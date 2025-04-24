import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.pyplot as plt
from llmlex.images import generate_base64_image, generate_base64_image_with_parents
from llmlex.llm import get_prompt, call_model, async_rate_limit_api_call, clear_rate_limit_lock, check_key_usage, async_call_model
from llmlex.response import extract_ansatz, fun_convert
import logging
import llmlex.fit as fit
import asyncio
import concurrent.futures
import importlib.util
import time
import re
from llmlex.response import APICallStats
from llmlex.fit import get_n_chi_squared

# Check if nest_asyncio is available
try:
    import nest_asyncio
    nest_asyncio_available = True
except ImportError:
    nest_asyncio_available = False

# Get module logger
logger = logging.getLogger("LLMLEx.llmlex")

# Helper function to execute a coroutine in a way that works with any event loop state
async def _run_in_nested_loop(coro):
    return await coro

def run_async_safely(coro):
    """
    Run a coroutine in a way that works both in and outside of an event loop.
    This function handles the complexity of running async code in different contexts.
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If the loop is already running (e.g., in a test), we create a task and await it
            return asyncio.create_task(_run_in_nested_loop(coro))
        else:
            # No running loop, but we have a loop
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop in this thread, create a new one
        return asyncio.run(coro)

def execute_async_in_loop(coro):
    """
    Helper function to safely execute an async function in any context.
    Works with running event loops (like in Jupyter) or creates a new one as needed.
    
    Args:
        coro: The coroutine to execute
        
    Returns:
        The result of the coroutine
    """
    # Try to get the current event loop
    current_loop = None
    try:
        current_loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop in this thread
        pass
    
    # If we have a running loop and nest_asyncio is available, use it
    if current_loop and current_loop.is_running() and nest_asyncio_available:
        # Apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()
        return current_loop.run_until_complete(coro)
        
    # If we have a running loop but no nest_asyncio, use our polling approach
    elif current_loop and current_loop.is_running():
        # We're in a running event loop (like in Jupyter)
        # Create a task and wait for it with polling
        future = asyncio.ensure_future(coro, loop=current_loop)
        
        # Wait for the result using a polling approach
        while not future.done():
            time.sleep(0.1)
        
        return future.result()
        
    # Standard case - no running loop, create a new one
    else:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
            # Restore original loop if there was one
            if current_loop:
                asyncio.set_event_loop(current_loop)
            else:
                asyncio.set_event_loop(None)
    
def single_call(client, img, x, y, model="openai/gpt-4o-mini", function_list=None, system_prompt=None, max_retries=3, stats=None, imports=None):
    """
    Executes a single call of to a specified llm-model with given parameters and processes the response.
    Args:
        client: The openai client object used to interact with the llm.
        img: The base64 encoded image to use for the model.
        x: The x-values for curve fitting.
        y: The y-values for curve fitting.
        model (str, optional): The model identifier to be used. Defaults to "openai/gpt-4o-mini".
        function_list (list, optional): A list of functions to be included in the prompt. Defaults to None.
        system_prompt (str, optional): A system-level prompt to guide the model's behavior. Defaults to None.
        max_retries (int, optional): Maximum number of retries for parsing errors. Default is 3.
        stats (APICallStats, optional): Statistics tracking object. If None, a new one will be created.
        imports (list, optional): A list of import statements to include in the prompt. Defaults to ["import numpy as np"].
    Returns:
        dict: A dictionary containing the following keys on success:
            - "params": The parameters resulting from the curve fitting.
            - "score": The score of the curve fitting.
            - "ansatz": The ansatz extracted from the model's response.
            - "Num_params": The number of parameters in the ansatz.
            - "response": The raw response from the model.
            - "prompt": The prompt used in the model call.
            - "function_list": The list of functions included in the prompt.
            - "stats": Statistics object (only if created locally).
        
        If all attempts fail, it raises the last exception.
    """
    logger.debug(f"Starting single_call with model={model}, function_list size={len(function_list) if function_list else 0}")
    
    # Create a local stats tracker if none provided
    local_stats = stats is not None
    if not local_stats:
        from llmlex.response import APICallStats
        stats = APICallStats()
    
    retry_count = 0
    last_error = None
    response = None
    
    while retry_count <= max_retries:
        retry_count += 1
        try:
            # Generate the prompt
            logger.debug("Generating prompt")
            prompt = get_prompt(function_list, imports=imports)
            
            # Make API call
            try:
                # Only make a new API call on the first attempt or if we need to retry with a new call
                if retry_count == 1 or response is None:
                    logger.debug(f"Calling model {model}")
                    response = call_model(client, model, img, prompt, system_prompt=system_prompt)
                stats.stage_success("api_call")
            except Exception as e:
                stats.stage_failure("api_call", e)
                last_error = e
                logger.error(f"API call failed: {e}")
                continue
            
            # Extract ansatz
            try:
                logger.debug("Extracting ansatz from response")
                ansatz, num_params = extract_ansatz(response)
                logger.info(f"Extracted ansatz: {ansatz[:50]}{'...' if len(ansatz) > 50 else ''} with {num_params} parameters")
                stats.stage_success("ansatz_extraction")
            except Exception as e:
                stats.stage_failure("ansatz_extraction", e)
                last_error = e
                logger.debug(f"Ansatz extraction failed: {e}")
                # For these errors, we might want to try a new API call
                response = None
                continue
            
            # Convert ansatz to function
            try:
                logger.debug("Converting ansatz to function")
                curve, num_params, lambda_str = fun_convert(ansatz)
                stats.stage_success("function_conversion")
            except Exception as e:
                stats.stage_failure("function_conversion", e)
                last_error = e
                logger.debug(f"Function conversion failed: {e}")
                # For these errors, we might want to try a new API call
                response = None
                continue
            
            # Fit curve to data
            try:
                logger.debug("Fitting curve to data")
                params, score = fit.fit_curve(x, y, curve, num_params, allow_using_jax=True, curve_str=lambda_str, stats=stats)
                logger.info(f"Fit result: score={-score}, params={params}")
                stats.stage_success("curve_fitting")
            except Exception as e:
                stats.stage_failure("curve_fitting", e)
                last_error = e
                logger.debug(f"Curve fitting in single_call failed: {e}")
                # For these errors, we might want to try a new API call
                response = None
                continue

            # If we get here, everything worked
            stats.add_success()
            result = {
                "params": params,
                "score": -score,
                "ansatz": ansatz,
                "Num_params": num_params,
                "response": response,
                "prompt": prompt,
                "function_list": function_list,
                "stats": None if local_stats else stats  # Only include stats if we created them locally
            }
            logger.debug("single_call completed successfully")
            return result
                
        except Exception as e:
            # This catch-all shouldn't be reached due to the inner try-except blocks
            stats.stage_failure("other", e)
            last_error = e
            logger.error(f"Unexpected error in single_call: {e}", exc_info=True)
            try:
                if response and hasattr(response, 'choices'):
                    logger.error(f"Response content: {response.choices[0].message.content}")
            except:
                logger.error("Could not access response content")
            response = None
            continue
    
    # If we've exhausted all retries
    if last_error:
        logger.error(f"All {max_retries} attempts failed in single_call. Last error: {last_error}")
        if not local_stats:
            logger.info(f"Call statistics:\n{stats}")
        raise last_error
    else:
        logger.error("Unknown error in single_call")
        raise RuntimeError("Unknown error in single_call")

# @async_rate_limit_api_call
# async def async_call_model(client, model, image, prompt, system_prompt=None):
#     """
#     Asynchronous version of call_model.
#     This function makes a direct async call to the LLM API with rate limiting.
#     """
#     logger.debug(f"Async calling model {model}")
    
#     # Set default system prompt if not provided
#     if system_prompt is None:
#         system_prompt = ("You are a symbolic regression expert. Analyze the data in the image and provide an improved mathematical ansatz (formula template). "
#                          "Respond with ONLY the ansatz formula, without any explanation or commentary. Ensure it is in valid python. You may use numpy functions. "
#                          "params is a list of parameters that can be of any length or complexity. Index into it with params[0], params[1], etc. "
#                          "Since the data contains noise, prioritize simpler, more elegant functions that capture the underlying pattern rather than fitting every point. ")
#         logger.debug("Using default system prompt: \n" + system_prompt)
    
#     # Track image size for debugging purposes
#     image_size = len(image) if image else 0
#     logger.debug(f"Image size: {image_size} characters (base64)")
#     logger.debug(f"Prompt length: {len(prompt)} characters")
    
#     try:
#         # Create and send the API request asynchronously or synchronously depending on client capabilities
#         logger.debug("Creating async chat completion request")
        
#         # Check if the client supports async operations directly
#         if hasattr(client.chat.completions, 'acreate'):
#             # Use the async API if available
#             response = await client.chat.completions.acreate(
#                 model=model,
#                 messages=[
#                     { "role": "system", 
#                      "content": system_prompt},
#                     {
#                         "role": "user",
#                         "content": [
#                             {
#                                 "type": "image_url",
#                                 "image_url": {"url": f"data:image/png;base64,{image}"},
#                             },
#                             {
#                                 "type": "text",
#                                 "text": prompt,
#                             },
#                         ],
#                     }
#                 ],
#                 max_tokens=4096,
#             )
#         else:
#             # If no async API is available, use the sync API in a thread pool
#             import concurrent.futures
#             loop = asyncio.get_event_loop()
#             with concurrent.futures.ThreadPoolExecutor() as pool:
#                 response = await loop.run_in_executor(
#                     pool,
#                     lambda: client.chat.completions.create(
#                         model=model,
#                         messages=[
#                             { "role": "system", 
#                              "content": system_prompt},
#                             {
#                                 "role": "user",
#                                 "content": [
#                                     {
#                                         "type": "image_url",
#                                         "image_url": {"url": f"data:image/png;base64,{image}"},
#                                     },
#                                     {
#                                         "type": "text",
#                                         "text": prompt,
#                                     },
#                                 ],
#                             }
#                         ],
#                         max_tokens=4096,
#                     )
#                 )
        
#         # Log response info
#         try:
#             if hasattr(response, 'usage'):
#                 token_usage = response.usage.total_tokens
#                 logger.debug(f"Async model response received. Total tokens: {token_usage}")
            
#             if hasattr(response, 'choices') and len(response.choices) > 0 and hasattr(response.choices[0], 'finish_reason'):
#                 logger.debug(f"Response finish reason: {response.choices[0].finish_reason}")
#         except Exception as e:
#             logger.debug(f"Could not access token usage information: {e}")
        
#         # Process response if needed
#         if hasattr(response, 'choices') and len(response.choices) > 0:
#             # Standard OpenAI API response format
#             if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
#                 # Return just the content for simpler processing later
#                 return response.choices[0].message.content
#             elif hasattr(response.choices[0], 'text'):
#                 return response.choices[0].text
#             # Fall back to returning the full response object
        
#         return response
        
#     except Exception as e:
#         # Log and re-raise any exceptions
# #         logger.error(f"Error calling model {model} asynchronously: {e}", exc_info=True)
# #         raise

#        raise

async def async_single_call(client, img, x, y, model="openai/gpt-4o-mini", function_list=None, system_prompt=None, max_retries=3, stats=None, plot_parents=False, imports=None):
    """
    Asynchronous version of single_call. Executes a single call to a specified llm-model with given parameters.
    This function is meant to be used with asyncio to allow for concurrent model calls.
    
    Args: 
        client: The API client
        img: Base64 encoded image
        x: x-values for fitting
        y: y-values for fitting
        model: Model name to use
        function_list: Optional list of functions
        system_prompt: Optional system prompt
        max_retries: Maximum number of retries for parsing/formatting errors
        stats: Optional APICallStats object for tracking statistics
        plot_parents: Whether to plot parents in the genetic algorithm
        imports: A list of import statements to include in the prompt. Defaults to ["import numpy as np"]
    
    Returns: 
        dict: Result dictionary or None if all attempts fail
    """
    logger.debug(f"Starting async_single_call with model={model}, function_list size={len(function_list) if function_list else 0}, plot_parents={plot_parents}")
    
    # Create a local stats tracker if none provided
    local_stats = stats is not None
    if not local_stats:
        from llmlex.response import APICallStats
        stats = APICallStats()
    
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        retry_count += 1
        try:
            # Get the proper prompt based on function_list
            prompt = get_prompt(function_list, imports=imports)
            if plot_parents:
                prompt = "\n\nThe listed curve_# functions (faded and broken lines) are plotted on the same image as the data (solid blue line).\
                  The coefficients the curve_# functions are plotted with are optimised with gradient descent. Use this information to improve the ansatz.\n" + prompt
                logger.debug(f"Generating plot for activation function with parents")
                img = generate_base64_image_with_parents(x, y, function_list)
            
            # Make the LLM call using our async rate-limited function
            try:
                resp = await async_call_model(client, model, img, prompt, system_prompt)
                stats.stage_success("api_call")
            except Exception as e:
                stats.stage_failure("api_call", e)
                last_error = e
                logger.error(f"API call failed: {e}")
                continue
            
            # Extract the ansatz from the response
            logger.debug("Extracting ansatz from response")
            
            # Process API response to get text content
            try:
                # At this point, resp should be a string because we're extracting the content in async_call_model
                # But let's be defensive and handle different response formats
                if isinstance(resp, str):
                    # Direct string response
                    response_text = resp
                elif hasattr(resp, 'choices') and len(resp.choices) > 0:
                    # Standard OpenAI API response
                    if hasattr(resp.choices[0], 'message') and hasattr(resp.choices[0].message, 'content'):
                        response_text = resp.choices[0].message.content
                    elif hasattr(resp.choices[0], 'text'):
                        response_text = resp.choices[0].text
                    else:
                        raise ValueError("Response format not recognized: no content found in choices")
                else:
                    raise ValueError(f"Unexpected response format: {type(resp)}")
            except Exception as e:
                stats.stage_failure("api_call", e, "invalid_response")
                last_error = e
                logger.warning(f"Failed to process API response: {e}")
                continue
                
            # Extract ansatz, convert to function, and fit curve
            try:
                # Try to extract a valid ansatz
                try:
                    ansatz, num_params = extract_ansatz(response_text)
                    stats.stage_success("ansatz_extraction")
                except Exception as e:
                    stats.stage_failure("ansatz_extraction", e)
                    last_error = e
                    logger.warning(f"Ansatz extraction failed: {e}")
                    continue
                
                # Try to convert to a function
                try:
                    f, num_params, lambda_str = fun_convert(ansatz)
                    stats.stage_success("function_conversion")
                except Exception as e:
                    stats.stage_failure("function_conversion", e)
                    last_error = e
                    logger.warning(f"Function conversion failed: {e}")
                    continue
                
                # Try to fit the curve
                try:
                    params, chi2 = fit.fit_curve(x, y, f, num_params, allow_using_jax=True, curve_str=lambda_str, stats=stats)
                    if chi2 == float('inf'):
                        logger.debug(f"Curve fitting failed: chi2 is infinite {ansatz} {num_params} {f}")# already logged elsewhere
                        stats.stage_failure("curve_fitting", RuntimeError("chi2 is infinite"))
                        continue
                    stats.stage_success("curve_fitting")
                except Exception as e:
                    stats.stage_failure("curve_fitting", e)
                    last_error = e
                    logger.warning(f"Curve fitting in async_single_call failed: {e}")
                    continue

                # Validate the function output
                try:
                    validate_function_output(x, f, params, stats)
                except Exception as e:
                    last_error = e
                    logger.warning(f"Function validation failed: {e}")
                    continue

                # If we got here, everything worked
                logger.debug(f"Fit complete. Chi²: {chi2}, parameters: {params}")
                stats.add_success()
                
                # Create result dictionary
                result = {
                    "params": params,
                    "score": -chi2,
                    "ansatz": ansatz,
                    "Num_params": num_params,
                    "response": response_text,
                    "prompt": prompt,
                    "function_list": function_list,
                    "stats": None if local_stats else stats  # Only include stats if we created them locally
                }
                
                logger.debug("async_single_call completed successfully")
                return result
                
            except Exception as e:
                # This catch-all shouldn't be reached due to the inner try-except blocks,
                # but it's here as a safety net
                logger.error(f"Unexpected error in processing: {e}", exc_info=True)
                last_error = e
                stats.stage_failure("other", e)
                continue
            
        except Exception as e:
            # Log any other errors that weren't caught by the inner try-except blocks
            logger.error(f"Error in async_single_call: {e}", exc_info=True)
            last_error = e
            stats.stage_failure("other", e)
            continue
    
    # If we've exhausted all retries
    if last_error:
        logger.error(f"All {max_retries} attempts failed in async_single_call. Last error: {last_error}")
        if not local_stats:
            logger.info(f"Call statistics:\n{stats}")
        raise last_error
    else:
        logger.error("Unknown error in async_single_call")
        raise RuntimeError("Unknown error in async_single_call")

def run_genetic(client, base64_image, x, y, population_size, num_of_generations,
                temperature=1., model="openai/gpt-4o-mini", exit_condition=1e-5, system_prompt=None, 
                elite=False, for_kan=False, use_async=True, plot_parents=False, demonstrate_parent_plotting=False, constant_on_failure=False, disable_parse_warnings=False, imports=None):
    """
        Run a genetic algorithm to fit a model to the given data.
        Parameters:
            client (object): The client object to use for API calls.
            base64_image (str): The base64 encoded image to use for the model.
            x (array-like): The input data for the model.
            y (array-like): The target data for the model.
            population_size (int): The size of the population for the genetic algorithm.
            num_of_generations (int): The number of generations to run the genetic algorithm.
            temperature (float, optional): The temperature parameter for the selection process. Default is 1.
            model (str, optional): The model to use for the API calls. Default is "openai/gpt-4o-mini".
            exit_condition (float, optional): The exit condition for the genetic algorithm. Default is 1e-5.
            system_prompt (str, optional): The system prompt to use for the API calls. Default is None.
            elite (bool, optional): Whether to use elitism in the genetic algorithm. Default is False.
            use_async (bool, optional): Whether to use async calls for population generation. Default is True.
            plot_parents (bool, optional): Whether to plot the parents in the genetic algorithm, showing the model the shape of the optimise parents. Default is False.
            demonstrate_parent_plotting (bool, optional): Whether to show to the user an example of the parent plotting. Default is False.
            constant_on_failure (bool, optional): Whether to return the constant function if the genetic algorithm fails. Default is False.
            disable_parse_warnings (bool, optional): Whether to disable parse warnings. Default is False.
            imports (list, optional): A list of import statements to include in the prompt. Defaults to ["import numpy as np"].
        Returns:
            list: A list of populations, where each population is a list of individuals.
        """
    clear_rate_limit_lock()

    logger.debug(f"Starting genetic algorithm with population_size={population_size}, generations={num_of_generations}, model={model}")
    logger.debug(f"Parameters: temperature={temperature}, exit_condition={exit_condition}, elite={elite}, for_kan={for_kan}")
    if plot_parents:
        logger.debug(f"Plotting parents in faded colours on the same image as the data.")
    

    
    # Initialize statistics tracker
    api_stats = APICallStats()
    
    population = []
    populations = []
    
    logger.debug("Checking constant function as baseline")
    curve = lambda x, *params: params[0] * np.ones(len(x))
    params, _ = curve_fit(curve, x, y, p0=[1.0])
    n_chi_squared = get_n_chi_squared(x, y, curve, params)

    if n_chi_squared <= exit_condition:
        logger.info(f"Constant function meets exit condition and is a good fit - returning early. Score: {-n_chi_squared}, constant: {params}")
        # Since we're returning early, there are no API calls to report
        logger.info(f"\nNo API calls were made (using constant function).")
        
        populations.append([{
            "params": params,
            "score": -n_chi_squared,
            "ansatz": "params[0]" if for_kan else "params[0]",
            "Num_params": 0,
            "response": None,
            "prompt": None,
            "function_list": None
        }])
        return populations
        
    logger.info(f"Constant function is not a good fit: Score: {-n_chi_squared}, for constant: {params}")

    
    # Always use async mode
    use_async = True
    logger.info(f"Generating initial population asynchronously")
    
    async def generate_population():
        tasks = []
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests to 10
        
        async def create_individual():
            nonlocal api_stats
            async with semaphore:
                max_attempts = 5
                for attempt in range(max_attempts):
                    try:
                        # Calculate exponential backoff delay
                        backoff_time = 0.1 * (2 ** attempt)  # 0.5s, 1s, 2s, 4s, 8s
                        
                        logger.debug(f"Async: Generating individual, attempt {attempt+1}/{max_attempts}")
                        result = await async_single_call(
                            client, base64_image, x, y, model=model, system_prompt=system_prompt,
                            stats=api_stats, imports=imports
                        )
                        if result is not None:
                            # Stats already updated in async_single_call
                            return result
                        
                        # No specific error, but call failed - don't add failure here as it's already recorded in async_single_call
                        logger.warning(f"Async: Failed attempt {attempt+1}/{max_attempts}, waiting {backoff_time}s before retry")
                        await asyncio.sleep(backoff_time)
                    except Exception as e:
                        api_stats.stage_failure("api_call", e)
                        logger.error(f"Async: Error in attempt {attempt+1}/{max_attempts}: {e}")
                        logger.warning(f"Async: Waiting {backoff_time}s before retry")
                        await asyncio.sleep(backoff_time)
                
                logger.error("Async: Failed to generate individual after 5 attempts with exponential backoff")
                return None
        for i in range(population_size):
            tasks.append(create_individual())
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    
    # Use our helper function to safely execute the async code in any context
    population = execute_async_in_loop(generate_population())
        
    logger.info(f"Generated {len(population)} individuals")

    # Check if we have a valid population
    if not population:
        error_msg = "Failed to generate any valid population members after multiple attempts"
        logger.error(error_msg)
        
        # If constant_on_failure is True, return the constant function
        if constant_on_failure:
            logger.info(f"Returning constant function as fallback (constant_on_failure=True). Score: {-n_chi_squared}, constant: {params}")
            populations.append([{
                "params": params,
                "score": -n_chi_squared,
                "ansatz": "params[0]" if for_kan else "params[0]",
                "Num_params": 0,
                "response": None,
                "prompt": None,
                "function_list": None
            }])
            return populations
        
        raise RuntimeError(error_msg)
    
    # Handle NaN and infinite scores
    # Count non-finite scores
    non_finite_scores = [p for p in population if not np.isfinite(p['score'])]
    if non_finite_scores:
        # Find the minimum finite score to use as reference
        finite_scores = [ind['score'] for ind in population if np.isfinite(ind['score'])]
        if finite_scores:
            min_score = min(finite_scores)
            # Set bad score to twice the worst score (assuming scores are negative)
            bad_score = 2 * min_score if min_score < 0 else min_score - abs(min_score)
        else:
            bad_score = -1e8  # Fallback if no finite scores
            
        logger.info(f"Found {len(non_finite_scores)} non-finite scores, setting all to {bad_score}. Example ansatz: {non_finite_scores[0]['ansatz']}")
        
        # Set all non-finite scores to the same bad_score
        for p in non_finite_scores:
            p['score'] = bad_score
            
    population.sort(key=lambda x: x['score'])
    populations.append(population)
    best_pop = population[-1]
    logger.info(f"Initial population best: score={best_pop['score']}, params={best_pop['params']}, ansatz: {best_pop['ansatz'][:100]}...")
    
    if best_pop['score'] > -exit_condition:
        logger.info("Exit condition met after initial population")
        print(f"\n{api_stats}")
        return populations

    if plot_parents and demonstrate_parent_plotting:
        demo_func_list = [(pop['ansatz'], pop['params']) for pop in populations[-1][-min(2, len(populations[-1])):]] # take the last two, so the best fit
        demo_image = generate_base64_image_with_parents(x, y, demo_func_list, fig=None, ax=None, actually_plot=True, title_override="Example of a plot with parents added")
    # Evolution loop
    for generation in range(num_of_generations-1):
        
        logger.debug("Computing selection probabilities")
        scores = np.array([ind['score'] for ind in population])
        finite_scores = scores[np.isfinite(scores)]
        
        # Handle case where all scores might be non-finite
        if len(finite_scores) == 0:
            logger.warning("No finite scores found, using uniform selection probabilities")
            probs = np.ones(len(scores)) / len(scores)
        else:
            normalized_scores = (scores - np.min(finite_scores)) / (np.max(finite_scores) - np.min(finite_scores) + 1e-6)
            # Ensure normalized scores are finite
            normalized_scores = np.nan_to_num(normalized_scores, nan=0.0, posinf=0.0, neginf=0.0)
            
            exp_scores = np.exp((normalized_scores - np.max(normalized_scores))/temperature)
            exp_scores = np.nan_to_num(exp_scores, nan=0.0)
            
            if np.sum(exp_scores) < 1e-10:
                logger.warning("All selection probabilities are effectively zero, using uniform distribution")
                probs = np.ones_like(exp_scores) / len(exp_scores)
            else:
                probs = exp_scores / np.sum(exp_scores)
        
        logger.debug("Selecting parents for next generation")
        selected_population = [np.random.choice(populations[-1], size=2,
                                               p=probs, replace=True) for _ in range(population_size)]

        func_lists = [[(pops[0]['ansatz'], pops[0]['params']), (pops[1]['ansatz'], pops[1]['params'])] for pops in selected_population]
        
        population = []
        if elite:
            logger.debug("Using elitism - keeping best individual")
            population.append(best_pop)
            
        if use_async:
            logger.info(f"Generation {generation+1}/{num_of_generations-1}: Generating {population_size} new individuals. Elitism? {elite}")
            
            async def generate_generation_population():
                tasks = []
                semaphore = asyncio.Semaphore(10)  # Limit concurrent requests to 10
                
                async def create_individual(idx):
                    nonlocal api_stats
                    func_list = func_lists[idx]
                    async with semaphore:
                        for attempt in range(5):  # Limit retries
                            try:
                                logger.debug(f"Async: Generation {generation+1}: Creating individual {idx+1}/{population_size}, attempt {attempt+1}")
                                result = await async_single_call(
                                    client, base64_image, x, y, model=model,
                                    function_list=func_list, system_prompt=system_prompt,
                                    stats=api_stats,
                                    plot_parents=plot_parents,
                                    imports=imports
                                )
                                if result is not None:
                                    # Stats already updated in async_single_call
                                    return result
                                # No specific error, but call failed - don't add failure here as it's already recorded in async_single_call
                                logger.warning(f"Async: Generation {generation+1}: Failed attempt {attempt+1} for individual {idx+1}")
                            except Exception as e:
                                api_stats.stage_failure("api_call", e)
                                logger.error(f"Async: Generation {generation+1}: Error in attempt {attempt+1} for individual {idx+1}: {e}")
                        
                        logger.error(f"Async: Generation {generation+1}: Failed to generate individual {idx+1} after 5 attempts")
                        return None
                
                for i in range(population_size - (1 if elite else 0)):
                    tasks.append(create_individual(i))
                
                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks)
                return [r for r in results if r is not None]
            
            # Use our helper function to safely execute the async code in any context
            gen_population = execute_async_in_loop(generate_generation_population())
                
            if elite:
                population.extend(gen_population)
            else:
                population = gen_population
        
        else:
            logger.info(f"Generation {generation+1}/{num_of_generations-1}: Generated {population_size} individuals. Elitism? {elite}")
            # Original synchronous implementation
            for funcs in tqdm(range(population_size - (1 if elite else 0))):
                good = False
                attempts = 0
                while not good and attempts < 5:  # Limit retries
                    attempts += 1
                    logger.debug(f"Generation {generation+1}: Creating individual {funcs+1}/{population_size}, attempt {attempts}")
                    try:
                        result = single_call(client, base64_image, x, y, model=model,
                                            function_list=func_lists[funcs], system_prompt=system_prompt,
                                            stats=api_stats, imports=imports)
                        if result is not None:
                            population.append(result)
                            good = True
                            # Stats already updated in single_call
                        else:
                            # Stats already recorded in single_call
                            logger.warning(f"Failed to generate individual {funcs+1}, attempt {attempts}")
                    except Exception as e:
                        api_stats.stage_failure("api_call", e)
                        logger.warning(f"Failed to generate individual {funcs+1}, attempt {attempts}: {e}")
                        
                if not good:
                    logger.error(f"Failed to generate individual {funcs+1} after {attempts} attempts")
        
        # Check if we have a valid population after this generation
        if not population:
            error_msg = f"Failed to generate any valid population members in generation {generation+1}"
            logger.error(error_msg)
            
            # If constant_on_failure is True, return the constant function
            if constant_on_failure:
                logger.info(f"Returning constant function as fallback (constant_on_failure=True). Score: {n_chi_squared}, constant: {params}")
                populations.append([{
                    "params": params,
                    "score": -n_chi_squared,
                    "ansatz": "params[0]" if for_kan else "params[0]",
                    "Num_params": 0,
                    "response": None,
                    "prompt": None,
                    "function_list": None
                }])
        population.sort(key=lambda x: x['score'])
        best_pop = population[-1]
        populations.append(population)
        
        logger.info(f"Generation {generation+1} best: score={best_pop['score']}, params={best_pop['params']}, ansatz: {best_pop['ansatz'][:100]}...")

        if best_pop['score'] > -exit_condition:
            logger.info(f"Exit condition met after generation {generation+1}: {best_pop['score']}>{-exit_condition}")
            
            # Print API call and validation statistics
            print(f"\n{api_stats}")
            
            # Print validation issue summary if any issues detected
            if api_stats.total_validation_issues() > 0:
                logger.info(f"Validation issues detected during genetic algorithm run, see summary.")
            
            return populations
    
    logger.info(f"Genetic algorithm completed after {num_of_generations} generations")
    
    # Print comprehensive statistics
    print(f"\n{api_stats}")
    
    # Print validation issue summary if any issues detected
    if api_stats.total_validation_issues() > 0:
        logger.info(f"Validation issues detected during genetic algorithm run, see summary.")
    
    return populations

def kan_to_symbolic(model, client, population=10, generations=3, temperature=0.1, gpt_model="openai/gpt-4o-mini", exit_condition=1e-3, verbose=0, use_async=True, plot_fit=True, plot_parents=False, demonstrate_parent_plotting=False, constant_on_failure=False, disable_parse_warnings=False, imports=None):
    """
    Converts a given kan model symbolic representations using llmlexs.
    Parameters:
        model (object): The kan model.
        client (object): The openai client object used to access the llm.
        population (int, optional): The population size for the genetic algorithm. Default is 10.
        generations (int, optional): The number of generations for the genetic algorithm. Default is 3.
        temperature (float, optional): The temperature parameter for the genetic algorithm. Default is 0.1.
        llm_model (str, optional): The GPT model to use for generating symbolic functions. Default is "openai/gpt-4o-mini".
        exit_condition (float, optional): The exit condition for the genetic algorithm. Default is 1e-3.
        verbose (int, optional): Verbosity level for logging. Default is 0.
        use_async (bool, optional): Whether to use asynchronous processing for population generation. Default is True.
        plot_fit (bool, optional): Whether to plot the fit of the symbolic functions. Default is True.
        plot_parents (bool, optional): Whether to plot the parents of the symbolic functions. Default is False.
        demonstrate_parent_plotting (bool, optional): Whether to demonstrate the parent plotting. Default is False.
        constant_on_failure (bool, optional): Whether to return the constant function if the genetic algorithm fails. Default is False.
        disable_parse_warnings (bool, optional): Whether to disable parse warnings. Default is False.
        imports (list, optional): A list of import statements to include in the prompt. Default is None.
    Returns:
        - res_fcts (dict): A dictionary mapping layer, input, and output indices to their corresponding symbolic functions.
    """
    start_usage = check_key_usage(client)    
    logger.debug(f"Starting KAN to symbolic conversion with population={population}, generations={generations}")
    logger.debug(f"KAN model has {len(model.width_in)} layers")

    res_fcts = {}
    
    # Initialize symb_formula to hold placeholders for all connections
    symb_formula = []
    for l in range(len(model.width_in) - 1):
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l]):
                symb_formula.append(f'f_{{{l},{i},{j}}}')
    
    # Setup layer connections
    logger.debug("Setting up layer connections")
    layer_connections = {0: {i: [] for i in range(model.width_in[0])}}
    for l in range(len(model.width_in) - 1):
        layer_connections[l] = {i: list(range(model.width_out[l-1])) if l > 0 else []  for i in range(model.width_in[l])}
    
    # Process each connection in the KAN model
    total_connections = 0
    symbolic_connections = 0
    zero_connections = 0
    processed_connections = 0
    
    # Track fitting warning statistics across all connections
    total_fitting_warnings = {
        "invalid_sqrt": 0,
        "covariance_estimation": 0,
        "other_warnings": 0
    }
    
    logger.info("Processing KAN model connections")
    for l in range(len(model.width_in) - 1):
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l + 1]):
                total_connections += 1
                if (model.symbolic_fun[l].mask[j, i] > 0. and model.act_fun[l].mask[i][j] == 0.):
                    logger.info(f'Skipping ({l},{i},{j}) - already symbolic')
                    symb_formula = [s.replace(f'f_{{{l},{i},{j}}}', 'TODO') for s in symb_formula]
                    symbolic_connections += 1
                    
                elif (model.symbolic_fun[l].mask[j, i] == 0. and model.act_fun[l].mask[i][j] == 0.):
                    logger.info(f'Fixing ({l},{i},{j}) with 0')
                    model.fix_symbolic(l, i, j, '0', verbose=verbose > 1, log_history=False)
                    symb_formula = [s.replace(f'f_{{{l},{i},{j}}}', '0') for s in symb_formula]
                    res_fcts[(l, i, j)] = None
                    zero_connections += 1
                    
                else:
                    logger.info(f'Processing non-symbolic activation function ({l},{i},{j})')
                    processed_connections += 1
                    
                    # Generate data for the connection
                    logger.debug(f"Getting range data for activation function ({l},{i},{j})")
                    x_min, x_max, y_min, y_max = model.get_range(l, i, j, verbose=False)
                    # Handle PyTorch tensors or NumPy arrays
                    x_data = model.acts[l][:, i]
                    y_data = model.spline_postacts[l][:, j, i]
                    
                    # Convert to numpy if it's a PyTorch tensor
                    if hasattr(x_data, 'cpu') and hasattr(x_data, 'detach'):
                        x = x_data.cpu().detach().numpy()
                    else:
                        x = np.array(x_data)
                        
                    if hasattr(y_data, 'cpu') and hasattr(y_data, 'detach'):
                        y = y_data.cpu().detach().numpy()
                    else:
                        y = np.array(y_data)
                    
                    # Sort data by x values
                    ordered_in = np.argsort(x)
                    x, y = x[ordered_in], y[ordered_in]
                    # Generate plot
                    if plot_fit:
                        logger.debug(f"Generating plot for activation function ({l},{i},{j})")
                        fig, ax = plt.subplots(figsize=(4, 3))
                        plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                        plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                        base64_image = generate_base64_image(fig, ax, x, y)
                        plt.title(f"Activation ({l},{i},{j})")
                        plt.tight_layout()
                        plt.show()
                    # Get activation function mask
                    mask = model.act_fun[l].mask
                    
                    # Run genetic algorithm to find symbolic expression
                    try:
                        # change system prompt for KANs -  we need less summands, since these are already covered by the KAN architecture
                        system_prompt = "You are a symbolic regression expert. Analyze the data in the image and provide an improved mathematical ansatz. The ansatz should have as few terms as possible and ideally no sums." +\
                         				"Respond with ONLY the ansatz formula, without any explanation or commentary. Ensure it is in valid python. You may use numpy functions." +\
                         				"params is a list of parameters that can be of any length or complexity."
                        logger.info(f"Running genetic algorithm for connection ({l},{i},{j})")
                        res = run_genetic(
                            client, base64_image, x, y, population, generations, 
                            temperature=temperature, model=gpt_model, 
                            system_prompt=system_prompt, elite=False, 
                            exit_condition=exit_condition, for_kan=True,
                            use_async=use_async, plot_parents=plot_parents,demonstrate_parent_plotting=demonstrate_parent_plotting, 
                            constant_on_failure=constant_on_failure, disable_parse_warnings=disable_parse_warnings,
                            imports=imports
                        )
                        res_fcts[(l,i,j)] = res
                        logger.info(f"Successfully found expression for connection ({l},{i},{j})")
                        
                        # Track fitting warnings from this run
                        if res is not None and len(res) > 0 and len(res[-1]) > 0:
                            # Find the last population stats if available
                            for pop in reversed(res):
                                for individual in pop:
                                    if individual.get('stats') and hasattr(individual['stats'], 'fitting_warnings'):
                                        for warning_type, count in individual['stats'].fitting_warnings.items():
                                            if count > 0:
                                                total_fitting_warnings[warning_type] += count
                                        break
                                break
                        
                        # Plot the fitted function on top of the original data
                        if res is not None and len(res) > 0 and len(res[-1]) > 0:
                            # Find the highest scoring element across all generations
                            highest_score_element = max((item for sublist in res for item in sublist), key=lambda item: item['score'])
                            print(f"Approximation for ({l},{i},{j}): {highest_score_element['ansatz'].strip()}, with score {highest_score_element['score']} and parameters {np.round(highest_score_element['params'], 3)}")
                            
                            # Use the highest scoring individual for plotting
                            if plot_fit:
                                try:
                                    # Convert the ansatz to a function
                                    curve, _,_ = fun_convert(highest_score_element['ansatz'])
                                    # Generate y values using the fitted function
                                    fitted_y = curve(x, *highest_score_element['params'])
                                    # Check if fitted_y is a scalar or has different shape than x
                                    if isinstance(fitted_y, (int, float, np.number)):
                                        fitted_y = np.full_like(x, fitted_y)
                                    elif np.isscalar(fitted_y) or len(fitted_y) == 1 or fitted_y.shape != x.shape:
                                        # Handle case where fitted_y is array-like but wrong shape
                                        if np.size(fitted_y) == 1:
                                            # Single value in array
                                            fitted_y = np.full_like(x, fitted_y.item() if hasattr(fitted_y, 'item') else fitted_y)
                                        else:
                                            # Try to reshape or broadcast
                                            fitted_y = np.broadcast_to(fitted_y, x.shape)
                                    
                                    # Create a new figure for the fitted function
                                    fig2, ax2 = plt.subplots(figsize=(8,6))
                                    ax2.plot(x, y, 'b-', linewidth=4, label='Original data', alpha=0.5)
                                    ax2.plot(x, fitted_y, 'r-', linewidth=2, label='Fitted function', alpha=1)
                                    plt.title(f"Fitted activation function ({l},{i},{j})")
                                    plt.legend()
                                    plt.tight_layout()
                                    plt.show()
                                except Exception as fit_err:
                                    logger.warning(f"Could not plot fitted function: {fit_err}")
                    except Exception as e:
                        logger.error(f"Error in genetic algorithm for connection ({l},{i},{j}): {e}", exc_info=True)
                        res_fcts[(l,i,j)] = res
    # Clean up
    logger.debug("Cleaning up matplotlib resources")
    try:
        ax.clear()
        plt.close()
    except Exception as e:
        logger.debug(f"Could not clean up matplotlib resources: {e}. Not a cause for concern.")
    
    # Log summary
    logger.info(f"KAN conversion complete: {total_connections} total connections")
    logger.info(f"Connection breakdown: {symbolic_connections} symbolic, {zero_connections} zero, {processed_connections} processed")
    
    # Log fitting warning summary if any warnings were detected
    if sum(total_fitting_warnings.values()) > 0:
        logger.info("Fitting warnings encountered during processing:")
        for warning_type, count in total_fitting_warnings.items():
            if count > 0:
                logger.info(f"  - {warning_type.replace('_', ' ')}: {count}")
                
    end_usage = check_key_usage(client)
    cost = f"${(end_usage - start_usage):.2f}" if isinstance(end_usage, (float, int)) and isinstance(start_usage, (float, int)) else 'unknown'
    logger.info(f"API key usage whilst this kan_to_symbolic was running: {cost}")
    
    return res_fcts

def generate_learned_f(sym_expr):
    """
    Generate a Python function from symbolic expressions discovered by the genetic algorithm.
    
    Args:
        sym_expr: Dictionary mapping connection tuples (layer, input_node, output_node) 
                 to the results of the genetic algorithm for that connection, or mapping to a
                 flattened and sorted list of results (as in the output of KANSR._sort_symbolic_expressions).
                 
    Returns:
        Tuple containing:
        - The generated Python function that implements the learned model
        - Total number of parameters in the learned model
        - List of best parameters for all connections
        
    Raises:
        ValueError: If any connection in sym_expr has a None value

    Assumes the conn_keys are formatted as (layer, input, output)
    """

    import re
    # Determine input nodes from layer 0
    conn_keys = list(sym_expr.keys())
    # Check for None values in sym_expr
    none_connections = [(l, i, j) for (l, i, j), value in sym_expr.items() if value is None]
    if none_connections:
        error_msg = f"Found {len(none_connections)} connections with None values: {none_connections}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    input_nodes = sorted({i for (l, i, j) in conn_keys if l == 0})
    layers = sorted({l for (l, i, j) in conn_keys})
    final_layer = max(layers) + 1
    # Build mapping: for each layer l, map target node j to list of (source i, modified ansatz)
    conns = {}
    param_index = 0
    param_counts = {}
    best_params = []
    
    # First pass: determine parameter counts for each connection
    for (l, i, j) in conn_keys:
        if sym_expr[(l, i, j)] is None:
            continue
        if isinstance(sym_expr[(l, i, j)], list) and all(isinstance(sub, list) for sub in sym_expr[(l, i, j)]):
            best = max((item for sub in sym_expr[(l, i, j)] for item in sub), key=lambda item: item['score'])
        else:
            best = max(sym_expr[(l, i, j)], key=lambda item: item['score'])
        param_counts[(l, i, j)] = len(best['params'])
    
    # Second pass: build connections with proper parameter indexing
    for (l, i, j) in conn_keys:
        if sym_expr[(l, i, j)] is None:
            continue
        if isinstance(sym_expr[(l, i, j)], list) and all(isinstance(sub, list) for sub in sym_expr[(l, i, j)]):
            best = max((item for sub in sym_expr[(l, i, j)] for item in sub), key=lambda item: item['score'])
        else:
            best = max(sym_expr[(l, i, j)], key=lambda item: item['score'])
        ansatz = best['ansatz'].strip()
        
        # Collect best parameters
        best_params.extend(best['params'])
        
        # Replace parameter references with indexed params
        param_count = param_counts[(l, i, j)]
        param_indices = list(range(param_index, param_index + param_count))
        param_index += param_count
        
        # Replace standalone 'x' with the source activation variable x_l_i
        ansatz_mod = re.sub(r'\bx\b', f"x_{l}_{i}", ansatz)
        
        # Replace params references with properly indexed params
        for p_idx, orig_idx in enumerate(range(len(best['params']))):
            ansatz_mod = ansatz_mod.replace(f"params[{orig_idx}]", f"paramsp[{param_indices[p_idx]}]")

        #stop overlapping replacements
        ansatz_mod = ansatz_mod.replace("paramsp", "params")
        
        conns.setdefault(l, {}).setdefault(j, []).append((i, ansatz_mod))
    lines = []
    lines.append("def learned_f(X, *params):")
    lines.append("    # Layer 0 activations")
    inp = ", ".join([f"x_0_{i}" for i in input_nodes])
    lines.append(f"    {inp} = " + ", ".join([f"X[..., {i}]" for i in input_nodes]))
    # Compute activations layer by layer
    for l in sorted(conns.keys()):
        for j in sorted(conns[l].keys()):
            temp = []
            for i, expr in conns[l][j]:
                var = f"px_{l}_{i}_{j}"
                lines.append(f"    {var} = {expr}")
                temp.append(var)
            lines.append(f"    x_{l+1}_{j} = " + " + ".join(temp))
    # Return final layer activation(s)
    final_nodes = []
    if (final_layer - 1) in conns:
        final_nodes = sorted(conns[final_layer - 1].keys())
    if len(final_nodes) == 1:
        lines.append(f"    return x_{final_layer}_{final_nodes[0]}")
    else:
        ret = ", ".join([f"x_{final_layer}_{j}" for j in final_nodes])
        lines.append(f"    return {ret}")
    
    total_num_params = param_index
    return "\n".join(lines), total_num_params, np.array(best_params)
              
def validate_function_output(x, f, params, stats):
    """
    Validates that a function produces correct output shape for the input data.
    
    Args:
        x: Input data array
        f: Function to validate
        params: Parameters for the function
        stats: Statistics object for tracking failures
    
    Raises:
        ValueError: If function output doesn't match input shape and can't be fixed
    """
    # Test with a small sample of the actual data
    test_x = x[:min(57, len(x))]
    try:
        test_y = f(test_x, *params)
        
        # Check if output shape matches input
        if test_y.shape != test_x.shape:
            logger.warning(f"Function output shape {test_y.shape} doesn't match input shape {test_x.shape}")
            # Try to reshape or broadcast if possible
            if np.isscalar(test_y) or len(test_y) == 1:
                # Handle scalar output by broadcasting
                logger.debug("Attempting to broadcast scalar output to match input shape")
                test_y = np.full_like(test_x, test_y)
                if hasattr(stats, 'add_validation_issue'):
                    stats.add_validation_issue('scalar_output')
            elif len(test_y) != len(test_x):
                if hasattr(stats, 'add_validation_issue'):
                    stats.add_validation_issue('shape_mismatch')
                raise ValueError(f"Function returns {len(test_y)} values for {len(test_x)} inputs")
        
        # Additional check for NaN or inf values
        if np.any(np.isnan(test_y)):
            logger.debug("Function returns NaN values")
            if hasattr(stats, 'add_validation_issue'):
                stats.add_validation_issue('nan_values')
        
        if np.any(np.isinf(test_y)):
            logger.debug("Function returns inf values")
            if hasattr(stats, 'add_validation_issue'):
                stats.add_validation_issue('inf_values')
    
    except Exception as e:
        # Track other kinds of function evaluation errors
        logger.warning(f"Error evaluating function: {e}")
        if hasattr(stats, 'add_validation_issue'):
            stats.add_validation_issue('evaluation_error', str(e))
        raise

    return 