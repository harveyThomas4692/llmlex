import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.pyplot as plt
from LLMSR.images import generate_base64_image
from LLMSR.llm import get_prompt, call_model, async_rate_limit_api_call, clear_rate_limit_lock
from LLMSR.response import extract_ansatz, fun_convert
import logging
import LLMSR.fit as fit
import asyncio
import concurrent.futures
import importlib.util
import time

# Check if nest_asyncio is available
try:
    import nest_asyncio
    nest_asyncio_available = True
except ImportError:
    nest_asyncio_available = False

# Get module logger
logger = logging.getLogger("LLMSR.llmSR")

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
    
def single_call(client, img, x, y, model="openai/gpt-4o-mini", function_list=None, system_prompt=None, max_retries=3):
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
    Returns:
        dict: A dictionary containing the following keys on success:
            - "params": The parameters resulting from the curve fitting.
            - "score": The score of the curve fitting.
            - "ansatz": The ansatz extracted from the model's response.
            - "Num_params": The number of parameters in the ansatz.
            - "response": The raw response from the model.
            - "prompt": The prompt used in the model call.
            - "function_list": The list of functions included in the prompt.
        
        If all attempts fail, it raises the last exception.
    """
    logger.debug(f"Starting single_call with model={model}, function_list size={len(function_list) if function_list else 0}")
    
    retry_count = 0
    last_error = None
    response = None
    
    while retry_count <= max_retries:
        try:
            # Generate the prompt
            logger.debug("Generating prompt")
            prompt = get_prompt(function_list)
            
            # Only make a new API call on the first attempt or if we need to retry with a new call
            if retry_count == 0 or response is None:
                logger.debug(f"Calling model {model}")
                response = call_model(client, model, img, prompt, system_prompt=system_prompt)
            
            # Try to parse the response and fit the curve
            logger.debug("Extracting ansatz from response")
            ansatz, num_params = extract_ansatz(response)
            logger.info(f"Extracted ansatz: {ansatz[:50]}{'...' if len(ansatz) > 50 else ''} with {num_params} parameters")
            
            logger.debug("Converting ansatz to function")
            curve, num_params = fun_convert(ansatz)
            
            logger.debug("Fitting curve to data")
            params, score = fit.fit_curve(x, y, curve, num_params)
            logger.info(f"Fit result: score={-score}, params={params}")

            # If we get here, everything worked
            result = {
                "params": params,
                "score": -score,
                "ansatz": ansatz,
                "Num_params": num_params,
                "response": response,
                "prompt": prompt,
                "function_list": function_list
            }
            logger.debug("single_call completed successfully")
            return result
            
        except (ValueError, SyntaxError, TypeError) as e:
            # These are parsing/formatting errors that might be worth retrying
            retry_count += 1
            last_error = e
            logger.debug(f"Attempt {retry_count}/{max_retries} failed: {str(e)}")
            
            if retry_count > max_retries:
                logger.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
                raise
            
            # Reset the response to force a new API call on the next iteration
            response = None
                
        except Exception as e:
            # Other errors like API errors shouldn't be retried
            logger.error(f"Error in single_call: {e}", exc_info=True)
            try:
                if response and hasattr(response, 'choices'):
                    logger.error(f"Response content: {response.choices[0].message.content}")
            except:
                logger.error("Could not access response content")
            raise
    
    # This code should never be reached because we either return or raise above
    if last_error:
        raise last_error
    else:
        raise RuntimeError("Unknown error in single_call")

@async_rate_limit_api_call
async def async_model_call(client, model, image, prompt, system_prompt=None):
    """
    Asynchronous version of call_model.
    This function makes a direct async call to the LLM API with rate limiting.
    """
    logger.debug(f"Async calling model {model}")
    
    # Set default system prompt if not provided
    if system_prompt is None:
        system_prompt = ("Give an improved ansatz to the list for the image. Follow on from the users text with no explaining."
                         "Params can be any length. There's some noise in the data, give preference to simpler functions.")
        logger.debug("Using default system prompt")
    
    # Track image size for debugging purposes
    image_size = len(image) if image else 0
    logger.debug(f"Image size: {image_size} characters (base64)")
    logger.debug(f"Prompt length: {len(prompt)} characters")
    
    try:
        # Create and send the API request asynchronously or synchronously depending on client capabilities
        logger.debug("Creating async chat completion request")
        
        # Check if the client supports async operations directly
        if hasattr(client.chat.completions, 'acreate'):
            # Use the async API if available
            response = await client.chat.completions.acreate(
                model=model,
                messages=[
                    { "role": "system", 
                     "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image}"},
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                max_tokens=4096,
            )
        else:
            # If no async API is available, use the sync API in a thread pool
            import concurrent.futures
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                response = await loop.run_in_executor(
                    pool,
                    lambda: client.chat.completions.create(
                        model=model,
                        messages=[
                            { "role": "system", 
                             "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{image}"},
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt,
                                    },
                                ],
                            }
                        ],
                        max_tokens=4096,
                    )
                )
        
        # Log response info
        try:
            if hasattr(response, 'usage'):
                token_usage = response.usage.total_tokens
                logger.debug(f"Async model response received. Total tokens: {token_usage}")
            
            if hasattr(response, 'choices') and len(response.choices) > 0 and hasattr(response.choices[0], 'finish_reason'):
                logger.debug(f"Response finish reason: {response.choices[0].finish_reason}")
        except Exception as e:
            logger.debug(f"Could not access token usage information: {e}")
        
        # Process response if needed
        if hasattr(response, 'choices') and len(response.choices) > 0:
            # Standard OpenAI API response format
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                # Return just the content for simpler processing later
                return response.choices[0].message.content
            elif hasattr(response.choices[0], 'text'):
                return response.choices[0].text
            # Fall back to returning the full response object
        
        return response
        
    except Exception as e:
        # Log and re-raise any exceptions
        logger.error(f"Error calling model {model} asynchronously: {e}", exc_info=True)
        raise

async def async_single_call(client, img, x, y, model="openai/gpt-4o-mini", function_list=None, system_prompt=None, max_retries=3):
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
    
    Returns: 
        dict: Result dictionary or None if all attempts fail
    """
    logger.debug(f"Starting async_single_call with model={model}, function_list size={len(function_list) if function_list else 0}")
    
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            # Get the proper prompt based on function_list
            prompt = get_prompt(function_list)
            
            # Make the LLM call using our async rate-limited function
            resp = await async_model_call(client, model, img, prompt, system_prompt)
            
            # Extract the ansatz from the response
            logger.debug("Extracting ansatz from response")
            
            # At this point, resp should be a string because we're extracting the content in async_model_call
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
                
            try:
                # Try to extract a valid ansatz (will raise ValueError if none found)
                ansatz, num_params = extract_ansatz(response_text)
                
                # Try to convert to a function (will raise if invalid)
                f, num_params = fun_convert(ansatz)
                
                # Try to fit the curve (this is the real test of validity)
                params, chi2 = fit.fit_curve(x, y, f, num_params)
                
                # If we got here, everything worked
                logger.debug(f"Fit complete. Chi²: {chi2}, parameters: {params}")
                
                # Create result dictionary
                result = {
                    "params": params,
                    "score": -chi2,
                    "ansatz": ansatz,
                    "Num_params": num_params,
                    "response": response_text,
                    "prompt": prompt,
                    "function_list": function_list
                }
                
                logger.debug("async_single_call completed successfully")
                return result
                
            except (ValueError, SyntaxError, TypeError) as e:
                # These are parsing/formatting errors worth retrying
                retry_count += 1
                last_error = e
                logger.warning(f"Attempt {retry_count}/{max_retries} failed: {str(e)}")
                if retry_count <= max_retries:
                    continue
                else:
                    logger.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
                    raise
            
        except Exception as e:
            # Log any other errors but don't retry them, they may be API errors
            logger.error(f"Error in async_single_call: {e}", exc_info=True)
            raise
    
    # If we've exhausted all retries
    if last_error:
        logger.error(f"All {max_retries} attempts failed in async_single_call. Last error: {last_error}")
        raise last_error
    else:
        logger.error("Unknown error in async_single_call")
        raise RuntimeError("Unknown error in async_single_call")

def run_genetic(client, base64_image, x, y, population_size, num_of_generations,
                temperature=1., model="openai/gpt-4o-mini", exit_condition=1e-5, system_prompt=None, 
                elite=False, for_kan=False, use_async=False):
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
            use_async (bool, optional): Whether to use async calls for population generation. Default is False.
        Returns:
            list: A list of populations, where each population is a list of individuals.
        """
    clear_rate_limit_lock()

    logger.debug(f"Starting genetic algorithm with population_size={population_size}, generations={num_of_generations}, model={model}")
    logger.debug(f"Parameters: temperature={temperature}, exit_condition={exit_condition}, elite={elite}, for_kan={for_kan}")
    
    population = []
    populations = []
    
    logger.debug("Checking constant function as baseline")
    curve = lambda x, *params: params[0] * np.ones(len(x))
    params, _ = curve_fit(curve, x, y, p0=[1])
    residuals = y - params[0]*np.ones(len(x))
    chi_squared = np.mean((residuals ** 2) / (np.square(curve(x, *params))+1e-6))
    logger.info(f"Constant function baseline: score={-chi_squared}, constant={params[0]}")

    if chi_squared <= exit_condition:
        logger.info("Constant function meets exit condition - returning early")
        print("Constant function is good fit.")
        print("Score: ", -chi_squared)
        print("Constant: ", params)
        populations.append([{
            "params": params,
            "score": -chi_squared,
            "ansatz": "lambda x,*params: params[0] * np.ones(len(x))" if for_kan else "params[0]",
            "Num_params": 0,
            "response": None,
            "prompt": None,
            "function_list": None
        }])
        return populations
        
    logger.info(f"Constant function is not a good fit: Score: {chi_squared}, for constant: {params}")

    
    if use_async:
        logger.info("Generating initial population asynchronously")
        
        async def generate_population():
            tasks = []
            semaphore = asyncio.Semaphore(10)  # Limit concurrent requests to 10
            
            async def create_individual():
                async with semaphore:
                    max_attempts = 5
                    for attempt in range(max_attempts):
                        try:
                            # Calculate exponential backoff delay
                            backoff_time = 0.1 * (2 ** attempt)  # 0.5s, 1s, 2s, 4s, 8s
                            
                            logger.debug(f"Async: Generating individual, attempt {attempt+1}/{max_attempts}")
                            result = await async_single_call(
                                client, base64_image, x, y, model=model, system_prompt=system_prompt
                            )
                            if result is not None:
                                return result
                            
                            logger.warning(f"Async: Failed attempt {attempt+1}/{max_attempts}, waiting {backoff_time}s before retry")
                            await asyncio.sleep(backoff_time)
                        except Exception as e:
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
            
        logger.info(f"Generated {len(population)} individuals asynchronously")
    
    else:
        logger.info("Generating initial population synchronously")
        # Original synchronous implementation
        for i in tqdm(range(population_size)):
            good = False
            attempts = 0
            while not good and attempts < 5:  # Limit retries
                attempts += 1
                logger.debug(f"Generating individual {i+1}/{population_size}, attempt {attempts}")
                result = single_call(client, base64_image, x, y, model=model, system_prompt=system_prompt)
                if result is not None:
                    population.append(result)
                    good = True
                else:
                    logger.warning(f"Failed to generate individual {i+1}, attempt {attempts}")
                    
            if not good:
                logger.error(f"Failed to generate individual {i+1} after {attempts} attempts")

    # Check if we have a valid population
    if not population:
        error_msg = "Failed to generate any valid population members after multiple attempts"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Handle NaN scores
    for p in population:
        if np.isnan(np.sum(p['score'])):
            logger.warning(f"Found NaN score, setting to -1e8")
            p['score'] = -1e8
            
    population.sort(key=lambda x: x['score'])
    populations.append(population)
    best_pop = population[-1]
    logger.info(f"Initial population best: score={best_pop['score']}, params={best_pop['params']}")
    logger.info(f"Best ansatz: {best_pop['ansatz'][:50]}...")
    
    logger.info(f"Best score: {best_pop['score']}")
    logger.info(f"Best ansatz: {best_pop['ansatz'][:50]}...")
    logger.info(f"Best params: {best_pop['params']}")
    
    if best_pop['score'] > -exit_condition:
        logger.info("Exit condition met after initial population")
        return populations

    # Evolution loop
    for generation in range(num_of_generations-1):
        logger.info(f"Starting generation {generation+1}/{num_of_generations-1}")
        
        logger.debug("Computing selection probabilities")
        scores = np.array([ind['score'] for ind in population])
        finite_scores = scores[np.isfinite(scores)]
        normalized_scores = (scores - np.min(finite_scores)) / (np.max(finite_scores) - np.min(finite_scores) + 1e-6)
        exp_scores = np.exp((normalized_scores - np.max(normalized_scores))/temperature)
        exp_scores = np.nan_to_num(exp_scores, nan=0.0)
        if np.all(exp_scores == 0):
            logger.warning("All selection probabilities are zero, using uniform distribution")
            exp_scores = np.ones_like(exp_scores)
        probs = exp_scores / np.sum(exp_scores)
        
        logger.debug("Selecting parents for next generation")
        selected_population = [np.random.choice(populations[-1], size=2,
                                               p=probs, replace=True) for _ in range(population_size)]

        func_lists = [[pops[0]['ansatz'],pops[1]['ansatz']] for pops in selected_population]
        
        population = []
        if elite:
            logger.debug("Using elitism - keeping best individual")
            population.append(best_pop)
            
        logger.info(f"Generating {population_size} new individuals")
        
        if use_async:
            logger.info(f"Generation {generation+1}: Using async mode for population generation")
            
            async def generate_generation_population():
                tasks = []
                semaphore = asyncio.Semaphore(10)  # Limit concurrent requests to 10
                
                async def create_individual(idx):
                    func_list = func_lists[idx]
                    async with semaphore:
                        for attempt in range(5):  # Limit retries
                            try:
                                logger.debug(f"Async: Generation {generation+1}: Creating individual {idx+1}/{population_size}, attempt {attempt+1}")
                                result = await async_single_call(
                                    client, base64_image, x, y, model=model,
                                    function_list=func_list, system_prompt=system_prompt
                                )
                                if result is not None:
                                    return result
                                logger.warning(f"Async: Generation {generation+1}: Failed attempt {attempt+1} for individual {idx+1}")
                            except Exception as e:
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
            logger.info(f"Generation {generation+1}: Generated {len(gen_population)} individuals asynchronously")
        
        else:
            # Original synchronous implementation
            for funcs in tqdm(range(population_size - (1 if elite else 0))):
                good = False
                attempts = 0
                while not good and attempts < 5:  # Limit retries
                    attempts += 1
                    logger.debug(f"Generation {generation+1}: Creating individual {funcs+1}/{population_size}, attempt {attempts}")
                    result = single_call(client, base64_image, x, y, model=model,
                                        function_list=func_lists[funcs], system_prompt=system_prompt)
                    if result is not None:
                        population.append(result)
                        good = True
                    else:
                        logger.warning(f"Failed to generate individual {funcs+1}, attempt {attempts}")
                        
                if not good:
                    logger.error(f"Failed to generate individual {funcs+1} after {attempts} attempts")
        
        population.sort(key=lambda x: x['score'])
        best_pop = population[-1]
        populations.append(population)
        
        logger.info(f"Generation {generation+1} best: score={best_pop['score']}, params={best_pop['params']}")
        logger.debug(f"Best ansatz: {best_pop['ansatz'][:50]}...")
        
        print("Best score: ", best_pop['score'])
        print("Best ansatz: ", best_pop['ansatz'])
        print("Best params: ", best_pop['params'])
        
        if best_pop['score'] > -exit_condition:
            logger.info(f"Exit condition met after generation {generation+1}")
            print("Exit condition met.")
            return populations
    
    logger.info(f"Genetic algorithm completed after {num_of_generations} generations")
    return populations


def kan_to_symbolic(model, client, population=10, generations=3, temperature=0.1, gpt_model="openai/gpt-4o-mini", exit_condition=1e-3, verbose=0, use_async=False):
    """
    Converts a given kan model symbolic representations using llmsr.
    Parameters:
        model (object): The kan model.
        client (object): The openai client object used to access the llm.
        population (int, optional): The population size for the genetic algorithm. Default is 10.
        generations (int, optional): The number of generations for the genetic algorithm. Default is 3.
        temperature (float, optional): The temperature parameter for the genetic algorithm. Default is 0.1.
        gpt_model (str, optional): The GPT model to use for generating symbolic functions. Default is "openai/gpt-4o-mini".
        exit_condition (float, optional): The exit condition for the genetic algorithm. Default is 1e-3.
        verbose (int, optional): Verbosity level for logging. Default is 0.
        use_async (bool, optional): Whether to use asynchronous processing for population generation. Default is False.
    Returns:
        - res_fcts (dict): A dictionary mapping layer, input, and output indices to their corresponding symbolic functions.
    """
    logger.debug(f"Starting KAN to symbolic conversion with population={population}, generations={generations}")
    logger.debug(f"KAN model has {len(model.width_in)} layers")

    res, res_fcts = 'Sin', {}
    
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
    
    logger.info("Processing KAN model connections")
    for l in range(len(model.width_in) - 1):
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l + 1]):
                total_connections += 1
                logger.debug(f"Processing connection ({l},{i},{j})")
                
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
                    logger.info(f"Generating plot for activation function ({l},{i},{j}) - this is what we're fitting.")
                    fig, ax = plt.subplots()
                    plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                    plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                    base64_image = generate_base64_image(fig, ax, x, y)
                    print((l,i,j))
                    plt.show()
                    
                    # Get activation function mask
                    mask = model.act_fun[l].mask
                    
                    # Run genetic algorithm to find symbolic expression
                    try:
                        logger.info(f"Running genetic algorithm for connection ({l},{i},{j})")
                        res = run_genetic(
                            client, base64_image, x, y, population, generations, 
                            temperature=temperature, model=gpt_model, 
                            system_prompt=None, elite=False, 
                            exit_condition=exit_condition, for_kan=True,
                            use_async=use_async
                        )
                        res_fcts[(l,i,j)] = res
                        logger.info(f"Successfully found expression for connection ({l},{i},{j})")
                        
                    except Exception as e:
                        logger.error(f"Error in genetic algorithm for connection ({l},{i},{j}): {e}", exc_info=True)
                        print(e)
                        res_fcts[(l,i,j)] = res
    
    # Clean up
    logger.debug("Cleaning up matplotlib resources")
    try:
        ax.clear()
        plt.close()
    except Exception as e:
        logger.info(f"Could not clean up matplotlib resources: {e}. Not a cause for concern.")
    
    # Log summary
    logger.info(f"KAN conversion complete: {total_connections} total connections")
    logger.info(f"Connection breakdown: {symbolic_connections} symbolic, {zero_connections} zero, {processed_connections} processed")
    
    return res_fcts