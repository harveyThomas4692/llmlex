import requests
import openai
import logging
import time
import os
import threading
from functools import wraps

# Get module logger
logger = logging.getLogger("LLMLEx.llm")

# Default API call rate limiter values
DEFAULT_MAX_CALLS_PER_MINUTE = 120
DEFAULT_TEST_MAX_CALLS_PER_MINUTE = 10  # More restrictive rate limit for tests

# Global tracking variables for rate limiting
_last_api_call_time = 0
_api_call_count = 0
_api_call_times = []  # Rolling window of call times
_rate_limit_lock = None  # Will be initialized later as threading.Lock()

# Get rate limit from environment or use default
MAX_CALLS_PER_MINUTE = int(os.environ.get('LLMLEx_MAX_CALLS_PER_MINUTE', DEFAULT_MAX_CALLS_PER_MINUTE))

# Check if we're in test mode
if os.environ.get('LLMLEx_TEST_REAL_API'):
    # Use more restrictive test rate limit
    MAX_CALLS_PER_MINUTE = int(os.environ.get('LLMLEx_TEST_MAX_CALLS_PER_MINUTE', DEFAULT_TEST_MAX_CALLS_PER_MINUTE))
    logger.info(f"Test mode detected - using restricted rate limit: {MAX_CALLS_PER_MINUTE} calls/min")

# Minimum time between API calls in seconds
MIN_TIME_BETWEEN_CALLS = 60.0 / MAX_CALLS_PER_MINUTE

# Initialize the lock
_rate_limit_lock = threading.Lock()

def rate_limit_api_call(func):
    """
    Decorator to enforce rate limits on API calls.
    This ensures we don't exceed the maximum calls per minute allowed.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _last_api_call_time, _api_call_times
        
        with _rate_limit_lock:
            current_time = time.time()
            
            # Clean up old call times (older than 60 seconds)
            _api_call_times = [t for t in _api_call_times if current_time - t < 60]
            
            # Check if we're about to exceed the rate limit
            if len(_api_call_times) >= MAX_CALLS_PER_MINUTE:
                # Calculate how long to wait
                oldest_call = min(_api_call_times) if _api_call_times else current_time - 60
                wait_time = 60 - (current_time - oldest_call)
                
                if wait_time > 0:
                    logger.warning(f"Rate limit reached! Waiting {wait_time:.2f} seconds before making API call")
                    # Release the lock while we sleep
                    _rate_limit_lock.release()
                    try:
                        time.sleep(wait_time)
                    finally:
                        # Re-acquire the lock
                        _rate_limit_lock.acquire()
                    current_time = time.time()  # Update current time after waiting
            
            # Record this call
            _api_call_times.append(current_time)
            _last_api_call_time = current_time
        
        # Make the API call outside the lock to avoid blocking other calls
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in rate-limited API call: {e}")
            raise
    
    return wrapper

def clear_rate_limit_lock():
    global _last_api_call_time, _api_call_times
    _last_api_call_time = 0
    _api_call_times = []
    logger.debug("Rate limit state cleared")

# Async version of the rate limiter
def async_rate_limit_api_call(func):
    """
    Decorator to enforce rate limits on async API calls.
    This ensures we don't exceed the maximum calls per minute allowed.
    
    This version uses a thread lock for rate limiting but releases it during the await.
    """
    import asyncio
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        global _last_api_call_time, _api_call_times
        
        # Use a lock for thread safety when checking/updating the rate limit
        need_to_wait = False
        wait_time = 0
        
        with _rate_limit_lock:
            current_time = time.time()
            
            # Clean up old call times (older than 60 seconds)
            _api_call_times = [t for t in _api_call_times if current_time - t < 60]
            
            # Check if we're about to exceed the rate limit
            if len(_api_call_times) >= MAX_CALLS_PER_MINUTE:
                # Calculate how long to wait
                oldest_call = min(_api_call_times) if _api_call_times else current_time - 60
                wait_time = 60 - (current_time - oldest_call)
                
                if wait_time > 0:
                    need_to_wait = True
        
        # Do the waiting outside the lock so we don't block other threads/tasks
        if need_to_wait:
            logger.warning(f"Rate limit reached! Waiting {wait_time:.2f} seconds before making async API call")
            try:
                await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error(f"Error during asyncio.sleep: {e}, falling back to time.sleep")
                # Fallback to synchronous sleep if asyncio.sleep fails
                time.sleep(wait_time)
            
            # Re-acquire lock after waiting to update the tracking info
            with _rate_limit_lock:
                current_time = time.time()
                # Record this call
                _api_call_times.append(current_time)
                _last_api_call_time = current_time
        else:
            # If no waiting needed, record the call time with the lock
            with _rate_limit_lock:
                current_time = time.time()
                _api_call_times.append(current_time)
                _last_api_call_time = current_time
        
        # Make the API call outside the lock
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in rate-limited API call: {e}")
            raise
    
    return wrapper

def check_key_limit(client):
    '''
    Checks the remaining limit for the provided API key.
    Args:
        client (object): An object containing the API key and base URL.
    Returns:
        int: The remaining limit for the API key if the request is successful.
        str: The error message if the request fails.
    '''
    logger.debug("Checking API key usage limit")
    
    # Create request headers
    headers = {"Authorization": f"Bearer {client.api_key}",}
    
    try:
        # Send request to check key limit
        base_url = str(client.base_url).rstrip('/')
        url_for_key_limit = f"{base_url}/auth/key"
        logger.debug(f"Sending request to {url_for_key_limit}")
        response = requests.get(url_for_key_limit, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Extract and return limit data
            limit_remaining = response.json()["data"]['limit_remaining']
            logger.info(f"API key check successful. Remaining limit: {limit_remaining}")
            return limit_remaining
        else:
            # Log error and return response text
            logger.error(f"API key check failed with status code {response.status_code}, sent request to {url_for_key_limit} ")
            print(f"Request failed with status code {response.status_code}, sent request to {url_for_key_limit}")
            return response.text
            
    except Exception as e:
        # Log and re-raise any exceptions
        logger.error(f"Error checking API key limit: {e}", exc_info=True)
        raise


def check_key_usage(client):
    '''
    Checks the current spend for the provided API key.
    Args:
        client (object): An object containing the API key and base URL.
    Returns:
        float: The current spend for the API key if the request is successful.
        str: The error message if the request fails.
    '''
    logger.debug("Checking API key current spend")
    
    # Create request headers
    headers = {"Authorization": f"Bearer {client.api_key}",}
    
    try:
        # Send request to check key usage
        # Ensure no duplicate slashes between base_url and endpoint
        base_url = str(client.base_url).rstrip('/')
        url_for_key_usage = f"{base_url}/auth/key"
        response = requests.get(url_for_key_usage, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Extract and return spend data
            current_usage = response.json()["data"]['usage']
            logger.info(f"API key usage check successful. Current usage: {current_usage}")
            return current_usage
        else:
            # Log error and return response text
            logger.error(f"API key usage check failed with status code {response.status_code}, sent request to {url_for_key_usage}")
            print(f"Request failed with status code {response.status_code}, sent request to {url_for_key_usage}")
            return response.text
            
    except Exception as e:
        # Log and re-raise any exceptions
        logger.error(f"Error checking API key usage: {e}", exc_info=True)
        raise

def check_credits_remaining(client):
    '''
    Checks the current credit balance for the provided API key.
    Args:
        client (object): An object containing the API key and base URL.
    Returns:
        float: The current credit balance for the API key if the request is successful.
    '''
    logger.debug("Checking API key credit balance")
    
    # Create request headers
    headers = {"Authorization": f"Bearer {client.api_key}",}
    try:
        # Send request to check credit balance
        logger.debug(f"Sending request to {client.base_url}/api/v1/auth/key")
        response = requests.get(f"{client.base_url}/api/v1/auth/key", headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Extract and return credit balance data
            data = response.json()["data"]
            credit_limit = data.get("limit", 0)
            credit_usage = data.get("usage", 0)
            credit_balance = credit_limit - credit_usage if credit_limit is not None else "unlimited"
            logger.info(f"API key credit balance check successful. Current balance: {credit_balance}")
            return credit_balance
        else:
            # Log error and return response text
            logger.error(f"API key credit balance check failed with status code {response.status_code}")
            print(f"Request failed with status code {response.status_code}")
            return response.text
            
    except Exception as e:
        # Log and re-raise any exceptions
        logger.error(f"Error checking API key credit balance: {e}", exc_info=True)
        raise
    
def get_prompt(function_list=None):
    """
    Generates the user prompt given a list of functions.
    Args:
        function_list (list, optional): A list of strings where each string is a mathematical expression to be used in the lambda functions. 
                                        If None, defaults to ["params[0]"].
    Returns:
        str: A string containing the generated user prompt with lambda function definitions.
    """
    logger.debug(f"Generating prompt with function_list: {function_list}")
    
    # Use default if no function list provided
    if function_list is None:
        function_list = [("params[0]", 1)]
        logger.debug("No function list provided, using default: ['params[0]']")
    
    # Build the prompt
    prompt = "import numpy as np \n"
    for n in range(len(function_list)):
        prompt += f"curve_{n} = lambda x, *params: {function_list[n][0]} \n"
    prompt += f"curve_{len(function_list)} = lambda x, *params:"

    logger.debug(f"Generated prompt with {len(function_list)} functions")
    return prompt

@rate_limit_api_call
def call_model(client, model, image, prompt, system_prompt=None):
    """
    In initiates a single call of the llm given an image. 
    Args:
        client (object): The client object used to interact with the model.
        model (str): The name or identifier of the model to be used.
        image (str): The image data encoded in base64 format.
        prompt (str): The text prompt provided by the user.
        system_prompt (str, optional): The system prompt to guide the model's response. If none provided it defaults to:
                                        "Give an improved ansatz in python to the list for the image. Follow on from the users text with no explaining.
                                        Params can be any length. If there's some noise in the data, give preference to simpler functions"
    Returns:
        dict: The response from the model, typically containing the generated text or other relevant information.
    """

    logger.debug(f"Calling model {model}")
    
    # Set default system prompt if not provided
    if system_prompt is None:
        #system_prompt = ("Give an improved ansatz to the list for the image. Follow on from the users text with no explaining."
        #                 "Params can be any length. If there's some noise in the data, give preference to simpler functions"
        # THIS IS THE SYSTEM PROMPT FOR THE SYNC MODEL - see LLMLEx.py for the async version
        system_prompt = ("You are a symbolic regression expert. Analyze the data in the image and provide an improved mathematical ansatz. "
                         "Respond with ONLY the ansatz formula, without any explanation or commentary. Ensure it is in valid python. You may use numpy functions. "
                         "params is a list of parameters that can be of any length or complexity. "
                        )
                         #"Since the data contains noise, prioritize simpler, more elegant functions that capture the underlying pattern rather than fitting every point. ")
        logger.debug("Using default system prompt: \n" + system_prompt)
    
    # Track image size for debugging purposes
    image_size = len(image) if image else 0
    logger.debug(f"Image size: {image_size} characters (base64)")
    logger.debug(f"Prompt length: {len(prompt)} characters")
    
    try:
        # Create and send the API request
        logger.debug("Creating chat completion request")
        response = client.chat.completions.create(
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
        
        # Log response info
        try:
            token_usage = response.usage.total_tokens if hasattr(response, 'usage') else 'unknown'
            logger.debug(f"Model response received. Total tokens: {token_usage}")
            logger.debug(f"Response finish reason: {response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else 'unknown'}")
        except:
            logger.debug("Could not access token usage information")
        
        return response
        
    except Exception as e:
        # Log and re-raise any exceptions
        logger.error(f"Error calling model {model}: {e}", exc_info=True)
        raise

@async_rate_limit_api_call
async def async_call_model(client, model, image, prompt, system_prompt=None):
    """
    Asynchronous version of call_model. Initiates a call to the LLM with an image.
    
    Args:
        client (object): The client object used to interact with the model.
        model (str): The name or identifier of the model to be used.
        image (str): The image data encoded in base64 format.
        prompt (str): The text prompt provided by the user.
        system_prompt (str, optional): The system prompt to guide the model's response.
                                       If None, uses the default system prompt.
    
    Returns:
        dict: The response from the model.
    """
    import asyncio
    
    logger.debug(f"Async calling model {model}")
    
    # Set default system prompt if not provided
    if system_prompt is None:
        system_prompt = ("You are a symbolic regression expert. Analyze the data in the image and provide an improved mathematical ansatz. "
                         "Respond with ONLY the ansatz formula, without any explanation or commentary. Ensure it is in valid python. You may use numpy functions. "
                         "params is a list of parameters that can be of any length or complexity. "
                        )
                         #"Since the data contains noise, prioritize simpler, more elegant functions that capture the underlying pattern rather than fitting every point. ")
        logger.debug("Using default system prompt: \n" + system_prompt)
    
    # Track image size for debugging purposes
    image_size = len(image) if image else 0
    logger.debug(f"Image size: {image_size} characters (base64)")
    logger.debug(f"Prompt length: {len(prompt)} characters")
    
    try:
        # Check if the client supports async operations directly
        if hasattr(client.chat.completions, 'acreate'):
            # Use the async API if available
            logger.debug("Creating async chat completion request")
            response = await client.chat.completions.acreate(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
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
                            {"role": "system", "content": system_prompt},
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
            token_usage = response.usage.total_tokens if hasattr(response, 'usage') else 'unknown'
            logger.debug(f"Model response received. Total tokens: {token_usage}")
            logger.debug(f"Response finish reason: {response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else 'unknown'}")
        except:
            logger.debug("Could not access token usage information")
        
        return response
        
    except Exception as e:
        # Log and re-raise any exceptions
        logger.error(f"Error calling model {model} asynchronously: {e}", exc_info=True)
        # Check for insufficient credits error (code 402)
        is_credit_error = (
            (hasattr(e, 'code') and e.code == 402) or
            (isinstance(e, dict) and 'error' in e and 'code' in e['error'] and e['error']['code'] == 402) or
            (str(e).find("402") != -1 and str(e).find("Insufficient credits") != -1)
        )
        
        if is_credit_error:
            logger.warning("Insufficient credits error detected. Pausing execution...")
            import time
            
            # Pause and poll for credits
            while True:
                logger.info("Waiting for credits to be added. Will check again in 60 seconds.")
                time.sleep(60)
                
                try:
                    # Check if credits have been added
                    logger.info("Checking if credits have been added...")
                    if check_credits_remaining(client) > 1.0 or check_credits_remaining(client) == "unlimited":
                        logger.info("Credits added. Resuming execution.")
                        return await async_call_model(client, model, image, prompt, system_prompt)
                except Exception as credit_check_error:
                    logger.error(f"Error checking credits: {credit_check_error}")
        
        raise
 