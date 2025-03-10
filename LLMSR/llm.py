import requests
import openai
import logging
import time
import os
import threading
from functools import wraps

# Get module logger
logger = logging.getLogger("LLMSR.llm")

# Default API call rate limiter values
DEFAULT_MAX_CALLS_PER_MINUTE = 20
DEFAULT_TEST_MAX_CALLS_PER_MINUTE = 3  # More restrictive rate limit for tests

# Global tracking variables for rate limiting
_last_api_call_time = 0
_api_call_count = 0
_api_call_times = []  # Rolling window of call times
_rate_limit_lock = None  # Will be initialized later as threading.Lock()

# Get rate limit from environment or use default
MAX_CALLS_PER_MINUTE = int(os.environ.get('LLMSR_MAX_CALLS_PER_MINUTE', DEFAULT_MAX_CALLS_PER_MINUTE))

# Check if we're in test mode
if os.environ.get('LLMSR_TEST_REAL_API'):
    # Use more restrictive test rate limit
    MAX_CALLS_PER_MINUTE = int(os.environ.get('LLMSR_TEST_MAX_CALLS_PER_MINUTE', DEFAULT_TEST_MAX_CALLS_PER_MINUTE))
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
        logger.debug(f"Sending request to {client.base_url}/auth/key")
        response = requests.get(f"{client.base_url}/auth/key", headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Extract and return limit data
            limit_remaining = response.json()["data"]['limit_remaining']
            logger.info(f"API key check successful. Remaining limit: {limit_remaining}")
            return limit_remaining
        else:
            # Log error and return response text
            logger.error(f"API key check failed with status code {response.status_code}")
            print(f"Request failed with status code {response.status_code}")
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
        logger.debug(f"Sending request to {client.base_url}/auth/key")
        response = requests.get(f"{client.base_url}/auth/key", headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Extract and return spend data
            current_usage = response.json()["data"]['usage']
            logger.info(f"API key usage check successful. Current usage: {current_usage}")
            return current_usage
        else:
            # Log error and return response text
            logger.error(f"API key usage check failed with status code {response.status_code}")
            print(f"Request failed with status code {response.status_code}")
            return response.text
            
    except Exception as e:
        # Log and re-raise any exceptions
        logger.error(f"Error checking API key usage: {e}", exc_info=True)
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
        function_list = ["params[0]"]
        logger.debug("No function list provided, using default: ['params[0]']")
    
    # Build the prompt
    prompt = "import numpy as np \n"
    for n in range(len(function_list)):
        prompt += f"curve_{n} = lambda x,*params: {function_list[n]} \n"
    prompt += f"curve_{len(function_list)} = lambda x,*params:"

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
                                        "Give an improved ansatz to the list for the image. Follow on from the users text with no explaining.
                                        Params can be any length. If there's some noise in the data, give preference to simpler functions"
    Returns:
        dict: The response from the model, typically containing the generated text or other relevant information.
    """

    logger.debug(f"Calling model {model}")
    
    # Set default system prompt if not provided
    if system_prompt is None:
        system_prompt = ("Give an improved ansatz to the list for the image. Follow on from the users text with no explaining."
                         "Params can be any length. If there's some noise in the data, give preference to simpler functions.")
        logger.debug("Using default system prompt")
    
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