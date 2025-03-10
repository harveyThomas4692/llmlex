import re
import numpy as np
from scipy import special
import logging

# Get module logger
logger = logging.getLogger("LLMSR.response")

def extract_ansatz(response):
    """
    Extracts the ansatz and the largest parameter index from the given response.
    Args:
        response (Union[str, object]): The response string or object from the llm call.
    Returns:
        tuple: A tuple containing:
            - ansatz (str): The extracted ansatz string.
            - largest_entry (int): The largest parameter index plus one.
    """
    logger.debug("Extracting ansatz from model response")
    
    # Handle different response types
    if isinstance(response, str):
        text = response
    elif hasattr(response, 'choices') and len(response.choices) > 0:
        if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
            text = response.choices[0].message.content
        elif hasattr(response.choices[0], 'text'):
            text = response.choices[0].text
        else:
            logger.error("Unexpected response format - can't find content")
            raise ValueError("Response format not recognized: no content found")
    else:
        logger.error(f"Unexpected response type: {type(response)}")
        raise ValueError(f"Unexpected response type: {type(response)}")
        
    logger.debug(f"Response content length: {len(text)} characters")
    
    # Split by newlines
    lines = text.split('\n')
    logger.debug(f"Response has {len(lines)} lines")
    
    # Look for lines containing "params"
    potential_ansatz_lines = []
    for line in lines:
        if "params" in line:
            potential_ansatz_lines.append(line)
    
    # If we found potential ansatz lines, use the first one
    if potential_ansatz_lines:
        ansatz = potential_ansatz_lines[0]
        logger.debug(f"Using first function found: {ansatz[:50]}{'...' if len(ansatz) > 50 else ''}")
    else:
        # Fallback to original approach
        if len(lines) < 2:
            logger.debug("Response has fewer than 2 lines, using first line")
            ansatz = lines[0]
        else:
            ansatz = lines[1]
        
        # If ansatz is empty, go to the next line
        if not ansatz.strip() and len(lines) > 2:
            ansatz = lines[2]
    
    # Handle code blocks
    if "```" in ansatz:
        ansatz = ansatz.split("```", 1)[1]
    
    # Extract the actual function part if it's in a variable assignment
    if "=" in ansatz:
        ansatz = ansatz.split("=", 1)[1].strip()
    
    # If it's a lambda function, remove the lambda part
    if "lambda" in ansatz and ":" in ansatz:
        ansatz = ansatz.split(":", 1)[1].strip()
    
    # Clean up the ansatz if it starts with curve_X
    if ansatz.startswith("_curve") or ansatz.startswith("curve"):
        logger.debug("Ansatz starts with curve prefix, removing it")
        ansatz = ansatz.split(":", 1)[1]
    
    logger.debug(f"Extracted raw ansatz: {ansatz[:50]}{'...' if len(ansatz) > 50 else ''}")

    # Extract parameter indices
    logger.debug("Extracting parameter indices from ansatz")
    params_values = re.findall(r'params\[(\d+)\]', ansatz)
    
    # Convert to integers
    params_values = list(map(int, params_values))
    
    if not params_values:
        logger.warning(f"No parameters found in initial ansatz extraction, searching full response")
        # Try to find any expression with params in the entire response
        for line in lines:
            if "params" in line:
                params_values = re.findall(r'params\[(\d+)\]', line)
                if params_values:
                    params_values = list(map(int, params_values))
                    ansatz = line
                    if "=" in ansatz:
                        ansatz = ansatz.split("=", 1)[1].strip()
                    if "lambda" in ansatz and ":" in ansatz:
                        ansatz = ansatz.split(":", 1)[1].strip()
                    logger.debug(f"Found alternative ansatz: {ansatz[:50]}{'...' if len(ansatz) > 50 else ''}")
                    break
        
        # Check for backtick-enclosed expressions like `params[0] + params[1] * x`
        if not params_values:
            backtick_expressions = re.findall(r'`([^`]*params\[\d+\][^`]*)`', text)
            if backtick_expressions:
                ansatz = backtick_expressions[0]
                params_values = re.findall(r'params\[(\d+)\]', ansatz)
                params_values = list(map(int, params_values))
                logger.debug(f"Found backtick-enclosed ansatz: {ansatz[:50]}{'...' if len(ansatz) > 50 else ''}")
        
        if not params_values:
            logger.error(f"No parameters found in ansatz or full response: {text[:200]}...")
            raise ValueError(f"No parameters found in ansatz: '{ansatz}'")
    
    # Find the largest parameter index
    largest_entry = max(params_values) + 1
        
    logger.debug(f"Determined number of parameters: {largest_entry}")
    return ansatz, largest_entry

def fun_convert(ansatz):
    """
    Converts a string representation of a lambda function into an actual lambda function.
    Args:
        ansatz (str or tuple): A string representing the body of a lambda function,
                              or a tuple of (string, largest_entry) from extract_ansatz.
    Returns:
        tuple: A tuple containing:
            - function: A lambda function created from the input string.
            - num_params (int): The number of parameters in the function.
    Example:
        >>> func, num_params = fun_convert("x + 2")
        >>> func(3, 1)
        4
    """
    logger.debug("Converting ansatz string to lambda function")
    
    # Handle if ansatz is a tuple (ansatz, largest_entry) returned from extract_ansatz
    if isinstance(ansatz, tuple) and len(ansatz) == 2:
        ansatz_str, num_params = ansatz
    else:
        # Count parameters if not provided
        ansatz_str = ansatz
        params_values = re.findall(r'params\[(\d+)\]', ansatz_str)
        if params_values:
            num_params = max(map(int, params_values)) + 1
        else:
            # No parameters found
            logger.debug("No parameters found in ansatz")
            raise ValueError("No parameters found in ansatz string")
    
    # Create complete lambda function string
    lambda_str = "lambda x,*params: " + ansatz_str
    logger.debug(f"Full lambda string: {lambda_str[:50]}{'...' if len(lambda_str) > 50 else ''}")
    
    # Evaluate the lambda function
    logger.debug("Evaluating lambda expression")
    try:
        curve = eval(lambda_str)
    except Exception as e:
        logger.debug(f"Failed to evaluate lambda expression: {e}")
        raise
    
    logger.debug("Successfully converted ansatz to lambda function")
    return curve, num_params