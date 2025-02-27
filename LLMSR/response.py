import re
import numpy as np
from scipy import special

def extract_ansatz(response):
    """
    Extracts the ansatz and the largest parameter index from the given response.
    Args:
        response (object): The response object from the llm call.
    Returns:
        tuple: A tuple containing:
            - ansatz (str): The extracted ansatz string.
            - largest_entry (int): The largest parameter index plus one.
    """

    text = response.choices[0].message.content
    ansatz = text.split('\n')[1]
    if ansatz.startswith("_curve") or ansatz.startswith("curve"):
        ansatz = ansatz.split(":", 1)[1]

    # Extract the values of params from the string
    params_values = re.findall(r'params\[(\d+)\]', ansatz)

    # Convert the extracted values to integers
    params_values = list(map(int, params_values))

    # Find the largest entry
    largest_entry = max(params_values) + 1
    return ansatz, largest_entry

def fun_convert(ansatz):
    """
    Converts a string representation of a lambda function into an actual lambda function.
    Args:
        ansatz (str): A string representing the body of a lambda function. 
                      For example, "x + 2" would be converted to a lambda function equivalent to `lambda x: x + 2`.
    Returns:
        function: A lambda function created from the input string.
    Example:
        >>> func = fun_convert("x + 2")
        >>> func(3)
        5
    """

    ansatz = "lambda x,*params: " + ansatz
    curve = eval(ansatz)
    return curve