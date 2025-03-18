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
        logger.debug(f"No parameters found in initial ansatz extraction, searching full response")
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
            logger.debug(f"No parameters found in ansatz or full response: {text[:200]}...")
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
    # Check for lambda in ansatz and remove it
    if "lambda" in ansatz_str and ":" in ansatz_str:
        lambda_start = ansatz_str.find("lambda")
        colon_pos = ansatz_str.find(":", lambda_start)
        if colon_pos > lambda_start:
            logger.error(f"Removing lambda from ansatz: {ansatz_str[lambda_start:colon_pos+1]}. There shouldn't be a lambda in the ansatz. {ansatz_str}")
            ansatz_str = ansatz_str[:lambda_start] + ansatz_str[colon_pos+1:]
        else:
            raise ValueError("Lambda expressions not allowed in ansatz")
    
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
    return curve, num_params, lambda_str

class APICallStats:
    def __init__(self):
        self.success_count = 0
        
        # Track errors by stage of processing
        self.stages = {
            "api_call": {"success": 0, "failure": 0},
            "ansatz_extraction": {"success": 0, "failure": 0},
            "function_conversion": {"success": 0, "failure": 0},
            "curve_fitting": {"success": 0, "failure": 0}
        }
        
        # Track detailed error types
        self.error_types = {
            # API errors
            "rate_limit": 0,
            "api_connection": 0,
            "timeout": 0,
            
            # Extraction errors
            "no_parameters": 0,
            "llm_refusal": 0,
            "invalid_response": 0,
            "empty_response": 0,
            
            # Function conversion errors
            "syntax_error": 0,
            "name_error": 0,
            "type_error": 0,
            
            # Curve fitting errors
            "convergence_error": 0,
            "singular_matrix": 0,
            "numerical_error": 0,
            
            # Other errors
            "other": 0
        }
        
        # Track validation issues
        self.validation_issues = {
            "scalar_output": 0,       # Function returns scalar instead of array
            "shape_mismatch": 0,      # Function returns wrong shape
            "nan_values": 0,          # Function returns NaN values
            "inf_values": 0,          # Function returns inf values
            "evaluation_error": 0,    # General errors in function evaluation
            "other_validation": 0     # Other validation issues
        }
        
        # Track fitting warnings
        self.fitting_warnings = {
            "invalid_sqrt": 0,        # "invalid value encountered in sqrt"
            "covariance_estimation": 0, # "Covariance of the parameters could not be estimated"
            "other_warnings": 0,        # Other warning types
            "invalid_log": 0,          # "invalid value encountered in log"
            "invalid_power": 0,        # "invalid value encountered in power"
        }
        
        # Details for specific validation issues
        self.validation_details = {}
    
    def add_success(self):
        """Record a completely successful API call cycle"""
        self.success_count += 1
    
    def stage_success(self, stage):
        """Record success for a specific stage"""
        if stage in self.stages:
            self.stages[stage]["success"] += 1
        
    def stage_failure(self, stage, error, error_type=None):
        """
        Record failure for a specific stage
        
        Args:
            stage: The processing stage where the error occurred
            error: The exception object or error message
            error_type: Optional specific error type to increment
        """
        if stage in self.stages:
            self.stages[stage]["failure"] += 1
        
        # If no specific error type was provided, categorize based on the error
        if error_type is None:
            self._categorize_error(error)
        else:
            if error_type in self.error_types:
                self.error_types[error_type] += 1
            else:
                self.error_types["other"] += 1
    
    def add_validation_issue(self, issue_type, details=None):
        """
        Record a validation issue
        
        Args:
            issue_type: Type of validation issue (e.g., 'scalar_output', 'nan_values')
            details: Optional details about the issue
        """
        if issue_type in self.validation_issues:
            self.validation_issues[issue_type] += 1
        else:
            self.validation_issues["other_validation"] += 1
        
        # Store details if provided
        if details:
            if issue_type not in self.validation_details:
                self.validation_details[issue_type] = []
            self.validation_details[issue_type].append(details)
    
    def add_fitting_warning(self, warning_type, details=None):
        """
        Record a fitting warning
        
        Args:
            warning_type: Type of fitting warning (e.g., 'invalid_sqrt', 'covariance_estimation')
            details: Optional details about the warning
        """
        if warning_type in self.fitting_warnings:
            self.fitting_warnings[warning_type] += 1
        else:
            logger.warning(f"Unknown fitting warning type: {warning_type}, adding it to tracking")
            self.fitting_warnings[warning_type] += 1
    
    def _categorize_error(self, error):
        """Categorize an error based on its type and message"""
        error_str = str(error).lower()
        
        # API errors
        if isinstance(error, Exception) and any(term in error_str for term in ["rate limit", "ratelimit", "too many requests"]):
            self.error_types["rate_limit"] += 1
        elif isinstance(error, Exception) and any(term in error_str for term in ["connect", "connection", "network", "timeout", "timed out"]):
            self.error_types["api_connection"] += 1
        elif isinstance(error, Exception) and "timeout" in error_str:
            self.error_types["timeout"] += 1
            
        # Extraction errors
        elif isinstance(error, ValueError) and "no parameters found" in error_str and any(phrase in error_str.lower() for phrase in ["can't assist", "can't help", "i'm sorry", "i apologise"]):
            self.error_types["llm_refusal"] += 1
        elif isinstance(error, ValueError) and "no parameters found" in error_str:
            self.error_types["no_parameters"] += 1
        elif isinstance(error, ValueError) and any(term in error_str for term in ["format", "unexpected", "not recognized"]):
            self.error_types["invalid_response"] += 1
        elif isinstance(error, ValueError) and any(term in error_str for term in ["empty", "no content"]):
            self.error_types["empty_response"] += 1
            
        # Function conversion errors
        elif isinstance(error, SyntaxError):
            self.error_types["syntax_error"] += 1
        elif isinstance(error, NameError):
            self.error_types["name_error"] += 1
        elif isinstance(error, TypeError):
            self.error_types["type_error"] += 1
            
        # Curve fitting errors
        elif any(term in error_str for term in ["convergence", "converge", "maximum number of iterations"]):
            self.error_types["convergence_error"] += 1
        elif any(term in error_str for term in ["singular", "invert", "invertible"]):
            self.error_types["singular_matrix"] += 1
        elif any(term in error_str for term in ["overflow", "underflow", "divide by zero", "domain", "nan", "inf"]):
            self.error_types["numerical_error"] += 1
            
        # Other errors
        else:
            self.error_types["other"] += 1
    
    def total_failures(self):
        """Get the total number of failures across all error types"""
        return sum(self.error_types.values())
    
    def total_validation_issues(self):
        """Get the total number of validation issues"""
        return sum(self.validation_issues.values())
    
    def total_fitting_warnings(self):
        """Get the total number of fitting warnings"""
        return sum(self.fitting_warnings.values())
    
    def get_stage_stats(self, stage):
        """Get statistics for a specific processing stage"""
        if stage not in self.stages:
            return None
            
        stats = self.stages[stage]
        total = stats["success"] + stats["failure"]
        success_rate = (stats["success"] / total * 100) if total > 0 else 0
        
        return {
            "success": stats["success"],
            "failure": stats["failure"],
            "total": total,
            "success_rate": success_rate
        }
        
    def __str__(self):
        """Generate a comprehensive statistics report"""
        total_calls = self.success_count + self.total_failures()
        success_rate = (self.success_count / total_calls * 100) if total_calls > 0 else 0
        
        result = []
        result.append(f"API Call Statistics:")
        result.append(f"  Successful calls (end-to-end): {self.success_count}")
        result.append(f"  Failed calls: {self.total_failures()}")
        result.append(f"  Success rate: {success_rate:.2f}%")
        
        # Add stage-specific statistics
        result.append("\nBreakdown by processing stage:")
        for stage, stats in self.stages.items():
            total = stats["success"] + stats["failure"]
            if total > 0:
                success_rate = (stats["success"] / total * 100)
                result.append(f"  {stage.replace('_', ' ').title()}: {stats['success']} succeeded, {stats['failure']} failed ({success_rate:.2f}% success)")
        
        # Add error type statistics
        result.append("\nError types:")
        
        # Group error types by category
        categories = {
            "API Errors": ["rate_limit", "api_connection", "timeout"],
            "Extraction Errors": ["no_parameters", "invalid_response", "empty_response", "llm_refusal"],
            "Function Errors": ["syntax_error", "name_error", "type_error"],
            "Fitting Errors": ["convergence_error", "singular_matrix", "numerical_error"],
            "Other": ["other"]
        }
        temp_result = []
        for category, error_types in categories.items():
            category_errors = [(error_type, self.error_types[error_type]) for error_type in error_types if self.error_types[error_type] > 0]
            if category_errors:
                temp_result.append(f"  {category}:")
                for error_type, count in category_errors:
                    temp_result.append(f"    - {error_type.replace('_', ' ')}: {count}")
        if len(temp_result)==0:
            result.append("  No errors")
        else:
            result.extend(temp_result)
            
        # Add validation issue statistics
        total_validation = self.total_validation_issues()
        if total_validation > 0:
            result.append("\nValidation issues:")
            for issue_type, count in self.validation_issues.items():
                if count > 0:
                    result.append(f"  - {issue_type.replace('_', ' ')}: {count}")
        
        # Add fitting warning statistics
        total_warnings = self.total_fitting_warnings()
        if total_warnings > 0:
            result.append("\nFitting warnings:")
            for warning_type, count in self.fitting_warnings.items():
                if count > 0:
                    result.append(f"  - {warning_type.replace('_', ' ')}: {count}")
        
        return "\n".join(result)