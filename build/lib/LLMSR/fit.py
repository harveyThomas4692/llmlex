"""
Curve fitting utilities for symbolic regression.

This module provides functions for fitting mathematical expressions to data points
and evaluating the quality of fit using a robust normalized chi-squared metric.
"""

import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import special
from scipy.optimize import OptimizeWarning, curve_fit, minimize

# Configure JAX
os.environ["JAX_PLATFORMS"] = "cpu"  # Explicitly tell JAX to only use CPU
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize as jax_minimize

# Disable 64-bit precision - for some reason 64-bit doesn't work!
# That's why we iterate and use the regular fit after JAX.
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_platform_name", "cpu")  # Disable JAX on GPU

# Get module logger
logger = logging.getLogger("LLMSR.fit")


#########################################
# CLASSES AND DATASTRUCTURES
#########################################

class FittingError(Exception):
    """Base class for fitting related errors."""
    pass


class ConvergenceError(FittingError):
    """Error raised when fitting fails to converge."""
    pass


class InputError(FittingError):
    """Error raised when there's an issue with input data."""
    pass


# Common optimisation error types and categories
OptimisATION_ERROR_CATEGORIES = {
    "RuntimeError": "Optimisation convergence issues",
    "ValueError": "Parameter or function value issues", 
    "OptimizeWarning": "Optimisation warning",
    "LinAlgError": "Linear algebra computation issue"
}


@dataclass
class FitResult:
    """Class for holding curve fitting results."""
    params: np.ndarray
    n_chi_squared: float
    success: bool = True
    methods_used: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error_msg: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if the fit result is valid (converged and not infinite)."""
        return self.success and not np.isinf(self.n_chi_squared)


class WarningManager:
    """Context manager for handling warnings during fitting."""
    
    def __init__(self, stats=None, context="curve fitting", print_warnings=True):
        self.stats = stats
        self.context = context
        self.original_showwarning = None
        self.warnings_list = []
        self.print_warnings = print_warnings
    
    def __enter__(self):
        self.original_showwarning = warnings.showwarning
        warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: self._warning_handler(
            message, category, filename, lineno, file, line, print_warnings=self.print_warnings
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.showwarning = self.original_showwarning
        return False  # Don't suppress exceptions
    
    def _warning_handler(self, message, category, filename, lineno, file=None, line=None, print_warnings=True):
        """
        Custom warning handler that categorizes and logs fitting warnings.
        """
        warning_msg = str(message).lower()
        self.warnings_list.append((category, warning_msg))
        
        if self.stats is not None and hasattr(self.stats, 'add_fitting_warning'):
            if (category == RuntimeWarning and "invalid value" in warning_msg and "sqrt" in warning_msg) or (
                "nan" in warning_msg or "inf" in warning_msg):
                self.stats.add_fitting_warning('invalid_sqrt')
                if print_warnings:
                    logger.warning(f"{self.context}: invalid value encountered in sqrt - {message}")
            elif category == OptimizeWarning and "covariance" in warning_msg:
                self.stats.add_fitting_warning('covariance_estimation')
                if print_warnings:
                    logger.warning(f"{self.context}: Covariance of the parameters could not be estimated")
            elif "convergence" in warning_msg:
                self.stats.add_fitting_warning('convergence_error')
                if print_warnings:
                    logger.warning(f"{self.context}: convergence warning - {message}")
            elif category == RuntimeWarning and "invalid value" in warning_msg and "log" in warning_msg:
                self.stats.add_fitting_warning('invalid_log')
                if print_warnings:
                    logger.warning(f"{self.context}: invalid value encountered in log")
            elif category == RuntimeWarning and "invalid value" in warning_msg and "power" in warning_msg:
                self.stats.add_fitting_warning('invalid_power')
                if print_warnings:
                    logger.warning(f"{self.context}: invalid value encountered in power")
            else:
                self.stats.add_fitting_warning('other_warnings')
                if print_warnings:
                    logger.warning(f"Warning during {self.context}: {category.__name__}: {message}")


#########################################
# ERROR METRIC AND FIT QUALITY FUNCTIONS
#########################################

def _get_n_chi_squared_from_predictions_jax(x, y, predictions, alpha=0.01, eps_scale=1e-4):
    """Internal function for normalized chi-squared calculation using JAX."""
    # Residuals
    residuals = y - predictions

    # A simple scale measure: median absolute deviation
    median_y = jnp.median(y)
    mad_y = jnp.median(jnp.abs(y - median_y))

    # Mean of absolute y (for additional scale-based floor)
    mean_abs_y = jnp.mean(jnp.abs(y))

    # Define a robust global scale, ensuring it never collapses to zero
    # (ties scale to MAD and a fraction of the mean magnitude)
    global_scale = jnp.maximum(mad_y, alpha * mean_abs_y)
    global_scale = jnp.maximum(global_scale, eps_scale)

    # For each data point, define a local denominator to smoothly handle
    # relative vs. absolute error. This prevents blow-up near zero.
    # delta_i = max(global_scale, alpha * |y_i|)
    abs_y = jnp.abs(y)
    local_scale = jnp.maximum(global_scale, alpha * abs_y)

    # Compute normalized chi-squared
    # (residual^2 / local_scale^2), then average
    n_chi_squared = jnp.mean((residuals ** 2) / (local_scale ** 2))

    return n_chi_squared 

#### NOT USED AT THE MOMENT
def _get_chi_squared_from_predictions_jax_THOMASSORIGINAL(x, y, predictions, eps_scale=1e-6):
    """Internal function for chi-squared calculation using JAX, for Thomas's original code."""
    # Residuals
    residuals = y - predictions
    chi_squared = jnp.mean(residuals**2 / (y**2 + eps_scale))

    return chi_squared 

def get_n_chi_squared_from_predictions(x, y, predictions, alpha=0.01, eps_scale=1e-4):
    """
    Calculate a robust, scale-invariant "normalized chi-squared" from actual values and predictions.
    This is a score which is approximately scale invariant.

    Args:
        x: Input data points (not used, kept for API consistency)
        y: Actual target values (1D array)
        predictions: Predicted values (1D array)
        alpha: Small fraction for relative scale (default 0.01)
        eps_scale: Absolute floor to avoid zero in scale (default 1e-4)

    Returns:
        float: Robust normalized chi-squared value
    """
    return float(_get_n_chi_squared_from_predictions_jax(x, y, predictions, alpha, eps_scale))

def get_n_chi_squared(x, y, curve, params, explain_if_inf=False, string_for_explanation=None):
    """
    Calculate normalized chi-squared for a curve fit.
    
    Args:
        x: Input data points
        y: Target values
        curve: Function to evaluate
        params: Parameters for the curve function
        explain_if_inf: Whether to explain infinite chi-squared values (default: False)
        string_for_explanation: Expression string for explanation (default: None)
        
    Returns:
        float: Normalized chi-squared value
    """
    # Create a wrapped curve that handles multivariate inputs internally
    wrapped_curve = create_multivariate_wrapper(curve)
    
    # Calculate predictions
    predicted = wrapped_curve(x, *params)
    
    # Get dimension information for explanation function if needed
    if explain_if_inf:
        Ninputs = x.shape[1] if len(x.squeeze().shape) > 1 else 1
    
    # Use the from_predictions function to calculate n_chi_squared
    n_chi_squared = get_n_chi_squared_from_predictions(x, y, predicted)
    
    # If raw_n_chi_squared is inf, call explanation function
    if explain_if_inf and np.isinf(n_chi_squared):
        explain_inf_n_chi_squared(x, y, curve, Ninputs, string_for_explanation)
    return n_chi_squared

def explain_inf_n_chi_squared(x, y, curve, Ninputs, string_for_explanation=None):
    """
    Explains why n_chi_squared is infinite by finding problematic data points.
    
    Args:
        x: Input data points
        y: Target values
        curve: Function to evaluate
        Ninputs: Number of input dimensions
        string_for_explanation: Expression string for explanation (default: None)
    """
    logger.warning(f"n_chi_squared is inf. Expression: {string_for_explanation}")
    try:
        # Create a wrapped curve that handles multivariate inputs
        wrapped_curve = create_multivariate_wrapper(curve, Ninputs)
        
        # Evaluate the function on each data point to find where it fails
        problematic_points = []
        for i, xi in enumerate(x):
            try:
                # Just use the wrapper regardless of dimensionality
                result = wrapped_curve(xi[np.newaxis] if Ninputs > 1 else xi)
                if np.isinf(result) or np.isnan(result):
                    problematic_points.append((i, xi, result))
            except Exception as e:
                problematic_points.append((i, xi, str(e)))
        
        # Report the number of problematic points and a sample
        if problematic_points:
            total_count = len(problematic_points)
            sample_size = min(5, total_count)
            
            # Handle sampling safely to avoid array shape issues
            if total_count > sample_size:
                indices = np.random.choice(range(total_count), sample_size, replace=False)
                sample_points = [problematic_points[i] for i in indices]
            else:
                sample_points = problematic_points
            
            # Format the output safely
            sample_strings = []
            for i, xi, result in sample_points:
                if isinstance(result, (int, float)) and not np.isnan(result) and not np.isinf(result):
                    result_str = str(round(result))
                else:
                    result_str = str(result)
                sample_strings.append(f'idx {i}: {xi}, result: {result_str}')
            
            logger.warning(f"------Found {total_count} problematic points. Sample: {', '.join(sample_strings)}")
    except Exception as e:
        logger.warning(f"Error while debugging inf n_chi_squared: {e}")


#########################################
# INPUT PREPROCESSING
#########################################

def create_multivariate_wrapper(curve, n_inputs=None):
    """
    Creates a wrapper for a curve function that handles multivariate inputs elegantly.
    
    Args:
        curve: The original curve function taking separate x coordinates
        n_inputs: Optional number of input dimensions, auto-detected if None
        
    Returns:
        A wrapped function that takes a single x array and unpacks it internally
    """
    def wrapped_curve(x, *params):
        # Auto-detect if multivariate (has shape with > 1 dimension)
        x_squeezed = x.squeeze()
        is_multivariate = len(x_squeezed.shape) > 1
        dimension_of_x = n_inputs or (x.shape[1] if is_multivariate else 1)
        
        # Call the curve with unpacked coordinates if multivariate
        if is_multivariate:
            return curve(*[x[:, i] for i in range(dimension_of_x)], *params)
        else:
            return curve(x, *params)
            
    return wrapped_curve

def prepare_input_data(x, y):
    """
    Preprocess input data to ensure consistency for curve fitting.
    
    Args:
        x: Input data array
        y: Target data array
        
    Returns:
        Tuple of (x, y, is_multivariate, dimension_of_x)
    """
    # Ensure y is squeezed
    y = y.squeeze()
    
    # Check if this is multivariate data
    is_multivariate = len(x.squeeze().shape) > 1
    dimension_of_x = x.shape[1] if is_multivariate else 1
    
    return x.squeeze(), y.squeeze(), is_multivariate, dimension_of_x

def create_curve_fit_wrapper(curve, dimension_of_x):
    """
    Create a wrapper for curve_fit that handles multivariate inputs correctly.
    
    Args:
        curve: The curve function to wrap
        dimension_of_x: Dimension of the input data
        
    Returns:
        Wrapped curve_fit function
    """
    if dimension_of_x > 1:
        def curve_fit_wrapper(curve, x, y, p0=None, **kwargs):
            xtranspose = np.transpose(x)  # needs to be shape(k,M)-shaped array for functions with k predictor, batch size M
            
            # Use the multivariate wrapper to handle coordinate unpacking
            multivariate_curve = create_multivariate_wrapper(curve, dimension_of_x)
            
            # For multivariate case where curve expects transposed inputs
            def wrapped_curve_for_scipy(X, *params):
                # curve_fit passes X as the entire dataset at once in a transposed format
                # we need to handle this special case
                if len(X.shape) == 2:  # Multiple input dimensions (k,M) format
                    # Transpose back to (M,k) format for our wrapper
                    X_correct = np.transpose(X)
                    return multivariate_curve(X_correct, *params)
                elif X.shape[0] == dimension_of_x:  # Single input point, but full dimension
                    # Single point, reshape to (1,k) format
                    X_reshaped = X.reshape(1, dimension_of_x)
                    return multivariate_curve(X_reshaped, *params)[0]  # Return scalar
                else:
                    raise ValueError(f"Invalid input shape: {X.shape}")

            return curve_fit(wrapped_curve_for_scipy, xtranspose, y, p0=p0, **kwargs)
        return curve_fit_wrapper
    else:
        return curve_fit


#########################################
# CURVE FITTING FUNCTIONS
#########################################

def fit_curve_with_guess(x, y, curve, params_initial, try_all_methods=False, log_everything=False, 
                         stats=None, curve_str=None, timeout_curve_fit=30):
    """
    Fit a curve to data with specified initial parameters.
    
    Args:
        x: Input data array
        y: Target data array
        curve: Function to fit
        params_initial: Initial parameters
        try_all_methods: Whether to try multiple optimisation methods
        log_everything: Whether to log details of the fitting process
        stats: Statistics object for tracking warnings and errors
        curve_str: String representation of the curve function
        timeout_curve_fit: Maximum time in seconds for curve fitting
        
    Returns:
        Tuple of (best_params, best_n_chi_squared)
    """
    result = FitResult(
        params=np.array(params_initial),
        n_chi_squared=np.inf,
        success=False,
    )
    
    try:
        # Preprocess input data
        x, y, is_multivariate, dimension_of_x = prepare_input_data(x, y)
        
        # Create appropriate wrapper for curve_fit
        curve_fit_here = create_curve_fit_wrapper(curve, dimension_of_x)
        
        # Use warning manager to handle warnings during fitting
        with WarningManager(stats, "regular curve fitting", print_warnings=log_everything) as warning_mgr:
            # Attempt to fit the curve
            if try_all_methods:
                result = _fit_with_all_methods(
                    x, y, curve, params_initial, curve_fit_here,
                    log_everything, stats, timeout_curve_fit, curve_str
                )
            else:
                result = _fit_with_default_method(
                    x, y, curve, params_initial, curve_fit_here, 
                    log_everything, stats
                )
            
            # Store warnings
            result.warnings.extend([str(w) for w in warning_mgr.warnings_list])
            
    except Exception as e:
        logger.error(f"Exception in fit_curve_with_guess: {str(e)}, all params passed: "
                   f"{x.shape}, {y.shape}, {curve}, {params_initial},"
                   f"{curve_str if curve_str is not None else curve}, "
                   f"{try_all_methods}, {log_everything}, {stats}")
        result.error_msg = str(e)
        result.success = False
    
    # Return params and n_chi_squared for backward compatibility
    return result.params, result.n_chi_squared

def _fit_with_default_method(x, y, curve, params_initial, curve_fit_here, log_everything, stats):
    """
    Helper function to fit using only the default method.
    
    Returns:
        FitResult object
    """
    result = FitResult(
        params=np.array(params_initial),
        n_chi_squared=np.inf,
        success=False,
        methods_used=[]
    )
    
    try:
        if log_everything:
            logger.info(f"Fitting curve with initial parameters {str(params_initial[0:3])[:-1]}...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            params_opt, _ = curve_fit_here(curve, x, y, p0=params_initial, maxfev=1000*len(params_initial))
        
        n_chi_squared = get_n_chi_squared(x, y, curve, params_opt)
        result.methods_used.append("default")
        result.params = params_opt
        result.n_chi_squared = n_chi_squared
        result.success = True
        
        if log_everything:
            logger.info(f"Fit complete: n_chi-squared={n_chi_squared}")
            logger.info(f"Methods used: {', '.join(result.methods_used)}")
        
    except RuntimeError as e:
        result.methods_used.append("failed")
        result.error_msg = str(e)
        
        if log_everything:
            logger.info(f"Curve fitting failed: {str(e)[:100]}, methods tried: {', '.join(result.methods_used)}")
            logger.info(f"Methods used: {', '.join(result.methods_used)}")
    
    return result

def _fit_with_all_methods(x, y, curve, params_initial, curve_fit_here, log_everything, 
                         stats, timeout_curve_fit, curve_str):
    """
    Helper function to fit using all available methods.
    
    Returns:
        FitResult object
    """
    result = FitResult(
        params=np.array(params_initial),
        n_chi_squared=np.inf,
        success=False,
        methods_used=[]
    )
    
    # First try default method
    try:
        if log_everything:
            logger.info(f"Fitting curve with initial parameters {str(params_initial[0:3])[:-1]}...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            params_opt, _ = curve_fit_here(curve, x, y, p0=params_initial, maxfev=1000*len(params_initial))
        
        n_chi_squared = get_n_chi_squared(x, y, curve, params_opt)
        result.methods_used.append("default")
        result.params = params_opt
        result.n_chi_squared = n_chi_squared
        result.success = True
        
        if log_everything:
            logger.info(f"Fit complete: n_chi-squared={n_chi_squared}")
            logger.info(f"Methods used: {', '.join(result.methods_used)}")
        
        return result
    
    except RuntimeError as e:
        initial_error = str(e)
        
        # If default method fails, try alternative methods
        methods = ['lm', 'trf', 'dogbox']
        for method in methods:
            try:
                if log_everything:
                    logger.info(f"Fitting curve with method {method}, "
                               f"{'trf may take some time' if method == 'trf' else ''} "
                               f"with timeout: {timeout_curve_fit} seconds")
                
                # Use stopit for timeout functionality
                import stopit
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    with stopit.ThreadingTimeout(timeout_curve_fit) as timeout_ctx:
                        params_opt, _ = curve_fit_here(curve, x, y, p0=params_initial, 
                                                     method=method, maxfev=1000*len(params_initial))
                    
                    if timeout_ctx.state == timeout_ctx.TIMED_OUT:
                        if log_everything:
                            logger.info(f"Curve fitting {method} timed out after {timeout_curve_fit} seconds")
                        raise RuntimeError(f"Curve fitting timed out after {timeout_curve_fit} seconds")
                
                n_chi_squared = get_n_chi_squared(x, y, curve, params_opt)
                result.methods_used.append(method)
                result.params = params_opt
                result.n_chi_squared = n_chi_squared
                result.success = True
                
                if log_everything:
                    logger.info(f"Fit complete {method}: n_chi-squared={n_chi_squared}")
                    logger.info(f"Methods used: {', '.join(result.methods_used)}")
                
                return result
            
            except RuntimeError:
                continue
        
        # If all methods fail
        result.methods_used.append("failed_all")
        result.error_msg = initial_error
        logger.info(f"All methods failed for this fit {curve_str if curve_str is not None else curve} "
                   f"{initial_error[:100]}, methods tried: {', '.join(result.methods_used)}")
        
        if log_everything:
            logger.info(f"Methods used: {', '.join(result.methods_used)}")
        
        return result

def fit_curve(x, y, curve, largest_entry, curve_str=None, allow_using_jax=True, 
              force_using_jax=False, stats=None, log_methods=False):
    """
    Fits a given curve to the provided data points (x, y) and calculates the n_chi-squared value.
    
    Args:
        x (array-like): The independent variable data points.
        y (array-like): The dependent variable data points.
        curve (callable): The curve function to fit, which should take x and parameters as inputs.
        largest_entry (int): The number of parameters for the curve function.
        curve_str (str, optional): The string representation of the curve function.
        allow_using_jax (bool, optional): Whether to allow using JAX for fitting.
        force_using_jax (bool, optional): Whether to force using JAX for fitting.
        stats (APICallStats, optional): Statistics object for tracking warnings and errors.
        log_methods (bool, optional): Whether to log the methods used in fitting.
        
    Returns:
        tuple: A tuple containing:
            - params_opt (array-like): The optimised parameters for the curve.
            - n_chi_squared (float): The n_chi-squared value indicating the goodness of fit.
    """
    result = FitResult(
        params=np.ones(largest_entry),
        n_chi_squared=np.inf,
        success=False,
        methods_used=[]
    )
    
    logger.debug(f"Fitting curve with {largest_entry} parameters")
    logger.debug(f"Data shape: x={len(x)}, y={len(y)}")
    
    # Validate JAX usage
    if (allow_using_jax or force_using_jax) and curve_str is None:
        logging.error(f"Curve string is None, but allow_using_jax is True. This is not allowed.")
        logging.error(f"proceeding, setting allow_using_jax to False")
        allow_using_jax = False
        force_using_jax = False
    
    # Use warning manager to handle warnings during fitting
    with WarningManager(stats, "curve fitting", print_warnings=False) as warning_mgr:
        try:
            # Try regular scipy curve fitting first
            scipy_result = _try_scipy_curve_fit_comprehensive(
                x, y, curve, result.params, largest_entry)
            
            result.params = scipy_result.params
            result.n_chi_squared = scipy_result.n_chi_squared
            result.success = scipy_result.success
            result.methods_used.extend(scipy_result.methods_used)
            result.error_msg = scipy_result.error_msg
            
            # If JAX is forced, try JAX optimisation
            if force_using_jax and result.success:
                jax_result = fit_curve_with_guess_jax(
                    x, y, curve, result.params, 
                    log_everything=True, 
                    log_methods=log_methods,
                    then_do_reg_fitting=True, 
                    numpy_curve_str=curve_str, 
                    stats=stats
                )
                
                # Use JAX results if they're better
                if jax_result.n_chi_squared < result.n_chi_squared:
                    result.params = jax_result.params
                    result.n_chi_squared = jax_result.n_chi_squared
                    result.methods_used.extend(jax_result.methods_used)
                    result.methods_used.append("jax_forced")
            
        except Exception as e:
            # Try JAX as fallback if allowed
            if allow_using_jax:
                try:
                    jax_result = fit_curve_with_guess_jax(
                        x, y, curve, result.params, 
                        log_everything=True, 
                        log_methods=log_methods,
                        then_do_reg_fitting=True, 
                        numpy_curve_str=curve_str, 
                        stats=stats
                    )
                    
                    if jax_result.n_chi_squared < np.inf:
                        result.params = jax_result.params
                        result.n_chi_squared = jax_result.n_chi_squared
                        result.success = True
                        result.methods_used.extend(jax_result.methods_used)
                        result.methods_used.append("jax_fallback")
                    else:
                        result.methods_used.append("jax_fallback_failed")
                        
                except Exception as e2:
                    logger.info(f"JAX method fallback in fit_curve failed, numpy curve str: {curve_str}, {e2}")
                    result.methods_used.append("jax_fallback_failed")
            
            # Log the original error if we didn't recover
            if not result.success:
                result.error_msg = str(e)
                
                if log_methods:
                    methods_str = ", ".join(result.methods_used)
                    logger.info(f"All methods failed for this fit {curve_str if curve_str is not None else curve} "
                               f"{str(e)[:100]}, methods tried: {methods_str}")
                else:
                    # Keep exactly the same message format as the original for test compatibility
                    logger.info(f"All methods failed for this fit {curve_str if curve_str is not None else curve} {str(e)[:100]}")
        
        # Report methods used if requested
        if log_methods and result.success:
            methods_str = ", ".join(result.methods_used)
            logger.info(f"Curve fitting completed using methods: {methods_str}")
        
        # Store warnings
        result.warnings.extend([str(w) for w in warning_mgr.warnings_list])
    
    # Return params and n_chi_squared for backward compatibility
    return result.params, result.n_chi_squared

def _try_scipy_curve_fit_comprehensive(x, y, curve, params_initial, largest_entry):
    """
    Attempt scipy curve fitting with multiple strategies.
    
    Returns:
        FitResult object
    """
    result = FitResult(
        params=np.array(params_initial),
        n_chi_squared=np.inf,
        success=False,
        methods_used=[]
    )
    
    try:
        # Standard curve fitting with initial params
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            params_opt, covariance = curve_fit(curve, x, y, p0=params_initial, 
                                             maxfev=1000*largest_entry)
        
        result.methods_used.append("default")
        result.params = params_opt
        logger.debug(f"Optimised parameters: {params_opt}")
        
    except RuntimeError as e:
        initial_error = str(e)
        result.error_msg = initial_error
        
        # Try different methods if standard fitting fails
        logger.debug(f"Initial fit failed: {e}. Trying with different methods")
        
        success = False
        for method in ['lm', 'trf', 'dogbox']:
            try:
                logger.debug(f"Trying method: {method}")
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    params_opt, covariance = curve_fit(curve, x, y, p0=params_initial, 
                                                     method=method, maxfev=1000*largest_entry)
                
                result.methods_used.append(method)
                result.params = params_opt
                logger.debug(f"Optimised parameters with method {method}: {params_opt}")
                success = True
                break
                
            except RuntimeError as e2:
                logger.debug(f"Method {method} failed: {e2} whilst handling {e}")
        
        # Try with random initialization as last resort
        if not success:
            try:
                random_params = np.random.uniform(-1.0, 1.0, largest_entry)
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    params_opt, covariance = curve_fit(curve, x, y, p0=random_params, 
                                                     maxfev=1000*largest_entry)
                
                result.methods_used.append("random_initialization")
                result.params = params_opt
                logger.debug(f"Optimised parameters with random init: {params_opt}")
                success = True
                
            except RuntimeError as e3:
                result.error_msg = str(e3)
                # Log the error in a format compatible with tests
                logger.info(f"All methods failed for this fit {curve} {str(e3)[:100]}")
                return result
    
    # Calculate chi-squared for successful fit
    logger.debug("Calculating fit quality metrics")
    result.n_chi_squared = get_n_chi_squared(x, y, curve, result.params)
    result.success = True
    logger.debug(f"Fit complete: n_chi-squared={result.n_chi_squared}")
    
    return result

def fit_curve_with_guess_jax(x, y, curve, params_initial, log_everything=False, 
                            log_methods=False, then_do_reg_fitting=False, 
                            numpy_curve_str=None, stats=None, timeout_regular=30):
    """
    JAX implementation of curve fitting that mimics fit_curve_with_guess.
    
    Args:
        x: Input data array
        y: Target data array
        curve: Function to fit
        params_initial: Initial parameters
        log_everything: Whether to log details of the fitting process
        log_methods: Whether to only log the methods used (summary at the end)
        then_do_reg_fitting: Whether to try regular (scipy) fitting after JAX
        numpy_curve_str: String representation of the curve function for NumPy
        stats: Statistics object for tracking warnings and errors
        timeout_regular: Timeout for regular fitting in seconds
    
    Returns:
        FitResult object with (best_params, best_n_chi_squared)
    """
    y = y.squeeze()
    result = FitResult(
        params=np.array(params_initial),
        n_chi_squared=np.inf,
        success=False,
        methods_used=["jax"]
    )
    
    # Use warning manager to handle warnings during fitting
    with WarningManager(stats, "JAX fitting", print_warnings=log_everything) as warning_mgr:
        try:
            # Convert inputs to JAX arrays
            x_jax = jnp.array(x)
            y_jax = jnp.array(y)
            params_initial_jax = jnp.array(params_initial)
            multivariateQ = len(x_jax.shape) > 1
            
            # Process curve functions
            jax_curve, np_curve, jax_curve_str = _prepare_jax_curve(
                curve, numpy_curve_str, log_everything, result.methods_used)
            
            x_dim = x_jax.shape[1] if len(x_jax.shape) == 2 else 1
            
            # Handle constant functions with empty parameter list
            if len(params_initial) == 0:
                result.params = np.array([])
                result.n_chi_squared = _calculate_constant_chi_squared(
                    x_jax, y_jax, jax_curve, multivariateQ, x_dim)
                result.success = True
                return result.params, result.n_chi_squared
            
            # Perform JAX optimisation
            jax_optimised = _run_jax_optimisation(
                x_jax, y_jax, jax_curve, params_initial_jax, 
                multivariateQ, x_dim, log_everything, result.methods_used)
            
            result.params = jax_optimised.params
            result.n_chi_squared = jax_optimised.n_chi_squared
            result.success = jax_optimised.success
            
            # Try additional optimisation methods if requested
            if then_do_reg_fitting:
                post_optimised = _run_post_jax_optimisations(
                    x, y, jax_curve, np_curve, result.params, result.n_chi_squared, 
                    log_everything, stats, timeout_regular, result.methods_used)
                
                # Update result if post-optimisation improved the fit
                if post_optimised.success and post_optimised.n_chi_squared < result.n_chi_squared:
                    result.params = post_optimised.params
                    result.n_chi_squared = post_optimised.n_chi_squared
            
            # Report methods used
            if log_methods or log_everything:
                _report_optimisation_methods(result.methods_used, result.success)
            
            # Store warnings
            result.warnings.extend([str(w) for w in warning_mgr.warnings_list])
            return result.params, result.n_chi_squared
        
        except Exception as e:
            # Handle exceptions properly
            logger.error(f"JAX fitting failed with error: {str(e)}, on "
                       f"{jax_curve_str if 'jax_curve_str' in locals() else None}, "
                       f"{numpy_curve_str}. Methods tried: {', '.join(result.methods_used)}")
            
            result.error_msg = str(e)
            result.success = False
            result.warnings.extend([str(w) for w in warning_mgr.warnings_list])
            raise

def _prepare_jax_curve(curve, numpy_curve_str, log_everything, methods_used):
    """
    Prepare JAX curve function from original curve or string representation.
    
    Returns:
        Tuple of (jax_curve, np_curve, jax_curve_str)
    """
    jax_curve = curve
    np_curve = None
    jax_curve_str = None
    
    if numpy_curve_str is not None:
        jax_curve_str = numpy_curve_str.replace('np.', 'jnp.')
        try:
            jax_curve = jax.jit(eval(jax_curve_str))  # Try to JIT compile the lambda function
        except Exception as e:
            logger.debug(f"Failed to JIT compile curve: {e}, using unjitted version")
            jax_curve = eval(jax_curve_str)  # Fall back to unjitted lambda function
        np_curve = eval(numpy_curve_str)
        methods_used.append("jax_from_string")
        
        if log_everything:
            logger.debug(f"Converting numpy curve string to JAX curve string: "
                       f"{numpy_curve_str}, jax_curve_str: {jax_curve_str}")
    
    return jax_curve, np_curve, jax_curve_str

def _calculate_constant_chi_squared(x_jax, y_jax, jax_curve, multivariateQ, x_dim):
    """
    Calculate chi-squared for constant functions with no parameters.
    
    Returns:
        float: chi-squared value
    """
    # Create a wrapped curve that handles multivariate inputs internally
    wrapped_curve = create_multivariate_wrapper(jax_curve, x_dim)
    pred = wrapped_curve(x_jax)
    
    return float(_get_n_chi_squared_from_predictions_jax(x_jax, y_jax, pred))

def _run_jax_optimisation(x_jax, y_jax, jax_curve, params_initial_jax, 
                         multivariateQ, x_dim, log_everything, methods_used):
    """
    Run JAX-based optimisation.
    
    Returns:
        FitResult object
    """
    result = FitResult(
        params=np.array(params_initial_jax),
        n_chi_squared=np.inf,
        success=False
    )
    
    # Create a wrapped curve that handles multivariate inputs internally
    wrapped_curve = create_multivariate_wrapper(jax_curve, x_dim)
    
    # Define the loss function (sum of squared residuals)
    def loss_fn(params):
        pred = wrapped_curve(x_jax, *params)
        residuals = y_jax - pred
        return jnp.mean(jnp.square(residuals))
    
    def get_n_chi_squared_jax(params):
        pred = wrapped_curve(x_jax, *params)
        return _get_n_chi_squared_from_predictions_jax(x_jax, y_jax, pred)

    # Define optimisation methods to try
    methods = ['BFGS']  # only BFGS for now
    
    # Catch warnings during the optimisation process
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        
        for method in methods:
            try:
                if log_everything:
                    logger.info(f"Fitting curve with JAX method {method}, initial parameters {str(params_initial_jax[:3])[:-1]}...")
                
                jax_result = jax_minimize(loss_fn, params_initial_jax, method=method, 
                                         options={'maxiter': 10000*len(params_initial_jax)})
                methods_used.append(f"jax_{method}")
                
                if jax_result.success:
                    params_opt = jax_result.x
                    n_chi_squared = get_n_chi_squared_jax(params_opt)
                    
                    if log_everything:
                        logger.debug(f"JAX fit complete: n_chi-squared={n_chi_squared}, params={params_opt[0:3]}...")
                    
                    result.params = np.array(params_opt)
                    result.n_chi_squared = float(n_chi_squared)
                    result.success = True
                    break
                else:
                    if log_everything:
                        logger.debug(f"JAX optimisation failed: {jax_result.success}, {jax_result.x[0:3]}...")
                    continue
            except Exception as e:
                if log_everything:
                    logger.info(f"JAX method in _run_jax_optimisation failed: {str(e)[:100]}")
                continue
    
    return result

def _run_post_jax_optimisations(x, y, jax_curve, np_curve, best_params, best_n_chi_squared,
                               log_everything, stats, timeout_regular, methods_used):
    """
    Run post-JAX optimisations using various methods to improve results.
    
    Returns:
        FitResult object
    """
    result = FitResult(
        params=np.array(best_params),
        n_chi_squared=best_n_chi_squared,
        success=True
    )
    
    # Strategy 1: Try regular curve_fit with JAX results as initial guess
    try:
        reg_curve = np_curve if np_curve is not None else jax_curve
        reg_params, reg_n_chi_squared = fit_curve_with_guess(
            np.array(x), np.array(y), reg_curve, 
            np.array(best_params), try_all_methods=True, 
            log_everything=log_everything,
            stats=stats, timeout_curve_fit=timeout_regular
        )
        methods_used.append("regular_after_jax")
        
        # Compare results and take the better one
        if np.isfinite(reg_n_chi_squared) and reg_n_chi_squared < best_n_chi_squared:
            if log_everything:
                logger.info(f"Regular fitting improved results: n_chi-squared from {best_n_chi_squared} to {reg_n_chi_squared}")
            result.params = reg_params
            result.n_chi_squared = reg_n_chi_squared
        elif log_everything and np.isfinite(reg_n_chi_squared):
            logger.info(f"JAX fitting had better results: n_chi-squared {best_n_chi_squared} vs {reg_n_chi_squared}")
    except Exception as e:
        methods_used.append("regular_after_jax_failed")
        if log_everything:
            # Always log the full error message - this could be a bug that needs fixing
            error_type = type(e).__name__
            error_msg = str(e)
            logger.info(f"Regular fitting failed ({error_type}), keeping JAX results. Error: {error_msg}")
    
    # Strategy 2: Try scipy.optimize.minimize
    try:
        # Create a robust wrapper that handles multivariate inputs
        wrapped_curve = create_multivariate_wrapper(jax_curve)
        
        def objective(params):
            # We don't want exception handling here - we want to see all errors
            predictions = wrapped_curve(x, *params)
            return np.mean((predictions - y)**2)
    
        optimised_result = minimize(objective, best_params)
        methods_used.append("scipy_minimize_after_jax")
        
        if optimised_result.success:
            opt_params = optimised_result.x
            opt_n_chi_squared = get_n_chi_squared(np.array(x), np.array(y), jax_curve, opt_params)
            
            if np.isfinite(opt_n_chi_squared) and opt_n_chi_squared < 0.999*result.n_chi_squared:
                result.params = opt_params
                result.n_chi_squared = opt_n_chi_squared
                if log_everything:
                    logger.info(f"scipy minimize improved results: n_chi-squared from {best_n_chi_squared} to {opt_n_chi_squared}")
            
            if log_everything:
                logger.info(f"scipy minimize final results: n_chi-squared {opt_n_chi_squared}, params {opt_params[0:3]}...")
        elif log_everything:
            logger.info(f"scipy minimize optimisation failed to converge: {optimised_result.message}")
    except Exception as e:
        methods_used.append("scipy_minimize_after_jax_failed")
        if log_everything:
            error_type = type(e).__name__
            error_msg = str(e)
            # Log the error in full for debugging
            logger.info(f"scipy.optimize.minimize failed ({error_type}): {error_msg}")
    
    # Strategy 3: Try scipy.optimize.minimize with bounds
    try:
        wrapped_curve = create_multivariate_wrapper(jax_curve)
        
        def objective(params):
            # Constrain parameters to -10 to 10 range
            bounded_params = np.clip(params, -10, 10)
            predictions = wrapped_curve(x, *bounded_params)
            return np.mean((predictions - y)**2)
        
        bounds = [(-10, 10) for _ in range(len(best_params))]
        # Use L-BFGS-B which works well with bounds
        optimised_result = minimize(objective, best_params, bounds=bounds, method='L-BFGS-B')
        methods_used.append("scipy_minimize_after_jax_with_bounds")
        
        if optimised_result.success:
            opt_params = np.clip(optimised_result.x, -10, 10)  # Ensure final params are in bounds
            opt_n_chi_squared = get_n_chi_squared(np.array(x), np.array(y), jax_curve, opt_params)
            
            if np.isfinite(opt_n_chi_squared) and opt_n_chi_squared < 0.999*result.n_chi_squared:
                result.params = opt_params
                result.n_chi_squared = opt_n_chi_squared
                if log_everything:
                    logger.info(f"scipy minimize with [-10,10] bounds improved results: "
                               f"n_chi-squared from {best_n_chi_squared} to {opt_n_chi_squared}")
            
            if log_everything:
                logger.info(f"scipy minimize with [-10,10] bounds final results: "
                           f"n_chi-squared {opt_n_chi_squared}, params {opt_params[0:3]}...")
        elif log_everything:
            logger.info(f"scipy minimize with bounds optimisation failed to converge: {optimised_result.message}")
    except Exception as e:
        methods_used.append("scipy_minimize_after_jax_with_bounds_failed")
        if log_everything:
            error_type = type(e).__name__
            error_msg = str(e)
            # Log full error information for debugging
            logger.info(f"scipy.optimize.minimize with bounds failed ({error_type}): {error_msg}")
    
    # Strategy 4: Try scipy.optimize.minimize with bounds based on current best params
    try:
        # Only attempt if we have valid parameters
        if result.params is None or len(result.params) == 0:
            if log_everything:
                logger.info("No valid parameters available for adaptive bounds")
            raise ValueError("No valid parameters to use for adaptive bounds")
        
        # Create bounds as 2*opt_params to -2*opt_params
        param_bounds = []
        for p in result.params:
            lower = -2 * abs(p) if p != 0 else -2
            upper = 2 * abs(p) if p != 0 else 2
            param_bounds.append((lower, upper))
        
        wrapped_curve = create_multivariate_wrapper(jax_curve)
        
        def objective(params):
            predictions = wrapped_curve(x, *params)
            return np.mean((predictions - y)**2)
        
        # Use SLSQP which can handle bounded problems well
        optimised_result = minimize(objective, result.params, bounds=param_bounds, method='SLSQP')
        methods_used.append("scipy_minimize_with_adaptive_bounds")
        
        if optimised_result.success:
            opt_params = optimised_result.x
            opt_n_chi_squared = get_n_chi_squared(np.array(x), np.array(y), jax_curve, opt_params)
            
            if np.isfinite(opt_n_chi_squared) and opt_n_chi_squared < 0.999*result.n_chi_squared:
                result.params = opt_params
                result.n_chi_squared = opt_n_chi_squared
                if log_everything:
                    logger.info(f"scipy minimize with adaptive bounds improved results: "
                               f"n_chi-squared from {best_n_chi_squared} to {opt_n_chi_squared}")
            
            if log_everything:
                logger.info(f"scipy minimize with adaptive bounds final results: "
                           f"n_chi-squared {opt_n_chi_squared}, params {opt_params[0:3]}...")
        elif log_everything:
            logger.info(f"scipy minimize with adaptive bounds failed to converge: {optimised_result.message}")
    except Exception as e:
        methods_used.append("scipy_minimize_with_adaptive_bounds_failed")
        if log_everything:
            error_type = type(e).__name__
            error_msg = str(e)
            # Log full error information including error type
            logger.info(f"scipy.optimize.minimize with adaptive bounds failed ({error_type}): {error_msg}")

    # Strategy 5: Try scipy.optimize.minimize with much tighter bounds
    try:
        # Only attempt if we have valid parameters
        if result.params is None or len(result.params) == 0:
            if log_everything:
                logger.info("No valid parameters available for tight bounds")
            raise ValueError("No valid parameters to use for tight bounds")
        
        # Create much tighter bounds: 1.2*opt_params to 0.8*opt_params
        param_bounds = []
        for p in result.params:
            if np.abs(p) > 1e-1:
                lower = 0.8 * p if p > 0 else 1.2 * p
                upper = 1.2 * p if p > 0 else 0.8 * p
            else:
                # For zero parameters, use very small bounds
                lower, upper = -0.2, 0.2
            param_bounds.append((lower, upper))
        
        wrapped_curve = create_multivariate_wrapper(jax_curve)
        
        def objective(params):
            predictions = wrapped_curve(x, *params)
            return np.mean((predictions - y)**2)
        
        # Use SLSQP which can handle bounded problems well
        optimised_result = minimize(objective, result.params, bounds=param_bounds, method='SLSQP')
        methods_used.append("scipy_minimize_with_tight_bounds")
        
        if optimised_result.success:
            opt_params = optimised_result.x
            opt_n_chi_squared = get_n_chi_squared(np.array(x), np.array(y), jax_curve, opt_params)
            
            if np.isfinite(opt_n_chi_squared) and opt_n_chi_squared < 0.999*result.n_chi_squared:
                result.params = opt_params
                result.n_chi_squared = opt_n_chi_squared
                if log_everything:
                    logger.info(f"scipy minimize with tight bounds improved results: "
                               f"n_chi-squared from {best_n_chi_squared} to {opt_n_chi_squared}")
            
            if log_everything:
                logger.info(f"scipy minimize with tight bounds final results: "
                           f"n_chi-squared {opt_n_chi_squared}, params {opt_params[0:3]}...")
        elif log_everything:
            logger.info(f"scipy minimize with tight bounds failed to converge: {optimised_result.message}")
    except Exception as e:
        methods_used.append("scipy_minimize_with_tight_bounds_failed")
        if log_everything:
            error_type = type(e).__name__
            error_msg = str(e)
            # Log full error information including error type
            logger.info(f"scipy.optimize.minimize with tight bounds failed ({error_type}): {error_msg}")

    
    return result

def _report_optimisation_methods(methods_used, success):
    """
    Log methods used for optimisation.
    """
    methods_str = ", ".join(methods_used)
    
    if success:
        logger.info(f"Curve fitting completed using methods: {methods_str}")
    else:
        logger.info(f"Curve fitting failed using methods: {methods_str}")


#########################################
# EXPRESSION EVALUATION AND TESTING
#########################################

def test_np_function_equivalence(f1, f2, test_xs, rtol=1e-3, almost_eveywhere_fraction=0.99):
    """
    Test if two expressions produce similar outputs for the same inputs.
    
    Args:
        f1: First function to compare
        f2: Second function to compare
        test_xs: Test input values
        rtol: Relative tolerance for comparison
        almost_eveywhere_fraction: Fraction of points required to be equal
        
    Returns:
        Tuple of (is_equivalent, message)
    """
    try:
        # Always use the provided test_xs
        test_inputs = test_xs
        
        # Handle both scalar and array inputs
        if not hasattr(test_inputs, '__iter__'):
            # For scalar inputs, wrap in a list with a tuple
            test_values = [(test_inputs,)]
        elif not hasattr(test_inputs[0], '__iter__'):
            # For 1D array inputs, wrap each value in a tuple
            test_values = [(x,) for x in test_inputs]
        else:
            # For multi-dimensional inputs, use as is
            test_values = test_inputs
        
        # Evaluate both expressions at test points
        results1 = np.array([f1(*x) for x in test_values])
        results2 = np.array([f2(*x) for x in test_values])
        
        return test_data_equivalence(results1, results2, rtol, almost_eveywhere_fraction)
    except Exception as e:
        return False, str(e)

def test_data_equivalence(y_1, y_2, rtol, almost_everywhere_fraction=0.99):
    """
    Test if two sets of data are functionally equivalent.
    
    Args:
        y_1: First set of data
        y_2: Second set of data
        rtol: Relative tolerance for comparison
        almost_everywhere_fraction: Fraction of points required to be equal
        
    Returns:
        Tuple of (is_equivalent, message)
    """
    # Check if results are close almost everywhere
    try:
        if np.allclose(y_1, y_2, rtol=rtol):
            return True, None

        # Calculate percentage of points that are close
        close_points = np.isclose(y_1, y_2, rtol=rtol)
        percent_close = np.mean(close_points) * 100
    
        # If most points are close, return True without warning if all points are close
        if percent_close > almost_everywhere_fraction:
            if percent_close == 100:
                return True, None

            # Calculate average relative difference for warning
            rel_diff = np.abs((y_1 - y_2) / np.maximum(np.abs(y_1), 1e-10))
            avg_rel_diff = np.mean(rel_diff)
            return True, f"Warning: Expressions differ at {100-percent_close:.1f}% of points (avg diff: {avg_rel_diff:.4e})"
        else:
            # Calculate average relative difference
            rel_diff = np.abs((y_1 - y_2) / np.maximum(np.abs(y_1), 1e-10))
            avg_rel_diff = np.mean(rel_diff)
            return False, avg_rel_diff
    except Exception as e:
        return False, str(e)


#########################################
# TEST CODE
#########################################

if __name__ == "__main__":
    # Set up logging for test
    logging.basicConfig(level=logging.INFO)
    test_func_str = "lambda x, a, b, c: a * np.exp(-b * x) + c"
    test_func = eval(test_func_str)
    test_func_jax_str = test_func_str.replace('np.', 'jnp.')
    test_func_jax = eval(test_func_jax_str)
    
    # Generate synthetic data
    true_params = [3.0, 0.5, 1.0]
    x_data = jnp.linspace(0, 10, 100)
    y_true = test_func(x_data, *true_params)
    
    # Add some noise
    key = jax.random.PRNGKey(42)
    y_noisy = y_true + jax.random.normal(key, shape=y_true.shape) * 0.2
    
    # Test with NumPy version
    initial_guess = [1.0, 1.0, 1.0]
    
    # Time NumPy version
    np_start_time = time.time()
    np_params, np_chi2 = fit_curve_with_guess(
        np.array(x_data), np.array(y_noisy), 
        test_func, 
        initial_guess, 
        log_everything=True
    )
    np_elapsed = time.time() - np_start_time
    
    print('\nTesting JAX version with minimal logging (default)')
    # Time JAX version with minimal logging
    jax_start_time = time.time()
    jax_params, jax_chi2 = fit_curve_with_guess_jax(  # Now unpacking the tuple
        x_data, y_noisy, 
        test_func_jax, 
        initial_guess,
    )
    jax_elapsed = time.time() - jax_start_time
    
    print('\nTesting JAX version with method logging')
    # Time JAX version with method logging
    jax_methods_start_time = time.time()
    jax_methods_params, jax_methods_chi2 = fit_curve_with_guess_jax(  # Now unpacking the tuple
        x_data, y_noisy, 
        test_func_jax, 
        initial_guess,
        log_methods=True
    )
    jax_methods_elapsed = time.time() - jax_methods_start_time
    
    print('\nTesting JAX version with full logging')
    # Time JAX version with full logging
    jax_full_start_time = time.time()
    jax_full_params, jax_full_chi2 = fit_curve_with_guess_jax(  # Now unpacking the tuple
        x_data, y_noisy, 
        test_func_jax, 
        initial_guess,
        log_everything=True
    )
    jax_full_elapsed = time.time() - jax_full_start_time
    
    print("\nResults comparison:")
    print(f"True parameters: {true_params}")
    print(f"NumPy fit parameters: {np_params}")
    print(f"JAX fit parameters: {jax_params}")
    print(f"JAX with method logging parameters: {jax_methods_params}")
    print(f"JAX with full logging parameters: {jax_full_params}")
    print(f"NumPy n_chi-squared: {np_chi2}")
    print(f"JAX n_chi-squared: {jax_chi2}")
    print(f"JAX with method logging n_chi-squared: {jax_methods_chi2}")
    print(f"JAX with full logging n_chi-squared: {jax_full_chi2}")
    print("\nTiming comparison:")
    print(f"NumPy version: {np_elapsed:.4f} seconds")
    print(f"JAX version (minimal logging): {jax_elapsed:.4f} seconds")
    print(f"JAX version (method logging): {jax_methods_elapsed:.4f} seconds")
    print(f"JAX version (full logging): {jax_full_elapsed:.4f} seconds")
    print(f"Speedup (JAX vs NumPy): {np_elapsed/jax_elapsed:.2f}x")
    
    # Test new string-based function with fallback
    print("\nTesting string-based function with secondary regular optimisation:")
    secondary_start_time = time.time()
    secondary_params, secondary_chi2 = fit_curve_with_guess_jax(  # Now unpacking the tuple
        x_data, y_noisy,
        None,  # We're using the string version instead
        initial_guess,
        log_everything=False,
        log_methods=True,
        then_do_reg_fitting=True,
        numpy_curve_str=test_func_str
    )
    secondary_elapsed = time.time() - secondary_start_time
    
    print("\nSecondary regular optimisation test results:")
    print(f"True parameters: {true_params}")
    print(f"Secondary regular optimisation fit parameters: {secondary_params}")
    print(f"Secondary regular optimisation n_chi-squared: {secondary_chi2}")
    print(f"Secondary regular optimisation execution time: {secondary_elapsed:.4f} seconds")
    
    # Compare with previous results
    print("\nParameter differences:")
    print(f"NumPy vs Secondary regular optimisation: {np.abs(np_params - secondary_params).sum():.6f}")
    print(f"JAX vs Secondary regular optimisation: {np.abs(jax_params - secondary_params).sum():.6f}")
    print(f"All params: {np_params}, {jax_params}, {secondary_params}")