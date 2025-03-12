from scipy.optimize import curve_fit
import numpy as np
from scipy import special
import logging
import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Explicitly tell JAX to only use CPU
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from functools import partial
import time
import warnings
from scipy.optimize import OptimizeWarning

# Get module logger
logger = logging.getLogger("LLMSR.fit")

# Disable 64-bit precision - for some reason 64-bit doesn't work! That's why we iterate and use the regular fit after JAX.
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_platform_name", "cpu")  # Disable JAX on GPU

# Unified warning handler for all fitting functions
def warning_handler(message, category, filename, lineno, file=None, line=None, stats=None, context="curve fitting"):
    warning_msg = str(message).lower()
    if stats is not None and hasattr(stats, 'add_fitting_warning'):
        if (category == RuntimeWarning and "invalid value" in warning_msg and "sqrt" in warning_msg) or (
            "nan" in warning_msg or "inf" in warning_msg):
            stats.add_fitting_warning('invalid_sqrt')
            logger.warning(f"{context}: invalid value encountered in sqrt - {message}")
        elif category == OptimizeWarning and "covariance" in warning_msg:
            stats.add_fitting_warning('covariance_estimation')
            logger.warning(f"{context}: Covariance of the parameters could not be estimated")
        elif "convergence" in warning_msg:
            stats.add_fitting_warning('convergence_error')
            logger.warning(f"{context}: convergence warning - {message}")
        elif category == RuntimeWarning and "invalid value" in warning_msg and "log" in warning_msg:
            stats.add_fitting_warning('invalid_log')
            logger.warning(f"{context}: invalid value encountered in log")
        elif category == RuntimeWarning and "invalid value" in warning_msg and "power" in warning_msg:
            stats.add_fitting_warning('invalid_power')
            logger.warning(f"{context}: invalid value encountered in power")
        else:
            stats.add_fitting_warning('other_warnings')
            logger.warning(f"Warning during {context}: {category.__name__}: {message}")
    return

def get_n_chi_squared(x, y, curve, params, explain_if_inf=False, string_for_explanation=None):
    # Calculate predictions based on input dimensions
    if len(x.squeeze().shape) > 1:
        Ninputs = x.shape[1]        
        predicted = curve(*[x[:, i] for i in range(x.shape[1])], *params)
    else:   
        Ninputs = 1
        predicted = curve(x, *params)
    
    # Use the from_predictions function to calculate n_chi_squared
    n_chi_squared = get_n_chi_squared_from_predictions(x, y, predicted)
    # If raw_n_chi_squared is inf, call explanation function
    if explain_if_inf and np.isinf(n_chi_squared):
        explain_inf_n_chi_squared(x, y, curve, Ninputs, string_for_explanation)
    return n_chi_squared

def explain_inf_n_chi_squared(x, y, curve, Ninputs, string_for_explanation=None):
    """Explains why n_chi_squared is infinite by finding problematic data points"""
    logger.warning(f"n_chi_squared is inf. Expression: {string_for_explanation}")
    try:
        # Evaluate the function on each data point to find where it fails
        problematic_points = []
        for i, xi in enumerate(x):
            try:
                if Ninputs == 1:
                    result = curve(xi)
                else:
                    result = curve(*xi)
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

def get_n_chi_squared_from_predictions(x, y, predictions):
    # Calculate residuals between actual and predicted values
    residuals = y - predictions
    
    # n_chi-squared calculation with robust normalization using standard deviation
    # This provides better scaling for constant fits than using variance
    data_std = jnp.std(y) + 1e-6
    n_chi_squared = jnp.mean((residuals ** 2) / data_std)
    return n_chi_squared
def fit_curve_with_guess(x, y, curve, params_initial, try_all_methods=False, log_everything=False, stats=None):
    """
    Fit a curve to data with specified initial parameters.
    
    Args:
        x: Input data array
        y: Target data array
        curve: Function to fit
        params_initial: Initial parameters
        try_all_methods: Whether to try multiple optimization methods
        log_everything: Whether to log details of the fitting process
        stats: Statistics object for tracking warnings and errors
        
    Returns:
        Tuple of (best_params, best_n_chi_squared)
    """
    # Setup warning handling
    original_showwarning = warnings.showwarning
    if stats is not None:
        warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: warning_handler(
            message, category, filename, lineno, file, line, stats, "regular curve fitting"
        )
    
    # Track methods used for reporting
    methods_used = []
    
    try:
        if len(x.shape) > 1:
            y = y.squeeze()
            dimension_of_x = x.shape[1]
            # Create a wrapper function to handle multivariate inputs
            def curve_fit_wrapper(curve, x, y, p0=None, **kwargs):
                xtranspose = np.transpose(x)# needs to be shape(k,M)-shaped array for functions with k predictor, batch size M
                # For multivariate case where curve expects individual coordinates
                def wrapped_curve(X, *params):
                    # curve_fit passes X as the entire dataset at once
                    # but our curve function expects individual coordinates
                    if len(X.shape) == 2:  # Multiple input dimensions
                        return curve(*[X[i] for i in range(dimension_of_x)], *params)
                    elif X.shape[0] == dimension_of_x:  # Single input dimension
                        return curve(*X, *params)
                    else:
                        raise ValueError(f"Invalid input shape: {X.shape}")

                return curve_fit(wrapped_curve, xtranspose, y, p0=p0, **kwargs)
            curve_fit_here = curve_fit_wrapper
        elif len(x.shape) == 1:
            curve_fit_here = curve_fit
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")
            
        if try_all_methods:
            try:
                if log_everything:
                    logger.info(f"Fitting curve with initial parameters {params_initial[0:3]}...")
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    params_opt, _ = curve_fit_here(curve, x, y, p0=params_initial, maxfev=1000*len(params_initial))
                n_chi_squared = get_n_chi_squared(x, y, curve, params_opt)
                methods_used.append("default")
                if log_everything:
                    logger.info(f"Fit complete: n_chi-squared={n_chi_squared}")
                
                # Restore warning handler
                warnings.showwarning = original_showwarning
                
                if log_everything:
                    logger.info(f"Methods used: {', '.join(methods_used)}")
                return params_opt, n_chi_squared 
            except RuntimeError as e:
                methods = ['lm', 'trf', 'dogbox']
                for method in methods:
                    try:
                        if log_everything:
                            logger.info(f"Fitting curve with method {method}, " + "trf may take some time" if method == 'trf' else "")
                        with warnings.catch_warnings():
                            warnings.simplefilter("always")
                            params_opt, _ = curve_fit_here(curve, x, y, p0=params_initial, method=method, maxfev=1000*len(params_initial))
                        n_chi_squared = get_n_chi_squared(x, y, curve, params_opt)
                        methods_used.append(method)
                        if log_everything:
                            logger.info(f"Fit complete: n_chi-squared={n_chi_squared}")
                        
                        # Restore warning handler
                        warnings.showwarning = original_showwarning
                        
                        if log_everything:
                            logger.info(f"Methods used: {', '.join(methods_used)}")
                        return params_opt, n_chi_squared
                    except RuntimeError:
                        continue
                methods_used.append("failed_all")
                logger.info(f"All methods failed for this fit {curve} {str(e)[:100]}, methods tried: {', '.join(methods_used)}")
                
                # Restore warning handler
                warnings.showwarning = original_showwarning
                
                if log_everything:
                    logger.info(f"Methods used: {', '.join(methods_used)}")
                return np.array(params_initial), np.inf
        else:
            try:
                if log_everything:
                    logger.info(f"Fitting curve with initial parameters {params_initial[0:3]}...")
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    params_opt, _ = curve_fit_here(curve, x, y, p0=params_initial, maxfev=1000*len(params_initial))
                n_chi_squared = get_n_chi_squared(x, y, curve, params_opt)
                methods_used.append("default")
                if log_everything:
                    logger.info(f"Fit complete: n_chi-squared={n_chi_squared}")
                
                # Restore warning handler
                warnings.showwarning = original_showwarning
                
                if log_everything:
                    logger.info(f"Methods used: {', '.join(methods_used)}")
                return params_opt, n_chi_squared
            except RuntimeError as e:
                methods_used.append("failed")
                if log_everything:
                    logger.info(f"Curve fitting failed: {str(e)[:100]}, methods tried: {', '.join(methods_used)}")
                
                # Restore warning handler
                warnings.showwarning = original_showwarning
                
                if log_everything:
                    logger.info(f"Methods used: {', '.join(methods_used)}")
                return np.array(params_initial), np.inf
    
    except Exception as e:
        # Make sure to restore the warning handler in case of any exception
        warnings.showwarning = original_showwarning
        
        logger.error(f"Exception in fit_curve_with_guess: {str(e)}, all params passed: {x.shape}, {y.shape}, {curve}, {params_initial},{ try_all_methods}, {log_everything}, {stats}")
        raise

def fit_curve(x, y, curve, largest_entry: int, curve_str: str = None, allow_using_jax: bool = True, force_using_jax: bool = False, stats=None, log_methods=False):
    """
    Fits a given curve to the provided data points (x, y) and calculates the n_chi-squared value.
    Parameters:
        x (array-like): The independent variable data points.
        y (array-like): The dependent variable data points.
        curve (callable): The curve function to fit, which should take x and parameters as inputs.
        largest_entry (int): The number of parameters for the curve function.
        curve_str (str, optional): The string representation of the curve function. Defaults to None.
        allow_using_jax (bool, optional): Whether to allow using JAX for fitting. Defaults to True.
        force_using_jax (bool, optional): Whether to force using JAX for fitting. Defaults to False.
        stats (APICallStats, optional): Statistics object for tracking warnings and errors. Defaults to None.
        log_methods (bool, optional): Whether to log the methods used in fitting. Defaults to True.
    Returns:
    tuple: A tuple containing:
        - params_opt (array-like): The optimized parameters for the curve.
        - n_chi_squared (float): The n_chi-squared value indicating the goodness of fit.
    """
    # Setup warning handling
    original_showwarning = warnings.showwarning
    if stats is not None:
        warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: warning_handler(
            message, category, filename, lineno, file, line, stats, "curve fitting"
        )
    
    # Track methods used for reporting
    methods_used = []
    
    logger.debug(f"Fitting curve with {largest_entry} parameters")
    logger.debug(f"Data shape: x={len(x)}, y={len(y)}")
    params_initial = np.ones(largest_entry)
    n_chi_squared = np.inf
    if (allow_using_jax or force_using_jax) and curve_str is None:
        logging.error(f"Curve string is None, but allow_using_jax is True. This is not allowed.")
        logging.error(f"proceeding, setting allow_using_jax to False")
        allow_using_jax = False
    
    try:
        # Initialize parameter, perform curve fitting
        logger.debug(f"Running curve_fit optimization, initial parameters {params_initial}")
        try:
            # Enable all warnings during curve_fit
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                params_opt, covariance = curve_fit(curve, x, y, p0=params_initial, maxfev=1000*largest_entry)
            methods_used.append("default")
            logger.debug(f"Optimized parameters: {params_opt}")
        except RuntimeError as e:
            # If initial fit fails, try with different methods
            logger.debug(f"Initial fit failed: {e}. Trying with different methods")
            methods = ['lm', 'trf', 'dogbox']
            for method in methods:
                try:
                    logger.debug(f"Trying method: {method}")
                    with warnings.catch_warnings():
                        warnings.simplefilter("always")
                        params_opt, covariance = curve_fit(curve, x, y, p0=params_initial, method=method, maxfev=1000*largest_entry)
                    methods_used.append(method)
                    logger.debug(f"Optimized parameters with method {method}: {params_opt}")
                    break
                except RuntimeError as e2:
                    logger.debug(f"Method {method} failed: {e2} whilst handling {e}")
            else:
                # If all methods fail, try with random parameters
                random_params = np.random.uniform(-1.0, 1.0, largest_entry)
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    params_opt, covariance = curve_fit(curve, x, y, p0=random_params, maxfev=1000*largest_entry)
                methods_used.append("random_initialization")
                logger.debug(f"Optimized parameters with random init: {params_opt}")

        
        # Calculate residuals and n_chi-squared
        logger.debug("Calculating fit quality metrics")
        n_chi_squared = get_n_chi_squared(x, y, curve, params_opt)
        
        logger.debug(f"Fit complete: n_chi-squared={n_chi_squared}")
        jax_used = False
        if force_using_jax:
            params_opt_jax, n_chi_squared_jax = fit_curve_with_guess_jax(
                x, y, curve, params_opt, 
                log_everything=True, 
                log_methods=log_methods,
                then_do_reg_fitting=True, 
                numpy_curve_str=curve_str, 
                stats=stats
            )
            if n_chi_squared_jax < n_chi_squared:
                params_opt = params_opt_jax
                n_chi_squared = n_chi_squared_jax
                jax_used = True
                methods_used.append("jax_forced")
        
        # Report methods used before returning
        if log_methods:
            methods_str = ", ".join(methods_used)
            logger.info(f"Curve fitting completed using methods: {methods_str}" + (" with JAX improvement" if jax_used else ""))
        
        # Restore original warning handler before returning
        warnings.showwarning = original_showwarning
        return params_opt, n_chi_squared
        
    except Exception as e:
        params_opt = params_initial
        n_chi_squared = np.inf        
        if allow_using_jax:
            try:
                params_opt_jax, n_chi_squared_jax = fit_curve_with_guess_jax(
                    x, y, curve, params_initial, 
                    log_everything=True, 
                    log_methods=log_methods,
                    then_do_reg_fitting=True, 
                    numpy_curve_str=curve_str, 
                    stats=stats
                )
                if n_chi_squared_jax < n_chi_squared:
                    params_opt = params_opt_jax
                    n_chi_squared = n_chi_squared_jax
                    methods_used.append("jax_fallback")
            except Exception as e2:
                logger.info(f"JAX method fallback in fit_curve failed, numpy curve str: {curve_str}, {e} whilst handling {e2}")
                methods_used.append("jax_fallback_failed")
        # Log error and return initial parameters with infinite n_chi-squared
        if log_methods:
            methods_str = ", ".join(methods_used)
            logger.info(f"All methods failed for this fit {curve} {str(e)[:100]}, methods tried: {methods_str}")
        else:
            logger.info(f"All methods failed for this fit {curve} {str(e)[:100]}")
        
        # Restore original warning handler before returning
        warnings.showwarning = original_showwarning
        return np.array(params_initial), np.inf

def fit_curve_with_guess_jax(x, y, curve, params_initial, log_everything=False, log_methods=False, then_do_reg_fitting=False, numpy_curve_str=None, stats=None):
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
    
    Returns:
        Tuple of (best_params, best_n_chi_squared)
    """
    # Setup warning handling
    original_showwarning = warnings.showwarning
    if stats is not None:
        warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: warning_handler(
            message, category, filename, lineno, file, line, stats, "JAX fitting"
        )
    
    # Track methods used for reporting
    methods_used = ["jax"]
    
    try:
        # Convert inputs to JAX arrays
        x_jax = jnp.array(x)
        y_jax = jnp.array(y)
        params_initial_jax = jnp.array(params_initial)
        multivariateQ = len(x_jax.shape) > 1

        
        # If numpy_curve_str is provided, extract JAX version for the fitting
        jax_curve = curve
        np_curve = None
        if numpy_curve_str is not None:
            jax_curve_str = numpy_curve_str.replace('np.', 'jnp.')
            jax_curve = eval(jax_curve_str) #(should be a lambda function)
            np_curve = eval(numpy_curve_str)
            methods_used.append("jax_from_string")
            if log_everything:
                logger.debug(f"Converting numpy curve string to JAX curve string: {numpy_curve_str}, jax_curve_str: {jax_curve_str}")

        x_dim = x_jax.shape[1] if len(x_jax.shape) == 2 else 1
        
        # Define the loss function (sum of squared residuals)
        def loss_fn(params):
            if multivariateQ:
                # Handle multivariate case
                pred = jax_curve(*[x_jax[:, i] for i in range(x_dim)], *params)
            else:
                pred = jax_curve(x_jax, *params)
            
            residuals = y_jax - pred
            return jnp.mean(jnp.square(residuals))
        
        # Method to calculate n_chi-squared for JAX
        def get_n_chi_squared_jax(params):
            if multivariateQ:
                pred = jax_curve(*[x_jax[:, i] for i in range(x_dim)], *params)
            else:
                pred = jax_curve(x_jax, *params)
                
            residuals = y_jax - pred
            return jnp.mean(jnp.square(residuals) / (jnp.square(pred) + 1e-6))
        
        # Define optimization methods to try
        methods = ['BFGS']# if try_all_methods else ['BFGS'] # only bfgs
        
        best_params = jnp.array(params_initial)
        best_n_chi_squared = jnp.inf
        jax_success = False
        
        # Catch warnings during the optimization process
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            
            for method in methods:
                try:
                    if log_everything:
                        logger.info(f"Fitting curve with JAX method {method}, initial parameters {params_initial}")
                    
                    result = minimize(loss_fn, params_initial_jax, method=method, options={'maxiter': 100000})
                    methods_used.append(f"jax_{method}")
                    
                    if result.success:
                        params_opt = result.x
                        n_chi_squared = get_n_chi_squared_jax(params_opt)
                        
                        if log_everything:
                            logger.debug(f"JAX fit complete: n_chi-squared={n_chi_squared}, params={params_opt[0:3]}...")
                        
                        best_params = params_opt
                        best_n_chi_squared = n_chi_squared
                        jax_success = True
                        break
                    else:
                        if log_everything:
                            logger.debug(f"JAX optimization failed: {result.success}, {result.x[0:3]}...")
                        continue
                except Exception as e:
                    if log_everything:
                        logger.info(f"JAX method in fit_curve_with_guess_jax failed: {str(e)[:100]}")
                        if "traced" in str(e):
                            logger.info(f"Here's the function we're fitting: {jax_curve_str}, {numpy_curve_str}")
                    continue
        
        # If requested, try regular fitting with the JAX results as initial guess
        reg_fitting_success = False
        if then_do_reg_fitting:
            try:
                reg_curve = np_curve if np_curve is not None else curve
                reg_params, reg_n_chi_squared = fit_curve_with_guess(
                    np.array(x), np.array(y), reg_curve, 
                    np.array(best_params), try_all_methods=False, 
                    log_everything=log_everything,  # Only pass through detailed logging if requested
                    stats=stats
                )
                methods_used.append("regular_after_jax")
                
                # Compare results and take the better one
                if reg_n_chi_squared < best_n_chi_squared:
                    if log_everything:
                        logger.info(f"Regular fitting improved results: n_chi-squared from {best_n_chi_squared} to {reg_n_chi_squared}")
                    best_params = reg_params
                    best_n_chi_squared = reg_n_chi_squared
                    reg_fitting_success = True
                elif log_everything and jax_success:
                    logger.info(f"JAX fitting had better results: n_chi-squared {best_n_chi_squared} vs {reg_n_chi_squared}")
            except Exception as e:
                methods_used.append("regular_after_jax_failed")
                if log_everything:
                    logger.info(f"Regular fitting failed, keeping JAX results. Error: {str(e)[:100]}")
        
        # Report methods used before returning - but only if explicitly requested
        if log_methods or log_everything:
            methods_str = ", ".join(methods_used)
            if jax_success:
                success_method = "JAX" if not reg_fitting_success else "Regular after JAX"
                logger.info(f"Curve fitting completed using methods: {methods_str}. Best method: {success_method}")
            else:
                logger.info(f"Curve fitting completed using methods: {methods_str}. No successful optimization.")
        
        # Restore original warning handler before returning
        warnings.showwarning = original_showwarning
        # Always return a numpy array
        return np.array(best_params), float(best_n_chi_squared)
    
    except Exception as e:
        # Restore original warning handler in case of exception
        warnings.showwarning = original_showwarning
        
        # Report methods used before re-raising - but only if explicitly requested
        if log_methods or log_everything:
            methods_str = ", ".join(methods_used)
            logger.error(f"JAX fitting failed with error: {str(e)}. Methods tried: {methods_str}")
        else:
            logger.error(f"JAX fitting failed with error: {str(e)}")
        
        # Re-raise the exception to be handled by the caller
        raise

# Test if simplified expression is functionally equivalent to raw expression
def test_expression_equivalence(expr1_np, expr2_np, lambda_xi, test_xs, rtol=1e-3, almost_eveywhere_fraction=0.99):
    """Test if two expressions produce similar outputs for the same inputs."""
    try:
        f1 = eval(lambda_xi + ": " + expr1_np, {"np": np})
        f2 = eval(lambda_xi + ": " + expr2_np, {"np": np})
        
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
        
        # Check if results are close almost everywhere
        if np.allclose(results1, results2, rtol=rtol):
            return True, None
        
        # Calculate percentage of points that are close
        close_points = np.isclose(results1, results2, rtol=rtol)
        percent_close = np.mean(close_points) * 100
        
        # If most points are close (>90%), return True without warning if all points are close
        if percent_close > almost_eveywhere_fraction:
            if percent_close == 100:
                return True, None
            
            # Calculate average relative difference for warning
            rel_diff = np.abs((results1 - results2) / np.maximum(np.abs(results1), 1e-10))
            avg_rel_diff = np.mean(rel_diff)
            return True, f"Warning: Expressions differ at {100-percent_close:.1f}% of points (avg diff: {avg_rel_diff:.4e})"
        else:
            # Calculate average relative difference
            rel_diff = np.abs((results1 - results2) / np.maximum(np.abs(results1), 1e-10))
            avg_rel_diff = np.mean(rel_diff)
            return False, avg_rel_diff
    except Exception as e:
        return False, str(e)

# Test function
if __name__ == "__main__":
    # Set up logging for test
    logging.basicConfig(level=logging.INFO)
    test_func_str = "lambda x, a, b, c: a * np.exp(-b * x) + c"
    test_func = eval(test_func_str)
    test_func_jax_str = test_func_str.replace('np.', 'jnp.')
    test_func_jax = eval(test_func_jax_str)
    # Define a test function to fit

    
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
    jax_params, jax_chi2 = fit_curve_with_guess_jax(
        x_data, y_noisy, 
        test_func_jax, 
        initial_guess,
    )
    jax_elapsed = time.time() - jax_start_time
    
    print('\nTesting JAX version with method logging')
    # Time JAX version with method logging
    jax_methods_start_time = time.time()
    jax_methods_params, jax_methods_chi2 = fit_curve_with_guess_jax(
        x_data, y_noisy, 
        test_func_jax, 
        initial_guess,
        log_methods=True
    )
    jax_methods_elapsed = time.time() - jax_methods_start_time
    
    print('\nTesting JAX version with full logging')
    # Time JAX version with full logging
    jax_full_start_time = time.time()
    jax_full_params, jax_full_chi2 = fit_curve_with_guess_jax(
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
    print("\nTesting string-based function with secondary regular optimization:")
    secondary_start_time = time.time()
    secondary_params, secondary_chi2 = fit_curve_with_guess_jax(
        x_data, y_noisy,
        None,  # We're using the string version instead
        initial_guess,
        log_everything=False,
        log_methods=True,
        then_do_reg_fitting=True,
        numpy_curve_str=test_func_str
    )
    secondary_elapsed = time.time() - secondary_start_time
    
    print("\nSecondary regular optimization test results:")
    print(f"True parameters: {true_params}")
    print(f"Secondary regular optimization fit parameters: {secondary_params}")
    print(f"Secondary regular optimization n_chi-squared: {secondary_chi2}")
    print(f"Secondary regular optimization execution time: {secondary_elapsed:.4f} seconds")
    
    # Compare with previous results
    print("\nParameter differences:")
    print(f"NumPy vs Secondary regular optimization: {np.abs(np_params - secondary_params).sum():.6f}")
    print(f"JAX vs Secondary regular optimization: {np.abs(jax_params - secondary_params).sum():.6f}")
    print(f"All params: {np_params}, {jax_params}, {secondary_params}")