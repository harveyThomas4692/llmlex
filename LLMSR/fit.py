from scipy.optimize import curve_fit
import numpy as np
from scipy import special
import logging
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from functools import partial
import time
# Get module logger
logger = logging.getLogger("LLMSR.fit")

# Disable 64-bit precision - for some reason 64-bit doesn't work! That's why we iterate and use the regular fit after JAX.
jax.config.update("jax_enable_x64", False)

def get_chi_squared(x, y, curve, params):
    if len(x.squeeze().shape) > 1:
        residuals = y - curve(*[x[:, i] for i in range(x.shape[1])], *params)
        curvevals = curve(*[x[:, i] for i in range(x.shape[1])], *params)       
    else:   
        residuals = y - curve(x, *params)
        curvevals = curve(x, *params)
    chi_squared = np.mean((residuals ** 2) / (np.square(curvevals)+1e-6))
    return chi_squared

def fit_curve_with_guess(x, y, curve, params_initial, try_all_methods=False, log_everything=False):
    if len(x.squeeze().shape) > 1:
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
            params_opt, _ = curve_fit_here(curve, x, y, p0=params_initial, maxfev=1000*len(params_initial))
            chi_squared = get_chi_squared(x, y, curve, params_opt)
            if log_everything:
                logger.info(f"Fit complete: chi-squared={chi_squared}")
            return params_opt, chi_squared 
        except RuntimeError as e:
            methods = ['lm', 'trf', 'dogbox']
            for method in methods:
                try:
                    if log_everything:
                        logger.info(f"Fitting curve with method {method}, " + "trf may take some time" if method == 'trf' else "")
                    params_opt, _ = curve_fit_here(curve, x, y, p0=params_initial, method=method, maxfev=1000*len(params_initial))
                    chi_squared = get_chi_squared(x, y, curve, params_opt)
                    if log_everything:
                        logger.info(f"Fit complete: chi-squared={chi_squared}")
                    return params_opt, chi_squared
                except RuntimeError:
                    continue
            logger.info(f"All methods failed for this fit {curve} {str(e)[:100]}, ")
            return np.array(params_initial), np.inf
    else:
        try:
            if log_everything:
                logger.info(f"Fitting curve with initial parameters {params_initial[0:3]}...")
            params_opt, _ = curve_fit_here(curve, x, y, p0=params_initial, maxfev=1000*len(params_initial))
            chi_squared = get_chi_squared(x, y, curve, params_opt)
            if log_everything:
                logger.info(f"Fit complete: chi-squared={chi_squared}")
            return params_opt, chi_squared
        except RuntimeError as e:
            if log_everything:
                logger.info(f"Curve fitting failed: {str(e)[:100]}")
            return np.array(params_initial), np.inf

def fit_curve(x, y, curve, largest_entry: int, curve_str: str = None, allow_using_jax: bool = True, force_using_jax: bool = False):
    """
    Fits a given curve to the provided data points (x, y) and calculates the chi-squared value.
    Parameters:
        x (array-like): The independent variable data points.
        y (array-like): The dependent variable data points.
        curve (callable): The curve function to fit, which should take x and parameters as inputs.
        largest_entry (int): The number of parameters for the curve function.
    Returns:
    tuple: A tuple containing:
        - params_opt (array-like): The optimized parameters for the curve.
        - chi_squared (float): The chi-squared value indicating the goodness of fit.
    """
    logger.debug(f"Fitting curve with {largest_entry} parameters")
    logger.debug(f"Data shape: x={len(x)}, y={len(y)}")
    params_initial = np.ones(largest_entry)
    chi_squared = np.inf
    
    try:
        # Initialize parameter, perform curve fitting
        logger.debug(f"Running curve_fit optimization, initial parameters {params_initial}")
        try:
            params_opt, covariance = curve_fit(curve, x, y, p0=params_initial, maxfev=1000*largest_entry)
            logger.debug(f"Optimized parameters: {params_opt}")
        except RuntimeError as e:
            # If initial fit fails, try with different methods
            logger.debug(f"Initial fit failed: {e}. Trying with different methods")
            methods = ['lm', 'trf', 'dogbox']
            for method in methods:
                try:
                    logger.debug(f"Trying method: {method}")
                    params_opt, covariance = curve_fit(curve, x, y, p0=params_initial, method=method, maxfev=1000*largest_entry)
                    logger.debug(f"Optimized parameters with method {method}: {params_opt}")
                    break
                except RuntimeError as e:
                    logger.debug(f"Method {method} failed: {e}")
            else:
                # If all methods fail, try with random parameters
                random_params = np.random.uniform(-1.0, 1.0, largest_entry)
                params_opt, covariance = curve_fit(curve, x, y, p0=random_params, maxfev=1000*largest_entry)
                logger.debug(f"Optimized parameters with random init: {params_opt}")

        
        # Calculate residuals and chi-squared
        logger.debug("Calculating fit quality metrics")
        chi_squared = get_chi_squared(x, y, curve, params_opt)
        
        logger.debug(f"Fit complete: chi-squared={chi_squared}")
        if force_using_jax:
            params_opt_jax, chi_squared_jax = fit_curve_with_guess_jax(x, y, curve, params_opt, try_all_methods=True, log_everything=True, then_do_reg_fitting=True, numpy_curve_str=curve_str)
            if chi_squared_jax < chi_squared:
                params_opt = params_opt_jax
                chi_squared = chi_squared_jax
        return params_opt, chi_squared
        
    except Exception as e:
        if allow_using_jax:
            params_opt_jax, chi_squared_jax = fit_curve_with_guess_jax(x, y, curve, params_initial, try_all_methods=True, log_everything=True, then_do_reg_fitting=True, numpy_curve_str=curve_str)
            if chi_squared_jax < chi_squared:
                params_opt = params_opt_jax
                chi_squared = chi_squared_jax
        # Log error and return initial parameters with infinite chi-squared
        logger.info(f"All methods failed for this fit {curve} {str(e)[:100]}, ")
        return np.array(params_initial), np.inf

def fit_curve_with_guess_jax(x, y, curve, params_initial, try_all_methods=False, log_everything=False, then_do_reg_fitting=False, numpy_curve_str=None):
    """JAX implementation of curve fitting that mimics fit_curve_with_guess."""
    # Convert inputs to JAX arrays
    x_jax = jnp.array(x)
    y_jax = jnp.array(y)
    params_initial_jax = jnp.array(params_initial)
    multivariateQ = len(x_jax.squeeze().shape) > 1

    
    # If numpy_curve_str is provided, extract JAX version for the fitting
    jax_curve = curve
    np_curve = None
    if numpy_curve_str is not None:
        jax_curve_str = numpy_curve_str.replace('np.', 'jnp.')
        jax_curve = eval(jax_curve_str)
        np_curve = eval(numpy_curve_str)
        print(f"Converting numpy curve string to JAX curve string: {numpy_curve_str}, jax_curve_str: {jax_curve_str}")

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
    
    # Method to calculate chi-squared for JAX
    def get_chi_squared_jax(params):
        if multivariateQ:
            pred = jax_curve(*[x_jax[:, i] for i in range(x_dim)], *params)
        else:
            pred = jax_curve(x_jax, *params)
            
        residuals = y_jax - pred
        return jnp.mean(jnp.square(residuals) / (jnp.square(pred) + 1e-6))
    
    # Define optimization methods to try
    methods = ['BFGS'] if try_all_methods else ['BFGS'] # only bfgs
    
    best_params = jnp.array(params_initial)
    best_chi_squared = jnp.inf
    jax_success = False
    
    for method in methods:
        try:
            if log_everything:
                logger.info(f"Fitting curve with JAX method {method}, initial parameters {params_initial}")
            
            result = minimize(loss_fn, params_initial_jax, method=method, options={'maxiter': 100000})
            
            if result.success:
                params_opt = result.x
                chi_squared = get_chi_squared_jax(params_opt)
                
                if log_everything:
                    print(f"JAX fit complete: chi-squared={chi_squared}, params={params_opt[0:3]}...")
                
                best_params = params_opt
                best_chi_squared = chi_squared
                jax_success = True
                break
            else:
                if log_everything:
                    print(f"JAX optimization failed: {result.success}, {result.x[0:3]}...")
                continue
        except Exception as e:
            if log_everything:
                logger.info(f"JAX method {method} failed: {str(e)[:100]}")
            continue
    
    # If requested, try regular fitting with the JAX results as initial guess
    if then_do_reg_fitting:
        try:
            reg_curve = np_curve if np_curve is not None else curve
            reg_params, reg_chi_squared = fit_curve_with_guess(
                np.array(x), np.array(y), reg_curve, 
                np.array(best_params), try_all_methods=False, log_everything=log_everything
            )
            
            # Compare results and take the better one
            if reg_chi_squared < best_chi_squared:
                if log_everything:
                    logger.info(f"Regular fitting improved results: chi-squared from {best_chi_squared} to {reg_chi_squared}")
                best_params = reg_params
                best_chi_squared = reg_chi_squared
            elif log_everything and jax_success:
                logger.info(f"JAX fitting had better results: chi-squared {best_chi_squared} vs {reg_chi_squared}")
        except Exception as e:
            if log_everything:
                logger.info(f"Regular fitting failed, keeping JAX results. Error: {str(e)[:100]}")
    
    # Always return a numpy array
    return np.array(best_params), float(best_chi_squared)

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
    
    print('testing jax') 
    # Time JAX version
    jax_start_time = time.time()
    jax_params, jax_chi2 = fit_curve_with_guess_jax(
        x_data, y_noisy, 
        test_func_jax, 
        initial_guess, 
        log_everything=True,
        try_all_methods=True
    )
    jax_elapsed = time.time() - jax_start_time
    
    print("\nResults comparison:")
    print(f"True parameters: {true_params}")
    print(f"NumPy fit parameters: {np_params}")
    print(f"JAX fit parameters: {jax_params}")
    print(f"NumPy chi-squared: {np_chi2}")
    print(f"JAX chi-squared: {jax_chi2}")
    print("\nTiming comparison:")
    print(f"NumPy version: {np_elapsed:.4f} seconds")
    print(f"JAX version: {jax_elapsed:.4f} seconds")
    print(f"Speedup: {np_elapsed/jax_elapsed:.2f}x")
    
    # Test new string-based function with fallback
    print("\nTesting string-based function with secondary regular optimization:")
    secondary_start_time = time.time()
    secondary_params, secondary_chi2 = fit_curve_with_guess_jax(
        x_data, y_noisy,
        None,  # We're using the string version instead
        initial_guess,
        log_everything=True,
        then_do_reg_fitting=True,
        numpy_curve_str=test_func_str
    )
    secondary_elapsed = time.time() - secondary_start_time
    
    print("\nSecondary regular optimization test results:")
    print(f"True parameters: {true_params}")
    print(f"Secondary regular optimization fit parameters: {secondary_params}")
    print(f"Secondary regular optimization chi-squared: {secondary_chi2}")
    print(f"Secondary regular optimization execution time: {secondary_elapsed:.4f} seconds")
    
    # Compare with previous results
    print("\nParameter differences:")
    print(f"NumPy vs Secondary regular optimization: {np.abs(np_params - secondary_params).sum():.6f}")
    print(f"JAX vs Secondary regular optimization: {np.abs(jax_params - secondary_params).sum():.6f}")
    print(f"All params: {np_params}, {jax_params}, {secondary_params}")