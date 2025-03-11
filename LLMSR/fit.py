from scipy.optimize import curve_fit
import numpy as np
from scipy import special
import logging

# Get module logger
logger = logging.getLogger("LLMSR.fit")


def get_chi_squared(x, y, curve, params):
    residuals = y - curve(x, *params)
    chi_squared = np.mean((residuals ** 2) / (np.square(curve(x, *params))+1e-6))
    return chi_squared

def fit_curve_with_guess(x, y, curve, params_initial, try_all_methods=False, log_everything=False):
    if try_all_methods:
        try:
            if log_everything:
                logger.info(f"Fitting curve with initial parameters {params_initial}")
            params_opt, _ = curve_fit(curve, x, y, p0=params_initial, maxfev=1000*len(params_initial))
            chi_squared = get_chi_squared(x, y, curve, params_opt)
            if log_everything:
                logger.info(f"Fit complete: chi-squared={chi_squared}")
            return params_opt, chi_squared 
        except RuntimeError as e:
            methods = ['lm', 'trf', 'dogbox']
            for method in methods:
                try:
                    if log_everything:
                        logger.info(f"Fitting curve with method {method}")
                    params_opt, _ = curve_fit(curve, x, y, p0=params_initial, method=method, maxfev=1000*len(params_initial))
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
                logger.info(f"Fitting curve with initial parameters {params_initial}")
            params_opt, _ = curve_fit(curve, x, y, p0=params_initial, maxfev=1000*len(params_initial))
            chi_squared = get_chi_squared(x, y, curve, params_opt)
            if log_everything:
                logger.info(f"Fit complete: chi-squared={chi_squared}")
            return params_opt, chi_squared
        except RuntimeError as e:
            if log_everything:
                logger.info(f"Curve fitting failed: {str(e)[:100]}")
            return np.array(params_initial), np.inf

def fit_curve(x, y, curve, largest_entry: int):
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
        return params_opt, chi_squared
        
    except Exception as e:
        # Log error and return initial parameters with infinite chi-squared
        logger.info(f"All methods failed for this fit {curve} {str(e)[:100]}, ")
        return np.array(params_initial), np.inf