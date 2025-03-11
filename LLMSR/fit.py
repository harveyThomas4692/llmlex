from scipy.optimize import curve_fit
import numpy as np
from scipy import special
import logging

# Get module logger
logger = logging.getLogger("LLMSR.fit")

def fit_curve(x, y, curve, largest_entry):
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
    
    try:
        # Initialize parameters
        params_initial = np.ones(largest_entry)
        logger.debug(f"Initial parameters: {params_initial}")
        
        # Perform curve fitting
        logger.debug("Running curve_fit optimization")
        params_opt, covariance = curve_fit(curve, x, y, p0=params_initial)
        logger.debug(f"Optimized parameters: {params_opt}")
        
        # Calculate residuals and chi-squared
        logger.debug("Calculating fit quality metrics")
        residuals = y - curve(x, *params_opt)
        chi_squared = np.mean((residuals ** 2) / (np.square(curve(x, *params_opt))+1e-6))
        
        logger.debug(f"Fit complete: chi-squared={chi_squared}")
        return params_opt, chi_squared
        
    except Exception as e:
        # Log error and return initial parameters with infinite chi-squared
        logger.debug(f"Error during curve fitting, getting another response: {str(e)[:100]}", exc_info=True)
        return np.array(params_initial), np.inf