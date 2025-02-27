from scipy.optimize import curve_fit
import numpy as np
from scipy import special

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
    try:
        params_initial = np.ones(largest_entry)
        params_opt, _ = curve_fit(curve, x, y, p0=params_initial)
        residuals = y - curve(x, *params_opt)
        chi_squared = np.mean((residuals ** 2) / (np.square(curve(x, *params_opt))+1e-6))
        return params_opt, chi_squared
    except Exception as e:
        print(e)
        print("Getting another response from LLM")
        return np.array(params_initial), np.inf