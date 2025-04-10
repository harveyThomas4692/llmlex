import os
import sys
import unittest
import numpy as np
import warnings

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from llmlex.fit import fit_curve, get_n_chi_squared_from_predictions

class TestFit(unittest.TestCase):
    def setUp(self):
        # Generate synthetic test data for a quadratic function: 2x^2 + 3x + 5
        self.x = np.linspace(0, 10, 100)
        self.y = 2 * self.x**2 + 3 * self.x + 5
        
        # Add a small amount of noise
        np.random.seed(42)
        self.y_noisy = self.y + np.random.normal(0, 1, size=len(self.y))
        
        # Define test function
        self.test_func = lambda x, a, b, c: a * x**2 + b * x + c
    
    def test_fit_curve_accurate(self):
        """Test that fit_curve accurately fits data without noise"""
        # Fit the curve to data without noise
        params, n_chi_squared = fit_curve(self.x, self.y, self.test_func, 3)
        
        # Check that fit produces a good result (low chi-squared)
        self.assertLess(n_chi_squared, 1e-6)
        
        # Check that the function with fit parameters matches the data
        predicted = self.test_func(self.x, *params)
        np.testing.assert_array_almost_equal(predicted, self.y, decimal=3)
    
    def test_fit_curve_with_noise(self):
        """Test that fit_curve works well with noisy data"""
        # Fit the curve to noisy data
        params, n_chi_squared = fit_curve(self.x, self.y_noisy, self.test_func, 3)
        
        # With noise, chi-squared will be higher but should still represent a good fit
        self.assertLess(n_chi_squared, 0.5)
        
        # The parameters should be approximately correct despite noise
        predicted = self.test_func(self.x, *params)
        
        # Calculate R^2 value (coefficient of determination)
        ss_total = np.sum((self.y_noisy - np.mean(self.y_noisy))**2)
        ss_residual = np.sum((self.y_noisy - predicted)**2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # R^2 should be high for a good fit
        self.assertGreater(r_squared, 0.95)
    
    def test_fit_curve_wrong_function(self):
        """Test fitting with the wrong function type"""
        # Try to fit a linear function to quadratic data
        linear_func = lambda x, a, b: a * x + b
        
        params, n_chi_squared = fit_curve(self.x, self.y, linear_func, 2)
        
        # Chi-squared should be positive for a poor fit
        # The exact value depends on implementation so use a very small threshold
        self.assertGreater(n_chi_squared, 0.001)
    
    def test_fit_curve_error_handling(self):
        """Test that fit_curve properly handles errors"""
        # Create a function that will raise an exception
        def bad_func(x, *params):
            raise ValueError("Intentional test error")
        
        # Temporarily suppress fit module logging
        import logging
        logger = logging.getLogger("llmlex.fit")
        original_level = logger.level
        logger.setLevel(logging.CRITICAL)
        
        try:
            # Fit should complete despite the error
            params, n_chi_squared = fit_curve(self.x, self.y, bad_func, 3)
            
            # Should return infinite chi-squared for failed fits
            self.assertEqual(n_chi_squared, np.inf)
            
            # Should return default parameters
            self.assertEqual(len(params), 3)
        finally:
            # Restore original logging level
            logger.setLevel(original_level)
    
    def test_n_chi_squared_calculation(self):
        """Test that get_n_chi_squared_from_predictions works correctly"""
        # Generate predictions with known error
        perfect_predictions = self.y.copy()
        poor_predictions = self.y + 10  # Large constant offset
        
        # Calculate n_chi_squared for perfect and poor fits
        perfect_chi = get_n_chi_squared_from_predictions(self.x, self.y, perfect_predictions)
        poor_chi = get_n_chi_squared_from_predictions(self.x, self.y, poor_predictions)
        
        # Perfect predictions should have very low chi-squared, but allow for numerical precision issues
        self.assertLess(perfect_chi, 0.1)
        
        # Poor predictions should have non-zero chi-squared
        self.assertGreater(poor_chi, 0.0)
        
        # Chi-squared should increase as predictions get worse
        self.assertLess(perfect_chi, poor_chi)

if __name__ == '__main__':
    unittest.main()