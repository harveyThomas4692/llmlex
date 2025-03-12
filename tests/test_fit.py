import os
import sys
import unittest
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from LLMSR.fit import fit_curve
from tests.test_data.generate_test_data import generate_test_data

class TestFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate test data if it doesn't exist
        if not os.path.exists('tests/test_data/test_data.npz'):
            cls.x, cls.y, _ = generate_test_data()
        else:
            # Load existing test data
            test_data = np.load('tests/test_data/test_data.npz')
            cls.x, cls.y = test_data['x'], test_data['y']
    
    def test_fit_curve(self):
        """Test fitting a curve to data"""
        # Create a test function
        def test_func(x, a, b, c):
            return a * x**2 + b * x + c
        
        # Fit the curve
        params, n_chi_squared = fit_curve(self.x, self.y, test_func, 3)
        
        # Check that we get reasonable parameters
        # We expect close to [2, 3, 5] since our test data is 2x^2 + 3x + 5
        np.testing.assert_array_almost_equal(params, np.array([2, 3, 5]), decimal=2)
        
        # Check that n_chi-squared is small
        self.assertLess(n_chi_squared, 1e-6)
    
    def test_fit_curve_error_handling(self):
        """Test error handling in fit_curve"""
        # Create a function that will raise an exception
        def bad_func(x, *params):
            raise ValueError("Intentional test error")
        # Try to fit the curve
        # Temporarily disable logging during this test so we don't see the error message
        # Temporarily suppress fit module logging
        import logging
        logger = logging.getLogger("LLMSR.fit")
        original_level = logger.level
        logger.setLevel(logging.CRITICAL)
        
        try:
            params, n_chi_squared = fit_curve(self.x, self.y, bad_func, 3)
        finally:
            # Restore original logging level
            logger.setLevel(original_level)
            
        # Check that we get the expected default values
        np.testing.assert_array_equal(params, np.ones(3))
        self.assertEqual(n_chi_squared, np.inf)

if __name__ == '__main__':
    unittest.main()