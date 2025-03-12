import os
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from LLMSR.response import extract_ansatz, fun_convert

class TestResponse(unittest.TestCase):
    def test_extract_ansatz(self):
        """Test extraction of ansatz from response"""
        # Create a mock response object
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "```python\ncurve_1 = lambda x, *params: params[0] * x**2 + params[1] * x + params[2]\n```"
        
        ansatz, largest_entry = extract_ansatz(mock_response)
        
        # Check the ansatz - using assertIn because the exact whitespace might vary
        self.assertIn("params[0] * x**2 + params[1] * x + params[2]", ansatz)
        # Check the largest entry (should be 3 as we have params[0], params[1], params[2])
        self.assertEqual(largest_entry, 3)
    
    def test_fun_convert(self):
        """Test conversion of string to lambda function"""
        ansatz = "params[0] * x**2 + params[1] * x + params[2]"
        func, num_params, lambda_str = fun_convert(ansatz)
        
        # Test function with some values
        x = np.array([1, 2, 3])
        params = [2, 3, 5]  # 2x^2 + 3x + 5
        
        # Check that num_params is correctly determined
        self.assertEqual(num_params, 3)
        
        # Expected values: 2*1^2 + 3*1 + 5 = 10, 2*2^2 + 3*2 + 5 = 19, 2*3^2 + 3*3 + 5 = 32
        expected = np.array([10, 19, 32])
        result = func(x, *params)
        
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == '__main__':
    unittest.main()