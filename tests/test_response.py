import os
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from LLMSR.response import extract_ansatz, fun_convert

class TestResponse(unittest.TestCase):
    def test_extract_ansatz_markdown_code_block(self):
        """Test extraction of ansatz from markdown code block"""
        # Create a mock response object with markdown code block
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "```python\ncurve_1 = lambda x, *params: params[0] * x**2 + params[1] * x + params[2]\n```"
        
        ansatz, largest_entry = extract_ansatz(mock_response)
        
        # Check the ansatz - using assertIn because the exact whitespace might vary
        self.assertIn("params[0] * x**2 + params[1] * x + params[2]", ansatz)
        # Check the largest entry (should be 3 as we have params[0], params[1], params[2])
        self.assertEqual(largest_entry, 3)
    
    def test_extract_ansatz_plain_text(self):
        """Test extraction of ansatz from plain text response"""
        # Test extraction from plain text
        text_response = "I think the best function to fit this data is:\nparams[0] * np.sin(x) + params[1] * np.cos(x)"
        ansatz, largest_entry = extract_ansatz(text_response)
        
        self.assertIn("params[0] * np.sin(x) + params[1] * np.cos(x)", ansatz)
        self.assertEqual(largest_entry, 2)
    
    def test_extract_ansatz_backtick_expression(self):
        """Test extraction of ansatz from backtick-enclosed expression"""
        # Test extraction from backtick-enclosed expression
        backtick_response = "The function `params[0] * np.exp(params[1] * x)` would fit this data well."
        ansatz, largest_entry = extract_ansatz(backtick_response)
        
        self.assertIn("params[0] * np.exp(params[1] * x)", ansatz)
        self.assertEqual(largest_entry, 2)
    
    def test_extract_ansatz_error_handling(self):
        """Test error handling in extract_ansatz"""
        # Test with a response that doesn't contain params
        invalid_response = "This is a response with no function"
        
        with self.assertRaises(ValueError):
            extract_ansatz(invalid_response)
    
    def test_fun_convert_polynomial(self):
        """Test conversion of polynomial string to lambda function"""
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
    
    def test_fun_convert_trig(self):
        """Test conversion of trigonometric string to lambda function"""
        ansatz = "params[0] * np.sin(x) + params[1] * np.cos(x)"
        func, num_params, lambda_str = fun_convert(ansatz)
        
        # Test function with some values
        x = np.array([0, np.pi/2, np.pi])
        params = [2, 3]  # 2*sin(x) + 3*cos(x)
        
        # Check that num_params is correctly determined
        self.assertEqual(num_params, 2)
        
        # Expected values at 0, π/2, π: 3, 2, -3
        expected = np.array([3, 2, -3])
        result = func(x, *params)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_fun_convert_exponential(self):
        """Test conversion of exponential string to lambda function"""
        ansatz = "params[0] * np.exp(params[1] * x)"
        func, num_params, lambda_str = fun_convert(ansatz)
        
        # Test function with some values
        x = np.array([0, 1, 2])
        params = [3, 0.5]  # 3*exp(0.5*x)
        
        # Check that num_params is correctly determined
        self.assertEqual(num_params, 2)
        
        # Expected values at 0, 1, 2: 3, 3*exp(0.5), 3*exp(1)
        expected = np.array([3, 3 * np.exp(0.5), 3 * np.exp(1)])
        result = func(x, *params)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_fun_convert_error_handling(self):
        """Test error handling in fun_convert"""
        # Test with an invalid expression
        invalid_ansatz = "this is not a valid expression"
        
        with self.assertRaises(ValueError):
            fun_convert(invalid_ansatz)
    
    def test_fun_convert_tuple_input(self):
        """Test fun_convert with tuple input"""
        # Test with tuple input as returned by extract_ansatz
        ansatz_tuple = ("params[0] * x + params[1]", 2)
        func, num_params, lambda_str = fun_convert(ansatz_tuple)
        
        # Test function with some values
        x = np.array([1, 2, 3])
        params = [2, 5]  # 2x + 5
        
        # Check that num_params is correctly determined
        self.assertEqual(num_params, 2)
        
        # Expected values
        expected = np.array([7, 9, 11])
        result = func(x, *params)
        
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == '__main__':
    unittest.main()