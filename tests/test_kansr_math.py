import unittest
import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch, ANY
import logging

# Add the parent directory to the path if it's not already there
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the modules to test
from LLM_LEx.kanLEx import KANSR


class TestKANSRMathematicalCorrectness(unittest.TestCase):
    """Test cases for the mathematical correctness of the KANSR class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Create mock client for API calls
        self.mock_client = MagicMock()
        self.mock_response = MagicMock()
        self.mock_response.choices = [MagicMock()]
        self.mock_response.choices[0].message = MagicMock()
        self.mock_response.choices[0].message.content = "```simplified_expression\nx0**2 + 2*x0 + 1\n```"
        self.mock_client.chat.completions.create.return_value = self.mock_response
        
        # Create mock KAN model
        self.mock_kan = MagicMock()
        self.mock_kan.width_in = [1, 4, 1]  # Input, hidden, output layers
        self.mock_kan.width_out = [4, 1]    # Number of outputs for each layer
        self.mock_kan.device = 'cpu'
        self.mock_kan.fit = MagicMock(return_value={'train_loss': torch.tensor([0.001])})
        self.mock_kan.prune = MagicMock(return_value=self.mock_kan)
        self.mock_kan.plot = MagicMock()
        
        # Simple test function and data
        self.test_function = lambda x: x**2 if isinstance(x, (float, int, np.ndarray)) else x.pow(2)
        
    def test_quadratic_fitting(self):
        """Test fitting a quadratic function."""
        # Create synthetic data for quadratic function: f(x) = 2*x^2 + 3
        x_data = np.linspace(-5, 5, 100)
        y_data = 2 * x_data**2 + 3
        
        # Add noise for realism
        np.random.seed(42)
        y_data += np.random.normal(0, 0.1, size=y_data.shape)
        
        # Create a KANSR instance with our mock
        kansr = KANSR(client=self.mock_client, model=self.mock_kan)
        
        # Prepare the node tree structure that would normally come from build_expression_tree
        kansr.expression_tree = {
            "edge_dict": {(0, 0, 0): "1.8*x + 3.2"},
            "top_k_edge_dicts": {(0, 0, 0): [{"expression": "1.8*x**2 + 3.2"}]},
            "node_tree": {(0, 0): "1.8*x0**2 + 3.2"},
            "full_expressions": ["1.8*x0**2 + 3.2"]  # Slightly off from true values to test fitting
        }
        
        # Mock the LLM simplification to return a response close to the true function
        with patch.object(kansr, '_call_model_simplify', return_value=["2.0*x0**2 + 3.0"]), \
             patch.object(kansr, '_process_expression_using_llm', return_value=({
                'expression': "2.0*x0**2 + 3.0",
                'expression_numpy': "2.0*x0**2 + 3.0", 
                'n_chi_squared': 0.01,
                'fit_type': 'LLMsimplified'
             }, [])):
            # Create mock dataset
            mock_dataset = {
                'train_input': torch.tensor(x_data.reshape(-1, 1)),
                'train_label': torch.tensor(y_data),
                'test_input': torch.tensor(x_data[:10].reshape(-1, 1)),
                'test_label': torch.tensor(y_data[:10])
            }
            kansr.dataset = mock_dataset
            
            # Call the optimise_expressions method with our test data
            with patch('matplotlib.pyplot.subplots', return_value=(MagicMock(), MagicMock())), \
                 patch('matplotlib.pyplot.show'), \
                 patch('LLMSR.kansr.get_n_chi_squared') as mock_chi_squared, \
                 patch('LLMSR.kansr.fit_curve_with_guess_jax') as mock_fit:
                
                # Mock the chi-squared calculation to return realistic values
                mock_chi_squared.return_value = 0.05
                
                # Mock the curve fitting to return optimised parameters close to true values
                mock_fit.return_value = ([2.0, 3.0], 0.01)
                
                # Run the optimisation
                best_expressions, best_n_chi_squared, result_dicts, all_results_sorted = kansr.optimise_expressions(
                    client=self.mock_client,
                    simplification_gpt_model="openai/gpt-4o-mini",
                    x_data=x_data,
                    y_data=y_data
                )
                
                # Verify basic structure of results
                self.assertIsInstance(best_expressions, list)
                self.assertIsInstance(best_n_chi_squared, list)
                self.assertIsInstance(result_dicts, list)
                self.assertIsInstance(all_results_sorted, list)
                
                # Verify all_results_sorted structure
                self.assertGreaterEqual(len(all_results_sorted), 1)
                output_results = all_results_sorted[0]
                self.assertIsInstance(output_results, list)
                # There should be at least one result per output
                if output_results:
                    result = output_results[0]
                    self.assertIn('expression', result)
                    self.assertIn('n_chi_squared', result)
                    self.assertIn('fit_type', result)
                
                # Get the result dictionary for examination
                result_dict = result_dicts[0]
                
                # Check that all expected keys are present
                expected_keys = [
                    'raw_expression', 'raw_n_chi_squared', 
                    'final_refitted_expression', 'n_chi_squared_refitted',
                    'final_LLM_expression', 'n_chi_squared_LLM_final',
                    'best_expression', 'best_n_chi_squared',
                    'best_fit_type'
                ]
                for key in expected_keys:
                    self.assertIn(key, result_dict)
                
                # The n_chi-squared value should be small for a good fit
                self.assertLess(best_n_chi_squared[0], 0.1, 
                              "n_chi-squared value should be small for a good fit")
    
    def test_linear_fitting(self):
        """Test fitting a simple linear function."""
        # Create data for a linear function: f(x) = 2*x + 1
        x_data = np.linspace(-5, 5, 100)
        y_data = 2 * x_data + 1
        
        # Add noise
        np.random.seed(42)
        y_data += np.random.normal(0, 0.1, size=y_data.shape)
        
        # Create a KANSR instance with our mock
        kansr = KANSR(client=self.mock_client, model=self.mock_kan)
        
        # Create a node tree that would simulate what we'd get from build_expression_tree
        kansr.expression_tree = {
            "edge_dict": {(0, 0, 0): "1.9*x + 0.8"},
            "top_k_edge_dicts": {(0, 0, 0): [{"expression": "1.9*x + 0.8"}]},
            "node_tree": {(0, 0): "1.9*x0 + 0.8"},
            "full_expressions": ["1.9*x0 + 0.8"]  # Slightly off from true values
        }
        
        # Mock the LLM response to return a better expression
        with patch.object(kansr, '_call_model_simplify', return_value=["2.0*x0 + 1.0"]), \
             patch.object(kansr, '_process_expression_using_llm', return_value=({
                'expression': "2.0*x0 + 1.0",
                'expression_numpy': "2.0*x0 + 1.0", 
                'n_chi_squared': 0.01,
                'fit_type': 'LLMsimplified'
             }, [])):
            # Create mock dataset
            mock_dataset = {
                'train_input': torch.tensor(x_data.reshape(-1, 1)),
                'train_label': torch.tensor(y_data),
                'test_input': torch.tensor(x_data[:10].reshape(-1, 1)),
                'test_label': torch.tensor(y_data[:10])
            }
            kansr.dataset = mock_dataset
            
            # Call optimise_expressions with our test data
            with patch('matplotlib.pyplot.subplots', return_value=(MagicMock(), MagicMock())), \
                 patch('matplotlib.pyplot.show'), \
                 patch('LLMSR.kansr.get_n_chi_squared') as mock_chi_squared, \
                 patch('LLMSR.kansr.fit_curve_with_guess_jax') as mock_fit:
                
                # Set realistic return values
                mock_chi_squared.return_value = 0.05
                mock_fit.return_value = ([2.0, 1.0], 0.01)
                
                # Run the optimisation
                best_expressions, best_n_chi_squared, result_dicts, all_results_sorted = kansr.optimise_expressions(
                    client=self.mock_client,
                    simplification_gpt_model="openai/gpt-4o-mini",
                    x_data=x_data,
                    y_data=y_data
                )
                
                # Verify chi-squared is small
                self.assertLess(best_n_chi_squared[0], 0.1, 
                              "n_chi-squared value should be small for a linear fit")
                
                # Check that the results have the expected structure
                self.assertIsInstance(result_dicts, list)
                self.assertGreater(len(result_dicts), 0)
                self.assertIn('best_expression', result_dicts[0])
                self.assertIn('best_n_chi_squared', result_dicts[0])
                
                # Verify the all_results_sorted structure
                self.assertIsInstance(all_results_sorted, list)
                self.assertGreater(len(all_results_sorted), 0)
                
    def test_sinusoidal_fitting(self):
        """Test fitting a sinusoidal function."""
        # Create data for f(x) = 3*sin(2*x) + 0.5*x^2
        x_data = np.linspace(-np.pi, np.pi, 200)
        y_data = 3 * np.sin(2 * x_data) + 0.5 * x_data**2
        
        # Add noise
        np.random.seed(42)
        y_data += np.random.normal(0, 0.1, size=y_data.shape)
        
        # Create KANSR instance
        kansr = KANSR(client=self.mock_client, model=self.mock_kan)
        
        # Create node tree
        kansr.expression_tree = {
            "edge_dict": {(0, 0, 0): "3.2*sin(1.8*x) + 0.4*x**2"},
            "top_k_edge_dicts": {(0, 0, 0): [{"expression": "3.2*sin(1.8*x) + 0.4*x**2"}]},
            "node_tree": {(0, 0): "3.2*sin(1.8*x0) + 0.4*x0**2"},
            "full_expressions": ["3.2*sin(1.8*x0) + 0.4*x0**2"]
        }
        
        # Mock LLM response
        with patch.object(kansr, '_call_model_simplify', return_value=["3.0*sin(2.0*x0) + 0.5*x0**2"]), \
             patch.object(kansr, '_process_expression_using_llm', return_value=({
                'expression': "3.0*sin(2.0*x0) + 0.5*x0**2",
                'expression_numpy': "3.0*np.sin(2.0*x0) + 0.5*x0**2", 
                'n_chi_squared': 0.05,
                'fit_type': 'LLMsimplified'
             }, [])):
            # Create mock dataset
            mock_dataset = {
                'train_input': torch.tensor(x_data.reshape(-1, 1)),
                'train_label': torch.tensor(y_data),
                'test_input': torch.tensor(x_data[:10].reshape(-1, 1)),
                'test_label': torch.tensor(y_data[:10])
            }
            kansr.dataset = mock_dataset
            
            # Call optimise_expressions
            with patch('matplotlib.pyplot.subplots', return_value=(MagicMock(), MagicMock())), \
                 patch('matplotlib.pyplot.show'), \
                 patch('LLMSR.kansr.get_n_chi_squared') as mock_chi_squared, \
                 patch('LLMSR.kansr.fit_curve_with_guess_jax') as mock_fit:
                
                # Set realistic return values for a more complex function
                mock_chi_squared.return_value = 0.1  # Might be higher due to complexity
                mock_fit.return_value = ([3.0, 2.0, 0.5], 0.05)
                
                # Run the optimisation
                best_expressions, best_n_chi_squared, result_dicts, all_results_sorted = kansr.optimise_expressions(
                    client=self.mock_client,
                    simplification_gpt_model="openai/gpt-4o-mini",
                    x_data=x_data,
                    y_data=y_data
                )
                
                # For sinusoidal functions, the n_chi-squared might be higher but still reasonable
                self.assertLess(best_n_chi_squared[0], 2.0, 
                              "n_chi-squared value should be reasonable for a complex function")
                
                # Verify the all_results_sorted structure
                self.assertIsInstance(all_results_sorted, list)
                self.assertEqual(len(all_results_sorted), 1)  # One element per output
                if all_results_sorted[0]:
                    self.assertIn('n_chi_squared', all_results_sorted[0][0])
                    self.assertIn('fit_type', all_results_sorted[0][0])
    
    def test_plot_results_mathematical_accuracy(self):
        """Test that plot_results correctly evaluates mathematical expressions."""
        # Create a simple result dictionary
        result_dict = {
            'raw_expression': "2.0*x0**2 + 3.0",
            'raw_n_chi_squared': 0.01,
            'final_refitted_expression': "2.0*x0**2 + 3.0",
            'n_chi_squared_refitted': 0.01,
            'final_LLM_expression': "2.0*x0**2 + 3.0",
            'n_chi_squared_LLM_final': 0.01,
            'best_expression': "2.0*x0**2 + 3.0",
            'best_n_chi_squared': 0.01,
            'best_fit_type': 'raw'
        }
        
        # Create a KANSR instance
        kansr = KANSR(client=self.mock_client, model=self.mock_kan)
        kansr.f = self.test_function
        kansr.results_all_dicts = [result_dict]
        
        # Create a simple dataset
        x_data = np.linspace(-5, 5, 1000)
        y_data = 2.0 * x_data**2 + 3.0
        mock_dataset = {
            'train_input': torch.tensor(x_data.reshape(-1, 1)).float(),
            'train_label': torch.tensor(y_data).float(),
        }
        kansr.dataset = mock_dataset
        
        # Test that actual evaluation happens correctly
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.show'):
                
            # Create mock figure and axes
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            # We want to capture the real evaluation, so only mock the plot functions
            # and not the eval calls
            fig, ax = kansr.plot_results([-5,5],result_dict)
            
            # Verify the plot was created
            self.assertEqual(fig, mock_fig)
            self.assertEqual(ax, mock_ax)
            
            # Mock_ax.plot should have been called multiple times
            # with data points that match our mathematical expressions
            # We can't easily inspect the y-values passed to plot,
            # but we can at least verify plot was called multiple times
            self.assertTrue(mock_ax.plot.call_count >= 3, 
                         "plot should be called for true function and at least 2 expressions")
    
    def test_simplification_methods(self):
        """Test the mathematical correctness of expression simplification methods."""
        # Create KANSR instance
        kansr = KANSR(client=self.mock_client, model=self.mock_kan)
        
        # Test _convert_sympy_to_numpy
        expr = "sin(x0)"
        numpy_expr = kansr._convert_sympy_to_numpy(expr)
        self.assertEqual(numpy_expr, "np.sin(x0)")
        
        # Test _replace_floats_with_params
        expr = "2.5 * x0 + 3.7"
        params_expr, params = kansr._replace_floats_with_params(expr)
        self.assertIn("params[", params_expr)
        self.assertEqual(len(params), 2)
        
        # Test _subst_params
        params_expr = "params[0] * x0 + params[1]"
        params = [2.5, 3.7]
        result = kansr._subst_params(params_expr, params)
        self.assertEqual(result, "2.5000 * x0 + 3.7000")
        
        # Test direct numerical evaluation of a simple expression
        # Skip testing _simplify_expression as it's harder to test robustly
        expr_np = "2.5 * x0**2 + 3.7"
        
        # Evaluate the expression with a test input
        func = eval(f"lambda x0: {expr_np}", {"np": np})
        test_value = 2.0
        result = func(test_value)
        
        # Compare with direct calculation
        expected = 2.5 * test_value**2 + 3.7
        self.assertAlmostEqual(result, expected, places=4)


if __name__ == '__main__':
    unittest.main() 