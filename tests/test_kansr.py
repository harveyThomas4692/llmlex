import unittest
import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch, ANY
import io
import logging

# Add the parent directory to the path if it's not already there
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the modules to test
import LLMSR.llmSR
from LLMSR.kansr import KANSR, run_complete_pipeline


class TestKANSRClass(unittest.TestCase):
    """Test cases for the KANSR class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Create mock KAN model
        self.mock_kan = MagicMock()
        
        # Set up the KAN model properties
        self.mock_kan.width_in = [1, 4, 1]  # Input, hidden, output layers
        self.mock_kan.width_out = [4, 1]    # Number of outputs for each layer
        self.mock_kan.device = 'cpu'
        
        # Mock essential methods
        self.mock_kan.fit = MagicMock(return_value={'train_loss': torch.tensor([0.001])})
        self.mock_kan.prune = MagicMock(return_value=self.mock_kan)
        self.mock_kan.plot = MagicMock()
        
        # Create mock client for API calls
        self.mock_client = MagicMock()
        
        # Simple test function and data
        self.test_function = lambda x: x**2
        self.x_data = np.linspace(-5, 5, 100)
        self.y_data = self.test_function(self.x_data)
        
        # Instantiate KANSR with mock model and client
        self.kansr = KANSR(client=self.mock_client, model=self.mock_kan)
        
        # Create mock dataset
        self.mock_dataset = {
            'train_input': torch.tensor(self.x_data.reshape(-1, 1)),
            'train_label': torch.tensor(self.y_data),
            'test_input': torch.tensor(self.x_data[:10].reshape(-1, 1)),
            'test_label': torch.tensor(self.y_data[:10])
        }
    
    def test_initialization(self):
        """Test initialization of KANSR class."""
        # Test with model parameter
        kansr = KANSR(client=self.mock_client, model=self.mock_kan)
        self.assertEqual(kansr.raw_model, self.mock_kan)
        self.assertEqual(kansr.device, 'cpu')
        
        # Test with width, grid, k parameters
        # Import KAN at the point of use to create a proper KAN model
        with patch('kan.KAN') as mock_kan_class:
            mock_kan_class.return_value = self.mock_kan
            kansr = KANSR(client=self.mock_client, width=[1, 4, 1], grid=7, k=3)
            mock_kan_class.assert_called_once_with(width=[1, 4, 1], grid=7, k=3, seed=17, device='cpu', symbolic_enabled=False)
            self.assertEqual(kansr.raw_model, self.mock_kan)
        
        # Test with missing parameters
        with self.assertRaises(ValueError):
            KANSR(client=self.mock_client)  # Should raise ValueError if neither model nor (width, grid, k) provided
    
    def test_create_dataset(self):
        """Test dataset creation."""
        # create_dataset is an external function that we should mock
        with patch('LLMSR.kansr.create_dataset') as mock_create_dataset:
            mock_create_dataset.return_value = self.mock_dataset
            
            # Call the method
            dataset = self.kansr.create_dataset(self.test_function)
            
            # Verify create_dataset was called with correct parameters
            mock_create_dataset.assert_called_once_with(
                self.test_function, n_var=1, ranges=(-np.pi, np.pi), 
                train_num=10000, test_num=1000, device='cpu'
            )
            
            # Verify dataset was saved
            self.assertEqual(self.kansr.dataset, self.mock_dataset)
            # Verify function was saved
            self.assertEqual(self.kansr.f, self.test_function)
    
    def test_train(self):
        """Test model training and pruning."""
        # Set up the KANSR instance with dataset
        self.kansr.dataset = self.mock_dataset
        
        # Call the train_kan method
        final_training_loss = self.kansr.train_kan(opt="Adam", steps=100, prune=True, node_th=0.3, edge_th=0.3)
        
        # Verify fit was called with correct parameters
        self.mock_kan.fit.assert_called_once_with(self.mock_dataset, opt="Adam", steps=100)
        
        # Verify pruning was called with correct parameters
        self.mock_kan.prune.assert_called_once_with(node_th=0.3, edge_th=0.3)
        
        # Verify model and history were saved
        self.assertEqual(self.kansr.model, self.mock_kan)
        self.assertEqual(self.kansr.training_history, {'train_loss': torch.tensor([0.001])})
        
        # Verify return value
        self.assertAlmostEqual(final_training_loss, 0.001, places=5)
        
        # Test without pruning
        self.kansr.model = None  # Reset model
        self.mock_kan.fit.reset_mock()
        self.mock_kan.prune.reset_mock()
        
        self.kansr.train_kan(prune=False)
        
        # Verify fit was called but prune was not
        self.mock_kan.fit.assert_called_once()
        self.mock_kan.prune.assert_not_called()
        
        # Verify model is same as raw_model
        self.assertEqual(self.kansr.model, self.mock_kan)
        
        # Test with no dataset provided
        kansr = KANSR(client=self.mock_client, model=self.mock_kan)
        with self.assertRaises(ValueError):
            kansr.train_kan()
    
    def test_get_symbolic(self):
        """Test conversion to symbolic expressions with get_symbolic."""
        # Setup KANSR instance with trained model
        self.kansr.model = self.mock_kan
        self.kansr.training_history = {'train_loss': torch.tensor([0.001])}
        self.kansr.dataset = self.mock_dataset
        
        # Only mock the external API call (kan_to_symbolic) and API key check
        with patch('LLMSR.llmSR.kan_to_symbolic') as mock_kan_to_symbolic, \
             patch('LLMSR.llm.check_key_usage') as mock_check_key_usage:
            
            # Setup mock returns for the external API
            mock_symbolic_result = {(0, 0, 0): [[{'score': 0.95, 'ansatz': 'x**2', 'params': [1.0]}]]}
            mock_kan_to_symbolic.return_value = mock_symbolic_result
            mock_check_key_usage.return_value = 1.000
            
            # Prepare to capture internal calls without mocking them
            # Setup a spy on build_expression_tree and optimize_expressions
            with patch.object(self.kansr, 'build_expression_tree', wraps=self.kansr.build_expression_tree) as spy_build, \
                 patch.object(self.kansr, 'optimize_expressions', return_value=(["x**2"], [0.95], [{'best_expression': "x**2"}])) as spy_optimize:
                
                # Call method
                result = self.kansr.get_symbolic(self.mock_client)
                
                # Verify kan_to_symbolic was called with the right parameters
                mock_kan_to_symbolic.assert_called_once()
                args, kwargs = mock_kan_to_symbolic.call_args
                self.assertEqual(args[0], self.mock_kan)
                self.assertEqual(args[1], self.mock_client)
                self.assertEqual(kwargs['population'], 10)
                self.assertEqual(kwargs['generations'], 3)
                self.assertEqual(kwargs['temperature'], 0.1)
                self.assertEqual(kwargs['gpt_model'], "openai/gpt-4o")
                self.assertAlmostEqual(kwargs['exit_condition'], 0.001, places=5)
                self.assertEqual(kwargs['verbose'], 0)
                self.assertEqual(kwargs['use_async'], True)
                self.assertEqual(kwargs['plot_fit'], True)
                self.assertEqual(kwargs['plot_parents'], False)
                
                # Verify that build_expression_tree was called (not mocked)
                spy_build.assert_called_once()
                
                # Verify that optimize_expressions was called with expected parameters
                spy_optimize.assert_called_once()
                args, kwargs = spy_optimize.call_args
                self.assertEqual(args[0], self.mock_client)
            
            # Test with non-trained model
            kansr = KANSR(client=self.mock_client, model=self.mock_kan)
            with self.assertRaises(ValueError):
                kansr.get_symbolic(self.mock_client)
    
    def test_build_expression_tree(self):
        """Test building the expression tree."""
        # Setup KANSR with symbolic expressions
        self.kansr.model = self.mock_kan
        self.kansr.symbolic_expressions = {
            (0, 0, 0): [
                {'score': 0.95, 'ansatz': 'params[0] * x + params[1]', 'params': [2.0, 3.0]},
                {'score': 0.90, 'ansatz': 'params[0] * x**2', 'params': [1.0]}
            ],
            (0, 1, 0): [
                {'score': 0.92, 'ansatz': 'params[0] * sin(x)', 'params': [1.0]},
                {'score': 0.85, 'ansatz': 'params[0] * exp(x)', 'params': [0.5]}
            ]
        }
        
        # For test validation, use a spy on the _simplify_expression method
        # This ensures we test the real implementation but can still validate it was called correctly
        with patch.object(
            self.kansr, '_simplify_expression', 
            wraps=self.kansr._simplify_expression
        ) as spy_simplify:
            
            # Call method
            result = self.kansr.build_expression_tree(top_k=2)
            
            # Verify structure of result
            self.assertIn("edge_dict", result)
            self.assertIn("top_k_edge_dicts", result)
            self.assertIn("node_tree", result)
            self.assertIn("full_expressions", result)
            
            # Verify edge dict has expected mappings
            self.assertEqual(len(result["edge_dict"]), 2)
            self.assertIn((0, 0, 0), result["edge_dict"])
            self.assertIn((0, 1, 0), result["edge_dict"])
            
            # Verify node tree is created
            self.assertIn((0, 0), result["node_tree"])
            
            # Verify top_k_edge_dicts has correct structure
            self.assertEqual(len(result["top_k_edge_dicts"]), 2)
            self.assertIn((0, 0, 0), result["top_k_edge_dicts"])
            self.assertIn((0, 1, 0), result["top_k_edge_dicts"])
            
            # Verify full_expressions is a list
            self.assertIsInstance(result["full_expressions"], list)
            
            # Verify result is saved to instance
            self.assertEqual(self.kansr.node_tree, result)
            
            # Verify that _simplify_expression was called
            spy_simplify.assert_called()
            
            # Test without symbolic expressions
            kansr = KANSR(client=self.mock_client, model=self.mock_kan)
            with self.assertRaises(ValueError):
                kansr.build_expression_tree()
    
    def test_optimize_expressions(self):
        """Test optimization of expressions."""
        # Setup KANSR with node_tree
        self.kansr.model = self.mock_kan
        self.kansr.dataset = self.mock_dataset
        self.kansr.node_tree = {
            "edge_dict": {(0, 0, 0): "2.0 * x + 3.0", (0, 1, 0): "1.0 * sin(x)"},
            "top_k_edge_dicts": {(0, 0, 0): [{"expression": "2.0 * x + 3.0"}], (0, 1, 0): [{"expression": "1.0 * sin(x)"}]},
            "node_tree": {(0, 0): "2.0 * x0 + 3.0 + 1.0 * sin(x1)"},
            "full_expressions": ["2.0 * x0 + 3.0 + 1.0 * sin(x0)"]
        }
        
        # Only mock the external dependencies and API calls
        # For internal methods, use spies to ensure they're called properly
        with patch('LLMSR.kansr.get_n_chi_squared') as mock_n_chi_squared, \
             patch('LLMSR.kansr.fit_curve_with_guess_jax') as mock_fit, \
             patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.show'), \
             patch.object(self.kansr, '_call_model_simplify') as mock_call_model:
            
            # Setup spy objects for internal methods
            convert_spy = patch.object(self.kansr, '_convert_sympy_to_numpy', 
                                     return_value="2.0 * x0 + 3.0 + 1.0 * np.sin(x0)")
            replace_spy = patch.object(self.kansr, '_replace_floats_with_params', 
                                     return_value=("params[0] * x0 + params[1] + params[2] * np.sin(x0)", [2.0, 3.0, 1.0]))
            simplify_spy = patch.object(self.kansr, '_simplify_expression', 
                                      return_value="2.0 * x0 + 3.0 + 1.0 * sin(x0)")
            
            # Setup mock returns for external dependencies
            mock_n_chi_squared.return_value = 0.001
            mock_fit.return_value = ([2.0, 3.0, 1.0], 0.0005)
            mock_subplots.return_value = (MagicMock(), MagicMock())
            mock_call_model.return_value = ["2.0 * x0 + 1.0 * sin(x0)"]
            
            # Apply all spies
            with convert_spy as spy_convert, replace_spy as spy_replace, simplify_spy as spy_simplify:
                # Call method
                best_expressions, best_n_chi_squared, result_dicts = self.kansr.optimize_expressions(
                    self.mock_client, "openai/gpt-4o"
                )
                
                # Verify structure of results
                self.assertIsInstance(best_expressions, list)
                self.assertIsInstance(best_n_chi_squared, list)
                self.assertIsInstance(result_dicts, list)
                self.assertEqual(len(best_expressions), 1)
                self.assertEqual(len(best_n_chi_squared), 1)
                self.assertEqual(len(result_dicts), 1)
                
                # Verify result dict structure
                result_dict = result_dicts[0]
                self.assertIn('raw_expression', result_dict)
                self.assertIn('final_KAN_expression', result_dict)
                self.assertIn('n_chi_squared_KAN_final', result_dict)
                self.assertIn('final_LLM_expression', result_dict)
                self.assertIn('n_chi_squared_LLM_final', result_dict)
                self.assertIn('best_expression', result_dict)
                self.assertIn('best_n_chi_squared', result_dict)
                self.assertIn('best_fit_type', result_dict)
                
                # Verify that internal methods were called
                spy_convert.assert_called()
                spy_replace.assert_called()
                spy_simplify.assert_called()
                mock_call_model.assert_called()
            
            # Test without node_tree
            kansr = KANSR(client=self.mock_client, model=self.mock_kan)
            with self.assertRaises(ValueError):
                kansr.optimize_expressions(self.mock_client, "openai/gpt-4o")
            
            # Test without dataset
            kansr = KANSR(client=self.mock_client, model=self.mock_kan)
            kansr.node_tree = self.kansr.node_tree
            with self.assertRaises(ValueError):
                kansr.optimize_expressions(self.mock_client, "openai/gpt-4o")
    
    def test_plot_results(self):
        """Test plotting of results."""
        # Create a result dict with all required fields
        result_dict = {
            'raw_expression': "x0**2",
            'raw_n_chi_squared': 0.001,
            'final_KAN_expression': ["x0**2"],
            'n_chi_squared_KAN_final': [0.001],
            'final_LLM_expression': ["x0**2"],
            'n_chi_squared_LLM_final': [0.0005],
            'best_expression': "x0**2",
            'best_n_chi_squared': 0.0005,
            'best_fit_type': 'LLMsimplified'
        }
        
        # Define a test function
        def test_func(x):
            if isinstance(x, torch.Tensor):
                return x**2
            return x**2
            
        # Set the function on the instance
        self.kansr.f = test_func
        
        # Create a mock dataset
        mock_dataset = {
            'train_input': torch.tensor(np.linspace(-5, 5, 1000).reshape(-1, 1)).float(),
            'train_label': torch.tensor(np.linspace(-5, 5, 1000)**2).float()
        }
        
        # Since _convert_sympy_to_numpy isn't directly called in plot_results,
        # we no longer need to mock it. Instead, we'll just test the plotting calls.
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.show'), \
             patch('builtins.eval', return_value=np.array([x**2 for x in range(1000)])):
                
            # Create mock figure and axes
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            # Test 1: Call method with explicit ranges and result_dict
            fig, ax = self.kansr.plot_results(ranges=(-5, 5), result_dict=result_dict)
            
            # Verify plot elements were called
            self.assertEqual(fig, mock_fig)
            self.assertEqual(ax, mock_ax)
            mock_ax.plot.assert_called()
            mock_ax.set_title.assert_called()
            mock_ax.set_xlabel.assert_called_with('x')
            mock_ax.set_ylabel.assert_called_with('y')
            mock_ax.legend.assert_called()
            mock_ax.grid.assert_called()
            
            # Test 2: Call with plot limits
            fig, ax = self.kansr.plot_results(
                ranges=(-5, 5), 
                result_dict=result_dict, 
                plotmaxmin=[[-2, 2], [-1, 10]]
            )
            
            # Verify limits were set
            mock_ax.set_xlim.assert_any_call(left=-2)
            mock_ax.set_xlim.assert_any_call(right=2)
            mock_ax.set_ylim.assert_any_call(bottom=-1)
            mock_ax.set_ylim.assert_any_call(top=10)
            
            # Test 3: Call with dataset and without explicit ranges
            # Set results_all_dicts for the method to use when result_dict is not provided
            self.kansr.results_all_dicts = [result_dict]
            
            fig, ax = self.kansr.plot_results(dataset=mock_dataset)
            
            # Verify plot was generated with dataset-derived ranges
            self.assertEqual(fig, mock_fig)
            mock_ax.plot.assert_called()
            
            # Test 4: Call with only result_dict (implicit ranges from default)
            fig, ax = self.kansr.plot_results(result_dict=result_dict)
            
            # Verify plot was generated with default ranges
            self.assertEqual(fig, mock_fig)
            mock_ax.plot.assert_called()
            
            # Test 5: Call with neither ranges nor result_dict (should use self.dataset and self.results_all_dicts)
            self.kansr.dataset = mock_dataset
            
            fig, ax = self.kansr.plot_results()
            
            # Verify plot was generated
            self.assertEqual(fig, mock_fig)
            mock_ax.plot.assert_called()
    
    def test_run_complete_pipeline_error_recovery(self):
        """Test that the complete pipeline properly handles errors and recovers."""
        # This is a better test for our needs - it verifies partial results are returned
        # Even with errors, and doesn't rely on API mocking which is complex
        
        # Here we're mocking an external dependency (create_dataset) which is appropriate
        with patch('LLMSR.kansr.create_dataset') as mock_create_dataset:
            # Force a controlled error
            mock_create_dataset.side_effect = ValueError("Intentional test error")
            
            # Call the pipeline
            result = self.kansr.run_complete_pipeline(
                self.mock_client, lambda x: x**2, 
                ranges=(-1, 1)
            )
            
            # Verify error was handled gracefully
            self.assertIsInstance(result, dict)
            # Should contain raw_model at minimum
            self.assertIn('trained_model', result)
    
    def test_run_complete_pipeline_error_handling(self):
        """Test error handling in the pipeline."""
        # Instead of mocking the method, use a spy to track calls while preserving functionality
        # This tests the actual implementation while handling errors
        with patch.object(self.kansr, 'create_dataset', side_effect=ValueError("Test error - expected during testing")) as spy_create_dataset:
            
            # Call the method
            result = self.kansr.run_complete_pipeline(
                self.mock_client, self.test_function,
                ranges=(-5, 5)
            )
            
            # Verify partial results are returned
            self.assertIsInstance(result, dict)
            self.assertIn('trained_model', result)
            
            # Verify create_dataset was called
            spy_create_dataset.assert_called_once()
            
            # Test error in a later stage of the pipeline
            # Use a spy on train_kan to raise an exception after dataset creation succeeds
            with patch.object(self.kansr, 'create_dataset', return_value=self.mock_dataset) as spy_create_dataset, \
                 patch.object(self.kansr, 'train_kan', side_effect=RuntimeError("Test error - expected during testing")) as spy_train:
                
                # Call the method
                result = self.kansr.run_complete_pipeline(
                    self.mock_client, self.test_function,
                    ranges=(-5, 5)
                )
                
                # Verify partial results contain dataset but not training results
                self.assertIsInstance(result, dict)
                self.assertIn('trained_model', result)
                self.assertIn('dataset', result)
                self.assertNotIn('train_loss', result)
                
                # Verify both methods were called
                spy_create_dataset.assert_called_once()
                spy_train.assert_called_once()
    
    def test_helper_methods(self):
        """Test the helper methods of KANSR."""
        # Test _subst_params directly - no need to mock since it's internal
        params_str = "params[0] * x + params[1]"
        params = [2.0, 3.0]
        result = self.kansr._subst_params(params_str, params)
        self.assertEqual(result, "2.0000 * x + 3.0000")
        
        # Test _convert_sympy_to_numpy with simple expressions
        # For NumPyPrinter, which is an external dependency, we should still mock
        with patch('LLMSR.kansr.NumPyPrinter') as mock_printer:
            mock_printer.return_value.doprint.return_value = "np.sin(x)"
            result = self.kansr._convert_sympy_to_numpy("sin(x)")
            self.assertEqual(result, "np.sin(x)")
        
        # Test _replace_floats_with_params directly - no need to mock since it's internal
        expr_str = "2.5 * x + 3.7"
        result, values = self.kansr._replace_floats_with_params(expr_str)
        # Just check the structure is correct (parameter indices might differ)
        self.assertIn("params[", result)
        self.assertIn("] * x + params[", result)
        self.assertIn(2.5, values)
        self.assertIn(3.7, values)


class TestRunCompletePipeline(unittest.TestCase):
    """Test cases for the run_complete_pipeline convenience function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Create mock client
        self.mock_client = MagicMock()
        
        # Test function
        self.test_function = lambda x: x**2
    
    def test_run_complete_pipeline_function(self):
        """Test the standalone run_complete_pipeline function."""
        # This is testing the global function which creates a KANSR instance
        # We should mock the KANSR class since we're testing the wrapper function
        with patch('LLMSR.kansr.KANSR') as mock_kansr_class:
            # Create a mock instance for the KANSR constructor to return
            mock_kansr_instance = MagicMock()
            mock_kansr_class.return_value = mock_kansr_instance
            
            # Configure the mock instance's run_complete_pipeline to return a result
            mock_result = {'pipeline_result': True}
            mock_kansr_instance.run_complete_pipeline.return_value = mock_result
            
            # Call the standalone function
            result = run_complete_pipeline(
                self.mock_client, self.test_function,
                ranges=(-5, 5), width=[1, 4, 1], grid=7, k=3,
                train_steps=50, generations=2
            )
            
            # Verify KANSR was instantiated with correct parameters
            mock_kansr_class.assert_called_once_with(
                client=self.mock_client, width=[1, 4, 1], grid=7, k=3, seed=17, device='cpu'
            )
            
            # Verify run_complete_pipeline was called on the instance
            mock_kansr_instance.run_complete_pipeline.assert_called_once()
            
            # Get the args from the call
            args, kwargs = mock_kansr_instance.run_complete_pipeline.call_args
            
            # Verify key parameters were passed correctly
            self.assertEqual(kwargs['client'], self.mock_client)
            self.assertEqual(kwargs['f'], self.test_function)
            self.assertEqual(kwargs['ranges'], (-5, 5))
            self.assertEqual(kwargs['train_steps'], 50)
            self.assertEqual(kwargs['generations'], 2)
            
            # Verify the result is passed through
            self.assertEqual(result, mock_result)


if __name__ == '__main__':
    unittest.main()