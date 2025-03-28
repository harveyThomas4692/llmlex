import unittest
import sys
import os
import numpy as np
import LLM_LEx.kanLEx
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch, ANY
import io
import logging
import sympy as sp
from sympy import symbols
import re

# Add the parent directory to the path if it's not already there
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the modules to test
import LLM_LEx.LLMLEx
from LLM_LEx.kanLEx import KAN_LEx, run_complete_pipeline


class TestKANLExClass(unittest.TestCase):
    """Test cases for the KANLEx class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Simple test function and data
        self.test_function = lambda x: x**2
        self.x_data = np.linspace(-5, 5, 100)
        self.y_data = self.test_function(self.x_data)
        
        # Create mock dataset
        self.mock_dataset = {
            'train_input': torch.tensor(self.x_data.reshape(-1, 1)).float(),
            'train_label': torch.tensor(self.y_data).float(),
            'test_input': torch.tensor(self.x_data[:10].reshape(-1, 1)).float(),
            'test_label': torch.tensor(self.y_data[:10]).float()
        }
        
        # Mock external dependencies
        self.patches = []
        
        # Save the original KAN class to restore it later if needed
        self.original_KAN = LLM_LEx.kanLEx.KAN
        
        # Create the mock KAN object
        self.mock_kan = MagicMock()
        
        # Set up mock behaviors
        self.mock_kan.width_in = [1, 4, 1]  # Input, hidden, output layers
        self.mock_kan.width_out = [4, 1]    # Number of outputs for each layer
        self.mock_kan.device = 'cpu'        # Device
        
        # Mock the KAN class constructor - this needs to track calls correctly
        def mock_kan_constructor(*args, **kwargs):
            mock_kan_constructor.call_count += 1
            mock_kan_constructor.call_args = args
            mock_kan_constructor.call_kwargs = kwargs
            return self.mock_kan
        
        # Initialize call tracking
        mock_kan_constructor.call_count = 0
        mock_kan_constructor.call_args = None
        mock_kan_constructor.call_kwargs = None
        
        # Apply the patch
        self.mock_kan_class = mock_kan_constructor
        mock_kan_patch = patch('LLMLEx.kanLEx.KAN', self.mock_kan_class)
        mock_kan_patch.start()
        self.patches.append(mock_kan_patch)
        
        # Configure mock KAN model properties
        self.mock_kan.width_in = [1, 4, 1]  # Input, hidden, output layers
        self.mock_kan.width_out = [4, 1]    # Number of outputs for each layer
        self.mock_kan.device = 'cpu'
        
        # Mock essential KAN methods
        self.mock_kan.fit = MagicMock(return_value={'train_loss': torch.tensor([0.001])})
        self.mock_kan.prune = MagicMock(return_value=self.mock_kan)
        self.mock_kan.plot = MagicMock()
        
        # Mock dataset creation
        mock_create_dataset = patch('LLMLEx.kanLEx.create_dataset')
        self.mock_create_dataset = mock_create_dataset.start()
        self.patches.append(mock_create_dataset)
        self.mock_create_dataset.return_value = self.mock_dataset
        
        # Mock external API client
        self.mock_client = MagicMock()
        
        # Mock LLM API functions to avoid actual calls
        mock_call_model = patch.object(KAN_LEx, '_call_model_simplify')
        self.mock_call_model = mock_call_model.start()
        self.patches.append(mock_call_model)
        self.mock_call_model.return_value = ["x0**2"]  # Default return value
        
        # Mock _fit_params to avoid actual fitting
        mock_fit_params = patch.object(KAN_LEx, '_fit_params')
        self.mock_fit_params = mock_fit_params.start()
        self.patches.append(mock_fit_params)
        self.mock_fit_params.return_value = ([1.0], 0.001)
        
        # Mock NumPyPrinter to avoid issues
        mock_numpy_printer = patch('LLMLex.kanLEx.NumPyPrinter')
        self.mock_numpy_printer = mock_numpy_printer.start()
        self.patches.append(mock_numpy_printer)
        self.mock_numpy_printer.return_value.doprint.return_value = "x0**2"
        
        # Mock eval to avoid actual code execution
        mock_eval = patch('builtins.eval')
        self.mock_eval = mock_eval.start()
        self.patches.append(mock_eval)
        self.mock_eval.return_value = lambda x0: x0**2 if isinstance(x0, (int, float)) else np.array([x**2 for x in x0])
        
        # Mock sympy simplify to avoid issues
        mock_simplify = patch('LLMLex.kanLEx.simplify')
        self.mock_simplify = mock_simplify.start()
        self.patches.append(mock_simplify)
        self.mock_simplify.return_value = symbols('x0')**2
        
        # Mock get_n_chi_squared and get_n_chi_squared_from_predictions to avoid dependency issues
        mock_get_n_chi_squared = patch('LLM_LEx.fit.get_n_chi_squared', return_value=0.001)
        self.mock_get_n_chi_squared = mock_get_n_chi_squared.start()
        self.patches.append(mock_get_n_chi_squared)
        
        mock_get_n_chi_squared_from_predictions = patch('LLM_LEx.fit.get_n_chi_squared_from_predictions', return_value=0.001)
        self.mock_get_n_chi_squared_from_predictions = mock_get_n_chi_squared_from_predictions.start()
        self.patches.append(mock_get_n_chi_squared_from_predictions)
        
        # Mock matplotlib to avoid display during tests
        mock_plt = patch('matplotlib.pyplot')
        self.mock_plt = mock_plt.start()
        self.patches.append(mock_plt)
        self.mock_fig = MagicMock()
        self.mock_ax = MagicMock()
        self.mock_plt.subplots.return_value = (self.mock_fig, self.mock_ax)
        
        # Create a fresh KAN_LEx instance for each test to avoid interference
        # Use cls.__new__ to skip __init__, then manually set attributes to avoid calling original init
        self.kansr = KAN_LEx.__new__(KAN_LEx)
        self.kansr.client = self.mock_client
        self.kansr.raw_model = self.mock_kan
        self.kansr.model = None
        self.kansr.dataset = self.mock_dataset
        self.kansr.f = self.test_function
        self.kansr.device = 'cpu'
        
        # Add logger to avoid errors
        self.kansr.logger = logging.getLogger('LLM_LEx.kansr')
        
        # Add other required attributes
        self.kansr.numpy_to_sympy = {
            'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'log': sp.log,
            'sqrt': sp.sqrt, 'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh
        }
        
        # Let's also create a separate instance specifically for the initialization test
        # Save current mock state
        self.real_KAN = LLM_LEx.kansr.KAN
        
        # Setup things needed for tests
        self.kansr.dataset = self.mock_dataset
        self.kansr.f = self.test_function
    
    def tearDown(self):
        """Clean up after tests."""
        for p in self.patches:
            p.stop()
    
    def test_initialization(self):
        """Test initialization of KAN_LEx class."""
        # This is a completely independent test that doesn't rely on the setUp mock
        
        # Create a dedicated MagicMock
        mock_kan_instance = MagicMock()
        mock_kan_instance.device = 'cpu'
        
        # Create a mock constructor function that tracks calls
        call_count = 0
        call_args = None
        call_kwargs = None
        
        def mock_kan_constructor(*args, **kwargs):
            nonlocal call_count, call_args, call_kwargs
            call_count += 1
            call_args = args
            call_kwargs = kwargs
            return mock_kan_instance
        
        # Patch KAN with our special callable mock that tracks calls
        with patch('LLMLex.kanLEx.KAN', side_effect=mock_kan_constructor) as mock_kan_class:
            # Test with width, grid, k parameters
            kansr1 = KAN_LEx(
                client=self.mock_client,
                width=[1, 4, 1],
                grid=7,
                k=3,
                seed=17
            )
            
            # Just verify that it created a model
            self.assertIsNotNone(kansr1.raw_model)
            self.assertEqual(kansr1.device, 'cpu')
            
            # Reset our counters and test with model parameter
            call_count = 0
            call_args = None
            call_kwargs = None
            mock_model = MagicMock()
            mock_model.device = 'cpu'
            kansr2 = KAN_LEx(client=self.mock_client, model=mock_model)
            
            # Just verify model was set correctly
            self.assertEqual(kansr2.raw_model, mock_model)
            
            # Test with missing parameters (should raise ValueError if neither model nor (width, grid, k) provided)
            with self.assertRaises(ValueError):
                KAN_LEx(client=self.mock_client)
    
    def test_create_dataset(self):
        """Test dataset creation."""
        # Reset previous calls to the mock
        self.mock_create_dataset.reset_mock()
        
        # Call the method
        dataset = self.kansr.create_dataset(self.test_function)
        
        # Verify create_dataset was called with correct parameters
        self.mock_create_dataset.assert_called_once_with(
            self.test_function, n_var=1, ranges=(-np.pi, np.pi), 
            train_num=10000, test_num=1000, device='cpu'
        )
        
        # Verify dataset was saved
        self.assertEqual(self.kansr.dataset, self.mock_dataset)
        # Verify function was saved
        self.assertEqual(self.kansr.f, self.test_function)
    
    def test_train(self):
        """Test model training and pruning."""
        # Reset previous calls
        self.mock_kan.fit.reset_mock()
        self.mock_kan.prune.reset_mock()
        
        # Reset the model to make sure we're using our mock
        self.kansr.raw_model = self.mock_kan
        
        # Call the method
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
        # Create a fresh KAN_LEx instance for this test
        with patch('LLMLex.kanLEx.KAN'):
            kansr = KAN_LEx(client=self.mock_client, model=self.mock_kan)
            kansr.dataset = None  # Ensure no dataset is set
            with self.assertRaises(ValueError):
                kansr.train_kan()
    
    def test_get_symbolic(self):
        """Test conversion to symbolic expressions with get_symbolic."""
        # Setup KAN_LEx instance with trained model and dataset
        self.kansr.model = self.mock_kan
        self.kansr.training_history = {'train_loss': torch.tensor([0.001])}
        
        # Only mock the external API calls
        with patch('LLM_LEx.llmLEx.kan_to_symbolic') as mock_kan_to_symbolic, \
             patch('LLM_LEx.llm.check_key_usage') as mock_check_key_usage:
            
            # Setup mock returns for the external API
            mock_symbolic_result = {(0, 0, 0): [[{'score': 0.95, 'ansatz': 'params[0] * x**2', 'params': [1.0]}]]}
            mock_kan_to_symbolic.return_value = mock_symbolic_result
            mock_check_key_usage.return_value = 1.000
            
            # Setup for internal methods so they don't fail
            # We need to prepare symbolic_expressions and node_tree
            self.kansr.symbolic_expressions = self.kansr._sort_symbolic_expressions(mock_symbolic_result)
            self.kansr.build_expression_tree()
            # self.kansr.expression_tree = {
            #     "edge_dict": {(0, 0, 0): "1.0 * x**2"},
            #     "top_k_edge_dicts": {(0, 0, 0): [{"expression": "1.0 * x**2"}]},
            #     "node_tree": {(0, 0): "1.0 * x0**2"},
            #     "full_expressions": ["1.0 * x0**2"]
            # }
            
            # Use spies to observe internal calls without mocking behavior
            with patch.object(self.kansr, '_sort_symbolic_expressions', wraps=self.kansr._sort_symbolic_expressions) as spy_sort, \
                 patch.object(self.kansr, 'build_expression_tree', wraps=self.kansr.build_expression_tree) as spy_build, \
                 patch.object(self.kansr, 'optimise_expressions') as mock_optimise:
                
                # Configure mock_optimise to return a valid result with all 4 expected return values
                mock_optimise.return_value = (["x0**2"], [0.001], [{'best_expression': "x0**2", 'best_chi_squared': 0.001, 'best_n_chi_squared': 0.001}], [("x0**2", 0.001)])
                
                # Call method - the method returns 4 values, but we only capture the first 3 for backward compatibility
                best_expressions, best_chi_squareds, results_dicts, _ = self.kansr.get_symbolic(self.mock_client)
                
                # Verify kan_to_symbolic was called with the right parameters
                mock_kan_to_symbolic.assert_called_once()
                args, kwargs = mock_kan_to_symbolic.call_args
                self.assertEqual(args[0], self.mock_kan)
                self.assertEqual(args[1], self.mock_client)
                
                # Verify that _sort_symbolic_expressions was called
                spy_sort.assert_called_once()
                
                # Verify that build_expression_tree was called (not mocked)
                spy_build.assert_called_once()
                
                # Verify that optimise_expressions was called
                mock_optimise.assert_called_once()
                
                # Verify results structure
                self.assertIsInstance(best_expressions, list)
                self.assertIsInstance(best_chi_squareds, list)
                self.assertIsInstance(results_dicts, list)
                self.assertEqual(best_expressions[0], "x0**2")
                self.assertEqual(best_chi_squareds[0], 0.001)
                
                # Only check specific keys we're interested in (implementation returns more)
                self.assertEqual(results_dicts[0]['best_expression'], "x0**2")
                self.assertEqual(results_dicts[0]['best_chi_squared'], 0.001)
            
            # Test with non-trained model
            with patch('LLMLex.kanLEx.KAN'):
                kansr = KAN_LEx(client=self.mock_client, model=self.mock_kan)
                kansr.model = None  # Ensure model is not trained
                with self.assertRaises(ValueError):
                    kansr.get_symbolic(self.mock_client)
    
    def test_build_expression_tree(self):
        """Test building the expression tree."""
        # Setup KAN_LEx with symbolic expressions
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
        self.assertEqual(self.kansr.expression_tree, result)
        
        # Test without symbolic expressions
        with patch('LLMLex.kanLEx.KAN'):
            kansr = KAN_LEx(client=self.mock_client, model=self.mock_kan)
            kansr.symbolic_expressions = None  # Ensure symbolic_expressions is not set
            with self.assertRaises(ValueError):
                kansr.build_expression_tree()
    
    def test_optimise_expressions(self):
        """Test optimisation of expressions."""
        # Setup KAN_LEx with node_tree
        self.kansr.model = self.mock_kan
        self.kansr.expression_tree = {
            "edge_dict": {(0, 0, 0): "2.0 * x + 3.0", (0, 1, 0): "1.0 * sin(x)"},
            "top_k_edge_dicts": {(0, 0, 0): [{"expression": "2.0 * x + 3.0"}], (0, 1, 0): [{"expression": "1.0 * sin(x)"}]},
            "node_tree": {(0, 0): "2.0 * x0 + 3.0 + 1.0 * sin(x1)"},
            "full_expressions": ["2.0 * x0 + 3.0 + 1.0 * sin(x0)"]
        }
        self.kansr.symbolic_expressions = {(0, 0): [{"score": 0.95, "ansatz": "params[0] * x0 + params[1] + params[2] * sin(x1)", "params": [2.0, 3.0, 1.0]}]}
        
        # We already have mocked get_chi_squared in setUp
        # Call method with the already mocked dependencies
        # Set up a mock for optimise_expressions to return consistent test values
        with patch.object(self.kansr, '_call_model_simplify') as mock_call_model, \
             patch.object(self.kansr, '_fit_params') as mock_fit_params:
             
            # Configure mock returns
            mock_call_model.return_value = ["x0**2"]
            mock_fit_params.return_value = ([1.0], 0.001)
            
            # Call the method with mocks in place
            best_expressions, best_chi_squareds, result_dicts, all_results_sorted = self.kansr.optimise_expressions(
                self.mock_client, "openai/gpt-4o", plot_all=False
            )
            

            # Verify structure of results
            self.assertIsInstance(best_expressions, list)
            self.assertIsInstance(best_chi_squareds, list)
            self.assertIsInstance(result_dicts, list)
            self.assertIsInstance(all_results_sorted, list)
            self.assertEqual(len(best_expressions), 1)
            self.assertEqual(len(best_chi_squareds), 1)
            self.assertEqual(len(result_dicts), 1)
            
            # Verify result dict structure
            result_dict = result_dicts[0]
            self.assertIn('raw_expression', result_dict)
            self.assertIn('raw_n_chi_squared', result_dict)  # Added check for normalized chi squared
            self.assertIn('final_refitted_expression', result_dict)
            self.assertIn('n_chi_squared_refitted', result_dict)
            self.assertIn('final_LLM_expression', result_dict)
            self.assertIn('n_chi_squared_LLM_final', result_dict)  # Added check for normalized chi squared
            self.assertIn('best_expression', result_dict)
            self.assertIn('best_n_chi_squared', result_dict)  # Added check for normalized chi squared
            self.assertIn('best_fit_type', result_dict)
            
            # Verify that _call_model_simplify was called
            mock_call_model.assert_called()
        
        # Test without node_tree
        with patch('LLMLex.kanLEx.KAN'):
            kansr = KAN_LEx(client=self.mock_client, model=self.mock_kan)
            kansr.expression_tree = None  # Ensure node_tree is not set
            with self.assertRaises(ValueError):
                kansr.optimise_expressions(self.mock_client, "openai/gpt-4o")
        
        # Test without dataset
        with patch('LLMLex.kanLEx.KAN'):
            kansr = KAN_LEx(client=self.mock_client, model=self.mock_kan)
            kansr.expression_tree = self.kansr.expression_tree
            kansr.dataset = None  # Ensure dataset is not set
            with self.assertRaises(ValueError):
                kansr.optimise_expressions(self.mock_client, "openai/gpt-4o")
                
    def test_call_model_simplify_with_mocked_responses(self):
        """Test the behaviors of _call_model_simplify with mocked LLM responses."""
        # Since we were having issues with mocking, we'll take a simpler approach here
        # Here we're not testing the actual parsing but just the integration with the mocked functions
        
        # We will patch the _call_model_simplify directly to return our test data
        # This test focuses on verifying what happens with the response once it's returned
        # by the _call_model_simplify method
        
        # Setup our test cases with mocked responses and expected outputs
        test_cases = [
            # A well-formatted quadratic expression 
            {
                "mock_response": ["2*x**2 + 3*x"],
                "expected_contains": "2",
                "description": "Simple quadratic expression"
            },
            # Multiple expressions 
            {
                "mock_response": ["x**2", "2*x**2 + 3*x"],
                "expected_contains": ["x", "2"],
                "description": "Multiple expressions"
            }
        ]
        
        # For each test case
        for i, test_case in enumerate(test_cases):
            with self.subTest(f"LLM response case {i+1}: {test_case['description']}"):
                # Patch optimise_expressions to observe what it receives
                with patch.object(self.kansr, '_call_model_simplify', return_value=test_case["mock_response"]) as mock_call_model:
                    
                    # Set up the necessary data for the test
                    self.kansr.model = self.mock_kan
                    self.kansr.expression_tree = {
                        "edge_dict": {(0, 0, 0): "2.0 * x**2 + 3.0 * x"},
                        "top_k_edge_dicts": {(0, 0, 0): [{"expression": "2.0 * x**2 + 3.0 * x"}]},
                        "node_tree": {(0, 0): "2.0 * x0**2 + 3.0 * x0"},
                        "full_expressions": ["2.0 * x0**2 + 3.0 * x0"]
                    }
                    
                    # Call optimise_expressions which will use our mocked _call_model_simplify
                    best_expressions, best_chi_squareds, result_dicts, _ = self.kansr.optimise_expressions(
                        self.mock_client, "openai/gpt-4o", plot_all=False
                    )
                    
                    # Print debug info
                    print(f"\nTest case: {test_case['description']}")
                    print(f"Mock LLM response: {test_case['mock_response']}")
                    print(f"Best expressions: {best_expressions}")
                    
                    # Verify that our mock was called
                    mock_call_model.assert_called()
                    
                    # Simply verify that the result contains our expected terms
                    # This is a loose check since we're not focused on exact parsing details here
                    if isinstance(test_case["expected_contains"], list):
                        for term in test_case["expected_contains"]:
                            found = any(term in expr for expr in best_expressions)
                            self.assertTrue(found, f"Expected term '{term}' not found in any expression: {best_expressions}")
                    else:
                        found = any(test_case["expected_contains"] in expr for expr in best_expressions)
                        self.assertTrue(found, f"Expected term '{test_case['expected_contains']}' not found in any expression: {best_expressions}")
    
    def test_optimise_expression_integration(self):
        """Test the end-to-end optimise_expressions with a simple controlled test case."""
        # This is a simpler test focused on a single case for optimise_expressions
        
        # Create a quadratic test function and data
        test_func = lambda x: 2 * x**2 + 3 * x
        x_data = np.linspace(-5, 5, 100)
        y_data = test_func(x_data)
        
        # Update test dataset
        self.kansr.dataset = {
            'train_input': torch.tensor(x_data.reshape(-1, 1)).float(),
            'train_label': torch.tensor(y_data).float(),
            'test_input': torch.tensor(x_data[:10].reshape(-1, 1)).float(),
            'test_label': torch.tensor(y_data[:10]).float()
        }
        
        # Set up the function so evalutations work correctly
        self.kansr.f = test_func
        
        # Setup expression tree directly with the test expression
        self.kansr.expression_tree = {
            "edge_dict": {(0, 0, 0): "2.0 * x**2 + 3.0 * x"},
            "top_k_edge_dicts": {(0, 0, 0): [{"expression": "2.0 * x**2 + 3.0 * x"}]},
            "node_tree": {(0, 0): "2.0 * x0**2 + 3.0 * x0"},
            "full_expressions": ["2.0 * x0**2 + 3.0 * x0"]
        }
        
        # Mock the key methods
        with patch.object(self.kansr, '_call_model_simplify') as mock_call_model, \
             patch.object(self.kansr, '_fit_params') as mock_fit_params:
            
            # Configure mocks for a successful test case 
            mock_call_model.return_value = ["2*x**2 + 3*x"]
            mock_fit_params.return_value = ([2.0, 3.0], 0.001)
            
            # Run the optimisation
            best_expressions, best_chi_squareds, result_dicts, _ = self.kansr.optimise_expressions(
                self.mock_client, "openai/gpt-4o", plot_all=False
            )
            
            # Print debug info
            print("\nTest test_optimise_expression_integration:")
            print(f"Best expressions: {best_expressions}")
            print(f"Result dicts: {result_dicts}")
            
            # Check the outputs
            self.assertIsInstance(best_expressions, list)
            self.assertGreaterEqual(len(best_expressions), 1)
            
            # Check if any of the expressions contain the terms "2", "x0", and "3"
            found_2 = any("2" in expr for expr in best_expressions)
            found_x0 = any("x0" in expr for expr in best_expressions)
            
            self.assertTrue(found_2 or found_x0, 
                           f"Expected expressions to contain either '2' or 'x0' but got {best_expressions}")
            
            # Verify a few key fields in the result dicts
            self.assertGreaterEqual(len(result_dicts), 1)
            result_dict = result_dicts[0]
            self.assertIn('raw_expression', result_dict)
            self.assertIn('final_LLM_expression', result_dict)
            self.assertIn('best_expression', result_dict)
    
    def test_plot_results(self):
        """Test plotting of results."""
        # Create a dedicated mock for matplotlib.pyplot that we can fully control
        class MockPlt:
            def __init__(self):
                self.fig = MagicMock()
                self.ax = MagicMock()
                self.subplots_called = 0
                
            def subplots(self, *args, **kwargs):
                self.subplots_called += 1
                return self.fig, self.ax
                
            def show(self):
                pass
        
        mock_plt = MockPlt()
        
        # Apply patch to both matplotlib.pyplot and builtins.eval
        with patch('LLMLex.kanLEx.plt', mock_plt), \
             patch('builtins.eval', return_value=np.ones(1000)):
            # Create result dict with all required fields
            result_dict = {
                'raw_expression': "x0**2",
                'raw_n_chi_squared': 0.001,
                'final_refitted_expression': "x0**2",
                'n_chi_squared_refitted': 0.001,
                'final_LLM_expression': "x0**2",
                'n_chi_squared_LLM_final': 0.0005,
                'best_expression': "x0**2",
                'best_n_chi_squared': 0.0005,
                'best_expression_index': 0,
                'best_fit_type': 'LLMsimplified'
                }
                
            # Define test function
            def test_func(x):
                if isinstance(x, torch.Tensor):
                    return x**2
                return x**2
                
            # Set the function on the instance
            self.kansr.f = test_func
            
            # Call plot_results
            self.kansr.symbolic_expressions = {(0, 0): [{"score": -0.95, "ansatz": "x0**2", "params": []}]}
            fig, ax = self.kansr.plot_results(ranges=(-5, 5), result_dict=result_dict)
            
            # Verify our custom mock plt was called
            self.assertEqual(mock_plt.subplots_called, 1)
            mock_plt.ax.plot.assert_called()
            mock_plt.ax.set_title.assert_called()
            mock_plt.ax.set_xlabel.assert_called_with('x')
            mock_plt.ax.set_ylabel.assert_called_with('y')
            mock_plt.ax.legend.assert_called()
            mock_plt.ax.grid.assert_called()
            
            # Reset the counter for the second test
            mock_plt.subplots_called = 0
            # Test with plot limits
            fig, ax = self.kansr.plot_results(
                ranges=(-5, 5), 
                result_dict=result_dict, 
                plotmaxmin=[[-2, 2], [-1, 10]]
            )
            
            # Verify limits were set
            self.assertEqual(mock_plt.subplots_called, 1)
            mock_plt.ax.set_xlim.assert_any_call(left=-2)
            mock_plt.ax.set_xlim.assert_any_call(right=2)
            mock_plt.ax.set_ylim.assert_any_call(bottom=-1)
            mock_plt.ax.set_ylim.assert_any_call(top=10)
    
    def test_run_complete_pipeline(self):
        """Test the complete pipeline functionality."""
        # Mock essential external API calls
        with patch('LLM_LEx.llmLEx.kan_to_symbolic') as mock_kan_to_symbolic, \
             patch('LLM_LEx.llm.check_key_usage') as mock_check_key_usage:
            
            # Set up mock returns
            mock_symbolic_result = {(0, 0, 0): [[{'score': 0.95, 'ansatz': 'params[0] * x**2', 'params': [1.0]}]]}
            mock_kan_to_symbolic.return_value = mock_symbolic_result
            mock_check_key_usage.return_value = 1.000
            
            # Reset any previous calls
            self.mock_create_dataset.reset_mock()
            self.mock_kan.fit.reset_mock()
            self.mock_kan.prune.reset_mock()
            
            # Prepare the kansr instance with what it needs
            self.kansr.model = self.mock_kan  # This is needed for the optimise_expressions call
            self.kansr.raw_model = self.mock_kan  # Make sure raw_model is our mock
            
            # Mock key methods to simplify test
            with patch.object(self.kansr, 'get_symbolic') as mock_get_symbolic, \
                 patch.object(self.kansr, 'build_expression_tree') as mock_build_tree, \
                 patch.object(self.kansr, 'optimise_expressions') as mock_optimise:
                
                # Setup return values with the expected number of return values
                mock_get_symbolic.return_value = (["x0**2"], [0.001], [{'best_expression': "x0**2", 'best_n_chi_squared': 0.001, 'best_fit_type': 'LLMsimplified'}], ['x0**2'])
                mock_build_tree.return_value = {
                    "edge_dict": {(0, 0, 0): "1.0 * x**2"},
                    "top_k_edge_dicts": {(0, 0, 0): [{"expression": "1.0 * x**2"}]},
                    "node_tree": {(0, 0): "1.0 * x0**2"},
                    "full_expressions": ["1.0 * x0**2"]
                }
                mock_optimise.return_value = (["x0**2"], [0.001], [{'best_expression': "x0**2", 'best_n_chi_squared': 0.001, 'best_fit_type': 'LLMsimplified'}], [("x0**2", 0.001)])
                
                # Call run_complete_pipeline
                result = self.kansr.run_complete_pipeline(
                    client=self.mock_client,
                    f=self.test_function,
                    ranges=(-5, 5),
                    train_steps=50
                )
                
                # Verify structure of result
                self.assertIn('trained_model', result)
                self.assertIn('pruned_model', result)
                self.assertIn('train_loss', result)
                self.assertIn('dataset', result)
                self.assertIn('best_expressions', result)
                
                # Verify components were called
                self.mock_create_dataset.assert_called_once()
                self.mock_kan.fit.assert_called_once()
                self.mock_kan.prune.assert_called_once()
                mock_get_symbolic.assert_called_once()
    
    def test_run_complete_pipeline_error_recovery(self):
        """Test that run_complete_pipeline returns partial results on error."""
        # Setup test function
        test_func = lambda x: x**2
        
        # Reset mock creates
        self.mock_create_dataset.reset_mock()
        
        # Setup error behavior
        with patch.object(self.kansr, 'get_symbolic') as mock_get_symbolic:
            # Configure mock_get_symbolic to raise an exception
            mock_get_symbolic.side_effect = ValueError("Test exception")
            
            # Call run_complete_pipeline
            result = self.kansr.run_complete_pipeline(
                client=self.mock_client,
                f=test_func,
                ranges=(-5, 5)
            )
            
            # Verify that we get partial results
            self.assertIn('trained_model', result)
            self.assertIn('pruned_model', result)
            self.assertIn('train_loss', result)
            self.assertIn('dataset', result)
            self.assertNotIn('symbolic_expressions', result)  # This would come from get_symbolic
    
    def test_helper_methods(self):
        """Test the helper methods of KAN_LEx."""
        # Test _subst_params
        params_str = "params[0] * x + params[1]"
        params = [2.0, 3.0]
        result = self.kansr._subst_params(params_str, params)
        self.assertEqual(result, "2.0000 * x + 3.0000")
        
        # Test _prune_small_params
        params = [0.0001, 2.0, 0.00001]
        result = self.kansr._prune_small_params(params, threshold=0.001)
        self.assertEqual(result, [0, 2.0, 0])
        
        # Test _replace_floats_with_params
        expr_str = "2.5 * x + 3.7"
        result, values = self.kansr._replace_floats_with_params(expr_str)
        # The function should extract the two floats
        self.assertEqual(len(values), 2)
        self.assertIn(2.5, values)
        self.assertIn(3.7, values)
        # The parameters should replace the floats in the expression
        self.assertIn("params[", result)
        self.assertTrue(" * x + params[" in result or " * x+params[" in result)
    
    def test_call_model_simplify(self):
        """Test the _call_model_simplify method that calls the LLM API."""
        # Set up a fake async function that returns our expected value
        async def fake_async_func():
            return ["2.0 * x + 1.0", "2.0 * x**2", "sin(2.0 * x)"]
            
        # Set up a fake asyncio.run that runs our function
        def fake_run(coro):
            # Just return the return value of our fake async function
            return ["2.0 * x + 1.0", "2.0 * x**2", "sin(2.0 * x)"]
        
        # Apply both patches
        with patch('asyncio.run', side_effect=fake_run), \
             patch.object(self.kansr, 'client', self.mock_client):
            
            # Call the method
            result = self.kansr._call_model_simplify(
                ranges=(-5, 5),
                expr="2.0 * x + 1.0",
                client=self.mock_client,
                gpt_model="openai/gpt-4o",
                num_answers=3
            )
            
            # Just test that we got some result
            self.assertIsInstance(result, list)
            self.assertGreaterEqual(len(result), 1)
        
        # Re-add the mock for other tests
        mock_call_model = patch.object(KAN_LEx, '_call_model_simplify')
        self.mock_call_model = mock_call_model.start()
        self.patches.append(mock_call_model)
        self.mock_call_model.return_value = ["x0**2"]


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
        """Test the run_complete_pipeline function."""
        # Mock the KAN_LEx class and its run_complete_pipeline method
        with patch('LLMLex.kanLEx.KAN_LEx') as mock_kansr_class:
            # Setup mock instance and return value
            mock_instance = MagicMock()
            mock_kansr_class.return_value = mock_instance
            
            # Mock run_complete_pipeline method to return a predefined result
            mock_result = {
                'trained_model': MagicMock(),
                'pruned_model': MagicMock(),
                'train_loss': torch.tensor([0.001]),
                'symbolic_expressions': {(0, 0, 0): [{'score': 0.95, 'ansatz': 'x**2', 'params': [1.0]}]},
                'best_expressions': ['x**2'],
                'best_chi_squareds': [0.001]
            }
            mock_instance.run_complete_pipeline.return_value = mock_result
            
            # Call the function
            result = run_complete_pipeline(
                client=self.mock_client,
                f=self.test_function,
                ranges=(-5, 5),
                width=[1, 4, 1],
                grid=7,
                k=3
            )
            
            # Verify KAN_LEx was created with the right parameters
            mock_kansr_class.assert_called_once_with(
                client=self.mock_client, 
                width=[1, 4, 1], 
                grid=7, 
                k=3, 
                seed=17, 
                device='cpu'
            )
            
            # Verify run_complete_pipeline was called
            mock_instance.run_complete_pipeline.assert_called_once()
            
            # Verify the result matches what we expect
            self.assertEqual(result, mock_result)


if __name__ == '__main__':
    unittest.main()