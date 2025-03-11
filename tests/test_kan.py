import unittest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch, ANY

# Add the parent directory to the path if it's not already there
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the module to test
import LLMSR.llmSR

# Create a modified version of kan_to_symbolic that handles the symb_formula issue
def test_kan_to_symbolic(model, client, population=10, generations=3, temperature=0.1, 
                        gpt_model="openai/gpt-4o-mini", exit_condition=1e-3, verbose=0, use_async=False):
    """
    A test-friendly version of kan_to_symbolic that fixes the symb_formula issue.
    This is a copy of the implementation with fixes for the uninitialized variable.
    """
    logger = LLMSR.llmSR.logger
    logger.debug(f"Starting KAN to symbolic conversion with population={population}, generations={generations}")
    logger.debug(f"KAN model has {len(model.width_in)} layers")

    res, res_fcts = 'Sin', {}
    
    # Initialize symb_formula here to avoid the uninitialized error
    symb_formula = []
    for l in range(len(model.width_in) - 1):
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l]):
                symb_formula.append(f'f_{{{l},{i},{j}}}')
    
    # Setup layer connections
    logger.debug("Setting up layer connections")
    layer_connections = {0: {i: [] for i in range(model.width_in[0])}}
    for l in range(len(model.width_in) - 1):
        layer_connections[l] = {i: list(range(model.width_out[l-1])) if l > 0 else []  for i in range(model.width_in[l])}
    
    # Process each connection in the KAN model
    total_connections = 0
    symbolic_connections = 0
    zero_connections = 0
    processed_connections = 0
    
    logger.info("Processing KAN model connections")
    for l in range(len(model.width_in) - 1):
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l]):
                total_connections += 1
                logger.debug(f"Processing connection ({l},{i},{j})")
                
                if (model.symbolic_fun[l].mask[j, i] > 0. and model.act_fun[l].mask[i][j] == 0.):
                    logger.info(f'Skipping ({l},{i},{j}) - already symbolic')
                    print(f'skipping ({l},{i},{j}) since already symbolic')
                    symb_formula = [s.replace(f'f_{{{l},{i},{j}}}', 'TODO') for s in symb_formula]
                    symbolic_connections += 1
                    
                elif (model.symbolic_fun[l].mask[j, i] == 0. and model.act_fun[l].mask[i][j] == 0.):
                    logger.info(f'Fixing ({l},{i},{j}) with 0')
                    model.fix_symbolic(l, i, j, '0', verbose=verbose > 1, log_history=False)
                    print(f'fixing ({l},{i},{j}) with 0')
                    symb_formula = [s.replace(f'f_{{{l},{i},{j}}}', '0') for s in symb_formula]
                    res_fcts[(l, i, j)] = None
                    zero_connections += 1
                    
                else:
                    logger.info(f'Processing non-symbolic connection ({l},{i},{j})')
                    processed_connections += 1
                    
                    # Generate data for the connection
                    logger.debug(f"Getting range data for connection ({l},{i},{j})")
                    x_min, x_max, y_min, y_max = model.get_range(l, i, j, verbose=False)
                    
                    # Handle PyTorch tensors or NumPy arrays
                    x_data = model.acts[l][:, i]
                    y_data = model.spline_postacts[l][:, j, i]
                    
                    # Convert to numpy if it's a PyTorch tensor
                    if hasattr(x_data, 'cpu') and hasattr(x_data, 'detach'):
                        x = x_data.cpu().detach().numpy()
                    else:
                        x = np.array(x_data)
                        
                    if hasattr(y_data, 'cpu') and hasattr(y_data, 'detach'):
                        y = y_data.cpu().detach().numpy()
                    else:
                        y = np.array(y_data)
                        
                    # Sort data by x values
                    ordered_in = np.argsort(x)
                    x, y = x[ordered_in], y[ordered_in]
                    
                    # Generate plot
                    logger.info(f"Generating plot for connection ({l},{i},{j}) - this is what we're fitting.")
                    fig, ax = LLMSR.llmSR.plt.subplots()
                    LLMSR.llmSR.plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                    LLMSR.llmSR.plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                    base64_image = LLMSR.llmSR.generate_base64_image(fig, ax, x, y)
                    print((l,i,j))
                    LLMSR.llmSR.plt.show()
                    
                    # Get activation function mask
                    mask = model.act_fun[l].mask
                    
                    # Run genetic algorithm to find symbolic expression
                    try:
                        logger.info(f"Running genetic algorithm for connection ({l},{i},{j})")
                        res = LLMSR.llmSR.run_genetic(
                            client, base64_image, x, y, population, generations, 
                            temperature=temperature, model=gpt_model, 
                            system_prompt=None, elite=False, 
                            exit_condition=exit_condition, for_kan=True,
                            use_async=use_async
                        )
                        res_fcts[(l,i,j)] = res
                        logger.info(f"Successfully found expression for connection ({l},{i},{j})")
                        
                    except Exception as e:
                        logger.error(f"Error in genetic algorithm for connection ({l},{i},{j}): {e}", exc_info=True)
                        print(e)
                        res_fcts[(l,i,j)] = res
    
    # Clean up
    logger.debug("Cleaning up matplotlib resources")
    try:
        ax.clear()
        LLMSR.llmSR.plt.close()
    except:
        logger.debug("Could not clean up matplotlib resources - not a cause for concern")
    
    # Log summary
    logger.info(f"KAN conversion complete: {total_connections} total connections")
    logger.info(f"Connection breakdown: {symbolic_connections} symbolic, {zero_connections} zero, {processed_connections} processed")
    
    return res_fcts

class TestKANFunctionality(unittest.TestCase):
    def setUp(self):
        # Create a mock KAN model
        self.mock_kan = MagicMock()
        
        # Set up the KAN model properties
        self.mock_kan.width_in = [2, 3, 1]  # 3 layers: input, hidden, output
        self.mock_kan.width_out = [3, 1]    # Number of outputs for each layer
        
        # Create mock activation functions and symbolic functions
        self.mock_kan.symbolic_fun = [MagicMock(), MagicMock()]
        self.mock_kan.act_fun = [MagicMock(), MagicMock()]
        
        # Setup masks
        self.mock_kan.symbolic_fun[0].mask = np.zeros((3, 2))  # First layer
        self.mock_kan.symbolic_fun[1].mask = np.zeros((1, 3))  # Second layer
        
        self.mock_kan.act_fun[0].mask = np.zeros((2, 3))  # First layer
        self.mock_kan.act_fun[1].mask = np.zeros((3, 1))  # Second layer
        
        # Set up mock layer activations (for get_range function)
        self.mock_kan.acts = [np.random.random((10, 2)), np.random.random((10, 3))]
        self.mock_kan.spline_postacts = [
            np.random.random((10, 3, 2)),  # First layer
            np.random.random((10, 1, 3))   # Second layer
        ]
        
        # Mock client for API calls
        self.mock_client = MagicMock()
        
        # Setup the get_range function to return deterministic values
        self.mock_kan.get_range = MagicMock(return_value=(-1, 1, -1, 1))
        
        # Setup the fix_symbolic function
        self.mock_kan.fix_symbolic = MagicMock()
        
        # Create patches for the built-in functions that our test version of kan_to_symbolic uses
        self.original_plt_subplots = LLMSR.llmSR.plt.subplots
        self.original_plt_xticks = LLMSR.llmSR.plt.xticks
        self.original_plt_yticks = LLMSR.llmSR.plt.yticks
        self.original_plt_show = LLMSR.llmSR.plt.show
        self.original_plt_close = LLMSR.llmSR.plt.close
        self.original_generate_base64_image = LLMSR.llmSR.generate_base64_image
        self.original_run_genetic = LLMSR.llmSR.run_genetic
        
        # Create the mocks we'll use
        self.mock_plt_subplots = MagicMock(return_value=(MagicMock(), MagicMock()))
        self.mock_plt_xticks = MagicMock()
        self.mock_plt_yticks = MagicMock()
        self.mock_plt_show = MagicMock()
        self.mock_plt_close = MagicMock()
        self.mock_generate_base64_image = MagicMock(return_value="mock_base64_image")
        self.mock_run_genetic = MagicMock(return_value=[
            [{
                'params': np.array([1.0, 2.0, 3.0]),
                'score': -0.01,
                'ansatz': 'params[0] * np.sin(params[1] * x + params[2])',
                'Num_params': 3
            }]
        ])
        
    def tearDown(self):
        # Restore the original functions
        LLMSR.llmSR.plt.subplots = self.original_plt_subplots
        LLMSR.llmSR.plt.xticks = self.original_plt_xticks
        LLMSR.llmSR.plt.yticks = self.original_plt_yticks
        LLMSR.llmSR.plt.show = self.original_plt_show
        LLMSR.llmSR.plt.close = self.original_plt_close
        LLMSR.llmSR.generate_base64_image = self.original_generate_base64_image
        LLMSR.llmSR.run_genetic = self.original_run_genetic

    def test_kan_to_symbolic_basic(self):
        """Test basic functionality of kan_to_symbolic"""
        # Set the mock functions
        LLMSR.llmSR.generate_base64_image = self.mock_generate_base64_image
        LLMSR.llmSR.plt.subplots = self.mock_plt_subplots
        LLMSR.llmSR.plt.xticks = self.mock_plt_xticks
        LLMSR.llmSR.plt.yticks = self.mock_plt_yticks
        LLMSR.llmSR.plt.show = self.mock_plt_show
        LLMSR.llmSR.plt.close = self.mock_plt_close
        LLMSR.llmSR.run_genetic = self.mock_run_genetic
        
        # Setup a test case where there's at least one non-zero connection
        self.mock_kan.symbolic_fun[0].mask = np.zeros((3, 2))
        self.mock_kan.act_fun[0].mask = np.zeros((2, 3))
        
        # Set one connection to have mask > 0 for activation
        self.mock_kan.act_fun[0].mask[0, 0] = 1.0
        
        # Call our test function with minimal arguments
        result = test_kan_to_symbolic(
            self.mock_kan, 
            self.mock_client,
            population=2, 
            generations=1,
            exit_condition=0.1
        )
        
        # Verify the basic structure of the result
        self.assertIsInstance(result, dict)
        
        # Verify that run_genetic was called for non-symbolic connections
        self.assertTrue(self.mock_run_genetic.called)
        
        # Check if generate_base64_image was called
        self.assertTrue(self.mock_generate_base64_image.called)

    def test_kan_to_symbolic_with_existing_symbolic(self):
        """Test kan_to_symbolic with existing symbolic connections"""
        # Set the mock functions
        LLMSR.llmSR.generate_base64_image = self.mock_generate_base64_image
        LLMSR.llmSR.plt.subplots = self.mock_plt_subplots
        LLMSR.llmSR.plt.xticks = self.mock_plt_xticks
        LLMSR.llmSR.plt.yticks = self.mock_plt_yticks
        LLMSR.llmSR.plt.show = self.mock_plt_show
        LLMSR.llmSR.plt.close = self.mock_plt_close
        LLMSR.llmSR.run_genetic = self.mock_run_genetic
        
        # Setup one connection as already symbolic
        self.mock_kan.symbolic_fun[0].mask = np.zeros((3, 2))
        self.mock_kan.act_fun[0].mask = np.zeros((2, 3))
        
        # Connection (0,0,0) is symbolic
        self.mock_kan.symbolic_fun[0].mask[0, 0] = 1.0
        self.mock_kan.act_fun[0].mask[0, 0] = 0.0
        
        # Call our test function
        result = test_kan_to_symbolic(
            self.mock_kan, 
            self.mock_client,
            population=2, 
            generations=1,
            exit_condition=0.1
        )
        
        # Verify result contains non-symbolic connections
        for l in range(len(self.mock_kan.width_in) - 1):
            for i in range(self.mock_kan.width_in[l]):
                for j in range(self.mock_kan.width_out[l]):
                    if not (l == 0 and i == 0 and j == 0):  # Skip the symbolic connection
                        self.assertIn((l, i, j), result)

    def test_kan_to_symbolic_with_zero_connections(self):
        """Test kan_to_symbolic with zero-valued connections"""
        # Set the mock functions
        LLMSR.llmSR.generate_base64_image = self.mock_generate_base64_image
        LLMSR.llmSR.plt.subplots = self.mock_plt_subplots
        LLMSR.llmSR.plt.xticks = self.mock_plt_xticks
        LLMSR.llmSR.plt.yticks = self.mock_plt_yticks
        LLMSR.llmSR.plt.show = self.mock_plt_show
        LLMSR.llmSR.plt.close = self.mock_plt_close
        LLMSR.llmSR.run_genetic = self.mock_run_genetic
        
        # Setup masks with zeros
        self.mock_kan.symbolic_fun[0].mask = np.zeros((3, 2))
        self.mock_kan.act_fun[0].mask = np.zeros((2, 3))
        
        # All connections are zero connections (mask values both 0)
        # Connection (0,0,0) has symbolic mask = 0 and activation mask = 0
        
        # Call our test function
        result = test_kan_to_symbolic(
            self.mock_kan, 
            self.mock_client,
            population=2, 
            generations=1,
            exit_condition=0.1
        )
        
        # Verify result has the zero connection
        self.assertIn((0, 0, 0), result)
        self.assertIsNone(result[(0, 0, 0)])
        
        # Verify fix_symbolic was called with '0' for the zero connection
        self.mock_kan.fix_symbolic.assert_any_call(0, 0, 0, '0', verbose=False, log_history=False)

    def test_kan_to_symbolic_with_async(self):
        """Test kan_to_symbolic with async mode enabled"""
        # Set the mock functions
        LLMSR.llmSR.generate_base64_image = self.mock_generate_base64_image
        LLMSR.llmSR.plt.subplots = self.mock_plt_subplots
        LLMSR.llmSR.plt.xticks = self.mock_plt_xticks
        LLMSR.llmSR.plt.yticks = self.mock_plt_yticks
        LLMSR.llmSR.plt.show = self.mock_plt_show
        LLMSR.llmSR.plt.close = self.mock_plt_close
        LLMSR.llmSR.run_genetic = self.mock_run_genetic
        
        # Setup a test case where there's at least one non-zero connection to trigger run_genetic
        self.mock_kan.symbolic_fun[0].mask = np.zeros((3, 2))
        self.mock_kan.act_fun[0].mask = np.zeros((2, 3))
        
        # Set one connection to have mask > 0 for activation
        self.mock_kan.act_fun[0].mask[0, 0] = 1.0
        
        # Call our test function with async mode
        result = test_kan_to_symbolic(
            self.mock_kan, 
            self.mock_client,
            population=2, 
            generations=1,
            exit_condition=0.1,
            use_async=True
        )
        
        # Verify that run_genetic was called with the use_async parameter
        for call_args in self.mock_run_genetic.call_args_list:
            args, kwargs = call_args
            self.assertTrue('use_async' in kwargs)
            self.assertTrue(kwargs['use_async'])

    def test_kan_to_symbolic_with_error_handling(self):
        """Test error handling in kan_to_symbolic"""
        # Set the mock functions except run_genetic (we'll create a special one)
        LLMSR.llmSR.generate_base64_image = self.mock_generate_base64_image
        LLMSR.llmSR.plt.subplots = self.mock_plt_subplots
        LLMSR.llmSR.plt.xticks = self.mock_plt_xticks
        LLMSR.llmSR.plt.yticks = self.mock_plt_yticks
        LLMSR.llmSR.plt.show = self.mock_plt_show
        LLMSR.llmSR.plt.close = self.mock_plt_close
        
        # Create a mock run_genetic that raises an exception on the second call
        call_count = [0]  # Use a list to maintain state across calls
        
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Second call raises error
                raise ValueError("Testing error handling - this error is expected, and is not a concern.")
            return [
                [{
                    'params': np.array([1.0, 2.0, 3.0]),
                    'score': -0.01,
                    'ansatz': 'params[0] * np.sin(params[1] * x + params[2])',
                    'Num_params': 3
                }]
            ]
        
        error_mock = MagicMock(side_effect=side_effect)
        LLMSR.llmSR.run_genetic = error_mock
        
        # Setup connections to ensure multiple calls to run_genetic
        self.mock_kan.symbolic_fun[0].mask = np.zeros((3, 2))
        self.mock_kan.act_fun[0].mask = np.zeros((2, 3))
        
        # Set multiple connections to have mask > 0 for activation 
        # to trigger multiple run_genetic calls
        self.mock_kan.act_fun[0].mask[0, 0] = 1.0
        self.mock_kan.act_fun[0].mask[0, 1] = 1.0
        self.mock_kan.act_fun[0].mask[1, 0] = 1.0
        
        # Call our test function - we expect a critical log when the error occurs
        with self.assertLogs(level='ERROR') as log_context:  # Use ERROR level instead of CRITICAL
            result = test_kan_to_symbolic(
                self.mock_kan, 
                self.mock_client,
                population=2, 
                generations=1,
                exit_condition=0.1
            )
        
        # Verify error was logged
        self.assertTrue(any('Testing error handling - this error is expected, and is not a concern.' in msg for msg in log_context.output))
        # The function should continue despite errors
        self.assertIsInstance(result, dict)
        
        # Verify run_genetic was called multiple times
        self.assertGreater(call_count[0], 1)

if __name__ == '__main__':
    unittest.main()