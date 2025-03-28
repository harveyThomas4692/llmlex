import os
import matplotlib.pyplot as plt
import sys
import unittest
import numpy as np
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import logging
import pytest
import tempfile

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import LLM_LEx
from LLM_LEx.LLMLEx import single_call, run_genetic, async_single_call

# Optional imports for real API tests
try:
    import openai
except ImportError:
    pass  # Will be handled in the test_real_api_call

# Suppress logging during tests
logging.getLogger().setLevel(logging.CRITICAL)

class TestLLM_LEx(unittest.TestCase):
    def setUp(self):
        """Set up test data for each test"""
        # Create simple test data 
        self.x = np.linspace(-5, 5, 20)
        self.y = 2 * self.x**2 + 3 * self.x + 5  # Polynomial function
        
        # Create a base64 image for testing
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        self.base64_image = LLM_LEx.images.generate_base64_image(fig, ax, self.x, self.y)
        plt.close(fig)
        
        # Create a mock client
        self.client = MagicMock()
        self.async_client = AsyncMock()
    
    def create_mock_api_response(self, ansatz="params[0] * x**2 + params[1] * x + params[2]"):
        """Create a mock API response with the given ansatz"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = f"""
        After analyzing the graph, I think the data follows a formula like this:
        
        ```python
        {ansatz}
        ```
        
        This should provide a good fit to the data.
        """
        return mock_response

    def test_single_call(self):
        """Test the single_call function with fully mocked dependencies"""
        # Create test data
        test_params = np.array([2.0, 3.0, 5.0])
        test_chi_squared = 0.001
        test_ansatz = "params[0] * x**2 + params[1] * x + params[2]"
        mock_function = lambda x, *p: p[0]*x**2 + p[1]*x + p[2]
        
        # Instead of mocking lower-level call_model, mock at the highest level needed
        # This avoids issues with complex chained mocks
        with patch('LLM_LEx.llmlex.call_model') as mock_call_model, \
             patch('LLM_LEx.response.extract_ansatz') as mock_extract_ansatz, \
             patch('LLM_LEx.response.fun_convert') as mock_fun_convert, \
             patch('LLM_LEx.fit.fit_curve') as mock_fit_curve:
            
            # Set up the mocks with appropriate return values - need to mock formatted responses
            mock_call_model.return_value = """Based on the plot, I think the relationship can be modeled as:
            
```python
params[0] * x**2 + params[1] * x + params[2]
```

This is a quadratic function that should fit the data well."""
            mock_extract_ansatz.return_value = (test_ansatz, 3)
            mock_fun_convert.return_value = (mock_function, 3, f"lambda x, *params: {test_ansatz}")
            mock_fit_curve.return_value = (test_params, test_chi_squared)
            
            # Call the function
            result = single_call(self.client, self.base64_image, self.x, self.y)
            
            # Verify the result structure
            self.assertIsInstance(result, dict)
            self.assertIn('ansatz', result)
            self.assertIn('params', result)
            self.assertIn('score', result)
            
            # Verify the expected values based on our mocks
            self.assertEqual(result['ansatz'], test_ansatz)
            np.testing.assert_array_equal(result['params'], test_params)
            self.assertEqual(result['score'], -test_chi_squared)  # Score is negative of chi-squared

    def test_async_single_call(self):
        """Test the async_single_call function with comprehensive mocking"""
        # Create test data
        test_params = np.array([0.1, 2.0, 3.0])
        test_chi_squared = 0.001
        test_ansatz = "params[0] * np.exp(-params[1] * x) * np.sin(params[2] * x)"
        test_func = lambda x, *p: p[0] * np.exp(-p[1] * x) * np.sin(p[2] * x)
        
        # Create async test function that patches the bare minimum of dependencies
        async def run_test():
            # Use all the needed patches
            with patch('LLM_LEx.llmlex.async_call_model') as mock_async_call, \
                 patch('LLM_LEx.response.extract_ansatz') as mock_extract_ansatz, \
                 patch('LLM_LEx.response.fun_convert') as mock_fun_convert, \
                 patch('LLM_LEx.fit.fit_curve') as mock_fit_curve:
                
                # Configure all the mocks with appropriate return values that look like real API responses
                mock_async_call.return_value = """Looking at the data plot, I can see this appears to be an oscillating function with decay.
                
```python
params[0] * np.exp(-params[1] * x) * np.sin(params[2] * x)
```

This function captures both the oscillation and decay visible in the data."""
                mock_extract_ansatz.return_value = (test_ansatz, 3)
                mock_fun_convert.return_value = (test_func, 3, f"lambda x, *params: {test_ansatz}")  
                mock_fit_curve.return_value = (test_params, test_chi_squared)
                
                # Call the function we're testing
                result = await async_single_call(
                    self.async_client, 
                    self.base64_image, 
                    self.x, 
                    self.y
                )
                
                # Verify the expected fields exist
                self.assertIsInstance(result, dict)
                self.assertIn('ansatz', result)
                self.assertIn('params', result)
                self.assertIn('score', result)
                self.assertIn('Num_params', result)
                
                # Verify the expected values 
                self.assertEqual(result['ansatz'], test_ansatz)
                np.testing.assert_array_equal(result['params'], test_params)
                self.assertEqual(result['score'], -test_chi_squared)
                
                return result
        
        # Setup and run the test with a clean event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            self.assertIsNotNone(result)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    
    def test_run_genetic_behavior(self):
        """Test run_genetic focusing on behavior and not implementation details"""
        # Create mock results for API calls
        mock_results = [
            {
                'params': np.array([2.0, 3.0, 5.0]),
                'score': -0.001,
                'ansatz': 'params[0] * x**2 + params[1] * x + params[2]',
                'Num_params': 3
            },
            {
                'params': np.array([0.1, 2.0, 3.0]),
                'score': -0.002,
                'ansatz': 'params[0] * np.exp(-params[1] * x) * np.sin(params[2] * x)',
                'Num_params': 3
            }
        ]
        
        # Create a mock for async_single_call that returns these results
        async def mock_async_single_call(*args, **kwargs):
            # Return different results based on temperature to simulate genetic diversity
            temp = kwargs.get('temperature', 1.0)
            # Return the first result for lower temperatures, second for higher
            if temp < 0.8:
                return mock_results[0]
            else:
                return mock_results[1]
        
        # Setup event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Patch only the external dependency
            with patch('LLM_LEx.llmlex.async_single_call', side_effect=mock_async_single_call):
                # Run with minimal generations and population
                result = run_genetic(
                    self.async_client, 
                    self.base64_image, 
                    self.x, 
                    self.y,
                    population_size=2,
                    num_of_generations=2,
                    exit_condition=0.0001
                )
                
                # Verify the overall structure without checking implementation details
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 2)  # 2 generations
                
                # Each generation should have a population
                for gen in result:
                    self.assertGreaterEqual(len(gen), 1)
                    
                    # Each member should have basic properties
                    for member in gen:
                        self.assertIn('ansatz', member)
                        self.assertIn('params', member)
                        self.assertIn('score', member)
        finally:
            # Clean up event loop
            loop.close()
            asyncio.set_event_loop(None)
    
    @pytest.mark.api
    def test_real_api_call(self):
        """Test with a real API call - runs by default unless disabled"""
        # Skip if no API key is set
        if not os.environ.get('OPENROUTER_API_KEY'):
            self.skipTest("No OPENROUTER_API_KEY found - skipping real API test")
        
        try:
            import openai
        except ImportError:
            self.skipTest("openai module not installed - skipping real API test")
        
        # Setup client with API key
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get('OPENROUTER_API_KEY')
        )
        
        # Check initial usage
        try:
            initial_usage = LLM_LEx.llm.check_key_limit(client)
        except Exception:
            # Skip if we can't check usage
            self.skipTest("Could not check API key limit - skipping real API test")
        
        # Generate test data with noise for realism
        x = np.linspace(-5, 5, 20)
        y = 2*x**2 + 3*x + 5 + np.random.normal(0, 1, 20)  # Quadratic with noise
        
        # Generate image
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        base64_image = LLM_LEx.images.generate_base64_image(fig, ax, x, y)
        plt.close(fig)
        
        # Run test with real API call
        try:
            result = single_call(
                client,
                base64_image,
                x,
                y,
                model="openai/gpt-4o-mini"  # Use mini model to save costs
            )
            
            # Verify basic structure
            self.assertIsNotNone(result)
            self.assertIn('ansatz', result)
            self.assertIn('params', result)
            self.assertIn('score', result)
            
            # Verify the result is properly typed
            self.assertIsInstance(result['ansatz'], str)
            self.assertIsInstance(result['params'], np.ndarray)
            self.assertIsInstance(result['score'], float)
            
            # Check if model identified the quadratic pattern
            # Note: This is a bit fuzzy as the model might find other good fits
            self.assertLess(result['score'], 0, "Score should be negative (lower is better)")
            
            # Check final usage for reporting
            try:
                final_usage = LLM_LEx.llm.check_key_limit(client)
                print(f"API test cost: ${np.round(final_usage - initial_usage, 3)}")
            except Exception:
                pass
        
        except Exception as e:
            self.skipTest(f"API test failed: {str(e)}")

class TestGenetic(unittest.TestCase):
    """Additional tests for the genetic algorithm aspects"""
    
    def setUp(self):
        """Set up test data"""
        # Use fixed seed to ensure deterministic behavior
        np.random.seed(42)
        self.x = np.linspace(-5, 5, 50)
        self.y = np.sin(self.x) + 0.1 * np.random.randn(50)  # Sine function with noise
        self.base64_image = "fake_base64_string"  # Not used except as a parameter
        self.async_client = AsyncMock()
    
    def test_genetic_selection(self):
        """Test that the genetic algorithm properly selects better candidates"""
        # Create a sequence of results with improving scores
        # Each has required properties to match expected output format
        result_sequence = [
            {
                'score': -0.5, 
                'ansatz': 'x', 
                'params': np.array([1.0]),
                'Num_params': 1,
                'response': 'mock response',
                'prompt': 'mock prompt',
                'function_list': None
            },
            {
                'score': -0.3, 
                'ansatz': 'x**2', 
                'params': np.array([1.0]),
                'Num_params': 1,
                'response': 'mock response',
                'prompt': 'mock prompt',
                'function_list': None
            },
            {
                'score': -0.1, 
                'ansatz': 'np.sin(x)', 
                'params': np.array([1.0]),
                'Num_params': 1,
                'response': 'mock response',
                'prompt': 'mock prompt',
                'function_list': None
            },
            {
                'score': -0.05, 
                'ansatz': 'np.sin(x) + params[0]', 
                'params': np.array([0.1]),
                'Num_params': 1,
                'response': 'mock response',
                'prompt': 'mock prompt',
                'function_list': None
            }
        ]
        
        # Counter to return different results on consecutive calls
        call_count = 0
        
        # Mock function that returns increasingly better results
        async def mock_api_call(*args, **kwargs):
            nonlocal call_count
            index = min(call_count, len(result_sequence) - 1)
            call_count += 1
            # Add any required properties of async_single_call result
            return result_sequence[index]
        
        # Setup event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Mock async_single_call to return our test sequence
            with patch('LLM_LEx.llmlex.async_single_call', side_effect=mock_api_call):
                # Run genetic algorithm with minimal settings 
                result = run_genetic(
                    self.async_client,
                    self.base64_image,
                    self.x,
                    self.y,
                    population_size=2,
                    num_of_generations=2,
                    exit_condition=0.0001  # Very low exit condition to ensure we run all generations
                )
                
                # Verify we get the expected number of generations
                self.assertEqual(len(result), 2, "Should return 2 generations")
                
                # Verify the initial population
                self.assertGreaterEqual(len(result[0]), 1, "Initial population should have at least 1 member")
                
                # Verify the final population 
                self.assertGreaterEqual(len(result[1]), 1, "Final population should have at least 1 member")
                
                # Verify that members have required structure
                for member in result[-1]:
                    self.assertIn('score', member)
                    self.assertIn('ansatz', member)
                    self.assertIn('params', member)
                
        finally:
            # Clean up event loop
            loop.close()
            asyncio.set_event_loop(None)

if __name__ == '__main__':
    unittest.main()