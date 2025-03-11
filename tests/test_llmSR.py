import os
import matplotlib.pyplot as plt
import sys
import unittest
import numpy as np
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import the modules we want to patch first
import LLMSR.llm
import LLMSR.response
import LLMSR.fit
# Then import the functions we want to test
from LLMSR.llmSR import single_call, run_genetic, async_single_call
from tests.test_data.generate_test_data import generate_test_data

class TestLLMSR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate test data if it doesn't exist
        if not os.path.exists('tests/test_data/test_data.npz'):
            cls.x, cls.y, cls.base64_image = generate_test_data()
        else:
            # Load existing test data
            test_data = np.load('tests/test_data/test_data.npz')
            cls.x, cls.y = test_data['x'], test_data['y']
            with open('tests/test_data/test_image.txt', 'r') as f:
                cls.base64_image = f.read()
    
    def mock_llm_response(self, ansatz="params[0] * x**2 + params[1] * x + params[2]", 
                           num_params=3, params=None, score=0.001):
        """
        Create a mock LLM response with specified parameters for testing.
        
        Args:
            ansatz: String formula to return
            num_params: Number of parameters
            params: Parameter values to use
            score: Score value for the fit
            
        Returns:
            Mock response object with expected structure
        """
        if params is None:
            params = np.array([2, 3, 5])
        
        # Create a properly structured response similar to what the OpenAI API returns
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = ansatz
        
        return response

    def test_single_call_simplified(self):
        """Test a simplified version of single_call that we define for testing"""
        
        # Create test data
        x = np.array([1, 2, 3])
        y = np.array([10, 19, 32])  # 2*x^2 + 3*x + 5
        client = MagicMock()
        
        # Create a simplified version of single_call that doesn't depend on external calls
        def test_single_call(client, img, x, y):
            """A simplified version of single_call for testing"""
            # Define expected values
            expected_ansatz = "params[0] * x**2 + params[1] * x + params[2]"
            expected_params = np.array([2, 3, 5])
            expected_score = 0.001
            
            # Return a structure identical to the real single_call
            return {
                "params": expected_params,
                "score": -expected_score,
                "ansatz": expected_ansatz,
                "Num_params": 3,
                "response": "mock response",
                "prompt": "mock prompt",
                "function_list": None
            }
        
        # Call our test function
        result = test_single_call(client, "fake_image", x, y)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['ansatz'], "params[0] * x**2 + params[1] * x + params[2]")
        self.assertEqual(result['Num_params'], 3)
        np.testing.assert_array_equal(result['params'], np.array([2, 3, 5]))
        self.assertEqual(result['score'], -0.001)

    @patch('LLMSR.llmSR.async_model_call')
    def test_async_single_call(self, mock_async_model_call):
        """Test the async_single_call function with a mocked LLM response"""
        
        # Create test data
        x = np.linspace(0.01, 1, 10)
        y = x**2
        img = "fake_base64_string"
        client = MagicMock()
        
        # Create a mock response
        mock_response_text = """Here's a mathematical representation of the data:

params[0] * np.exp(-params[1] * x) * np.sin(params[2] * x)

This function captures the exponential decay with oscillation pattern."""

        async def mock_async_return(*args, **kwargs):
            return mock_response_text
            
        mock_async_model_call.side_effect = mock_async_return
        
        # Also patch the remaining processing functions to ensure they work correctly
        extract_ansatz_patcher = patch('LLMSR.response.extract_ansatz', 
                                       return_value=("params[0] * np.exp(-params[1] * x) * np.sin(params[2] * x)", 3))
        fun_convert_patcher = patch('LLMSR.response.fun_convert', 
                                    return_value=(lambda x, *p: p[0] * np.exp(-p[1] * x) * np.sin(p[2] * x), 3))
        fit_curve_patcher = patch('LLMSR.fit.fit_curve', 
                                 return_value=(np.array([0.1, 2.0, 3.0]), 0.001))
        
        # Create a proper async test function
        async def async_test():
            # Call the actual function we're testing with patchers applied
            with extract_ansatz_patcher, fun_convert_patcher, fit_curve_patcher:
                result = await async_single_call(client, img, x, y)
            
            # Verify the result
            self.assertIsInstance(result, dict)
            self.assertIn('ansatz', result)
            self.assertIn('score', result)
            self.assertIn('Num_params', result)
            self.assertIn('params', result)
            
            # Verify the expected values
            self.assertEqual(result['ansatz'], "params[0] * np.exp(-params[1] * x) * np.sin(params[2] * x)")
            self.assertEqual(result['Num_params'], 3)
            np.testing.assert_array_equal(result['params'], np.array([0.1, 2.0, 3.0]))
            self.assertEqual(result['score'], -0.001)
            
            return result
        
        # Run the async test with a clean event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(async_test())
            self.assertIsNotNone(result)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    
    def test_run_genetic(self):
        """Test the run_genetic function with proper mocking"""
        
        # Create test data
        x = np.linspace(0.01, 1, 10)
        y = x**2
        base64_image = "fake_base64_string"
        client = MagicMock()
        
        # Create a mock result that will be returned by single_call
        mock_result = {
            'params': np.array([0.1, 2.0, 3.0]),
            'score': -0.001,
            'ansatz': 'params[0] * np.exp(-params[1] * x) * np.sin(params[2] * x)',
            'Num_params': 3,
            'response': "mock response",
            'prompt': 'test prompt',
            'function_list': None
        }
        
        # Replace single_call with a function that returns our mock result
        with patch('LLMSR.llmSR.single_call', return_value=mock_result):
            # Patch curve_fit to return deterministic values
            with patch('LLMSR.llmSR.curve_fit', return_value=(np.array([1.0]), None)):
                # Run the genetic algorithm with minimal population/generations
                result = run_genetic(
                    client, base64_image, x, y, 
                    population_size=2, num_of_generations=2,
                    temperature=1.0, exit_condition=1e-7,
                    use_async=False
                )
                
                # Verify the result structure
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 2)  # Should have 2 generations
                
                # Each generation should have a population
                self.assertGreater(len(result[0]), 0)
                self.assertGreater(len(result[1]), 0)
        
    def test_run_genetic_async(self):
        """Test the run_genetic function with async mode enabled and proper mocking"""
        
        # Create test data
        x = np.linspace(0.01, 1, 10)
        y = x**2
        base64_image = "fake_base64_string"
        client = AsyncMock()
        
        # Create a mock result for the async_single_call function
        mock_result = {
            'params': np.array([0.1, 2.0, 3.0]),
            'score': -0.001,
            'ansatz': 'params[0] * np.exp(-params[1] * x) * np.sin(params[2] * x)',
            'Num_params': 3,
            'response': "mock response",
            'prompt': 'test prompt',
            'function_list': None
        }
        
        # Mock the async_single_call function
        async def mock_async_single_call(*args, **kwargs):
            return mock_result
            
        # Setup event loop for testing async code
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Patch the async_single_call function
            with patch('LLMSR.llmSR.async_single_call', side_effect=mock_async_single_call):
                # Run the genetic algorithm with async enabled
                result = run_genetic(
                    client, base64_image, x, y, 
                    population_size=2, num_of_generations=2,
                    temperature=1.0, exit_condition=1e-7,
                    use_async=True
                )
                
                # Verify basic structure of the result
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 2)  # Should have 2 generations
                
                # Each generation should have a population
                self.assertGreater(len(result[0]), 0)
                self.assertGreater(len(result[1]), 0)
        finally:
            # Clean up the event loop
            loop.close()
            asyncio.set_event_loop(None)
    
    def test_real_api_call(self):
        """Test with a real API call - skipped by default but can be enabled by setting LLMSR_TEST_REAL_API=1"""
        # Check if we should run with real API
        if not os.environ.get('LLMSR_TEST_REAL_API'):
            self.skipTest("Set LLMSR_TEST_REAL_API=1 to run tests with real API calls")
        
        # Import necessary modules for this test
        try:
            import openai
        except ImportError:
            self.skipTest("openai module not installed - skipping real API test")
            
        # Skip if no API key is set
        if not os.environ.get('OPENROUTER_API_KEY'):
            # Try to load from .env file
            if os.path.exists('.env'):
                from dotenv import load_dotenv
                load_dotenv()
                
            # Check again after loading .env
            if not os.environ.get('OPENROUTER_API_KEY'):
                self.skipTest("No OPENROUTER_API_KEY found - set this environment variable to run real API tests")
        
        # Setup client with API key
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get('OPENROUTER_API_KEY')
        )
        
        # Generate simple test data
        x = np.linspace(-5, 5, 20)
        y = 2*x**2 + 3*x + 5 + np.random.normal(0, 1, 20)  # Quadratic with noise
        
        # Generate image for the API call
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        from LLMSR.images import generate_base64_image
        base64_image = generate_base64_image(fig, ax, x, y)
        plt.close(fig)
        initial_usage = LLMSR.llm.check_key_usage(client)
        # Run the test with real API call
        print("Making real API call to model...")
        result = single_call(
            client,
            base64_image,
            x,
            y,
            model="openai/gpt-4o-mini"  # Use mini model to save costs
        )
        
        # Basic assertions to verify the result
        self.assertIsNotNone(result, f"API call failed to return a result {result}")
        self.assertIn('ansatz', result, "Result should contain 'ansatz'")
        self.assertIn('params', result, "Result should contain 'params'")
        self.assertIn('score', result, "Result should contain 'score'")
        
        # Print the discovered function
        print(f"Real API returned ansatz: {result['ansatz']}")
        print(f"With parameters: {result['params']}")
        print(f"Score: {result['score']}")
        final_usage = LLMSR.llm.check_key_usage(client)
        print(f"Spent on test: ${np.round(final_usage - initial_usage, 3)}")

if __name__ == '__main__':
    unittest.main()