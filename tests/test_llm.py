import os
import sys
import unittest
import asyncio
from unittest.mock import MagicMock, patch

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from LLMSR.llm import get_prompt, call_model, async_call_model

class TestLLM(unittest.TestCase):
    def test_get_prompt_default(self):
        """Test get_prompt with default parameters"""
        prompt = get_prompt()
        expected = "import numpy as np \ncurve_0 = lambda x, *params: params[0] \ncurve_1 = lambda x, *params:"
        self.assertEqual(prompt, expected)
    
    def test_get_prompt_custom(self):
        """Test get_prompt with custom function list"""
        function_list = [("params[0] * x**2", 1), ("params[0] * np.sin(params[1] * x)", 2)]
        prompt = get_prompt(function_list)
        expected = (
            "import numpy as np \n"
            "curve_0 = lambda x, *params: params[0] * x**2 \n"
            "curve_1 = lambda x, *params: params[0] * np.sin(params[1] * x) \n"
            "curve_2 = lambda x, *params:"
        )
        self.assertEqual(prompt, expected)
    
    @patch('LLMSR.llm.openai')
    def test_call_model(self, mock_openai):
        """Test call_model function with mocked OpenAI"""
        # Create mock client and response
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_client.chat = mock_chat
        
        mock_response = MagicMock()
        mock_chat.completions.create.return_value = mock_response
        
        # Call the function
        result = call_model(
            mock_client,
            "openai/gpt-4o",
            "dummy_base64_image",
            "test prompt",
            "test system prompt"
        )
        
        # Check that the call was made with the right parameters
        mock_chat.completions.create.assert_called_once()
        call_kwargs = mock_chat.completions.create.call_args[1]
        
        self.assertEqual(call_kwargs['model'], "openai/gpt-4o")
        self.assertEqual(call_kwargs['messages'][0]['content'], "test system prompt")
        self.assertEqual(call_kwargs['messages'][1]['role'], "user")
        
        # Check that the image was included correctly
        image_content = call_kwargs['messages'][1]['content'][0]
        self.assertEqual(image_content['type'], "image_url")
        self.assertEqual(image_content['image_url']['url'], "data:image/png;base64,dummy_base64_image")
        
        # Check that the prompt was included correctly
        text_content = call_kwargs['messages'][1]['content'][1]
        self.assertEqual(text_content['type'], "text")
        self.assertEqual(text_content['text'], "test prompt")
        
        # Check that the result was returned
        self.assertEqual(result, mock_response)
    
    @patch('LLMSR.llm._rate_limit_lock')
    def test_async_call_model(self, mock_lock):
        """Test async_call_model function with mocked client"""
        # Create mocks for lock functionality
        mock_lock.__enter__ = MagicMock(return_value=None)
        mock_lock.__exit__ = MagicMock(return_value=None)
        
        # Create mock client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        
        # Set up the mock to handle async calls properly
        async def mock_acreate(*args, **kwargs):
            return mock_response
        
        # Attach the async method to our mock
        mock_client.chat.completions.acreate = mock_acreate
        
        # Create a test coroutine
        async def test_coro():
            result = await async_call_model(
                mock_client,
                "openai/gpt-4o",
                "dummy_base64_image",
                "test prompt",
                "test system prompt"
            )
            return result
        
        # Run the coroutine in an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(test_coro())
            self.assertEqual(result, mock_response)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

if __name__ == '__main__':
    unittest.main()