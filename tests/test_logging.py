import unittest
import logging
import io
import sys
import asyncio
import LLM_LEx
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from unittest.mock import patch, MagicMock, AsyncMock

class TestLogging(unittest.TestCase):
    """Test suite for verifying proper logging functionality in LLMSR"""
    
    def setUp(self):
        """Set up for each test case"""
        # Capture logs for testing
        self.log_capture = io.StringIO()
        self.handler = logging.StreamHandler(self.log_capture)
        self.handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
        self.root_logger = logging.getLogger("LLMSR")
        self.root_logger.addHandler(self.handler)
        
        # Store original level
        self.original_level = self.root_logger.level
        
        # Set to DEBUG for testing
        self.root_logger.setLevel(logging.DEBUG)
        
    def tearDown(self):
        """Clean up after each test"""
        self.root_logger.removeHandler(self.handler)
        self.root_logger.setLevel(self.original_level)
        self.log_capture.close()
    
    def test_logger_configuration(self):
        """Test that the module configures loggers properly"""
        # Verify root logger has handlers
        self.assertIsNotNone(self.root_logger.handlers)
        
        # Verify level is set correctly
        self.assertEqual(logging.INFO, self.original_level)
        
        # Check submodule loggers
        submodules = ['llmSR', 'llm', 'fit', 'images', 'response']
        for submodule in submodules:
            logger = logging.getLogger(f"LLMSR.{submodule}")
            # Verify logger exists and inherits from root logger
            self.assertEqual(logger.parent, self.root_logger)
    
    def test_log_levels(self):
        """Test that different log levels work properly"""
        logger = logging.getLogger("LLMSR.test")
        
        # Log at different levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Get log output
        log_content = self.log_capture.getvalue()
        
        # Check messages at all levels
        self.assertIn("DEBUG:LLMSR.test:Debug message", log_content)
        self.assertIn("INFO:LLMSR.test:Info message", log_content)
        self.assertIn("WARNING:LLMSR.test:Warning message", log_content)
        self.assertIn("ERROR:LLMSR.test:Error message", log_content)
    
    def test_images_logging_success_path(self):
        """Test logging in images module for successful operations"""
        # Create test data and figure
        fig, ax = plt.subplots()
        x = np.linspace(0, 1, 10)
        y = x**2
        
        # Call the function
        result = LLM_LEx.images.generate_base64_image(fig, ax, x, y)
        plt.close(fig)
        
        # Get log output
        log_content = self.log_capture.getvalue()
        
        # Verify appropriate log levels for normal operation
        self.assertIn("DEBUG:LLMSR.images:Generating base64 image", log_content)
        # Success should be logged at debug level, not higher
        self.assertNotIn("INFO:LLMSR.images:Successfully generated", log_content.upper())
        self.assertNotIn("ERROR", log_content)
    
    def test_images_logging_error_path(self):
        """Test logging in images module for error conditions"""
        # Create temporary test path that doesn't exist
        nonexistent_path = os.path.join(tempfile.gettempdir(), 'nonexistent_image.png')
        if os.path.exists(nonexistent_path):
            os.remove(nonexistent_path)
            
        # Attempt to encode a nonexistent image
        with self.assertRaises(FileNotFoundError):
            LLM_LEx.images.encode_image(nonexistent_path)
            
        # Get log output
        log_content = self.log_capture.getvalue()
        
        # Verify errors are logged appropriately
        self.assertIn("ERROR:LLMSR.images:Image file not found", log_content)

    @patch('LLMSR.llm.requests.get')
    def test_llm_logging_network_events(self, mock_get):
        """Test that network operations in llm module are logged properly"""
        # Setup success response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"limit_remaining": 1000}}
        mock_get.return_value = mock_response
        
        # Setup client mock
        client = MagicMock()
        client.api_key = "fake_key"
        client.base_url = "https://example.com"
        
        # Call the function - success path
        result = LLM_LEx.llm.check_key_limit(client)
        
        # Clear the log capture
        log_content = self.log_capture.getvalue()
        self.log_capture.truncate(0)
        self.log_capture.seek(0)
        
        # Check success logs contain appropriate information without implementation details
        self.assertIn("DEBUG:LLMSR.llm:Checking API key usage limit", log_content)
        self.assertIn("API key check successful", log_content)
        
        # Now test error path
        mock_response.status_code = 403
        mock_get.return_value = mock_response
        
        # Call the function - error path
        result = LLM_LEx.llm.check_key_limit(client)
        
        # Get new log content
        log_content = self.log_capture.getvalue()
        
        # Check error is logged at appropriate level
        self.assertIn("ERROR:LLMSR.llm:", log_content)
    
    def test_fit_module_logging_behavior(self):
        """Test that the fit module logs appropriate events at appropriate levels"""
        # Mock curve_fit to avoid actual fitting
        with patch('LLMSR.fit.curve_fit') as mock_curve_fit, \
             patch('LLMSR.fit.get_n_chi_squared_from_predictions') as mock_chi_squared:
            
            # Setup successful return
            mock_curve_fit.return_value = (np.array([1.0, 2.0]), None)
            mock_chi_squared.return_value = 0.001
            
            # Create test data and function
            x = np.linspace(0, 1, 10)
            y = x**2
            curve = lambda x, a, b: a * x**b
            
            # Call function - success path
            params, n_chi_squared = LLM_LEx.fit.fit_curve(x, y, curve, 2)
            
            # Get log output
            log_content = self.log_capture.getvalue()
            self.log_capture.truncate(0)
            self.log_capture.seek(0)
            
            # Verify debug logs for successful operation
            self.assertIn("DEBUG:LLMSR.fit:Fitting curve with", log_content)
            self.assertIn("DEBUG:LLMSR.fit:Optimised parameters", log_content)
            self.assertNotIn("ERROR:LLMSR.fit:", log_content)
            
            # Now test error path by making curve_fit raise an exception
            mock_curve_fit.side_effect = RuntimeError("Test error")
            
            # Call function - error path
            params, n_chi_squared = LLM_LEx.fit.fit_curve(x, y, curve, 2)
            
            # Get log output
            log_content = self.log_capture.getvalue()
            
            # Verify error is logged appropriately
            self.assertIn("INFO:LLMSR.fit:All methods failed for this fit", log_content)
    
    def test_response_logging_with_parse_errors(self):
        """Test that the response module logs parsing errors appropriately"""
        # Create a malformed response with no parameters
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "This response has no parameters"
        
        # Attempt to extract ansatz - should fail
        with self.assertRaises(ValueError):
            LLM_LEx.response.extract_ansatz(mock_response)
        
        # Get log output
        log_content = self.log_capture.getvalue()
        
        # Verify appropriate debug logs leading up to error
        self.assertIn("DEBUG:LLMSR.response:Extracting ansatz from model response", log_content)
        self.assertIn("DEBUG:LLMSR.response:No parameters found", log_content)

if __name__ == '__main__':
    unittest.main()