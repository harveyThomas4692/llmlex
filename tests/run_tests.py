import unittest
import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import the test modules
from tests.test_images import TestImages
from tests.test_response import TestResponse
from tests.test_fit import TestFit
from tests.test_llm import TestLLM
from tests.test_llmSR import TestLLMSR
from tests.test_logging import TestLogging
from tests.test_kan import TestKANFunctionality, TestKanSrFunctions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLMSR.tests")

# Load environment variables from .env file if it exists
if os.path.exists('.env'):
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Loaded environment variables from .env file")
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env loading")

def check_api_key_available():
    """Check if an API key is available for running real API tests"""
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if api_key:
        logger.info("API key found, real API tests can be run")
        return True
    else:
        logger.info("No API key found, real API tests will be skipped")
        return False

def run_tests(include_real_api_tests=None):
    """
    Run all tests, optionally including real API tests
    
    Args:
        include_real_api_tests: If True, always run real API tests.
                               If False, never run real API tests.
                               If None (default), run if API key is available.
    """
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases to the suite
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestImages))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestResponse))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestFit))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestLLM))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestLLMSR))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestLogging))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestKANFunctionality))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestKanSrFunctions))
    
    # Check if we should run real API tests
    should_run_api_tests = include_real_api_tests
    if should_run_api_tests is None:
        should_run_api_tests = check_api_key_available()
    
    # Add real API tests if requested and API key is available
    if should_run_api_tests:
        # Set environment flag to enable real API tests
        os.environ['LLMSR_TEST_REAL_API'] = '1'
        print("Instructed to run real API tests")
        
        # Create a separate test suite just for API tests
        api_test_suite = unittest.TestSuite()
        api_test_suite.addTest(TestLLMSR('test_real_api_call'))
        
        # Add to main test suite
        test_suite.addTest(api_test_suite)
        logger.info("Including real API tests in test suite")
    else:
        # Make sure the flag is not set
        if 'LLMSR_TEST_REAL_API' in os.environ:
            del os.environ['LLMSR_TEST_REAL_API']
        logger.info("Skipping real API tests")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return the result
    return result

if __name__ == "__main__":
    # Parse any command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run LLMSR tests")
    parser.add_argument('--api-tests', action='store_true', help="Force run real API tests (requires API key)")
    parser.add_argument('--no-api-tests', action='store_true', help="Skip real API tests even if key is available")
    args = parser.parse_args()
    
    # Determine whether to run API tests
    include_api_tests = None
    if args.api_tests:
        include_api_tests = True
        print("Instructed to run real API tests")
        if not check_api_key_available():
            print("Warning: API key not found but API tests requested. Tests may fail.")
            logger.warning("API key not found but API tests requested. Tests may fail.")
            raise ValueError("API key not found but API tests requested. Tests may fail.")
    elif args.no_api_tests:
        include_api_tests = False
        print("Instructed to skip real API tests")
    
    # First generate test data
    from tests.test_data.generate_test_data import generate_test_data
    print("Generating test data...")
    generate_test_data()
    
    # Run tests
    print("\nRunning tests...\n")
    result = run_tests(include_real_api_tests=include_api_tests)
    
    # Exit with appropriate status code
    sys.exit(not result.wasSuccessful())