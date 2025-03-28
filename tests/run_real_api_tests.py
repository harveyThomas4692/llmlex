#!/usr/bin/env python
"""
Run tests with a real API key.
This script is a wrapper around run_tests.py that forces real API tests to run.
"""

import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLM_LEx.api_tests")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM_LEx tests with real API calls")
    parser.add_argument('--max-rate', type=int, help="Maximum API calls per minute")
    args = parser.parse_args()
    
    # Try to load API key from .env file
    if os.path.exists('.env'):
        try:
            from dotenv import load_dotenv
            print("Loading environment variables from .env file...")
            load_dotenv()
            logger.info("Loaded environment variables from .env file")
        except ImportError:
            logger.warning("python-dotenv not installed, skipping .env loading")
    
    if not os.environ.get('OPENROUTER_API_KEY'):
        print("WARNING: No OPENROUTER_API_KEY environment variable found.")
        print("Please set this variable to run the real API tests:")
        print("  export OPENROUTER_API_KEY=your_api_key")
        print("  # or create a .env file with this variable")
        sys.exit(1)
        
    # Set custom rate limit if provided
    if args.max_rate:
        os.environ['LLM_LEx_TEST_MAX_CALLS_PER_MINUTE'] = str(args.max_rate)
        print(f"Setting maximum API call rate to {args.max_rate} calls per minute")
    
    # Import run_tests function from run_tests.py
    from tests.run_tests import run_tests, generate_test_data
    
    # Generate test data
    print("Generating test data...")
    generate_test_data()
    
    # Run tests with real API calls
    print("\nRunning tests with real API calls...\n")
    result = run_tests(include_real_api_tests=True)
    
    # Exit with appropriate status code
    sys.exit(not result.wasSuccessful())