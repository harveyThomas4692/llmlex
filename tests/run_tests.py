#!/usr/bin/env python
"""
Run tests for the LLMSR package.

This script provides a convenient way to run the test suite with command line arguments
to control which tests are run, including API tests and archived tests.
"""

import os
import sys
import logging
import warnings
import subprocess
import shlex

# Add parent directory to path so that 'tests' can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

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

def run_tests(no_api=False, run_archived=False):
    """
    Run all tests using pytest.
    
    Args:
        no_api: If True, skip API tests.
        run_archived: If True, run archived tests.
    
    Returns:
        Exit code from pytest
    """
    # First generate test data
    from tests.test_data.generate_test_data import generate_test_data
    print("Generating test data...")
    generate_test_data()
    
    # Build the pytest command
    pytest_cmd = ["python", "-m", "pytest", "tests/"]
    
    # Add arguments
    if no_api:
        pytest_cmd.append("--no-api")
    
    if run_archived:
        # When run_archived is True, we want to run all tests including the archive directory
        # Since we're using pytest_ignore_collect to ignore archived tests by default, we need to
        # modify the command to explicitly include the archive directory
        pytest_cmd = ["python", "-m", "pytest", "tests/", "tests/archive/"]
    
    # Add verbosity
    pytest_cmd.append("-v")
    
    # Run pytest
    print(f"\nRunning command: {' '.join(pytest_cmd)}\n")
    return subprocess.call(pytest_cmd)

if __name__ == "__main__":
    # Parse any command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run LLMSR tests using pytest")
    parser.add_argument('--no-api', action='store_true', help="Skip API tests")
    parser.add_argument('--run-archived', action='store_true', help="Run archived tests")
    parser.add_argument('--pytest-args', type=str, help="Additional arguments to pass to pytest")
    args = parser.parse_args()
    
    # Check for API key if needed
    if not args.no_api and not check_api_key_available():
        print("Warning: API key not found, API tests will be skipped or may fail.")
        logger.warning("API key not found, API tests will be skipped or may fail.")
    
    # Run the tests
    print("\nRunning tests...\n")
    exit_code = run_tests(
        no_api=args.no_api,
        run_archived=args.run_archived
    )
    
    # If additional pytest args were provided, run again with those
    if args.pytest_args:
        additional_cmd = ["python", "-m", "pytest"] + shlex.split(args.pytest_args)
        print(f"\nRunning additional pytest command: {' '.join(additional_cmd)}\n")
        exit_code = subprocess.call(additional_cmd)
    
    # Exit with appropriate status code
    sys.exit(exit_code)