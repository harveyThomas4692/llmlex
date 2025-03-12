import pytest
import os
import pathlib

def pytest_addoption(parser):
    parser.addoption(
        "--no-api", action="store_true", default=False, help="Skip API tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "api: mark test as requiring API access")

def pytest_ignore_collect(collection_path, config):
    # Completely ignore archived tests by default
    if os.path.join("tests", "archive") in str(collection_path):
        return True
    return False

def pytest_collection_modifyitems(config, items):
    # Skip API tests if --no-api flag is set
    if config.getoption("--no-api"):
        skip_api = pytest.mark.skip(reason="API tests disabled with --no-api flag")
        for item in items:
            if "api" in item.keywords:
                item.add_marker(skip_api) 