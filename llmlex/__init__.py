import logging
import os

# Configure logging
logger = logging.getLogger("llmlex")
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)

# Allow log level to be set via environment variable
log_level = os.environ.get('llmlex_LOG_LEVEL', 'INFO').upper()
if log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
    logger.setLevel(getattr(logging, log_level))

from .llmlex import *
from . import images
from . import llm
from . import fit
from . import response