"""
This file makes the pages directory a Python package.
This allows importing modules from the pages directory directly.
"""
import os
import sys
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Get the current directory
pages_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(pages_dir)

# Add the parent directory to the path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    logger.info(f"Added parent directory to Python path: {parent_dir}")

# Add the pages directory to the path
if pages_dir not in sys.path:
    sys.path.insert(0, pages_dir)
    logger.info(f"Added pages directory to Python path: {pages_dir}")

# Log the Python path for debugging
logger.info(f"Python path: {sys.path}")
