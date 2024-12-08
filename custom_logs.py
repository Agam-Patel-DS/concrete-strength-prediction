import logging
import os
from datetime import datetime

def get_logger(log_file_name="app.log"):
    """
    Creates and configures a logger to save logs in the 'logs/' folder.

    Args:
        log_file_name (str): Name of the log file (default: 'app.log').

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Define log file path
    log_file_path = os.path.join(log_dir, log_file_name)

    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Log all levels (DEBUG and above)

    # Check if handlers are already set to prevent duplicates
    if not logger.hasHandlers():
        # File handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_format = logging.Formatter('%(levelname)s - %(message)s')
        stream_handler.setFormatter(stream_format)
        logger.addHandler(stream_handler)

    return logger
