# logger.py

import logging
import colorlog

def setup_logger(log_level=logging.DEBUG):
    """
    Sets up a logger that logs only to the console (CLI) with color formatting and timestamps in [].
    
    Args:
        log_level (int): The level of logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    
    Returns:
        logger (logging.Logger): The configured logger object.
    """
    
    # Create a custom logger
    logger = logging.getLogger(__name__)
    
    # Set the global logging level
    logger.setLevel(log_level)
    
    # Define log format with colored output and custom timestamp format
    log_format = (
        '%(log_color)s[%(levelname)s | %(asctime)s]: %(name)s - %(message)s'
    )
    
    formatter = colorlog.ColoredFormatter(
        log_format,
        datefmt='%Y-%m-%d %H:%M:%S',  # Custom timestamp format
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    # Create console handler and set the level to the provided log_level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger
