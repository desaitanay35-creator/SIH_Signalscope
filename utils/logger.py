"""
Structured Logging Module.
Provides standardized logger with timestamp formatting and log level controls.
Responsible Team Member: Member 6 (MLOps & Infrastructure)
"""

import logging

def get_logger(name: str) -> logging.Logger:
    """Returns configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
