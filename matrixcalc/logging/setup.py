"""Logging configuration and setup."""

import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

# Config will be imported when needed to avoid circular imports


def _get_log_level_from_config() -> int:
    """Get log level from config."""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    try:
        from ..config.settings import Config
        return level_map.get(Config.log_level, logging.INFO)
    except Exception:
        return logging.INFO


def setup_logging(log_file: str = 'matrix_calculator.log', 
                 max_bytes: int = 1_000_000, 
                 backup_count: int = 5) -> logging.Logger:
    """
    Configure application logging with rotation, respecting Config.log_level.
    
    Args:
        log_file: Path to log file
        max_bytes: Maximum bytes per log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('matrix_calculator')
    logger.setLevel(_get_log_level_from_config())

    # Clear existing handlers to allow reconfiguration
    while logger.handlers:
        logger.handlers.pop().close()

    # Choose formatter based on config
    try:
        from ..config.settings import Config
        if getattr(Config, 'log_format', 'plain') == 'json':
            log_formatter = logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s","logger":"%(name)s"}')
        else:
            log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    except Exception:
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Rotating file handler
    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(_get_log_level_from_config())

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(_get_log_level_from_config())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance."""
    if name:
        return logging.getLogger(f'matrix_calculator.{name}')
    return logging.getLogger('matrix_calculator')
