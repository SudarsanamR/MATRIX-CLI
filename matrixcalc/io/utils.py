"""File I/O utilities and helpers."""

import os
import tempfile
from typing import Union

from ..logging.setup import get_logger

logger = get_logger(__name__)


def ensure_parent_dir(filepath: str) -> None:
    """Ensure the parent directory of a file path exists."""
    parent_dir = os.path.dirname(filepath) or '.'
    try:
        os.makedirs(parent_dir, exist_ok=True)
        logger.debug(f"Ensured directory exists: {parent_dir}")
    except Exception as e:
        logger.error(f"Failed to create directory {parent_dir}: {str(e)}")
        raise


def write_text_atomic(filepath: str, content: str, encoding: str = 'utf-8') -> None:
    """
    Atomically write text content to filepath.
    
    Args:
        filepath: Path to write to
        content: Text content to write
        encoding: Text encoding to use
    """
    ensure_parent_dir(filepath)
    dir_path = os.path.dirname(filepath) or '.'
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding=encoding, dir=dir_path, delete=False) as tf:
            tf.write(content)
            temp_name = tf.name
        
        os.replace(temp_name, filepath)
        logger.debug(f"Atomically wrote text to: {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to write text to {filepath}: {str(e)}")
        raise


def write_bytes_atomic(filepath: str, data: bytes) -> None:
    """
    Atomically write bytes to filepath.
    
    Args:
        filepath: Path to write to
        data: Bytes data to write
    """
    ensure_parent_dir(filepath)
    dir_path = os.path.dirname(filepath) or '.'
    
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=dir_path, delete=False) as tf:
            tf.write(data)
            temp_name = tf.name
        
        os.replace(temp_name, filepath)
        logger.debug(f"Atomically wrote bytes to: {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to write bytes to {filepath}: {str(e)}")
        raise


def ensure_file_extension(filename: str, default_ext: str) -> str:
    """
    Ensure a filename has the specified extension.
    
    Args:
        filename: The filename to check
        default_ext: Default extension to add if missing
        
    Returns:
        Filename with extension
        
    Raises:
        ValueError: If filename is empty
    """
    if not filename:
        raise ValueError("Filename cannot be empty")
    
    _, ext = os.path.splitext(filename)
    if not ext:
        return filename + default_ext
    return filename


def get_file_size_mb(filepath: str) -> float:
    """Get file size in megabytes."""
    try:
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def is_file_readable(filepath: str) -> bool:
    """Check if file is readable."""
    try:
        return os.path.isfile(filepath) and os.access(filepath, os.R_OK)
    except OSError:
        return False


def is_file_writable(filepath: str) -> bool:
    """Check if file is writable (or parent directory is writable)."""
    try:
        if os.path.exists(filepath):
            return os.access(filepath, os.W_OK)
        else:
            parent_dir = os.path.dirname(filepath) or '.'
            return os.access(parent_dir, os.W_OK)
    except OSError:
        return False
