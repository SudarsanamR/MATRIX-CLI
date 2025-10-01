"""Configuration settings and management."""

import json
import os
import tempfile
from typing import List, Dict, Any
import importlib

# Logger will be created when needed to avoid circular imports


class Config:
    """Global configuration settings."""
    
    # Core settings
    precision: int = 4
    default_export_format: str = 'csv'
    colored_output: bool = True
    auto_save: bool = False
    save_directory: str = './matrices'
    show_progress: bool = True
    
    # Advanced settings
    max_history_size: int = 20
    enable_caching: bool = True
    matrix_size_warning_threshold: int = 1000
    recent_files_list: List[str] = []
    log_level: str = 'INFO'
    
    # Cache settings
    cache_size: int = 100
    cache_ttl_seconds: int = 3600  # 1 hour
    numeric_atol: float = 1e-9
    numeric_rtol: float = 1e-12
    log_format: str = 'plain'  # 'plain' or 'json'
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_mb: int = 1024  # 1GB
    
    # Plugin settings
    enable_plugins: bool = False
    plugin_directories: List[str] = ['./plugins']
    
    # Security settings
    security_level: str = 'moderate'  # 'strict', 'moderate', 'permissive'
    enable_audit_log: bool = False
    expression_timeout_seconds: int = 30
    
    @classmethod
    def load_from_file(cls, filename: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(filename, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)
            print_success(f"Configuration loaded from {filename}")
            # Get logger when needed to avoid circular imports
            from ..logging.setup import get_logger
            logger = get_logger(__name__)
            logger.info(f"Configuration loaded from {filename}")
            
            # Reconfigure logging to apply new level
            from ..logging.setup import setup_logging
            setup_logging()
            
        except Exception as e:
            print_error(f"Error loading configuration: {str(e)}")
            logger.error(f"Error loading configuration: {str(e)}")
    
    @classmethod
    def save_to_file(cls, filename: str) -> None:
        """Save configuration to JSON file."""
        try:
            config_data = {
                'precision': cls.precision,
                'default_export_format': cls.default_export_format,
                'colored_output': cls.colored_output,
                'auto_save': cls.auto_save,
                'save_directory': cls.save_directory,
                'show_progress': cls.show_progress,
                'max_history_size': cls.max_history_size,
                'enable_caching': cls.enable_caching,
                'matrix_size_warning_threshold': cls.matrix_size_warning_threshold,
                'recent_files_list': cls.recent_files_list,
                'log_level': cls.log_level,
                'cache_size': cls.cache_size,
                'cache_ttl_seconds': cls.cache_ttl_seconds,
                'numeric_atol': cls.numeric_atol,
                'numeric_rtol': cls.numeric_rtol,
                'log_format': cls.log_format,
                'parallel_processing': cls.parallel_processing,
                'max_workers': cls.max_workers,
                'memory_limit_mb': cls.memory_limit_mb,
                'enable_plugins': cls.enable_plugins,
                'plugin_directories': cls.plugin_directories,
                'security_level': cls.security_level,
                'enable_audit_log': cls.enable_audit_log,
                'expression_timeout_seconds': cls.expression_timeout_seconds
            }
            
            from ..io.utils import ensure_parent_dir
            ensure_parent_dir(filename)
            
            dir_path = os.path.dirname(filename) or '.'
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=dir_path, delete=False) as tf:
                json.dump(config_data, tf, indent=2)
                temp_name = tf.name
            os.replace(temp_name, filename)
            
            print_success(f"Configuration saved to {filename}")
            from ..logging.setup import get_logger
            logger = get_logger(__name__)
            logger.info(f"Configuration saved to {filename}")
            
        except Exception as e:
            print_error(f"Error saving configuration: {str(e)}")
            from ..logging.setup import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error saving configuration: {str(e)}")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate all configuration values."""
        try:
            # Basic schema-driven checks (without external deps)
            try:
                import json
                import importlib.resources as pkg_resources
                with pkg_resources.files(__package__).joinpath('schema.json').open('r') as f:  # type: ignore[attr-defined]
                    schema = json.load(f)
                allowed_keys = set(schema.get('properties', {}).keys())
                cfg_keys = set(cls.get_all_settings().keys())
                unknown = cfg_keys - allowed_keys
                if unknown:
                    raise ValueError(f"Unknown configuration keys: {sorted(list(unknown))}")
            except Exception:
                # Schema file missing or resource loading failed; continue with built-in checks
                pass
            
            if cls.precision < 0:
                raise ValueError("Precision must be non-negative")
            
            if not os.path.exists(cls.save_directory) and not os.access(os.path.dirname(cls.save_directory) or '.', os.W_OK):
                raise ValueError("Save directory does not exist or is not writable")
            
            if cls.default_export_format not in ['csv', 'json', 'latex', 'numpy', 'matlab', 'text']:
                raise ValueError("Invalid default export format")
            
            if cls.max_history_size < 1:
                raise ValueError("Max history size must be positive")
            
            if cls.matrix_size_warning_threshold < 1:
                raise ValueError("Matrix size warning threshold must be positive")
            
            if cls.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                raise ValueError("Invalid log level")
            if cls.log_format not in ['plain', 'json']:
                raise ValueError("Invalid log format")
            
            if cls.cache_size < 1:
                raise ValueError("Cache size must be positive")
            
            if cls.cache_ttl_seconds < 0:
                raise ValueError("Cache TTL must be non-negative")
            
            if cls.max_workers < 1 or cls.max_workers > 32:
                raise ValueError("Max workers must be between 1 and 32")
            
            if cls.memory_limit_mb < 64:
                raise ValueError("Memory limit must be at least 64 MB")
            
            if cls.security_level not in ['strict', 'moderate', 'permissive']:
                raise ValueError("Invalid security level")
            
            if cls.expression_timeout_seconds < 1 or cls.expression_timeout_seconds > 300:
                raise ValueError("Expression timeout must be between 1 and 300 seconds")
            
            return True
            
        except ValueError as e:
            from ..logging.setup import get_logger
            logger = get_logger(__name__)
            logger.error(f"Configuration validation failed: {str(e)}")
            raise
    
    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary."""
        return {
            'precision': cls.precision,
            'default_export_format': cls.default_export_format,
            'colored_output': cls.colored_output,
            'auto_save': cls.auto_save,
            'save_directory': cls.save_directory,
            'show_progress': cls.show_progress,
            'max_history_size': cls.max_history_size,
            'enable_caching': cls.enable_caching,
            'matrix_size_warning_threshold': cls.matrix_size_warning_threshold,
            'recent_files_list': cls.recent_files_list,
            'log_level': cls.log_level,
            'cache_size': cls.cache_size,
            'cache_ttl_seconds': cls.cache_ttl_seconds,
            'numeric_atol': cls.numeric_atol,
            'numeric_rtol': cls.numeric_rtol,
            'log_format': cls.log_format,
            'parallel_processing': cls.parallel_processing,
            'max_workers': cls.max_workers,
            'memory_limit_mb': cls.memory_limit_mb,
            'enable_plugins': cls.enable_plugins,
            'plugin_directories': cls.plugin_directories,
            'security_level': cls.security_level,
            'enable_audit_log': cls.enable_audit_log,
            'expression_timeout_seconds': cls.expression_timeout_seconds
        }


# Import print functions to avoid circular imports
def print_success(message: str) -> None:
    """Print success message in green."""
    from colorama import Fore, Style
    if Config.colored_output:
        print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")
    else:
        print(f"✓ {message}")


def print_error(message: str) -> None:
    """Print error message in red."""
    from colorama import Fore, Style
    if Config.colored_output:
        print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")
    else:
        print(f"✗ {message}")
