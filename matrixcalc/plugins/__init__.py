"""Plugin system for extensible matrix operations."""

import os
import sys
import importlib
import importlib.util
from typing import Dict, List, Callable, Any, Optional
from abc import ABC, abstractmethod
import inspect

from ..config.settings import Config
from ..logging.setup import get_logger
from ..core.matrix import Matrix

logger = get_logger(__name__)


class PluginInterface(ABC):
    """Abstract interface for matrix operation plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass
    
    @abstractmethod
    def get_operations(self) -> Dict[str, Callable]:
        """Get dictionary of operation_name -> function mappings."""
        pass
    
    def initialize(self) -> None:
        """Initialize plugin (called when loaded)."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup plugin (called when unloaded)."""
        pass


class PluginManager:
    """Manages loading and execution of plugins."""
    
    def __init__(self):
        self.plugins: Dict[str, PluginInterface] = {}
        self.operations: Dict[str, Callable] = {}
        self._plugin_directories: List[str] = []
    
    def add_plugin_directory(self, directory: str) -> None:
        """Add a directory to search for plugins."""
        if os.path.isdir(directory) and directory not in self._plugin_directories:
            self._plugin_directories.append(directory)
            logger.info(f"Added plugin directory: {directory}")
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins in plugin directories."""
        discovered = []
        
        # Add configured plugin directories
        for directory in Config.plugin_directories:
            self.add_plugin_directory(directory)
        
        for directory in self._plugin_directories:
            if not os.path.exists(directory):
                continue
            
            for filename in os.listdir(directory):
                if filename.endswith('.py') and not filename.startswith('_'):
                    plugin_path = os.path.join(directory, filename)
                    discovered.append(plugin_path)
                    logger.debug(f"Discovered plugin: {plugin_path}")
        
        return discovered
    
    def load_plugin(self, plugin_path: str) -> bool:
        """
        Load a plugin from file path.
        
        Args:
            plugin_path: Path to the plugin file
            
        Returns:
            True if plugin loaded successfully, False otherwise
        """
        try:
            # Import the plugin module
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to load plugin spec: {plugin_path}")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class that implements PluginInterface
            plugin_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, PluginInterface) and 
                    obj is not PluginInterface and 
                    not obj.__name__.startswith('_')):
                    plugin_class = obj
                    break
            
            if plugin_class is None:
                logger.error(f"No valid plugin class found in: {plugin_path}")
                return False
            
            # Instantiate and register plugin
            plugin_instance = plugin_class()
            plugin_name = plugin_instance.name
            
            if plugin_name in self.plugins:
                logger.warning(f"Plugin '{plugin_name}' already loaded, skipping")
                return False
            
            # Initialize plugin
            plugin_instance.initialize()
            
            # Register plugin and its operations
            self.plugins[plugin_name] = plugin_instance
            plugin_operations = plugin_instance.get_operations()
            
            for op_name, op_func in plugin_operations.items():
                if op_name in self.operations:
                    logger.warning(f"Operation '{op_name}' already exists, overriding with plugin '{plugin_name}'")
                self.operations[op_name] = op_func
            
            logger.info(f"Loaded plugin '{plugin_name}' v{plugin_instance.version} with {len(plugin_operations)} operations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugin from {plugin_path}: {str(e)}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin by name.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if plugin unloaded successfully, False otherwise
        """
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin '{plugin_name}' not found")
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            
            # Remove plugin operations
            plugin_operations = plugin.get_operations()
            for op_name in plugin_operations.keys():
                if op_name in self.operations:
                    del self.operations[op_name]
            
            # Cleanup plugin
            plugin.cleanup()
            
            # Remove plugin
            del self.plugins[plugin_name]
            
            logger.info(f"Unloaded plugin '{plugin_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin '{plugin_name}': {str(e)}")
            return False
    
    def load_all_plugins(self) -> int:
        """
        Load all discovered plugins.
        
        Returns:
            Number of plugins loaded successfully
        """
        if not Config.enable_plugins:
            logger.info("Plugin system disabled")
            return 0
        
        discovered = self.discover_plugins()
        loaded_count = 0
        
        for plugin_path in discovered:
            if self.load_plugin(plugin_path):
                loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} out of {len(discovered)} discovered plugins")
        return loaded_count
    
    def execute_operation(self, operation_name: str, *args, **kwargs) -> Any:
        """
        Execute a plugin operation.
        
        Args:
            operation_name: Name of the operation to execute
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation
            
        Returns:
            Result of the operation
            
        Raises:
            ValueError: If operation not found
        """
        if operation_name not in self.operations:
            raise ValueError(f"Operation '{operation_name}' not found")
        
        try:
            operation_func = self.operations[operation_name]
            result = operation_func(*args, **kwargs)
            logger.debug(f"Executed plugin operation '{operation_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute operation '{operation_name}': {str(e)}")
            raise
    
    def get_available_operations(self) -> List[str]:
        """Get list of available plugin operations."""
        return list(self.operations.keys())
    
    def get_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded plugins."""
        info = {}
        for name, plugin in self.plugins.items():
            info[name] = {
                'version': plugin.version,
                'description': plugin.description,
                'operations': list(plugin.get_operations().keys())
            }
        return info
    
    def shutdown(self) -> None:
        """Shutdown plugin manager and cleanup all plugins."""
        plugin_names = list(self.plugins.keys())
        for plugin_name in plugin_names:
            self.unload_plugin(plugin_name)
        
        logger.info("Plugin manager shutdown complete")


# Global plugin manager instance
plugin_manager = PluginManager()


# Example plugin for demonstration
class ExamplePlugin(PluginInterface):
    """Example plugin showing how to implement custom operations."""
    
    @property
    def name(self) -> str:
        return "example_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Example plugin with custom matrix operations"
    
    def get_operations(self) -> Dict[str, Callable]:
        return {
            'matrix_sum': self.matrix_sum,
            'matrix_mean': self.matrix_mean,
            'matrix_max': self.matrix_max,
            'matrix_min': self.matrix_min
        }
    
    def matrix_sum(self, matrix: Matrix) -> float:
        """Calculate sum of all matrix elements."""
        total = 0
        for row in matrix.data:
            for elem in row:
                total += float(elem)
        return total
    
    def matrix_mean(self, matrix: Matrix) -> float:
        """Calculate mean of all matrix elements."""
        total = self.matrix_sum(matrix)
        count = matrix.rows * matrix.cols
        return total / count if count > 0 else 0
    
    def matrix_max(self, matrix: Matrix) -> float:
        """Find maximum element in matrix."""
        max_val = float('-inf')
        for row in matrix.data:
            for elem in row:
                val = float(elem)
                if val > max_val:
                    max_val = val
        return max_val
    
    def matrix_min(self, matrix: Matrix) -> float:
        """Find minimum element in matrix."""
        min_val = float('inf')
        for row in matrix.data:
            for elem in row:
                val = float(elem)
                if val < min_val:
                    min_val = val
        return min_val