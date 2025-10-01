"""Performance optimization utilities for matrix operations."""

import os
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Callable, Any, Optional, Union
import multiprocessing as mp
import time

from ..config.settings import Config
from ..logging.setup import get_logger

logger = get_logger(__name__)


class MemoryMonitor:
    """Monitor memory usage and enforce limits."""
    
    def __init__(self, limit_mb: Optional[int] = None):
        """
        Initialize memory monitor.
        
        Args:
            limit_mb: Memory limit in MB (uses config if None)
        """
        self.limit_mb = limit_mb or Config.memory_limit_mb
        self.process = psutil.Process()
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        memory_info = self.process.memory_info()
        return memory_info.rss / 1024 / 1024
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        current_usage = self.get_memory_usage_mb()
        if current_usage > self.limit_mb:
            logger.warning(f"Memory usage ({current_usage:.1f} MB) exceeds limit ({self.limit_mb} MB)")
            return False
        return True
    
    def enforce_memory_limit(self) -> None:
        """Enforce memory limit by raising exception if exceeded."""
        if not self.check_memory_limit():
            raise MemoryError(f"Memory usage exceeds limit of {self.limit_mb} MB")


class ParallelProcessor:
    """Handle parallel processing of matrix operations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of workers (uses config if None)
        """
        self.max_workers = max_workers or Config.max_workers
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self._lock = threading.Lock()
    
    def get_thread_pool(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor."""
        if self.thread_pool is None:
            with self._lock:
                if self.thread_pool is None:
                    self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
                    logger.debug(f"Created thread pool with {self.max_workers} workers")
        return self.thread_pool
    
    def get_process_pool(self) -> ProcessPoolExecutor:
        """Get or create process pool executor."""
        if self.process_pool is None:
            with self._lock:
                if self.process_pool is None:
                    # Use fewer processes than threads for CPU-bound tasks
                    process_workers = min(self.max_workers, mp.cpu_count())
                    self.process_pool = ProcessPoolExecutor(max_workers=process_workers)
                    logger.debug(f"Created process pool with {process_workers} workers")
        return self.process_pool
    
    def parallel_map(self, func: Callable, items: List[Any], use_processes: bool = False) -> List[Any]:
        """
        Execute function in parallel on list of items.
        
        Args:
            func: Function to execute
            items: List of items to process
            use_processes: Use processes instead of threads
            
        Returns:
            List of results
        """
        if not Config.parallel_processing or len(items) < 2:
            # Fall back to sequential processing
            return [func(item) for item in items]
        
        try:
            if use_processes:
                executor = self.get_process_pool()
            else:
                executor = self.get_thread_pool()
            
            start_time = time.time()
            results = list(executor.map(func, items))
            end_time = time.time()
            
            logger.debug(f"Parallel execution completed in {end_time - start_time:.3f}s")
            return results
            
        except Exception as e:
            logger.warning(f"Parallel execution failed: {str(e)}, falling back to sequential")
            return [func(item) for item in items]
    
    def shutdown(self) -> None:
        """Shutdown executor pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None
        
        logger.debug("Parallel processor shutdown complete")


class PerformanceProfiler:
    """Profile performance of matrix operations."""
    
    def __init__(self):
        self.operation_times: List[tuple] = []
        self.memory_monitor = MemoryMonitor()
    
    def profile_operation(self, operation_name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Profile a matrix operation.
        
        Args:
            operation_name: Name of the operation
            func: Function to execute
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Result of function execution
        """
        start_time = time.time()
        start_memory = self.memory_monitor.get_memory_usage_mb()
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = self.memory_monitor.get_memory_usage_mb()
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.operation_times.append((
                operation_name,
                execution_time,
                start_memory,
                end_memory,
                memory_delta
            ))
            
            logger.debug(f"Operation '{operation_name}' completed in {execution_time:.3f}s, "
                        f"memory: {start_memory:.1f} -> {end_memory:.1f} MB ({memory_delta:+.1f} MB)")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"Operation '{operation_name}' failed after {execution_time:.3f}s: {str(e)}")
            raise
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        if not self.operation_times:
            return {}
        
        times = [op[1] for op in self.operation_times]
        memory_deltas = [op[4] for op in self.operation_times]
        
        return {
            'total_operations': len(self.operation_times),
            'total_time': sum(times),
            'average_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'total_memory_delta': sum(memory_deltas),
            'average_memory_delta': sum(memory_deltas) / len(memory_deltas),
            'current_memory_mb': self.memory_monitor.get_memory_usage_mb()
        }
    
    def clear_stats(self) -> None:
        """Clear performance statistics."""
        self.operation_times.clear()
        logger.debug("Performance statistics cleared")


# Global instances
memory_monitor = MemoryMonitor()
parallel_processor = ParallelProcessor()
performance_profiler = PerformanceProfiler()


def performance_decorator(operation_name: str):
    """Decorator to profile function performance."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return performance_profiler.profile_operation(operation_name, func, *args, **kwargs)
        return wrapper
    return decorator


def memory_limit_decorator(func: Callable) -> Callable:
    """Decorator to enforce memory limits."""
    def wrapper(*args, **kwargs):
        memory_monitor.enforce_memory_limit()
        result = func(*args, **kwargs)
        memory_monitor.enforce_memory_limit()
        return result
    return wrapper