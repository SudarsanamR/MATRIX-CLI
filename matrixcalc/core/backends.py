"""Dual backend system for numeric and symbolic operations."""

import numpy as np
import sympy as sp
from typing import Union, List, Tuple, Optional, Any
import json
import hashlib
from abc import ABC, abstractmethod

from ..logging.setup import get_logger
from ..caching.cache import cache_manager

logger = get_logger(__name__)


class BackendInterface(ABC):
    """Abstract interface for matrix operation backends."""
    
    @abstractmethod
    def add(self, a: Any, b: Any) -> Any:
        """Add two matrices."""
        pass
    
    @abstractmethod
    def subtract(self, a: Any, b: Any) -> Any:
        """Subtract two matrices."""
        pass
    
    @abstractmethod
    def multiply(self, a: Any, b: Any) -> Any:
        """Multiply two matrices."""
        pass
    
    @abstractmethod
    def transpose(self, matrix: Any) -> Any:
        """Transpose matrix."""
        pass
    
    @abstractmethod
    def inverse(self, matrix: Any) -> Any:
        """Compute matrix inverse."""
        pass
    
    @abstractmethod
    def determinant(self, matrix: Any) -> Any:
        """Compute determinant."""
        pass
    
    @abstractmethod
    def trace(self, matrix: Any) -> Any:
        """Compute trace."""
        pass
    
    @abstractmethod
    def eigenvalues(self, matrix: Any) -> List[Any]:
        """Compute eigenvalues."""
        pass


class NumericBackend(BackendInterface):
    """Numeric backend using NumPy/SciPy for fast numerical operations."""
    
    def __init__(self):
        self.cache = cache_manager.get_cache('numeric_ops', max_size=50, ttl_seconds=1800)
    
    def _to_numpy(self, data: Union[List, np.ndarray]) -> np.ndarray:
        """Convert data to NumPy array."""
        if isinstance(data, np.ndarray):
            return data
        return np.array(data, dtype=float)
    
    def _key(self, prefix: str, *objs: Any) -> str:
        payload = json.dumps([prefix] + [obj.tolist() if hasattr(obj, 'tolist') else obj for obj in objs], default=str, sort_keys=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def add(self, a: Union[List, np.ndarray], b: Union[List, np.ndarray]) -> np.ndarray:
        """Add two matrices using NumPy."""
        cache_key = self._key('add', a, b)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        np_a = self._to_numpy(a)
        np_b = self._to_numpy(b)
        result = np_a + np_b
        
        self.cache.put(cache_key, result)
        return result
    
    def subtract(self, a: Union[List, np.ndarray], b: Union[List, np.ndarray]) -> np.ndarray:
        """Subtract two matrices using NumPy."""
        cache_key = self._key('subtract', a, b)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        np_a = self._to_numpy(a)
        np_b = self._to_numpy(b)
        result = np_a - np_b
        
        self.cache.put(cache_key, result)
        return result
    
    def multiply(self, a: Union[List, np.ndarray], b: Union[List, np.ndarray]) -> np.ndarray:
        """Multiply two matrices using NumPy."""
        cache_key = self._key('multiply', a, b)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        np_a = self._to_numpy(a)
        np_b = self._to_numpy(b)
        result = np.dot(np_a, np_b)
        
        self.cache.put(cache_key, result)
        return result
    
    def transpose(self, matrix: Union[List, np.ndarray]) -> np.ndarray:
        """Transpose matrix using NumPy."""
        cache_key = self._key('transpose', matrix)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        np_matrix = self._to_numpy(matrix)
        result = np_matrix.T
        
        self.cache.put(cache_key, result)
        return result
    
    def inverse(self, matrix: Union[List, np.ndarray]) -> np.ndarray:
        """Compute matrix inverse using NumPy."""
        cache_key = self._key('inverse', matrix)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        np_matrix = self._to_numpy(matrix)
        result = np.linalg.inv(np_matrix)
        
        self.cache.put(cache_key, result)
        return result
    
    def determinant(self, matrix: Union[List, np.ndarray]) -> float:
        """Compute determinant using NumPy."""
        cache_key = self._key('determinant', matrix)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        np_matrix = self._to_numpy(matrix)
        result = float(np.linalg.det(np_matrix))
        
        self.cache.put(cache_key, result)
        return result
    
    def trace(self, matrix: Union[List, np.ndarray]) -> float:
        """Compute trace using NumPy."""
        cache_key = self._key('trace', matrix)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        np_matrix = self._to_numpy(matrix)
        result = float(np.trace(np_matrix))
        
        self.cache.put(cache_key, result)
        return result
    
    def eigenvalues(self, matrix: Union[List, np.ndarray]) -> List[complex]:
        """Compute eigenvalues using NumPy."""
        cache_key = self._key('eigenvalues', matrix)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        np_matrix = self._to_numpy(matrix)
        eigenvals = np.linalg.eigvals(np_matrix)
        result = eigenvals.tolist()
        
        self.cache.put(cache_key, result)
        return result


class SymbolicBackend(BackendInterface):
    """Symbolic backend using SymPy for exact symbolic operations."""
    
    def __init__(self):
        self.cache = cache_manager.get_cache('symbolic_ops', max_size=30, ttl_seconds=3600)
    
    def _to_sympy(self, data: Union[List, Any]) -> sp.Matrix:
        """Convert data to SymPy Matrix."""
        if isinstance(data, sp.Matrix):
            return data
        return sp.Matrix(data)
    
    def _key(self, prefix: str, *objs: Any) -> str:
        payload = json.dumps([prefix] + [str(obj) for obj in objs], sort_keys=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def add(self, a: Union[List, Any], b: Union[List, Any]) -> sp.Matrix:
        """Add two matrices using SymPy."""
        cache_key = self._key('sym_add', a, b)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        sp_a = self._to_sympy(a)
        sp_b = self._to_sympy(b)
        result = sp_a + sp_b
        
        self.cache.put(cache_key, result)
        return result
    
    def subtract(self, a: Union[List, Any], b: Union[List, Any]) -> sp.Matrix:
        """Subtract two matrices using SymPy."""
        cache_key = self._key('sym_subtract', a, b)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        sp_a = self._to_sympy(a)
        sp_b = self._to_sympy(b)
        result = sp_a - sp_b
        
        self.cache.put(cache_key, result)
        return result
    
    def multiply(self, a: Union[List, Any], b: Union[List, Any]) -> sp.Matrix:
        """Multiply two matrices using SymPy."""
        cache_key = self._key('sym_multiply', a, b)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        sp_a = self._to_sympy(a)
        sp_b = self._to_sympy(b)
        result = sp_a * sp_b
        
        self.cache.put(cache_key, result)
        return result
    
    def transpose(self, matrix: Union[List, Any]) -> sp.Matrix:
        """Transpose matrix using SymPy."""
        cache_key = self._key('sym_transpose', matrix)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        sp_matrix = self._to_sympy(matrix)
        result = sp_matrix.T
        
        self.cache.put(cache_key, result)
        return result
    
    def inverse(self, matrix: Union[List, Any]) -> sp.Matrix:
        """Compute matrix inverse using SymPy."""
        cache_key = self._key('sym_inverse', matrix)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        sp_matrix = self._to_sympy(matrix)
        result = sp_matrix.inv()
        
        self.cache.put(cache_key, result)
        return result
    
    def determinant(self, matrix: Union[List, Any]) -> sp.Expr:
        """Compute determinant using SymPy."""
        cache_key = self._key('sym_determinant', matrix)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        sp_matrix = self._to_sympy(matrix)
        result = sp_matrix.det()
        
        self.cache.put(cache_key, result)
        return result
    
    def trace(self, matrix: Union[List, Any]) -> sp.Expr:
        """Compute trace using SymPy."""
        cache_key = self._key('sym_trace', matrix)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        sp_matrix = self._to_sympy(matrix)
        result = sp_matrix.trace()
        
        self.cache.put(cache_key, result)
        return result
    
    def eigenvalues(self, matrix: Union[List, Any]) -> List[sp.Expr]:
        """Compute eigenvalues using SymPy."""
        cache_key = self._key('sym_eigenvalues', matrix)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        sp_matrix = self._to_sympy(matrix)
        eigenvals = sp_matrix.eigenvals()
        result = list(eigenvals.keys())
        
        self.cache.put(cache_key, result)
        return result


class BackendFactory:
    """Factory for creating appropriate backends based on data type."""
    
    @staticmethod
    def create_backend(data: Union[List, np.ndarray, sp.Matrix]) -> BackendInterface:
        """
        Create appropriate backend based on data type.
        
        Args:
            data: Matrix data to analyze
            
        Returns:
            Appropriate backend instance
        """
        # Check if data contains symbolic expressions
        if _contains_symbolic_data(data):
            logger.debug("Using symbolic backend for symbolic data")
            return SymbolicBackend()
        else:
            logger.debug("Using numeric backend for numeric data")
            return NumericBackend()
    
    @staticmethod
    def force_numeric_backend() -> BackendInterface:
        """Force creation of numeric backend."""
        return NumericBackend()
    
    @staticmethod
    def force_symbolic_backend() -> BackendInterface:
        """Force creation of symbolic backend."""
        return SymbolicBackend()


def _contains_symbolic_data(data: Union[List, np.ndarray, sp.Matrix]) -> bool:
    """Check if data contains symbolic expressions."""
    if isinstance(data, sp.Matrix):
        return True
    
    if isinstance(data, np.ndarray):
        data = data.tolist()
    
    if not isinstance(data, list):
        return False
    
    # Check if any element is a SymPy expression
    for row in data:
        if isinstance(row, list):
            for elem in row:
                if isinstance(elem, (sp.Expr, sp.Symbol, sp.Number)):
                    return True
                if isinstance(elem, str) and any(c.isalpha() for c in str(elem)):
                    return True
        else:
            if isinstance(row, (sp.Expr, sp.Symbol, sp.Number)):
                return True
            if isinstance(row, str) and any(c.isalpha() for c in str(row)):
                return True
    
    return False
