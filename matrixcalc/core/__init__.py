"""Core matrix operations and data structures."""

from .matrix import Matrix
from .manager import MatrixManager
from .backends import NumericBackend, SymbolicBackend, BackendFactory

__all__ = ["Matrix", "MatrixManager", "NumericBackend", "SymbolicBackend", "BackendFactory"]
