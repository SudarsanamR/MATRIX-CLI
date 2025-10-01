"""
Matrix Calculator CLI - A powerful command-line matrix calculator.

This package provides comprehensive matrix operations with support for:
- Basic operations (add, subtract, multiply, transpose, inverse, determinant)
- Advanced decompositions (LU, QR, SVD, Cholesky)
- Matrix properties and analysis
- Multiple file formats (CSV, JSON, LaTeX, NumPy, MATLAB)
- Symbolic computation with SymPy
- Interactive and command-line interfaces
"""

__version__ = "2.3.0"
__author__ = "Matrix Calculator Team"

from .core.matrix import Matrix
from .core.manager import MatrixManager
from .config.settings import Config

__all__ = ["Matrix", "MatrixManager", "Config"]
