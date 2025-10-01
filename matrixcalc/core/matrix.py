"""Core Matrix class with dual backend support."""

import numpy as np
import sympy as sp
from typing import List, Union, Optional, Tuple, Any
import copy

# Imports will be done when needed to avoid circular imports


class Matrix:
    """
    A class representing a matrix with support for symbolic computations.
    
    This class provides various matrix operations including addition, subtraction,
    multiplication, transpose, determinant, inverse, eigenvalues, and more.
    All elements are stored as sympy expressions to support symbolic computations.
    
    Attributes:
        data (List[List[sp.Expr]]): The matrix elements as sympy expressions
        rows (int): Number of rows in the matrix
        cols (int): Number of columns in the matrix
        backend (BackendInterface): Backend for operations (numeric or symbolic)
    """
    
    def __init__(self, data: List[List[Union[int, float, str, sp.Expr]]]) -> None:
        """
        Initialize a matrix with the given data.
        
        Args:
            data: A 2D list containing matrix elements. Elements can be numbers,
                 strings representing mathematical expressions, or sympy expressions.
        
        Raises:
            ValueError: If data is empty or rows have unequal lengths.
            SecurityError: If any element contains dangerous patterns.
        """
        if not data or not all(len(row) == len(data[0]) for row in data):
            raise ValueError("Data must be a non-empty 2D list with equal row lengths.")
        
        # Check matrix size and warn if too large
        from ..config.settings import Config
        if len(data) > Config.matrix_size_warning_threshold or len(data[0]) > Config.matrix_size_warning_threshold:
            from ..logging.setup import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Large matrix created: {len(data)}x{len(data[0])}")
        
        # Convert every element using sp.sympify to allow symbolic expressions
        # Preprocess strings to allow ^ for power and e for Euler's number
        self.data = []
        for row in data:
            processed_row = []
            for elem in row:
                if isinstance(elem, str):
                    # Validate expression for security
                    from ..security.validation import expression_validator
                    validated_expr = expression_validator.validate_expression(elem)
                    processed_row.append(sp.sympify(validated_expr))
                else:
                    processed_row.append(sp.sympify(elem))
            self.data.append(processed_row)
        
        self.rows = len(data)
        self.cols = len(data[0])
        
        # Choose backend based on data type
        from .backends import BackendFactory
        self.backend = BackendFactory.create_backend(self.data)
        
        # Initialize cache
        self._det_cache: Optional[sp.Expr] = None
        self._eigenval_cache: Optional[List[sp.Expr]] = None
        self._rank_cache: Optional[int] = None
        self._inverse_cache: Optional['Matrix'] = None
    
    def _invalidate_caches(self) -> None:
        """Invalidate all cached computations."""
        self._det_cache = None
        self._eigenval_cache = None
        self._rank_cache = None
        self._inverse_cache = None
    
    def __str__(self) -> str:
        """
        Return a string representation of the matrix.
        
        Returns:
            A formatted string showing each element with its position.
        """
        s = ""
        for i in range(self.rows):
            row_str = "\t".join(f"a{i+1}{j+1} = {self._format_element(self.data[i][j])}" 
                               for j in range(self.cols))
            s += row_str + "\n"
        return s
    
    def _format_element(self, elem: sp.Expr) -> str:
        """Format element for display."""
        try:
            # Convert ** back to ^ for display
            elem_str = str(sp.pretty(elem))
            return elem_str.replace('**', '^')
        except Exception:
            return str(elem)
    
    def is_square(self) -> bool:
        """Check if the matrix is square (equal number of rows and columns)."""
        return self.rows == self.cols
    
    def add(self, other: 'Matrix') -> 'Matrix':
        """Add this matrix with another matrix."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Addition requires matrices of the same dimensions.")
        
        try:
            result = self.backend.add(self.data, other.data)
            return Matrix(result.tolist() if hasattr(result, 'tolist') else result)
        except Exception as e:
            from ..logging.setup import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error in matrix addition: {str(e)}")
            raise ValueError(f"Error in matrix addition: {str(e)}")
    
    def subtract(self, other: 'Matrix') -> 'Matrix':
        """Subtract another matrix from this matrix."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Subtraction requires matrices of the same dimensions.")
        
        try:
            result = self.backend.subtract(self.data, other.data)
            return Matrix(result.tolist() if hasattr(result, 'tolist') else result)
        except Exception as e:
            logger.error(f"Error in matrix subtraction: {str(e)}")
            raise ValueError(f"Error in matrix subtraction: {str(e)}")
    
    def multiply(self, other: Union['Matrix', int, float, sp.Number]) -> 'Matrix':
        """Multiply this matrix by another matrix or a scalar."""
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("For matrix multiplication, the number of columns in the first matrix must equal the number of rows in the second.")
            
            try:
                result = self.backend.multiply(self.data, other.data)
                return Matrix(result.tolist() if hasattr(result, 'tolist') else result)
            except Exception as e:
                logger.error(f"Error in matrix multiplication: {str(e)}")
                raise ValueError(f"Error in matrix multiplication: {str(e)}")
        
        elif isinstance(other, (int, float, sp.Number)):
            # Scalar multiplication
            result = [[self.data[i][j] * other for j in range(self.cols)] for i in range(self.rows)]
            return Matrix(result)
        else:
            raise ValueError("Multiplication is only supported with a matrix or a scalar number.")
    
    def transpose(self) -> 'Matrix':
        """Compute the transpose of this matrix."""
        try:
            result = self.backend.transpose(self.data)
            return Matrix(result.tolist() if hasattr(result, 'tolist') else result)
        except Exception as e:
            logger.error(f"Error computing transpose: {str(e)}")
            raise ValueError(f"Error computing transpose: {str(e)}")
    
    def trace(self) -> Union[sp.Expr, float]:
        """Compute the trace of this matrix (sum of diagonal elements)."""
        if not self.is_square():
            raise ValueError("Trace is defined only for square matrices.")
        
        try:
            return self.backend.trace(self.data)
        except Exception as e:
            logger.error(f"Error computing trace: {str(e)}")
            raise ValueError(f"Error computing trace: {str(e)}")
    
    def determinant(self) -> Union[sp.Expr, float]:
        """Compute the determinant of this matrix."""
        if not self.is_square():
            raise ValueError("Determinant is defined only for square matrices.")
        
        # Use cache if available and enabled
        from ..config.settings import Config
        if Config.enable_caching and self._det_cache is not None:
            return self._det_cache
        
        try:
            result = self.backend.determinant(self.data)
            if Config.enable_caching:
                self._det_cache = result
            return result
        except Exception as e:
            logger.error(f"Error computing determinant: {str(e)}")
            raise ValueError(f"Error computing determinant: {str(e)}")
    
    def inverse(self) -> 'Matrix':
        """Compute the inverse of this matrix."""
        if not self.is_square():
            raise ValueError("Inverse is defined only for square matrices.")
        
        # Use cache if available and enabled
        from ..config.settings import Config
        if Config.enable_caching and self._inverse_cache is not None:
            return self._inverse_cache
        
        try:
            # Check if determinant is zero (matrix is singular)
            det = self.determinant()
            if det == 0:
                raise ValueError("Matrix is singular (determinant is zero).")
            
            result = self.backend.inverse(self.data)
            inverse_matrix = Matrix(result.tolist() if hasattr(result, 'tolist') else result)
            
            if Config.enable_caching:
                self._inverse_cache = inverse_matrix
            
            return inverse_matrix
        except Exception as e:
            logger.error(f"Error computing inverse: {str(e)}")
            raise ValueError(f"Error computing inverse: {str(e)}")
    
    def eigenvalues(self, numeric: bool = False) -> List[Union[sp.Expr, complex]]:
        """Compute the eigenvalues of this matrix."""
        if not self.is_square():
            raise ValueError("Eigenvalues are defined only for square matrices.")
        
        # Use cache if available and enabled
        from ..config.settings import Config
        if Config.enable_caching and self._eigenval_cache is not None:
            return self._eigenval_cache
        
        try:
            result = self.backend.eigenvalues(self.data)
            if Config.enable_caching:
                self._eigenval_cache = result
            return result
        except Exception as e:
            logger.error(f"Error computing eigenvalues: {str(e)}")
            raise ValueError(f"Error computing eigenvalues: {str(e)}")
    
    def rank(self) -> int:
        """Compute the rank of this matrix."""
        # Use cache if available and enabled
        from ..config.settings import Config
        if Config.enable_caching and self._rank_cache is not None:
            return self._rank_cache
        
        try:
            rank_val = sp.Matrix(self.data).rank()
            if Config.enable_caching:
                self._rank_cache = rank_val
            return rank_val
        except Exception as e:
            logger.error(f"Error computing rank: {str(e)}")
            raise ValueError(f"Error computing rank: {str(e)}")
    
    def condition_number(self) -> float:
        """Compute the condition number of this matrix."""
        if not self.is_square():
            raise ValueError("Condition number is defined only for square matrices.")
        
        try:
            # Convert to numpy for condition number computation
            numeric_data = [[float(sp.N(elem)) for elem in row] for row in self.data]
            np_matrix = np.array(numeric_data)
            return float(np.linalg.cond(np_matrix))
        except Exception as e:
            logger.error(f"Error computing condition number: {str(e)}")
            raise ValueError(f"Error computing condition number: {str(e)}")
    
    def norm(self, norm_type: str = 'frobenius') -> float:
        """Compute the norm of this matrix."""
        try:
            # Convert to numpy for norm computation
            numeric_data = [[float(sp.N(elem)) for elem in row] for row in self.data]
            np_matrix = np.array(numeric_data)
            
            if norm_type.lower() == 'frobenius':
                return float(np.linalg.norm(np_matrix, 'fro'))
            elif norm_type.lower() == 'l1':
                return float(np.linalg.norm(np_matrix, 1))
            elif norm_type.lower() == 'l2':
                return float(np.linalg.norm(np_matrix, 2))
            elif norm_type.lower() == 'inf':
                return float(np.linalg.norm(np_matrix, np.inf))
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")
        except Exception as e:
            logger.error(f"Error computing norm: {str(e)}")
            raise ValueError(f"Error computing norm: {str(e)}")
    
    def is_symmetric(self) -> bool:
        """Check if the matrix is symmetric (equal to its transpose)."""
        if not self.is_square():
            return False
        try:
            return self.data == self.transpose().data
        except Exception:
            return False
    
    def is_orthogonal(self) -> bool:
        """Check if the matrix is orthogonal (its transpose equals its inverse)."""
        if not self.is_square():
            return False
        try:
            # Check if A * A^T = I
            product = self.multiply(self.transpose())
            # Tolerance-aware comparison
            from ..config.settings import Config
            numeric_product = [[float(sp.N(elem)) for elem in row] for row in product.data]
            identity = [[1.0 if i == j else 0.0 for j in range(self.rows)] for i in range(self.rows)]
            import numpy as np
            return bool(np.allclose(np.array(numeric_product), np.array(identity), atol=Config.numeric_atol, rtol=Config.numeric_rtol))
        except Exception:
            return False
    
    def is_positive_definite(self) -> bool:
        """Check if the matrix is positive definite."""
        if not self.is_square():
            return False
        try:
            # Convert to numpy for positive definite check
            numeric_data = [[float(sp.N(elem)) for elem in row] for row in self.data]
            np_matrix = np.array(numeric_data)
            return np.all(np.linalg.eigvals(np_matrix) > 0)
        except Exception:
            return False
    
    def lu_decomposition(self) -> Tuple['Matrix', 'Matrix']:
        """Compute the LU decomposition."""
        if not self.is_square():
            raise ValueError("LU decomposition is defined only for square matrices.")
        
        try:
            sym_matrix = sp.Matrix(self.data)
            L, U, _ = sym_matrix.LUdecomposition()
            return Matrix(L.tolist()), Matrix(U.tolist())
        except Exception as e:
            logger.error(f"Error computing LU decomposition: {str(e)}")
            raise ValueError(f"Error computing LU decomposition: {str(e)}")
    
    def qr_decomposition(self) -> Tuple['Matrix', 'Matrix']:
        """Compute the QR decomposition."""
        try:
            sym_matrix = sp.Matrix(self.data)
            Q, R = sym_matrix.QRdecomposition()
            return Matrix(Q.tolist()), Matrix(R.tolist())
        except Exception as e:
            logger.error(f"Error computing QR decomposition: {str(e)}")
            raise ValueError(f"Error computing QR decomposition: {str(e)}")
    
    def svd(self) -> Tuple['Matrix', 'Matrix', 'Matrix']:
        """Compute the Singular Value Decomposition."""
        try:
            # Convert to numpy for SVD computation
            numeric_data = [[float(sp.N(elem)) for elem in row] for row in self.data]
            np_matrix = np.array(numeric_data)
            U, s, Vt = np.linalg.svd(np_matrix)
            S = np.zeros((U.shape[1], Vt.shape[0]))
            S[:len(s), :len(s)] = np.diag(s)
            return Matrix(U.tolist()), Matrix(S.tolist()), Matrix(Vt.T.tolist())
        except Exception as e:
            logger.error(f"Error computing SVD: {str(e)}")
            raise ValueError(f"Error computing SVD: {str(e)}")
    
    def cholesky_decomposition(self) -> 'Matrix':
        """Compute the Cholesky decomposition."""
        if not self.is_square():
            raise ValueError("Cholesky decomposition is defined only for square matrices.")
        
        try:
            # Convert to numpy for Cholesky decomposition
            numeric_data = [[float(sp.N(elem)) for elem in row] for row in self.data]
            np_matrix = np.array(numeric_data)
            L = np.linalg.cholesky(np_matrix)
            return Matrix(L.tolist())
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is not positive definite")
        except Exception as e:
            logger.error(f"Error computing Cholesky decomposition: {str(e)}")
            raise ValueError(f"Error computing Cholesky decomposition: {str(e)}")
    
    def preview(self, max_rows: int = 5, max_cols: int = 5) -> str:
        """Show a preview of the matrix for large matrices."""
        if self.rows <= max_rows and self.cols <= max_cols:
            return str(self)
        
        s = f"Matrix Preview ({self.rows}x{self.cols}):\n"
        
        # Show top rows
        for i in range(min(max_rows, self.rows)):
            row_str = "\t".join(f"a{i+1}{j+1} = {self._format_element(self.data[i][j])}" 
                               for j in range(min(max_cols, self.cols)))
            if self.cols > max_cols:
                row_str += "\t..."
            s += row_str + "\n"
        
        if self.rows > max_rows:
            s += "...\n"
        
        return s
    
    def __eq__(self, other: 'Matrix') -> bool:
        """Check if two matrices are equal."""
        if not isinstance(other, Matrix):
            return False
        if self.rows != other.rows or self.cols != other.cols:
            return False
        
        try:
            return self.data == other.data
        except Exception:
            return False
    
    def __copy__(self) -> 'Matrix':
        """Create a copy of the matrix."""
        return Matrix([[elem for elem in row] for row in self.data])
    
    def __deepcopy__(self, memo) -> 'Matrix':
        """Create a deep copy of the matrix."""
        return Matrix([[copy.deepcopy(elem, memo) for elem in row] for row in self.data])
