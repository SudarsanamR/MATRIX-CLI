import copy
import math
import os
import sys
import json
import csv
import argparse
import signal
import logging
import time
import re
from typing import List, Union, Optional, Tuple, Dict, Any
import numpy as np
import sympy as sp
from colorama import init, Fore, Back, Style
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm
import scipy.io as sio

# Initialize colorama for cross-platform colored output
init(autoreset=True)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('matrix_calculator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration Class
# -----------------------------
class Config:
    """Global configuration settings."""
    precision: int = 4
    default_export_format: str = 'csv'
    colored_output: bool = True
    auto_save: bool = False
    save_directory: str = './matrices'
    show_progress: bool = True
    max_history_size: int = 20
    enable_caching: bool = True
    matrix_size_warning_threshold: int = 1000
    recent_files_list: List[str] = []
    log_level: str = 'INFO'
    
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
            logger.info(f"Configuration loaded from {filename}")
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
                'log_level': cls.log_level
            }
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2)
            print_success(f"Configuration saved to {filename}")
            logger.info(f"Configuration saved to {filename}")
        except Exception as e:
            print_error(f"Error saving configuration: {str(e)}")
            logger.error(f"Error saving configuration: {str(e)}")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate all configuration values."""
        try:
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
            return True
        except ValueError as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise

# -----------------------------
# Utility Functions
# -----------------------------
def print_success(message: str) -> None:
    """Print success message in green."""
    if Config.colored_output:
        print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")
    else:
        print(f"✓ {message}")

def print_error(message: str) -> None:
    """Print error message in red."""
    if Config.colored_output:
        print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")
    else:
        print(f"✗ {message}")

def print_warning(message: str) -> None:
    """Print warning message in yellow."""
    if Config.colored_output:
        print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")
    else:
        print(f"⚠ {message}")

def print_info(message: str) -> None:
    """Print info message in cyan."""
    if Config.colored_output:
        print(f"{Fore.CYAN}ℹ {message}{Style.RESET_ALL}")
    else:
        print(f"ℹ {message}")

def print_header(message: str) -> None:
    """Print header message in bold."""
    if Config.colored_output:
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{message}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'=' * len(message)}{Style.RESET_ALL}")
    else:
        print(f"\n{message}")
        print("=" * len(message))

def clear_screen() -> None:
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def confirm_action(message: str) -> bool:
    """Ask user for confirmation."""
    while True:
        response = input(f"{message} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        print_warning("Please enter 'y' or 'n'.")

def signal_handler(sig, frame) -> None:
    """Handle Ctrl+C gracefully."""
    print_info("\n\nGracefully exiting... Goodbye!")
    logger.info("Application terminated by user")
    sys.exit(0)

def validate_matrix_input(elements: List[str]) -> bool:
    """
    Validate matrix elements for batch input.
    
    Args:
        elements: List of element strings to validate.
    
    Returns:
        True if all elements are valid, False otherwise.
    """
    for elem in elements:
        try:
            sp.sympify(preprocess_expression(elem.strip()))
        except (sp.SympifyError, ValueError):
            return False
    return True

def check_file_exists(filename: str) -> bool:
    """
    Check if a file exists and is readable.
    
    Args:
        filename: Path to the file.
    
    Returns:
        True if file exists and is readable, False otherwise.
    """
    if not os.path.exists(filename):
        return False
    if not os.access(filename, os.R_OK):
        return False
    return True

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def ensure_file_extension(filename: str, default_ext: str) -> str:
    """
    Ensure a filename has the specified extension.
    
    Args:
        filename: The filename to check.
        default_ext: The default extension to add if missing.
    
    Returns:
        Filename with extension.
    """
    if not filename:
        raise ValueError("Filename cannot be empty")
    
    _, ext = os.path.splitext(filename)
    if not ext:
        return filename + default_ext
    return filename

def preprocess_expression(expr: str) -> str:
    """
    Preprocess mathematical expressions for user convenience.
    Converts ^ to ** for power and handles e for Euler's number.
    
    Args:
        expr: The expression string to preprocess.
    
    Returns:
        Preprocessed expression string.
    
    Raises:
        ValueError: If expression contains disallowed functions or patterns.
    """
    # Whitelist of allowed functions
    allowed_functions = {
        'sin', 'cos', 'tan', 'sec', 'csc', 'cot',
        'asin', 'acos', 'atan', 'asec', 'acsc', 'acot',
        'sinh', 'cosh', 'tanh', 'sech', 'csch', 'coth',
        'asinh', 'acosh', 'atanh', 'asech', 'acsch', 'acoth',
        'exp', 'log', 'ln', 'sqrt', 'abs', 'ceil', 'floor',
        'factorial', 'gamma', 'beta'
    }
    
    # Check for potentially dangerous patterns
    dangerous_patterns = [
        r'__',  # Double underscore (potential dunder methods)
        r'import',  # Import statements
        r'exec',  # Exec statements
        r'eval',  # Eval statements
        r'open',  # File operations
        r'file',  # File operations
        r'input',  # Input operations
        r'raw_input',  # Raw input operations
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, expr, re.IGNORECASE):
            raise ValueError(f"Expression contains potentially dangerous pattern: {pattern}")
    
    # Replace ^ with ** for power
    expr = expr.replace('^', '**')
    
    # Replace standalone 'e' with 'E' (Euler's number)
    # But be careful not to replace 'e' in function names like 'exp'
    # Match 'e' that is not part of a word (standalone or with operators)
    expr = re.sub(r'\be\b', 'E', expr)
    
    return expr

def format_expression_for_display(expr: str) -> str:
    """
    Format SymPy expressions for user-friendly display.
    Converts ** to ^ for power display.
    
    Args:
        expr: The expression string to format.
    
    Returns:
        Formatted expression string.
    """
    # Convert ** to ^ for display
    expr = str(expr).replace('**', '^')
    return expr

# -----------------------------
# Matrix Class
# -----------------------------
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
    
    Examples:
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> print(m)
        a11 = 1    a12 = 2
        a21 = 3    a22 = 4
    """
    
    def __init__(self, data: List[List[Union[int, float, str, sp.Expr]]]) -> None:
        """
        Initialize a matrix with the given data.
        
        Args:
            data: A 2D list containing matrix elements. Elements can be numbers,
                 strings representing mathematical expressions, or sympy expressions.
        
        Raises:
            ValueError: If data is empty or rows have unequal lengths.
            SympifyError: If any element cannot be converted to a sympy expression.
        """
        if not data or not all(len(row) == len(data[0]) for row in data):
            raise ValueError("Data must be a non-empty 2D list with equal row lengths.")
        
        # Check matrix size and warn if too large
        if len(data) > Config.matrix_size_warning_threshold or len(data[0]) > Config.matrix_size_warning_threshold:
            print_warning(f"Large matrix detected ({len(data)}x{len(data[0])}). Performance may be affected.")
            logger.warning(f"Large matrix created: {len(data)}x{len(data[0])}")
        
        # Convert every element using sp.sympify to allow symbolic expressions
        # Preprocess strings to allow ^ for power and e for Euler's number
        self.data = [[sp.sympify(preprocess_expression(str(elem)) if isinstance(elem, str) else elem) 
                      for elem in row] for row in data]
        self.rows = len(data)
        self.cols = len(data[0])
        
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
        
        # Initialize cache
        self._det_cache: Optional[sp.Expr] = None
        self._eigenval_cache: Optional[List[sp.Expr]] = None
        self._rank_cache: Optional[int] = None
        self._inverse_cache: Optional['Matrix'] = None
    
    def __str__(self) -> str:
        """
        Return a string representation of the matrix.
        
        Returns:
            A formatted string showing each element with its position.
        """
        s = ""
        for i in range(self.rows):
            row_str = "\t".join(f"a{i+1}{j+1} = {format_expression_for_display(sp.pretty(self.data[i][j]))}" 
                               for j in range(self.cols))
            s += row_str + "\n"
        return s

    def is_square(self) -> bool:
        """
        Check if the matrix is square (equal number of rows and columns).
        
        Returns:
            True if the matrix is square, False otherwise.
        """
        return self.rows == self.cols

    def add(self, other: 'Matrix') -> 'Matrix':
        """
        Add this matrix with another matrix.
        
        Args:
            other: The matrix to add with this matrix.
        
        Returns:
            A new matrix containing the sum.
        
        Raises:
            ValueError: If matrices have different dimensions.
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Addition requires matrices of the same dimensions.")
        result = [[self.data[i][j] + other.data[i][j] for j in range(self.cols)]
                  for i in range(self.rows)]
        return Matrix(result)

    def subtract(self, other: 'Matrix') -> 'Matrix':
        """
        Subtract another matrix from this matrix.
        
        Args:
            other: The matrix to subtract from this matrix.
        
        Returns:
            A new matrix containing the difference.
        
        Raises:
            ValueError: If matrices have different dimensions.
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Subtraction requires matrices of the same dimensions.")
        result = [[self.data[i][j] - other.data[i][j] for j in range(self.cols)]
                  for i in range(self.rows)]
        return Matrix(result)

    def multiply(self, other: Union['Matrix', int, float, sp.Number]) -> 'Matrix':
        """
        Multiply this matrix by another matrix or a scalar.
        
        Args:
            other: Either a Matrix instance or a scalar number.
        
        Returns:
            A new matrix containing the product.
        
        Raises:
            ValueError: If matrix multiplication dimensions are incompatible.
        """
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("For matrix multiplication, the number of columns in the first matrix must equal the number of rows in the second.")
            result = [[sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                       for j in range(other.cols)] for i in range(self.rows)]
        elif isinstance(other, (int, float, sp.Number)):
            result = [[self.data[i][j] * other for j in range(self.cols)] for i in range(self.rows)]
        else:
            raise ValueError("Multiplication is only supported with a matrix or a scalar number.")
        return Matrix(result)

    def transpose(self) -> 'Matrix':
        """
        Compute the transpose of this matrix.
        
        Returns:
            A new matrix containing the transpose.
        """
        try:
            result = [[self.data[j][i] for j in range(self.rows)]
                      for i in range(self.cols)]
            return Matrix(result)
        except Exception as e:
            raise ValueError(f"Error computing transpose: {str(e)}")

    def trace(self) -> sp.Expr:
        """
        Compute the trace of this matrix (sum of diagonal elements).
        
        Returns:
            The trace as a sympy expression.
        
        Raises:
            ValueError: If the matrix is not square.
        """
        if not self.is_square():
            raise ValueError("Trace is defined only for square matrices.")
        try:
            return sum(self.data[i][i] for i in range(self.rows))
        except Exception as e:
            raise ValueError(f"Error computing trace: {str(e)}")

    def determinant(self) -> sp.Expr:
        """
        Compute the determinant of this matrix.
        
        Returns:
            The determinant as a sympy expression.
        
        Raises:
            ValueError: If the matrix is not square.
        """
        if not self.is_square():
            raise ValueError("Determinant is defined only for square matrices.")
        
        # Use cache if available and enabled
        if Config.enable_caching and self._det_cache is not None:
            return self._det_cache
        
        try:
            # Use sympy's built-in determinant computation
            det = sp.Matrix(self.data).det()
            if Config.enable_caching:
                self._det_cache = det
            return det
        except Exception as e:
            raise ValueError(f"Error computing determinant: {str(e)}")

    def inverse(self) -> 'Matrix':
        """
        Compute the inverse of this matrix.
        
        Returns:
            A new matrix containing the inverse.
        
        Raises:
            ValueError: If the matrix is not square or not invertible.
        """
        if not self.is_square():
            raise ValueError("Inverse is defined only for square matrices.")
        try:
            # Check if determinant is zero (matrix is singular)
            det = self.determinant()
            if det == 0:
                raise ValueError("Matrix is singular (determinant is zero).")
            
            inv = sp.Matrix(self.data).inv()
            return Matrix(inv.tolist())
        except sp.NonInvertibleMatrixError:
            raise ValueError("Matrix is not invertible.")
        except Exception as e:
            raise ValueError(f"Error computing inverse: {str(e)}")

    def eigenvalues(self, numeric: bool = False) -> List[sp.Expr]:
        """
        Compute the eigenvalues of this matrix.
        
        Args:
            numeric: If True, return numerical approximations of eigenvalues.
                    If False, return symbolic eigenvalues.
        
        Returns:
            List of eigenvalues as sympy expressions.
        
        Raises:
            ValueError: If the matrix is not square.
        """
        if not self.is_square():
            raise ValueError("Eigenvalues are defined only for square matrices.")
        try:
            sym_eigs = list(sp.Matrix(self.data).eigenvals().keys())
            if numeric:
                return [sp.N(e) for e in sym_eigs]
            return sym_eigs
        except Exception as e:
            raise ValueError(f"Error computing eigenvalues: {str(e)}")

    def characteristic_equation(self) -> sp.Expr:
        """
        Compute the characteristic polynomial of this matrix.
        
        Returns:
            The characteristic polynomial as a sympy expression.
        
        Raises:
            ValueError: If the matrix is not square.
        """
        if not self.is_square():
            raise ValueError("Characteristic equation is defined only for square matrices.")
        try:
            X = sp.symbols('X')
            A_sym = sp.Matrix(self.data)
            char_poly = sp.expand((X * sp.eye(self.rows) - A_sym).det())
            return char_poly
        except Exception as e:
            raise ValueError(f"Error computing characteristic equation: {str(e)}")

    def power(self, exponent: Union[int, float]) -> 'Matrix':
        """
        Raise this matrix to a real number exponent.
        
        Args:
            exponent: The real number exponent.
        
        Returns:
            A new matrix containing the result.
        
        Raises:
            ValueError: If the matrix is not square or if the operation is not defined.
        """
        if not self.is_square():
            raise ValueError("Matrix power is defined only for square matrices.")
        try:
            M = sp.Matrix(self.data)
            # Try diagonalization first
            try:
                P, D = M.diagonalize()
                # Raise each diagonal entry to the exponent
                diag_entries = [d**exponent for d in D.diagonal()]
                D_power = sp.diag(*diag_entries)
                M_power = P * D_power * P.inv()
                return Matrix(M_power.tolist())
            except sp.MatrixError:
                # If not diagonalizable, try using logarithm and exponential
                try:
                    M_log = sp.logm(M)
                    M_power = sp.exp(M_log * exponent)
                    return Matrix(M_power.tolist())
                except Exception as e:
                    raise ValueError(f"Matrix power for real exponent is not defined for this matrix: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error computing matrix power: {str(e)}")

    def is_symmetric(self) -> bool:
        """
        Check if the matrix is symmetric (equal to its transpose).
        
        Returns:
            True if the matrix is symmetric, False otherwise.
        """
        if not self.is_square():
            return False
        try:
            return self.data == self.transpose().data
        except Exception:
            return False

    def is_orthogonal(self) -> bool:
        """
        Check if the matrix is orthogonal (its transpose equals its inverse).
        
        Returns:
            True if the matrix is orthogonal, False otherwise.
        """
        if not self.is_square():
            return False
        try:
            # Check if A * A^T = I
            product = self.multiply(self.transpose())
            identity = Matrix([[1 if i == j else 0 for j in range(self.rows)] for i in range(self.rows)])
            return product.data == identity.data
        except Exception:
            return False
    
    def rank(self) -> int:
        """
        Compute the rank of this matrix.
        
        Returns:
            The rank as an integer.
        """
        # Use cache if available and enabled
        if Config.enable_caching and self._rank_cache is not None:
            return self._rank_cache
        
        try:
            rank_val = sp.Matrix(self.data).rank()
            if Config.enable_caching:
                self._rank_cache = rank_val
            return rank_val
        except Exception as e:
            raise ValueError(f"Error computing rank: {str(e)}")
    
    def condition_number(self) -> float:
        """
        Compute the condition number of this matrix.
        
        Returns:
            The condition number as a float.
        
        Raises:
            ValueError: If the matrix is not square or singular.
        """
        if not self.is_square():
            raise ValueError("Condition number is defined only for square matrices.")
        try:
            # Convert to numpy for condition number computation
            numeric_data = [[float(sp.N(elem)) for elem in row] for row in self.data]
            np_matrix = np.array(numeric_data)
            return float(np.linalg.cond(np_matrix))
        except Exception as e:
            raise ValueError(f"Error computing condition number: {str(e)}")
    
    def is_positive_definite(self) -> bool:
        """
        Check if the matrix is positive definite.
        
        Returns:
            True if the matrix is positive definite, False otherwise.
        """
        if not self.is_square():
            return False
        try:
            # Convert to numpy for positive definite check
            numeric_data = [[float(sp.N(elem)) for elem in row] for row in self.data]
            np_matrix = np.array(numeric_data)
            return np.all(np.linalg.eigvals(np_matrix) > 0)
        except Exception:
            return False
    
    def is_diagonalizable(self) -> bool:
        """
        Check if the matrix is diagonalizable.
        
        Returns:
            True if the matrix is diagonalizable, False otherwise.
        """
        if not self.is_square():
            return False
        try:
            sym_matrix = sp.Matrix(self.data)
            sym_matrix.diagonalize()
            return True
        except (sp.MatrixError, ValueError):
            return False
    
    def norm(self, norm_type: str = 'frobenius') -> float:
        """
        Compute the norm of this matrix.
        
        Args:
            norm_type: Type of norm ('frobenius', 'L1', 'L2', 'inf').
        
        Returns:
            The norm as a float.
        
        Raises:
            ValueError: If norm_type is not supported.
        """
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
            raise ValueError(f"Error computing norm: {str(e)}")
    
    def hadamard_product(self, other: 'Matrix') -> 'Matrix':
        """
        Compute the Hadamard (element-wise) product with another matrix.
        
        Args:
            other: The matrix to multiply element-wise.
        
        Returns:
            A new matrix containing the Hadamard product.
        
        Raises:
            ValueError: If matrices have different dimensions.
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Hadamard product requires matrices of the same dimensions.")
        
        result = [[self.data[i][j] * other.data[i][j] for j in range(self.cols)]
                  for i in range(self.rows)]
        return Matrix(result)
    
    def kronecker_product(self, other: 'Matrix') -> 'Matrix':
        """
        Compute the Kronecker product with another matrix.
        
        Args:
            other: The matrix to compute Kronecker product with.
        
        Returns:
            A new matrix containing the Kronecker product.
        """
        try:
            # Manual implementation of Kronecker product
            result_rows = self.rows * other.rows
            result_cols = self.cols * other.cols
            result = [[0 for _ in range(result_cols)] for _ in range(result_rows)]
            
            for i in range(self.rows):
                for j in range(self.cols):
                    for k in range(other.rows):
                        for l in range(other.cols):
                            result[i * other.rows + k][j * other.cols + l] = self.data[i][j] * other.data[k][l]
            
            return Matrix(result)
        except Exception as e:
            raise ValueError(f"Error computing Kronecker product: {str(e)}")
    
    def pseudoinverse(self) -> 'Matrix':
        """
        Compute the Moore-Penrose pseudoinverse.
        
        Returns:
            A new matrix containing the pseudoinverse.
        """
        try:
            # Convert to numpy for pseudoinverse computation
            numeric_data = [[float(sp.N(elem)) for elem in row] for row in self.data]
            np_matrix = np.array(numeric_data)
            pinv = np.linalg.pinv(np_matrix)
            return Matrix(pinv.tolist())
        except Exception as e:
            raise ValueError(f"Error computing pseudoinverse: {str(e)}")
    
    def lu_decomposition(self) -> Tuple['Matrix', 'Matrix']:
        """
        Compute the LU decomposition.
        
        Returns:
            Tuple of (L, U) matrices.
        
        Raises:
            ValueError: If the matrix is not square or decomposition fails.
        """
        if not self.is_square():
            raise ValueError("LU decomposition is defined only for square matrices.")
        
        try:
            sym_matrix = sp.Matrix(self.data)
            L, U, _ = sym_matrix.LUdecomposition()
            return Matrix(L.tolist()), Matrix(U.tolist())
        except Exception as e:
            raise ValueError(f"Error computing LU decomposition: {str(e)}")
    
    def qr_decomposition(self) -> Tuple['Matrix', 'Matrix']:
        """
        Compute the QR decomposition.
        
        Returns:
            Tuple of (Q, R) matrices.
        
        Raises:
            ValueError: If decomposition fails.
        """
        try:
            sym_matrix = sp.Matrix(self.data)
            Q, R = sym_matrix.QRdecomposition()
            return Matrix(Q.tolist()), Matrix(R.tolist())
        except Exception as e:
            raise ValueError(f"Error computing QR decomposition: {str(e)}")
    
    def svd(self) -> Tuple['Matrix', 'Matrix', 'Matrix']:
        """
        Compute the Singular Value Decomposition.
        
        Returns:
            Tuple of (U, S, V) matrices.
        
        Raises:
            ValueError: If decomposition fails.
        """
        try:
            # Convert to numpy for SVD computation
            numeric_data = [[float(sp.N(elem)) for elem in row] for row in self.data]
            np_matrix = np.array(numeric_data)
            U, s, Vt = np.linalg.svd(np_matrix)
            S = np.zeros((U.shape[1], Vt.shape[0]))
            S[:len(s), :len(s)] = np.diag(s)
            return Matrix(U.tolist()), Matrix(S.tolist()), Matrix(Vt.T.tolist())
        except Exception as e:
            raise ValueError(f"Error computing SVD: {str(e)}")
    
    def cholesky_decomposition(self) -> 'Matrix':
        """
        Compute the Cholesky decomposition.
        
        Returns:
            The lower triangular matrix L such that A = L * L^T.
        
        Raises:
            ValueError: If the matrix is not positive definite.
        """
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
            raise ValueError(f"Error computing Cholesky decomposition: {str(e)}")
    
    def preview(self, max_rows: int = 5, max_cols: int = 5) -> str:
        """
        Show a preview of the matrix for large matrices.
        
        Args:
            max_rows: Maximum number of rows to show.
            max_cols: Maximum number of columns to show.
        
        Returns:
            A formatted string showing the matrix preview.
        """
        if self.rows <= max_rows and self.cols <= max_cols:
            return str(self)
        
        s = f"Matrix Preview ({self.rows}x{self.cols}):\n"
        
        # Show top rows
        for i in range(min(max_rows, self.rows)):
            row_str = "\t".join(f"a{i+1}{j+1} = {format_expression_for_display(sp.pretty(self.data[i][j]))}" 
                               for j in range(min(max_cols, self.cols)))
            if self.cols > max_cols:
                row_str += "\t..."
            s += row_str + "\n"
        
        if self.rows > max_rows:
            s += "...\n"
        
        return s
    
    @staticmethod
    def from_csv(filename: str) -> 'Matrix':
        """
        Load a matrix from a CSV file.
        
        Args:
            filename: Path to the CSV file.
        
        Returns:
            A new Matrix instance.
        """
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                data = [[elem.strip() for elem in row] for row in reader if row]
            return Matrix(data)
        except Exception as e:
            raise ValueError(f"Error loading from CSV: {str(e)}")
    
    @staticmethod
    def from_json(filename: str) -> Tuple['Matrix', Optional[str]]:
        """
        Load a matrix from a JSON file.
        
        Args:
            filename: Path to the JSON file.
        
        Returns:
            Tuple of (Matrix instance, optional name).
        """
        try:
            with open(filename, 'r') as f:
                json_data = json.load(f)
            data = json_data.get('data', json_data)
            name = json_data.get('name', None)
            return Matrix(data), name
        except Exception as e:
            raise ValueError(f"Error loading from JSON: {str(e)}")
    
    @staticmethod
    def from_numpy(filename: str) -> 'Matrix':
        """
        Load a matrix from a NumPy .npy file.
        
        Args:
            filename: Path to the .npy file.
        
        Returns:
            A new Matrix instance.
        """
        try:
            data = np.load(filename)
            return Matrix(data.tolist())
        except Exception as e:
            raise ValueError(f"Error loading from NumPy file: {str(e)}")
    
    @staticmethod
    def from_matlab(filename: str, variable_name: str = 'matrix') -> 'Matrix':
        """
        Load a matrix from a MATLAB .mat file.
        
        Args:
            filename: Path to the .mat file.
            variable_name: Name of the variable in the .mat file.
        
        Returns:
            A new Matrix instance.
        """
        try:
            mat_data = sio.loadmat(filename)
            data = mat_data[variable_name]
            return Matrix(data.tolist())
        except Exception as e:
            raise ValueError(f"Error loading from MATLAB file: {str(e)}")
    
    def to_csv(self, filename: str) -> None:
        """
        Export the matrix to a CSV file.
        
        Args:
            filename: Path to save the CSV file.
        """
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                for row in self.data:
                    writer.writerow([str(elem) for elem in row])
            print_success(f"Matrix exported to {filename}")
        except Exception as e:
            print_error(f"Error exporting to CSV: {str(e)}")
    
    def to_json(self, filename: str, name: Optional[str] = None) -> None:
        """
        Export the matrix to a JSON file.
        
        Args:
            filename: Path to save the JSON file.
            name: Optional name for the matrix.
        """
        try:
            json_data = {
                'data': [[str(elem) for elem in row] for row in self.data],
                'rows': self.rows,
                'cols': self.cols
            }
            if name:
                json_data['name'] = name
            with open(filename, 'w') as f:
                json.dump(json_data, f, indent=2)
            print_success(f"Matrix exported to {filename}")
        except Exception as e:
            print_error(f"Error exporting to JSON: {str(e)}")
    
    def to_latex(self, filename: str) -> None:
        """
        Export the matrix to a LaTeX file.
        
        Args:
            filename: Path to save the LaTeX file.
        """
        try:
            latex_str = "\\begin{bmatrix}\n"
            for i, row in enumerate(self.data):
                row_str = " & ".join([sp.latex(elem) for elem in row])
                latex_str += "  " + row_str
                if i < self.rows - 1:
                    latex_str += " \\\\\n"
                else:
                    latex_str += "\n"
            latex_str += "\\end{bmatrix}"
            
            with open(filename, 'w') as f:
                f.write(latex_str)
            print_success(f"Matrix exported to {filename}")
        except Exception as e:
            print_error(f"Error exporting to LaTeX: {str(e)}")
    
    def to_numpy(self, filename: str) -> None:
        """
        Export the matrix to a NumPy .npy file.
        
        Args:
            filename: Path to save the .npy file.
        """
        try:
            # Convert to float for numerical storage
            numeric_data = [[float(sp.N(elem)) for elem in row] for row in self.data]
            np_array = np.array(numeric_data)
            np.save(filename, np_array)
            print_success(f"Matrix exported to {filename}")
        except Exception as e:
            print_error(f"Error exporting to NumPy file: {str(e)}")
    
    def to_matlab(self, filename: str, variable_name: str = 'matrix') -> None:
        """
        Export the matrix to a MATLAB .mat file.
        
        Args:
            filename: Path to save the .mat file.
            variable_name: Name of the variable in the .mat file.
        """
        try:
            # Convert to float for numerical storage
            numeric_data = [[float(sp.N(elem)) for elem in row] for row in self.data]
            np_array = np.array(numeric_data)
            sio.savemat(filename, {variable_name: np_array})
            print_success(f"Matrix exported to {filename}")
        except Exception as e:
            print_error(f"Error exporting to MATLAB file: {str(e)}")
    
    @staticmethod
    def from_batch_input(rows: int, cols: int) -> 'Matrix':
        """
        Create a matrix from batch input (paste entire matrix).
        
        Args:
            rows: Number of rows.
            cols: Number of columns.
        
        Returns:
            A new Matrix instance.
        
        Raises:
            ValueError: If input validation fails.
        """
        print_info(f"Paste your {rows}x{cols} matrix (space or tab separated):")
        data = []
        timeout_count = 0
        max_attempts = 3
        
        for i in range(rows):
            attempt_count = 0
            while attempt_count < max_attempts:
                try:
                    line = input().strip()
                    if not line:
                        print_warning("Empty line detected. Please enter matrix elements:")
                        attempt_count += 1
                        continue
                    
                    elements = line.split()
                    if len(elements) != cols:
                        print_warning(f"Expected {cols} elements, got {len(elements)}. Try again:")
                        attempt_count += 1
                        continue
                    
                    # Validate elements
                    if not validate_matrix_input(elements):
                        print_error("Invalid elements detected. Please check your input and try again:")
                        attempt_count += 1
                        continue
                    
                    data.append(elements)
                    break
                    
                except KeyboardInterrupt:
                    raise ValueError("Input cancelled by user")
                except EOFError:
                    raise ValueError("Unexpected end of input")
                except Exception as e:
                    print_error(f"Invalid input: {str(e)}. Try again:")
                    attempt_count += 1
                    
            if attempt_count >= max_attempts:
                raise ValueError(f"Too many failed attempts for row {i+1}. Please check your input format.")
        
        return Matrix(data)

# -----------------------------
# Matrix Manager Class
# -----------------------------
class MatrixManager:
    """
    A class to manage multiple matrices and their operations.
    
    This class provides functionality to create, delete, edit, and perform operations
    on multiple matrices. It maintains a dictionary of named matrices and provides
    a user interface for matrix management.
    
    Attributes:
        matrices (Dict[str, Matrix]): Dictionary mapping matrix names to Matrix instances
        counter (int): Counter for generating unique matrix names
        history (List[Tuple[str, str]]): List of operations performed (operation, matrix name)
    """
    
    def __init__(self):
        self.matrices = {}  # maps a name (like "A", "B", etc.) to a Matrix instance
        self.counter = 0   # for assigning names
        self.history = []  # track operation history

    def _get_new_name(self) -> str:
        """
        Generate a new unique name for a matrix.
        
        Returns:
            A string representing the new matrix name.
        """
        if self.counter < 26:
            name = chr(65 + self.counter)
        else:
            name = chr(65 + (self.counter % 26)) + str(self.counter // 26)
        self.counter += 1
        return name

    def create_matrix(self) -> None:
        """
        Create a new matrix through user input.
        
        The user is prompted to enter dimensions and elements of the matrix.
        The matrix is stored with a unique name.
        """
        name = self._get_new_name()
        try:
            while True:
                try:
                    rows = int(input(f"Enter number of rows for Matrix {name}: "))
                    cols = int(input(f"Enter number of columns for Matrix {name}: "))
                    if rows <= 0 or cols <= 0:
                        print_warning("Dimensions must be positive integers.")
                        continue
                    break
                except ValueError:
                    print_warning("Please enter valid integers for dimensions.")

            # Ask if user wants batch input
            batch_choice = input("Enter elements (1) one-by-one or (2) batch paste? [1/2]: ").strip()
            
            if batch_choice == '2':
                self.matrices[name] = Matrix.from_batch_input(rows, cols)
            else:
                data = []
                for i in range(rows):
                    row = []
                    for j in range(cols):
                        while True:
                            expr = input(f"Enter element a{i+1}{j+1}: ")
                            try:
                                # Preprocess to allow ^ and e notation
                                row.append(sp.sympify(preprocess_expression(expr)))
                                break
                            except (sp.SympifyError, ValueError):
                                print_warning("Invalid input. Please enter a valid number or expression.")
                    data.append(row)
                self.matrices[name] = Matrix(data)
            
            self.history.append(("create", name, None))
            print_success(f"Matrix {name} has been created successfully.")
            
        except Exception as e:
            print_error(f"Error creating matrix: {str(e)}")
            if name in self.matrices:
                del self.matrices[name]

    def delete_matrix(self) -> None:
        """
        Delete a matrix selected by the user.
        """
        chosen = self.select_matrix()
        if chosen is None:
            return
        name, mat = chosen
        
        # Confirmation before delete
        if not confirm_action(f"Are you sure you want to delete Matrix {name}?"):
            print_info("Deletion cancelled.")
            return
        
        try:
            # Store matrix data for undo
            self.history.append(("delete", name, mat))
            del self.matrices[name]
            print_success(f"Matrix {name} has been deleted successfully.")
        except Exception as e:
            print_error(f"Error deleting matrix: {str(e)}")

    def edit_matrix(self) -> None:
        """
        Edit an existing matrix through user input.
        """
        chosen = self.select_matrix()
        if chosen is None:
            return
        name, old_matrix = chosen
        try:
            while True:
                try:
                    rows = int(input(f"Enter new number of rows for Matrix {name}: "))
                    cols = int(input(f"Enter new number of columns for Matrix {name}: "))
                    if rows <= 0 or cols <= 0:
                        print_warning("Dimensions must be positive integers.")
                        continue
                    break
                except ValueError:
                    print_warning("Please enter valid integers for dimensions.")

            # Ask if user wants batch input
            batch_choice = input("Enter elements (1) one-by-one or (2) batch paste? [1/2]: ").strip()
            
            if batch_choice == '2':
                new_matrix = Matrix.from_batch_input(rows, cols)
            else:
                new_data = []
                for i in range(rows):
                    row = []
                    for j in range(cols):
                        while True:
                            expr = input(f"Enter new element a{i+1}{j+1}: ")
                            try:
                                # Preprocess to allow ^ and e notation
                                row.append(sp.sympify(preprocess_expression(expr)))
                                break
                            except (sp.SympifyError, ValueError):
                                print_warning("Invalid input. Please enter a valid number or expression.")
                    new_data.append(row)
                new_matrix = Matrix(new_data)
            
            # Store old matrix for undo
            self.history.append(("edit", name, old_matrix))
            self.matrices[name] = new_matrix
            print_success(f"Matrix {name} has been updated successfully.")
            
        except Exception as e:
            print_error(f"Error editing matrix: {str(e)}")

    def list_matrices(self) -> List[Tuple[str, Matrix]]:
        """
        List all stored matrices.
        
        Returns:
            List of tuples containing (matrix_name, matrix_instance).
        """
        if not self.matrices:
            print("No matrices defined yet.")
            return []
        sorted_names = sorted(self.matrices.keys())
        print("\nStored Matrices:")
        numbered = []
        for idx, name in enumerate(sorted_names, start=1):
            mat = self.matrices[name]
            print(f"{idx}. Matrix {name} ({mat.rows}x{mat.cols})")
            numbered.append((name, mat))
        print("0. Back")
        return numbered

    def show_matrix(self) -> None:
        """
        Display the contents of a selected matrix.
        """
        numbered = self.list_matrices()
        if not numbered:
            return
        try:
            while True:
                try:
                    choice = int(input("Select a matrix number to show (or 0 to cancel): "))
                    if choice == 0:
                        return
                    if 1 <= choice <= len(numbered):
                        name, mat = numbered[choice - 1]
                        print(f"\nMatrix {name}:")
                        print(mat)
                        return
                    print(f"Please enter a number between 0 and {len(numbered)}.")
                except ValueError:
                    print("Please enter a valid number.")
        except Exception as e:
            print(f"Error displaying matrix: {str(e)}")

    def select_matrix(self) -> Optional[Tuple[str, Matrix]]:
        """
        Let the user select a matrix from the stored matrices.
        
        Returns:
            Tuple of (matrix_name, matrix_instance) if a matrix is selected,
            None if selection is cancelled or an error occurs.
        """
        numbered = self.list_matrices()
        if not numbered:
            return None
        try:
            while True:
                try:
                    choice = int(input("Select a matrix by number (or 0 to cancel): "))
                    if choice == 0:
                        return None
                    if 1 <= choice <= len(numbered):
                        return numbered[choice - 1]
                    print(f"Please enter a number between 0 and {len(numbered)}.")
                except ValueError:
                    print("Please enter a valid number.")
        except Exception as e:
            print(f"Error selecting matrix: {str(e)}")
            return None

    def store_result(self, result_matrix: Matrix) -> None:
        """
        Store a result matrix with a new name.
        
        Args:
            result_matrix: The matrix to store.
        """
        try:
            while True:
                choice = input("Store result as a new matrix? (Enter 1 for yes, 0 for no): ").strip()
                if choice in ['0', '1']:
                    break
                print_warning("Please enter 0 or 1.")
            
            if choice == '1':
                name = self._get_new_name()
                self.matrices[name] = result_matrix
                self.history.append(("store_result", name, None))
                print_success(f"Result stored as Matrix {name}.")
        except Exception as e:
            print_error(f"Error storing result: {str(e)}")

    def undo_last_operation(self) -> None:
        """
        Undo the last operation performed on matrices.
        """
        if not self.history:
            print_warning("No operations to undo.")
            return
        
        operation, name, data = self.history.pop()
        if operation in ["create", "create_identity", "create_zeros", "create_ones", "create_random", "create_diagonal"]:
            if name in self.matrices:
                del self.matrices[name]
            print_success(f"Undid creation of Matrix {name}")
        elif operation == "delete":
            # Restore deleted matrix
            if data is not None:
                self.matrices[name] = data
                print_success(f"Restored Matrix {name}")
            else:
                print_warning(f"Cannot restore Matrix {name} (no backup data)")
        elif operation == "edit":
            # Restore previous matrix state
            if data is not None:
                self.matrices[name] = data
                print_success(f"Restored previous state of Matrix {name}")
            else:
                print_warning(f"Cannot restore Matrix {name} (no backup data)")
        elif operation == "store_result":
            if name in self.matrices:
                del self.matrices[name]
            print_success(f"Undid storing of result Matrix {name}")
        elif operation == "import":
            if name in self.matrices:
                del self.matrices[name]
            print_success(f"Undid import of Matrix {name}")
    
    def view_history(self) -> None:
        """
        Display the operation history.
        """
        if not self.history:
            print_info("No operations in history.")
            return
        
        print_header("Operation History")
        for idx, (operation, name, _) in enumerate(self.history, 1):
            print(f"{idx}. {operation.upper()}: Matrix {name}")
    
    def import_matrix(self, filename: str, name: Optional[str] = None) -> None:
        """
        Import a matrix from a file.
        
        Args:
            filename: Path to the file.
            name: Optional name for the matrix. If None, auto-generate.
        """
        try:
            # Check if file exists
            if not check_file_exists(filename):
                print_error(f"File not found or not readable: {filename}")
                return
            
            ext = os.path.splitext(filename)[1].lower()
            
            if ext == '.csv':
                matrix = Matrix.from_csv(filename)
            elif ext == '.json':
                matrix, json_name = Matrix.from_json(filename)
                if name is None and json_name:
                    name = json_name
            elif ext == '.npy':
                matrix = Matrix.from_numpy(filename)
            elif ext == '.mat':
                var_name = input("Enter MATLAB variable name [default: matrix]: ").strip() or 'matrix'
                matrix = Matrix.from_matlab(filename, var_name)
            else:
                print_error(f"Unsupported file format: {ext}")
                return
            
            if name is None:
                name = self._get_new_name()
            
            self.matrices[name] = matrix
            self.history.append(("import", name, None))
            
            # Add to recent files list
            if filename not in Config.recent_files_list:
                Config.recent_files_list.insert(0, filename)
                # Keep only last 5 files
                Config.recent_files_list = Config.recent_files_list[:5]
            
            print_success(f"Matrix imported as {name} from {filename}")
            logger.info(f"Matrix imported as {name} from {filename}")
            
        except Exception as e:
            print_error(f"Error importing matrix: {str(e)}")
            logger.error(f"Error importing matrix: {str(e)}")
    
    def export_matrix(self, name: str, filename: str, format_type: Optional[str] = None) -> None:
        """
        Export a matrix to a file.
        
        Args:
            name: Name of the matrix to export.
            filename: Path to save the file.
            format_type: Export format (csv, json, latex, numpy, matlab, text).
        """
        if name not in self.matrices:
            print_error(f"Matrix {name} not found.")
            return
        
        matrix = self.matrices[name]
        
        # Use file extension helper
        if format_type:
            ext_map = {
                'csv': '.csv',
                'json': '.json',
                'latex': '.tex',
                'numpy': '.npy',
                'matlab': '.mat',
                'text': '.txt'
            }
            filename = ensure_file_extension(filename, ext_map.get(format_type, '.csv'))
        else:
            ext_map = {
                'csv': '.csv',
                'json': '.json',
                'latex': '.tex',
                'numpy': '.npy',
                'matlab': '.mat',
                'text': '.txt'
            }
            filename = ensure_file_extension(filename, ext_map.get(Config.default_export_format, '.csv'))
            format_type = Config.default_export_format
        
        ext = os.path.splitext(filename)[1].lower()
        
        if format_type is None:
            format_map = {
                '.csv': 'csv',
                '.json': 'json',
                '.tex': 'latex',
                '.npy': 'numpy',
                '.mat': 'matlab',
                '.txt': 'text'
            }
            format_type = format_map.get(ext, Config.default_export_format)
        
        try:
            if format_type == 'csv':
                matrix.to_csv(filename)
            elif format_type == 'json':
                matrix.to_json(filename, name)
            elif format_type == 'latex':
                matrix.to_latex(filename)
            elif format_type == 'numpy':
                matrix.to_numpy(filename)
            elif format_type == 'matlab':
                var_name = input("Enter MATLAB variable name [default: matrix]: ").strip() or 'matrix'
                matrix.to_matlab(filename, var_name)
            elif format_type == 'text':
                with open(filename, 'w') as f:
                    f.write(f"Matrix {name}:\n")
                    f.write(str(matrix))
                print_success(f"Matrix exported to {filename}")
            else:
                print_error(f"Unsupported export format: {format_type}")
                return
            
            # Add to recent files list
            if filename not in Config.recent_files_list:
                Config.recent_files_list.insert(0, filename)
                # Keep only last 5 files
                Config.recent_files_list = Config.recent_files_list[:5]
            
            logger.info(f"Matrix {name} exported to {filename}")
            
        except Exception as e:
            print_error(f"Error exporting matrix: {str(e)}")
            logger.error(f"Error exporting matrix: {str(e)}")
    
    def batch_import(self, filenames: List[str]) -> None:
        """
        Import multiple matrices from files.
        
        Args:
            filenames: List of file paths.
        """
        if Config.show_progress:
            for filename in tqdm(filenames, desc="Importing matrices"):
                self.import_matrix(filename)
        else:
            for filename in filenames:
                self.import_matrix(filename)
    
    def create_identity_matrix(self, size: int) -> None:
        """
        Create an identity matrix of given size.
        
        Args:
            size: Size of the identity matrix.
        """
        try:
            if size <= 0:
                print_error("Size must be positive")
                return
            
            data = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
            matrix = Matrix(data)
            name = self._get_new_name()
            self.matrices[name] = matrix
            self.history.append(("create_identity", name, None))
            print_success(f"Identity matrix {name} ({size}x{size}) created")
            logger.info(f"Identity matrix {name} ({size}x{size}) created")
        except Exception as e:
            print_error(f"Error creating identity matrix: {str(e)}")
            logger.error(f"Error creating identity matrix: {str(e)}")
    
    def create_zeros_matrix(self, rows: int, cols: int) -> None:
        """
        Create a zeros matrix of given dimensions.
        
        Args:
            rows: Number of rows.
            cols: Number of columns.
        """
        try:
            if rows <= 0 or cols <= 0:
                print_error("Dimensions must be positive")
                return
            
            data = [[0 for j in range(cols)] for i in range(rows)]
            matrix = Matrix(data)
            name = self._get_new_name()
            self.matrices[name] = matrix
            self.history.append(("create_zeros", name, None))
            print_success(f"Zeros matrix {name} ({rows}x{cols}) created")
            logger.info(f"Zeros matrix {name} ({rows}x{cols}) created")
        except Exception as e:
            print_error(f"Error creating zeros matrix: {str(e)}")
            logger.error(f"Error creating zeros matrix: {str(e)}")
    
    def create_ones_matrix(self, rows: int, cols: int) -> None:
        """
        Create a ones matrix of given dimensions.
        
        Args:
            rows: Number of rows.
            cols: Number of columns.
        """
        try:
            if rows <= 0 or cols <= 0:
                print_error("Dimensions must be positive")
                return
            
            data = [[1 for j in range(cols)] for i in range(rows)]
            matrix = Matrix(data)
            name = self._get_new_name()
            self.matrices[name] = matrix
            self.history.append(("create_ones", name, None))
            print_success(f"Ones matrix {name} ({rows}x{cols}) created")
            logger.info(f"Ones matrix {name} ({rows}x{cols}) created")
        except Exception as e:
            print_error(f"Error creating ones matrix: {str(e)}")
            logger.error(f"Error creating ones matrix: {str(e)}")
    
    def create_random_matrix(self, rows: int, cols: int, min_val: float = 0, max_val: float = 1) -> None:
        """
        Create a random matrix of given dimensions.
        
        Args:
            rows: Number of rows.
            cols: Number of columns.
            min_val: Minimum value for random elements.
            max_val: Maximum value for random elements.
        """
        try:
            if rows <= 0 or cols <= 0:
                print_error("Dimensions must be positive")
                return
            
            import random
            data = [[random.uniform(min_val, max_val) for j in range(cols)] for i in range(rows)]
            matrix = Matrix(data)
            name = self._get_new_name()
            self.matrices[name] = matrix
            self.history.append(("create_random", name, None))
            print_success(f"Random matrix {name} ({rows}x{cols}) created with values in [{min_val}, {max_val}]")
            logger.info(f"Random matrix {name} ({rows}x{cols}) created")
        except Exception as e:
            print_error(f"Error creating random matrix: {str(e)}")
            logger.error(f"Error creating random matrix: {str(e)}")
    
    def create_diagonal_matrix(self, diagonal_elements: List[Union[int, float, str]]) -> None:
        """
        Create a diagonal matrix from given diagonal elements.
        
        Args:
            diagonal_elements: List of elements for the diagonal.
        """
        try:
            if not diagonal_elements:
                print_error("Diagonal elements list cannot be empty")
                return
            
            size = len(diagonal_elements)
            data = [[diagonal_elements[i] if i == j else 0 for j in range(size)] for i in range(size)]
            matrix = Matrix(data)
            name = self._get_new_name()
            self.matrices[name] = matrix
            self.history.append(("create_diagonal", name, None))
            print_success(f"Diagonal matrix {name} ({size}x{size}) created")
            logger.info(f"Diagonal matrix {name} ({size}x{size}) created")
        except Exception as e:
            print_error(f"Error creating diagonal matrix: {str(e)}")
            logger.error(f"Error creating diagonal matrix: {str(e)}")
    
    def show_help(self) -> None:
        """
        Display help information.
        """
        print_header("Matrix Calculator Help")
        print(f"{Fore.CYAN}Commands (type anywhere):{Style.RESET_ALL}")
        print("  ? or help    - Show this help message")
        print("  history      - View operation history")
        print("  clear        - Clear the screen")
        print("  config       - Open configuration menu")
        print("  exit         - Exit the program")
        print(f"\n{Fore.CYAN}Symbolic Expressions:{Style.RESET_ALL}")
        print("  Variables: x, y, z, a, b, etc.")
        print("  Functions: sin(x), cos(x), exp(x), log(x), sqrt(x)")
        print("  Constants: pi, e (Euler's number)")
        print("  Operators: +, -, *, /, ^ (power)")
        print(f"\n{Fore.CYAN}Examples:{Style.RESET_ALL}")
        print("  x^2 + 2*x + 1")
        print("  e^x")
        print("  sin(pi*x)")
        print("  sqrt(x^2 + y^2)")
        print("  2*e^(-x^2)")
        print(f"\n{Fore.CYAN}Matrix Operations:{Style.RESET_ALL}")
        print("  Basic: Add, subtract, multiply, transpose, inverse")
        print("  Advanced: LU, QR, SVD, Cholesky decomposition")
        print("  Properties: rank, condition number, norms")
        print("  Templates: identity, zeros, ones, random, diagonal")
        print(f"\n{Fore.CYAN}File Tips:{Style.RESET_ALL}")
        print("  File extension optional - will auto-add if missing")
        print("  With extension: matrix.csv, result.json, data.txt")
        print("  Without extension: matrix (saves as matrix.csv)")
        print(f"  Default format: {Config.default_export_format}")
        print(f"\n{Fore.CYAN}Performance Tips:{Style.RESET_ALL}")
        print("  Use batch input for large matrices")
        print("  Enable caching for repeated operations")
        print("  Check matrix size warnings for large matrices")

    def save_matrices(self, filename: str) -> None:
        """
        Save all matrices to a file.
        
        Args:
            filename: Name of the file to save matrices to.
        """
        try:
            filename = ensure_file_extension(filename, '.txt')
            
            with open(filename, 'w') as f:
                for name, matrix in self.matrices.items():
                    f.write(f"Matrix {name}:\n")
                    f.write(str(matrix))
                    f.write("\n\n")
            print_success(f"Matrices saved to {filename}")
            logger.info(f"All matrices saved to {filename}")
        except Exception as e:
            print_error(f"Error saving matrices: {str(e)}")
            logger.error(f"Error saving matrices: {str(e)}")

    def load_matrices(self, filename: str) -> None:
        """
        Load matrices from a file.
        
        Args:
            filename: Name of the file to load matrices from.
        """
        try:
            # Check if file exists
            if not check_file_exists(filename):
                print_error(f"File not found or not readable: {filename}")
                return
            
            with open(filename, 'r') as f:
                content = f.read()
                matrices_data = content.split("\n\n")
                for matrix_data in matrices_data:
                    if not matrix_data.strip():
                        continue
                    lines = matrix_data.strip().split("\n")
                    name = lines[0].split()[1].rstrip(":")
                    data = []
                    for line in lines[1:]:
                        row = []
                        for elem in line.split("\t"):
                            value = elem.split("=")[1].strip()
                            row.append(value)
                        data.append(row)
                    self.matrices[name] = Matrix(data)
            print_success(f"Matrices loaded from {filename}")
            logger.info(f"Matrices loaded from {filename}")
        except Exception as e:
            print_error(f"Error loading matrices: {str(e)}")
            logger.error(f"Error loading matrices: {str(e)}")

# -----------------------------
# Menus
# -----------------------------
def management_menu(manager: MatrixManager) -> None:
    """
    Display and handle the matrix management menu.
    
    Args:
        manager: The MatrixManager instance to use.
    """
    while True:
        print_header("Matrix Management Menu")
        print("1. New Matrix")
        print("2. Delete Matrix")
        print("3. Edit Matrix")
        print("4. List/Show Matrices")
        print("5. Import Matrix from File")
        print("6. Export Matrix to File")
        print("7. Matrix Templates")
        print("8. Undo Last Operation")
        print("9. View History")
        print("10. Save All Matrices")
        print("11. Load All Matrices")
        print("0. Back to Main Menu")
        print("\nType '?' for help, 'clear' to clear screen")
        
        choice = input(f"\n{Fore.YELLOW}Select an option: {Style.RESET_ALL}").strip().lower()
        
        if choice == '?':
            manager.show_help()
        elif choice == 'clear':
            clear_screen()
        elif choice == '1':
            manager.create_matrix()
        elif choice == '2':
            manager.delete_matrix()
        elif choice == '3':
            manager.edit_matrix()
        elif choice == '4':
            manager.show_matrix()
        elif choice == '5':
            filename = input("Enter filename to import: ").strip()
            if filename:
                manager.import_matrix(filename)
            else:
                print_warning("Invalid filename.")
        elif choice == '6':
            chosen = manager.select_matrix()
            if chosen:
                name, _ = chosen
                filename = input("Enter filename to export: ").strip()
                if filename:
                    manager.export_matrix(name, filename)
                else:
                    print_warning("Invalid filename.")
        elif choice == '7':  # Matrix Templates
            print_header("Matrix Templates")
            print("1. Identity Matrix")
            print("2. Zeros Matrix")
            print("3. Ones Matrix")
            print("4. Random Matrix")
            print("5. Diagonal Matrix")
            print("0. Back")
            
            template_choice = input(f"\n{Fore.YELLOW}Select template: {Style.RESET_ALL}").strip()
            
            if template_choice == '1':
                try:
                    size = int(input("Enter size of identity matrix: "))
                    manager.create_identity_matrix(size)
                except ValueError:
                    print_error("Invalid size. Please enter a positive integer.")
            elif template_choice == '2':
                try:
                    rows = int(input("Enter number of rows: "))
                    cols = int(input("Enter number of columns: "))
                    manager.create_zeros_matrix(rows, cols)
                except ValueError:
                    print_error("Invalid dimensions. Please enter positive integers.")
            elif template_choice == '3':
                try:
                    rows = int(input("Enter number of rows: "))
                    cols = int(input("Enter number of columns: "))
                    manager.create_ones_matrix(rows, cols)
                except ValueError:
                    print_error("Invalid dimensions. Please enter positive integers.")
            elif template_choice == '4':
                try:
                    rows = int(input("Enter number of rows: "))
                    cols = int(input("Enter number of columns: "))
                    min_val = float(input("Enter minimum value [default: 0]: ") or "0")
                    max_val = float(input("Enter maximum value [default: 1]: ") or "1")
                    manager.create_random_matrix(rows, cols, min_val, max_val)
                except ValueError:
                    print_error("Invalid input. Please enter valid numbers.")
            elif template_choice == '5':
                try:
                    elements_input = input("Enter diagonal elements (space-separated): ").strip()
                    elements = elements_input.split()
                    manager.create_diagonal_matrix(elements)
                except ValueError:
                    print_error("Invalid diagonal elements.")
        elif choice == '8':
            manager.undo_last_operation()
        elif choice == '9':
            manager.view_history()
        elif choice == '10':
            filename = input("Enter filename to save all matrices: ").strip()
            if filename:
                manager.save_matrices(filename)
            else:
                print_warning("Invalid filename.")
        elif choice == '11':
            filename = input("Enter filename to load matrices: ").strip()
            if filename:
                manager.load_matrices(filename)
            else:
                print_warning("Invalid filename.")
        elif choice == '0':
            break
        else:
            print_error("Invalid choice. Please try again.")

def operations_menu(manager: MatrixManager) -> None:
    """
    Display and handle the matrix operations menu.
    
    Args:
        manager: The MatrixManager instance to use.
    """
    while True:
        print_header("Matrix Operations Menu")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. Power")
        print("5. Transpose")
        print("6. Inverse")
        print("7. Determinant")
        print("8. Trace")
        print("9. Eigenvalues")
        print("10. Characteristic Equation")
        print("11. Check if Symmetric")
        print("12. Check if Orthogonal")
        print("13. Matrix Properties")
        print("14. Matrix Decompositions")
        print("15. Advanced Operations")
        print("0. Back to Main Menu")
        print("\nType '?' for help, 'clear' to clear screen")
        
        choice = input(f"\n{Fore.YELLOW}Select an option: {Style.RESET_ALL}").strip().lower()
        
        if choice == '?':
            manager.show_help()
            continue
        elif choice == 'clear':
            clear_screen()
            continue
        
        try:
            if choice in ['1', '2']:
                print_info("Select first matrix:")
                first = manager.select_matrix()
                if first is None:
                    continue
                _, mat1 = first
                print_info("Select second matrix:")
                second = manager.select_matrix()
                if second is None:
                    continue
                _, mat2 = second
                
                if Config.show_progress:
                    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                        task = progress.add_task("Computing...", total=None)
                        if choice == '1':
                            result = mat1.add(mat2)
                            operation = "Addition"
                        else:
                            result = mat1.subtract(mat2)
                            operation = "Subtraction"
                        progress.update(task, completed=True)
                else:
                    if choice == '1':
                        result = mat1.add(mat2)
                        operation = "Addition"
                    else:
                        result = mat1.subtract(mat2)
                        operation = "Subtraction"
                
                print_success(f"{operation} successful. Result:")
                print(result)
                manager.store_result(result)

            elif choice == '3':  # Multiplication
                print("\nSelect a matrix:")
                first = manager.select_matrix()
                if first is None:
                    continue
                _, mat1 = first
                while True:
                    scalar_or_matrix = input("\nMultiply by (1) another matrix or (2) a scalar? Enter 1 or 2: ").strip()
                    if scalar_or_matrix in ['1', '2']:
                        break
                    print("Please enter 1 or 2.")
                
                if scalar_or_matrix == '1':
                    print("\nSelect second matrix:")
                    second = manager.select_matrix()
                    if second is None:
                        continue
                    _, mat2 = second
                    result = mat1.multiply(mat2)
                
                elif scalar_or_matrix == '2':
                    while True:
                        scalar = input("\nEnter the scalar value: ").strip()
                        try:
                            scalar = sp.N(sp.sympify(scalar))
                            break
                        except (ValueError, sp.SympifyError):
                            print("Invalid scalar value. Please try again.")
                    result = mat1.multiply(scalar)
                
                print("\nMultiplication successful. Result:")
                print(result)
                manager.store_result(result)

            elif choice == '4':
                chosen = manager.select_matrix()
                if chosen is None: 
                    continue
                _, mat = chosen
                while True:
                    try:
                        exp = float(input("\nEnter the real exponent: "))
                        break
                    except ValueError:
                        print("Invalid input for exponent. Please try again.")
                
                result = mat.power(exp)
                # Automatically convert to numerical approximations
                result = Matrix([[sp.N(elem) for elem in row] for row in result.data])
                print_success(f"Matrix^{exp} computed successfully. Result:")
                print(result)
                manager.store_result(result)

            elif choice in ['5', '6', '7', '8', '9', '10', '11', '12']:
                chosen = manager.select_matrix()
                if chosen is None: 
                    continue
                _, mat = chosen
                
                if choice == '5':
                    result = mat.transpose()
                    print("\nTranspose successful. Result:")
                    print(result)
                    manager.store_result(result)
                elif choice == '6':
                    result = mat.inverse()
                    print("\nInverse successful. Result:")
                    print(result)
                    manager.store_result(result)
                elif choice == '7':
                    det = mat.determinant()
                    print_success(f"Determinant: {format_expression_for_display(str(det))}")
                elif choice == '8':
                    tr = mat.trace()
                    print_success(f"Trace: {format_expression_for_display(str(tr))}")
                elif choice == '9':
                    while True:
                        numeric = input("\nEvaluate eigenvalues numerically? (Enter 1 for yes, 0 for no): ").strip()
                        if numeric in ['0', '1']:
                            break
                        print("Please enter 0 or 1.")
                    
                    eigvals = mat.eigenvalues(numeric=(numeric == '1'))
                    print_success("Eigenvalues:")
                    for idx, val in enumerate(eigvals, start=1):
                        print(f"  λ{idx} = {format_expression_for_display(str(val))}")
                elif choice == '10':
                    char_eq = mat.characteristic_equation()
                    print_success("Characteristic Equation:")
                    print(f"  {format_expression_for_display(str(char_eq))} = 0")
                elif choice == '11':
                    is_sym = mat.is_symmetric()
                    print(f"\nMatrix is {'symmetric' if is_sym else 'not symmetric'}")
                elif choice == '12':
                    is_orth = mat.is_orthogonal()
                    print(f"\nMatrix is {'orthogonal' if is_orth else 'not orthogonal'}")
            
            elif choice == '13':  # Matrix Properties
                chosen = manager.select_matrix()
                if chosen is None:
                    continue
                _, mat = chosen
                
                print_header("Matrix Properties")
                try:
                    rank = mat.rank()
                    print_success(f"Rank: {rank}")
                except Exception as e:
                    print_error(f"Error computing rank: {str(e)}")
                
                try:
                    cond_num = mat.condition_number()
                    print_success(f"Condition Number: {cond_num:.4f}")
                except Exception as e:
                    print_error(f"Error computing condition number: {str(e)}")
                
                try:
                    is_pd = mat.is_positive_definite()
                    print_success(f"Positive Definite: {is_pd}")
                except Exception as e:
                    print_error(f"Error checking positive definiteness: {str(e)}")
                
                try:
                    is_diag = mat.is_diagonalizable()
                    print_success(f"Diagonalizable: {is_diag}")
                except Exception as e:
                    print_error(f"Error checking diagonalizability: {str(e)}")
                
                try:
                    frobenius_norm = mat.norm('frobenius')
                    print_success(f"Frobenius Norm: {frobenius_norm:.4f}")
                except Exception as e:
                    print_error(f"Error computing Frobenius norm: {str(e)}")
            
            elif choice == '14':  # Matrix Decompositions
                chosen = manager.select_matrix()
                if chosen is None:
                    continue
                _, mat = chosen
                
                print_header("Matrix Decompositions")
                print("1. LU Decomposition")
                print("2. QR Decomposition")
                print("3. SVD (Singular Value Decomposition)")
                print("4. Cholesky Decomposition")
                print("0. Back")
                
                decomp_choice = input(f"\n{Fore.YELLOW}Select decomposition: {Style.RESET_ALL}").strip()
                
                try:
                    if decomp_choice == '1':
                        L, U = mat.lu_decomposition()
                        print_success("LU Decomposition:")
                        print("L matrix:")
                        print(L)
                        print("U matrix:")
                        print(U)
                        manager.store_result(L)
                        manager.store_result(U)
                    elif decomp_choice == '2':
                        Q, R = mat.qr_decomposition()
                        print_success("QR Decomposition:")
                        print("Q matrix:")
                        print(Q)
                        print("R matrix:")
                        print(R)
                        manager.store_result(Q)
                        manager.store_result(R)
                    elif decomp_choice == '3':
                        U, S, V = mat.svd()
                        print_success("SVD Decomposition:")
                        print("U matrix:")
                        print(U)
                        print("S matrix:")
                        print(S)
                        print("V matrix:")
                        print(V)
                        manager.store_result(U)
                        manager.store_result(S)
                        manager.store_result(V)
                    elif decomp_choice == '4':
                        L = mat.cholesky_decomposition()
                        print_success("Cholesky Decomposition:")
                        print("L matrix:")
                        print(L)
                        manager.store_result(L)
                except Exception as e:
                    print_error(f"Decomposition error: {str(e)}")
            
            elif choice == '15':  # Advanced Operations
                print_header("Advanced Operations")
                print("1. Hadamard Product (Element-wise)")
                print("2. Kronecker Product")
                print("3. Pseudoinverse (Moore-Penrose)")
                print("0. Back")
                
                adv_choice = input(f"\n{Fore.YELLOW}Select operation: {Style.RESET_ALL}").strip()
                
                try:
                    if adv_choice == '1':
                        print_info("Select first matrix:")
                        first = manager.select_matrix()
                        if first is None:
                            continue
                        _, mat1 = first
                        print_info("Select second matrix:")
                        second = manager.select_matrix()
                        if second is None:
                            continue
                        _, mat2 = second
                        result = mat1.hadamard_product(mat2)
                        print_success("Hadamard Product:")
                        print(result)
                        manager.store_result(result)
                    elif adv_choice == '2':
                        print_info("Select first matrix:")
                        first = manager.select_matrix()
                        if first is None:
                            continue
                        _, mat1 = first
                        print_info("Select second matrix:")
                        second = manager.select_matrix()
                        if second is None:
                            continue
                        _, mat2 = second
                        result = mat1.kronecker_product(mat2)
                        print_success("Kronecker Product:")
                        print(result)
                        manager.store_result(result)
                    elif adv_choice == '3':
                        chosen = manager.select_matrix()
                        if chosen is None:
                            continue
                        _, mat = chosen
                        result = mat.pseudoinverse()
                        print_success("Pseudoinverse:")
                        print(result)
                        manager.store_result(result)
                except Exception as e:
                    print_error(f"Advanced operation error: {str(e)}")

            elif choice == '0':
                break
            else:
                print_error("Invalid choice. Please try again.")
                
        except ValueError as e:
            print_error(f"Error: {str(e)}")
        except Exception as e:
            print_error(f"Unexpected error: {str(e)}")

def config_menu() -> None:
    """
    Display and handle the configuration menu.
    """
    while True:
        print_header("Configuration Settings")
        print(f"1. Precision: {Config.precision} decimal places")
        print(f"2. Default Export Format: {Config.default_export_format}")
        print(f"3. Colored Output: {'Enabled' if Config.colored_output else 'Disabled'}")
        print(f"4. Auto Save: {'Enabled' if Config.auto_save else 'Disabled'}")
        print(f"5. Save Directory: {Config.save_directory}")
        print(f"6. Show Progress: {'Enabled' if Config.show_progress else 'Disabled'}")
        print(f"7. Max History Size: {Config.max_history_size}")
        print(f"8. Enable Caching: {'Enabled' if Config.enable_caching else 'Disabled'}")
        print(f"9. Matrix Size Warning Threshold: {Config.matrix_size_warning_threshold}")
        print(f"10. Log Level: {Config.log_level}")
        print("11. Save Configuration to File")
        print("12. Load Configuration from File")
        print("0. Back to Main Menu")
        
        choice = input(f"\n{Fore.YELLOW}Select an option: {Style.RESET_ALL}").strip()
        
        if choice == '1':
            try:
                precision = int(input("Enter precision (decimal places): "))
                Config.precision = precision
                print_success(f"Precision set to {precision}")
            except ValueError:
                print_error("Invalid input. Please enter a number.")
        elif choice == '2':
            print("Available formats: csv, json, latex, numpy, matlab, text")
            fmt = input("Enter default export format: ").strip().lower()
            if fmt in ['csv', 'json', 'latex', 'numpy', 'matlab', 'text']:
                Config.default_export_format = fmt
                print_success(f"Default export format set to {fmt}")
            else:
                print_error("Invalid format.")
        elif choice == '3':
            Config.colored_output = not Config.colored_output
            print_success(f"Colored output {'enabled' if Config.colored_output else 'disabled'}")
        elif choice == '4':
            Config.auto_save = not Config.auto_save
            print_success(f"Auto save {'enabled' if Config.auto_save else 'disabled'}")
        elif choice == '5':
            directory = input("Enter save directory: ").strip()
            Config.save_directory = directory
            print_success(f"Save directory set to {directory}")
        elif choice == '6':
            Config.show_progress = not Config.show_progress
            print_success(f"Show progress {'enabled' if Config.show_progress else 'disabled'}")
        elif choice == '7':
            try:
                history_size = int(input("Enter max history size: "))
                if history_size > 0:
                    Config.max_history_size = history_size
                    print_success(f"Max history size set to {history_size}")
                else:
                    print_error("History size must be positive")
            except ValueError:
                print_error("Invalid input. Please enter a positive integer.")
        elif choice == '8':
            Config.enable_caching = not Config.enable_caching
            print_success(f"Caching {'enabled' if Config.enable_caching else 'disabled'}")
        elif choice == '9':
            try:
                threshold = int(input("Enter matrix size warning threshold: "))
                if threshold > 0:
                    Config.matrix_size_warning_threshold = threshold
                    print_success(f"Matrix size warning threshold set to {threshold}")
                else:
                    print_error("Threshold must be positive")
            except ValueError:
                print_error("Invalid input. Please enter a positive integer.")
        elif choice == '10':
            print("Available log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL")
            log_level = input("Enter log level: ").strip().upper()
            if log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                Config.log_level = log_level
                logger.setLevel(getattr(logging, log_level))
                print_success(f"Log level set to {log_level}")
            else:
                print_error("Invalid log level")
        elif choice == '11':
            filename = input("Enter filename to save config: ").strip() or "config.json"
            Config.save_to_file(filename)
        elif choice == '12':
            filename = input("Enter filename to load config: ").strip() or "config.json"
            Config.load_from_file(filename)
        elif choice == '0':
            break
        else:
            print_error("Invalid choice. Please try again.")

def main_menu() -> None:
    """
    Display and handle the main menu of the matrix calculator.
    """
    manager = MatrixManager()
    clear_screen()
    print_header("Matrix Calculator v2.2.0")
    print_info("Type '?' for help at any time")
    
    while True:
        print_header("Main Menu")
        print("1. Matrix Management")
        print("2. Matrix Operations")
        print("3. Configuration")
        print("4. Help")
        print("0. Exit")
        
        choice = input(f"\n{Fore.YELLOW}Select an option: {Style.RESET_ALL}").strip().lower()
        
        if choice == '?' or choice == 'help':
            manager.show_help()
        elif choice == 'clear':
            clear_screen()
        elif choice == 'history':
            manager.view_history()
        elif choice == 'config':
            config_menu()
        elif choice == 'exit' or choice == '0':
            print_success("\nThank you for using Matrix Calculator!")
            break
        elif choice == '1':
            management_menu(manager)
        elif choice == '2':
            operations_menu(manager)
        elif choice == '3':
            config_menu()
        elif choice == '4':
            manager.show_help()
        else:
            print_error("Invalid choice. Please try again.")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Matrix Calculator CLI - Advanced matrix operations with symbolic support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python Matrixcodes.py                                    # Interactive mode
  python Matrixcodes.py --import matrix.csv                # Import and show matrix
  python Matrixcodes.py --import A.csv B.csv --operation multiply --export result.json
  python Matrixcodes.py --config myconfig.json             # Load configuration
  python Matrixcodes.py --precision 6 --import matrix.csv --operation eigenvalues
        '''
    )
    
    parser.add_argument('--import', dest='import_files', nargs='+', metavar='FILE',
                        help='Import matrix/matrices from file(s)')
    parser.add_argument('--operation', choices=['add', 'subtract', 'multiply', 'transpose', 
                                                'inverse', 'determinant', 'trace', 'eigenvalues',
                                                'characteristic', 'power'],
                        help='Operation to perform on imported matrices')
    parser.add_argument('--export', metavar='FILE',
                        help='Export result to file')
    parser.add_argument('--format', choices=['csv', 'json', 'latex', 'numpy', 'matlab', 'text'],
                        help='Export format')
    parser.add_argument('--precision', type=int, metavar='N',
                        help='Set numerical precision (decimal places)')
    parser.add_argument('--config', metavar='FILE',
                        help='Load configuration from file')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    parser.add_argument('--exponent', type=float, metavar='N',
                        help='Exponent for power operation')
    
    return parser.parse_args()

def run_cli_mode(args: argparse.Namespace) -> None:
    """
    Run the calculator in non-interactive CLI mode.
    
    Args:
        args: Parsed command-line arguments.
    """
    manager = MatrixManager()
    
    # Load configuration if specified
    if args.config:
        Config.load_from_file(args.config)
    
    # Set precision if specified
    if args.precision:
        Config.precision = args.precision
    
    # Disable colors if requested
    if args.no_color:
        Config.colored_output = False
    
    # Import matrices
    if args.import_files:
        for idx, filename in enumerate(args.import_files):
            try:
                manager.import_matrix(filename)
                print_success(f"Imported matrix from {filename}")
            except Exception as e:
                print_error(f"Failed to import {filename}: {str(e)}")
                return
    
    # Perform operation if specified
    if args.operation and manager.matrices:
        try:
            matrix_names = list(manager.matrices.keys())
            
            if args.operation == 'add' and len(matrix_names) >= 2:
                result = manager.matrices[matrix_names[0]].add(manager.matrices[matrix_names[1]])
                print_success("Addition result:")
                print(result)
            elif args.operation == 'subtract' and len(matrix_names) >= 2:
                result = manager.matrices[matrix_names[0]].subtract(manager.matrices[matrix_names[1]])
                print_success("Subtraction result:")
                print(result)
            elif args.operation == 'multiply' and len(matrix_names) >= 2:
                result = manager.matrices[matrix_names[0]].multiply(manager.matrices[matrix_names[1]])
                print_success("Multiplication result:")
                print(result)
            elif args.operation == 'transpose':
                result = manager.matrices[matrix_names[0]].transpose()
                print_success("Transpose result:")
                print(result)
            elif args.operation == 'inverse':
                result = manager.matrices[matrix_names[0]].inverse()
                print_success("Inverse result:")
                print(result)
            elif args.operation == 'determinant':
                det = manager.matrices[matrix_names[0]].determinant()
                print_success(f"Determinant: {det}")
                result = None
            elif args.operation == 'trace':
                tr = manager.matrices[matrix_names[0]].trace()
                print_success(f"Trace: {tr}")
                result = None
            elif args.operation == 'eigenvalues':
                eigvals = manager.matrices[matrix_names[0]].eigenvalues(numeric=True)
                print_success("Eigenvalues:")
                for idx, val in enumerate(eigvals, 1):
                    print(f"  λ{idx} = {val}")
                result = None
            elif args.operation == 'characteristic':
                char_eq = manager.matrices[matrix_names[0]].characteristic_equation()
                print_success(f"Characteristic equation: {char_eq} = 0")
                result = None
            elif args.operation == 'power':
                if args.exponent is None:
                    print_error("--exponent required for power operation")
                    return
                result = manager.matrices[matrix_names[0]].power(args.exponent)
                print_success(f"Matrix^{args.exponent} result:")
                print(result)
            else:
                print_error(f"Operation '{args.operation}' requires appropriate number of matrices")
                return
            
            # Export result if specified
            if args.export and result is not None:
                name = manager._get_new_name()
                manager.matrices[name] = result
                manager.export_matrix(name, args.export, args.format)
        
        except Exception as e:
            print_error(f"Operation failed: {str(e)}")
    
    # Just show matrices if no operation specified
    elif manager.matrices and not args.operation:
        for name, matrix in manager.matrices.items():
            print_header(f"Matrix {name}")
            print(matrix)

if __name__ == '__main__':
    args = parse_arguments()
    
    # Check if running in CLI mode (any arguments provided)
    if len(sys.argv) > 1:
        run_cli_mode(args)
    else:
        # Interactive mode
        main_menu()
