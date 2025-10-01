"""Matrix manager for handling multiple matrices."""

from typing import Dict, List, Tuple, Optional
import random

from .matrix import Matrix
from ..config.settings import Config
from ..logging.setup import get_logger

logger = get_logger(__name__)


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
        self.matrices: Dict[str, Matrix] = {}
        self.counter = 0
        self.history: List[Tuple[str, str, Optional[Matrix]]] = []
    
    def _get_new_name(self) -> str:
        """Generate a new unique name for a matrix."""
        if self.counter < 26:
            name = chr(65 + self.counter)
        else:
            name = chr(65 + (self.counter % 26)) + str(self.counter // 26)
        self.counter += 1
        return name
    
    def _add_to_history(self, operation: str, matrix_name: str, matrix: Optional[Matrix] = None) -> None:
        """Add operation to history."""
        self.history.append((operation, matrix_name, matrix))
        
        # Limit history size
        if len(self.history) > Config.max_history_size:
            self.history = self.history[-Config.max_history_size:]
    
    def create_matrix(self, data: List[List], name: Optional[str] = None) -> str:
        """Create a new matrix with the given data."""
        if name is None:
            name = self._get_new_name()
        
        matrix = Matrix(data)
        self.matrices[name] = matrix
        self._add_to_history("create", name)
        
        logger.info(f"Created matrix '{name}' with dimensions {matrix.rows}x{matrix.cols}")
        return name
    
    def create_identity_matrix(self, size: int, name: Optional[str] = None) -> str:
        """Create an identity matrix."""
        if name is None:
            name = self._get_new_name()
        
        data = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
        matrix = Matrix(data)
        self.matrices[name] = matrix
        self._add_to_history("create_identity", name)
        
        logger.info(f"Created identity matrix '{name}' with size {size}x{size}")
        return name
    
    def create_zeros_matrix(self, rows: int, cols: int, name: Optional[str] = None) -> str:
        """Create a zeros matrix."""
        if name is None:
            name = self._get_new_name()
        
        data = [[0 for _ in range(cols)] for _ in range(rows)]
        matrix = Matrix(data)
        self.matrices[name] = matrix
        self._add_to_history("create_zeros", name)
        
        logger.info(f"Created zeros matrix '{name}' with dimensions {rows}x{cols}")
        return name
    
    def create_ones_matrix(self, rows: int, cols: int, name: Optional[str] = None) -> str:
        """Create a ones matrix."""
        if name is None:
            name = self._get_new_name()
        
        data = [[1 for _ in range(cols)] for _ in range(rows)]
        matrix = Matrix(data)
        self.matrices[name] = matrix
        self._add_to_history("create_ones", name)
        
        logger.info(f"Created ones matrix '{name}' with dimensions {rows}x{cols}")
        return name
    
    def create_random_matrix(self, rows: int, cols: int, min_val: float = 0, max_val: float = 1, name: Optional[str] = None) -> str:
        """Create a random matrix."""
        if name is None:
            name = self._get_new_name()
        
        data = [[random.uniform(min_val, max_val) for _ in range(cols)] for _ in range(rows)]
        matrix = Matrix(data)
        self.matrices[name] = matrix
        self._add_to_history("create_random", name)
        
        logger.info(f"Created random matrix '{name}' with dimensions {rows}x{cols}")
        return name
    
    def create_diagonal_matrix(self, diagonal_elements: List[float], name: Optional[str] = None) -> str:
        """Create a diagonal matrix."""
        if name is None:
            name = self._get_new_name()
        
        size = len(diagonal_elements)
        data = [[diagonal_elements[i] if i == j else 0 for j in range(size)] for i in range(size)]
        matrix = Matrix(data)
        self.matrices[name] = matrix
        self._add_to_history("create_diagonal", name)
        
        logger.info(f"Created diagonal matrix '{name}' with size {size}x{size}")
        return name
    
    def delete_matrix(self, name: str) -> bool:
        """Delete a matrix by name."""
        if name not in self.matrices:
            logger.warning(f"Attempted to delete non-existent matrix '{name}'")
            return False
        
        # Store matrix for undo
        matrix = self.matrices[name]
        self._add_to_history("delete", name, matrix)
        
        del self.matrices[name]
        logger.info(f"Deleted matrix '{name}'")
        return True
    
    def get_matrix(self, name: str) -> Optional[Matrix]:
        """Get a matrix by name."""
        return self.matrices.get(name)
    
    def list_matrices(self) -> Dict[str, Matrix]:
        """Get all matrices."""
        return self.matrices.copy()
    
    def get_matrix_info(self) -> List[Tuple[str, int, int]]:
        """Get information about all matrices."""
        return [(name, matrix.rows, matrix.cols) for name, matrix in self.matrices.items()]
    
    def clear_all_matrices(self) -> None:
        """Clear all matrices."""
        self.matrices.clear()
        self.history.clear()
        self.counter = 0
        logger.info("Cleared all matrices")
    
    def get_history(self) -> List[Tuple[str, str, Optional[Matrix]]]:
        """Get operation history."""
        return self.history.copy()
    
    def undo_last_operation(self) -> bool:
        """Undo the last operation."""
        if not self.history:
            return False
        
        operation, matrix_name, matrix_data = self.history.pop()
        
        if operation == "delete" and matrix_data is not None:
            # Restore deleted matrix
            self.matrices[matrix_name] = matrix_data
            logger.info(f"Restored matrix '{matrix_name}'")
        elif operation.startswith("create"):
            # Remove created matrix
            if matrix_name in self.matrices:
                del self.matrices[matrix_name]
                logger.info(f"Removed matrix '{matrix_name}'")
        
        return True
