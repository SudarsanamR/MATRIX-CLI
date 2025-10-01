"""File format handlers for matrix I/O operations."""

import csv
import json
import numpy as np
import sympy as sp
import scipy.io as sio
from typing import Tuple, Optional, Any
from abc import ABC, abstractmethod

import os
from pathlib import Path
from .utils import ensure_parent_dir, write_text_atomic, write_bytes_atomic
from ..logging.setup import get_logger
from ..core.matrix import Matrix

# Optional imports for additional formats
try:
    import pandas as pd
    import openpyxl
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

logger = get_logger(__name__)


class FormatHandler(ABC):
    """Abstract base class for format handlers."""
    
    @abstractmethod
    def load(self, filepath: str) -> Tuple[Matrix, Optional[str]]:
        """Load matrix from file."""
        pass
    
    @abstractmethod
    def save(self, matrix: Matrix, filepath: str, name: Optional[str] = None) -> None:
        """Save matrix to file."""
        pass


class CSVHandler(FormatHandler):
    """CSV format handler."""
    
    def load(self, filepath: str) -> Tuple[Matrix, Optional[str]]:
        """Load matrix from CSV file."""
        try:
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                data = [[elem.strip() for elem in row] for row in reader if row]
            
            matrix = Matrix(data)
            logger.debug(f"Loaded CSV matrix from {filepath}")
            return matrix, None
            
        except Exception as e:
            logger.error(f"Error loading CSV from {filepath}: {str(e)}")
            raise ValueError(f"Error loading from CSV: {str(e)}")
    
    def save(self, matrix: Matrix, filepath: str, name: Optional[str] = None) -> None:
        """Save matrix to CSV file."""
        try:
            ensure_parent_dir(filepath)
            dir_path = os.path.dirname(filepath) or '.'
            
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', newline='', dir=dir_path, delete=False) as tf:
                writer = csv.writer(tf)
                for row in matrix.data:
                    writer.writerow([str(elem) for elem in row])
                temp_name = tf.name
            
            import os
            os.replace(temp_name, filepath)
            logger.debug(f"Saved CSV matrix to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving CSV to {filepath}: {str(e)}")
            raise ValueError(f"Error exporting to CSV: {str(e)}")


class JSONHandler(FormatHandler):
    """JSON format handler."""
    
    def load(self, filepath: str) -> Tuple[Matrix, Optional[str]]:
        """Load matrix from JSON file."""
        try:
            with open(filepath, 'r') as f:
                json_data = json.load(f)
            
            data = json_data.get('data', json_data)
            name = json_data.get('name', None)
            
            matrix = Matrix(data)
            logger.debug(f"Loaded JSON matrix from {filepath}")
            return matrix, name
            
        except Exception as e:
            logger.error(f"Error loading JSON from {filepath}: {str(e)}")
            raise ValueError(f"Error loading from JSON: {str(e)}")
    
    def save(self, matrix: Matrix, filepath: str, name: Optional[str] = None) -> None:
        """Save matrix to JSON file."""
        try:
            json_data = {
                'data': [[str(elem) for elem in row] for row in matrix.data],
                'rows': matrix.rows,
                'cols': matrix.cols
            }
            if name:
                json_data['name'] = name
            
            ensure_parent_dir(filepath)
            dir_path = os.path.dirname(filepath) or '.'
            
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=dir_path, delete=False) as tf:
                json.dump(json_data, tf, indent=2)
                temp_name = tf.name
            
            import os
            os.replace(temp_name, filepath)
            logger.debug(f"Saved JSON matrix to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving JSON to {filepath}: {str(e)}")
            raise ValueError(f"Error exporting to JSON: {str(e)}")


class LaTeXHandler(FormatHandler):
    """LaTeX format handler."""
    
    def load(self, filepath: str) -> Tuple[Matrix, Optional[str]]:
        """Load matrix from LaTeX file (not supported)."""
        raise ValueError("LaTeX import not supported")
    
    def save(self, matrix: Matrix, filepath: str, name: Optional[str] = None) -> None:
        """Save matrix to LaTeX file."""
        try:
            latex_str = "\\begin{bmatrix}\n"
            for i, row in enumerate(matrix.data):
                row_str = " & ".join([sp.latex(elem) for elem in row])
                latex_str += "  " + row_str
                if i < matrix.rows - 1:
                    latex_str += " \\\\\n"
                else:
                    latex_str += "\n"
            latex_str += "\\end{bmatrix}"
            
            write_text_atomic(filepath, latex_str)
            logger.debug(f"Saved LaTeX matrix to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving LaTeX to {filepath}: {str(e)}")
            raise ValueError(f"Error exporting to LaTeX: {str(e)}")


class NumPyHandler(FormatHandler):
    """NumPy format handler."""
    
    def load(self, filepath: str) -> Tuple[Matrix, Optional[str]]:
        """Load matrix from NumPy file."""
        try:
            data = np.load(filepath)
            matrix = Matrix(data.tolist())
            logger.debug(f"Loaded NumPy matrix from {filepath}")
            return matrix, None
            
        except Exception as e:
            logger.error(f"Error loading NumPy from {filepath}: {str(e)}")
            raise ValueError(f"Error loading from NumPy file: {str(e)}")
    
    def save(self, matrix: Matrix, filepath: str, name: Optional[str] = None) -> None:
        """Save matrix to NumPy file."""
        try:
            # Convert to float for numerical storage
            numeric_data = [[float(sp.N(elem)) for elem in row] for row in matrix.data]
            np_array = np.array(numeric_data)
            
            ensure_parent_dir(filepath)
            dir_path = os.path.dirname(filepath) or '.'
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.npy', dir=dir_path, delete=False) as tf:
                temp_name = tf.name
            
            np.save(temp_name, np_array)
            
            import os
            os.replace(temp_name, filepath)
            logger.debug(f"Saved NumPy matrix to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving NumPy to {filepath}: {str(e)}")
            raise ValueError(f"Error exporting to NumPy file: {str(e)}")


class MATLABHandler(FormatHandler):
    """MATLAB format handler."""
    
    def load(self, filepath: str, variable_name: str = 'matrix') -> Tuple[Matrix, Optional[str]]:
        """Load matrix from MATLAB file."""
        try:
            mat_data = sio.loadmat(filepath)
            data = mat_data[variable_name]
            matrix = Matrix(data.tolist())
            logger.debug(f"Loaded MATLAB matrix from {filepath}")
            return matrix, None
            
        except Exception as e:
            logger.error(f"Error loading MATLAB from {filepath}: {str(e)}")
            raise ValueError(f"Error loading from MATLAB file: {str(e)}")
    
    def save(self, matrix: Matrix, filepath: str, name: Optional[str] = None, variable_name: str = 'matrix') -> None:
        """Save matrix to MATLAB file."""
        try:
            # Convert to float for numerical storage
            numeric_data = [[float(sp.N(elem)) for elem in row] for row in matrix.data]
            np_array = np.array(numeric_data)
            
            ensure_parent_dir(filepath)
            dir_path = os.path.dirname(filepath) or '.'
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mat', dir=dir_path, delete=False) as tf:
                temp_name = tf.name
            
            sio.savemat(temp_name, {variable_name: np_array})
            
            import os
            os.replace(temp_name, filepath)
            logger.debug(f"Saved MATLAB matrix to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving MATLAB to {filepath}: {str(e)}")
            raise ValueError(f"Error exporting to MATLAB file: {str(e)}")


class TextHandler(FormatHandler):
    """Plain text format handler."""
    
    def load(self, filepath: str) -> Tuple[Matrix, Optional[str]]:
        """Load matrix from text file."""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            data = []
            for line in lines:
                line = line.strip()
                if line:
                    # Split by whitespace
                    elements = line.split()
                    data.append(elements)
            
            matrix = Matrix(data)
            logger.debug(f"Loaded text matrix from {filepath}")
            return matrix, None
            
        except Exception as e:
            logger.error(f"Error loading text from {filepath}: {str(e)}")
            raise ValueError(f"Error loading from text file: {str(e)}")
    
    def save(self, matrix: Matrix, filepath: str, name: Optional[str] = None) -> None:
        """Save matrix to text file."""
        try:
            lines = []
            for row in matrix.data:
                line = ' '.join([str(elem) for elem in row])
                lines.append(line)
            
            content = '\n'.join(lines) + '\n'
            write_text_atomic(filepath, content)
            logger.debug(f"Saved text matrix to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving text to {filepath}: {str(e)}")
            raise ValueError(f"Error exporting to text file: {str(e)}")


class ExcelHandler(FormatHandler):
    """Excel format handler."""
    
    def load(self, filepath: str, sheet_name: str = 'Sheet1') -> Tuple[Matrix, Optional[str]]:
        """Load matrix from Excel file."""
        if not PANDAS_AVAILABLE:
            raise ValueError("pandas and openpyxl required for Excel support")
        
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
            data = df.values.tolist()
            matrix = Matrix(data)
            logger.debug(f"Loaded Excel matrix from {filepath}")
            return matrix, None
            
        except Exception as e:
            logger.error(f"Error loading Excel from {filepath}: {str(e)}")
            raise ValueError(f"Error loading from Excel file: {str(e)}")
    
    def save(self, matrix: Matrix, filepath: str, name: Optional[str] = None, sheet_name: str = 'Matrix') -> None:
        """Save matrix to Excel file."""
        if not PANDAS_AVAILABLE:
            raise ValueError("pandas and openpyxl required for Excel support")
        
        try:
            # Convert to string data for Excel
            string_data = [[str(elem) for elem in row] for row in matrix.data]
            df = pd.DataFrame(string_data)
            
            ensure_parent_dir(filepath)
            df.to_excel(filepath, sheet_name=sheet_name, index=False, header=False)
            logger.debug(f"Saved Excel matrix to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving Excel to {filepath}: {str(e)}")
            raise ValueError(f"Error exporting to Excel file: {str(e)}")


class ParquetHandler(FormatHandler):
    """Parquet format handler for efficient columnar storage."""
    
    def load(self, filepath: str) -> Tuple[Matrix, Optional[str]]:
        """Load matrix from Parquet file."""
        if not PYARROW_AVAILABLE:
            raise ValueError("pyarrow required for Parquet support")
        
        try:
            table = pq.read_table(filepath)
            df = table.to_pandas()
            data = df.values.tolist()
            matrix = Matrix(data)
            logger.debug(f"Loaded Parquet matrix from {filepath}")
            return matrix, None
            
        except Exception as e:
            logger.error(f"Error loading Parquet from {filepath}: {str(e)}")
            raise ValueError(f"Error loading from Parquet file: {str(e)}")
    
    def save(self, matrix: Matrix, filepath: str, name: Optional[str] = None) -> None:
        """Save matrix to Parquet file."""
        if not PYARROW_AVAILABLE:
            raise ValueError("pyarrow required for Parquet support")
        
        try:
            # Convert to float for numerical storage
            numeric_data = [[float(sp.N(elem)) for elem in row] for row in matrix.data]
            
            # Create column names
            column_names = [f'col_{i}' for i in range(matrix.cols)]
            
            # Create PyArrow table
            arrays = [pa.array([row[i] for row in numeric_data]) for i in range(matrix.cols)]
            table = pa.table(arrays, names=column_names)
            
            ensure_parent_dir(filepath)
            pq.write_table(table, filepath)
            logger.debug(f"Saved Parquet matrix to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving Parquet to {filepath}: {str(e)}")
            raise ValueError(f"Error exporting to Parquet file: {str(e)}")


# Format registry
_FORMAT_HANDLERS = {
    'csv': CSVHandler(),
    'json': JSONHandler(),
    'latex': LaTeXHandler(),
    'numpy': NumPyHandler(),
    'matlab': MATLABHandler(),
    'text': TextHandler()
}

# Add optional format handlers if dependencies are available
if PANDAS_AVAILABLE:
    _FORMAT_HANDLERS['excel'] = ExcelHandler()

if PYARROW_AVAILABLE:
    _FORMAT_HANDLERS['parquet'] = ParquetHandler()


def get_handler_by_format(format_name: str) -> FormatHandler:
    """Get handler for the specified format."""
    if format_name not in _FORMAT_HANDLERS:
        raise ValueError(f"Unsupported format: {format_name}")
    return _FORMAT_HANDLERS[format_name]


def get_supported_formats() -> list[str]:
    """Get list of supported formats."""
    return list(_FORMAT_HANDLERS.keys())
