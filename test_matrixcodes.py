#!/usr/bin/env python3
"""
Comprehensive unit tests for Matrix Calculator CLI v2.2.0

This test suite covers all functionality including:
- Matrix creation and validation
- All mathematical operations
- File I/O operations
- Configuration management
- Error handling
- New features: decompositions, properties, templates
"""

import unittest
import tempfile
import os
import sys
import json
import numpy as np
import sympy as sp

# Add the current directory to the path to import Matrixcodes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Matrixcodes import Matrix, MatrixManager, Config, ensure_file_extension, preprocess_expression


class TestMatrixClass(unittest.TestCase):
    """Test cases for the Matrix class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simple_matrix = Matrix([[1, 2], [3, 4]])
        self.symbolic_matrix = Matrix([['x', 0], [0, 'y']])
        self.large_matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    
    def test_matrix_creation(self):
        """Test matrix creation with various input types."""
        # Test with integers
        m1 = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m1.rows, 2)
        self.assertEqual(m1.cols, 2)
        
        # Test with floats
        m2 = Matrix([[1.5, 2.5], [3.5, 4.5]])
        self.assertEqual(m2.rows, 2)
        self.assertEqual(m2.cols, 2)
        
        # Test with symbolic expressions
        m3 = Matrix([['x', 'y'], ['z', 'w']])
        self.assertEqual(m3.rows, 2)
        self.assertEqual(m3.cols, 2)
    
    def test_matrix_creation_errors(self):
        """Test matrix creation error handling."""
        # Empty matrix
        with self.assertRaises(ValueError):
            Matrix([])
        
        # Inconsistent row lengths
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3]])
        
        # Invalid expressions
        with self.assertRaises(Exception):
            Matrix([['invalid_expression_@#$%']])
    
    def test_basic_operations(self):
        """Test basic matrix operations."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        
        # Addition
        result = m1.add(m2)
        expected = Matrix([[6, 8], [10, 12]])
        self.assertEqual(result.data, expected.data)
        
        # Subtraction
        result = m1.subtract(m2)
        expected = Matrix([[-4, -4], [-4, -4]])
        self.assertEqual(result.data, expected.data)
        
        # Scalar multiplication
        result = m1.multiply(2)
        expected = Matrix([[2, 4], [6, 8]])
        self.assertEqual(result.data, expected.data)
        
        # Matrix multiplication
        result = m1.multiply(m2)
        expected = Matrix([[19, 22], [43, 50]])
        self.assertEqual(result.data, expected.data)
    
    def test_transpose(self):
        """Test matrix transpose."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        result = m.transpose()
        expected = Matrix([[1, 4], [2, 5], [3, 6]])
        self.assertEqual(result.data, expected.data)
    
    def test_determinant(self):
        """Test determinant calculation."""
        m = Matrix([[1, 2], [3, 4]])
        det = m.determinant()
        self.assertEqual(det, -2)
        
        # Test caching
        det2 = m.determinant()
        self.assertEqual(det, det2)
    
    def test_inverse(self):
        """Test matrix inverse."""
        m = Matrix([[1, 2], [3, 4]])
        result = m.inverse()
        # Check that A * A^(-1) = I
        identity = m.multiply(result)
        expected = Matrix([[1, 0], [0, 1]])
        self.assertEqual(identity.data, expected.data)
    
    def test_trace(self):
        """Test matrix trace."""
        m = Matrix([[1, 2], [3, 4]])
        trace = m.trace()
        self.assertEqual(trace, 5)
    
    def test_eigenvalues(self):
        """Test eigenvalue calculation."""
        m = Matrix([[1, 0], [0, 2]])
        eigenvals = m.eigenvalues()
        self.assertEqual(len(eigenvals), 2)
    
    def test_characteristic_equation(self):
        """Test characteristic equation."""
        m = Matrix([[1, 0], [0, 2]])
        char_eq = m.characteristic_equation()
        self.assertIsInstance(char_eq, sp.Expr)
    
    def test_matrix_properties(self):
        """Test matrix property checks."""
        # Symmetric matrix
        sym_matrix = Matrix([[1, 2], [2, 1]])
        self.assertTrue(sym_matrix.is_symmetric())
        
        # Non-symmetric matrix
        non_sym_matrix = Matrix([[1, 2], [3, 4]])
        self.assertFalse(non_sym_matrix.is_symmetric())
        
        # Square matrix
        self.assertTrue(self.simple_matrix.is_square())
        self.assertFalse(self.large_matrix.is_square())
    
    def test_rank(self):
        """Test matrix rank calculation."""
        m = Matrix([[1, 2], [3, 4]])
        rank = m.rank()
        self.assertEqual(rank, 2)
        
        # Test caching
        rank2 = m.rank()
        self.assertEqual(rank, rank2)
    
    def test_condition_number(self):
        """Test condition number calculation."""
        m = Matrix([[1, 0], [0, 2]])
        cond = m.condition_number()
        self.assertIsInstance(cond, float)
        self.assertGreater(cond, 0)
    
    def test_norm(self):
        """Test matrix norm calculation."""
        m = Matrix([[3, 4], [0, 0]])
        frobenius_norm = m.norm('frobenius')
        self.assertEqual(frobenius_norm, 5.0)  # sqrt(3^2 + 4^2)
        
        # Test different norm types
        l1_norm = m.norm('L1')
        l2_norm = m.norm('L2')
        inf_norm = m.norm('inf')
        
        self.assertIsInstance(l1_norm, float)
        self.assertIsInstance(l2_norm, float)
        self.assertIsInstance(inf_norm, float)
    
    def test_hadamard_product(self):
        """Test Hadamard (element-wise) product."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        result = m1.hadamard_product(m2)
        expected = Matrix([[5, 12], [21, 32]])
        self.assertEqual(result.data, expected.data)
    
    def test_kronecker_product(self):
        """Test Kronecker product."""
        m1 = Matrix([[1, 2]])
        m2 = Matrix([[3], [4]])
        result = m1.kronecker_product(m2)
        expected = Matrix([[3, 6], [4, 8]])
        self.assertEqual(result.data, expected.data)
    
    def test_pseudoinverse(self):
        """Test pseudoinverse calculation."""
        m = Matrix([[1, 2], [3, 4]])
        pinv = m.pseudoinverse()
        self.assertIsInstance(pinv, Matrix)
    
    def test_lu_decomposition(self):
        """Test LU decomposition."""
        m = Matrix([[2, 1], [1, 1]])
        L, U = m.lu_decomposition()
        self.assertIsInstance(L, Matrix)
        self.assertIsInstance(U, Matrix)
    
    def test_qr_decomposition(self):
        """Test QR decomposition."""
        m = Matrix([[1, 2], [3, 4]])
        Q, R = m.qr_decomposition()
        self.assertIsInstance(Q, Matrix)
        self.assertIsInstance(R, Matrix)
    
    def test_svd(self):
        """Test SVD decomposition."""
        m = Matrix([[1, 2], [3, 4]])
        U, S, V = m.svd()
        self.assertIsInstance(U, Matrix)
        self.assertIsInstance(S, Matrix)
        self.assertIsInstance(V, Matrix)
    
    def test_cholesky_decomposition(self):
        """Test Cholesky decomposition."""
        # Create a positive definite matrix
        m = Matrix([[4, 2], [2, 3]])
        L = m.cholesky_decomposition()
        self.assertIsInstance(L, Matrix)
    
    def test_preview(self):
        """Test matrix preview for large matrices."""
        # Create a large matrix
        data = [[i + j for j in range(10)] for i in range(10)]
        large_matrix = Matrix(data)
        preview = large_matrix.preview(max_rows=3, max_cols=3)
        self.assertIn("Matrix Preview", preview)
        self.assertIn("10x10", preview)


class TestMatrixManager(unittest.TestCase):
    """Test cases for the MatrixManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = MatrixManager()
    
    def test_matrix_creation_and_storage(self):
        """Test matrix creation and storage."""
        # Create a test matrix
        test_data = [[1, 2], [3, 4]]
        matrix = Matrix(test_data)
        name = self.manager._get_new_name()
        self.manager.matrices[name] = matrix
        
        self.assertIn(name, self.manager.matrices)
        self.assertEqual(self.manager.matrices[name].data, matrix.data)
    
    def test_matrix_templates(self):
        """Test matrix template generation."""
        # Test identity matrix
        self.manager.create_identity_matrix(3)
        identity_name = list(self.manager.matrices.keys())[-1]
        identity_matrix = self.manager.matrices[identity_name]
        self.assertEqual(identity_matrix.rows, 3)
        self.assertEqual(identity_matrix.cols, 3)
        
        # Test zeros matrix
        self.manager.create_zeros_matrix(2, 3)
        zeros_name = list(self.manager.matrices.keys())[-1]
        zeros_matrix = self.manager.matrices[zeros_name]
        self.assertEqual(zeros_matrix.rows, 2)
        self.assertEqual(zeros_matrix.cols, 3)
        
        # Test ones matrix
        self.manager.create_ones_matrix(2, 2)
        ones_name = list(self.manager.matrices.keys())[-1]
        ones_matrix = self.manager.matrices[ones_name]
        self.assertEqual(ones_matrix.rows, 2)
        self.assertEqual(ones_matrix.cols, 2)
        
        # Test random matrix
        self.manager.create_random_matrix(2, 2, 0, 1)
        random_name = list(self.manager.matrices.keys())[-1]
        random_matrix = self.manager.matrices[random_name]
        self.assertEqual(random_matrix.rows, 2)
        self.assertEqual(random_matrix.cols, 2)
        
        # Test diagonal matrix
        self.manager.create_diagonal_matrix([1, 2, 3])
        diagonal_name = list(self.manager.matrices.keys())[-1]
        diagonal_matrix = self.manager.matrices[diagonal_name]
        self.assertEqual(diagonal_matrix.rows, 3)
        self.assertEqual(diagonal_matrix.cols, 3)
    
    def test_history_tracking(self):
        """Test operation history tracking."""
        initial_history_length = len(self.manager.history)
        
        # Create a matrix
        self.manager.create_identity_matrix(2)
        
        # Check history was updated
        self.assertEqual(len(self.manager.history), initial_history_length + 1)
        self.assertEqual(self.manager.history[-1][0], "create_identity")


class TestConfiguration(unittest.TestCase):
    """Test cases for the Config class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset config to defaults
        Config.precision = 4
        Config.default_export_format = 'csv'
        Config.colored_output = True
        Config.auto_save = False
        Config.save_directory = './matrices'
        Config.show_progress = True
        Config.max_history_size = 20
        Config.enable_caching = True
        Config.matrix_size_warning_threshold = 1000
        Config.recent_files_list = []
        Config.log_level = 'INFO'
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        self.assertTrue(Config.validate())
        
        # Invalid precision
        Config.precision = -1
        with self.assertRaises(ValueError):
            Config.validate()
        
        # Reset precision
        Config.precision = 4
        
        # Invalid export format
        Config.default_export_format = 'invalid'
        with self.assertRaises(ValueError):
            Config.validate()
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
        
        try:
            # Save configuration
            Config.save_to_file(temp_filename)
            
            # Modify configuration
            Config.precision = 8
            Config.default_export_format = 'json'
            
            # Load configuration
            Config.load_from_file(temp_filename)
            
            # Check that values were restored
            self.assertEqual(Config.precision, 4)
            self.assertEqual(Config.default_export_format, 'csv')
        
        finally:
            # Clean up
            os.unlink(temp_filename)


class TestFileOperations(unittest.TestCase):
    """Test cases for file operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_matrix = Matrix([[1, 2], [3, 4]])
    
    def test_csv_operations(self):
        """Test CSV import and export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_filename = f.name
        
        try:
            # Export to CSV
            self.test_matrix.to_csv(temp_filename)
            
            # Import from CSV
            imported_matrix = Matrix.from_csv(temp_filename)
            
            # Check that matrices are equal
            self.assertEqual(self.test_matrix.data, imported_matrix.data)
        
        finally:
            os.unlink(temp_filename)
    
    def test_json_operations(self):
        """Test JSON import and export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
        
        try:
            # Export to JSON
            self.test_matrix.to_json(temp_filename, "test_matrix")
            
            # Import from JSON
            imported_matrix, name = Matrix.from_json(temp_filename)
            
            # Check that matrices are equal
            self.assertEqual(self.test_matrix.data, imported_matrix.data)
            self.assertEqual(name, "test_matrix")
        
        finally:
            os.unlink(temp_filename)
    
    def test_latex_export(self):
        """Test LaTeX export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
            temp_filename = f.name
        
        try:
            # Export to LaTeX
            self.test_matrix.to_latex(temp_filename)
            
            # Check that file was created and contains LaTeX
            with open(temp_filename, 'r') as f:
                content = f.read()
                self.assertIn('\\begin{bmatrix}', content)
                self.assertIn('\\end{bmatrix}', content)
        
        finally:
            os.unlink(temp_filename)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_ensure_file_extension(self):
        """Test file extension helper function."""
        # Test with extension
        filename = "test.csv"
        result = ensure_file_extension(filename, ".json")
        self.assertEqual(result, "test.csv")
        
        # Test without extension
        filename = "test"
        result = ensure_file_extension(filename, ".csv")
        self.assertEqual(result, "test.csv")
        
        # Test empty filename
        with self.assertRaises(ValueError):
            ensure_file_extension("", ".csv")
    
    def test_preprocess_expression(self):
        """Test expression preprocessing."""
        # Test power conversion
        expr = "x^2"
        result = preprocess_expression(expr)
        self.assertEqual(result, "x**2")
        
        # Test Euler's number conversion
        expr = "e^x"
        result = preprocess_expression(expr)
        self.assertEqual(result, "E**x")
        
        # Test dangerous patterns
        with self.assertRaises(ValueError):
            preprocess_expression("__import__('os')")
        
        with self.assertRaises(ValueError):
            preprocess_expression("exec('malicious code')")


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling."""
    
    def test_matrix_operation_errors(self):
        """Test error handling in matrix operations."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2, 3], [4, 5, 6]])
        m3 = Matrix([[1, 2], [3, 4], [5, 6]])  # 3x2 matrix
        
        # Test dimension mismatch in addition
        with self.assertRaises(ValueError):
            m1.add(m2)
        
        # Test dimension mismatch in multiplication (3x2 * 2x3 should work, but 2x2 * 3x2 should fail)
        with self.assertRaises(ValueError):
            m1.multiply(m3)  # 2x2 * 3x2 - invalid
        
        # Test non-square matrix operations
        with self.assertRaises(ValueError):
            m2.determinant()
        
        with self.assertRaises(ValueError):
            m2.inverse()
    
    def test_decomposition_errors(self):
        """Test error handling in decompositions."""
        # Test non-square matrix for LU decomposition
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            m.lu_decomposition()
        
        # Test non-positive definite matrix for Cholesky
        m = Matrix([[1, 2], [2, 1]])  # Not positive definite
        with self.assertRaises(ValueError):
            m.cholesky_decomposition()


class TestCaching(unittest.TestCase):
    """Test cases for caching functionality."""
    
    def test_determinant_caching(self):
        """Test determinant caching."""
        m = Matrix([[1, 2], [3, 4]])
        
        # First computation
        det1 = m.determinant()
        
        # Second computation should use cache
        det2 = m.determinant()
        
        self.assertEqual(det1, det2)
        
        # Cache should be populated
        self.assertIsNotNone(m._det_cache)
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        m = Matrix([[1, 2], [3, 4]])
        
        # Compute determinant to populate cache
        m.determinant()
        self.assertIsNotNone(m._det_cache)
        
        # Invalidate cache
        m._invalidate_caches()
        self.assertIsNone(m._det_cache)


if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2)
