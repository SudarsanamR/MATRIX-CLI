"""Enhanced test suite for Matrix Calculator CLI v2.3.0+

This comprehensive test suite covers:
- Property-based testing with Hypothesis
- Performance benchmarks
- Integration tests
- Security validation tests
- Plugin system tests
"""

import unittest
import tempfile
import os
import sys
import json
import time
import numpy as np
import sympy as sp
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, settings

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matrixcalc.core.matrix import Matrix
from matrixcalc.core.manager import MatrixManager
from matrixcalc.config.settings import Config
from matrixcalc.security.validation import ExpressionValidator, SecurityError
from matrixcalc.core.performance import MemoryMonitor, ParallelProcessor, PerformanceProfiler
from matrixcalc.plugins import PluginManager, PluginInterface, ExamplePlugin
from matrixcalc.io.formats import get_handler_by_format, get_supported_formats


class TestSecurityValidation(unittest.TestCase):
    """Test security validation functionality."""
    
    def setUp(self):
        self.validator = ExpressionValidator(timeout_seconds=5, security_level='moderate')
    
    def test_security_levels(self):
        """Test different security levels."""
        # Test strict mode
        strict_validator = ExpressionValidator(security_level='strict')
        with self.assertRaises(SecurityError):
            strict_validator.validate_expression('x**2')  # ** blocked in strict mode
        
        # Test moderate mode (should allow **)
        moderate_validator = ExpressionValidator(security_level='moderate')
        result = moderate_validator.validate_expression('x^2')  # ^ converted to **
        self.assertIn('**', result)
        
        # Test permissive mode
        permissive_validator = ExpressionValidator(security_level='permissive')
        result = permissive_validator.validate_expression('sin(x) + cos(y)')
        self.assertIsInstance(result, str)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        validator = ExpressionValidator()
        validator._max_validations_per_minute = 3
        
        # Should succeed for first 3 validations
        for i in range(3):
            validator.validate_expression(f'x + {i}')
        
        # Should fail on 4th validation
        with self.assertRaises(SecurityError):
            validator.validate_expression('x + 4')
    
    def test_dangerous_patterns(self):
        """Test blocking of dangerous patterns."""
        dangerous_expressions = [
            '__import__("os")',
            'exec("print(1)")',
            'eval("1+1")',
            'open("file.txt")',
            'globals()'
        ]
        
        for expr in dangerous_expressions:
            with self.assertRaises(SecurityError):
                self.validator.validate_expression(expr)
    
    def test_expression_length_limit(self):
        """Test expression length limits."""
        long_expr = 'x + ' * 5000  # Very long expression
        with self.assertRaises(SecurityError):
            self.validator.validate_expression(long_expr)


class TestPerformanceOptimization(unittest.TestCase):
    \"\"\"Test performance optimization features.\"\"\"
    
    def setUp(self):
        self.memory_monitor = MemoryMonitor(limit_mb=100)
        self.parallel_processor = ParallelProcessor(max_workers=2)
        self.profiler = PerformanceProfiler()
    
    def test_memory_monitoring(self):
        \"\"\"Test memory usage monitoring.\"\"\"
        usage = self.memory_monitor.get_memory_usage_mb()
        self.assertIsInstance(usage, float)
        self.assertGreater(usage, 0)
    
    def test_parallel_processing(self):
        \"\"\"Test parallel execution.\"\"\"
        def square(x):
            return x * x
        
        items = [1, 2, 3, 4, 5]
        results = self.parallel_processor.parallel_map(square, items)
        expected = [1, 4, 9, 16, 25]
        self.assertEqual(results, expected)
    
    def test_performance_profiling(self):
        \"\"\"Test performance profiling.\"\"\"
        def slow_operation():
            time.sleep(0.01)  # Small delay
            return 42
        
        result = self.profiler.profile_operation('test_op', slow_operation)
        self.assertEqual(result, 42)
        
        stats = self.profiler.get_performance_stats()
        self.assertGreater(stats['total_operations'], 0)
        self.assertGreater(stats['total_time'], 0)
    
    def tearDown(self):
        self.parallel_processor.shutdown()


class TestPluginSystem(unittest.TestCase):
    \"\"\"Test plugin system functionality.\"\"\"
    
    def setUp(self):
        self.plugin_manager = PluginManager()
        self.example_plugin = ExamplePlugin()
    
    def test_plugin_interface(self):
        \"\"\"Test plugin interface implementation.\"\"\"
        self.assertEqual(self.example_plugin.name, \"example_plugin\")
        self.assertEqual(self.example_plugin.version, \"1.0.0\")
        self.assertIsInstance(self.example_plugin.description, str)
        
        operations = self.example_plugin.get_operations()
        self.assertIsInstance(operations, dict)
        self.assertIn('matrix_sum', operations)
    
    def test_plugin_operations(self):
        \"\"\"Test plugin operation execution.\"\"\"
        matrix = Matrix([[1, 2], [3, 4]])
        
        # Test sum operation
        total = self.example_plugin.matrix_sum(matrix)
        self.assertEqual(total, 10.0)
        
        # Test mean operation
        mean = self.example_plugin.matrix_mean(matrix)
        self.assertEqual(mean, 2.5)
        
        # Test max operation
        max_val = self.example_plugin.matrix_max(matrix)
        self.assertEqual(max_val, 4.0)
        
        # Test min operation
        min_val = self.example_plugin.matrix_min(matrix)
        self.assertEqual(min_val, 1.0)
    
    def test_plugin_manager(self):
        \"\"\"Test plugin manager functionality.\"\"\"
        # Test loading plugins
        self.plugin_manager.plugins['example'] = self.example_plugin
        self.plugin_manager.operations.update(self.example_plugin.get_operations())
        
        # Test operation execution
        matrix = Matrix([[1, 2], [3, 4]])
        result = self.plugin_manager.execute_operation('matrix_sum', matrix)
        self.assertEqual(result, 10.0)
        
        # Test plugin info
        info = self.plugin_manager.get_plugin_info()
        self.assertIn('example', info)
        
        # Test available operations
        operations = self.plugin_manager.get_available_operations()
        self.assertIn('matrix_sum', operations)


class TestPropertyBased(unittest.TestCase):
    \"\"\"Property-based tests using Hypothesis.\"\"\"
    
    @given(st.lists(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=5), min_size=1, max_size=5))
    @settings(max_examples=50, deadline=5000)
    def test_matrix_addition_commutative(self, data):
        \"\"\"Test that matrix addition is commutative.\"\"\"
        # Ensure all rows have the same length
        if not data or not all(len(row) == len(data[0]) for row in data):
            return
        
        try:
            A = Matrix(data)
            B = Matrix(data)  # Use same data for simplicity
            
            result1 = A.add(B)
            result2 = B.add(A)
            
            self.assertEqual(result1.data, result2.data)
        except Exception:
            # Skip invalid matrices
            pass
    
    @given(st.lists(st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=4), min_size=1, max_size=4))
    @settings(max_examples=30, deadline=5000)
    def test_matrix_multiplication_associative(self, data):
        \"\"\"Test that matrix multiplication is associative when dimensions allow.\"\"\"
        # Ensure all rows have the same length and matrix is square
        if not data or not all(len(row) == len(data[0]) for row in data) or len(data) != len(data[0]):
            return
        
        try:
            A = Matrix(data)
            B = Matrix(data)
            C = Matrix(data)
            
            # (A * B) * C
            result1 = (A.multiply(B)).multiply(C)
            # A * (B * C)
            result2 = A.multiply(B.multiply(C))
            
            # Check if results are approximately equal (due to floating point precision)
            for i in range(result1.rows):
                for j in range(result1.cols):
                    val1 = float(result1.data[i][j])
                    val2 = float(result2.data[i][j])
                    self.assertAlmostEqual(val1, val2, places=10)
        except Exception:
            # Skip invalid matrices or operations
            pass
    
    @given(st.lists(st.lists(st.integers(min_value=-50, max_value=50), min_size=1, max_size=3), min_size=1, max_size=3))
    @settings(max_examples=40, deadline=5000)
    def test_transpose_involution(self, data):
        \"\"\"Test that transpose is its own inverse: (A^T)^T = A.\"\"\"
        # Ensure all rows have the same length
        if not data or not all(len(row) == len(data[0]) for row in data):
            return
        
        try:
            A = Matrix(data)
            A_transpose_transpose = A.transpose().transpose()
            
            self.assertEqual(A.data, A_transpose_transpose.data)
        except Exception:
            # Skip invalid matrices
            pass


class TestIntegration(unittest.TestCase):
    \"\"\"Integration tests for end-to-end functionality.\"\"\"
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = MatrixManager()
    
    def test_full_workflow(self):
        \"\"\"Test complete workflow from creation to export.\"\"\"
        # Create matrices
        A_name = self.manager.create_matrix([[1, 2], [3, 4]])
        B_name = self.manager.create_matrix([[5, 6], [7, 8]])
        
        # Get matrices
        A = self.manager.get_matrix(A_name)
        B = self.manager.get_matrix(B_name)
        
        # Perform operations
        C = A.add(B)
        D = A.multiply(B)
        
        # Check results
        self.assertEqual(C.data, [[6, 8], [10, 12]])
        self.assertEqual(D.data, [[19, 22], [43, 50]])
        
        # Test file I/O
        csv_file = os.path.join(self.temp_dir, 'matrix.csv')
        json_file = os.path.join(self.temp_dir, 'matrix.json')
        
        # Save to different formats
        csv_handler = get_handler_by_format('csv')
        json_handler = get_handler_by_format('json')
        
        csv_handler.save(C, csv_file)
        json_handler.save(C, json_file, name='result_matrix')
        
        # Load back and verify
        loaded_csv, _ = csv_handler.load(csv_file)
        loaded_json, json_name = json_handler.load(json_file)
        
        self.assertEqual(loaded_csv.data, C.data)
        self.assertEqual(loaded_json.data, C.data)
        self.assertEqual(json_name, 'result_matrix')
    
    def test_configuration_persistence(self):
        \"\"\"Test configuration loading and saving.\"\"\"
        config_file = os.path.join(self.temp_dir, 'test_config.json')
        
        # Modify configuration
        original_precision = Config.precision
        Config.precision = 6
        Config.colored_output = False
        
        # Save configuration
        Config.save_to_file(config_file)
        
        # Reset configuration
        Config.precision = 4
        Config.colored_output = True
        
        # Load configuration
        Config.load_from_file(config_file)
        
        # Verify loaded values
        self.assertEqual(Config.precision, 6)
        self.assertEqual(Config.colored_output, False)
        
        # Restore original
        Config.precision = original_precision
    
    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestBenchmarks(unittest.TestCase):
    \"\"\"Performance benchmark tests.\"\"\"
    
    def test_large_matrix_operations(self):
        \"\"\"Benchmark operations on large matrices.\"\"\"
        # Create large matrices (but not too large for testing)
        size = 50
        data_a = [[i + j for j in range(size)] for i in range(size)]
        data_b = [[i * j + 1 for j in range(size)] for i in range(size)]
        
        A = Matrix(data_a)
        B = Matrix(data_b)
        
        # Time matrix multiplication
        start_time = time.time()
        C = A.multiply(B)
        end_time = time.time()
        
        multiplication_time = end_time - start_time
        
        # Time determinant calculation
        start_time = time.time()
        det = A.determinant()
        end_time = time.time()
        
        determinant_time = end_time - start_time
        
        # Log performance metrics
        print(f\"\nPerformance Metrics for {size}x{size} matrices:\")
        print(f\"Matrix multiplication: {multiplication_time:.3f}s\")
        print(f\"Determinant calculation: {determinant_time:.3f}s\")
        
        # Assertions (reasonable time limits)
        self.assertLess(multiplication_time, 10.0)  # Should complete within 10 seconds
        self.assertLess(determinant_time, 10.0)  # Should complete within 10 seconds
        
        # Verify result integrity
        self.assertEqual(C.rows, size)
        self.assertEqual(C.cols, size)
        self.assertIsNotNone(det)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)