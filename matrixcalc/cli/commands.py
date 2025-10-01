"""Command-line interface with subcommands."""

import argparse
import sys
from typing import List, Optional, Dict, Any

from ..core.matrix import Matrix
from ..core.manager import MatrixManager
from ..config.settings import Config
from ..logging.setup import setup_logging, get_logger
from ..io.formats import get_handler_by_format

logger = get_logger(__name__)


class MatrixCLI:
    """Command-line interface for matrix operations."""
    
    def __init__(self):
        self.manager = MatrixManager()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser with subcommands."""
        parser = argparse.ArgumentParser(
            prog='matrix',
            description='Advanced Matrix Calculator CLI',
            epilog='Use "matrix <command> --help" for command-specific help'
        )
        
        parser.add_argument('--version', action='version', version='%(prog)s 2.3.0')
        parser.add_argument('--config', type=str, help='Configuration file path')
        parser.add_argument('--no-color', action='store_true', help='Disable colored output')
        parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
        parser.add_argument('--log-format', choices=['plain', 'json'], help='Logging format')
        
        # Create subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Create matrix subcommand
        create_parser = subparsers.add_parser('create', help='Create matrices')
        create_parser.add_argument('--type', choices=['identity', 'zeros', 'ones', 'random', 'diagonal'], 
                                 required=True, help='Matrix type')
        create_parser.add_argument('--rows', type=int, help='Number of rows')
        create_parser.add_argument('--cols', type=int, help='Number of columns')
        create_parser.add_argument('--size', type=int, help='Size for square matrices')
        create_parser.add_argument('--diagonal', type=str, help='Diagonal elements (comma-separated)')
        create_parser.add_argument('--min', type=float, default=0, help='Minimum value for random matrix')
        create_parser.add_argument('--max', type=float, default=1, help='Maximum value for random matrix')
        create_parser.add_argument('--name', type=str, help='Matrix name')
        
        # Import matrix subcommand
        import_parser = subparsers.add_parser('import', help='Import matrices from files')
        import_parser.add_argument('files', nargs='+', help='Files to import')
        import_parser.add_argument('--format', choices=['csv', 'json', 'numpy', 'matlab'], 
                                 help='File format (auto-detected if not specified)')
        
        # Export matrix subcommand
        export_parser = subparsers.add_parser('export', help='Export matrices to files')
        export_parser.add_argument('matrix_name', help='Name of matrix to export')
        export_parser.add_argument('output_file', help='Output file path')
        export_parser.add_argument('--format', choices=['csv', 'json', 'latex', 'numpy', 'matlab', 'text'],
                                 default='csv', help='Export format')
        
        # Operations subcommands
        self._add_operation_commands(subparsers)
        
        # List matrices subcommand
        list_parser = subparsers.add_parser('list', help='List all matrices')
        
        # Show matrix subcommand
        show_parser = subparsers.add_parser('show', help='Show matrix contents')
        show_parser.add_argument('matrix_name', help='Name of matrix to show')
        show_parser.add_argument('--preview', action='store_true', help='Show preview for large matrices')
        
        # Delete matrix subcommand
        delete_parser = subparsers.add_parser('delete', help='Delete matrices')
        delete_parser.add_argument('matrix_names', nargs='+', help='Names of matrices to delete')
        
        # Configuration subcommand
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_parser.add_argument('--show', action='store_true', help='Show current configuration')
        config_parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), help='Set configuration value')
        config_parser.add_argument('--save', type=str, help='Save configuration to file')
        config_parser.add_argument('--load', type=str, help='Load configuration from file')
        config_parser.add_argument('--validate', action='store_true', help='Validate current configuration')
        
        return parser
    
    def _add_operation_commands(self, subparsers):
        """Add operation subcommands."""
        
        # Add operation
        add_parser = subparsers.add_parser('add', help='Add two matrices')
        add_parser.add_argument('matrix_a', help='First matrix name')
        add_parser.add_argument('matrix_b', help='Second matrix name')
        add_parser.add_argument('--result', type=str, help='Result matrix name')
        
        # Subtract operation
        sub_parser = subparsers.add_parser('subtract', help='Subtract two matrices')
        sub_parser.add_argument('matrix_a', help='First matrix name')
        sub_parser.add_argument('matrix_b', help='Second matrix name')
        sub_parser.add_argument('--result', type=str, help='Result matrix name')
        
        # Multiply operation
        mul_parser = subparsers.add_parser('multiply', help='Multiply two matrices')
        mul_parser.add_argument('matrix_a', help='First matrix name')
        mul_parser.add_argument('matrix_b', help='Second matrix name')
        mul_parser.add_argument('--result', type=str, help='Result matrix name')
        
        # Transpose operation
        transpose_parser = subparsers.add_parser('transpose', help='Transpose matrix')
        transpose_parser.add_argument('matrix_name', help='Matrix name')
        transpose_parser.add_argument('--result', type=str, help='Result matrix name')
        
        # Inverse operation
        inverse_parser = subparsers.add_parser('inverse', help='Compute matrix inverse')
        inverse_parser.add_argument('matrix_name', help='Matrix name')
        inverse_parser.add_argument('--result', type=str, help='Result matrix name')
        
        # Determinant operation
        det_parser = subparsers.add_parser('determinant', help='Compute matrix determinant')
        det_parser.add_argument('matrix_name', help='Matrix name')
        
        # Trace operation
        trace_parser = subparsers.add_parser('trace', help='Compute matrix trace')
        trace_parser.add_argument('matrix_name', help='Matrix name')
        
        # Eigenvalues operation
        eigen_parser = subparsers.add_parser('eigenvalues', help='Compute matrix eigenvalues')
        eigen_parser.add_argument('matrix_name', help='Matrix name')
        
        # Decompositions
        decomp_parser = subparsers.add_parser('decompose', help='Matrix decompositions')
        decomp_parser.add_argument('matrix_name', help='Matrix name')
        decomp_parser.add_argument('--type', choices=['lu', 'qr', 'svd', 'cholesky'], 
                                 required=True, help='Decomposition type')
        decomp_parser.add_argument('--result-prefix', type=str, help='Prefix for result matrices')
        
        # Properties
        props_parser = subparsers.add_parser('properties', help='Matrix properties')
        props_parser.add_argument('matrix_name', help='Matrix name')
        props_parser.add_argument('--property', choices=['rank', 'condition', 'norm', 'symmetric', 'orthogonal'], 
                                 help='Specific property to compute')
        # Performance subcommand
        perf_parser = subparsers.add_parser('performance', help='Performance monitoring and profiling')
        perf_parser.add_argument('--stats', action='store_true', help='Show performance statistics')
        perf_parser.add_argument('--memory', action='store_true', help='Show memory usage')
        perf_parser.add_argument('--clear-stats', action='store_true', help='Clear performance statistics')
        
        # Plugin subcommand
        plugin_parser = subparsers.add_parser('plugin', help='Plugin management')
        plugin_parser.add_argument('--list', action='store_true', help='List loaded plugins')
        plugin_parser.add_argument('--load', type=str, help='Load plugin from file')
        plugin_parser.add_argument('--unload', type=str, help='Unload plugin by name')
        plugin_parser.add_argument('--execute', nargs='+', help='Execute plugin operation: <operation> <matrix_name> [args]')
        
        # Security subcommand
        security_parser = subparsers.add_parser('security', help='Security management')
        security_parser.add_argument('--level', choices=['strict', 'moderate', 'permissive'], help='Set security level')
        security_parser.add_argument('--validate', type=str, help='Validate expression')
        security_parser.add_argument('--stats', action='store_true', help='Show security statistics')
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the CLI with given arguments.
        
        Args:
            args: Command line arguments (defaults to sys.argv[1:])
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Parse arguments
            parsed_args = self.parser.parse_args(args)
            
            # Load configuration if specified
            if parsed_args.config:
                Config.load_from_file(parsed_args.config)
            
            # Configure output
            if parsed_args.no_color:
                Config.colored_output = False
            
            # Configure logging
            if parsed_args.verbose:
                Config.log_level = 'DEBUG'
                setup_logging()
            if parsed_args.log_format:
                Config.log_format = parsed_args.log_format
                setup_logging()
            
            # Execute command
            if parsed_args.command is None:
                self.parser.print_help()
                return 1
            
            return self._execute_command(parsed_args)
            
        except Exception as e:
            logger.error(f"CLI error: {str(e)}")
            print(f"Error: {str(e)}")
            return 1
    
    def _execute_command(self, args: argparse.Namespace) -> int:
        """Execute the parsed command."""
        command = args.command
        
        try:
            if command == 'create':
                return self._cmd_create(args)
            elif command == 'import':
                return self._cmd_import(args)
            elif command == 'export':
                return self._cmd_export(args)
            elif command == 'list':
                return self._cmd_list(args)
            elif command == 'show':
                return self._cmd_show(args)
            elif command == 'delete':
                return self._cmd_delete(args)
            elif command == 'config':
                return self._cmd_config(args)
            elif command in ['add', 'subtract', 'multiply']:
                return self._cmd_binary_operation(args)
            elif command in ['transpose', 'inverse', 'determinant', 'trace', 'eigenvalues']:
                return self._cmd_unary_operation(args)
            elif command == 'decompose':
                return self._cmd_decompose(args)
            elif command == 'properties':
                return self._cmd_properties(args)
            elif command == 'performance':
                return self._cmd_performance(args)
            elif command == 'plugin':
                return self._cmd_plugin(args)
            elif command == 'security':
                return self._cmd_security(args)
            else:
                print(f"Unknown command: {command}")
                return 1
                
        except Exception as e:
            logger.error(f"Command execution error: {str(e)}")
            print(f"Error executing {command}: {str(e)}")
            return 1
    
    def _cmd_create(self, args: argparse.Namespace) -> int:
        """Handle create command."""
        matrix_type = args.type
        
        if matrix_type == 'identity':
            size = args.size or args.rows or 3
            self.manager.create_identity_matrix(size, args.name)
        elif matrix_type == 'zeros':
            rows = args.rows or 3
            cols = args.cols or rows
            self.manager.create_zeros_matrix(rows, cols, args.name)
        elif matrix_type == 'ones':
            rows = args.rows or 3
            cols = args.cols or rows
            self.manager.create_ones_matrix(rows, cols, args.name)
        elif matrix_type == 'random':
            rows = args.rows or 3
            cols = args.cols or rows
            self.manager.create_random_matrix(rows, cols, args.min, args.max, args.name)
        elif matrix_type == 'diagonal':
            if not args.diagonal:
                print("Error: --diagonal required for diagonal matrix")
                return 1
            diagonal_elements = [float(x.strip()) for x in args.diagonal.split(',')]
            self.manager.create_diagonal_matrix(diagonal_elements, args.name)
        
        print(f"✓ Created {matrix_type} matrix")
        return 0
    
    def _cmd_import(self, args: argparse.Namespace) -> int:
        """Handle import command."""
        for file_path in args.files:
            try:
                format_type = args.format
                if not format_type:
                    format_type = self._detect_format(file_path)
                
                handler = get_handler_by_format(format_type)
                matrix, name = handler.load(file_path)
                
                if args.name:
                    name = args.name
                
                self.manager.matrices[name] = matrix
                print(f"✓ Imported matrix '{name}' from {file_path}")
                
            except Exception as e:
                print(f"Error importing {file_path}: {str(e)}")
                return 1
        
        return 0
    
    def _cmd_export(self, args: argparse.Namespace) -> int:
        """Handle export command."""
        if args.matrix_name not in self.manager.matrices:
            print(f"Error: Matrix '{args.matrix_name}' not found")
            return 1
        
        matrix = self.manager.matrices[args.matrix_name]
        handler = get_handler_by_format(args.format)
        
        try:
            handler.save(matrix, args.output_file)
            print(f"✓ Exported matrix '{args.matrix_name}' to {args.output_file}")
            return 0
        except Exception as e:
            print(f"Error exporting matrix: {str(e)}")
            return 1
    
    def _cmd_list(self, args: argparse.Namespace) -> int:
        """Handle list command."""
        if not self.manager.matrices:
            print("No matrices loaded")
            return 0
        
        print("Loaded matrices:")
        for name, matrix in self.manager.matrices.items():
            print(f"  {name}: {matrix.rows}x{matrix.cols}")
        
        return 0
    
    def _cmd_show(self, args: argparse.Namespace) -> int:
        """Handle show command."""
        if args.matrix_name not in self.manager.matrices:
            print(f"Error: Matrix '{args.matrix_name}' not found")
            return 1
        
        matrix = self.manager.matrices[args.matrix_name]
        
        if args.preview:
            print(matrix.preview())
        else:
            print(matrix)
        
        return 0
    
    def _cmd_delete(self, args: argparse.Namespace) -> int:
        """Handle delete command."""
        for name in args.matrix_names:
            if name in self.manager.matrices:
                del self.manager.matrices[name]
                print(f"✓ Deleted matrix '{name}'")
            else:
                print(f"Warning: Matrix '{name}' not found")
        
        return 0
    
    def _cmd_config(self, args: argparse.Namespace) -> int:
        """Handle config command."""
        if args.show:
            settings = Config.get_all_settings()
            print("Current configuration:")
            for key, value in settings.items():
                print(f"  {key}: {value}")
        
        elif args.set:
            key, value = args.set
            if hasattr(Config, key):
                # Try to convert value to appropriate type
                current_value = getattr(Config, key)
                if isinstance(current_value, bool):
                    setattr(Config, key, value.lower() in ('true', '1', 'yes', 'on'))
                elif isinstance(current_value, int):
                    setattr(Config, key, int(value))
                elif isinstance(current_value, float):
                    setattr(Config, key, float(value))
                else:
                    setattr(Config, key, value)
                print(f"✓ Set {key} = {getattr(Config, key)}")
            else:
                print(f"Error: Unknown configuration key '{key}'")
                return 1
        
        elif args.save:
            Config.save_to_file(args.save)
        
        elif args.load:
            Config.load_from_file(args.load)

        elif args.validate:
            try:
                Config.validate()
                print("Configuration is valid.")
            except Exception as e:
                print(f"Configuration invalid: {e}")
        
        return 0
    
    def _cmd_binary_operation(self, args: argparse.Namespace) -> int:
        """Handle binary operations (add, subtract, multiply)."""
        if args.matrix_a not in self.manager.matrices:
            print(f"Error: Matrix '{args.matrix_a}' not found")
            return 1
        
        if args.matrix_b not in self.manager.matrices:
            print(f"Error: Matrix '{args.matrix_b}' not found")
            return 1
        
        matrix_a = self.manager.matrices[args.matrix_a]
        matrix_b = self.manager.matrices[args.matrix_b]
        
        try:
            if args.command == 'add':
                result = matrix_a.add(matrix_b)
            elif args.command == 'subtract':
                result = matrix_a.subtract(matrix_b)
            elif args.command == 'multiply':
                result = matrix_a.multiply(matrix_b)
            
            result_name = args.result or f"{args.matrix_a}_{args.command}_{args.matrix_b}"
            self.manager.matrices[result_name] = result
            
            print(f"✓ {args.command.title()} operation completed. Result saved as '{result_name}'")
            return 0
            
        except Exception as e:
            print(f"Error in {args.command} operation: {str(e)}")
            return 1
    
    def _cmd_unary_operation(self, args: argparse.Namespace) -> int:
        """Handle unary operations."""
        if args.matrix_name not in self.manager.matrices:
            print(f"Error: Matrix '{args.matrix_name}' not found")
            return 1
        
        matrix = self.manager.matrices[args.matrix_name]
        
        try:
            if args.command == 'transpose':
                result = matrix.transpose()
                result_name = args.result or f"{args.matrix_name}_transpose"
                self.manager.matrices[result_name] = result
                print(f"✓ Transpose completed. Result saved as '{result_name}'")
            
            elif args.command == 'inverse':
                result = matrix.inverse()
                result_name = args.result or f"{args.matrix_name}_inverse"
                self.manager.matrices[result_name] = result
                print(f"✓ Inverse completed. Result saved as '{result_name}'")
            
            elif args.command == 'determinant':
                result = matrix.determinant()
                print(f"Determinant: {result}")
            
            elif args.command == 'trace':
                result = matrix.trace()
                print(f"Trace: {result}")
            
            elif args.command == 'eigenvalues':
                result = matrix.eigenvalues()
                print(f"Eigenvalues: {result}")
            
            return 0
            
        except Exception as e:
            print(f"Error in {args.command} operation: {str(e)}")
            return 1
    
    def _cmd_decompose(self, args: argparse.Namespace) -> int:
        """Handle decomposition command."""
        if args.matrix_name not in self.manager.matrices:
            print(f"Error: Matrix '{args.matrix_name}' not found")
            return 1
        
        matrix = self.manager.matrices[args.matrix_name]
        prefix = args.result_prefix or f"{args.matrix_name}_{args.type}"
        
        try:
            if args.type == 'lu':
                L, U = matrix.lu_decomposition()
                self.manager.matrices[f"{prefix}_L"] = L
                self.manager.matrices[f"{prefix}_U"] = U
                print(f"✓ LU decomposition completed. Results saved as '{prefix}_L' and '{prefix}_U'")
            
            elif args.type == 'qr':
                Q, R = matrix.qr_decomposition()
                self.manager.matrices[f"{prefix}_Q"] = Q
                self.manager.matrices[f"{prefix}_R"] = R
                print(f"✓ QR decomposition completed. Results saved as '{prefix}_Q' and '{prefix}_R'")
            
            elif args.type == 'svd':
                U, S, V = matrix.svd()
                self.manager.matrices[f"{prefix}_U"] = U
                self.manager.matrices[f"{prefix}_S"] = S
                self.manager.matrices[f"{prefix}_V"] = V
                print(f"✓ SVD decomposition completed. Results saved as '{prefix}_U', '{prefix}_S', '{prefix}_V'")
            
            elif args.type == 'cholesky':
                L = matrix.cholesky_decomposition()
                self.manager.matrices[f"{prefix}_L"] = L
                print(f"✓ Cholesky decomposition completed. Result saved as '{prefix}_L'")
            
            return 0
            
        except Exception as e:
            print(f"Error in {args.type} decomposition: {str(e)}")
            return 1
    
    def _cmd_properties(self, args: argparse.Namespace) -> int:
        """Handle properties command."""
        if args.matrix_name not in self.manager.matrices:
            print(f"Error: Matrix '{args.matrix_name}' not found")
            return 1
        
        matrix = self.manager.matrices[args.matrix_name]
        
        try:
            if args.property:
                # Show specific property
                if args.property == 'rank':
                    result = matrix.rank()
                    print(f"Rank: {result}")
                elif args.property == 'condition':
                    result = matrix.condition_number()
                    print(f"Condition number: {result}")
                elif args.property == 'norm':
                    result = matrix.norm('frobenius')
                    print(f"Frobenius norm: {result}")
                elif args.property == 'symmetric':
                    result = matrix.is_symmetric()
                    print(f"Is symmetric: {result}")
                elif args.property == 'orthogonal':
                    result = matrix.is_orthogonal()
                    print(f"Is orthogonal: {result}")
            else:
                # Show all properties
                print(f"Matrix '{args.matrix_name}' properties:")
                print(f"  Size: {matrix.rows}x{matrix.cols}")
                print(f"  Rank: {matrix.rank()}")
                print(f"  Condition number: {matrix.condition_number()}")
                print(f"  Frobenius norm: {matrix.norm('frobenius')}")
                print(f"  Is symmetric: {matrix.is_symmetric()}")
                print(f"  Is orthogonal: {matrix.is_orthogonal()}")
                print(f"  Is square: {matrix.is_square()}")
                print(f"  Is positive definite: {matrix.is_positive_definite()}")
            
            return 0
            
        except Exception as e:
            print(f"Error computing properties: {str(e)}")
            return 1
    
    def _cmd_performance(self, args: argparse.Namespace) -> int:
        """Handle performance command."""
        try:
            from ..core.performance import performance_profiler, memory_monitor
            
            if args.stats:
                stats = performance_profiler.get_performance_stats()
                if stats:
                    print("Performance Statistics:")
                    print(f"  Total operations: {stats['total_operations']}")
                    print(f"  Total time: {stats['total_time']:.3f}s")
                    print(f"  Average time: {stats['average_time']:.3f}s")
                    print(f"  Min time: {stats['min_time']:.3f}s")
                    print(f"  Max time: {stats['max_time']:.3f}s")
                    print(f"  Memory delta: {stats['average_memory_delta']:.1f} MB avg")
                else:
                    print("No performance statistics available")
            
            elif args.memory:
                usage = memory_monitor.get_memory_usage_mb()
                limit = memory_monitor.limit_mb
                print(f"Memory Usage: {usage:.1f} MB / {limit} MB ({usage/limit*100:.1f}%)")
                
                if usage > limit * 0.8:
                    print("⚠ Warning: Memory usage is high")
            
            elif args.clear_stats:
                performance_profiler.clear_stats()
                print("✓ Performance statistics cleared")
            
            else:
                print("Use --stats, --memory, or --clear-stats")
            
            return 0
            
        except Exception as e:
            print(f"Error in performance command: {str(e)}")
            return 1
    
    def _cmd_plugin(self, args: argparse.Namespace) -> int:
        """Handle plugin command."""
        try:
            from ..plugins import plugin_manager
            
            if args.list:
                plugins = plugin_manager.get_plugin_info()
                if plugins:
                    print("Loaded plugins:")
                    for name, info in plugins.items():
                        print(f"  {name} v{info['version']}: {info['description']}")
                        print(f"    Operations: {', '.join(info['operations'])}")
                else:
                    print("No plugins loaded")
            
            elif args.load:
                success = plugin_manager.load_plugin(args.load)
                if success:
                    print(f"✓ Plugin loaded from {args.load}")
                else:
                    print(f"✗ Failed to load plugin from {args.load}")
                    return 1
            
            elif args.unload:
                success = plugin_manager.unload_plugin(args.unload)
                if success:
                    print(f"✓ Plugin '{args.unload}' unloaded")
                else:
                    print(f"✗ Failed to unload plugin '{args.unload}'")
                    return 1
            
            elif args.execute:
                if len(args.execute) < 2:
                    print("Usage: --execute <operation> <matrix_name> [args]")
                    return 1
                
                operation = args.execute[0]
                matrix_name = args.execute[1]
                
                if matrix_name not in self.manager.matrices:
                    print(f"Error: Matrix '{matrix_name}' not found")
                    return 1
                
                matrix = self.manager.matrices[matrix_name]
                result = plugin_manager.execute_operation(operation, matrix)
                print(f"Result: {result}")
            
            else:
                operations = plugin_manager.get_available_operations()
                print(f"Available plugin operations: {', '.join(operations) if operations else 'None'}")
            
            return 0
            
        except Exception as e:
            print(f"Error in plugin command: {str(e)}")
            return 1
    
    def _cmd_security(self, args: argparse.Namespace) -> int:
        """Handle security command."""
        try:
            from ..security.validation import expression_validator
            
            if args.level:
                expression_validator.update_security_level(args.level)
                print(f"✓ Security level set to '{args.level}'")
            
            elif args.validate:
                try:
                    result = expression_validator.validate_expression(args.validate)
                    print(f"✓ Expression is valid: {result}")
                except Exception as e:
                    print(f"✗ Expression validation failed: {str(e)}")
                    return 1
            
            elif args.stats:
                stats = expression_validator.get_validation_stats()
                print("Security Statistics:")
                print(f"  Security level: {stats['security_level']}")
                print(f"  Timeout: {stats['timeout_seconds']}s")
                print(f"  Allowed functions: {stats['allowed_functions_count']}")
                print(f"  Validation count: {stats['validation_count']}")
                print(f"  Rate limit: {stats['rate_limit']}/minute")
            
            else:
                print("Use --level, --validate, or --stats")
            
            return 0
            
        except Exception as e:
            print(f"Error in security command: {str(e)}")
            return 1
    
    def _detect_format(self, file_path: str) -> str:
        """Detect file format from extension."""
        ext = file_path.lower().split('.')[-1]
        format_map = {
            'csv': 'csv',
            'json': 'json',
            'npy': 'numpy',
            'mat': 'matlab'
        }
        return format_map.get(ext, 'csv')
