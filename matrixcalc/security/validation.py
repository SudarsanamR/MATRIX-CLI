"""Security validation and expression sanitization."""

import re
import time
from typing import Set, List, Dict, Any
import sympy as sp

from ..logging.setup import get_logger

logger = get_logger(__name__)


class SecurityError(Exception):
    """Security validation error."""
    pass


class ExpressionValidator:
    """Validates mathematical expressions for security."""
    
    def __init__(self, timeout_seconds: int = 5, security_level: str = 'moderate'):
        """
        Initialize expression validator.
        
        Args:
            timeout_seconds: Maximum time allowed for expression evaluation
            security_level: Security level ('strict', 'moderate', 'permissive')
        """
        self.timeout_seconds = timeout_seconds
        self.security_level = security_level
        
        # Whitelist of allowed functions
        self.allowed_functions: Set[str] = {
            'sin', 'cos', 'tan', 'sec', 'csc', 'cot',
            'asin', 'acos', 'atan', 'asec', 'acsc', 'acot',
            'sinh', 'cosh', 'tanh', 'sech', 'csch', 'coth',
            'asinh', 'acosh', 'atanh', 'asech', 'acsch', 'acoth',
            'exp', 'log', 'ln', 'sqrt', 'abs', 'ceil', 'floor',
            'factorial', 'gamma', 'beta', 'erf', 'erfc',
            'binomial', 'gcd', 'lcm', 'prime', 'isprime'
        }
        
        # Adjust allowed functions based on security level
        if security_level == 'permissive':
            self.allowed_functions.update({
                'summation', 'product', 'limit', 'diff', 'integrate',
                'solve', 'expand', 'factor', 'simplify'
            })
        
        # Dangerous patterns to block
        self.dangerous_patterns: List[str] = [
            r'__',  # Double underscore (potential dunder methods)
            r'import',  # Import statements
            r'exec',  # Exec statements
            r'eval',  # Eval statements
            r'open',  # File operations
            r'file',  # File operations
            r'input',  # Input operations
            r'raw_input',  # Raw input operations
            r'globals',  # Global variable access
            r'locals',  # Local variable access
            r'vars',  # Variable access
            r'dir',  # Directory listing
            r'getattr',  # Attribute access
            r'setattr',  # Attribute setting
            r'delattr',  # Attribute deletion
            r'hasattr',  # Attribute checking
            r'compile',  # Code compilation
            r'__import__',  # Import function
        ]
        
        # Add stricter patterns for strict mode
        if security_level == 'strict':
            self.dangerous_patterns.extend([
                r'while',  # While loops
                r'for',  # For loops
                r'lambda',  # Lambda functions
                r'\*\*',  # Power operator (limit to ^)
            ])
        
        # Rate limiting
        self._validation_count = 0
        self._last_reset_time = time.time()
        self._max_validations_per_minute = 100
    
    def validate_expression(self, expr: str) -> str:
        """
        Validate and preprocess mathematical expression.
        
        Args:
            expr: The expression string to validate
            
        Returns:
            Preprocessed and validated expression string
            
        Raises:
            SecurityError: If expression contains dangerous patterns
        """
        # Rate limiting check
        self._check_rate_limit()
        
        logger.debug(f"Validating expression: {expr}")
        
        # Basic length check
        if len(expr) > 10000:  # Prevent extremely long expressions
            raise SecurityError("Expression too long")
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, expr, re.IGNORECASE):
                logger.warning(f"Blocked dangerous pattern '{pattern}' in expression: {expr}")
                raise SecurityError(f"Expression contains potentially dangerous pattern: {pattern}")
        
        # Preprocess expression
        preprocessed = self._preprocess_expression(expr)
        
        # Validate with SymPy (with timeout)
        self._validate_with_sympy(preprocessed)
        
        logger.debug(f"Expression validated successfully: {preprocessed}")
        return preprocessed
    
    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Reset counter if a minute has passed
        if current_time - self._last_reset_time >= 60:
            self._validation_count = 0
            self._last_reset_time = current_time
        
        self._validation_count += 1
        
        if self._validation_count > self._max_validations_per_minute:
            logger.warning("Rate limit exceeded for expression validation")
            raise SecurityError("Rate limit exceeded. Please try again later.")
    
    def _preprocess_expression(self, expr: str) -> str:
        """Preprocess expression for user convenience."""
        # Replace ^ with ** for power
        expr = expr.replace('^', '**')
        
        # Replace standalone 'e' with 'E' (Euler's number)
        # But be careful not to replace 'e' in function names like 'exp'
        expr = re.sub(r'\be\b', 'E', expr)
        
        return expr
    
    def _validate_with_sympy(self, expr: str) -> None:
        """Validate expression with SymPy and timeout."""
        start_time = time.time()
        
        try:
            # Create a safe evaluation context
            safe_dict = {
                'x': sp.Symbol('x'),
                'y': sp.Symbol('y'),
                'z': sp.Symbol('z'),
                't': sp.Symbol('t'),
                'E': sp.E,
                'pi': sp.pi,
                'I': sp.I,
                'oo': sp.oo,
                'exp': sp.exp,
                'log': sp.log,
                'sin': sp.sin,
                'cos': sp.cos,
                'tan': sp.tan,
                'sqrt': sp.sqrt,
                'abs': sp.Abs,
            }
            
            # Add allowed functions to safe context
            for func_name in self.allowed_functions:
                if hasattr(sp, func_name):
                    safe_dict[func_name] = getattr(sp, func_name)
            
            # Parse and validate expression
            parsed_expr = sp.sympify(expr, locals=safe_dict)
            
            # Check evaluation time
            if time.time() - start_time > self.timeout_seconds:
                logger.warning(f"Expression evaluation timeout: {expr}")
                raise SecurityError("Expression evaluation timeout")
            
            # Additional validation - check for suspicious attributes
            if hasattr(parsed_expr, 'atoms'):
                atoms = parsed_expr.atoms()
                for atom in atoms:
                    if hasattr(atom, 'name') and atom.name.startswith('_'):
                        logger.warning(f"Blocked suspicious atom: {atom.name}")
                        raise SecurityError("Expression contains suspicious elements")
            
            logger.debug(f"Expression parsed successfully: {parsed_expr}")
            
        except Exception as e:
            if isinstance(e, SecurityError):
                raise
            logger.warning(f"Expression validation failed: {expr}, error: {str(e)}")
            raise SecurityError(f"Invalid expression: {str(e)}")
    
    def is_safe_function(self, func_name: str) -> bool:
        """Check if a function name is in the safe whitelist."""
        return func_name in self.allowed_functions
    
    def add_safe_function(self, func_name: str) -> None:
        """Add a function to the safe whitelist."""
        self.allowed_functions.add(func_name)
        logger.info(f"Added safe function: {func_name}")
    
    def remove_safe_function(self, func_name: str) -> None:
        """Remove a function from the safe whitelist."""
        self.allowed_functions.discard(func_name)
        logger.info(f"Removed safe function: {func_name}")
    
    def get_safe_functions(self) -> Set[str]:
        """Get the set of safe functions."""
        return self.allowed_functions.copy()
    
    def update_security_level(self, level: str) -> None:
        """Update security level and adjust validation rules."""
        if level not in ['strict', 'moderate', 'permissive']:
            raise ValueError("Invalid security level")
        
        self.security_level = level
        logger.info(f"Security level updated to: {level}")
        
        # Re-initialize with new security level
        self.__init__(self.timeout_seconds, level)
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'security_level': self.security_level,
            'timeout_seconds': self.timeout_seconds,
            'allowed_functions_count': len(self.allowed_functions),
            'dangerous_patterns_count': len(self.dangerous_patterns),
            'validation_count': self._validation_count,
            'rate_limit': self._max_validations_per_minute
        }


# Global validator instance
expression_validator = ExpressionValidator()
