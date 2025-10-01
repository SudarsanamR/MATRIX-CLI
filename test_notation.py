"""
Test script to verify ^ and e notation preprocessing
"""
import sympy as sp
import re

def preprocess_expression(expr: str) -> str:
    """
    Preprocess mathematical expressions for user convenience.
    Converts ^ to ** for power and handles e for Euler's number.
    """
    # Replace ^ with ** for power
    expr = expr.replace('^', '**')
    
    # Replace standalone 'e' with 'E' (Euler's number)
    expr = re.sub(r'\be\b', 'E', expr)
    
    return expr

# Test cases
test_cases = [
    "x^2",
    "e^x",
    "2*e^(-x^2)",
    "x^2 + 2*x + 1",
    "sin(pi*x)",
    "sqrt(x^2 + y^2)",
    "e",
    "exp(x)",  # Should not change 'e' in 'exp'
    "2^3",
    "e^(x^2)",
]

print("Testing expression preprocessing:\n")
print(f"{'Input':<25} {'Preprocessed':<30} {'SymPy Result':<30}")
print("=" * 85)

for test in test_cases:
    preprocessed = preprocess_expression(test)
    try:
        result = sp.sympify(preprocessed)
        print(f"{test:<25} {preprocessed:<30} {str(result):<30}")
    except Exception as e:
        print(f"{test:<25} {preprocessed:<30} ERROR: {str(e):<30}")

print("\n✓ All tests completed!")
print("\nNote: 'e' is converted to 'E' (Euler's number)")
print("      '^' is converted to '**' (power operator)")
