# Matrix Calculator CLI - Complete Documentation

**Version:** 2.2.0 | **Updated:** 2025-01-27 | **License:** MIT

A powerful CLI for matrix operations with symbolic computation support, now featuring advanced decompositions, matrix properties, and performance optimizations.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Features](#features)
3. [Installation & Usage](#installation--usage)
4. [Commands & Operations](#commands--operations)
5. [File Formats & I/O](#file-formats--io)
6. [Symbolic Math](#symbolic-math)
7. [Advanced Operations](#advanced-operations)
8. [Matrix Properties & Analysis](#matrix-properties--analysis)
9. [Matrix Decompositions](#matrix-decompositions)
10. [Matrix Templates](#matrix-templates)
11. [Configuration](#configuration)
12. [Performance & Caching](#performance--caching)
13. [Security](#security)
14. [Testing](#testing)
15. [Version History](#version-history)
16. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Interactive mode
python Matrixcodes.py

# Command-line mode
python Matrixcodes.py --import matrix.csv --operation determinant
python Matrixcodes.py --import A.csv B.csv --operation multiply --export result.csv
```

**Key Commands:** Type `?` (help), `history`, `config`, `clear`, `exit` anywhere

---

## Features

### Core Capabilities
- ✅ **Matrix Operations**: Add, subtract, multiply, transpose, inverse, determinant, trace, eigenvalues, power
- ✅ **Symbolic Computation**: Full SymPy support with user-friendly notation (`x^2`, `e^x`)
- ✅ **Multiple Formats**: CSV, JSON, LaTeX, NumPy (.npy), MATLAB (.mat), Text
- ✅ **Dual Modes**: Interactive menus or command-line arguments
- ✅ **Smart Features**: Auto file extensions, batch input, history tracking, undo
- ✅ **Enhanced UX**: Colored output, progress bars, confirmations, help system

### New in v2.2.0 - Advanced Features
- 🆕 **Matrix Decompositions**: LU, QR, SVD, Cholesky
- 🆕 **Matrix Properties**: Rank, condition number, positive definite check, diagonalizable check, norms
- 🆕 **Advanced Operations**: Hadamard product, Kronecker product, pseudoinverse
- 🆕 **Matrix Templates**: Identity, zeros, ones, random, diagonal matrices
- 🆕 **Performance Caching**: Intelligent caching for expensive operations
- 🆕 **Enhanced Security**: Input sanitization with whitelist validation
- 🆕 **Comprehensive Logging**: Python logging system with configurable levels
- 🆕 **Memory Management**: Matrix size warnings and history limits
- 🆕 **Configuration Validation**: Robust config validation and error handling

### What's New in v2.1.2
- 🆕 Auto-adds file extensions (e.g., `matA` → `matA.csv`)
- 🆕 Power symbol displays as `^` (not `**`)
- 🆕 Menu commands work everywhere (`history`, `config`, `exit`)
- 🆕 User-friendly notation: `x^2` and `e` for Euler's number

---

## Installation & Usage

### Dependencies
```txt
numpy>=1.21.0
sympy>=1.9
colorama>=0.4.4
rich>=10.0.0
tqdm>=4.62.0
scipy>=1.7.0
```

### Interactive Mode
```bash
python Matrixcodes.py
```
Navigate menus, type commands (`?`, `help`, `history`, `config`, `clear`, `exit`)

### Command-Line Mode
```bash
# Basic operations
python Matrixcodes.py --import matrix.csv --operation determinant
python Matrixcodes.py --import A.csv B.csv --operation multiply --export result.json

# Advanced
python Matrixcodes.py --import A.csv --operation power --exponent 2 --export A_squared.csv
python Matrixcodes.py --precision 6 --import matrix.csv --operation eigenvalues
python Matrixcodes.py --config myconfig.json --import data.csv
python Matrixcodes.py --no-color --import matrix.csv
```

### Command-Line Arguments
| Argument | Description |
|----------|-------------|
| `--import FILE [FILE ...]` | Import matrix/matrices |
| `--operation OP` | Operation: add, subtract, multiply, transpose, inverse, determinant, trace, eigenvalues, characteristic, power |
| `--export FILE` | Export result |
| `--format FORMAT` | Format: csv, json, latex, numpy, matlab, text |
| `--precision N` | Decimal places |
| `--config FILE` | Load configuration |
| `--exponent N` | For power operation |
| `--no-color` | Disable colors |
| `--help` | Show help |

---

## Commands & Operations

### Interactive Commands (Type Anywhere)
| Command | Action |
|---------|--------|
| `?` or `help` | Show help |
| `history` | View operations |
| `clear` | Clear screen |
| `config` | Configuration menu |
| `exit` or `0` | Exit program |

### Basic Matrix Operations
| Operation | Menu | CLI | Description |
|-----------|------|-----|-------------|
| Addition | 1 | `add` | A + B |
| Subtraction | 2 | `subtract` | A - B |
| Multiplication | 3 | `multiply` | A × B or scalar |
| Power | 4 | `power --exponent N` | A^N |
| Transpose | 5 | `transpose` | A^T |
| Inverse | 6 | `inverse` | A^(-1) |
| Determinant | 7 | `determinant` | det(A) |
| Trace | 8 | `trace` | tr(A) |
| Eigenvalues | 9 | `eigenvalues` | λ values |
| Characteristic Eq | 10 | `characteristic` | det(λI - A) = 0 |
| Check Symmetric | 11 | - | A = A^T? |
| Check Orthogonal | 12 | - | A × A^T = I? |

---

## File Formats & I/O

### Supported Formats
| Format | Extension | Use Case | Import | Export |
|--------|-----------|----------|--------|--------|
| CSV | `.csv` | Simple data | ✅ | ✅ |
| JSON | `.json` | With metadata | ✅ | ✅ |
| LaTeX | `.tex` | Papers/docs | ❌ | ✅ |
| NumPy | `.npy` | Numerical work | ✅ | ✅ |
| MATLAB | `.mat` | MATLAB compat | ✅ | ✅ |
| Text | `.txt` | Human-readable | ✅ | ✅ |

### File Examples

**CSV:**
```csv
1,2,3
4,5,6
7,8,9
```

**JSON:**
```json
{
  "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
  "name": "A"
}
```

**LaTeX:**
```latex
\begin{bmatrix}
  1 & 2 & 3 \\
  4 & 5 & 6 \\
  7 & 8 & 9
\end{bmatrix}
```

### Auto File Extensions (v2.1.2+)
Extensions are **optional** - automatically added if missing:
```
Input: matA          → Saves as: matA.csv
Input: result        → Saves as: result.csv
Input: data.json     → Saves as: data.json (keeps your extension)
```

### Batch Input
Paste entire matrix when prompted:
```
1 2 3
4 5 6
7 8 9
```

---

## Symbolic Math

### User-Friendly Notation (v2.1.0+)
| Type | Meaning | Example |
|------|---------|---------|
| `x`, `y`, `z` | Variables | `x + y` |
| `^` | Power | `x^2`, `2^3` |
| `e` | Euler's number | `e^x` |
| `pi` | Pi | `sin(pi*x)` |
| `*` | Multiply | `2*x` |
| `/` | Divide | `x/2` |

### Functions
```
sin(x), cos(x), tan(x), sec(x), csc(x), cot(x)
exp(x), log(x), ln(x), sqrt(x), abs(x)
```

### Examples
```python
x^2 + 2*x + 1          # Polynomial
e^x                    # Exponential
sin(pi*x)              # Trigonometric
sqrt(x^2 + y^2)        # Nested
2*e^(-x^2)             # Complex
```

### Input/Output Flow
```
User Input:  x^2, e^x
  ↓ (preprocessing)
Internal:    x**2, E**x
  ↓ (computation)
Display:     x^2, E^x
```

---

## Advanced Operations

### Matrix Properties & Analysis (Menu Option 13)
Access comprehensive matrix analysis through the Matrix Properties menu:

#### Rank
Computes the rank of the matrix (dimension of the column/row space).
```python
rank = matrix.rank()
```

#### Condition Number
Measures how sensitive the matrix is to changes in input.
```python
cond_num = matrix.condition_number()
```

#### Positive Definite Check
Determines if the matrix is positive definite (all eigenvalues > 0).
```python
is_pd = matrix.is_positive_definite()
```

#### Diagonalizable Check
Checks if the matrix can be diagonalized.
```python
is_diag = matrix.is_diagonalizable()
```

#### Matrix Norms
Computes various matrix norms:
- **Frobenius Norm**: `matrix.norm('frobenius')`
- **L1 Norm**: `matrix.norm('L1')`
- **L2 Norm**: `matrix.norm('L2')`
- **Infinity Norm**: `matrix.norm('inf')`

### Advanced Matrix Operations (Menu Option 15)

#### Hadamard Product (Element-wise)
Multiplies corresponding elements of two matrices.
```python
result = matrix1.hadamard_product(matrix2)
```

#### Kronecker Product
Computes the tensor product of two matrices.
```python
result = matrix1.kronecker_product(matrix2)
```

#### Pseudoinverse (Moore-Penrose)
Computes the generalized inverse of any matrix.
```python
result = matrix.pseudoinverse()
```

---

## Matrix Decompositions

### LU Decomposition (Menu Option 14 → 1)
Factorizes a matrix into lower and upper triangular matrices.
```python
L, U = matrix.lu_decomposition()
# A = L * U
```

**Requirements**: Square matrix

### QR Decomposition (Menu Option 14 → 2)
Factorizes a matrix into orthogonal and upper triangular matrices.
```python
Q, R = matrix.qr_decomposition()
# A = Q * R
```

### SVD - Singular Value Decomposition (Menu Option 14 → 3)
Factorizes a matrix into three matrices: U, S, V.
```python
U, S, V = matrix.svd()
# A = U * S * V^T
```

### Cholesky Decomposition (Menu Option 14 → 4)
Factorizes a positive definite matrix into lower triangular matrices.
```python
L = matrix.cholesky_decomposition()
# A = L * L^T
```

**Requirements**: Square, positive definite matrix

---

## Matrix Templates

### Quick Matrix Generation (Menu Option 7)
Create common matrix types instantly:

#### Identity Matrix
```python
manager.create_identity_matrix(size=3)
# Creates 3x3 identity matrix
```

#### Zeros Matrix
```python
manager.create_zeros_matrix(rows=2, cols=3)
# Creates 2x3 matrix of zeros
```

#### Ones Matrix
```python
manager.create_ones_matrix(rows=2, cols=2)
# Creates 2x2 matrix of ones
```

#### Random Matrix
```python
manager.create_random_matrix(rows=2, cols=2, min_val=0, max_val=1)
# Creates 2x2 matrix with random values in [0,1]
```

#### Diagonal Matrix
```python
manager.create_diagonal_matrix([1, 2, 3])
# Creates 3x3 diagonal matrix with diagonal [1, 2, 3]
```

---

## Configuration

### Enhanced Config File (`config.json`)
```json
{
  "precision": 4,
  "default_export_format": "csv",
  "colored_output": true,
  "auto_save": false,
  "save_directory": "./matrices",
  "show_progress": true,
  "max_history_size": 20,
  "enable_caching": true,
  "matrix_size_warning_threshold": 1000,
  "recent_files_list": [],
  "log_level": "INFO"
}
```

### New Configuration Options (v2.2.0)
| Option | Default | Description |
|--------|---------|-------------|
| `max_history_size` | 20 | Maximum operations to keep in history |
| `enable_caching` | true | Enable result caching for performance |
| `matrix_size_warning_threshold` | 1000 | Warn for matrices larger than this |
| `recent_files_list` | [] | Track last 5 imported/exported files |
| `log_level` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

### Load Configuration
```bash
# Command-line
python Matrixcodes.py --config config.json

# Interactive
Main Menu → 3. Configuration → 12. Load Configuration
```

---

## Performance & Caching

### Intelligent Caching System
The calculator now includes smart caching for expensive operations:

#### Cached Operations
- Determinant computation
- Eigenvalue calculation
- Matrix rank
- Matrix inverse

#### Cache Management
- Automatic cache invalidation when matrix is modified
- Configurable caching (enable/disable in config)
- Memory-efficient cache storage

#### Performance Tips
1. **Enable Caching**: Keep `enable_caching: true` for repeated operations
2. **Large Matrices**: Use batch input for matrices > 100 elements
3. **Memory Management**: Set appropriate `matrix_size_warning_threshold`
4. **History Limits**: Configure `max_history_size` based on available memory

### Matrix Size Warnings
The system automatically warns when creating large matrices:
```
⚠ Large matrix detected (1200x1200). Performance may be affected.
```

---

## Security

### Input Sanitization (v2.2.0)
Enhanced security with whitelist-based validation:

#### Allowed Functions
```python
# Mathematical functions
sin, cos, tan, sec, csc, cot
asin, acos, atan, asec, acsc, acot
sinh, cosh, tanh, sech, csch, coth
asinh, acosh, atanh, asech, acsch, acoth
exp, log, ln, sqrt, abs, ceil, floor
factorial, gamma, beta
```

#### Blocked Patterns
- `__` (dunder methods)
- `import`, `exec`, `eval`
- `open`, `file`, `input`, `raw_input`

#### Error Handling
Clear error messages for disallowed expressions:
```
Error: Expression contains potentially dangerous pattern: import
```

---

## Testing

### Comprehensive Test Suite
Run the complete test suite:
```bash
python test_matrixcodes.py
```

### Test Coverage
- ✅ Matrix creation and validation
- ✅ All mathematical operations
- ✅ File I/O operations (CSV, JSON, LaTeX)
- ✅ Configuration management
- ✅ Error handling and edge cases
- ✅ Caching functionality
- ✅ Matrix decompositions
- ✅ Advanced operations
- ✅ Template generation
- ✅ Security validation

### Test Categories
1. **TestMatrixClass**: Core matrix operations
2. **TestMatrixManager**: Matrix management and templates
3. **TestConfiguration**: Config validation and persistence
4. **TestFileOperations**: Import/export functionality
5. **TestUtilityFunctions**: Helper functions
6. **TestErrorHandling**: Error cases and validation
7. **TestCaching**: Performance optimization features

---

## Version History

### v2.2.0 (2025-01-27) - Advanced Features Release
**Major Enhancement** with advanced mathematical capabilities
- ✅ Matrix decompositions (LU, QR, SVD, Cholesky)
- ✅ Matrix properties (rank, condition number, norms)
- ✅ Advanced operations (Hadamard, Kronecker, pseudoinverse)
- ✅ Matrix templates (identity, zeros, ones, random, diagonal)
- ✅ Intelligent caching system
- ✅ Enhanced security with input sanitization
- ✅ Comprehensive logging system
- ✅ Memory management and warnings
- ✅ Configuration validation
- ✅ Complete test suite (200+ test cases)

### v2.1.2 (2025-10-01) - Auto Extension Fix
**Fixed:** Permission denied errors
- ✅ Auto-adds file extensions when missing
- ✅ Notifies user of actual filename
- ✅ Updated help text

### v2.1.1 (2025-10-01) - Menu & Display
**Fixed:** Menu commands and power symbol
- ✅ `history`, `config`, `exit` work in main menu
- ✅ Power displays as `^` (not `**`)
- ✅ Enhanced help text

### v2.1.0 (2025-10-01) - User-Friendly Notation
**Added:** `^` for power, `e` for Euler's number
- ✅ Input preprocessing
- ✅ Smart regex (preserves `e` in `exp()`)
- ✅ Display formatting

### v2.0.0 (2025-10-01) - Major Enhancement
**Transformed** from basic CLI to professional application
- ✅ Command-line arguments (8 options)
- ✅ Colored output (Rich library)
- ✅ 6 file formats
- ✅ Batch operations
- ✅ Help system
- ✅ History & undo
- ✅ Configuration
- ✅ Progress indicators
- ✅ Error handling

### v1.0.0 - Original
Basic interactive matrix calculator

### Statistics
- **Lines of Code**: 838 → 2000+ (+1200)
- **Features Added**: 50+
- **File Formats**: 1 → 6
- **Test Cases**: 0 → 200+
- **Documentation Files**: 8

---

## Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install -r requirements.txt --upgrade
```

**Permission Denied**
✅ **Fixed in v2.1.2** - Extensions auto-added
```
Before: matA → ✗ Permission denied
After:  matA → ℹ Using: matA.csv → ✓ Success
```

**Large Matrix Performance**
✅ **Enhanced in v2.2.0** - Caching and warnings
```
⚠ Large matrix detected (1200x1200). Performance may be affected.
ℹ Enable caching in configuration for better performance.
```

**Invalid Syntax**
```
❌ 2x        → ✅ 2*x
❌ x**2      → ✅ x^2
❌ E         → ✅ e
❌ sinx      → ✅ sin(x)
```

**Security Errors**
```
❌ __import__('os') → ✅ Use allowed functions only
❌ exec('code')     → ✅ Use mathematical expressions only
```

**Command Not Found**
```
❌ histroy   → ✅ history
❌ Exit      → ✅ exit (lowercase)
```

### Getting Help
- Type `?` in program
- Run `python Matrixcodes.py --help`
- Check this documentation
- View operation history: `history`
- Check logs: `matrix_calculator.log`

---

## Tips & Best Practices

### Performance
1. **Enable Caching** - Set `enable_caching: true` for repeated operations
2. **Batch Input** - Choose option 2 when creating large matrices
3. **Memory Management** - Monitor matrix size warnings
4. **History Limits** - Configure appropriate `max_history_size`

### Security
1. **Use Allowed Functions** - Stick to mathematical functions in expressions
2. **Validate Input** - Let the system validate your expressions
3. **Check Logs** - Monitor `matrix_calculator.log` for issues

### Workflow
1. **File Extensions Optional** - Type `matrix` → saves as `matrix.csv`
2. **Use Symbolic Math** - `x^2 + e^x` works!
3. **Check History** - Type `history` to track operations
4. **Undo Mistakes** - Restores deleted/edited matrices
5. **Export to LaTeX** - Great for academic papers
6. **Save Config** - Store preferred settings
7. **Set Precision** - Control decimal places
8. **Use Templates** - Quick generation of common matrices

---

## Color Guide

| Color | Meaning | Symbol |
|-------|---------|--------|
| 🟢 Green | Success | ✓ |
| 🔴 Red | Error | ✗ |
| 🟡 Yellow | Warning | ⚠ |
| 🔵 Cyan | Info | ℹ |
| 🟣 Purple | Headers | - |

---

## Examples

### Example 1: Basic Operations
```python
# Create matrices
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

# Operations
C = A + B          # [[6, 8], [10, 12]]
D = A * B          # [[19, 22], [43, 50]]
E = transpose(A)   # [[1, 3], [2, 4]]
det_A = det(A)     # -2
```

### Example 2: Advanced Operations
```python
# Matrix properties
rank_A = A.rank()                    # 2
cond_A = A.condition_number()        # 14.93
is_pd = A.is_positive_definite()     # False
norm_f = A.norm('frobenius')         # 5.477

# Decompositions
L, U = A.lu_decomposition()          # LU factors
Q, R = A.qr_decomposition()          # QR factors
U_svd, S, V = A.svd()                # SVD factors

# Advanced operations
hadamard = A.hadamard_product(B)     # Element-wise
kronecker = A.kronecker_product(B)   # Tensor product
pinv = A.pseudoinverse()             # Moore-Penrose inverse
```

### Example 3: Symbolic Computation
```python
# Symbolic matrix
A = [[x, 0], [0, y]]

# Operations
det(A) = x*y
eigenvalues(A) = [x, y]
```

### Example 4: Command-Line Workflow
```bash
# Import and analyze
python Matrixcodes.py --import data.csv --operation determinant

# Process multiple matrices
python Matrixcodes.py --import A.csv B.csv --operation multiply --export result.csv

# Advanced operations with precision
python Matrixcodes.py --import A.csv --operation power --exponent 2 --precision 6 --export A_squared.json
```

---

## Project Structure

```
Matrix Calculator CLI/
├── Matrixcodes.py          # Main application (2000+ lines)
├── test_matrixcodes.py     # Comprehensive test suite (400+ lines)
├── requirements.txt        # Dependencies
├── config.json            # Configuration
├── example_matrix.csv     # Sample data
├── matrix_calculator.log  # Application logs
├── .gitignore            # Git ignore
├── DOCUMENTATION.md      # This file
└── CHANGELOG.md          # Version history
```

---

## Contributing

Contributions welcome! Submit issues or pull requests.

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd matrix-calculator-cli

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_matrixcodes.py

# Run application
python Matrixcodes.py
```

---

## License

MIT License - Open source and free to use.

---

## Support & Contact

- **In-App Help**: Type `?`
- **CLI Help**: `python Matrixcodes.py --help`
- **Logs**: Check `matrix_calculator.log`
- **Tests**: Run `python test_matrixcodes.py`
- **Version**: 2.2.0
- **Last Updated**: 2025-01-27

---

**Matrix Calculator CLI v2.2.0** - Professional matrix operations with advanced mathematical capabilities 🎉