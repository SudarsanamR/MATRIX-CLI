#!/usr/bin/env python3
"""
Enhanced Matrix Calculator CLI v2.3.0
Comprehensive entry point supporting both interactive and command-line modes.

Usage:
    # Interactive mode (menu-driven)
    python matrix_main.py
    python matrix_main.py --interactive
    
    # Command-line mode (scripting)
    python matrix_main.py create --type identity --size 3
    python matrix_main.py import file.csv
    python matrix_main.py multiply A B --result C
    python matrix_main.py performance --stats
    python matrix_main.py plugin --list
"""

import sys
import os
import argparse
from typing import Optional, List

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_interactive_mode():
    """Run the original interactive mode."""
    print("🔢 Matrix Calculator CLI v2.3.0 - Interactive Mode")
    print("=" * 50)
    
    try:
        # Try to import and run the original interactive interface
        import Matrixcodes
        
        # Check if it has a main function
        if hasattr(Matrixcodes, 'main'):
            return Matrixcodes.main()
        else:
            # If no main function, try to run the module directly
            print("ℹ Starting interactive matrix calculator...")
            exec(open('Matrixcodes.py').read())
            return 0
            
    except ImportError as e:
        print(f"❌ Error importing Matrixcodes module: {str(e)}")
        print("💡 Trying alternative approach...")
        
        try:
            # Alternative: execute the file directly
            import subprocess
            import sys
            result = subprocess.run([sys.executable, 'Matrixcodes.py'], 
                                  cwd=os.path.dirname(os.path.abspath(__file__)))
            return result.returncode
        except Exception as e2:
            print(f"❌ Error running Matrixcodes.py: {str(e2)}")
            print("💡 Falling back to simplified interactive mode...")
            return run_simplified_interactive_mode()
            
    except FileNotFoundError:
        print("❌ Error: Matrixcodes.py file not found.")
        print("💡 Falling back to simplified interactive mode...")
        return run_simplified_interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\n👋 Interactive mode cancelled.")
        return 0
        
    except Exception as e:
        print(f"❌ Unexpected error in interactive mode: {str(e)}")
        print("💡 Falling back to simplified interactive mode...")
        return run_simplified_interactive_mode()


def run_simplified_interactive_mode():
    """Run a simplified interactive mode using modular components."""
    print("\n🚀 Simplified Interactive Mode")
    print("-" * 40)
    print("Available commands:")
    print("  help     - Show available commands")
    print("  create   - Create a new matrix")
    print("  list     - List all matrices")
    print("  show     - Show matrix contents")
    print("  multiply - Multiply two matrices")
    print("  add      - Add two matrices")
    print("  exit     - Exit the program")
    print()
    
    try:
        from matrixcalc.cli import MatrixCLI
        cli = MatrixCLI()
        
        while True:
            try:
                command = input("🔢 matrix> ").strip()
                
                if not command:
                    continue
                    
                if command.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if command.lower() in ['help', '?']:
                    print("Available commands: help, create, list, show, multiply, add, exit")
                    print("Example: create --type identity --size 3")
                    continue
                
                # Parse and execute command
                args = command.split()
                result = cli.run(args)
                
                if result != 0:
                    print(f"⚠ Command failed with exit code {result}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue
                
        return 0
        
    except ImportError as e:
        print(f"❌ Error: Cannot load CLI components: {str(e)}")
        print("💡 Please run 'python Matrixcodes.py' directly for full interactive mode.")
        return 1

def run_cli_mode(args: Optional[List[str]] = None):
    """Run the enhanced command-line mode."""
    try:
        from matrixcalc.cli import MatrixCLI
        cli = MatrixCLI()
        return cli.run(args)
    except ImportError as e:
        print(f"❌ Error: CLI mode not available: {str(e)}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1

def main():
    """Main entry point with mode selection."""
    # Quick check for CLI mode vs interactive mode
    if len(sys.argv) > 1 and not any(arg in ['--interactive', '-i'] for arg in sys.argv):
        # CLI mode - delegate to MatrixCLI
        print("⚡ Running CLI Mode...")
        return run_cli_mode()
    else:
        # Interactive mode
        print("🎯 Starting Interactive Mode...")
        return run_interactive_mode()

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled. Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Fatal error: {str(e)}")
        sys.exit(1)