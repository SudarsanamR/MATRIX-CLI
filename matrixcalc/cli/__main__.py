"""Main entry point for the matrix calculator CLI."""

import sys
import argparse
from typing import List, Optional

from .commands import MatrixCLI
from ..config.settings import Config
from ..logging.setup import setup_logging
from ..plugins import plugin_manager
from ..core.performance import parallel_processor


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the matrix calculator CLI.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Initialize logging
        setup_logging()
        
        # Load plugins if enabled
        if Config.enable_plugins:
            plugin_manager.load_all_plugins()
        
        # Create and run CLI
        cli = MatrixCLI()
        exit_code = cli.run(args)
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 130
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return 1
    finally:
        # Cleanup
        if Config.enable_plugins:
            plugin_manager.shutdown()
        parallel_processor.shutdown()


if __name__ == '__main__':
    sys.exit(main())