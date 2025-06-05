#!/usr/bin/env python3
"""
Radiochromic Film Analyzer - Main Entry Point

This script initializes and runs the Radiochromic Film Analyzer application.
"""

import sys
import os
import logging
from app.rc_analyzer import RCAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rc_analyzer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application."""
    try:
        logger.info("Starting Radiochromic Film Analyzer")
        app = RCAnalyzer()
        app.mainloop()
        logger.info("Application closed normally")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
