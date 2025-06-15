#!/usr/bin/env python3
"""
Radiochromic Film Analyzer - Main Entry Point

This script initializes and runs the Radiochromic Film Analyzer application.
"""

import sys
import os
import logging
import datetime
from app.rc_analyzer import RCAnalyzer

# Configure logging: create one log file per run, keep in 'logs' directory, include timestamp in filename
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create timestamped log filename, e.g. rc_analyzer_20250614_193835.log
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(logs_dir, f"rc_analyzer_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
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
