"""
Main application class for the Radiochromic Film Analyzer.

This module contains the main application class that initializes the UI
and connects the various components.
"""

import tkinter as tk
import logging
import os
import glob
import tempfile
import shutil
from app.ui.main_window import MainWindow
from app.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class RCAnalyzer(tk.Tk):
    """Main application class for the Radiochromic Film Analyzer."""
    
    def __init__(self):
        """Initialize the application."""
        super().__init__()
        
        logger.info("Initializing RCAnalyzer application")
        
        # Clean up log and temporary files
        self._cleanup_files()
        
        # Initialize configuration manager
        self.config_manager = ConfigManager()
        
        # Load configuration
        self.app_config = self.config_manager.load_config()
        
        # Set window size (10% taller)
        self.geometry("1200x770")  # Increased from 700 to 770 (10% taller)
        
        # Initialize main window
        self.main_window = MainWindow(self, self.app_config)
        
        # Store reference to main_window in the root window for access from child widgets
        self.main_window = self.main_window
        
        # Set up window close handler
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        logger.info("RCAnalyzer initialization complete")
    
    def _cleanup_files(self):
        """Clean up log and temporary files."""
        try:
            # Clean up log files – keep only the most recent 10
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
            if os.path.isdir(logs_dir):
                log_files = sorted(glob.glob(os.path.join(logs_dir, "rc_analyzer_*.log")))
                # Remove oldest logs while more than 10 remain
                while len(log_files) > 10:
                    old_log = log_files.pop(0)
                    try:
                        os.remove(old_log)
                        logger.debug(f"Removed old log file: {old_log}")
                    except Exception as e:
                        logger.warning(f"Could not remove log file {old_log}: {str(e)}")
            
            # Clean up temporary files in system temp directory
            temp_dir = tempfile.gettempdir()
            temp_files = glob.glob(os.path.join(temp_dir, "*_bin*.pkl"))
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    logger.debug(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Could not remove temporary file {temp_file}: {str(e)}")
            
            # Clean up local temp directory
            local_temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
            if os.path.exists(local_temp_dir):
                try:
                    # Remove all files in the temp directory
                    for file in os.listdir(local_temp_dir):
                        file_path = os.path.join(local_temp_dir, file)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            logger.warning(f"Could not remove {file_path}: {str(e)}")
                    
                    logger.info(f"Cleaned up local temp directory: {local_temp_dir}")
                except Exception as e:
                    logger.error(f"Error cleaning up local temp directory: {str(e)}", exc_info=True)
            else:
                # Create the temp directory if it doesn't exist
                os.makedirs(local_temp_dir)
                logger.info(f"Created local temp directory: {local_temp_dir}")
            
            logger.info("Cleaned up log and temporary files")
        except Exception as e:
            logger.error(f"Error cleaning up files: {str(e)}", exc_info=True)
    
    def on_close(self):
        """Handle window close event."""
        logger.info("Saving configuration before exit")
        
        # Save current configuration
        self.config_manager.save_config(self.main_window.get_config())
        
        # Clean up resources in the main window
        self.main_window.cleanup()
        
        # Clean up resources in the image processor
        if hasattr(self.main_window, 'image_processor'):
            self.main_window.image_processor.cleanup()
        
        # Clean up temporary files
        self._cleanup_files()
        
        # Destroy the window
        self.destroy()
