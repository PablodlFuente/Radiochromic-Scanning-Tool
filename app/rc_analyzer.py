"""
Main application class for the Radiochromic Film Analyzer.

This module contains the main application class that initializes the UI
and connects the various components.
"""

import tkinter as tk
from tkinter import messagebox
import logging
import os
import glob
import shutil
import threading
from app.ui.main_window import MainWindow
from app.utils.config_manager import ConfigManager
from app.utils.updater import UpdateChecker
from app.paths import PROJECT_ROOT

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
        
        # Check for updates on startup if enabled (after mainloop starts)
        if self.app_config.get("check_updates_on_startup", True):
            self.after(1500, self._check_updates_on_startup)
    
    def _check_updates_on_startup(self):
        """Check for updates in background and notify user if available."""
        def check_updates():
            try:
                checker = UpdateChecker()
                result = checker.check_for_updates()
                
                if result.get('success') and result.get('has_updates'):
                    latest_version = result.get('latest_version', '')
                    # Show notification on main thread
                    self.after(0, lambda: self._show_update_notification(latest_version))
            except Exception as e:
                logger.warning(f"Error checking for updates on startup: {e}")
        
        # Run in background thread to not block UI
        thread = threading.Thread(target=check_updates, daemon=True)
        thread.start()
    
    def _show_update_notification(self, latest_version):
        """Show update notification dialog."""
        msg = f"Version {latest_version} is available.\n\nGo to Help → Check for Updates to update."
        messagebox.showinfo("Update Available", msg, parent=self)

    def report_callback_exception(self, exception_type, exception, traceback):
        """Make uncaught Tk callback failures visible while preserving full logs."""
        logger.error(
            "Unhandled UI callback exception",
            exc_info=(exception_type, exception, traceback),
        )
        messagebox.showerror(
            "Unexpected application error",
            f"{exception_type.__name__}: {exception}\n\n"
            "The operation did not complete. Full diagnostic details were written to the logs folder.",
            parent=self,
        )
    
    def _cleanup_files(self):
        """Clean up log and temporary files."""
        try:
            # Clean up log files – keep only the most recent 10
            logs_dir = os.path.join(PROJECT_ROOT, "logs")
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
            
            # Temporary data is owned and cleaned by each ImageProcessor session.
            # Other running instances may still be using their directories.
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
