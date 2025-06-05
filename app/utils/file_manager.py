"""
File manager for the Radiochromic Film Analyzer.

This module contains the file manager class that handles file operations
such as loading, saving, and managing recent files.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class FileManager:
    """File manager for the Radiochromic Film Analyzer."""
    
    def __init__(self, config):
        """Initialize the file manager."""
        self.config = config
        self.recent_files_path = "recent_files.json"
        self.max_recent_files = 5  # Changed from 10 to 5 as requested
        
        # Load recent files
        self.recent_files = self._load_recent_files()
        
        logger.info("File manager initialized")
    
    def _load_recent_files(self) -> List[str]:
        """Load the list of recent files."""
        try:
            if os.path.exists(self.recent_files_path):
                with open(self.recent_files_path, "r") as f:
                    data = json.load(f)
                
                # Validate the data
                if not isinstance(data, list):
                    logger.warning("Recent files data is not a list, resetting")
                    return []
                
                # Filter out files that no longer exist
                valid_files = [f for f in data if os.path.exists(f)]
                
                # If some files were filtered out, save the updated list
                if len(valid_files) != len(data):
                    logger.info(f"Removed {len(data) - len(valid_files)} non-existent files from recent files list")
                    self._save_recent_files(valid_files)
                
                return valid_files
            else:
                logger.info("No recent files found")
                return []
        except Exception as e:
            logger.error(f"Error loading recent files: {str(e)}", exc_info=True)
            return []
    
    def _save_recent_files(self, files: List[str]) -> bool:
        """Save the list of recent files."""
        try:
            with open(self.recent_files_path, "w") as f:
                json.dump(files, f)
            
            logger.debug("Saved recent files")
            return True
        except Exception as e:
            logger.error(f"Error saving recent files: {str(e)}", exc_info=True)
            return False
    
    def add_recent_file(self, file_path: str) -> bool:
        """Add a file to the recent files list."""
        try:
            # Normalize path
            file_path = os.path.abspath(file_path)
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"Cannot add non-existent file to recent files: {file_path}")
                return False
            
            # Remove file if it already exists in the list
            if file_path in self.recent_files:
                self.recent_files.remove(file_path)
            
            # Add file to the beginning of the list
            self.recent_files.insert(0, file_path)
            
            # Limit the number of recent files
            if len(self.recent_files) > self.max_recent_files:
                self.recent_files = self.recent_files[:self.max_recent_files]
            
            # Save the updated list
            self._save_recent_files(self.recent_files)
            
            logger.debug(f"Added file to recent files: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error adding recent file: {str(e)}", exc_info=True)
            return False
    
    def get_recent_files(self) -> List[str]:
        """Get the list of recent files."""
        return self.recent_files
    
    def clear_recent_files(self) -> bool:
        """Clear the list of recent files."""
        try:
            self.recent_files = []
            self._save_recent_files(self.recent_files)
            
            logger.info("Cleared recent files")
            return True
        except Exception as e:
            logger.error(f"Error clearing recent files: {str(e)}", exc_info=True)
            return False
