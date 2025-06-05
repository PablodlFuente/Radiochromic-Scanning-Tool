"""
Configuration manager for the Radiochromic Film Analyzer.

This module contains the configuration manager class that handles loading
and saving application configuration.
"""

import json
import os
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for the Radiochromic Film Analyzer."""
    
    def __init__(self, config_file="rc_config.json"):
        """Initialize the configuration manager."""
        self.config_file = config_file
        self.default_config = {
            "remove_background": False,
            "negative_mode": False,
            "recent_files": []
        }
        
        logger.info("Configuration manager initialized")
    
    def load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                
                logger.info("Configuration loaded from file")
                return config
            
            logger.info("Configuration file not found, using defaults")
            return self.default_config.copy()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
            return self.default_config.copy()
    
    def save_config(self, config):
        """Save configuration to file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=4)
            
            logger.info("Configuration saved to file")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}", exc_info=True)
            return False
