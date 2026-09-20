"""
Configuration manager for the Radiochromic Film Analyzer.

This module contains the configuration manager class that handles loading
and saving application configuration.
"""

import json
import os
import logging
import tempfile
from copy import deepcopy
from app.models.config_model import DEFAULT_CONFIG
from app.paths import CONFIG_FILE

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for the Radiochromic Film Analyzer."""
    
    def __init__(self, config_file=None):
        """Initialize the configuration manager."""
        self.config_file = os.fspath(config_file or CONFIG_FILE)
        self.default_config = DEFAULT_CONFIG
        
        logger.info("Configuration manager initialized")
    
    def load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                
                merged = deepcopy(self.default_config)
                merged.update(config)
                logger.info("Configuration loaded from file")
                return merged
            
            logger.info("Configuration file not found, using defaults")
            return deepcopy(self.default_config)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
            return deepcopy(self.default_config)
    
    def save_config(self, config):
        """Save configuration to file."""
        temporary_name = None
        try:
            directory = os.path.dirname(os.path.abspath(self.config_file))
            os.makedirs(directory, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix="rc_config_", suffix=".json", dir=directory
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_name, self.config_file)
            
            logger.info("Configuration saved to file")
            return True
        except Exception as e:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
            logger.error(f"Error saving configuration: {str(e)}", exc_info=True)
            return False
