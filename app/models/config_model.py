"""
Configuration model for the Radiochromic Film Analyzer.

This module contains the configuration model class that represents
application configuration.
"""

import json
import os
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class ConfigModel:
    """Configuration model for the Radiochromic Film Analyzer."""
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize the configuration model with default values.
        
        Args:
            data: Optional dictionary containing configuration values
        """
        self.data: Dict[str, Any] = data or {}
        
        # Set defaults if not present
        self._set_defaults()
        
        logger.info("Configuration model initialized")
    
    def _set_defaults(self) -> None:
        """Set default values for missing configuration items."""
        defaults: Dict[str, Any] = {
            "remove_background": False,
            "negative_mode": False,
            "recent_files": [],
            "colormap": "viridis",
            "use_gpu": False,
            "gpu_force_enabled": False,
            "use_multithreading": True,
            "num_threads": os.cpu_count() or 4,
            "auto_measure": False,
            "max_memory_percent": 75,
            "max_cache_mb": 512,
            "log_level": "INFO",
            "detailed_logging": False
        }
        
        for key, value in defaults.items():
            if key not in self.data:
                self.data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: The configuration key to retrieve
            default: The default value if key is not found
            
        Returns:
            The configuration value or default if not found
        """
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: The configuration key to set
            value: The value to set
        """
        self.data[key] = value
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update configuration with new data.
        
        Args:
            data: Dictionary of configuration values to update
        """
        self.data.update(data)
    
    @classmethod
    def load_from_file(cls, file_path: str = "rc_config.json") -> 'ConfigModel':
        """Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            A ConfigModel instance with loaded configuration
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                logger.info("Configuration loaded from file")
                return cls(data)
            
            logger.info("Configuration file not found, using defaults")
            return cls()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {str(e)}", exc_info=True)
            logger.info("Using default configuration due to parsing error")
            return cls()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
            return cls()
    
    def save_to_file(self, file_path: str = "rc_config.json") -> bool:
        """Save configuration to a file.
        
        Args:
            file_path: Path to save the configuration file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(file_path, "w") as f:
                json.dump(self.data, f, indent=4)
            
            logger.info("Configuration saved to file")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}", exc_info=True)
            return False
    
    def validate(self) -> Dict[str, str]:
        """Validate configuration values.
        
        Returns:
            Dictionary of validation errors, empty if all valid
        """
        errors: Dict[str, str] = {}
        
        # Validate num_threads
        if "num_threads" in self.data:
            if not isinstance(self.data["num_threads"], int) or self.data["num_threads"] < 1:
                errors["num_threads"] = "Number of threads must be a positive integer"
        
        # Validate memory percent
        if "max_memory_percent" in self.data:
            mem_pct = self.data["max_memory_percent"]
            if not isinstance(mem_pct, (int, float)) or mem_pct < 10 or mem_pct > 95:
                errors["max_memory_percent"] = "Memory percentage must be between 10 and 95"
        
        # Validate cache size
        if "max_cache_mb" in self.data:
            cache_mb = self.data["max_cache_mb"]
            if not isinstance(cache_mb, (int, float)) or cache_mb < 16:
                errors["max_cache_mb"] = "Cache size must be at least 16 MB"
        
        return errors
