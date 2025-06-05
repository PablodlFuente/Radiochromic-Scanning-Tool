"""
Calibration model for the Radiochromic Film Analyzer.

This module contains the calibration model class that represents
calibration data.
"""

import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CalibrationModel:
    """Calibration model for the Radiochromic Film Analyzer."""
    
    def __init__(self, dose=0.0, factor=1.0, offset=0.0, date=None):
        """Initialize the calibration model."""
        self.dose = dose
        self.factor = factor
        self.offset = offset
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        
        logger.info("Calibration model initialized")
    
    @classmethod
    def from_dict(cls, data):
        """Create a calibration model from a dictionary."""
        return cls(
            dose=data.get("dose", 0.0),
            factor=data.get("factor", 1.0),
            offset=data.get("offset", 0.0),
            date=data.get("date")
        )
    
    def to_dict(self):
        """Convert the calibration model to a dictionary."""
        return {
            "dose": self.dose,
            "factor": self.factor,
            "offset": self.offset,
            "date": self.date
        }
    
    @classmethod
    def load_from_file(cls, file_path="rc_calibration.json"):
        """Load calibration from a file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                logger.info("Calibration loaded from file")
                return cls.from_dict(data)
            
            logger.info("Calibration file not found")
            return None
        except Exception as e:
            logger.error(f"Error loading calibration: {str(e)}", exc_info=True)
            return None
    
    def save_to_file(self, file_path="rc_calibration.json"):
        """Save calibration to a file."""
        try:
            with open(file_path, "w") as f:
                json.dump(self.to_dict(), f, indent=4)
            
            logger.info("Calibration saved to file")
            return True
        except Exception as e:
            logger.error(f"Error saving calibration: {str(e)}", exc_info=True)
            return False
    
    def calculate_dose(self, pixel_value):
        """Calculate dose from pixel value using calibration parameters."""
        try:
            # Simple linear model: dose = factor * pixel_value + offset
            return self.factor * pixel_value + self.offset
        except Exception as e:
            logger.error(f"Error calculating dose: {str(e)}", exc_info=True)
            return None
