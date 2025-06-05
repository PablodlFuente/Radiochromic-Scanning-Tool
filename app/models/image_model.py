"""
Image model for the Radiochromic Film Analyzer.

This module contains the image model class that represents
image data and metadata.
"""

import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ImageModel:
    """Image model for the Radiochromic Film Analyzer."""
    
    def __init__(self, data=None, file_path=None):
        """Initialize the image model."""
        self.data = data
        self.file_path = file_path
        self.load_time = datetime.now()
        self.metadata = {}
        
        logger.info("Image model initialized")
    
    @property
    def width(self):
        """Get the image width."""
        if self.data is not None:
            return self.data.shape[1]
        return 0
    
    @property
    def height(self):
        """Get the image height."""
        if self.data is not None:
            return self.data.shape[0]
        return 0
    
    @property
    def channels(self):
        """Get the number of image channels."""
        if self.data is not None and len(self.data.shape) > 2:
            return self.data.shape[2]
        return 1
    
    @property
    def is_color(self):
        """Check if the image is color."""
        return self.channels > 1
    
    def get_pixel(self, x, y):
        """Get the pixel value at the specified position."""
        if self.data is None or x < 0 or y < 0 or x >= self.width or y >= self.height:
            return None
        
        return self.data[y, x]
    
    def get_region(self, x, y, width, height):
        """Get a region of the image."""
        if self.data is None:
            return None
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.width, x + width)
        y2 = min(self.height, y + height)
        
        return self.data[y1:y2, x1:x2]
    
    def get_circular_region(self, center_x, center_y, radius):
        """Get a circular region of the image."""
        if self.data is None:
            return None
        
        # Create mask
        y, x = np.ogrid[:self.height, :self.width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Apply mask
        if self.is_color:
            result = []
            for c in range(self.channels):
                result.append(self.data[:, :, c][mask])
            return np.array(result).T
        else:
            return self.data[mask]
    
    def get_square_region(self, center_x, center_y, size):
        """Get a square region of the image."""
        if self.data is None:
            return None
        
        # Create mask
        y, x = np.ogrid[:self.height, :self.width]
        mask = (abs(x - center_x) <= size) & (abs(y - center_y) <= size)
        
        # Apply mask
        if self.is_color:
            result = []
            for c in range(self.channels):
                result.append(self.data[:, :, c][mask])
            return np.array(result).T
        else:
            return self.data[mask]
