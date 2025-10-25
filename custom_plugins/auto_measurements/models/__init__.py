"""
Data models for the AutoMeasurements plugin.

This module contains data classes and constants used throughout the plugin:
- Constants: CTR_DOSE_THRESHOLD, DASH_COUNT, Y_TOLERANCE_FACTOR
- DetectionParams: Parameters for film and circle detection
- Circle: Represents a circular ROI
- Film: Represents a rectangular film
- MeasurementResult: Results from a circle measurement
"""

from typing import List, Optional, Tuple

# DATA CLASSES AND CONSTANTS
# ============================================================================

# Dose (Gy) threshold below which a circle is considered a control (background) region
CTR_DOSE_THRESHOLD = 0.05

# Drawing constants
DASH_COUNT = 20  # Number of dashes for CTR circle visualization
Y_TOLERANCE_FACTOR = 0.5  # Factor for matrix row grouping


class DetectionParams:
    """Parameters for film and circle detection."""
    
    def __init__(self, 
                 rc_threshold: int = 180,
                 rc_min_area: int = 5000,
                 min_circle_radius: int = 200,
                 max_circle_radius: int = 400,
                 min_distance: int = 200,
                 param1: int = 15,
                 param2: int = 40,
                 default_diameter: int = 300,
                 restrict_diameter: bool = False):
        self.rc_threshold = rc_threshold
        self.rc_min_area = rc_min_area
        self.min_circle_radius = min_circle_radius
        self.max_circle_radius = max_circle_radius
        self.min_distance = min_distance
        self.param1 = param1
        self.param2 = param2
        self.default_diameter = default_diameter
        self.restrict_diameter = restrict_diameter
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'rc_threshold': self.rc_threshold,
            'rc_min_area': self.rc_min_area,
            'min_circle_radius': self.min_circle_radius,
            'max_circle_radius': self.max_circle_radius,
            'min_distance': self.min_distance,
            'param1': self.param1,
            'param2': self.param2,
            'default_diameter': self.default_diameter,
            'restrict_diameter': self.restrict_diameter
        }


class Circle:
    """Represents a circular ROI."""
    
    def __init__(self, cx: int, cy: int, radius: int, name: str = "", is_ctr: bool = False,
                 dose: Optional[float] = None, std: Optional[float] = None,
                 avg: Optional[float] = None, avg_unc: Optional[float] = None):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.name = name
        self.is_ctr = is_ctr
        self.dose = dose
        self.std = std
        self.avg = avg
        self.avg_unc = avg_unc
    
    def coords(self) -> Tuple[int, int, int]:
        """Return (cx, cy, radius) tuple."""
        return (self.cx, self.cy, self.radius)


class Film:
    """Represents a rectangular film."""
    
    def __init__(self, x: int, y: int, width: int, height: int, name: str, circles: Optional[List[Circle]] = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.name = name
        self.circles = circles if circles is not None else []
    
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)
    
    def contains_point(self, px: int, py: int) -> bool:
        """Check if point is inside film."""
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)


class MeasurementResult:
    """Results from a circle measurement."""
    
    def __init__(self, film_name: str, circle_name: str, dose: float, std: float,
                 avg: float, avg_unc: float, ci95: float,
                 file_path: Optional[str] = None, date: Optional[str] = None):
        self.film_name = film_name
        self.circle_name = circle_name
        self.dose = dose
        self.std = std
        self.avg = avg
        self.avg_unc = avg_unc
        self.ci95 = ci95
        self.file_path = file_path
        self.date = date
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'film_name': self.film_name,
            'circle_name': self.circle_name,
            'dose': self.dose,
            'std': self.std,
            'avg': self.avg,
            'avg_unc': self.avg_unc,
            'ci95': self.ci95,
            'file_path': self.file_path,
            'date': self.date
        }


# ============================================================================
# UTILITY CLASSES
# ============================================================================


__all__ = [
    'CTR_DOSE_THRESHOLD',
    'DASH_COUNT', 
    'Y_TOLERANCE_FACTOR',
    'DetectionParams',
    'Circle',
    'Film',
    'MeasurementResult'
]
