"""
Core functionality for the AutoMeasurements plugin.

This package contains the core classes for film and circle detection,
metadata extraction, CTR management, file data management, and CSV export.
"""

from .formatter import MeasurementFormatter
from .detection import DetectionEngine
from .metadata import MetadataExtractor
from .ctr_manager import CTRManager
from .file_manager import FileDataManager
from .exporter import CSVExporter

__all__ = [
    'MeasurementFormatter',
    'DetectionEngine',
    'MetadataExtractor',
    'CTRManager',
    'FileDataManager',
    'CSVExporter'
]
