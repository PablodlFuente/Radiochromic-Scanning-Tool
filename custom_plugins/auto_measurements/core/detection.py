"""
Film and circle detection algorithms.

This module contains the DetectionEngine class for detecting
radiochromic films and circular ROIs using OpenCV.
"""

from typing import List, Tuple
import numpy as np
import cv2

from ..models import DetectionParams, Y_TOLERANCE_FACTOR
from .formatter import MeasurementFormatter


class DetectionEngine:
    """Handles film and circle detection algorithms."""
    
    def __init__(self):
        self.formatter = MeasurementFormatter()
    
    @staticmethod
    def ensure_uint8(img: np.ndarray) -> np.ndarray:
        """Convert any dtype image to uint8 0-255 for OpenCV edge detection."""
        if img.dtype == np.uint8:
            return img
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img_norm.astype(np.uint8)
    
    def detect_films(self, gray: np.ndarray, params: DetectionParams) -> List[Tuple[int, int, int, int]]:
        """Detect rectangular films using fixed threshold.
        
        Args:
            gray: Grayscale image
            params: Detection parameters
            
        Returns:
            List of (x, y, width, height) tuples for detected films
        """
        gray8 = self.ensure_uint8(gray)
        
        thresh_val = params.rc_threshold
        _, thresh = cv2.threshold(gray8, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = params.rc_min_area
        films = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                films.append((x, y, w, h))
        
        # Sort by y-coordinate (top to bottom)
        films.sort(key=lambda b: b[1])
        return films
    
    def detect_circles(self, roi: np.ndarray, params: DetectionParams) -> List[Tuple[int, int, int]]:
        """Detect circles via Hough transform.
        
        Args:
            roi: Region of interest to search
            params: Detection parameters
            
        Returns:
            List of (cx, cy, radius) tuples for detected circles
        """
        min_r = params.min_circle_radius
        max_r = params.max_circle_radius
        roi8 = self.ensure_uint8(roi)
        roi_blur = cv2.medianBlur(roi8, 5)
        
        if min_r < 5:
            min_r = 5
        if max_r < min_r:
            max_r = min_r + 10
        
        param1_val = params.param1
        param2_val = params.param2
        min_dist_val = params.min_distance
        
        circles = cv2.HoughCircles(
            roi_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist_val if min_dist_val > 0 else max(30, min_r // 2),
            param1=param1_val,
            param2=param2_val,
            minRadius=min_r,
            maxRadius=max_r,
        )
        
        result = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for (cx, cy, r) in circles:
                result.append((int(cx), int(cy), int(r)))
        return result
    
    def organize_circles_as_matrix(self, circles: List[Tuple[int, int, int]]) -> List[Tuple[str, Tuple[int, int, int]]]:
        """Organize circles into a matrix grid (row-column) and assign names like C11, C12, C21, etc.
        
        Args:
            circles: List of (cx, cy, radius) tuples
            
        Returns:
            List of (name, (cx, cy, r)) tuples with matrix-based names
        """
        if not circles:
            return []
        
        # Extract center coordinates
        centers = [(cx, cy, r) for cx, cy, r in circles]
        
        # Group circles into rows based on Y coordinate
        sorted_by_y = sorted(centers, key=lambda c: c[1])  # Sort by Y
        
        rows = []
        current_row = [sorted_by_y[0]]
        
        # Tolerance for grouping into same row (average radius * Y_TOLERANCE_FACTOR)
        avg_radius = np.mean([r for _, _, r in circles])
        y_tolerance = avg_radius * Y_TOLERANCE_FACTOR
        
        for i in range(1, len(sorted_by_y)):
            prev_y = current_row[-1][1]
            curr_y = sorted_by_y[i][1]
            
            if abs(curr_y - prev_y) <= y_tolerance:
                current_row.append(sorted_by_y[i])
            else:
                rows.append(current_row)
                current_row = [sorted_by_y[i]]
        
        # Don't forget the last row
        if current_row:
            rows.append(current_row)
        
        # Sort circles within each row by X coordinate (left to right)
        for row in rows:
            row.sort(key=lambda c: c[0])
        
        # Assign names based on matrix position
        result = []
        for row_idx, row in enumerate(rows, start=1):
            for col_idx, (cx, cy, r) in enumerate(row, start=1):
                name = f"C{row_idx}{col_idx}"
                result.append((name, (cx, cy, r)))
        
        return result




__all__ = ['DetectionEngine']
