"""AutoMeasurements plugin - Multi-file package version

Detects rectangular radiochromic films and circular ROIs inside them,
computes dose and uncertainty, and shows results in a TreeView with export.
Includes complete CTR (Control) circle functionality for background subtraction.
"""

from __future__ import annotations

import numpy as np
import cv2
import logging

from .ui.main_tab import AutoMeasurementsTab

# Plugin interface
TAB_TITLE = "AutoMeasurements v2"

# Global instance to track the plugin for configuration updates
_AUTO_MEASUREMENTS_INSTANCE = None

def setup(main_window, notebook, image_processor):
    global _AUTO_MEASUREMENTS_INSTANCE
    _AUTO_MEASUREMENTS_INSTANCE = AutoMeasurementsTab(main_window, notebook, image_processor)
    return _AUTO_MEASUREMENTS_INSTANCE.frame


_OVERLAY: dict | None = None  # Holds last detection for drawing

# Keys used in _OVERLAY:
#   "films": list[(x, y, w, h)]
#   "circles": list[(cx, cy, r)]
#   "_shape": (h, w) of detection image
#   "highlight": ("film"|"circle", coords) – optional highlighted shape
#   "ctr_map": dict[film_name, circle_id] - map of control circles
#   "item_to_shape": dict[item_id, (type, coords)] - map of items to shapes

# Dose (Gy) threshold below which a circle is considered a control (background) region
CTR_DOSE_THRESHOLD = 0.05

def update_overlay_shape(image_shape):
    """Update the _OVERLAY shape after loading a new image.
    
    This ensures manual shapes can be drawn correctly even before Start Detection.
    Call this after image_processor.load_image() completes.
    
    Args:
        image_shape: Tuple (height, width) of the loaded image
    """
    global _OVERLAY
    if _OVERLAY is not None:
        _OVERLAY["_shape"] = image_shape[:2]  # Only take H, W
        logging.debug(f"[update_overlay_shape] Updated _OVERLAY shape to {image_shape[:2]}")
    else:
        # Initialize _OVERLAY if it doesn't exist yet
        _OVERLAY = {
            "films": [], 
            "circles": [], 
            "_shape": image_shape[:2], 
            "scale": 1.0, 
            "ctr_map": {}, 
            "item_to_shape": {}
        }
        logging.debug(f"[update_overlay_shape] Initialized _OVERLAY with shape {image_shape[:2]}")

def process(image: np.ndarray):
    """Draw bounding rectangles and circles from last detection.
    
    This function is called on every display update, so minimize logging
    to avoid performance issues.
    
    Args:
        image: Input image (grayscale or RGB) to draw overlays on
        
    Returns:
        Image with drawn overlays (films, circles, CTR markers, highlights)
    """
    global _OVERLAY
    
    # Check if there's anything to draw (films OR circles)
    if _OVERLAY is None:
        return image
    
    has_films = "films" in _OVERLAY and _OVERLAY["films"]
    has_circles = "circles" in _OVERLAY and _OVERLAY["circles"]
    
    if not has_films and not has_circles:
        return image
    
    # Create output image (convert to color if needed)
    if len(image.shape) == 2 or image.shape[2] == 1:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()
    
    h, w = image.shape[:2]
    orig_h, orig_w = _OVERLAY.get("_shape", (0, 0))
    
    if orig_h == 0 or orig_w == 0:
        logging.warning("[process] Original shape is 0x0 - cannot scale properly")
        return out
    
    sx, sy = w / orig_w, h / orig_h
    
    # Draw films (green rectangles)
    for (x, y, w_rect, h_rect) in _OVERLAY.get("films", []):
        cv2.rectangle(out, 
                      (int(x * sx), int(y * sy)),
                      (int((x + w_rect) * sx), int((y + h_rect) * sy)),
                      (0, 255, 0), 2)
    
    # Draw circles (green) or CTR circles (orange dashed)
    # Pre-compute CTR circle coordinates for efficient lookup
    ctr_coords = set()
    if "ctr_map" in _OVERLAY and "item_to_shape" in _OVERLAY:
        for ctr_item_id in _OVERLAY["ctr_map"].values():
            if ctr_item_id in _OVERLAY["item_to_shape"]:
                shape_type, coords = _OVERLAY["item_to_shape"][ctr_item_id]
                if shape_type == "circle":
                    ctr_coords.add(coords)
    
    for (cx, cy, r) in _OVERLAY.get("circles", []):
        # Check if this circle is marked as CTR
        is_ctr = (cx, cy, r) in ctr_coords
        
        if is_ctr:
            # Draw dashed circle (orange)
            _draw_dashed_circle(out, (int(cx * sx), int(cy * sy)), int(r * sx), (0, 165, 255), 2)
        else:
            # Draw normal circle (green)
            cv2.circle(out, (int(cx * sx), int(cy * sy)), int(r * sx), 
                      (0, 255, 0), 2, lineType=cv2.LINE_AA)
    
    # Draw highlighted shape on top (yellow)
    hl = _OVERLAY.get("highlight")
    if hl is not None:
        stype, coords = hl
        if stype == "film":
            x, y, w_rect, h_rect = coords
            cv2.rectangle(out, 
                          (int(x * sx), int(y * sy)),
                          (int((x + w_rect) * sx), int((y + h_rect) * sy)),
                          (0, 255, 255), 3)
        elif stype == "circle":
            cx, cy, r = coords
            cv2.circle(out, (int(cx * sx), int(cy * sy)), int(r * sx), (0, 255, 255), 3)
    return out


def _draw_dashed_circle(img, center, radius, color, thickness):
    """Draw a dashed circle for CTR circles."""
    import math
    
    # Number of dashes
    dash_count = 20
    dash_length = 2 * math.pi * radius / (2 * dash_count)
    
    for i in range(dash_count):
        if i % 2 == 0:  # Draw every other dash
            start_angle = i * (360 / dash_count)
            end_angle = (i + 0.5) * (360 / dash_count)
            
            # Convert angles to radians
            start_rad = math.radians(start_angle)
            end_rad = math.radians(end_angle)
            
            # Calculate start and end points
            start_x = int(center[0] + radius * math.cos(start_rad))
            start_y = int(center[1] + radius * math.sin(start_rad))
            end_x = int(center[0] + radius * math.cos(end_rad))
            end_y = int(center[1] + radius * math.sin(end_rad))
            
            # Draw arc segment
            cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, thickness)


# -------------------------------------------------------------------------
# Configuration change handler
# -------------------------------------------------------------------------

def on_config_change(config, image_processor):
    """Handle configuration changes - refresh measurements if uncertainty method changed."""
    global _AUTO_MEASUREMENTS_INSTANCE
    
    try:
        if _AUTO_MEASUREMENTS_INSTANCE is not None:
            # Force refresh of all existing measurements with new uncertainty method
            _AUTO_MEASUREMENTS_INSTANCE._refresh_all_measurements()
            logging.info("Auto-measurements refreshed due to uncertainty method change")
        else:
            logging.debug("Auto-measurements instance not available for config change notification")
    except Exception as e:
        logging.error(f"Error refreshing auto-measurements after config change: {e}")


__all__ = ['TAB_TITLE', 'setup', 'process', 'on_config_change']
