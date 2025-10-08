"""AutoMeasurements plugin - Improved version with full CTR functionality and manual metadata selection

Detects rectangular radiochromic films and circular ROIs inside them,
computes dose and uncertainty, and shows results in a TreeView with export.
Includes complete CTR (Control) circle functionality for background subtraction.
Includes manual metadata selection when resolution is not found automatically.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tkinter as tk
import logging
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from tkinter import ttk, filedialog, messagebox, simpledialog
import tkinter.font as tkfont
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3D
from math import log10, floor, sqrt

# Import tkcalendar - may need to install with pip install tkcalendar
try:
    from tkcalendar import Calendar
except ImportError:
    # Define a fallback calendar
    class Calendar:
        def __init__(self, master=None, **kw):
            self.toplevel = tk.Toplevel(master)
            self.toplevel.title("Calendar (Not Available)")
            tk.Label(self.toplevel, text="tkcalendar package not installed.", pady=20, padx=20).pack()
            tk.Label(self.toplevel, text="Please install with: pip install tkcalendar", pady=10).pack()
            tk.Button(self.toplevel, text="Close", command=self.toplevel.destroy).pack(pady=20)
        
        def selection_get(self):
            today = datetime.date.today()
            return today

# Plugin interface ---------------------------------------------------------
TAB_TITLE = "AutoMeasurements"

# Global instance to track the plugin for configuration updates
_AUTO_MEASUREMENTS_INSTANCE = None

def setup(main_window, notebook, image_processor):
    global _AUTO_MEASUREMENTS_INSTANCE
    _AUTO_MEASUREMENTS_INSTANCE = AutoMeasurementsTab(main_window, notebook, image_processor)
    return _AUTO_MEASUREMENTS_INSTANCE.frame


# NOTE: _OVERLAY is a global variable used for sharing detection state between plugin and drawing code.
# In multi-threaded or async contexts, this must be protected for thread safety.
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

def process(image: np.ndarray):
    """Draw bounding rectangles and circles from last detection."""
    global _OVERLAY
    if _OVERLAY is None or "films" not in _OVERLAY or not _OVERLAY["films"]:
        return image
    
    # Create output image (convert to color if needed)
    if len(image.shape) == 2 or image.shape[2] == 1:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()
    
    h, w = image.shape[:2]
    orig_h, orig_w = _OVERLAY["_shape"]
    if orig_h == 0 or orig_w == 0:
        return out
    sx, sy = w / orig_w, h / orig_h
    
    # Draw films (green)
    for (x, y, w_rect, h_rect) in _OVERLAY["films"]:
        cv2.rectangle(out, 
                      (int(x * sx), int(y * sy)),
                      (int((x + w_rect) * sx), int((y + h_rect) * sy)),
                      (0, 255, 0), 2)
    
    # Draw circles
    for (cx, cy, r) in _OVERLAY.get("circles", []):
        # Check if circle is CTR
        is_ctr = any((cx, cy, r) == coords 
                    for ctr_id, coords in _OVERLAY.get("item_to_shape", {}).items() 
                    if _OVERLAY.get("ctr_map", {}).get(_get_film_name_for_circle(ctr_id)) == ctr_id)
        
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


def _get_film_name_for_circle(circle_id):
    """Helper function to get film name for a circle item_id."""
    # This would need to be implemented based on how the tree structure works
    # For now, we'll use a placeholder approach
    return None


# -------------------------------------------------------------------------

class AutoMeasurementsTab(ttk.Frame):
    def __init__(self, main_window, notebook, image_processor):
        self.item_to_shape: dict[str, tuple] = {}  # maps TreeView items to shape info
        self.main_window = main_window
        self.image_processor = image_processor
        self.frame = ttk.Frame(notebook)
        
        # Reference to the image canvas for manual drawing
        self._canvas = (
            self.main_window.image_panel.canvas if hasattr(self.main_window, "image_panel") else None
        )

        # UI setup
        self._setup_ui()
        
        # Storage
        self.results = []  # List[Dict]
        self.original_radii = {}  # Store original radii for circle items
        # Map film name -> circle_id of control (CTR) circle
        self.ctr_map: dict[str, str] = {}
        # Store original measurements before CTR subtraction
        self.original_measurements: dict[str, dict] = {}
        # Store original values for unit conversion
        self.original_values = {}  # key: item_id, value: dict of numeric values
        
        # Drawing attributes
        self.draw_mode = None
        self.draw_dims = None
        self._dims_window = None
        self._dim_vars = []
        self._last_cursor = None
        self.preview_tag = "draw_preview"  # Unique tag for canvas preview items

        # Add sorting state variables
        self.sort_column = None
        self.sort_reverse = False
        self.name_sort_mode = "coords"  # 'coords' or 'name'
        
        # Session storage for manual metadata selection
        self.session_metadata_key = None  # Store selected metadata key for session
        self.metadata_date = None  # Store date from metadata
        
        # Storage for original parameter values (for unit conversion)
        self.original_parameters = {}
        self.parameters_converted = False
        self.updating_parameters_programmatically = False  # New flag

    def _setup_ui(self):
        """Setup the user interface."""
        # Metadata section
        metadata_frame = ttk.Frame(self.frame)
        metadata_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.use_metadata_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(metadata_frame, text="Use Metadata", variable=self.use_metadata_var,
                        command=self._update_unit_conversion).pack(side=tk.LEFT)

        # Frame for resolution and conversion factor labels
        info_frame = ttk.Frame(self.frame)
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 5))
        
        # Label to show detected resolution (DPI)
        self.resolution_label = ttk.Label(info_frame, text="Resolution: -", font=("Arial", 9, "italic"))
        self.resolution_label.pack(side=tk.LEFT, anchor=tk.W)
        
        # Create a new frame for date
        date_frame = ttk.Frame(self.frame)
        date_frame.pack(fill=tk.X, padx=20, pady=(0, 5))
        
        # Date label and entries for YYYY-MM-DD
        ttk.Label(date_frame, text="Date:", font=("Arial", 9)).pack(side=tk.LEFT, anchor=tk.W)
        
        # Create a frame for the date inputs
        date_inputs_frame = ttk.Frame(date_frame)
        date_inputs_frame.pack(side=tk.LEFT, padx=(5, 0))
        
        # Year entry
        self.year_var = tk.StringVar()
        self.year_entry = ttk.Entry(date_inputs_frame, textvariable=self.year_var, width=5)
        self.year_entry.pack(side=tk.LEFT)
        ttk.Label(date_inputs_frame, text="-").pack(side=tk.LEFT)
        
        # Month entry
        self.month_var = tk.StringVar()
        self.month_entry = ttk.Entry(date_inputs_frame, textvariable=self.month_var, width=3)
        self.month_entry.pack(side=tk.LEFT)
        ttk.Label(date_inputs_frame, text="-").pack(side=tk.LEFT)
        
        # Day entry
        self.day_var = tk.StringVar()
        self.day_entry = ttk.Entry(date_inputs_frame, textvariable=self.day_var, width=3)
        self.day_entry.pack(side=tk.LEFT)
        
        # Combined date variable (for compatibility with existing code)
        self.date_var = tk.StringVar()
        
        # Update the combined date_var when any component changes
        def update_combined_date(*args):
            try:
                year = self.year_var.get().strip()
                month = self.month_var.get().strip()
                day = self.day_var.get().strip()
                
                if year and month and day:
                    self.date_var.set(f"{year}-{month.zfill(2)}-{day.zfill(2)}")
                else:
                    self.date_var.set("")
                    
                # Hide metadata label if user has entered a date
                if self.date_var.get():
                    self.metadata_date_label.config(text="")
                elif hasattr(self, "metadata_date") and self.metadata_date:
                    self.metadata_date_label.config(text=f"(Metadata: {self.metadata_date})")
            except Exception:
                pass
        
        self.year_var.trace_add("write", update_combined_date)
        self.month_var.trace_add("write", update_combined_date)
        self.day_var.trace_add("write", update_combined_date)
        
        # Calendar button for date selection
        def show_calendar():
            from tkcalendar import Calendar
            
            top = tk.Toplevel(self.frame)
            top.title("Select Date")
            
            today = datetime.date.today()
            cal = Calendar(top, selectmode="day", year=today.year, month=today.month, day=today.day)
            cal.pack(pady=10)
            
            def set_date():
                selected_date = cal.selection_get()
                self.year_var.set(str(selected_date.year))
                self.month_var.set(str(selected_date.month).zfill(2))
                self.day_var.set(str(selected_date.day).zfill(2))
                top.destroy()
            
            ttk.Button(top, text="Select", command=set_date).pack(pady=10)
        
        calendar_button = ttk.Button(date_frame, text="📅", width=3, command=show_calendar)
        calendar_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Placeholder for metadata date
        self.metadata_date_label = ttk.Label(date_frame, text="", font=("Arial", 9, "italic"), foreground="gray")
        self.metadata_date_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Label to show conversion factor
        self.conversion_label = ttk.Label(info_frame, text="", font=("Arial", 9, "italic"), foreground="gray")
        self.conversion_label.pack(side=tk.LEFT, padx=(10, 0), anchor=tk.W)
        
        # Add button for manual metadata selection
        ttk.Button(metadata_frame, text="Select DPI from Metadata", 
                   command=self._manual_metadata_selection).pack(side=tk.LEFT, padx=(10, 0))
        
        # Title label
        ttk.Label(self.frame, text="Auto Measurements", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Main button frame
        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill=tk.X, padx=10)
        ttk.Button(btn_frame, text="Start Detection", command=self.start_detection).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Clear", command=self._clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Add RC", command=self._add_manual_film).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Add Circle", command=self._add_manual_circle).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5)

        # CTR control frame
        ctr_frame = ttk.Frame(self.frame)
        ctr_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(ctr_frame, text="Toggle CTR", command=self._toggle_ctr_manual).pack(side=tk.LEFT)
        
        # Add CTR subtraction checkbox
        self.subtract_ctr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctr_frame, text="Subtract CTR Background", variable=self.subtract_ctr_var,
                        command=self._update_ctr_subtraction).pack(side=tk.LEFT, padx=(10, 0))

        # TreeView setup
        self._setup_treeview()
        
        # Detection parameter panels
        self._setup_detection_params()

    def _setup_treeview(self):
        """Setup the TreeView widget."""
        # TreeView columns
        cols = ["dose", "sigma", "avg", "avg_unc"]

        self.tree = ttk.Treeview(self.frame, columns=tuple(cols), show="tree headings")
        self.tree.heading("#0", text="Element")
        self.tree.heading("dose", text="Dose")
        self.tree.heading("sigma", text="Uncertainty")
        self.tree.heading("avg", text="Average")
        self.tree.heading("avg_unc", text="Avg. Uncertainty")
        
        # Column widths
        self.tree.column("#0", width=120, anchor=tk.W)
        self.tree.column("dose", width=80, anchor=tk.CENTER)
        self.tree.column("sigma", width=100, anchor=tk.CENTER)
        self.tree.column("avg", width=80, anchor=tk.CENTER)
        self.tree.column("avg_unc", width=80, anchor=tk.CENTER)
        
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure CTR style
        self.tree.tag_configure("ctr", background="#FFE4B5", foreground="#8B4513", font=("TkDefaultFont", 9, "underline"))
        
        # Bind events
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Delete>", self._on_delete_key)
        self.tree.bind("<Button-3>", self._on_right_click)
        self.tree.bind("<Double-1>", self._on_edit_label)
        self.tree.bind("<Control-Button-1>", self._toggle_ctr_circle)
        
        # Bind column headers for sorting
        self.tree.heading("#0", command=lambda: self._sort_by_column("#0"))
        self.tree.heading("avg", command=lambda: self._sort_by_column("avg"))
        self.tree.heading("avg_unc", command=lambda: self._sort_by_column("avg_unc"))

    def _setup_detection_params(self):
        """Setup detection parameter frames."""
        # Film detection parameters
        self.rc_thresh_var = tk.IntVar(value=180)
        self.rc_min_area_var = tk.IntVar(value=5000)
        film_frame = ttk.LabelFrame(self.frame, text="RC Detection")
        film_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(film_frame, text="Threshold (0-255):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(film_frame, textvariable=self.rc_thresh_var, width=6).grid(row=0, column=1, sticky=tk.W)
        
        # Store reference to area label for unit updates
        self.area_label = ttk.Label(film_frame, text="Min Area (px²):")
        self.area_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(film_frame, textvariable=self.rc_min_area_var, width=8).grid(row=1, column=1, sticky=tk.W)

        # Circle detection parameters
        self.min_circle_var = tk.IntVar(value=200)
        self.max_circle_var = tk.IntVar(value=400)
        self.min_dist_var = tk.IntVar(value=200)
        self.param1_var = tk.IntVar(value=15)
        self.param2_var = tk.IntVar(value=40)
        self.default_diameter_var = tk.IntVar(value=300)
        
        circle_frame = ttk.LabelFrame(self.frame, text="Circle Detection")
        circle_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Store references to labels for unit updates
        self.min_radius_label = ttk.Label(circle_frame, text="Min Radius (px):")
        self.min_radius_label.grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.min_circle_var, width=6).grid(row=0, column=1)
        
        self.max_radius_label = ttk.Label(circle_frame, text="Max Radius (px):")
        self.max_radius_label.grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.max_circle_var, width=6).grid(row=0, column=3)
        
        self.min_dist_label = ttk.Label(circle_frame, text="Min Distance (px):")
        self.min_dist_label.grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.min_dist_var, width=6).grid(row=1, column=1)
        
        ttk.Label(circle_frame, text="Param1:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.param1_var, width=6).grid(row=2, column=1)
        ttk.Label(circle_frame, text="Param2:").grid(row=2, column=2, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.param2_var, width=6).grid(row=2, column=3)
        
        self.default_diameter_label = ttk.Label(circle_frame, text="Default Diameter (px):")
        self.default_diameter_label.grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.default_diameter_var, width=6).grid(row=3, column=1)
        
        # Add diameter restriction checkbox
        self.restrict_diameter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            circle_frame,
            text="Use default diameter for all circles",
            variable=self.restrict_diameter_var,
            command=self._apply_diameter_restriction,
        ).grid(row=4, column=0, columnspan=4, sticky=tk.W)
        
        # Recalculate when default diameter value changes
        self.default_diameter_var.trace_add("write", lambda *args: self._apply_diameter_restriction())
        # Update original_parameters when user manually changes detection parameters
        for name in ('rc_min_area', 'min_circle', 'max_circle', 'min_dist', 'default_diameter'):
            var = getattr(self, f'{name}_var')
            var.trace_add("write", lambda *args, n=name, v=var: self._on_param_change(n, v))

    # ---------------------------------------------------------------
    # Metadata handling functionality
    # ---------------------------------------------------------------
    
    def _get_image_metadata(self):
        """Get all metadata from the current image using multiple methods."""
        all_metadata = {}
        
        try:
            # Method 1: Try to get metadata from image processor
            if hasattr(self.image_processor, 'get_all_metadata'):
                processor_metadata = self.image_processor.get_all_metadata()
                if processor_metadata:
                    all_metadata.update(processor_metadata)
            
            # Method 2: Get image path
            img_path = None
            if hasattr(self.image_processor, 'original_image_path'):
                img_path = self.image_processor.original_image_path
            elif hasattr(self.image_processor, 'image_path'):
                img_path = self.image_processor.image_path
            elif hasattr(self.image_processor, 'current_file_path'):
                img_path = self.image_processor.current_file_path
            
            if img_path and os.path.exists(img_path):
                # Method 3: PIL/Pillow - EXIF and basic info
                try:
                    from PIL import Image
                    from PIL.ExifTags import TAGS, GPSTAGS
                    
                    with Image.open(img_path) as img:
                        # Get EXIF data
                        exifdata = img.getexif()
                        for tag_id, value in exifdata.items():
                            tag = TAGS.get(tag_id, f"Unknown_EXIF_{tag_id}")
                            all_metadata[f"EXIF_{tag}"] = value
                        
                        # Get image info (including DPI) - this is crucial for DPI detection
                        if hasattr(img, 'info') and img.info:
                            for key, value in img.info.items():
                                all_metadata[f"PIL_{key}"] = value
                        
                        # Also try to get DPI directly from PIL
                        try:
                            if hasattr(img, 'info') and 'dpi' in img.info:
                                all_metadata['PIL_dpi_direct'] = img.info['dpi']
                        except:
                            pass
                        
                        # Get format-specific info
                        if hasattr(img, 'format'):
                            all_metadata['Format'] = img.format
                        if hasattr(img, 'mode'):
                            all_metadata['Mode'] = img.mode
                        if hasattr(img, 'size'):
                            all_metadata['Size'] = img.size
                
                except Exception as e:
                    print(f"PIL metadata extraction failed: {e}")
                
                # Method 4: Try exifread library (if available)
                try:
                    import exifread
                    with open(img_path, 'rb') as f:
                        tags = exifread.process_file(f, details=True)
                        for tag, value in tags.items():
                            if tag not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']:
                                all_metadata[f"ExifRead_{tag}"] = str(value)
                except ImportError:
                    # Try to install exifread
                    all_metadata['_info_exifread'] = "ExifRead not installed - may provide more metadata"
                except Exception as e:
                    print(f"ExifRead metadata extraction failed: {e}")
                
                # Method 5: Try to read Windows properties using subprocess (Windows only)
                try:
                    import platform
                    import subprocess
                    import json
                    
                    if platform.system() == "Windows":
                        # Use PowerShell to get file properties
                        ps_command = f'''
                        $file = Get-Item "{img_path}"
                        $props = @{{}}
                        $file.PSObject.Properties | ForEach-Object {{
                            if ($_.Value -ne $null) {{
                                $props[$_.Name] = $_.Value.ToString()
                            }}
                        }}
                        $props | ConvertTo-Json
                        '''
                        
                        result = subprocess.run(['powershell', '-Command', ps_command], 
                                              capture_output=True, text=True, timeout=10)
                        if result.returncode == 0 and result.stdout.strip():
                            try:
                                windows_props = json.loads(result.stdout)
                                for key, value in windows_props.items():
                                    all_metadata[f"Windows_{key}"] = value
                            except json.JSONDecodeError:
                                pass
                except Exception as e:
                    print(f"Windows properties extraction failed: {e}")
                
                # Method 6: Manual EXIF parsing for common resolution tags
                try:
                    with open(img_path, 'rb') as f:
                        # Try to find resolution info manually in common EXIF tags
                        content = f.read(8192)  # Read first 8KB
                        
                        # Look for common DPI/resolution patterns
                        import re
                        dpi_patterns = [
                            rb'(\d+)\s*dpi',
                            rb'(\d+)\s*dots per inch',
                            rb'resolution[^\d]*(\d+)',
                        ]
                        
                        for pattern in dpi_patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            if matches:
                                all_metadata['Manual_DPI_Detection'] = [int(m) for m in matches if m.isdigit()]
                except Exception as e:
                    print(f"Manual EXIF parsing failed: {e}")
            
            # Add debug information
            all_metadata['_debug_total_keys'] = len(all_metadata)
            all_metadata['_debug_image_path'] = img_path if img_path else "No path found"
            
            return all_metadata
                        
        except Exception as e:
            print(f"Error getting metadata: {e}")
            return {}
    
    def _extract_resolution_from_metadata(self, metadata_key, metadata_value):
        """Extract resolution/DPI value from a metadata entry."""
        try:
            # Handle different metadata formats
            if isinstance(metadata_value, (tuple, list)) and len(metadata_value) >= 1:
                # DPI as tuple (x_dpi, y_dpi) or list
                if len(metadata_value) >= 2:
                    return float(metadata_value[0])  # Use X resolution
                else:
                    return float(metadata_value[0])
            elif isinstance(metadata_value, (int, float)):
                # Direct DPI value
                return float(metadata_value)
            elif isinstance(metadata_value, str):
                # Try to parse string representations
                import re
                
                # Remove common units and clean the string
                cleaned = metadata_value.replace(',', '.').lower()
                
                # Look for common DPI/resolution patterns
                patterns = [
                    r'(\d+\.?\d*)\s*dpi',
                    r'(\d+\.?\d*)\s*dots per inch',
                    r'(\d+\.?\d*)\s*pixels per inch',
                    r'(\d+\.?\d*)\s*ppi',
                    r'(\d+\.?\d*)\s*/\s*1',  # Rational format like "300/1"
                    r'(\d+\.?\d*)\s*x\s*\d+',  # Format like "300 x 300"
                    r'(\d+\.?\d*)',  # Just extract first number
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, cleaned)
                    if matches:
                        value = float(matches[0])
                        # Reasonable DPI range check
                        if 10 <= value <= 10000:
                            return value
                
                # Special handling for fractions (like "300/1")
                if '/' in cleaned:
                    parts = cleaned.split('/')
                    if len(parts) == 2:
                        try:
                            numerator = float(parts[0].strip())
                            denominator = float(parts[1].strip())
                            if denominator != 0:
                                value = numerator / denominator
                                if 10 <= value <= 10000:
                                    return value
                        except ValueError:
                            pass
            
            # Handle manual DPI detection results
            if metadata_key == 'Manual_DPI_Detection' and isinstance(metadata_value, list):
                # Return the first reasonable DPI value
                for val in metadata_value:
                    if 10 <= val <= 10000:
                        return float(val)
            
            return None
        except (ValueError, TypeError, IndexError, ZeroDivisionError):
            return None

    def _manual_metadata_selection(self):
        """Open dialog for manual metadata selection."""
        if not self.image_processor.has_image():
            messagebox.showwarning("Metadata", "Load an image first.")
            return
        
        # Get all metadata from the current image
        metadata = self._get_image_metadata()
        
        if not metadata or len(metadata) <= 2:  # Only debug keys
            # Show debug information and offer manual input
            debug_info = self._get_debug_info()
            result = messagebox.askyesno(
                "Metadata Not Found", 
                f"No metadata found in the current image.\n\n"
                f"Debug information:\n{debug_info}\n\n"
                f"Suggestions:\n"
                f"• Check if the image has EXIF metadata\n"
                f"• Try with a different image\n"
                f"• Some formats (PNG, BMP) may not have DPI in metadata\n\n"
                f"Do you want to enter the DPI manually?"

            )
            
            if result:
                self._manual_dpi_input()
            return
        
        # Create selection dialog
        self._show_metadata_selection_dialog(metadata)
    
    def _get_debug_info(self):
        """Get debug information about the current image."""
        debug_lines = []
        
        # Image path
        img_path = None
        if hasattr(self.image_processor, 'original_image_path'):
            img_path = self.image_processor.original_image_path
        elif hasattr(self.image_processor, 'image_path'):
            img_path = self.image_processor.image_path
        elif hasattr(self.image_processor, 'current_file_path'):
            img_path = self.image_processor.current_file_path
        
        debug_lines.append(f"Path: {img_path if img_path else 'Not available'}")
        
        if img_path and os.path.exists(img_path):
            debug_lines.append(f"Archivo existe: Sí")
            debug_lines.append(f"Tamaño: {os.path.getsize(img_path)} bytes")
            debug_lines.append(f"Extensión: {os.path.splitext(img_path)[1]}")
            
            # Try to get basic image info
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    debug_lines.append(f"Formato PIL: {img.format}")
                    debug_lines.append(f"Modo: {img.mode}")
                    debug_lines.append(f"Dimensiones: {img.size}")
                    debug_lines.append(f"Info keys: {list(img.info.keys()) if img.info else 'Ninguna'}")
            except Exception as e:
                debug_lines.append(f"Error PIL: {e}")
        else:
            debug_lines.append("Archivo no existe o no accesible")
        
        return "\n".join(debug_lines)

    
    def _manual_dpi_input(self):
        """Allow manual DPI input when metadata is not available."""
        # Crear diálogo personalizado para mantenerlo enfocado
        dialog = tk.Toplevel(self.frame)
        dialog.title("Enter DPI Manually")
        dialog.transient(self.frame)  # Mantener sobre la ventana principal
        dialog.grab_set()  # Hacer modal
        
        # Centrar diálogo
        dialog.geometry("400x300")
        dialog.resizable(False, False)
        
        # Título
        ttk.Label(dialog, text="Enter the image DPI (dots per inch):", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        # Texto informativo
        info_text = (
            "Common values:\n\n"
            "• 72 DPI - Standard screen\n"
            "• 96 DPI - Windows screen\n"
            "• 150 DPI - Basic printing\n"
            "• 300 DPI - High quality print\n"
            "• 600+ DPI - Professional print"
        )
        ttk.Label(dialog, text=info_text, justify=tk.LEFT).pack(pady=5)
        
        # Entrada de DPI
        dpi_var = tk.StringVar()
        entry_frame = ttk.Frame(dialog)
        entry_frame.pack(pady=10)
        ttk.Label(entry_frame, text="DPI:").pack(side=tk.LEFT, padx=5)
        dpi_entry = ttk.Entry(entry_frame, textvariable=dpi_var, width=10)
        dpi_entry.pack(side=tk.LEFT, padx=5)
        dpi_entry.focus_set()
        
        # Botones
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        def on_ok():
            try:
                dpi_input = float(dpi_var.get().replace(',', '.'))
                if not (10.0 <= dpi_input <= 10000.0):
                    raise ValueError("Value must be between 10 and 10000")
                
                # Store this as a manual selection for the session
                self.session_metadata_key = "_manual_dpi"
                self._manual_dpi_value = dpi_input
                
                # Enable metadata usage
                self.use_metadata_var.set(True)
                
                # Apply the conversion
                conversion_factor = 25.4 / dpi_input  # mm per pixel
                self._apply_conversion(conversion_factor)
                self._convert_parameters_to_mm(conversion_factor)
                
                # Actualizar etiquetas de resolución y conversión
                if hasattr(self, 'resolution_label'):
                    self.resolution_label.config(text=f"Resolution: {dpi_input:.3f} DPI")
                    self.conversion_label.config(text=f"({conversion_factor:.6f} mm/px)")
                
                dialog.destroy()
                
                messagebox.showinfo(
                    "Manual DPI Applied", 
                    f"Manually entered DPI: {dpi_input}\n"
                    f"Conversion factor: {conversion_factor:.6f} mm/pixel\n\n"
                    f"This setting will remain for the current session."
                )
                
            except ValueError as e:
                messagebox.showerror("Error", f"Valor de DPI no válido: {e}")
                dpi_entry.focus_set()
        
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Manejar tecla Enter
        def on_enter(event):
            on_ok()
        
        dpi_entry.bind('<Return>', on_enter)
        
        # Centrar diálogo
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def _convert_parameters_to_mm(self, conversion_factor):
        """Convert detection parameters from pixels to millimeters."""
        if self.parameters_converted:
            return  # Already converted
        
        # Store original values only once
        self.original_parameters = {
            'rc_min_area': self.rc_min_area_var.get(),
            'min_circle': self.min_circle_var.get(),
            'max_circle': self.max_circle_var.get(),
            'min_dist': self.min_dist_var.get(),
            'default_diameter': self.default_diameter_var.get()
        }
        
        # Convert area (pixels² to mm²)
        original_area = self.original_parameters['rc_min_area']
        converted_area = original_area * (conversion_factor ** 2)
        self.rc_min_area_var.set(round(converted_area, 2))
        
        # Convert linear measurements (pixels to mm), manteniendo decimales
        self.min_circle_var.set(round(self.original_parameters['min_circle'] * conversion_factor, 2))
        self.max_circle_var.set(round(self.original_parameters['max_circle'] * conversion_factor, 2))
        self.min_dist_var.set(round(self.original_parameters['min_dist'] * conversion_factor, 2))
        self.default_diameter_var.set(round(self.original_parameters['default_diameter'] * conversion_factor, 2))
        
        self.parameters_converted = True
        
        # Update UI labels to show units
        self._update_parameter_labels(True)
    
    def _restore_original_parameters(self):
        """Restore original parameter values in pixels."""
        if not self.parameters_converted or not self.original_parameters:
            return
        
        # Restore original values
        self.rc_min_area_var.set(self.original_parameters['rc_min_area'])
        self.min_circle_var.set(self.original_parameters['min_circle'])
        self.max_circle_var.set(self.original_parameters['max_circle'])
        self.min_dist_var.set(self.original_parameters['min_dist'])
        self.default_diameter_var.set(self.original_parameters['default_diameter'])
        
        self.parameters_converted = False
        
        # Update UI labels to show original units
        self._update_parameter_labels(False)
    
    def _update_parameter_labels(self, show_mm):
        """Update parameter labels to show current units."""
        if show_mm:
            # Update labels to show mm units
            self.area_label.config(text="Min Area (mm²):")
            self.min_radius_label.config(text="Min Radius (mm):")
            self.max_radius_label.config(text="Max Radius (mm):")
            self.min_dist_label.config(text="Min Distance (mm):")
            self.default_diameter_label.config(text="Default Diameter (mm):")
        else:
            # Update labels to show pixel units
            self.area_label.config(text="Min Area (px²):")
            self.min_radius_label.config(text="Min Radius (px):")
            self.max_radius_label.config(text="Max Radius (px):")
            self.min_dist_label.config(text="Min Distance (px):")
            self.default_diameter_label.config(text="Default Diameter (px):")
    


    def _on_param_change(self, name, var):
        """Update original_parameters dict when user changes a parameter."""
        try:
            val = float(var.get())
        except (ValueError, tk.TclError):
            return
        if not hasattr(self, 'original_parameters'):
            return
        if hasattr(self, 'updating_parameters_programmatically') and self.updating_parameters_programmatically:  # Ignore if updating programmatically
            return

        # If metadata is in use, convert the manually entered value (which is in mm) back to pixels
        if self.use_metadata_var.get() and hasattr(self, 'stored_resolution') and self.stored_resolution is not None:
            conversion_factor = 25.4 / self.stored_resolution
            if 'area' in name: # Area is factor squared
                self.original_parameters[name] = val / (conversion_factor ** 2)
            else:
                self.original_parameters[name] = val / conversion_factor
        else:
            # Otherwise, store the value directly (it's already in pixels)
            self.original_parameters[name] = val

    def _show_metadata_selection_dialog(self, metadata):
        """Show dialog for selecting metadata key for DPI/resolution."""
        dialog = tk.Toplevel(self.frame)
        dialog.title("Seleccionar Metadato de Resolución")
        dialog.geometry("600x400")
        dialog.resizable(True, True)
        dialog.grab_set()  # Make dialog modal
        
        # Center the dialog
        dialog.transient(self.frame.winfo_toplevel())
        
        # Instructions
        instruction_label = ttk.Label(
            dialog, 
            text="Selecciona el metadato que contiene la información de resolución (DPI):",
            font=("Arial", 10, "bold")
        )
        instruction_label.pack(pady=10, padx=10, anchor=tk.W)
        
        # Create treeview for metadata display
        metadata_frame = ttk.Frame(dialog)
        metadata_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview with scrollbars
        tree_frame = ttk.Frame(metadata_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        metadata_tree = ttk.Treeview(tree_frame, columns=("value",), show="tree headings")
        metadata_tree.heading("#0", text="Metadata Key")
        metadata_tree.heading("value", text="Value")
        
        # Configure column widths
        metadata_tree.column("#0", width=200)
        metadata_tree.column("value", width=300)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=metadata_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=metadata_tree.xview)
        metadata_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        metadata_tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Populate metadata tree with prioritized order
        metadata_items = {}
        
        # First add likely resolution-related entries
        resolution_keywords = ['dpi', 'resolution', 'density', 'inch', 'pixel', 'ppi']
        priority_items = []
        regular_items = []
        
        for key, value in metadata.items():
            # Skip debug entries for the main list
            if key.startswith('_debug'):
                continue
                
            # Format value for display
            display_value = str(value)
            if len(display_value) > 100:
                display_value = display_value[:97] + "..."
            
            # Check if this looks like a resolution-related field
            key_lower = str(key).lower()
            is_resolution_related = any(keyword in key_lower for keyword in resolution_keywords)
            
            item_data = (key, value, display_value, is_resolution_related)
            
            if is_resolution_related:
                priority_items.append(item_data)
            else:
                regular_items.append(item_data)
        
        # Add priority items first (likely resolution-related)
        for key, value, display_value, _ in sorted(priority_items, key=lambda x: x[0]):
            item_id = metadata_tree.insert("", "end", text=f"⭐ {key}", values=(display_value,))
            metadata_items[item_id] = (key, value)
        
        # Add separator if we have both types
        if priority_items and regular_items:
            separator_id = metadata_tree.insert("", "end", text="─── Otros metadatos ───", values=("",))
            # Don't add to metadata_items so it can't be selected
        
        # Add regular items
        for key, value, display_value, _ in sorted(regular_items, key=lambda x: x[0]):
            item_id = metadata_tree.insert("", "end", text=str(key), values=(display_value,))
            metadata_items[item_id] = (key, value)
        
        # Add info items at the end
        info_items = [(k, v) for k, v in metadata.items() if k.startswith('_info')]
        if info_items:
            separator_id = metadata_tree.insert("", "end", text="─── Información adicional ───", values=("",))
            for key, value in info_items:
                display_value = str(value)
                if len(display_value) > 100:
                    display_value = display_value[:97] + "..."
                item_id = metadata_tree.insert("", "end", text=f"ℹ️ {key[6:]}", values=(display_value,))  # Remove '_info_' prefix
                # Don't add to metadata_items so it can't be selected
        
        # Selection tracking
        selected_key = None
        selected_value = None
        
        def on_tree_select(event):
            nonlocal selected_key, selected_value
            selection = metadata_tree.selection()
            if selection:
                item_id = selection[0]
                if item_id in metadata_items:
                    selected_key, selected_value = metadata_items[item_id]
        
        metadata_tree.bind("<<TreeviewSelect>>", on_tree_select)
        
        # Buttons frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def on_accept():
            if selected_key is None:
                messagebox.showwarning("Selección", "Por favor selecciona un metadato.")
                return
            
            # Try to extract resolution from selected metadata
            resolution = self._extract_resolution_from_metadata(selected_key, selected_value)
            
            if resolution is None or resolution <= 0:
                messagebox.showerror(
                    "Error", 
                    f"No se pudo extraer un valor de resolución válido del metadato '{selected_key}'.\n"
                    f"Valor: {selected_value}"
                )
                return
            
            # Store the selected metadata key for the session
            self.session_metadata_key = selected_key
            
            # Enable metadata usage
            self.use_metadata_var.set(True)
            
            # Apply the conversion
            conversion_factor = 25.4 / resolution  # mm per pixel DESDE dpi
            self._apply_conversion(conversion_factor)
            self._convert_parameters_to_mm(conversion_factor)
            # Update the resolution and conversion labels in the main UI
            if hasattr(self, 'resolution_label'):
                self.resolution_label.config(text=f"Resolution: {resolution:.3f} DPI")
                conversion_factor = 25.4 / resolution
                self.conversion_label.config(text=f"({conversion_factor:.6f} mm/px)")
            
            messagebox.showinfo(
                "Success", 
                f"Metadata '{selected_key}' selected successfully.\n"
                f"Resolution: {resolution:.2f} DPI\n"
                f"Conversion factor: {conversion_factor:.6f} mm/pixel\n\n"
                f"This selection will be kept for the current session."
            )
            
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        def on_manual_input():
            """Allow manual DPI input."""
            dialog.destroy()
            self._manual_dpi_input()
        
        ttk.Button(button_frame, text="OK", command=on_accept).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Enter DPI Manually", command=on_manual_input).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT)
        
        # Handle dialog close
        def on_dialog_close():
            dialog.destroy()
        
        dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)

    def _extract_date_from_metadata(self):
        """Extract date from image metadata or file modification date."""
        try:
            # First try to get date from EXIF metadata
            if hasattr(self.image_processor, "image_metadata") and self.image_processor.image_metadata:
                # Try common date metadata keys
                date_keys = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized', 'ModifyDate', 'CreateDate']
                
                for key in date_keys:
                    if key in self.image_processor.image_metadata:
                        date_value = self.image_processor.image_metadata[key]
                        # Try to parse date - expecting format like "2023:06:15 14:30:22"
                        try:
                            # Extract just the date part (YYYY:MM:DD)
                            date_part = date_value.split()[0]
                            # Convert from YYYY:MM:DD to YYYY-MM-DD
                            formatted_date = date_part.replace(':', '-')
                            logging.info(f"Date extracted from image metadata: {formatted_date}")
                            return formatted_date
                        except (ValueError, IndexError):
                            continue
        
            # If we couldn't get a date from metadata, try to get file modification date
            if hasattr(self.image_processor, "image_path") and self.image_processor.image_path:
                import os
                import datetime
                try:
                    # Get file modification time
                    mod_time = os.path.getmtime(self.image_processor.image_path)
                    # Convert to datetime and format
                    mod_date = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d")
                    logging.info(f"Using file modification date: {mod_date}")
                    return mod_date
                except Exception as e:
                    logging.error(f"Error getting file modification time: {e}")
            
            return None
        except Exception as e:
            logging.error(f"Error extracting date from metadata: {e}")
            return None
    
    def _get_resolution_from_metadata(self):
        """Get resolution from metadata, using session key if available."""
        try:
            # First try the session metadata key if available
            if self.session_metadata_key:
                # Handle manual DPI input
                if self.session_metadata_key == "_manual_dpi":
                    # For manual DPI, we need to store the value separately
                    if hasattr(self, '_manual_dpi_value'):
                        return self._manual_dpi_value
                    return None
                
                # Handle metadata selection
                metadata = self._get_image_metadata()
                if self.session_metadata_key in metadata:
                    resolution = self._extract_resolution_from_metadata(
                        self.session_metadata_key, 
                        metadata[self.session_metadata_key]
                    )
                    if resolution and resolution > 0:
                        return resolution
            
            # Auto-detect from common metadata fields
            metadata = self._get_image_metadata()
            
            # Priority order for automatic detection
            priority_fields = [
                'PIL_dpi',
                'PIL_dpi_direct', 
                'EXIF_XResolution',
                'EXIF_YResolution',
                'PIL_resolution',
                'ExifRead_Image XResolution',
                'ExifRead_Image YResolution'
            ]
            
            for field in priority_fields:
                if field in metadata:
                    resolution = self._extract_resolution_from_metadata(field, metadata[field])
                    if resolution and resolution > 0:
                        logging.info(f"Auto-detected resolution from {field}: {resolution} DPI")
                        return resolution
            
            # Fallback to default DPI method
            dpi = self.image_processor.get_dpi() if hasattr(self.image_processor, "get_dpi") else None
            if dpi and dpi > 0:
                return dpi
            
            return None
        except Exception as e:
            logging.error(f"Error getting resolution from metadata: {e}")
            return None

    # ---------------------------------------------------------------
    # Event handlers
    # ---------------------------------------------------------------
    
    def _on_tree_select(self, event):
        """Highlight selected film or circle on the image."""
        global _OVERLAY
        sel = self.tree.selection()
        if not sel:
            if _OVERLAY is not None and "highlight" in _OVERLAY:
                _OVERLAY.pop("highlight", None)
        else:
            item_id = sel[0]
            if _OVERLAY and item_id in _OVERLAY.get("item_to_shape", {}):
                _OVERLAY["highlight"] = _OVERLAY["item_to_shape"][item_id]
        # Refresh display
        self.main_window.update_image()

    def _on_delete_key(self, event):
        """Delete selected film/circle (Delete key)."""
        sel = self.tree.selection()
        if not sel:
            return
        for item_id in list(sel):
            self._delete_item_recursive(item_id)
        self.main_window.update_image()

    def _on_right_click(self, event):
        """Show 3D view on right-click."""
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return
        if item_id in _OVERLAY.get("item_to_shape", {}) and _OVERLAY["item_to_shape"][item_id][0] == "circle":
            _, (cx, cy, r) = _OVERLAY["item_to_shape"][item_id]
            self._show_circle_3d(cx, cy, r)

    def _on_edit_label(self, event):
        """Edit item label on double-click."""
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return
        x, y, width, height = self.tree.bbox(item_id, "#0")
        entry = tk.Entry(self.tree)
        entry.place(x=x, y=y, width=width, height=height)
        entry.insert(0, self.tree.item(item_id, "text"))
        entry.focus()

        old_text = self.tree.item(item_id, "text")
        item_type = _OVERLAY.get("item_to_shape", {}).get(item_id, (None,))[0]

        def save_edit(event=None):
            new_text = entry.get()
            # Update TreeView
            self.tree.item(item_id, text=new_text)
            # Update cached results so CSV export reflects the rename
            if item_type == "film":
                for rec in self.results:
                    if rec["film"] == old_text:
                        rec["film"] = new_text
                # Update CTR map if this film has a CTR
                if old_text in self.ctr_map:
                    self.ctr_map[new_text] = self.ctr_map.pop(old_text)
            elif item_type == "circle":
                parent_id = self.tree.parent(item_id)
                film_name = self.tree.item(parent_id, "text") if parent_id else None
                for rec in self.results:
                    if rec["film"] == film_name and rec["circle"] == old_text:
                        rec["circle"] = new_text
            entry.destroy()
            self._autosize_columns()
        
        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", save_edit)

    def _toggle_ctr_circle(self, event):
        """Toggle CTR status for circle with Ctrl+click."""
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return
        
        # Check if it's a circle
        shape_info = _OVERLAY.get("item_to_shape", {}).get(item_id)
        if not shape_info or shape_info[0] != "circle":
            return
        
        # Get film name from parent
        film_id = self.tree.parent(item_id)
        if not film_id:
            return
        film_name = self.tree.item(film_id, "text")
        
        self._toggle_ctr_for_item(item_id, film_name)

    # ---------------------------------------------------------------
    # CTR functionality
    # ---------------------------------------------------------------
    
    def _toggle_ctr_manual(self):
        """Toggle CTR status for selected circle manually."""
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("CTR", "Selecciona un círculo para marcar/desmarcar como CTR.")
            return
        
        item_id = sel[0]
        shape_info = _OVERLAY.get("item_to_shape", {}).get(item_id)
        if not shape_info or shape_info[0] != "circle":
            messagebox.showinfo("CTR", "Solo se pueden marcar círculos como CTR.")
            return
        
        film_id = self.tree.parent(item_id)
        if not film_id:
            messagebox.showwarning("CTR", "No se puede determinar la radiocrómica del círculo.")
            return
        
        film_name = self.tree.item(film_id, "text")
        self._toggle_ctr_for_item(item_id, film_name)

    def _toggle_ctr_for_item(self, item_id: str, film_name: str):
        """Toggle CTR status for a specific item."""
        global _OVERLAY
        
        # Check if this circle is already CTR
        if self.ctr_map.get(film_name) == item_id:
            # Remove CTR status
            self.ctr_map.pop(film_name, None)
            current_text = self.tree.item(item_id, "text")
            if "(CTR)" in current_text:
                new_text = current_text.replace(" (CTR)", "")
                self.tree.item(item_id, text=new_text)
            self.tree.item(item_id, tags=())  # Remove CTR tag
            # Restore original values for all circles in this film
            self._restore_original_measurements()
        else:
            # Remove existing CTR if any
            if film_name in self.ctr_map:
                old_ctr_id = self.ctr_map[film_name]
                old_text = self.tree.item(old_ctr_id, "text")
                if "(CTR)" in old_text:
                    self.tree.item(old_ctr_id, text=old_text.replace(" (CTR)", ""))
                self.tree.item(old_ctr_id, tags=())  # Remove CTR tag
            
            # Set new CTR
            self.ctr_map[film_name] = item_id
            current_text = self.tree.item(item_id, "text")
            if "(CTR)" not in current_text:
                self.tree.item(item_id, text=f"{current_text} (CTR)")
            self.tree.item(item_id, tags=("ctr",))  # Apply CTR tag
        
        # Update CTR subtraction (if any CTR remains)
        self._update_ctr_subtraction()
        self.main_window.update_image()

    def _detect_ctr_automatically(self, film_name: str, film_circles: list):
        """Automatically detect CTR circle based on dose threshold."""
        ctr_candidate = None
        min_dose = float('inf')
        
        for circle_data in film_circles:
            avg_val = circle_data.get("avg_val", float('inf'))
            if avg_val <= CTR_DOSE_THRESHOLD and avg_val < min_dose:
                min_dose = avg_val
                ctr_candidate = circle_data
        
        if ctr_candidate:
            circle_id = ctr_candidate["circle_id"]
            self.ctr_map[film_name] = circle_id
            current_text = self.tree.item(circle_id, "text")
            self.tree.item(circle_id, text=f"{current_text} (CTR)")
            self.tree.item(circle_id, tags=("ctr",))
            return True
        return False

    def _update_ctr_subtraction(self):
        """Apply or remove CTR subtraction based on checkbox state."""
        if not self.subtract_ctr_var.get():
            # Show original measurements
            self._restore_original_measurements()
        else:
            # Apply CTR subtraction
            self._apply_ctr_subtraction()

    def _restore_original_measurements(self):
        """Restore original measurements without CTR subtraction."""
        for item_id, orig_data in self.original_measurements.items():
            if self.tree.exists(item_id):
                self.tree.item(item_id, values=(
                    orig_data["dose"],
                    orig_data["unc"],
                    orig_data["avg"],
                    orig_data["avg_unc"]
                ))

    def _apply_ctr_subtraction(self):
        """Apply CTR subtraction to all circles in films with CTR."""
        for film_name, ctr_id in self.ctr_map.items():
            if not self.tree.exists(ctr_id):
                continue
            
            # Get CTR measurement data
            ctr_orig_data = self.original_measurements.get(ctr_id)
            if not ctr_orig_data:
                continue
            
            ctr_avg = self._clean_numeric_string(ctr_orig_data["avg"])
            ctr_unc = self._clean_numeric_string(ctr_orig_data["avg_unc"])
            if ctr_avg == 0.0 and ctr_unc == 0.0:
                continue
            
            # Find parent film
            film_id = self.tree.parent(ctr_id)
            if not film_id:
                continue
            
            # Update all circles in this film
            for circle_id in self.tree.get_children(film_id):
                if not self.tree.exists(circle_id):
                    continue
                
                orig_data = self.original_measurements.get(circle_id)
                if not orig_data:
                    continue
                
                orig_avg = self._clean_numeric_string(orig_data["avg"])
                orig_unc = self._clean_numeric_string(orig_data["avg_unc"])
                if orig_avg == 0.0 and orig_unc == 0.0:
                    continue
                
                if circle_id == ctr_id:
                    # CTR circle: set to 0 ± uncertainty
                    corrected_avg = 0.0
                    corrected_unc = ctr_unc
                else:
                    # Other circles: subtract CTR with error propagation
                    corrected_avg = max(0.0, orig_avg - ctr_avg)
                    corrected_unc = sqrt(orig_unc**2 + ctr_unc**2)
                
                # Format and update
                avg_str, unc_str = self._format_val_unc(corrected_avg, corrected_unc, 2)
                
                # Update TreeView (keep original dose and sigma columns, update avg columns)
                current_values = list(self.tree.item(circle_id, "values"))
                if len(current_values) >= 4:
                    current_values[2] = avg_str  # avg
                    current_values[3] = unc_str   # avg_unc
                    self.tree.item(circle_id, values=tuple(current_values))

    def _store_original_measurement(self, item_id: str, dose_str: str, unc_str: str, avg_str: str, avg_unc_str: str, dose_val: float, unc_val: float, avg_val: float, avg_unc_val: float):
        """Store original measurement data before any CTR corrections."""
        self.original_measurements[item_id] = {
            "dose": dose_str,
            "unc": unc_str,
            "avg": avg_str,
            "avg_unc": avg_unc_str,
            "dose_val": dose_val,
            "unc_val": unc_val,
            "avg_val": avg_val,
            "avg_unc_val": avg_unc_val
        }

    def _refresh_all_measurements(self):
        """Refresh all measurements using the current uncertainty estimation method."""
        if not self.image_processor.has_image() or not hasattr(self, 'tree'):
            return
        
        # Get all items from the tree
        items = self.tree.get_children()
        if not items:
            return
        
        # Re-measure all circles with the new uncertainty method
        for item in items:
            item_id = self.tree.item(item, "values")[0]  # Get the ID from first column
            
            # Find the shape info for this item
            if item_id in _OVERLAY.get("item_to_shape", {}):
                shape_info = _OVERLAY["item_to_shape"][item_id]
                if shape_info[0] == "circle":
                    cx, cy, r = shape_info[1:]
                    
                    # Re-measure with current settings
                    prev_size = self.image_processor.measurement_size
                    try:
                        self.image_processor.measurement_size = r
                        res = self.image_processor.measure_area(
                            cx * self.image_processor.zoom,
                            cy * self.image_processor.zoom
                        )
                        if res:
                            # Extract measurement results (always 6-tuple format)
                            dose, std, unc, rgb_mean, rgb_mean_std, pixel_count = res
                            
                            # Format and update the tree
                            if isinstance(dose, tuple):
                                dose_parts = []
                                unc_parts = []
                                std_parts = []
                                for v, s, u in zip(dose, std, unc):
                                    v_str, u_str = self._format_val_unc(v, u, 2)
                                    _, s_str = self._format_val_unc(v, s, 2)
                                    dose_parts.append(v_str)
                                    unc_parts.append(u_str)
                                    std_parts.append(s_str.replace("±", "").strip())
                                dose_str = ", ".join(dose_parts)
                                unc_str = ", ".join(unc_parts)
                                std_str = ", ".join(std_parts)
                                
                                # Use combined uncertainty from ImageProcessor
                                avg_val = float(rgb_mean)
                                avg_unc = float(rgb_mean_std)
                            else:
                                dose_str, unc_str = self._format_val_unc(dose, unc, 2)
                                _, std_str_full = self._format_val_unc(dose, std, 2)
                                std_str = std_str_full.replace("±", "").strip()
                                
                                # For single channel, use combined uncertainty values
                                avg_val = float(rgb_mean)
                                avg_unc = float(rgb_mean_std)
                            
                            # Format average values
                            avg_str, avg_unc_str = self._format_val_unc(avg_val, avg_unc, 2)
                            
                            # Update tree values
                            current_values = list(self.tree.item(item, "values"))
                            if len(current_values) >= 4:
                                current_values[0] = dose_str    # dose
                                current_values[1] = unc_str     # sigma (uncertainty)
                                current_values[2] = avg_str     # avg
                                current_values[3] = avg_unc_str # avg_unc
                                self.tree.item(item, values=current_values)
                                
                                # Update original measurements for CTR calculations
                                self._store_original_measurement(item, dose_str, unc_str, avg_str, avg_unc_str)
                                
                    finally:
                        self.image_processor.measurement_size = prev_size

    # ---------------------------------------------------------------
    # Detection functionality
    # ---------------------------------------------------------------
    
    def start_detection(self):
        """Start automatic detection of films and circles."""
        if not self.image_processor.has_image():
            messagebox.showwarning("AutoMeasurements", "Carga una imagen primero.")
            return

        # Get image for detection
        img_rgb = (
            self.image_processor.original_image
            if getattr(self.image_processor, "original_image", None) is not None
            else self.image_processor.current_image
        )
        if img_rgb is None:
            return
        
        # Ensure 3 channels
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        # Extract date from metadata
        self.metadata_date = self._extract_date_from_metadata()
        
        # Clear any previous entries if they're in gray (metadata display)
        for entry in [self.year_entry, self.month_entry, self.day_entry]:
            if entry.cget("foreground") == "gray":
                entry.delete(0, tk.END)
                entry.config(foreground="black", font=("Arial", 9))
        
        # Update metadata date display
        if self.metadata_date:
            # Update metadata label
            self.metadata_date_label.config(text=f"(Metadata: {self.metadata_date})")
            
            # If user hasn't entered anything, pre-fill entries with gray text
            if "-" in self.metadata_date and not any([self.year_var.get(), self.month_var.get(), self.day_var.get()]):
                try:
                    y, m, d = self.metadata_date.split("-")
                    
                    self.year_entry.delete(0, tk.END)
                    self.year_entry.insert(0, y)
                    self.year_entry.config(foreground="gray", font=("Arial", 9, "italic"))
                    
                    self.month_entry.delete(0, tk.END)
                    self.month_entry.insert(0, m)
                    self.month_entry.config(foreground="gray", font=("Arial", 9, "italic"))
                    
                    self.day_entry.delete(0, tk.END)
                    self.day_entry.insert(0, d)
                    self.day_entry.config(foreground="gray", font=("Arial", 9, "italic"))
                    
                    # Define the behavior when clicking on an entry with metadata
                    def on_entry_click(event, entry):
                        if entry.cget("foreground") == "gray":
                            entry.delete(0, tk.END)
                            entry.config(foreground="black", font=("Arial", 9))
                    
                    # Bind the focus event to each entry field
                    self.year_entry.bind("<FocusIn>", lambda e: on_entry_click(e, self.year_entry))
                    self.month_entry.bind("<FocusIn>", lambda e: on_entry_click(e, self.month_entry))
                    self.day_entry.bind("<FocusIn>", lambda e: on_entry_click(e, self.day_entry))
                except ValueError:
                    pass



        # Check if parameters are converted to mm and need conversion to pixels
        restore_params = False
        if self.use_metadata_var.get() and self.parameters_converted:
            # Get resolution
            resolution = self._get_resolution_from_metadata()
            if resolution is None or resolution <= 0:
                messagebox.showerror("Error", "Resolución no disponible para conversión de unidades.")
                return
            
            conversion_factor = 25.4 / resolution
            
            # Convert parameters from mm to pixels
            # Area is mm² to pixels²
            rc_min_area_px = self.rc_min_area_var.get() / (conversion_factor ** 2)
            # Linear parameters are mm to pixels
            min_circle_px = self.min_circle_var.get() / conversion_factor
            max_circle_px = self.max_circle_var.get() / conversion_factor
            min_dist_px = self.min_dist_var.get() / conversion_factor
            default_diameter_px = self.default_diameter_var.get() / conversion_factor
            
            # Save original values
            self.orig_rc_min_area = self.rc_min_area_var.get()
            self.orig_min_circle = self.min_circle_var.get()
            self.orig_max_circle = self.max_circle_var.get()
            self.orig_min_dist = self.min_dist_var.get()
            self.orig_default_diameter = self.default_diameter_var.get()
            
            # Set to pixel values
            self.rc_min_area_var.set(int(round(rc_min_area_px)))
            self.min_circle_var.set(int(round(min_circle_px)))
            self.max_circle_var.set(int(round(max_circle_px)))
            self.min_dist_var.set(int(round(min_dist_px)))
            self.default_diameter_var.set(int(round(default_diameter_px)))
            restore_params = True

        # 1. Detect films
        films = self._detect_films(gray)
        if not films:
            messagebox.showinfo("AutoMeasurements", "No se detectaron radiocromías.")
            if restore_params:
                self._restore_detection_params()
            return

        # Clear previous results
        self.tree.delete(*self.tree.get_children())
        self.results.clear()
        self.original_measurements.clear()
        self.original_values.clear()  # Clear previous original values
        self.ctr_map.clear()
        
        global _OVERLAY
        _OVERLAY = {
            "films": films, 
            "circles": [], 
            "_shape": img_rgb.shape[:2], 
            "scale": 1.0, 
            "ctr_map": {}, 
            "item_to_shape": {}
        }

        # Process each film
        for idx, (x, y, w, h) in enumerate(films, 1):
            film_roi = gray[y : y + h, x : x + w]
            film_name = f"RC_{idx}"
            film_id = self.tree.insert("", "end", text=film_name, values=("", "", "", ""))
            _OVERLAY["item_to_shape"][film_id] = ("film", (x, y, w, h))

            # 2. Detect circles inside film
            film_circles = []
            detected_circles = self._detect_circles(film_roi)
            detected_circles = sorted(detected_circles, key=lambda c: (c[1], c[0]))

            # Store original radii
            original_circles = detected_circles.copy()

            # Apply diameter restriction if requested
            circles = detected_circles
            if self.restrict_diameter_var.get() and circles:
                default_radius = max(1, int(round(self.default_diameter_var.get() / 2)))
                circles = [(cx, cy, default_radius) for (cx, cy, _r) in detected_circles]

            # Process each circle
            for jdx, (cx, cy, adj_r) in enumerate(circles, 1):
                abs_cx = x + cx
                abs_cy = y + cy
                r_int = int(round(adj_r))
                orig_r_int = int(round(original_circles[jdx - 1][2]))

                # Measure dose
                prev_size = self.image_processor.measurement_size
                try:
                    self.image_processor.measurement_size = r_int
                    res = self.image_processor.measure_area(
                        abs_cx * self.image_processor.zoom,
                        abs_cy * self.image_processor.zoom
                    )
                finally:
                    self.image_processor.measurement_size = prev_size

                if res is None:
                    continue

                # Extract measurement results (always 6-tuple format)
                dose, std, unc, rgb_mean, rgb_mean_std, pixel_count = res

                # Format measurements
                if isinstance(dose, tuple):
                    dose_parts = []
                    unc_parts = []
                    std_parts = []
                    for v, s, u in zip(dose, std, unc):
                        v_str, u_str = self._format_val_unc(v, u, 2)
                        _, s_str = self._format_val_unc(v, s, 2)
                        dose_parts.append(v_str)
                        unc_parts.append(u_str)
                        std_parts.append(s_str.replace("±", "").strip())
                    dose_str = ", ".join(dose_parts)
                    unc_str = ", ".join(unc_parts)
                    std_str = ", ".join(std_parts)
                    
                    # Use combined uncertainty from ImageProcessor
                    avg_val = float(rgb_mean)
                    avg_unc = float(rgb_mean_std)
                    avg_std = float(np.mean(std))
                    avg_std_str = self._fmt_sig(avg_std, 2)
                else:
                    dose_str, unc_str = self._format_val_unc(dose, unc, 2)
                    _, std_str_full = self._format_val_unc(dose, std, 2)
                    std_str = std_str_full.replace("±", "").strip()
                    avg_std_str = std_str
                    
                    # For single channel, use combined uncertainty values
                    avg_val = float(rgb_mean)
                    avg_unc = float(rgb_mean_std)

                avg_str, avg_unc_str = self._format_val_unc(avg_val, avg_unc, 2)

                # Create circle item
                circ_name = f"C{jdx}"
                circ_id = self.tree.insert(film_id, "end", text=circ_name, 
                                         values=(dose_str, unc_str, avg_str, avg_unc_str))
                
                # Store mapping and original measurements
                _OVERLAY["item_to_shape"][circ_id] = ("circle", (abs_cx, abs_cy, r_int))
                self.original_radii[circ_id] = orig_r_int
                self._store_original_measurement(circ_id, dose_str, unc_str, avg_str, avg_unc_str, dose, unc, avg_val, avg_unc)
                
                # Add to overlay circles
                _OVERLAY["circles"].append((abs_cx, abs_cy, r_int))

                # Store for CTR detection and results
                film_circles.append({
                    "circle_id": circ_id,
                    "circ_name": circ_name,
                    "dose": dose,
                    "unc": unc,
                    "avg_val": avg_val,
                    "avg_unc": avg_unc,
                    "std": std,
                    "pixel_count": pixel_count,
                })

                # Store results
                self.results.append({
                    "film": film_name,
                    "circle": circ_name,
                    "dose": dose_str,
                    "unc": unc_str,
                    "avg": avg_str,
                    "avg_unc": avg_unc_str,
                    "std": avg_std_str,
                    "pixel_count": pixel_count,
                    "x": abs_cx,
                    "y": abs_cy,
                })

            # Automatically detect CTR circle
            self._detect_ctr_automatically(film_name, film_circles)

        # After detection, restore parameters if they were converted
        if restore_params:
            self._restore_detection_params()

        # Open all film nodes
        for child in self.tree.get_children():
            self.tree.item(child, open=True)

        # Apply CTR subtraction if enabled
        self._update_ctr_subtraction()

        # Update display
        self.main_window.update_image()
        self._autosize_columns()

    def _restore_detection_params(self):
        """Restore original detection parameters in mm."""
        if hasattr(self, 'orig_rc_min_area'):
            self.rc_min_area_var.set(self.orig_rc_min_area)
            self.min_circle_var.set(self.orig_min_circle)
            self.max_circle_var.set(self.orig_max_circle)
            self.min_dist_var.set(self.orig_min_dist)
            self.default_diameter_var.set(self.orig_default_diameter)
            # Clean up temporary attributes
            delattr(self, 'orig_rc_min_area')
            delattr(self, 'orig_min_circle')
            delattr(self, 'orig_max_circle')
            delattr(self, 'orig_min_dist')
            delattr(self, 'orig_default_diameter')
    
    # ---------------------------------------------------------------
    # Unit conversion functionality
    # ---------------------------------------------------------------

    def _update_unit_conversion(self):
        """Handle checkbox state change for unit conversion with debug prints."""
        logging.info("Unit Conversion Update initiated.")

        self.updating_parameters_programmatically = True  # Set flag to True

        # Disable checkbox if no image
        if not self.image_processor.has_image() and self.use_metadata_var.get():
            logging.warning("No image loaded - forcing checkbox OFF")
            messagebox.showwarning("No Image", "Load an image first.")
            self.use_metadata_var.set(False)
            return

        resolution = None
        if self.use_metadata_var.get():
            logging.info("Metadata usage enabled.")
            if not hasattr(self, 'stored_resolution') or self.stored_resolution is None:
                logging.info("No stored resolution found, attempting to detect new.")
                resolution = self._get_resolution_from_metadata()
                if resolution is None or resolution <= 0:
                    logging.error(f"Invalid resolution detected: {resolution}")
                    messagebox.showerror("Error", "Resolución no disponible para conversión de unidades.")
                    self.use_metadata_var.set(False)
                    return
                self.stored_resolution = resolution
                logging.info(f"New resolution stored: {resolution:.2f} DPI")
            else:
                logging.info(f"Using stored resolution: {self.stored_resolution:.2f} DPI")

            # Calculate conversion factor
            conversion_factor = 25.4 / self.stored_resolution
            logging.debug(f"Conversion factor: {conversion_factor:.6f} mm/px")

            # Always store current pixel values as original parameters before conversion
            self.original_parameters = {
                'rc_min_area': float(self.rc_min_area_var.get()),
                'min_circle': float(self.min_circle_var.get()),
                'max_circle': float(self.max_circle_var.get()),
                'min_dist': float(self.min_dist_var.get()),
                'default_diameter': float(self.default_diameter_var.get())
            }

            # Apply conversion to UI using exact values from original_parameters
            self.rc_min_area_var.set(self.original_parameters['rc_min_area'] * (conversion_factor ** 2))
            self.min_circle_var.set(self.original_parameters['min_circle'] * conversion_factor)
            self.max_circle_var.set(self.original_parameters['max_circle'] * conversion_factor)
            self.min_dist_var.set(self.original_parameters['min_dist'] * conversion_factor)
            self.default_diameter_var.set(self.original_parameters['default_diameter'] * conversion_factor)

            self.parameters_converted = True  # Set flag to indicate parameters are now in mm
        
            # Update labels
            self.resolution_label.config(text=f"Resolution: {self.stored_resolution:.3f} DPI")
            self.conversion_label.config(text=f"({conversion_factor:.6f} mm/px)")
            self._update_parameter_labels(True)

        else:
            logging.info("Metadata usage disabled.")
            # Convert current mm values back to pixels using stored resolution
            if hasattr(self, 'stored_resolution') and self.stored_resolution is not None:
                logging.debug(f"Using stored resolution: {self.stored_resolution:.2f} DPI")
                conversion_factor = 25.4 / self.stored_resolution

                # Use stored original parameters
                # Restore original parameters if they exist
                if hasattr(self, 'original_parameters') and self.original_parameters:
                    self.rc_min_area_var.set(self.original_parameters['rc_min_area'])
                    self.min_circle_var.set(self.original_parameters['min_circle'])
                    self.max_circle_var.set(self.original_parameters['max_circle'])
                    self.min_dist_var.set(self.original_parameters['min_dist'])
                    self.default_diameter_var.set(self.original_parameters['default_diameter'])
                    logging.info("Stored resolution cleared.")
                else:
                    logging.info("No stored parameters, cannot restore. UI remains as is.")

                # Always clear stored resolution and reset conversion flags
                self.stored_resolution = None
                self.parameters_converted = False
                self._update_parameter_labels(False) # Update labels to show pixel units

            # Restore original values display
            self._restore_original_values()
            self.resolution_label.config(text="Resolution: -")
            self.conversion_label.config(text="")

        self.updating_parameters_programmatically = False  # Reset flag to False
        logging.info("Unit Conversion Update complete.\n")
        logging.info("Unit Conversion Update complete.\n")

    def _apply_conversion(self, factor):
        """Convert displayed values to millimeters."""
        for film_id in self.tree.get_children(''):
            for circle_id in self.tree.get_children(film_id):
                if circle_id not in self.original_values:
                    continue
                orig_data = self.original_values[circle_id]
                # Convert values
                converted_dose = [v * factor for v in orig_data["dose"]]
                converted_sigma = [u * factor for u in orig_data["sigma"]]
                converted_avg = orig_data["avg"] * factor
                converted_avg_unc = orig_data["avg_unc"] * factor

                # Format
                if len(converted_dose) > 1:
                    dose_str_parts = []
                    sigma_str_parts = []
                    for v, u in zip(converted_dose, converted_sigma):
                        v_str, u_str = self._format_val_unc(v, u, 2, force_decimals=True)
                        dose_str_parts.append(v_str)
                        sigma_str_parts.append(u_str)
                    dose_str = ", ".join(dose_str_parts)
                    sigma_str = ", ".join(sigma_str_parts)
                else:
                    dose_str, sigma_str = self._format_val_unc(converted_dose[0], converted_sigma[0], 2, force_decimals=True)
                
                avg_str, avg_unc_str = self._format_val_unc(converted_avg, converted_avg_unc, 2, force_decimals=True)
                
                # Update TreeView
                self.tree.item(circle_id, values=(dose_str, sigma_str, avg_str, avg_unc_str))

    def _restore_original_values(self):
        """Restore original displayed values in pixels."""
        for film_id in self.tree.get_children(''):
            for circle_id in self.tree.get_children(film_id):
                if circle_id not in self.original_values:
                    continue
                orig_data = self.original_values[circle_id]
                # Re-format original values
                if len(orig_data["dose"]) > 1:
                    dose_str_parts = []
                    sigma_str_parts = []
                    for v, u in zip(orig_data["dose"], orig_data["sigma"]):
                        v_str, u_str = self._format_val_unc(v, u, 2, force_decimals=self.parameters_converted)
                        dose_str_parts.append(v_str)
                        sigma_str_parts.append(u_str)
                    dose_str = ", ".join(dose_str_parts)
                    sigma_str = ", ".join(sigma_str_parts)
                else:
                    dose_str, sigma_str = self._format_val_unc(orig_data["dose"][0], orig_data["sigma"][0], 2, force_decimals=self.parameters_converted)
                
                avg_str, avg_unc_str = self._format_val_unc(orig_data["avg"], orig_data["avg_unc"], 2, force_decimals=self.parameters_converted)
                
                # Update TreeView
                self.tree.item(circle_id, values=(dose_str, sigma_str, avg_str, avg_unc_str))

    # ---------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------

    def _fmt_sig(self, value: float, sig: int = 2) -> str:
        """Format number with the given significant figures."""
        if value == 0 or not np.isfinite(value):
            return f"{value}"
        return f"{value:.{sig}g}"

    def _clean_numeric_string(self, value_str: str) -> float:
        """Clean and parse numeric string that may contain ± symbols."""
        try:
            # Remove ± symbols and extra whitespace
            cleaned = str(value_str).replace("±", "").strip()
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0

    def _format_val_unc(self, value: float, unc: float, sig: int = 2, force_decimals: bool = False) -> tuple[str, str]:
        """Format value and uncertainty with consistent decimal places. Si force_decimals=True, siempre muestra 2 decimales."""
        unc_fmt = self._fmt_sig(unc, sig)

        # Scientific notation – fall back to significant-figure formatting
        if "e" in unc_fmt or "E" in unc_fmt:
            val_fmt = self._fmt_sig(value, sig)
        else:
            # Count decimals in uncertainty (0 if integer)
            if force_decimals:
                val_fmt = f"{value:.2f}"
            elif "." in unc_fmt:
                decimals = len(unc_fmt.split(".")[1])
                val_fmt = f"{value:.{decimals}f}"
            else:
                val_fmt = f"{value:.0f}"
        return val_fmt, f"±{unc_fmt}"

    def _ensure_uint8(self, img: np.ndarray) -> np.ndarray:
        """Convert any dtype image to uint8 0-255 for OpenCV edge detection."""
        if img.dtype == np.uint8:
            return img
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img_norm.astype(np.uint8)

    def _detect_films(self, gray) -> List[Tuple[int, int, int, int]]:
        """Detect rectangular films using fixed threshold."""
        gray8 = self._ensure_uint8(gray)

        thresh_val = self.rc_thresh_var.get()
        _, thresh = cv2.threshold(gray8, thresh_val, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.rc_min_area_var.get()
        films = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            films.append((x, y, w, h))

        films.sort(key=lambda b: b[1])
        return films

    def _detect_circles(self, roi) -> List[Tuple[int, int, int]]:
        """Detect circles via Hough transform."""
        min_r = self.min_circle_var.get()
        max_r = self.max_circle_var.get()
        roi8 = self._ensure_uint8(roi)
        roi_blur = cv2.medianBlur(roi8, 5)

        if min_r < 5:
            min_r = 100
        if max_r < min_r:
            max_r = min_r + 200

        param1_val = self.param1_var.get()
        param2_val = self.param2_var.get()
        min_dist_val = self.min_dist_var.get()
        
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
            circles = np.uint16(np.around(circles[0]))
            circles = circles.astype(int)
            result = [tuple(c) for c in circles]
        return result

    def _apply_diameter_restriction(self, *args):
        """Apply diameter restriction to detected circles."""
        global _OVERLAY
        if not _OVERLAY or "circles" not in _OVERLAY or not _OVERLAY["circles"]:
            return

        if self.restrict_diameter_var.get():
            # Apply default diameter restriction
            default_diameter = max(1, self.default_diameter_var.get())
            
            # Convert default diameter to pixels if in mm
            if self.use_metadata_var.get() and self.parameters_converted:
                resolution = self._get_resolution_from_metadata()
                if resolution is not None and resolution > 0:
                    conversion_factor = 25.4 / resolution
                    default_diameter = default_diameter / conversion_factor
            
            default_radius = max(1, int(round(default_diameter / 2)))  # integer radius
            
            new_circles = []
            for item_id, (stype, shape_data) in list(_OVERLAY.get("item_to_shape", {}).items()):
                if stype != "circle":
                    continue
                x_px, y_px, _ = shape_data
                new_circles.append((x_px, y_px, default_radius))
                
                # Update the stored shape data
                _OVERLAY["item_to_shape"][item_id] = ("circle", (x_px, y_px, default_radius))
                
                # Recalculate measurements
                prev_size = self.image_processor.measurement_size
                try:
                    self.image_processor.measurement_size = default_radius
                    res = self.image_processor.measure_area(
                        x_px * self.image_processor.zoom,
                        y_px * self.image_processor.zoom
                    )
                finally:
                    self.image_processor.measurement_size = prev_size
                
                if res is None:
                    continue
                
                # Update tree view with new measurements
                # Extract measurement results (always 6-tuple format)
                dose, std, unc, rgb_mean, rgb_mean_std, pixel_count = res
                
                if isinstance(dose, tuple):
                    dose_parts: list[str] = []
                    unc_parts: list[str] = []
                    std_parts: list[str] = []
                    for v, s, u in zip(dose, std, unc):
                        v_str, u_str = self._format_val_unc(v, u, 2, force_decimals=self.parameters_converted)
                        _, s_str = self._format_val_unc(v, s, 2)
                        dose_parts.append(v_str)
                        unc_parts.append(u_str)
                        std_parts.append(s_str.replace("±", "").strip())
                    dose_str = ", ".join(dose_parts)
                    unc_str = ", ".join(unc_parts)
                    
                    # Use combined uncertainty from ImageProcessor
                    avg_val = float(rgb_mean)
                    avg_unc = float(rgb_mean_std)
                    avg_std = float(np.mean(std))
                    avg_std_str = self._fmt_sig(avg_std, 2)
                else:
                    dose_str, unc_str = self._format_val_unc(dose, unc, 2, force_decimals=self.parameters_converted)
                    _, std_str_full = self._format_val_unc(dose, std, 2)
                    std_str = std_str_full.replace("±", "").strip()
                    avg_std_str = std_str
                    
                    # For single channel, use combined uncertainty values
                    avg_val = float(rgb_mean)
                    avg_unc = float(rgb_mean_std)
                
                avg_str, avg_unc_str = self._format_val_unc(avg_val, avg_unc, 2, force_decimals=self.parameters_converted)
                
                # Update original measurements
                self._store_original_measurement(item_id, dose_str, unc_str, avg_str, avg_unc_str)
                
                # Update TreeView values
                self.tree.item(item_id, values=(dose_str, unc_str, avg_str, avg_unc_str))
                
                # Update cached results for CSV
                circle_name = self.tree.item(item_id, "text").replace(" (CTR)", "")
                film_id = self.tree.parent(item_id)
                film_name = self.tree.item(film_id, "text") if film_id else None
                
                for rec in self.results:
                    if rec["film"] == film_name and rec["circle"] == circle_name:
                        rec["dose"] = dose_str
                        rec["unc"] = unc_str
                        rec["avg"] = avg_str
                        rec["avg_unc"] = avg_unc_str
                        rec["std"] = avg_std_str
                        rec["pixel_count"] = pixel_count
                        break

            _OVERLAY["circles"] = new_circles
        else:
            # Restore originally detected diameters
            new_circles = []
            for item_id, (stype, shape_data) in list(_OVERLAY.get("item_to_shape", {}).items()):
                if stype != "circle":
                    continue
                x, y, _ = shape_data
                original_r = int(self.original_radii.get(item_id, _))
                
                # Update stored shape to original radius
                _OVERLAY["item_to_shape"][item_id] = ("circle", (x, y, original_r))
                new_circles.append((x, y, original_r))
                
                # Recalculate measurements with original radii
                prev_size = self.image_processor.measurement_size
                try:
                    self.image_processor.measurement_size = original_r
                    res = self.image_processor.measure_area(
                        x * self.image_processor.zoom,
                        y * self.image_processor.zoom
                    )
                finally:
                    self.image_processor.measurement_size = prev_size
                
                if res is None:
                    continue
                
                # Update tree view with original measurements
                # Extract measurement results (always 6-tuple format)
                dose, std, unc, rgb_mean, rgb_mean_std, pixel_count = res
                
                if isinstance(dose, tuple):
                    dose_parts: list[str] = []
                    unc_parts: list[str] = []
                    std_parts: list[str] = []
                    for v, s, u in zip(dose, std, unc):
                        v_str, u_str = self._format_val_unc(v, u, 2, force_decimals=self.parameters_converted)
                        _, s_str = self._format_val_unc(v, s, 2)
                        dose_parts.append(v_str)
                        unc_parts.append(u_str)
                        std_parts.append(s_str.replace("±", "").strip())
                    dose_str = ", ".join(dose_parts)
                    unc_str = ", ".join(unc_parts)
                    
                    # Use combined uncertainty from ImageProcessor
                    avg_val = float(rgb_mean)
                    avg_unc = float(rgb_mean_std)
                    avg_std = float(np.mean(std))
                    avg_std_str = self._fmt_sig(avg_std, 2)
                else:
                    dose_str, unc_str = self._format_val_unc(dose, unc, 2, force_decimals=self.parameters_converted)
                    _, std_str_full = self._format_val_unc(dose, std, 2)
                    std_str = std_str_full.replace("±", "").strip()
                    avg_std_str = std_str
                    
                    # For single channel, use combined uncertainty values
                    avg_val = float(rgb_mean)
                    avg_unc = float(rgb_mean_std)
                    
                avg_str, avg_unc_str = self._format_val_unc(avg_val, avg_unc, 2, force_decimals=self.parameters_converted)
                
                # Update original measurements
                self._store_original_measurement(item_id, dose_str, unc_str, avg_str, avg_unc_str)
                
                # Update TreeView values
                self.tree.item(item_id, values=(dose_str, unc_str, avg_str, avg_unc_str))
                
                # Update cached results for CSV
                circle_name = self.tree.item(item_id, "text").replace(" (CTR)", "")
                film_id = self.tree.parent(item_id)
                film_name = self.tree.item(film_id, "text") if film_id else None
                
                for rec in self.results:
                    if rec["film"] == film_name and rec["circle"] == circle_name:
                        rec["dose"] = dose_str
                        rec["unc"] = unc_str
                        rec["avg"] = avg_str
                        rec["avg_unc"] = avg_unc_str
                        rec["std"] = avg_std_str
                        rec["pixel_count"] = pixel_count
                        break

            _OVERLAY["circles"] = new_circles

        # Apply CTR subtraction if enabled
        self._update_ctr_subtraction()
        self.main_window.update_image()
        self._autosize_columns()

    def _clear_all(self):
        """Clear all detections and data."""
        self.tree.delete(*self.tree.get_children())
        self.results.clear()
        self.original_measurements.clear()
        self.original_values.clear()  # Clear stored original values
        self.ctr_map.clear()
        
        # Reset parameter conversion state
        if self.parameters_converted:
            self._restore_original_parameters()
        
        self._clear_overlay()
        self.main_window.update_image()

    def _clear_overlay(self):
        """Clear overlay data."""
        global _OVERLAY
        _OVERLAY = {"films": [], "circles": [], "_shape": (0, 0), "scale": 1.0, "ctr_map": {}, "item_to_shape": {}}

    def _delete_item_recursive(self, item_id):
        """Recursively delete item and clean up data."""
        # First delete children recursively
        for child in self.tree.get_children(item_id):
            self._delete_item_recursive(child)

        # Remove from overlays and data structures
        if _OVERLAY and item_id in _OVERLAY.get("item_to_shape", {}):
            shape_type, coords = _OVERLAY["item_to_shape"].pop(item_id)
            
            if shape_type == "film":
                if coords in _OVERLAY.get("films", []):
                    _OVERLAY["films"].remove(coords)
                # Remove any CTR mapping for this film
                film_name = self.tree.item(item_id, "text")
                self.ctr_map.pop(film_name, None)
            elif shape_type == "circle":
                if coords in _OVERLAY.get("circles", []):
                    _OVERLAY["circles"].remove(coords)
                # Remove from CTR mapping if it was a CTR
                for film_name, ctr_id in list(self.ctr_map.items()):
                    if ctr_id == item_id:
                        self.ctr_map.pop(film_name)
                        break

        # Clean up stored data
        self.original_measurements.pop(item_id, None)
        self.original_values.pop(item_id, None)  # Remove from stored values
        self.original_radii.pop(item_id, None)

        # Remove from results
        if item_id in _OVERLAY.get("item_to_shape", {}):
            shape_type = _OVERLAY["item_to_shape"][item_id][0]
            if shape_type == "circle":
                circ_name = self.tree.item(item_id, "text").replace(" (CTR)", "")
                film_item = self.tree.parent(item_id)
                film_name = self.tree.item(film_item, "text") if film_item else None
                self.results = [rec for rec in self.results 
                              if not (rec["film"] == film_name and rec["circle"] == circ_name)]

        # Finally delete from TreeView
        self.tree.delete(item_id)

    def _show_circle_3d(self, cx: int, cy: int, r: int):
        """Show 3D visualization of circle area."""
        # Get image data
        if (
            self.image_processor.calibration_applied
            and getattr(self.image_processor, "dose_channels", None) is not None
        ):
            img = self.image_processor.dose_channels
            plot_title = "3D Dose Map"
            z_label = "Dose"
        else:
            img = (
                self.image_processor.original_image
                if getattr(self.image_processor, "original_image", None) is not None
                else self.image_processor.current_image
            )
            plot_title = "3D Intensity Map"
            z_label = "Value"
        
        if img is None:
            messagebox.showwarning("3D View", "No image loaded.")
            return

        # Crop ROI with margin
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(img.shape[1] - 1, cx + r)
        y2 = min(img.shape[0] - 1, cy + r)
        roi = img[y1 : y2 + 1, x1 : x2 + 1]

        # Prepare grid
        h, w = roi.shape[:2]
        X, Y = np.meshgrid(np.arange(w), np.arange(h))

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        if roi.ndim == 3 and roi.shape[2] >= 3:
            Z = np.mean(roi, axis=2)
        else:
            Z = roi.copy()

        # Clip outliers
        lo, hi = np.percentile(Z, [2, 98])
        Z = np.clip(Z, lo, hi)

        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
        ax.set_title(plot_title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel(z_label)

        plt.show()

    def _autosize_columns(self):
        """Adjust column widths to fit content."""
        font = tkfont.nametofont("TkDefaultFont")
        
        # For the name column, account for indentation
        max_name_width = 0
        for item in self.tree.get_children(''):
            text = self.tree.item(item, 'text')
            width = font.measure(text)
            if width > max_name_width:
                max_name_width = width
            
            for child in self.tree.get_children(item):
                child_text = self.tree.item(child, 'text')
                width = font.measure(child_text) + 30  # Add indentation
                if width > max_name_width:
                    max_name_width = width
        
        self.tree.column("#0", width=max_name_width + 20)
        
        # For other columns
        columns = self.tree["columns"]
        col_max_widths = {col: font.measure(self.tree.heading(col, "text")) for col in columns}
        
        all_items = self.tree.get_children('')
        for film in all_items:
            for col in columns:
                val = self.tree.set(film, col)
                if val:
                    width = font.measure(val)
                    if width > col_max_widths[col]:
                        col_max_widths[col] = width
            
            for circle in self.tree.get_children(film):
                for col in columns:
                    val = self.tree.set(circle, col)
                    if val:
                        width = font.measure(val)
                        if width > col_max_widths[col]:
                            col_max_widths[col] = width
        
        for col in columns:
            self.tree.column(col, width=col_max_widths[col] + 20)

    # ---------------------------------------------------------------
    # Manual addition and drawing
    # ---------------------------------------------------------------

    def _add_manual_film(self):
        """Start manual film addition mode."""
        self._start_draw_mode("film", (300, 200))

    def _add_manual_circle(self):
        """Start manual circle addition mode."""
        self._start_draw_mode("circle", 100)

    def _start_draw_mode(self, shape_type: str, dims):
        """Start interactive drawing mode."""
        if self._canvas is None or not self.image_processor.has_image():
            messagebox.showwarning("Draw", "No image loaded or canvas not available.")
            return

        self._cancel_draw()
        self.draw_mode = shape_type
        self.draw_dims = dims
        self._open_dims_window()
        
        self._canvas.config(cursor="crosshair")
        self._canvas.delete(self.preview_tag)
        self._canvas.bind("<Motion>", self._on_draw_move)
        self._canvas.bind("<Button-1>", self._on_draw_click)
        self.frame.winfo_toplevel().bind("<Escape>", self._cancel_draw)

    def _open_dims_window(self):
        """Open dimension editing window."""
        if self._dims_window is not None:
            self._dims_window.destroy()
        
        self._dims_window = tk.Toplevel(self.frame)
        self._dims_window.title("Dimensions")
        self._dims_window.resizable(False, False)
        self._dims_window.attributes("-topmost", True)
        self._dim_vars.clear()

        def on_var_change(*_):
            try:
                if self.draw_mode == "film" and len(self._dim_vars) == 2:
                    w = int(self._dim_vars[0].get())
                    h = int(self._dim_vars[1].get())
                    if w > 0 and h > 0:
                        self.draw_dims = (w, h)
                elif self.draw_mode == "circle" and self._dim_vars:
                    r = int(self._dim_vars[0].get())
                    if r > 0:
                        self.draw_dims = r
            except ValueError:
                pass
            self._update_preview()

        if self.draw_mode == "film":
            w, h = self.draw_dims
            w_var = tk.StringVar(value=str(w))
            h_var = tk.StringVar(value=str(h))
            self._dim_vars.extend([w_var, h_var])
            tk.Label(self._dims_window, text="Width:").grid(row=0, column=0, padx=4, pady=2)
            tk.Entry(self._dims_window, textvariable=w_var, width=6).grid(row=0, column=1, padx=4, pady=2)
            tk.Label(self._dims_window, text="Height:").grid(row=1, column=0, padx=4, pady=2)
            tk.Entry(self._dims_window, textvariable=h_var, width=6).grid(row=1, column=1, padx=4, pady=2)
        elif self.draw_mode == "circle":
            r = self.draw_dims
            r_var = tk.StringVar(value=str(r))
            self._dim_vars.append(r_var)
            tk.Label(self._dims_window, text="Radius:").grid(row=0, column=0, padx=4, pady=2)
            tk.Entry(self._dims_window, textvariable=r_var, width=6).grid(row=0, column=1, padx=4, pady=2)

        for var in self._dim_vars:
            var.trace_add("write", on_var_change)

    def _update_preview(self):
        """Update drawing preview."""
        if self.draw_mode is None or self._last_cursor is None:
            return
        dummy_evt = type("_e", (), {"x": self._last_cursor[0], "y": self._last_cursor[1]})()
        self._on_draw_move(dummy_evt)

    def _on_draw_move(self, event):
        """Update preview while moving cursor."""
        if self.draw_mode is None:
            return

        canvas = self._canvas
        canvas.delete(self.preview_tag)
        
        x = canvas.canvasx(event.x)
        y = canvas.canvasy(event.y)
        zoom = self.image_processor.zoom or 1.0
        
        self._last_cursor = (event.x, event.y)

        if self.draw_mode == "film":
            w, h = self.draw_dims
            dx = (w * zoom) / 2
            dy = (h * zoom) / 2
            canvas.create_rectangle(
                x - dx, y - dy, x + dx, y + dy,
                outline="yellow", dash=(4, 2), tags=self.preview_tag,
            )
        elif self.draw_mode == "circle":
            r = self.draw_dims
            dr = r * zoom
            canvas.create_oval(
                x - dr, y - dr, x + dr, y + dr,
                outline="yellow", dash=(4, 2), tags=self.preview_tag,
            )

    def _on_draw_click(self, event):
        """Finalize shape placement on click."""
        if self.draw_mode is None:
            return

        canvas = self._canvas
        x_canvas = canvas.canvasx(event.x)
        y_canvas = canvas.canvasy(event.y)
        zoom = self.image_processor.zoom or 1.0

        if self.draw_mode == "film":
            w, h = self.draw_dims
            x_top = int(x_canvas / zoom - w / 2)
            y_top = int(y_canvas / zoom - h / 2)
            self._insert_film(x_top, y_top, w, h)
        elif self.draw_mode == "circle":
            r = self.draw_dims
            cx = int(x_canvas / zoom)
            cy = int(y_canvas / zoom)
            self._insert_circle(cx, cy, r)

        self._cancel_draw()

    def _cancel_draw(self, event=None):
        """Cancel drawing mode."""
        if hasattr(self, "_canvas") and self._canvas is not None:
            self._canvas.delete(self.preview_tag)
            self._canvas.config(cursor="")
            self._canvas.unbind("<Motion>")
            self._canvas.unbind("<Button-1>")

        try:
            self.frame.winfo_toplevel().unbind("<Escape>")
        except Exception:
            pass

        if getattr(self, "_dims_window", None) is not None:
            self._dims_window.destroy()
            self._dims_window = None
        
        self._dim_vars.clear()
        self._last_cursor = None
        self.draw_mode = None
        self.draw_dims = None

    def _insert_film(self, x: int, y: int, w: int, h: int):
        """Insert a new film manually."""
        global _OVERLAY
        if _OVERLAY is None:
            _OVERLAY = {
                "films": [], "circles": [], "_shape": self.image_processor.current_image.shape[:2],
                "scale": 1.0, "ctr_map": {}, "item_to_shape": {}
            }

        film_idx = len([v for v in _OVERLAY["item_to_shape"].values() if v[0] == "film"]) + 1
        film_name = f"RC_{film_idx}M"
        film_id = self.tree.insert("", "end", text=film_name, values=("", "", "", ""))
        _OVERLAY["item_to_shape"][film_id] = ("film", (x, y, w, h))
        _OVERLAY["films"].append((x, y, w, h))
        self.tree.item(film_id, open=True)
        self.main_window.update_image()

    def _insert_circle(self, cx: int, cy: int, r: int):
        """Insert a new circle manually."""
        global _OVERLAY
        if _OVERLAY is None:
            _OVERLAY = {
                "films": [], "circles": [], "_shape": self.image_processor.current_image.shape[:2],
                "scale": 1.0, "ctr_map": {}, "item_to_shape": {}
            }

        # Find parent film
        parent_id = ""
        for fid, (item_type, coords) in _OVERLAY["item_to_shape"].items():
            if item_type != "film":
                continue
            fx, fy, fw, fh = coords
            if (fx <= cx <= fx + fw) and (fy <= cy <= fy + fh):
                parent_id = fid
                break

        circ_idx = len([v for v in _OVERLAY["item_to_shape"].values() if v[0] == "circle"]) + 1
        circ_name = f"C{circ_idx}M"
        
        # Measure the circle
        prev_size = self.image_processor.measurement_size
        try:
            self.image_processor.measurement_size = r
            res = self.image_processor.measure_area(
                cx * self.image_processor.zoom,
                cy * self.image_processor.zoom
            )
        finally:
            self.image_processor.measurement_size = prev_size

        if res:
            # Extract measurement results (always 6-tuple format)
            dose, std, unc, rgb_mean, rgb_mean_std, pixel_count = res

            # Format measurements
            if isinstance(dose, tuple):
                dose_parts = []
                unc_parts = []
                std_parts = []
                for v, s, u in zip(dose, std, unc):
                    v_str, u_str = self._format_val_unc(v, u, 2, force_decimals=self.parameters_converted)
                    _, s_str = self._format_val_unc(v, s, 2)
                    dose_parts.append(v_str)
                    unc_parts.append(u_str)
                    std_parts.append(s_str.replace("±", "").strip())
                dose_str = ", ".join(dose_parts)
                unc_str = ", ".join(unc_parts)
                std_str = ", ".join(std_parts)
                
                # Use combined uncertainty from ImageProcessor
                avg_val = float(rgb_mean)
                avg_unc = float(rgb_mean_std)
                avg_std = float(np.mean(std))
                avg_std_str = self._fmt_sig(avg_std, 2)
            else:
                dose_str, unc_str = self._format_val_unc(dose, unc, 2, force_decimals=self.parameters_converted)
                _, std_str_full = self._format_val_unc(dose, std, 2)
                std_str = std_str_full.replace("±", "").strip()
                avg_std_str = std_str
                
                # For single channel, use combined uncertainty values
                avg_val = float(rgb_mean)
                avg_unc = float(rgb_mean_std)
            
            avg_str, avg_unc_str = self._format_val_unc(avg_val, avg_unc, 2, force_decimals=self.parameters_converted)
        else:
            dose_str = unc_str = avg_str = avg_unc_str = std_str = ""
            pixel_count = 0
            avg_val = avg_unc = 0.0

        circ_id = self.tree.insert(parent_id, "end", text=circ_name, 
                                 values=(dose_str, unc_str, avg_str, avg_unc_str))
        
        # Store mapping and original measurements
        _OVERLAY["item_to_shape"][circ_id] = ("circle", (cx, cy, r))
        self.original_radii[circ_id] = r
        self._store_original_measurement(circ_id, dose_str, unc_str, avg_str, avg_unc_str)
        
        # Add to overlay circles
        _OVERLAY["circles"].append((cx, cy, r))

        # Store for CTR detection and results
        film_name = self.tree.item(parent_id, "text") if parent_id else None
        self.results.append({
            "film": film_name,
            "circle": circ_name,
            "dose": dose_str,
            "unc": unc_str,
            "avg": avg_str,
            "avg_unc": avg_unc_str,
            "x": cx,
            "y": cy,
        })

        # Store original values for unit conversion
        if isinstance(dose, tuple):
            dose_values = list(dose)
            sigma_values = list(unc)
        else:
            dose_values = [dose]
            sigma_values = [unc]
        
        self.original_values[circ_id] = {
            "dose": dose_values,
            "sigma": sigma_values,
            "avg": avg_val,
            "avg_unc": avg_unc,
        }

        if parent_id:
            self.tree.item(parent_id, open=True)

        # Add to results
        self.results.append({
            "film": film_name,
            "circle": circ_name,
            "dose": dose_str,
            "unc": unc_str,
            "avg": avg_str,
            "avg_unc": avg_unc_str,
            "std": avg_std_str,
            "pixel_count": pixel_count,
            "x": cx,
            "y": cy,
        })

        self.main_window.update_image()

    # ---------------------------------------------------------------
    # Sorting functionality
    # ---------------------------------------------------------------

    def _sort_by_column(self, col):
        """Sort tree items by column."""
        if col == "#0":
            if self.name_sort_mode == "coords":
                self._sort_by_coordinates()
                self.name_sort_mode = "name"
            else:
                self._sort_alphabetically()
                self.name_sort_mode = "coords"
        else:
            self.sort_reverse = not self.sort_reverse if self.sort_column == col else False
            self.sort_column = col
            self._sort_by_value(col, self.sort_reverse)

    def _sort_by_coordinates(self):
        """Sort circles by coordinates within each film."""
        for film_id in self.tree.get_children():
            circles = list(self.tree.get_children(film_id))
            
            circle_data = []
            for circle_id in circles:
                shape_data = _OVERLAY.get("item_to_shape", {}).get(circle_id)
                if shape_data and shape_data[0] == "circle":
                    x, y, _ = shape_data[1]
                    circle_data.append((circle_id, x, y))
            
            circle_data.sort(key=lambda item: (item[2], item[1]))
            
            for index, (circle_id, _, _) in enumerate(circle_data):
                self.tree.move(circle_id, film_id, index)

    def _sort_alphabetically(self):
        """Sort items alphabetically."""
        films = list(self.tree.get_children())
        films.sort(key=lambda film_id: self.tree.item(film_id, "text"))
        for index, film_id in enumerate(films):
            self.tree.move(film_id, '', index)
            
            circles = list(self.tree.get_children(film_id))
            circles.sort(key=lambda circle_id: self.tree.item(circle_id, "text"))
            for c_index, circle_id in enumerate(circles):
                self.tree.move(circle_id, film_id, c_index)

    def _sort_by_value(self, col, reverse=False):
        """Sort circles by column value."""
        for film_id in self.tree.get_children():
            circles = list(self.tree.get_children(film_id))
            
            circle_data = []
            for circle_id in circles:
                value_str = self.tree.set(circle_id, col)
                value = self._clean_numeric_string(value_str) if value_str else 0.0
                circle_data.append((circle_id, value))
            
            circle_data.sort(key=lambda item: item[1], reverse=reverse)
            
            for index, (circle_id, _) in enumerate(circle_data):
                self.tree.move(circle_id, film_id, index)

    # ---------------------------------------------------------------
    # Export functionality
    # ---------------------------------------------------------------

    def export_csv(self):
        """Export measurements to CSV."""
        if not self.results:
            messagebox.showwarning("Export", "No data to export.")
            return

        # Ask for file location
        filename = filedialog.asksaveasfilename(
            title="Save CSV",
            filetypes=[("CSV files", "*.csv")],
            defaultextension=".csv"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                cols = [
                    "date", "film", "circle", 
                    "original_dose", "original_unc", "original_avg", "original_avg_unc",
<<<<<<< HEAD
                    "std", "pixel_count"]
=======
                    "avg_dose", "uncertainty" 
                ]

                # Determine if CTR columns are needed based on whether CTR subtraction is enabled
                # and if there are any CTR circles defined.
                ctr_columns_needed = self.subtract_ctr_var.get() and bool(self.ctr_map)
>>>>>>> 39d0fcf05271feaf89d15f87209e2112e4e11ce5
                
                # Add CTR columns to the list of columns if they are needed
                if ctr_columns_needed:
                    cols.extend(["ctr_corrected_avg", "ctr_corrected_avg_unc"])

                # Initialize the DictWriter with the correctly defined fieldnames
                writer = csv.DictWriter(
                    csvfile, 
                    fieldnames=cols
                )

                # Write header with proper names
                header_map = {
                    "date": "Date",
                    "film": "Film",
                    "circle": "Circle", 
                    "original_dose": "Original Dose",
                    "original_unc": "Original Uncertainty",
                    "original_avg": "Original Average",
                    "original_avg_unc": "Original Average Uncertainty",
<<<<<<< HEAD
                    "std": "STD",
                    "pixel_count": "Number of pixels averaged",
=======
                    "avg_dose": "Average Dose",
                    "uncertainty": "Uncertainty",
>>>>>>> 39d0fcf05271feaf89d15f87209e2112e4e11ce5
                    "ctr_corrected_avg": "CTR Corrected Average",
                    "ctr_corrected_avg_unc": "CTR Corrected Average Uncertainty"
                }
                writer.writerow({field: header_map.get(field, field) for field in cols})


                # Get the date to use for all rows
                date_to_use = self.date_var.get() or self.metadata_date or ""

                # Sort results by Film, then by Circle
                try:
                    sorted_results = sorted(
                        [r for r in self.results if isinstance(r, dict)],
                        key=lambda rec: (
                            str(rec.get("film", "")),
                            str(rec.get("circle", ""))
                        )
                    )
                except Exception as e:
                    logging.error(f"Error sorting results: {e}")
                    sorted_results = [r for r in self.results if isinstance(r, dict)]

                # Process each result
                for rec in sorted_results:
                    film_name = rec.get("film", "")
                    circle_name = rec.get("circle", "")

                    # Find the corresponding tree item for this circle
                    circle_display_name = circle_name
                    item_id = None
                    is_ctr = False

                    # Search through all circle items to find the matching one
                    for iid, (stype, coords) in (_OVERLAY.get("item_to_shape", {}) if _OVERLAY else {}).items():
                        if stype == "circle" and self.tree.exists(iid):
                            parent_id = self.tree.parent(iid)
                            if parent_id and self.tree.exists(parent_id):
                                parent_name = self.tree.item(parent_id, "text")
                                current_circle_name = self.tree.item(iid, "text").replace(" (CTR)", "")

                                # Match by film name and circle name
                                if parent_name == film_name and current_circle_name == circle_name:
                                    item_id = iid
                                    # Check if this is a CTR circle
                                    if film_name in self.ctr_map and self.ctr_map[film_name] == iid:
                                        is_ctr = True
                                        circle_display_name = f"{circle_name} (CTR)"
                                    break

                                        # Get original values
                    orig_data = self.original_measurements.get(item_id, {}) if item_id else {}

                    # Get original numeric values
                    original_dose_val = orig_data.get("dose_val", 0.0) # Ensure these are numeric, default to 0.0
                    original_unc_val = orig_data.get("unc_val", 0.0)
                    original_avg_val = orig_data.get("avg_val", 0.0)
                    original_avg_unc_val = orig_data.get("avg_unc_val", 0.0)

                                        # Initialize corrected_avg and corrected_unc to original values
                    corrected_avg = original_avg_val
                    corrected_unc = original_avg_unc_val
                    
<<<<<<< HEAD
                    # Get values safely
                    original_avg = orig_data.get("avg", rec.get("avg", ""))
                    original_avg_unc = orig_data.get("avg_unc", rec.get("avg_unc", ""))
                    original_dose = orig_data.get("dose", rec.get("dose", ""))
                    original_unc = orig_data.get("unc", rec.get("unc", ""))
                    std = rec.get("std", "")
                    pixel_count = rec.get("pixel_count", "")
                    
                    # Calculate CTR-corrected values
                    ctr_corrected_avg = ""
                    ctr_corrected_unc = ""
                    
=======
                    # Initialize ctr_avg and ctr_unc for the CTR circle itself
                    ctr_avg_val_for_calc = 0.0
                    ctr_unc_val_for_calc = 0.0

                    # Apply CTR correction if enabled and applicable
>>>>>>> 39d0fcf05271feaf89d15f87209e2112e4e11ce5
                    if film_name in self.ctr_map and item_id and self.subtract_ctr_var.get():
                        ctr_id = self.ctr_map[film_name]
                        ctr_orig_data = self.original_measurements.get(ctr_id, {})

                        if ctr_orig_data:
                            try:
                                ctr_avg_val_for_calc = ctr_orig_data.get("avg_val", 0.0)
                                ctr_unc_val_for_calc = ctr_orig_data.get("avg_unc_val", 0.0)

                                if item_id == ctr_id:
                                    # CTR circle: set to 0 ± uncertainty
                                    corrected_avg = 0.0
                                    corrected_unc = ctr_unc_val_for_calc
                                else:
                                    # Other circles: subtract CTR with error propagation
                                    corrected_avg = max(0.0, original_avg_val - ctr_avg_val_for_calc)
                                    corrected_unc = sqrt(original_avg_unc_val**2 + ctr_unc_val_for_calc**2)
                                
                            except Exception as e:
                                logging.error(f"Error calculating CTR-corrected values: {e}")
                                messagebox.showerror("CTR Calculation Error", f"Error calculating CTR-corrected values for {circle_display_name}: {e}")

                    # Determine final values for avg_dose and uncertainty columns
                    # These should be the CTR-corrected values if CTR was applied, otherwise original
                    final_avg_dose = corrected_avg
                    final_uncertainty = corrected_unc

                    # Create a dictionary for the row
                    row_data = {
                        "date": date_to_use,
                        "film": film_name if isinstance(film_name, str) else "",
                        "circle": circle_display_name,
<<<<<<< HEAD
                        "original_dose": original_dose,
                        "original_unc": original_unc,
                        "original_avg": original_avg,
                        "original_avg_unc": original_avg_unc,
                        "std": std,
                        "pixel_count": pixel_count
=======
                        "original_dose": original_dose_val, # Use numeric values
                        "original_unc": original_unc_val,   # Use numeric values
                        "original_avg": original_avg_val,   # Use numeric values
                        "original_avg_unc": original_avg_unc_val, # Use numeric values
                        "avg_dose": final_avg_dose, # Final average dose (numeric)
                        "uncertainty": final_uncertainty, # Final uncertainty (numeric)
>>>>>>> 39d0fcf05271feaf89d15f87209e2112e4e11ce5
                    }

                    # Add CTR columns if they were calculated (numeric values)
                    # Only add if CTR was actually applied to this specific circle
                    # and if the CTR values are not zero (meaning a subtraction actually occurred)
                    if self.subtract_ctr_var.get() and bool(self.ctr_map) and (ctr_avg_val_for_calc != 0.0 or ctr_unc_val_for_calc != 0.0):
                        row_data["ctr_corrected_avg"] = corrected_avg
                        row_data["ctr_corrected_avg_unc"] = corrected_unc

                    # Write the row
                    writer.writerow(row_data)

            # Show success message
            messagebox.showinfo("Export", f"CSV successfully exported to:\n{filename}")
        except Exception as exc:
            logging.error(f"Error exporting CSV: {exc}")
            messagebox.showerror("Export Error", f"Error exporting CSV:\n{str(exc)}")


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
