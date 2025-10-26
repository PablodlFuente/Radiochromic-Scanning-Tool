"""
Main UI tab for the AutoMeasurements plugin.

This module contains the AutoMeasurementsTab class which provides
the complete user interface for film and circle detection, measurement,
and export functionality.
"""

from __future__ import annotations

import csv
import os
import sys
import logging
import datetime
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import tkinter.font as tkfont
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import log10, floor, sqrt

# Try to import tkcalendar - fallback if not available
try:
    from tkcalendar import DateEntry
    TKCALENDAR_AVAILABLE = True
except ImportError:
    TKCALENDAR_AVAILABLE = False
    # Fallback: simple entry
    class DateEntry(ttk.Entry):
        def __init__(self, parent, **kwargs):
            super().__init__(parent)
        def get_date(self):
            return self.get()

from ..models import DetectionParams, Circle, Film, MeasurementResult
from ..models import CTR_DOSE_THRESHOLD, DASH_COUNT, Y_TOLERANCE_FACTOR
from ..core import (
    MeasurementFormatter,
    DetectionEngine,
    MetadataExtractor,
    CTRManager,
    FileDataManager,
    CSVExporter
)

# Access to parent module _OVERLAY - done lazily to avoid circular imports
class _ParentModuleProxy:
    """Proxy to access parent module attributes lazily."""
    def __init__(self):
        self._module = None
    
    def _get_module(self):
        if self._module is None:
            import sys
            self._module = sys.modules.get('auto_measurements') or sys.modules.get('custom_plugins.auto_measurements')
        return self._module
    
    def __getattr__(self, name):
        return getattr(self._get_module(), name)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            # Internal attributes
            object.__setattr__(self, name, value)
        else:
            setattr(self._get_module(), name, value)

_parent_module = _ParentModuleProxy()

# Import parent module to access _OVERLAY directly (avoids proxy issues)
def _get_parent_module():
    """Get parent module reference."""
    import sys
    return sys.modules.get('auto_measurements') or sys.modules.get('custom_plugins.auto_measurements')

def _get_overlay():
    """Get _OVERLAY from parent module."""
    return _get_parent_module()._OVERLAY

def _set_overlay(value):
    """Set _OVERLAY in parent module."""
    parent = _get_parent_module()
    parent._OVERLAY = value


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
        
        # Initialize helper classes
        self.formatter = MeasurementFormatter()
        self.detector = DetectionEngine()
        self.metadata_extractor = MetadataExtractor(image_processor)
        
        # Initialize file data manager early (will set UI controls after tree is created)
        self.file_manager = FileDataManager(None, image_processor, main_window)  # tree will be set later

        # UI setup
        self._setup_ui()
        
        # Initialize CTR manager (needs tree widget from _setup_ui)
        self.ctr_manager = CTRManager(self.tree, self.formatter)
        
        # Update file manager with tree widget reference
        self.file_manager.tree = self.tree
        
        # Initialize CSV exporter (needs tree, image_processor, and file_manager)
        self.csv_exporter = CSVExporter(self.tree, image_processor, self.file_manager)
        
        # Storage
        self.results = []  # List[Dict]
        self.original_radii = {}  # Store current radii for circle items (may be restricted)
        self.detected_radii = {}  # Store originally detected radii before any restriction
        # Store original values for unit conversion
        self.original_values = {}  # key: item_id, value: dict of numeric values
        
        # Sorting and naming mode
        self.name_sort_mode = "coords"  # 'coords' or 'name'
        
        # Session storage for manual metadata selection
        self.metadata_date = None  # Store date from metadata
        
        # Storage for original parameter values (for unit conversion)
        self.original_parameters = {}
        self.parameters_converted = False
        self.updating_parameters_programmatically = False
        
        # Multi-file support: file_list, current_file_index, and file_data are now properties
        # that delegate to file_manager (no need to initialize them here)
        
        # Drawing attributes
        self.draw_mode = None
        self.draw_dims = None
        self._dims_window = None
        self._dim_vars = []
        self._last_cursor = None
        self.preview_tag = "draw_preview"  # Unique tag for canvas preview items
        
        # Store original canvas bindings to restore after drawing mode
        self._original_bindings = {}

        # Add sorting state variables
        self.sort_column = None
        self.sort_reverse = False
    
    # Properties for backward compatibility (delegate to CTR manager)
    @property
    def ctr_map(self):
        """Access CTR map (backward compatibility)."""
        return self.ctr_manager.ctr_map
    
    @ctr_map.setter
    def ctr_map(self, value):
        """Set CTR map (backward compatibility)."""
        self.ctr_manager.ctr_map = value
    
    @property
    def original_measurements(self):
        """Access original measurements (backward compatibility)."""
        return self.ctr_manager.original_measurements
    
    @original_measurements.setter
    def original_measurements(self, value):
        """Set original measurements (backward compatibility)."""
        self.ctr_manager.original_measurements = value
    
    # Properties for backward compatibility (delegate to file manager)
    @property
    def file_list(self):
        """Access file list (backward compatibility)."""
        return self.file_manager.file_list
    
    @file_list.setter
    def file_list(self, value):
        """Set file list (backward compatibility)."""
        self.file_manager.file_list = value
    
    @property
    def current_file_index(self):
        """Access current file index (backward compatibility)."""
        return self.file_manager.current_file_index
    
    @current_file_index.setter
    def current_file_index(self, value):
        """Set current file index (backward compatibility)."""
        self.file_manager.current_file_index = value
    
    @property
    def file_data(self):
        """Access file data (backward compatibility)."""
        return self.file_manager.file_data
    
    @file_data.setter
    def file_data(self, value):
        """Set file data (backward compatibility)."""
        self.file_manager.file_data = value

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
        ttk.Label(self.frame, text="Auto Measurements ", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Main button frame
        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill=tk.X, padx=10)
        ttk.Button(btn_frame, text="Add Files", command=self._add_files).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Start Detection", command=self.start_detection).pack(side=tk.LEFT, padx=5)
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
        cols = ["dose", "sigma", "avg", "avg_unc", "ci95"]

        self.tree = ttk.Treeview(self.frame, columns=tuple(cols), show="tree headings")
        self.tree.heading("#0", text="Element")
        self.tree.heading("dose", text="Dose")
        self.tree.heading("sigma", text="STD")
        self.tree.heading("avg", text="Average")
        self.tree.heading("avg_unc", text="SE of Avg")
        self.tree.heading("ci95", text="95% CI")
        
        # Column widths
        self.tree.column("#0", width=120, anchor=tk.W)
        self.tree.column("dose", width=80, anchor=tk.CENTER)
        self.tree.column("sigma", width=100, anchor=tk.CENTER)
        self.tree.column("avg", width=80, anchor=tk.CENTER)
        self.tree.column("avg_unc", width=80, anchor=tk.CENTER)
        self.tree.column("ci95", width=100, anchor=tk.CENTER)
        
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))
        
        # Multi-file navigation controls
        nav_frame = ttk.Frame(self.frame)
        nav_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Navigation buttons and label
        self.prev_button = ttk.Button(nav_frame, text="<", command=self._prev_file, state="disabled")
        self.prev_button.pack(side=tk.LEFT)
        
        self.file_counter_label = ttk.Label(nav_frame, text="0/0")
        self.file_counter_label.pack(side=tk.LEFT, padx=10)
        
        self.next_button = ttk.Button(nav_frame, text=">", command=self._next_file, state="disabled")
        self.next_button.pack(side=tk.LEFT)
        
        # Current file name label
        self.current_file_label = ttk.Label(nav_frame, text="No files loaded", foreground="gray")
        self.current_file_label.pack(side=tk.LEFT, padx=20)
        
        # Set file manager UI controls (after navigation controls are created)
        self.file_manager.set_ui_controls(
            self.prev_button, 
            self.next_button, 
            self.file_counter_label, 
            self.current_file_label
        )
        
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
        self.tree.heading("ci95", command=lambda: self._sort_by_column("ci95"))

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
    # Metadata handling functionality (delegates to MetadataExtractor)
    # ---------------------------------------------------------------
    
    def _get_image_metadata(self):
        """Get all metadata (delegates to metadata extractor)."""
        return self.metadata_extractor.get_all_metadata()
    
    def _extract_resolution_from_metadata(self, metadata_key, metadata_value):
        """Extract resolution from metadata (delegates to metadata extractor)."""
        return self.metadata_extractor.extract_resolution(metadata_key, metadata_value)

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
        
        img_path = self.metadata_extractor._get_image_path()
        debug_lines.append(f"Path: {img_path if img_path else 'Not available'}")
        
        if img_path and os.path.exists(img_path):
            debug_lines.append(f"File exists: Yes")
            debug_lines.append(f"Size: {os.path.getsize(img_path)} bytes")
            debug_lines.append(f"Extension: {os.path.splitext(img_path)[1]}")
            
            # Try to get basic image info
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    debug_lines.append(f"PIL Format: {img.format}")
                    debug_lines.append(f"Mode: {img.mode}")
                    debug_lines.append(f"Dimensions: {img.size}")
                    debug_lines.append(f"Info keys: {list(img.info.keys()) if img.info else 'None'}")
            except Exception as e:
                debug_lines.append(f"PIL Error: {e}")
        else:
            debug_lines.append("File doesn't exist or not accessible")
        
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
                
                # Store in metadata extractor
                self.metadata_extractor.session_metadata_key = "_manual_dpi"
                self.metadata_extractor.manual_dpi_value = dpi_input
                
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
            
            # Store the selected metadata key in the metadata extractor
            self.metadata_extractor.session_metadata_key = selected_key
            
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
        """Extract date from metadata (delegates to metadata extractor)."""
        return self.metadata_extractor.extract_date()
    
    def _get_resolution_from_metadata(self):
        """Get resolution automatically (delegates to metadata extractor)."""
        return self.metadata_extractor.get_resolution_auto()

    # ---------------------------------------------------------------
    # Event handlers
    # ---------------------------------------------------------------
    
    def _on_tree_select(self, event):
        """Highlight selected film or circle on the image."""
        # Using _get_parent_module()._OVERLAY (package-level global)
        sel = self.tree.selection()
        if not sel:
            if _get_parent_module()._OVERLAY is not None and "highlight" in _get_parent_module()._OVERLAY:
                _get_parent_module()._OVERLAY.pop("highlight", None)
        else:
            item_id = sel[0]
            if _get_parent_module()._OVERLAY and item_id in _get_parent_module()._OVERLAY.get("item_to_shape", {}):
                _get_parent_module()._OVERLAY["highlight"] = _get_parent_module()._OVERLAY["item_to_shape"][item_id]
        # Refresh display
        self.main_window.update_image()

    def _on_delete_key(self, event):
        """Delete selected film/circle (Delete key)."""
        sel = self.tree.selection()
        if not sel:
            return
        for item_id in list(sel):
            self._delete_item_recursive(item_id)
        
        # IMPORTANT: Store current file data after deletion to persist changes
        self._store_current_file_data()
        
        self.main_window.update_image()

    def _on_right_click(self, event):
        """Show 3D view on right-click."""
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return
        if item_id in _get_parent_module()._OVERLAY.get("item_to_shape", {}) and _get_parent_module()._OVERLAY["item_to_shape"][item_id][0] == "circle":
            _, (cx, cy, r) = _get_parent_module()._OVERLAY["item_to_shape"][item_id]
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
        item_type = _get_parent_module()._OVERLAY.get("item_to_shape", {}).get(item_id, (None,))[0]

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
        shape_info = _get_parent_module()._OVERLAY.get("item_to_shape", {}).get(item_id)
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
        shape_info = _get_parent_module()._OVERLAY.get("item_to_shape", {}).get(item_id)
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
        """Toggle CTR status (delegates to CTR manager)."""
        # Using _get_parent_module()._OVERLAY (package-level global)
        self.ctr_manager.toggle_ctr_for_item(item_id, film_name, _get_parent_module()._OVERLAY, self._update_ctr_subtraction)
        self.main_window.update_image()

    def _detect_ctr_automatically(self, film_name: str, film_circles: list):
        """Automatically detect CTR circle (delegates to CTR manager)."""
        return self.ctr_manager.detect_ctr_automatically(film_name, film_circles)

    def _update_ctr_subtraction(self):
        """Apply or remove CTR subtraction based on checkbox state and CTR availability."""
        # CRITICAL: If no CTR is selected, ALWAYS restore original values
        # regardless of checkbox state. Checkbox only applies when CTR exists.
        has_ctr = len(self.ctr_manager.ctr_map) > 0
        
        if not has_ctr or not self.subtract_ctr_var.get():
            # Restore original if: 1) No CTR selected, OR 2) Checkbox unchecked
            self.ctr_manager.restore_original_measurements(self.results)
        else:
            # Apply CTR only if: 1) CTR selected, AND 2) Checkbox checked
            self.ctr_manager.apply_ctr_subtraction(self.results)

    def _restore_original_measurements(self):
        """Restore original measurements (delegates to CTR manager)."""
        self.ctr_manager.restore_original_measurements(self.results)

    def _apply_ctr_subtraction(self):
        """Apply CTR subtraction (delegates to CTR manager)."""
        self.ctr_manager.apply_ctr_subtraction(self.results)
    
    def _store_original_measurement(self, item_id: str, dose_str: str, std_str: str, avg_str: str, avg_unc_str: str):
        """Store original measurement (delegates to CTR manager)."""
        self.ctr_manager.store_original_measurement(item_id, dose_str, std_str, avg_str, avg_unc_str)

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
            if item_id in _get_parent_module()._OVERLAY.get("item_to_shape", {}):
                shape_info = _get_parent_module()._OVERLAY["item_to_shape"][item_id]
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
                            
                            # Get numeric values
                            if isinstance(dose, tuple):
                                dose_values = list(dose)
                                sigma_values = list(std)
                                avg_val = float(rgb_mean)
                                avg_unc = float(rgb_mean_std)
                            else:
                                dose_values = dose
                                sigma_values = std
                                avg_val = float(rgb_mean)
                                avg_unc = float(rgb_mean_std)
                            
                            # Format for TreeView
                            dose_str, std_str, avg_str, avg_unc_str, ci95_str = self.formatter.format_for_treeview(
                                dose_values,
                                sigma_values,
                                avg_val,
                                avg_unc,
                                sig=2
                            )
                            
                            # For CSV export, we need unc_str with full precision
                            if isinstance(unc, tuple):
                                unc_parts = [f"±{u}" for u in unc]
                                unc_str = ", ".join(unc_parts)
                            else:
                                unc_str = f"±{unc}"
                            
                            # Update tree values
                            current_values = list(self.tree.item(item, "values"))
                            if len(current_values) >= 5:
                                current_values[0] = dose_str    # dose
                                current_values[1] = std_str     # STD (standard deviation)
                                current_values[2] = avg_str     # avg
                                current_values[3] = avg_unc_str # avg_unc (SE)
                                current_values[4] = ci95_str    # 95% CI
                                self.tree.item(item, values=current_values)
                                
                                # Update original measurements for CTR calculations (use formatted values)
                                self._store_original_measurement(item, dose_str, std_str, avg_str, avg_unc_str)
                                
                    finally:
                        self.image_processor.measurement_size = prev_size

    # ---------------------------------------------------------------
    # Multi-file functionality
    # ---------------------------------------------------------------
    
    def _add_files(self):
        """Allow user to select multiple image files for analysis (delegates to FileDataManager)."""
        self.file_manager.add_files(
            self._store_current_file_data,
            self._update_treeview_from_data
        )
    
    def _store_current_file_data(self):
        """Store current measurements and data for the current file (delegates to FileDataManager)."""
        self.file_manager.store_current_data(
            lambda: self.results,
            lambda: self.ctr_map,
            lambda: self.original_measurements,
            lambda: self.original_radii,
            lambda: self.original_values
        )
    
    def _load_file_data(self, file_path):
        """Load data for a specific file (delegates to FileDataManager)."""
        # Clear existing CTR data before loading new file data
        self.ctr_manager.ctr_map.clear()
        self.ctr_manager.original_measurements.clear()
        
        self._ctr_map_by_name = self.file_manager.load_file_data(
            file_path,
            lambda v: setattr(self, 'results', v),
            lambda v: self.ctr_manager.ctr_map.update(v) if v else None,
            lambda v: self.ctr_manager.original_measurements.update(v) if v else None,
            lambda v: setattr(self, 'original_radii', v),
            lambda v: setattr(self, 'original_values', v)
        )
    
    def _load_file(self, file_path):
        """Load a specific image file and its associated data (delegates to FileDataManager)."""
        # FileDataManager needs to call _update_treeview_from_data which internally calls _load_file_data
        # So we pass the update function as callback
        self.file_manager.load_file(file_path, self._update_treeview_from_data)
    
    def _update_treeview_from_data(self):
        """Update TreeView display with current file's data."""
        # Using _get_parent_module()._OVERLAY (package-level global)
        
        # Load file data first (this populates self.results, ctr_map, etc.)
        current_file = self.file_manager.get_current_file()
        if current_file:
            self._load_file_data(current_file)
        
        # Clear TreeView
        self.tree.delete(*self.tree.get_children())
        
        # Get shapes_by_name and radii_by_name from FileDataManager
        shapes_by_name, radii_by_name = self.file_manager.get_shapes_and_radii_for_current_file()
        
        # Rebuild item_to_shape mapping with new TreeView IDs
        new_item_to_shape = {}
        
        # Rebuild TreeView from stored results
        films = {}
        
        # FIRST: Create films that exist in shapes_by_name but might not have circles yet
        # This handles manually added films without children
        for key, coords in shapes_by_name.items():
            if key[0] == 'film':  # key is ('film', film_name)
                film_name = key[1]
                if film_name not in films:
                    film_id = self.tree.insert("", "end", text=film_name, values=("", "", "", "", ""))
                    films[film_name] = film_id
                    new_item_to_shape[film_id] = ('film', coords)
                    logging.debug(f"[_update_treeview] Restored film {film_name} from shapes_by_name")
        
        # SECOND: Rebuild circles from results
        for result in self.results:
            film_name = result['film']
            if film_name not in films:
                film_id = self.tree.insert("", "end", text=film_name, values=("", "", "", "", ""))
                films[film_name] = film_id
                
                # Restore film shape mapping
                film_key = ('film', film_name)
                if film_key in shapes_by_name:
                    new_item_to_shape[film_id] = ('film', shapes_by_name[film_key])
            else:
                film_id = films[film_name]
            
            # Add circle
            circle_name = result['circle']
            
            # Get numeric values for formatting
            avg_unc_value = result.get('avg_unc', 0.0)
            avg_display = result.get('avg_numeric', result.get('avg', 0.0))
            
            # Convert to float if needed
            if not isinstance(avg_unc_value, (int, float)):
                avg_unc_value = 0.0
            if not isinstance(avg_display, (int, float)):
                try:
                    avg_display = float(avg_display)
                except (ValueError, TypeError):
                    avg_display = 0.0
            
            # Get numeric values for formatting (preserve full precision)
            dose_for_format = result.get('dose_numeric', result.get('dose', 0.0))
            std_for_format = result.get('std_numeric', result.get('std_per_channel', result.get('unc', 0.0)))
            
            # Format values for TreeView display
            dose_str, sigma_str, avg_str, avg_unc_str, ci95_str = self.formatter.format_for_treeview(
                dose_for_format,
                std_for_format,
                avg_display,
                avg_unc_value,
                sig=2
            )
            
            values = (dose_str, sigma_str, avg_str, avg_unc_str, ci95_str)
            circle_id = self.tree.insert(film_id, "end", text=circle_name, values=values)
            
            # Store original measurements with new TreeView ID
            # IMPORTANT: Store FORMATTED values (avg_str, avg_unc_str) not raw numeric values
            # This ensures restore_original_measurements displays correctly formatted values
            # The formatter has already been applied above, so use those formatted strings
            self._store_original_measurement(circle_id, 
                                            dose_str,      # Already formatted by format_for_treeview
                                            sigma_str,     # Already formatted by format_for_treeview
                                            avg_str,       # Already formatted by format_for_treeview  
                                            avg_unc_str)   # Already formatted by format_for_treeview
            
            # Restore circle shape mapping
            circle_key = ('circle', film_name, circle_name)
            if circle_key in shapes_by_name:
                new_item_to_shape[circle_id] = ('circle', shapes_by_name[circle_key])
            else:
                # Fallback: use coordinates from result
                cx, cy = result.get('x', 0), result.get('y', 0)
                # Try to find radius from original_radii by matching coordinates
                r = 20  # default
                for coords_key, radius in self.original_radii.items():
                    if isinstance(coords_key, tuple) and len(coords_key) >= 2:
                        if coords_key[0] == cx and coords_key[1] == cy:
                            r = radius
                            break
                new_item_to_shape[circle_id] = ('circle', (cx, cy, r))
        
        # Update _OVERLAY with new mapping
        if _get_parent_module()._OVERLAY:
            _get_parent_module()._OVERLAY['item_to_shape'] = new_item_to_shape
            # Rebuild films list from shapes
            new_films = []
            for film_id in films.values():
                if film_id in new_item_to_shape and new_item_to_shape[film_id][0] == 'film':
                    new_films.append(new_item_to_shape[film_id][1])
            _get_parent_module()._OVERLAY['films'] = new_films
            logging.info(f"[_update_treeview] Restored {len(new_films)} films to _OVERLAY")
            
        # Restore original_radii with new IDs using radii_by_name
        new_original_radii = {}
        new_circles = []  # Rebuild circles list with correct radii
        
        for film_name, film_id in films.items():
            for child_id in self.tree.get_children(film_id):
                circle_name = self.tree.item(child_id, 'text')
                # Remove " (CTR)" suffix for key lookup
                circle_name_clean = circle_name.replace(" (CTR)", "")
                key = (film_name, circle_name_clean)
                
                # Get radius from radii_by_name or from shape mapping
                r = None
                if key in radii_by_name:
                    r = radii_by_name[key]
                elif child_id in new_item_to_shape and new_item_to_shape[child_id][0] == 'circle':
                    # Get radius from the already created shape mapping
                    _, (cx, cy, r_from_shape) = new_item_to_shape[child_id]
                    r = r_from_shape
                
                # If we have radius information, store it
                if r is not None:
                    new_original_radii[child_id] = r
                    # Update shape mapping with correct radius
                    if child_id in new_item_to_shape and new_item_to_shape[child_id][0] == 'circle':
                        cx, cy, _ = new_item_to_shape[child_id][1]
                        new_item_to_shape[child_id] = ('circle', (cx, cy, r))
                        # Add ALL circles to the list with correct radius
                        new_circles.append((cx, cy, r))
                else:
                    # Fallback: if no radius info, try to get from shape mapping
                    if child_id in new_item_to_shape and new_item_to_shape[child_id][0] == 'circle':
                        cx, cy, r_fallback = new_item_to_shape[child_id][1]
                        new_circles.append((cx, cy, r_fallback))
        
        self.original_radii = new_original_radii
        
        # Update _OVERLAY circles
        if _get_parent_module()._OVERLAY:
            _get_parent_module()._OVERLAY['circles'] = new_circles
            logging.info(f"[_update_treeview] Restored {len(new_circles)} circles to _OVERLAY")
        
        # Rebuild original_values with new IDs from results
        new_original_values = {}
        for result in self.results:
            film_name = result['film']
            circle_name = result['circle']
            # Remove " (CTR)" suffix if present
            circle_name_clean = circle_name.replace(" (CTR)", "")
            
            if film_name in films:
                film_id = films[film_name]
                for child_id in self.tree.get_children(film_id):
                    child_name = self.tree.item(child_id, 'text').replace(" (CTR)", "")
                    if child_name == circle_name_clean:
                        # Reconstruct original_values from result data
                        dose_numeric = result.get('dose_numeric', result.get('dose', 0.0))
                        unc_numeric = result.get('unc_numeric', result.get('unc', 0.0))
                        avg_numeric = result.get('avg_numeric', result.get('avg_val', 0.0))
                        avg_unc_numeric = result.get('avg_unc_numeric', result.get('avg_unc', 0.0))
                        
                        # Convert to lists if they're tuples or single values
                        if isinstance(dose_numeric, (tuple, list)):
                            dose_values = list(dose_numeric)
                        else:
                            dose_values = [dose_numeric]
                        
                        if isinstance(unc_numeric, (tuple, list)):
                            sigma_values = list(unc_numeric)
                        else:
                            sigma_values = [unc_numeric]
                        
                        new_original_values[child_id] = {
                            "dose": dose_values,
                            "sigma": sigma_values,
                            "avg": avg_numeric if isinstance(avg_numeric, (int, float)) else 0.0,
                            "avg_unc": avg_unc_numeric if isinstance(avg_unc_numeric, (int, float)) else 0.0,
                        }
                        break
        
        self.original_values = new_original_values
        
        # Update _OVERLAY circles with correct radii
        if _get_parent_module()._OVERLAY:
            _get_parent_module()._OVERLAY['circles'] = new_circles
        
        # Rebuild ctr_map using names from _ctr_map_by_name
        if hasattr(self, '_ctr_map_by_name') and self._ctr_map_by_name:
            new_ctr_map = {}
            for film_name, circle_name_with_ctr in self._ctr_map_by_name.items():
                # Find the film and circle IDs
                if film_name in films:
                    film_id = films[film_name]
                    for child_id in self.tree.get_children(film_id):
                        child_name = self.tree.item(child_id, 'text')
                        # Remove " (CTR)" suffix if present for comparison
                        child_name_clean = child_name.replace(" (CTR)", "")
                        if child_name_clean == circle_name_with_ctr:
                            new_ctr_map[film_name] = child_id
                            # Mark as CTR in TreeView
                            if "(CTR)" not in child_name:
                                self.tree.item(child_id, text=f"{child_name_clean} (CTR)")
                            self.tree.item(child_id, tags=("ctr",))
                            logging.debug(f"[_update_treeview] Marked {child_name_clean} as CTR from _ctr_map_by_name")
                            break
            self.ctr_map = new_ctr_map
            # Clear temporary storage
            del self._ctr_map_by_name
        
        # FALLBACK: Restore CTR visual markers from _OVERLAY ctr_map
        # This handles manual shapes added without Start Detection
        if _get_parent_module()._OVERLAY and 'ctr_map' in _get_parent_module()._OVERLAY:
            overlay_ctr = _get_parent_module()._OVERLAY['ctr_map']
            logging.debug(f"[_update_treeview] Found CTR in _OVERLAY: {list(overlay_ctr.keys())}")
            
            # For each film that has CTR in overlay
            for film_name in overlay_ctr.keys():
                if film_name in films and film_name not in self.ctr_map:
                    film_id = films[film_name]
                    # Get children circles
                    children = list(self.tree.get_children(film_id))
                    if children:
                        # Mark first circle as CTR (simplest approach for manual case)
                        # In practice, there should only be one circle per manually added film
                        first_circle_id = children[0]
                        circle_name = self.tree.item(first_circle_id, 'text').replace(" (CTR)", "")
                        
                        # Update ctr_map and TreeView
                        self.ctr_map[film_name] = first_circle_id
                        self.tree.item(first_circle_id, text=f"{circle_name} (CTR)")
                        self.tree.item(first_circle_id, tags=("ctr",))
                        logging.info(f"[_update_treeview] Restored CTR marker for {circle_name} in {film_name} from _OVERLAY")
            
        # Open all film nodes
        for film_id in films.values():
            self.tree.item(film_id, open=True)
            
        # Apply CTR subtraction if enabled
        self._update_ctr_subtraction()
    
    def _prev_file(self):
        """Navigate to previous file (delegates to FileDataManager)."""
        self.file_manager.navigate_previous(
            self._store_current_file_data,
            self._update_treeview_from_data
        )
    
    def _next_file(self):
        """Navigate to next file (delegates to FileDataManager)."""
        self.file_manager.navigate_next(
            self._store_current_file_data,
            self._update_treeview_from_data
        )
    
    def _update_navigation_controls(self):
        """Update navigation button states and counter label (delegates to FileDataManager)."""
        self.file_manager.update_navigation_controls()

    # ---------------------------------------------------------------
    # Detection functionality
    # ---------------------------------------------------------------
    
    def start_detection(self):
        """Start automatic detection of films and circles."""
        if not self.image_processor.has_image():
            messagebox.showwarning("AutoMeasurements", "Carga una imagen primero.")
            return
        
        # Get parent module reference for _OVERLAY access
        import sys
        parent_module = sys.modules.get('auto_measurements') or sys.modules.get('custom_plugins.auto_measurements')

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

        # 1. Create detection parameters
        params = DetectionParams(
            rc_threshold=self.rc_thresh_var.get(),
            rc_min_area=self.rc_min_area_var.get(),
            min_circle_radius=self.min_circle_var.get(),
            max_circle_radius=self.max_circle_var.get(),
            min_distance=self.min_dist_var.get(),
            param1=self.param1_var.get(),
            param2=self.param2_var.get(),
            default_diameter=self.default_diameter_var.get(),
            restrict_diameter=self.restrict_diameter_var.get()
        )
        
        # Detect films using DetectionEngine
        films = self.detector.detect_films(gray, params)
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
        
        # Set _OVERLAY in parent module (direct access, not via proxy)
        parent_module._OVERLAY = {
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
            film_id = self.tree.insert("", "end", text=film_name, values=("", "", "", "", ""))
            parent_module._OVERLAY["item_to_shape"][film_id] = ("film", (x, y, w, h))

            # 2. Detect circles inside film using DetectionEngine
            film_circles = []
            detected_circles = self.detector.detect_circles(film_roi, params)
            detected_circles = sorted(detected_circles, key=lambda c: (c[1], c[0]))

            # Store original radii
            original_circles = detected_circles.copy()

            # Apply diameter restriction if requested
            circles = detected_circles
            if params.restrict_diameter and circles:
                default_radius = max(1, int(round(params.default_diameter / 2)))
                circles = [(cx, cy, default_radius) for (cx, cy, _r) in detected_circles]

            # Organize circles as matrix and assign names using DetectionEngine
            named_circles = self.detector.organize_circles_as_matrix(circles)
            
            # Create mapping from new index to original index for radius lookup
            circle_to_orig_idx = {}
            for new_idx, (name, (cx, cy, r)) in enumerate(named_circles):
                # Find matching circle in original list
                for orig_idx, (ocx, ocy, _) in enumerate(original_circles):
                    if cx == ocx and cy == ocy:
                        circle_to_orig_idx[new_idx] = orig_idx
                        break

            # Process each circle
            for jdx, (circ_name, (cx, cy, adj_r)) in enumerate(named_circles):
                abs_cx = x + cx
                abs_cy = y + cy
                r_int = int(round(adj_r))
                orig_idx = circle_to_orig_idx.get(jdx, jdx)
                orig_r_int = int(round(original_circles[orig_idx][2]))

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

                # Format measurements with full precision
                if isinstance(dose, tuple):
                    dose_parts = []
                    unc_parts = []
                    std_parts = []
                    for v, s, u in zip(dose, std, unc):
                        dose_parts.append(f"{v}")
                        unc_parts.append(f"±{u}")
                        std_parts.append(f"{s}")
                    dose_str = ", ".join(dose_parts)
                    unc_str = ", ".join(unc_parts)
                    std_str = ", ".join(std_parts)
                    
                    # Use combined uncertainty from ImageProcessor
                    avg_val = float(rgb_mean)
                    avg_unc = float(rgb_mean_std)
                    avg_std = float(np.mean(std))
                    avg_std_str = f"{avg_std}"
                else:
                    # For single channel, use combined uncertainty values
                    avg_val = float(rgb_mean)
                    avg_unc = float(rgb_mean_std)

                # Format for TreeView display using formatter
                dose_str, std_str, avg_str, avg_unc_str, ci95_str = self.formatter.format_for_treeview(
                    dose,
                    std,
                    avg_val,
                    avg_unc,
                    sig=2
                )

                # Create circle item (circ_name already comes from matrix organization)
                circ_id = self.tree.insert(film_id, "end", text=circ_name, 
                                         values=(dose_str, std_str, avg_str, avg_unc_str, ci95_str))
                
                # Store mapping and original measurements
                parent_module._OVERLAY["item_to_shape"][circ_id] = ("circle", (abs_cx, abs_cy, r_int))
                # Store the actual radius used (r_int), which may be restricted
                self.original_radii[circ_id] = r_int
                # Store the detected radius before restriction
                self.detected_radii[circ_id] = orig_r_int
                self._store_original_measurement(circ_id, dose_str, std_str, avg_str, avg_unc_str)
                
                # Add to overlay circles
                parent_module._OVERLAY["circles"].append((abs_cx, abs_cy, r_int))

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
                    "dose": dose,                        # Raw numeric dose (NOT formatted string)
                    "std_per_channel": std,              # Raw numeric STD per channel (NOT formatted string)
                    "unc": unc,                          # Raw numeric SE/Uncertainty per channel
                    "avg": avg_val,                      # Store numeric value (not formatted string)
                    "avg_unc": avg_unc,                  # Store numeric value directly
                    "std": avg_std,                      # Raw numeric average std
                    "pixel_count": pixel_count,
                    "x": abs_cx,
                    "y": abs_cy,
                    # Store raw numeric values for CSV export (redundant but kept for compatibility)
                    "dose_numeric": dose,                # Raw dose values (tuple or float)
                    "std_numeric": std,                  # Raw std values (tuple or float)
                    "unc_numeric": unc,                  # Raw uncertainty values (tuple or float)
                    "avg_numeric": avg_val,              # Raw average value (float)
                    "std_avg_numeric": avg_std,          # Raw average std (float)
                    "avg_unc_numeric": avg_unc,          # Raw SE/uncertainty value (float)
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
        
        # Mark current file as measured (for multi-file support)
        if self.file_list and 0 <= self.current_file_index < len(self.file_list):
            current_file = self.file_list[self.current_file_index]
            if current_file in self.file_data:
                self.file_data[current_file]['measured'] = True

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
                
                # Format for TreeView
                dose_str, sigma_str, avg_str, avg_unc_str, ci95_str = self.formatter.format_for_treeview(
                    converted_dose,
                    converted_sigma,
                    converted_avg,
                    converted_avg_unc,
                    sig=2
                )
                
                # Update TreeView
                self.tree.item(circle_id, values=(dose_str, sigma_str, avg_str, avg_unc_str, ci95_str))

    def _restore_original_values(self):
        """Restore original displayed values in pixels."""
        for film_id in self.tree.get_children(''):
            for circle_id in self.tree.get_children(film_id):
                if circle_id not in self.original_values:
                    continue
                orig_data = self.original_values[circle_id]
                
                # Format for TreeView
                dose_str, sigma_str, avg_str, avg_unc_str, ci95_str = self.formatter.format_for_treeview(
                    orig_data["dose"],
                    orig_data["sigma"],
                    orig_data["avg"],
                    orig_data["avg_unc"],
                    sig=2
                )
                
                # Update TreeView
                self.tree.item(circle_id, values=(dose_str, sigma_str, avg_str, avg_unc_str, ci95_str))

    # ---------------------------------------------------------------
    # Helper methods (detection and formatting delegated to helper classes)
    # ---------------------------------------------------------------
    
    # Wrapper methods for backward compatibility - delegate to helper classes
    def _fmt_sig(self, value: float, sig: int = 2) -> str:
        """Format number with significant figures (delegates to formatter)."""
        return self.formatter.format_significant(value, sig)
    
    def _clean_numeric_string(self, value_str: str) -> float:
        """Clean numeric string (delegates to formatter)."""
        return self.formatter.clean_numeric_string(value_str)
    
    def _format_val_unc(self, value: float, unc: float, sig: int = 2, force_decimals: bool = False) -> tuple[str, str]:
        """Format value ± uncertainty (delegates to formatter)."""
        return self.formatter.format_value_uncertainty(value, unc, sig, force_decimals)
    
    def _ensure_uint8(self, img: np.ndarray) -> np.ndarray:
        """Convert image to uint8 (delegates to detector)."""
        return self.detector.ensure_uint8(img)
    
    def _detect_films(self, gray) -> List[Tuple[int, int, int, int]]:
        """Detect films (delegates to detector)."""
        params = DetectionParams(
            rc_threshold=self.rc_thresh_var.get(),
            rc_min_area=self.rc_min_area_var.get()
        )
        return self.detector.detect_films(gray, params)
    
    def _detect_circles(self, roi) -> List[Tuple[int, int, int]]:
        """Detect circles (delegates to detector)."""
        params = DetectionParams(
            min_circle_radius=self.min_circle_var.get(),
            max_circle_radius=self.max_circle_var.get(),
            min_distance=self.min_dist_var.get(),
            param1=self.param1_var.get(),
            param2=self.param2_var.get()
        )
        return self.detector.detect_circles(roi, params)
    
    def _organize_circles_as_matrix(self, circles: List[Tuple[int, int, int]]) -> List[Tuple[str, Tuple[int, int, int]]]:
        """Organize circles as matrix (delegates to detector)."""
        return self.detector.organize_circles_as_matrix(circles)

    def _apply_diameter_restriction(self, *args):
        """Apply diameter restriction to detected circles."""
        # Using _get_parent_module()._OVERLAY (package-level global)
        if not _get_parent_module()._OVERLAY or "circles" not in _get_parent_module()._OVERLAY or not _get_parent_module()._OVERLAY["circles"]:
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
            for item_id, (stype, shape_data) in list(_get_parent_module()._OVERLAY.get("item_to_shape", {}).items()):
                if stype != "circle":
                    continue
                x_px, y_px, _ = shape_data
                new_circles.append((x_px, y_px, default_radius))
                
                # Update the stored shape data
                _get_parent_module()._OVERLAY["item_to_shape"][item_id] = ("circle", (x_px, y_px, default_radius))
                
                # Also update original_radii so it persists correctly when saving
                self.original_radii[item_id] = default_radius
                
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
                
                # Update tree view with new measurements with full precision
                # Extract measurement results (always 6-tuple format)
                dose, std, unc, rgb_mean, rgb_mean_std, pixel_count = res
                
                if isinstance(dose, tuple):
                    dose_parts: list[str] = []
                    unc_parts: list[str] = []
                    std_parts: list[str] = []
                    for v, s, u in zip(dose, std, unc):
                        dose_parts.append(f"{v}")
                        unc_parts.append(f"±{u}")
                        std_parts.append(f"{s}")
                    dose_str = ", ".join(dose_parts)
                    std_str = ", ".join(std_parts)
                    
                    # Use combined uncertainty from ImageProcessor
                    avg_val = float(rgb_mean)
                    avg_unc = float(rgb_mean_std)
                    avg_std = float(np.mean(std))
                    avg_std_str = f"{avg_std}"
                else:
                    # For single channel, use combined uncertainty values
                    avg_val = float(rgb_mean)
                    avg_unc = float(rgb_mean_std)
                
                # Format for TreeView display using formatter
                dose_str, std_str, avg_str, avg_unc_str, ci95_str = self.formatter.format_for_treeview(
                    dose,
                    std,
                    avg_val,
                    avg_unc,
                    sig=2
                )
                
                # Update original measurements
                self._store_original_measurement(item_id, dose_str, std_str, avg_str, avg_unc_str)
                
                # Update TreeView values (only if item exists)
                try:
                    if self.tree.exists(item_id):
                        self.tree.item(item_id, values=(dose_str, std_str, avg_str, avg_unc_str, ci95_str))
                    else:
                        continue
                except tk.TclError:
                    # Item no longer exists, skip update
                    continue
                
                # Update cached results for CSV
                try:
                    circle_name = self.tree.item(item_id, "text").replace(" (CTR)", "")
                except tk.TclError:
                    # Item no longer exists, skip CSV update
                    continue
                film_id = self.tree.parent(item_id)
                film_name = self.tree.item(film_id, "text") if film_id else None
                
                for rec in self.results:
                    if rec["film"] == film_name and rec["circle"] == circle_name:
                        rec["dose"] = dose_str
                        rec["unc"] = std_str  # This should be std
                        rec["avg"] = avg_val  # Store numeric value, not formatted string
                        rec["avg_unc"] = avg_unc  # Store numeric value
                        rec["std"] = avg_std_str
                        rec["pixel_count"] = pixel_count
                        # Also update the _numeric fields
                        rec["avg_numeric"] = avg_val
                        rec["avg_unc_numeric"] = avg_unc
                        rec["dose_numeric"] = dose
                        rec["std_numeric"] = std
                        rec["unc_numeric"] = unc
                        break

            _get_parent_module()._OVERLAY["circles"] = new_circles
        else:
            # Restore originally detected diameters
            new_circles = []
            for item_id, (stype, shape_data) in list(_get_parent_module()._OVERLAY.get("item_to_shape", {}).items()):
                if stype != "circle":
                    continue
                x, y, _ = shape_data
                # Get the originally detected radius (before any restriction was applied)
                original_r = self.detected_radii.get(item_id, self.original_radii.get(item_id, 20))
                
                # Update stored shape to original radius
                _get_parent_module()._OVERLAY["item_to_shape"][item_id] = ("circle", (x, y, original_r))
                # Also update original_radii to the detected radius
                self.original_radii[item_id] = original_r
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
                
                # Update tree view with original measurements with full precision
                # Extract measurement results (always 6-tuple format)
                dose, std, unc, rgb_mean, rgb_mean_std, pixel_count = res
                
                if isinstance(dose, tuple):
                    dose_parts: list[str] = []
                    unc_parts: list[str] = []
                    std_parts: list[str] = []
                    for v, s, u in zip(dose, std, unc):
                        dose_parts.append(f"{v}")
                        unc_parts.append(f"±{u}")
                        std_parts.append(f"{s}")
                    dose_str = ", ".join(dose_parts)
                    std_str = ", ".join(std_parts)
                    
                    # Use combined uncertainty from ImageProcessor
                    avg_val = float(rgb_mean)
                    avg_unc = float(rgb_mean_std)
                    avg_std = float(np.mean(std))
                    avg_std_str = f"{avg_std}"
                else:
                    # For single channel, use combined uncertainty values
                    avg_val = float(rgb_mean)
                    avg_unc = float(rgb_mean_std)
                    
                # Format for TreeView display using formatter
                dose_str, std_str, avg_str, avg_unc_str, ci95_str = self.formatter.format_for_treeview(
                    dose,
                    std,
                    avg_val,
                    avg_unc,
                    sig=2
                )
                
                # Update original measurements
                self._store_original_measurement(item_id, dose_str, std_str, avg_str, avg_unc_str)
                
                # Update TreeView values (only if item exists)
                try:
                    if self.tree.exists(item_id):
                        self.tree.item(item_id, values=(dose_str, std_str, avg_str, avg_unc_str, ci95_str))
                    else:
                        continue
                except tk.TclError:
                    # Item no longer exists, skip update
                    continue
                
                # Update cached results for CSV
                try:
                    circle_name = self.tree.item(item_id, "text").replace(" (CTR)", "")
                    film_id = self.tree.parent(item_id)
                    film_name = self.tree.item(film_id, "text") if film_id else None
                except tk.TclError:
                    # Item no longer exists, skip CSV update
                    continue
                
                for rec in self.results:
                    if rec["film"] == film_name and rec["circle"] == circle_name:
                        rec["dose"] = dose_str
                        rec["unc"] = std_str  # This should be std
                        rec["avg"] = avg_val  # Store numeric value, not formatted string
                        rec["avg_unc"] = avg_unc  # Store numeric value
                        rec["std"] = avg_std_str
                        rec["pixel_count"] = pixel_count
                        # Also update the _numeric fields for CSV export
                        rec["avg_numeric"] = avg_val
                        rec["avg_unc_numeric"] = avg_unc
                        rec["dose_numeric"] = dose
                        rec["std_numeric"] = std
                        rec["unc_numeric"] = unc
                        break

            _get_parent_module()._OVERLAY["circles"] = new_circles

        # Apply CTR subtraction if enabled
        self._update_ctr_subtraction()
        self.main_window.update_image()
        self._autosize_columns()

    def _clear_all(self):
        """Clear detections and data with options for single/multi-file mode."""
        if self.file_list:
            # Multi-file mode - ask user what to clear
            response = messagebox.askyesnocancel(
                "Clear Data",
                "Clear current file only? \n\n"
                "Yes = Clear current file only\n" 
                "No = Clear all files and data\n"
                "Cancel = Don't clear anything"
            )
            
            if response is None:  # Cancel
                return
            elif response:  # Yes - clear current file only
                self._clear_current_file()
            else:  # No - clear all files
                self._clear_all_files()
        else:
            # Single file mode - clear normally
            self._clear_current_file()
    
    def _clear_current_file(self):
        """Clear data for the current file only."""
        self.tree.delete(*self.tree.get_children())
        self.results.clear()
        self.original_measurements.clear()
        self.original_values.clear()
        self.ctr_map.clear()
        
        # Mark current file as not measured
        if self.file_list and 0 <= self.current_file_index < len(self.file_list):
            current_file = self.file_list[self.current_file_index]
            if current_file in self.file_data:
                self.file_data[current_file] = {
                    'results': [],
                    'ctr_map': {},
                    'original_measurements': {},
                    'original_radii': {},
                    'original_values': {},
                    'measured': False,
                    'overlay_state': {
                        'films': [],
                        'circles': [],
                        'item_to_shape': {},
                        'ctr_map': {},
                        '_shape': (0, 0),
                        'scale': 1.0
                    }
                }
        
        # Reset parameter conversion state
        if self.parameters_converted:
            self._restore_original_parameters()
        
        self._clear_overlay()
        self.main_window.update_image()
    
    def _clear_all_files(self):
        """Clear all files and data completely."""
        self.tree.delete(*self.tree.get_children())
        self.results.clear()
        self.original_measurements.clear()
        self.original_values.clear()
        self.ctr_map.clear()
        
        # Clear multi-file data (delegates to FileDataManager)
        self.file_manager.clear_all()
        
        # Reset parameter conversion state
        if self.parameters_converted:
            self._restore_original_parameters()
        
        self._clear_overlay()
        self.main_window.update_image()

    def _get_image_shape(self):
        """Get the shape of the current image for overlay initialization.
        
        Returns:
            tuple: (height, width) of the image, or (0, 0) if no image loaded
        """
        if hasattr(self.image_processor, 'original_image') and self.image_processor.original_image is not None:
            return self.image_processor.original_image.shape[:2]
        elif self.image_processor.current_image is not None:
            return self.image_processor.current_image.shape[:2]
        else:
            return (0, 0)

    def _clear_overlay(self):
        """Clear overlay data while preserving image shape."""
        img_shape = self._get_image_shape()
        
        if img_shape == (0, 0):
            logging.debug("[_clear_overlay] No image available - using (0, 0) shape")
        
        _get_parent_module()._OVERLAY = {
            "films": [], 
            "circles": [], 
            "_shape": img_shape, 
            "scale": 1.0, 
            "ctr_map": {}, 
            "item_to_shape": {}
        }
        logging.debug(f"[_clear_overlay] Cleared overlay with shape {img_shape}")

    def _delete_item_recursive(self, item_id):
        """Recursively delete item and clean up data."""
        # Get item info before deletion for results cleanup
        item_text = self.tree.item(item_id, "text")
        parent_id = self.tree.parent(item_id)
        parent_text = self.tree.item(parent_id, "text") if parent_id else None
        
        # First delete children recursively
        for child in self.tree.get_children(item_id):
            self._delete_item_recursive(child)

        # Remove from overlays and data structures
        if _get_parent_module()._OVERLAY and item_id in _get_parent_module()._OVERLAY.get("item_to_shape", {}):
            shape_type, coords = _get_parent_module()._OVERLAY["item_to_shape"].pop(item_id)
            
            if shape_type == "film":
                if coords in _get_parent_module()._OVERLAY.get("films", []):
                    _get_parent_module()._OVERLAY["films"].remove(coords)
                # Remove any CTR mapping for this film
                film_name = item_text
                self.ctr_map.pop(film_name, None)
                
                # Remove all circles from this film from results
                self.results = [rec for rec in self.results 
                              if rec["film"] != film_name]
                
            elif shape_type == "circle":
                if coords in _get_parent_module()._OVERLAY.get("circles", []):
                    _get_parent_module()._OVERLAY["circles"].remove(coords)
                # Remove from CTR mapping if it was a CTR
                for film_name, ctr_id in list(self.ctr_map.items()):
                    if ctr_id == item_id:
                        self.ctr_map.pop(film_name)
                        break
                
                # Remove this specific circle from results
                circle_name_clean = item_text.replace(" (CTR)", "")
                self.results = [rec for rec in self.results 
                              if not (rec["film"] == parent_text and 
                                     rec["circle"].replace(" (CTR)", "") == circle_name_clean)]

        # Clean up stored data
        self.original_measurements.pop(item_id, None)
        self.original_values.pop(item_id, None)
        self.original_radii.pop(item_id, None)
        self.detected_radii.pop(item_id, None)

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
        
        # Save references to ImagePanel's original methods before overriding
        if hasattr(self.main_window, 'image_panel'):
            image_panel = self.main_window.image_panel
            self._original_bindings = {
                '<Motion>': image_panel._on_mouse_move,
                '<Button-1>': image_panel._on_click
            }
        
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
            
            # Restore original ImagePanel bindings
            if hasattr(self, "_original_bindings") and self._original_bindings:
                for event_type, callback in self._original_bindings.items():
                    if callback:
                        self._canvas.bind(event_type, callback)
                self._original_bindings.clear()

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
        # Initialize _OVERLAY if needed
        if _get_parent_module()._OVERLAY is None:
            img_shape = self._get_image_shape()
            if img_shape == (0, 0):
                logging.error("[_insert_film] No image available")
                return
            
            _get_parent_module()._OVERLAY = {
                "films": [], "circles": [], "_shape": img_shape,
                "scale": 1.0, "ctr_map": {}, "item_to_shape": {}
            }
            logging.info(f"[_insert_film] Initialized _OVERLAY with shape {img_shape}")

        film_idx = len([v for v in _get_parent_module()._OVERLAY["item_to_shape"].values() if v[0] == "film"]) + 1
        film_name = f"RC_{film_idx}M"
        film_id = self.tree.insert("", "end", text=film_name, values=("", "", "", "", ""))
        _get_parent_module()._OVERLAY["item_to_shape"][film_id] = ("film", (x, y, w, h))
        _get_parent_module()._OVERLAY["films"].append((x, y, w, h))
        logging.info(f"[_insert_film] Added {film_name} at ({x}, {y}, {w}, {h})")
        logging.info(f"[_insert_film] Total films in _OVERLAY: {len(_get_parent_module()._OVERLAY['films'])}")
        self.tree.item(film_id, open=True)
        self.main_window.update_image()

    def _insert_circle(self, cx: int, cy: int, r: int):
        """Insert a new circle manually."""
        # Initialize _OVERLAY if needed
        if _get_parent_module()._OVERLAY is None:
            img_shape = self._get_image_shape()
            if img_shape == (0, 0):
                logging.error("[_insert_circle] No image available")
                return
                
            _get_parent_module()._OVERLAY = {
                "films": [], "circles": [], "_shape": img_shape,
                "scale": 1.0, "ctr_map": {}, "item_to_shape": {}
            }
            logging.info(f"[_insert_circle] Initialized _OVERLAY with shape {img_shape}")

        # Find parent film
        parent_id = ""
        parent_film_name = None
        for fid, (item_type, coords) in _get_parent_module()._OVERLAY["item_to_shape"].items():
            if item_type != "film":
                continue
            fx, fy, fw, fh = coords
            if (fx <= cx <= fx + fw) and (fy <= cy <= fy + fh):
                parent_id = fid
                parent_film_name = self.tree.item(fid, 'text')
                break
        
        # VALIDATION: Circle must be inside a film
        if not parent_id:
            messagebox.showwarning(
                "Add Circle",
                "El círculo debe estar dentro de una radiocrómica (RC).\n\n"
                "Por favor, primero añade un RC (Add RC) y luego\n"
                "dibuja el círculo dentro de sus límites."
            )
            logging.warning(f"[_insert_circle] Circle at ({cx}, {cy}) rejected: no parent film found")
            return

        circ_idx = len([v for v in _get_parent_module()._OVERLAY["item_to_shape"].values() if v[0] == "circle"]) + 1
        circ_name = f"C{circ_idx}M"
        logging.debug(f"[_insert_circle] Circle {circ_name} will be added to parent film {parent_film_name}")
        
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

            # Format measurements with full precision
            if isinstance(dose, tuple):
                dose_parts = []
                unc_parts = []
                std_parts = []
                for v, s, u in zip(dose, std, unc):
                    dose_parts.append(f"{v}")
                    unc_parts.append(f"±{u}")
                    std_parts.append(f"{s}")
                dose_str = ", ".join(dose_parts)
                unc_str = ", ".join(unc_parts)
                std_str = ", ".join(std_parts)
                
                # Use combined uncertainty from ImageProcessor
                avg_val = float(rgb_mean)
                avg_unc = float(rgb_mean_std)
                avg_std = float(np.mean(std))
                avg_std_str = f"{avg_std}"
            else:
                # For single channel, use combined uncertainty values
                avg_val = float(rgb_mean)
                avg_unc = float(rgb_mean_std)
            
            # Format for TreeView display using formatter
            dose_str, std_str, avg_str, avg_unc_str, ci95_str = self.formatter.format_for_treeview(
                dose,
                std,
                avg_val,
                avg_unc,
                sig=2
            )
        else:
            dose_str = unc_str = avg_str = avg_unc_str = ci95_str = ""
            pixel_count = 0
            avg_val = avg_unc = 0.0

        circ_id = self.tree.insert(parent_id, "end", text=circ_name, 
                                 values=(dose_str, std_str, avg_str, avg_unc_str, ci95_str))
        
        # Store mapping and original measurements
        _get_parent_module()._OVERLAY["item_to_shape"][circ_id] = ("circle", (cx, cy, r))
        self.original_radii[circ_id] = r
        self._store_original_measurement(circ_id, dose_str, std_str, avg_str, avg_unc_str)
        
        # Add to overlay circles
        _get_parent_module()._OVERLAY["circles"].append((cx, cy, r))
        logging.info(f"[_insert_circle] Added {circ_name} at ({cx}, {cy}, {r})")
        logging.info(f"[_insert_circle] Total circles in _OVERLAY: {len(_get_parent_module()._OVERLAY['circles'])}")

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

        # Store for CTR detection and results
        # Store result
        film_name = self.tree.item(parent_id, "text") if parent_id else None
        self.results.append({
            "film": film_name,
            "circle": circ_name,
            "dose": dose if res else 0.0,               # Raw numeric dose (NOT formatted string)
            "std_per_channel": std if res else 0.0,     # Raw numeric STD per channel (NOT formatted string)
            "unc": unc if res else 0.0,                 # Raw numeric SE/Uncertainty per channel
            "avg": avg_val,                             # Store numeric value (not formatted string)
            "avg_unc": avg_unc,                         # Store numeric value directly
            "std": avg_std if res else 0.0,             # Raw numeric average std
            "pixel_count": pixel_count,
            "x": cx,
            "y": cy,
            # Store raw numeric values for CSV export (redundant but kept for compatibility)
            "dose_numeric": dose if res else 0.0,
            "std_numeric": std if res else 0.0,
            "unc_numeric": unc if res else 0.0,
            "avg_numeric": avg_val,
            "std_avg_numeric": avg_std if res else 0.0,
            "avg_unc_numeric": avg_unc,                 # Raw SE/uncertainty value (float)
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
                shape_data = _get_parent_module()._OVERLAY.get("item_to_shape", {}).get(circle_id)
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
    
    def _format_for_csv(self, value):
        """Format value for CSV (delegates to CSVExporter)."""
        return self.csv_exporter.format_for_csv(value)

    def _get_export_values_for_result(self, result, film_name=None, ctr_map_by_name=None, file_results=None):
        """Get export values for result (delegates to CSVExporter)."""
        return self.csv_exporter.get_export_values_for_result(result)

    def export_csv(self):
        """Export measurements to CSV (delegates to CSVExporter)."""
        self.csv_exporter.export_all_files(
            self.results,
            self.ctr_map,
            self.date_var,
            self.metadata_date,
            self.ctr_manager.original_measurements,
            self.original_radii,
            self.original_values
        )




__all__ = ['AutoMeasurementsTab']
