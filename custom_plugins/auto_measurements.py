"""AutoMeasurements plugin - Versión mejorada con funcionalidad CTR completa

Detects rectangular radiochromic films and circular ROIs inside them,
computes dose and uncertainty, and shows results in a TreeView with export.
Includes complete CTR (Control) circle functionality for background subtraction.

TAB_TITLE = "AutoMeasurements"
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import tkinter.font as tkfont
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3D
from math import log10, floor, sqrt

# Plugin interface ---------------------------------------------------------
TAB_TITLE = "AutoMeasurements"


def setup(main_window, notebook, image_processor):
    return AutoMeasurementsTab(main_window, notebook, image_processor).frame


_OVERLAY: dict | None = None  # Holds last detection for drawing

# Keys used in _OVERLAY:
#   "films": list[(x, y, w, h)]
#   "circles": list[(cx, cy, r)]
#   "_shape": (h, w) of detection image
#   "highlight": ("film"|"circle", coords) – optional highlighted shape
#   "ctr_map": dict[film_name, circle_id] - mapa de círculos de control
#   "item_to_shape": dict[item_id, (type, coords)] - mapeo de items a formas

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

    def _setup_ui(self):
        """Setup the user interface."""
        ttk.Label(self.frame, text="Auto Measurements", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Main button frame
        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill=tk.X, padx=10)
        ttk.Button(btn_frame, text="Iniciar Detección", command=self.start_detection).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Limpiar", command=self._clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Añadir RC", command=self._add_manual_film).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Añadir Círculo", command=self._add_manual_circle).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Exportar CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5)

        # CTR control frame
        ctr_frame = ttk.Frame(self.frame)
        ctr_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(ctr_frame, text="Marcar/Desmarcar CTR", command=self._toggle_ctr_manual).pack(side=tk.LEFT)
        
        # Add CTR subtraction checkbox
        self.subtract_ctr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctr_frame, text="Restar fondo CTR", variable=self.subtract_ctr_var,
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
        self.tree.heading("#0", text="Elemento")
        self.tree.heading("dose", text="Dosis")
        self.tree.heading("sigma", text="Incertidumbre")
        self.tree.heading("avg", text="Promedio")
        self.tree.heading("avg_unc", text="Inc. Promedio")
        
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
        film_frame = ttk.LabelFrame(self.frame, text="Detección RC")
        film_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(film_frame, text="Umbral (0-255):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(film_frame, textvariable=self.rc_thresh_var, width=6).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(film_frame, text="Área mín (px):").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(film_frame, textvariable=self.rc_min_area_var, width=8).grid(row=1, column=1, sticky=tk.W)

        # Circle detection parameters
        self.min_circle_var = tk.IntVar(value=200)
        self.max_circle_var = tk.IntVar(value=400)
        self.min_dist_var = tk.IntVar(value=200)
        self.param1_var = tk.IntVar(value=15)
        self.param2_var = tk.IntVar(value=40)
        self.default_diameter_var = tk.IntVar(value=300)
        
        circle_frame = ttk.LabelFrame(self.frame, text="Detección Círculos")
        circle_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(circle_frame, text="Radio mín:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.min_circle_var, width=6).grid(row=0, column=1)
        ttk.Label(circle_frame, text="Radio máx:").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.max_circle_var, width=6).grid(row=0, column=3)
        ttk.Label(circle_frame, text="Distancia mín:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.min_dist_var, width=6).grid(row=1, column=1)
        ttk.Label(circle_frame, text="Param1:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.param1_var, width=6).grid(row=2, column=1)
        ttk.Label(circle_frame, text="Param2:").grid(row=2, column=2, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.param2_var, width=6).grid(row=2, column=3)
        
        ttk.Label(circle_frame, text="Diámetro por defecto:").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.default_diameter_var, width=6).grid(row=3, column=1)
        
        # Add diameter restriction checkbox
        self.restrict_diameter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            circle_frame,
            text="Usar diámetro por defecto para todos los círculos",
            variable=self.restrict_diameter_var,
            command=self._apply_diameter_restriction,
        ).grid(row=4, column=0, columnspan=4, sticky=tk.W)
        
        # Recalculate when default diameter value changes
        self.default_diameter_var.trace_add("write", lambda *args: self._apply_diameter_restriction())

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
        else:
            # Remove existing CTR if any
            if film_name in self.ctr_map:
                old_ctr_id = self.ctr_map[film_name]
                old_text = self.tree.item(old_ctr_id, "text")
                if "(CTR)" in old_text:
                    self.tree.item(old_ctr_id, text=old_text.replace(" (CTR)", ""))
                self.tree.item(old_ctr_id, tags=())
            
            # Set new CTR
            self.ctr_map[film_name] = item_id
            current_text = self.tree.item(item_id, "text")
            if "(CTR)" not in current_text:
                self.tree.item(item_id, text=f"{current_text} (CTR)")
            self.tree.item(item_id, tags=("ctr",))
        
        # Update CTR subtraction
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

    def _store_original_measurement(self, item_id: str, dose_str: str, unc_str: str, avg_str: str, avg_unc_str: str):
        """Store original measurement data before any CTR corrections."""
        self.original_measurements[item_id] = {
            "dose": dose_str,
            "unc": unc_str,
            "avg": avg_str,
            "avg_unc": avg_unc_str
        }

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

        # 1. Detect films
        films = self._detect_films(gray)
        if not films:
            messagebox.showinfo("AutoMeasurements", "No se detectaron radiocromías.")
            return

        # Clear previous results
        self.tree.delete(*self.tree.get_children())
        self.results.clear()
        self.original_measurements.clear()
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

                dose, _, unc, _ = res
                
                # Format measurements
                if isinstance(dose, tuple):
                    dose_parts = []
                    unc_parts = []
                    for v, u in zip(dose, unc):
                        v_str, u_str = self._format_val_unc(v, u, 2)
                        dose_parts.append(v_str)
                        unc_parts.append(u_str)
                    dose_str = ", ".join(dose_parts)
                    unc_str = ", ".join(unc_parts)
                    avg_val = float(np.mean(dose))
                    avg_unc = float(np.mean(unc))
                else:
                    dose_str, unc_str = self._format_val_unc(dose, unc, 2)
                    avg_val = float(dose)
                    avg_unc = float(unc)

                avg_str, avg_unc_str = self._format_val_unc(avg_val, avg_unc, 2)

                # Create circle item
                circ_name = f"C{jdx}"
                circ_id = self.tree.insert(film_id, "end", text=circ_name, 
                                         values=(dose_str, unc_str, avg_str, avg_unc_str))
                
                # Store mapping and original measurements
                _OVERLAY["item_to_shape"][circ_id] = ("circle", (abs_cx, abs_cy, r_int))
                self.original_radii[circ_id] = orig_r_int
                self._store_original_measurement(circ_id, dose_str, unc_str, avg_str, avg_unc_str)
                
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
                })

                # Store results
                self.results.append({
                    "film": film_name,
                    "circle": circ_name,
                    "dose": dose_str,
                    "unc": unc_str,
                    "avg": avg_str,
                    "avg_unc": avg_unc_str,
                    "x": abs_cx,
                    "y": abs_cy,
                })

            # 3. Automatic CTR detection
            self._detect_ctr_automatically(film_name, film_circles)

        # Open all film nodes
        for child in self.tree.get_children():
            self.tree.item(child, open=True)

        # Apply CTR subtraction if enabled
        self._update_ctr_subtraction()

        # Update display
        self.main_window.update_image()
        self._autosize_columns()

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

    def _format_val_unc(self, value: float, unc: float, sig: int = 2) -> tuple[str, str]:
        """Format value and uncertainty with consistent decimal places."""
        unc_fmt = self._fmt_sig(unc, sig)

        # Scientific notation – fall back to significant-figure formatting
        if "e" in unc_fmt or "E" in unc_fmt:
            val_fmt = self._fmt_sig(value, sig)
        else:
            # Count decimals in uncertainty (0 if integer)
            if "." in unc_fmt:
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
                dose, std_dev, unc, _ = res
                if isinstance(dose, tuple):
                    dose_parts: list[str] = []
                    unc_parts: list[str] = []
                    for v, u in zip(dose, unc):
                        v_str, u_str = self._format_val_unc(v, u, 2)
                        dose_parts.append(v_str)
                        unc_parts.append(u_str)
                    dose_str = ", ".join(dose_parts)
                    unc_str = ", ".join(unc_parts)
                    avg_val = float(np.mean(dose))
                    avg_unc = float(np.mean(unc))
                else:
                    dose_str, unc_str = self._format_val_unc(dose, unc, 2)
                    avg_val = float(dose)
                    avg_unc = float(unc)
                
                avg_str, avg_unc_str = self._format_val_unc(avg_val, avg_unc, 2)
                
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
                dose, _, unc, _ = res  # mean(s), std, uncertainty
                if isinstance(dose, tuple):
                    dose_parts: list[str] = []
                    unc_parts: list[str] = []
                    for v, u in zip(dose, unc):
                        v_str, u_str = self._format_val_unc(v, u, 2)
                        dose_parts.append(v_str)
                        unc_parts.append(u_str)
                    dose_str = ", ".join(dose_parts)
                    unc_str = ", ".join(unc_parts)
                    avg_val = float(np.mean(dose))
                    avg_unc = float(np.mean(unc))
                else:
                    dose_str, unc_str = self._format_val_unc(dose, unc, 2)
                    avg_val = float(dose)
                    avg_unc = float(unc)
                    
                avg_str, avg_unc_str = self._format_val_unc(avg_val, avg_unc, 2)
                
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
        self.ctr_map.clear()
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
            plot_title = "Mapa 3D de Dosis"
            z_label = "Dosis"
        else:
            img = (
                self.image_processor.original_image
                if getattr(self.image_processor, "original_image", None) is not None
                else self.image_processor.current_image
            )
            plot_title = "Mapa 3D de Intensidad"
            z_label = "Valor"
        
        if img is None:
            messagebox.showwarning("Vista 3D", "No hay imagen cargada.")
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
            messagebox.showwarning("Dibujar", "No hay imagen cargada o canvas no disponible.")
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
        self._dims_window.title("Dimensiones")
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
            tk.Label(self._dims_window, text="Ancho:").grid(row=0, column=0, padx=4, pady=2)
            tk.Entry(self._dims_window, textvariable=w_var, width=6).grid(row=0, column=1, pady=2)
            tk.Label(self._dims_window, text="Alto:").grid(row=1, column=0, padx=4, pady=2)
            tk.Entry(self._dims_window, textvariable=h_var, width=6).grid(row=1, column=1, pady=2)
        elif self.draw_mode == "circle":
            r = self.draw_dims
            r_var = tk.StringVar(value=str(r))
            self._dim_vars.append(r_var)
            tk.Label(self._dims_window, text="Radio:").grid(row=0, column=0, padx=4, pady=2)
            tk.Entry(self._dims_window, textvariable=r_var, width=6).grid(row=0, column=1, pady=2)

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
                "ctr_map": {}, "item_to_shape": {}
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
                "ctr_map": {}, "item_to_shape": {}
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
            dose, _, unc, _ = res
            if isinstance(dose, tuple):
                dose_parts = []
                unc_parts = []
                for v, u in zip(dose, unc):
                    v_str, u_str = self._format_val_unc(v, u, 2)
                    dose_parts.append(v_str)
                    unc_parts.append(u_str)
                dose_str = ", ".join(dose_parts)
                unc_str = ", ".join(unc_parts)
                avg_val = float(np.mean(dose))
                avg_unc = float(np.mean(unc))
            else:
                dose_str, unc_str = self._format_val_unc(dose, unc, 2)
                avg_val = float(dose)
                avg_unc = float(unc)
            
            avg_str, avg_unc_str = self._format_val_unc(avg_val, avg_unc, 2)
        else:
            dose_str = unc_str = avg_str = avg_unc_str = ""
            avg_val = avg_unc = 0.0

        circ_id = self.tree.insert(parent_id, "end", text=circ_name, 
                                 values=(dose_str, unc_str, avg_str, avg_unc_str))
        _OVERLAY["item_to_shape"][circ_id] = ("circle", (cx, cy, r))
        _OVERLAY["circles"].append((cx, cy, r))
        
        # Store original measurement
        self._store_original_measurement(circ_id, dose_str, unc_str, avg_str, avg_unc_str)
        
        if parent_id:
            self.tree.item(parent_id, open=True)

        # Add to results
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
        for film_id in self.tree.get_children(''):
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
        films = list(self.tree.get_children(''))
        films.sort(key=lambda film_id: self.tree.item(film_id, "text"))
        for index, film_id in enumerate(films):
            self.tree.move(film_id, '', index)
            
            circles = list(self.tree.get_children(film_id))
            circles.sort(key=lambda circle_id: self.tree.item(circle_id, "text"))
            for c_index, circle_id in enumerate(circles):
                self.tree.move(circle_id, film_id, c_index)

    def _sort_by_value(self, col, reverse=False):
        """Sort circles by column value."""
        for film_id in self.tree.get_children(''):
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
        """Export results to CSV file with both original and CTR-corrected values."""
        if not self.results:
            messagebox.showinfo("Exportar", "No hay datos para exportar.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Guardar CSV", 
            defaultextension=".csv", 
            filetypes=[("CSV", "*.csv")]
        )
        if not file_path:
            return
        
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Write header with both original and corrected values
                writer.writerow([
                    "Radiocrómica", "Círculo", "X", "Y", 
                    "Dosis_Original", "Inc_Original", "Promedio_Original", "Inc_Promedio_Original",
                    "Promedio_Corregido_CTR", "Inc_Promedio_Corregido_CTR"
                ])
                
                # Sort results by Film, then by Circle
                # Use a key function that handles potential None values for film or circle names
                sorted_results = sorted(self.results, key=lambda rec: (str(rec.get("film") or ""), str(rec.get("circle") or "")))
                for rec in sorted_results:
                    film_name = rec["film"]
                    circle_name = rec["circle"]
                    
                    # Find the corresponding tree item for this circle
                    circle_display_name = circle_name
                    item_id = None
                    is_ctr = False
                    
                    # Search through all circle items to find the matching one
                    for iid, (stype, coords) in _OVERLAY.get("item_to_shape", {}).items():
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
                    original_avg = orig_data.get("avg", rec["avg"])
                    original_avg_unc = orig_data.get("avg_unc", rec["avg_unc"])
                    original_dose = orig_data.get("dose", rec["dose"])
                    original_unc = orig_data.get("unc", rec["unc"])
                    
                    # Calculate CTR-corrected values
                    ctr_corrected_avg = ""
                    ctr_corrected_unc = ""
                    
                    if film_name in self.ctr_map and item_id:
                        ctr_id = self.ctr_map[film_name]
                        ctr_orig_data = self.original_measurements.get(ctr_id, {})
                        
                        if ctr_orig_data:
                            ctr_avg = self._clean_numeric_string(ctr_orig_data.get("avg", "0"))
                            ctr_unc = self._clean_numeric_string(ctr_orig_data.get("avg_unc", "0"))
                            orig_avg_num = self._clean_numeric_string(original_avg)
                            orig_unc_num = self._clean_numeric_string(original_avg_unc)
                            
                            if item_id == ctr_id:
                                # This is the CTR circle
                                corrected_avg = 0.0
                                corrected_unc = ctr_unc
                            else:
                                # Other circles: subtract CTR with error propagation
                                corrected_avg = max(0.0, orig_avg_num - ctr_avg)
                                corrected_unc = sqrt(orig_unc_num**2 + ctr_unc**2)
                            
                            ctr_corrected_avg, ctr_corrected_unc = self._format_val_unc(corrected_avg, corrected_unc, 2)
                    
                    # If no CTR correction available, use original values
                    if not ctr_corrected_avg:
                        ctr_corrected_avg = original_avg
                        ctr_corrected_unc = original_avg_unc
                    
                    writer.writerow([
                        rec["film"] or "",
                        circle_display_name,
                        
                        rec.get("x", ""),
                        rec.get("y", ""),
                        original_dose,
                        original_unc,
                        original_avg,
                        original_avg_unc,
                        ctr_corrected_avg,
                        ctr_corrected_unc
                    ])
            messagebox.showinfo("Exportar", f"CSV exportado exitosamente a:\n{file_path}")
        except Exception as exc:
            messagebox.showerror("Error de Exportación", f"Error al exportar CSV:\n{str(exc)}")