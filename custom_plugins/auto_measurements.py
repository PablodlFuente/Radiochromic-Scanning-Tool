"""AutoMeasurements plugin

Detects rectangular radiochromic films and circular ROIs inside them,
computes dose and uncertainty, and shows results in a TreeView with export.

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
from math import log10, floor

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

def process(image: np.ndarray):
    """Draw bounding rectangles and circles from last detection."""
    global _OVERLAY
    if _OVERLAY is None:
        return image
    # Dynamically scale overlay if display size differs from detection image
    orig_h, orig_w = _OVERLAY.get("_shape", image.shape[:2])
    disp_h, disp_w = image.shape[:2]
    sx = disp_w / orig_w if orig_w else 1.0
    sy = disp_h / orig_h if orig_h else 1.0
    # Ensure we have 3 channels for colored overlays
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()
    # Draw films scaled (green)
    for (x, y, w, h) in _OVERLAY.get("films", []):
        cv2.rectangle(out,
                      (int(x * sx), int(y * sy)),
                      (int((x + w) * sx), int((y + h) * sy)),
                      (0, 255, 0), 2)
    # Draw circles scaled (blue)
    for (cx, cy, r) in _OVERLAY.get("circles", []):
        cv2.circle(out,
                   (int(cx * sx), int(cy * sy)),
                   int(r * sx),
                   (255, 0, 0), 2)
    # Draw highlighted shape on top (yellow)
    hl = _OVERLAY.get("highlight")
    if hl is not None:
        hl_type, coords = hl
        if hl_type == "film":
            x, y, w, h = coords
            cv2.rectangle(out,
                          (int(x * sx), int(y * sy)),
                          (int((x + w) * sx), int((y + h) * sy)),
                          (0, 255, 255), 3)
        elif hl_type == "circle":
            cx, cy, r = coords
            cv2.circle(out,
                       (int(cx * sx), int(cy * sy)),
                       int(r * sx),
                       (0, 255, 255), 3)
    return out




# -------------------------------------------------------------------------

class AutoMeasurementsTab(ttk.Frame):
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
            if item_id in self.item_to_shape:
                _OVERLAY["highlight"] = self.item_to_shape[item_id]
        # Refresh display
        self.main_window.update_image()

    def _fmt_sig(self, value: float, sig: int = 2) -> str:
        """Format number with the given significant figures."""
        if value == 0 or not np.isfinite(value):
            return f"{value}"
        return f"{value:.{sig}g}"

    def _format_val_unc(self, value: float, unc: float, sig: int = 2) -> tuple[str, str]:
        """Format *value* and its *uncertainty* so that *value* shows the
        same number of decimal places as the formatted uncertainty.

        The uncertainty is first rendered with *sig* significant figures
        (using :py:meth:`_fmt_sig`).  If the uncertainty ends up in
        scientific notation (e-format) we also format the value with the
        same number of significant figures.  Otherwise, the number of
        digits that appear after the decimal point in the uncertainty is
        used for the value as well.  A leading ``"±"`` is prepended to the
        returned uncertainty string.
        """
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

    def __init__(self, main_window, notebook, image_processor):
        self.item_to_shape: dict[str, tuple] = {}  # maps TreeView items to shape info
        self.main_window = main_window
        self.image_processor = image_processor
        self.frame = ttk.Frame(notebook)

        # Reference to the image canvas for manual drawing
        self._canvas = (
            self.main_window.image_panel.canvas if hasattr(self.main_window, "image_panel") else None
        )

        # UI
        ttk.Label(self.frame, text="Auto Measurements", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))

        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill=tk.X, padx=10)
        ttk.Button(btn_frame, text="Start", command=self.start_detection).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Clear", command=self._clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Add RC", command=self._add_manual_film).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Add Circle", command=self._add_manual_circle).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Export to CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5)

        # ----------------- Detection Parameter Panels -----------------
        # Bind selection event for highlighting
        # TreeView columns – always show channel data + global averages
        cols = ["dose", "sigma", "avg", "avg_unc"]

        self.tree = ttk.Treeview(self.frame, columns=tuple(cols), show="tree headings")
        self.tree.heading("dose", text="Dose")
        self.tree.heading("sigma", text="Uncertainty")
        self.tree.heading("avg", text="Avg")
        self.tree.heading("avg_unc", text="Avg uncert.")
        self.tree.column("dose", width=80, anchor=tk.CENTER)
        self.tree.column("sigma", width=100, anchor=tk.CENTER)
        self.tree.column("avg", width=80, anchor=tk.CENTER)
        self.tree.column("avg_unc", width=80, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Delete>", self._on_delete_key)
        self.tree.bind("<Button-3>", self._on_right_click)
        self.tree.bind("<Double-1>", self._on_edit_label)
        # Auto-adjust initial column widths
        self._autosize_columns()
        # Film detection parameters
        self.rc_thresh_var = tk.IntVar(value=180)
        self.rc_min_area_var = tk.IntVar(value=5000)
        film_frame = ttk.LabelFrame(self.frame, text="RC Detection")
        film_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(film_frame, text="Threshold (0-255):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(film_frame, textvariable=self.rc_thresh_var, width=6).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(film_frame, text="Min area (px):").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(film_frame, textvariable=self.rc_min_area_var, width=8).grid(row=1, column=1, sticky=tk.W)

        # Circle detection parameters
        self.min_circle_var = tk.IntVar(value=200)
        self.max_circle_var = tk.IntVar(value=400)
        self.min_dist_var = tk.IntVar(value=200)
        self.param1_var = tk.IntVar(value=15)  # Canny high threshold
        self.param2_var = tk.IntVar(value=40)  # Accumulator threshold
        self.default_diameter_var = tk.IntVar(value=300)  # New default diameter variable
        circle_frame = ttk.LabelFrame(self.frame, text="Circle Detection")
        circle_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(circle_frame, text="Min radius:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.min_circle_var, width=6).grid(row=0, column=1)
        ttk.Label(circle_frame, text="Max radius:").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.max_circle_var, width=6).grid(row=0, column=3)
        ttk.Label(circle_frame, text="MinDist:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.min_dist_var, width=6).grid(row=1, column=1)
        ttk.Label(circle_frame, text="Param1 (Canny high thr):").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.param1_var, width=6).grid(row=2, column=1)
        ttk.Label(circle_frame, text="Param2 (Accumulator thr):").grid(row=2, column=2, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.param2_var, width=6).grid(row=2, column=3)
        ttk.Label(circle_frame, text="Default diameter:").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.default_diameter_var, width=6).grid(row=3, column=1)
        # Add a checkbox for restricting diameter
        self.restrict_diameter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            circle_frame,
            text="Use default diameter for all circles",
            variable=self.restrict_diameter_var,
            command=self._apply_diameter_restriction,
        ).grid(row=4, column=0, columnspan=4, sticky=tk.W)
        # Recalculate when default diameter value changes
        self.default_diameter_var.trace_add("write", lambda *args: self._apply_diameter_restriction())
        # Enable inline editing of item text (film/circle names)
        # Storage
        self.results = []  # List[Dict]
        self.original_radii = {}  # Store original radii for circle items
        
        # Drawing attributes
        self.draw_mode = None
        self.draw_dims = None
        self._dims_window = None
        self._dim_vars = []
        self._last_cursor = None
        self.preview_tag = "draw_preview"  # Unique tag for canvas preview items

    def _apply_diameter_restriction(self, *args):
        """Adjust circle diameters based on checkbox state"""
        global _OVERLAY
        if not _OVERLAY or "circles" not in _OVERLAY or not _OVERLAY["circles"]:
            return

        if self.restrict_diameter_var.get():
            # Apply default diameter restriction
            default_diameter = max(1, self.default_diameter_var.get())
            default_radius = max(1, int(round(default_diameter / 2)))
            
            new_circles = []
            for item_id, (stype, shape_data) in list(self.item_to_shape.items()):
                if stype != "circle":
                    continue
                x_px, y_px, _ = shape_data
                new_circles.append((x_px, y_px, default_radius))
                
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
                
                # Update tree view
                dose, std_dev, unc, _ = res
                if isinstance(dose, tuple):
                    dose_parts, unc_parts = [], []
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

                val_tuple = (dose_str, unc_str, avg_str, avg_unc_str)
                self.tree.item(item_id, values=val_tuple)
                self.item_to_shape[item_id] = ("circle", (x_px, y_px, default_radius))

            _OVERLAY["circles"] = new_circles
        else:
            # Restore originally detected diameters
            new_circles = []
            for item_id, (stype, shape_data) in list(self.item_to_shape.items()):
                if stype != "circle":
                    continue
                x, y, _ = shape_data
                original_r = int(self.original_radii.get(item_id, _))
                # Update stored shape to original radius
                self.item_to_shape[item_id] = ("circle", (x, y, original_r))
                new_circles.append((x, y, original_r))
            
            _OVERLAY["circles"] = new_circles
            # Recalculate measurements with original radii
            for item_id, (stype, shape_data) in list(self.item_to_shape.items()):
                if stype != "circle":
                    continue
                x_px, y_px, r = shape_data
                
                # Recalculate measurements
                prev_size = self.image_processor.measurement_size
                try:
                    self.image_processor.measurement_size = r
                    res = self.image_processor.measure_area(
                        x_px * self.image_processor.zoom,
                        y_px * self.image_processor.zoom
                    )
                finally:
                    self.image_processor.measurement_size = prev_size
                
                if res is None:
                    continue
                
                # Update tree view
                dose, std_dev, unc, _ = res
                if isinstance(dose, tuple):
                    dose_parts, unc_parts = [], []
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

                val_tuple = (dose_str, unc_str, avg_str, avg_unc_str)
                self.tree.item(item_id, values=val_tuple)
                self.item_to_shape[item_id] = ("circle", (x_px, y_px, r))

        self.main_window.update_image()
        self._autosize_columns()

    # ---------------------------------------------------------------
    def _on_edit_label(self, event):
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return
        x, y, width, height = self.tree.bbox(item_id, "#0")
        entry = tk.Entry(self.tree)
        entry.place(x=x, y=y, width=width, height=height)
        entry.insert(0, self.tree.item(item_id, "text"))
        entry.focus()

        old_text = self.tree.item(item_id, "text")
        item_type = self.item_to_shape.get(item_id, (None,))[0]

        def save_edit(event=None):
            new_text = entry.get()
            # Update TreeView
            self.tree.item(item_id, text=new_text)
            # Update cached results so CSV export reflects the rename
            if item_type == "film":
                for rec in self.results:
                    if rec["film"] == old_text:
                        rec["film"] = new_text
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

    # ---------------------------------------------------------------
    def start_detection(self):
        if not self.image_processor.has_image():
            messagebox.showwarning("AutoMeasurements", "Load an image first.")
            return

        # Always use original RGB image for detection, irrespective of dose calibration
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

        # 1. Detect rectangular films via contour approx
        films = self._detect_films(gray)
        if not films:
            messagebox.showinfo("AutoMeasurements", "No films detected.")
            return

        self.tree.delete(*self.tree.get_children())
        self.results.clear()
        global _OVERLAY
        _OVERLAY = {"films": films, "circles": [], "_shape": img_rgb.shape[:2], "scale": 1.0}

        # Map TreeView items to shapes for highlighting
        self.item_to_shape = {}

        for idx, (x, y, w, h) in enumerate(films, 1):
            film_roi = gray[y : y + h, x : x + w]
            film_name = f"RC_{idx}"
            film_id = self.tree.insert("", "end", text=film_name, values=("", "", "", ""))
            self.item_to_shape[film_id] = ("film", (x, y, w, h))

            # 2. Detect circles inside film
            min_radius = self.min_circle_var.get()
            max_radius = self.max_circle_var.get()
            min_dist = self.min_dist_var.get()
            param1 = self.param1_var.get()
            param2 = self.param2_var.get()
            restrict_diameter = self.restrict_diameter_var.get()
            default_diameter = self.default_diameter_var.get()

            # Detect circles and sort them (top → bottom, left → right)
            detected_circles = self._detect_circles(film_roi)
            detected_circles = sorted(detected_circles, key=lambda c: (c[1], c[0]))

            # Keep copy of original radii for later restoration
            original_circles = detected_circles.copy()

            # Apply diameter restriction if requested
            circles = detected_circles
            if restrict_diameter and circles:
                default_radius = max(1, int(round(default_diameter / 2)))  # integer radius
                circles = [(cx, cy, default_radius) for (cx, cy, _r) in detected_circles]
            for jdx, (cx, cy, adj_r) in enumerate(circles, 1):
                abs_cx = x + cx
                abs_cy = y + cy
                r_int = int(round(adj_r))  # radius used for measurement/drawing
                orig_r_int = int(round(original_circles[jdx - 1][2]))

                # Temporarily set measurement size to current radius
                prev_size = self.image_processor.measurement_size
                try:
                    self.image_processor.measurement_size = r_int
                    res = self.image_processor.measure_area(
                        abs_cx * self.image_processor.zoom,
                        abs_cy * self.image_processor.zoom,
                    )
                finally:
                    self.image_processor.measurement_size = prev_size
                if res is None:
                    continue
                dose, _, unc, _ = res  # mean(s), std, uncertainty
                circ_name = f"C{jdx}"

                # ------------------------------------------------------------------
                # Format dose / uncertainty so that values have the same number of
                # decimal places as their corresponding uncertainties.  We keep two
                # significant figures for the uncertainty by default.
                # ------------------------------------------------------------------
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

                # Average strings
                avg_str, avg_unc_str = self._format_val_unc(avg_val, avg_unc, 2)

                val_tuple = (dose_str, unc_str, avg_str, avg_unc_str)
                circle_id = self.tree.insert(film_id, "end", text=circ_name, values=val_tuple)
                self.item_to_shape[circle_id] = ("circle", (abs_cx, abs_cy, r_int))
                self.original_radii[circle_id] = orig_r_int  # Store original detected radius
                self.results.append({
                    "film": film_name,
                    "circle": circ_name,
                    "dose": dose_str,
                    "unc": unc_str,
                    "avg": avg_str,
                    "avg_unc": avg_unc_str,
                })
                _OVERLAY["circles"].append((abs_cx, abs_cy, r_int))

        # Open all film nodes
        for child in self.tree.get_children():
            self.tree.item(child, open=True)

        # Force refresh of image to show overlays
        self.main_window.update_image()
        self._autosize_columns()

    # ---------------------------------------------------------------
    def _ensure_uint8(self, img: np.ndarray) -> np.ndarray:
        """Convert any dtype image to uint8 0-255 for OpenCV edge detection."""
        if img.dtype == np.uint8:
            return img
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img_norm.astype(np.uint8)

    def _detect_films(self, gray) -> List[Tuple[int, int, int, int]]:
        """Detect rectangular films using fixed threshold like interactivov10."""
        gray8 = self._ensure_uint8(gray)

        # Fixed inverse threshold; films darker than ~180 (on 0-255)
        thresh_val = self.rc_thresh_var.get()
        _, thresh = cv2.threshold(gray8, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # Find contours
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

    def _create_blob_detector(self, min_area: float, max_area: float):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByColor = False
        # OpenCV API difference
        try:
            return cv2.SimpleBlobDetector_create(params)
        except AttributeError:
            return cv2.SimpleBlobDetector(params)

    def _detect_circles(self, roi) -> List[Tuple[int, int, int]]:
        """Detect circles via Hough like interactivov10."""
        min_r = self.min_circle_var.get()
        max_r = self.max_circle_var.get()
        roi8 = self._ensure_uint8(roi)
        roi_blur = cv2.medianBlur(roi8, 5)

        # Use defaults similar to reference script if user hasn't set sensible radii
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
            circles = circles.astype(int)  # avoid uint16 overflow when computing distances
            # Keep all detected circles – allow overlaps
            result = [tuple(c) for c in circles]
        return result

    # ---------------------------------------------------------------
    def _clear_overlay(self):
        global _OVERLAY
        _OVERLAY = None

    def export_csv(self):
        if not self.results:
            messagebox.showinfo("AutoMeasurements", "No data to export.")
            return
        file_path = filedialog.asksaveasfilename(title="Save CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not file_path:
            return
        try:
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["RC", "Circle", "Dose", "Uncertainty", "Avg", "Avg uncert."])
                for rec in self.results:
                    writer.writerow([
                        rec["film"],
                        rec["circle"],
                        rec["dose"],
                        rec["unc"],
                        rec["avg"],
                        rec["avg_unc"],
                    ])
            messagebox.showinfo("AutoMeasurements", "CSV exported.")
        except Exception as exc:
            messagebox.showerror("AutoMeasurements", str(exc))

    # ------------------ new helpers ------------------
    def _clear_all(self):
        """Remove all detections and overlays"""
        self.tree.delete(*self.tree.get_children())
        self.item_to_shape.clear()
        self.results.clear()
        self._clear_overlay()
        self.main_window.update_image()

    def _on_delete_key(self, event):
        """Delete selected film/circle (Supr key)."""
        sel = self.tree.selection()
        if not sel:
            return
        for item_id in list(sel):
            self._delete_item_recursive(item_id)
        self.main_window.update_image()

    def _delete_item_recursive(self, item_id):
        """Remove item from tree, overlay, and results."""
        # First delete children recursively
        for child in self.tree.get_children(item_id):
            self._delete_item_recursive(child)

        # Remove overlay & results for this item
        if item_id in self.item_to_shape:
            shape_type, coords = self.item_to_shape.pop(item_id)
            if _OVERLAY is not None:
                if shape_type == "film":
                    if coords in _OVERLAY.get("films", []):
                        _OVERLAY["films"].remove(coords)
                elif shape_type == "circle":
                    if coords in _OVERLAY.get("circles", []):
                        _OVERLAY["circles"].remove(coords)

            # Remove from results list if circle
            if shape_type == "circle":
                cx, cy, r = coords
                # results keyed by circle name – easier: drop by tree text
                circ_name = self.tree.item(item_id, "text")
                film_item = self.tree.parent(item_id)
                film_name = self.tree.item(film_item, "text") if film_item else None
                self.results = [rec for rec in self.results if not (rec["film"] == film_name and rec["circle"] == circ_name)]

        # Finally delete from TreeView
        self.tree.delete(item_id)

    # Right-click 3D view
    def _on_right_click(self, event):
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return
        if item_id in self.item_to_shape and self.item_to_shape[item_id][0] == "circle":
            _, (cx, cy, r) = self.item_to_shape[item_id]
            self._show_circle_3d(cx, cy, r)

    def _show_circle_3d(self, cx: int, cy: int, r: int):
        """Open matplotlib window with 3-D surface of circle dose or intensity."""
        # Prefer calibrated dose data for plotting when available
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

        # Crop ROI with small margin
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(img.shape[1] - 1, cx + r)
        y2 = min(img.shape[0] - 1, cy + r)
        roi = img[y1 : y2 + 1, x1 : x2 + 1]

        # Prepare grid
        h, w = roi.shape[:2]
        X, Y = np.meshgrid(np.arange(w), np.arange(h))

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection="3d")

        if roi.ndim == 3 and roi.shape[2] >= 3:
            # Average across channels (RGB dose/intensity)
            Z = np.mean(roi, axis=2)
        else:
            Z = roi.copy()

        # Clip outliers (2nd–98th percentiles) to avoid extreme peaks
        lo, hi = np.percentile(Z, [2, 98])
        Z = np.clip(Z, lo, hi)

        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
        ax.set_title(plot_title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel(z_label)

        plt.show()

    # Manual addition ------------------------------------------------------
    def _add_manual_film(self):
        """Open dynamic dimension window and start film draw mode (no dialogs)."""
        # sensible defaults
        self._start_draw_mode("film", (300, 200))
        return  # Skip additional dialogs; dimensions can be edited live in the popup
        """Begin interactive drawing mode to add an RC rectangle (film)."""
        from tkinter import simpledialog

        # Ask for rectangle dimensions (in original image pixels)
        w = simpledialog.askinteger("Add RC", "Enter rectangle width (pixels):", minvalue=1)
        if w is None:
            return
        h = simpledialog.askinteger("Add RC", "Enter rectangle height (pixels):", minvalue=1)
        if h is None:
            return

        self._start_draw_mode("film", (w, h))
        return  # Exit early to skip deprecated manual-add code
        from tkinter import simpledialog
        ans = simpledialog.askstring("Add RC", "Enter rectangle x,y,w,h:")
        if not ans:
            return
        try:
            x, y, w, h = map(int, ans.replace(" ", "").split(","))
        except Exception:
            messagebox.showerror("Add RC", "Invalid input. Use: x,y,w,h")
            return

        global _OVERLAY
        if _OVERLAY is None:
            _OVERLAY = {"films": [], "circles": [], "_shape": self.image_processor.current_image.shape[:2]}

        film_name = f"RC_{len(_OVERLAY['films'])+1}M"
        film_id = self.tree.insert("", "end", text=film_name, values=("", "", "", ""))
        self.item_to_shape[film_id] = ("film", (x, y, w, h))
        _OVERLAY["films"].append((x, y, w, h))
        self.tree.item(film_id, open=True)
        self.main_window.update_image()

    def _add_manual_circle(self):
        """Open dynamic dimension window and start circle draw mode (no dialogs)."""
        self._start_draw_mode("circle", 100)
        return  # Skip additional dialogs; radius can be edited live in the popup
        """Begin interactive drawing mode to add a circle ROI."""
        from tkinter import simpledialog

        r = simpledialog.askinteger("Add Circle", "Enter circle radius (pixels):", minvalue=1)
        if r is None:
            return

        self._start_draw_mode("circle", r)

    # ------------------------------------------------------------------
    # Interactive drawing helpers
    # ------------------------------------------------------------------
    def _start_draw_mode(self, shape_type: str, dims):
        """Activate drawing mode on the image canvas.

        Parameters
        ----------
        shape_type : str
            'film' or 'circle'.
        dims : tuple | int
            (w, h) for film or radius for circle.
        """
        if self._canvas is None or not self.image_processor.has_image():
            messagebox.showwarning("Draw", "No image loaded or canvas unavailable.")
            return

        # If we were already in a drawing mode, cancel it first
        self._cancel_draw()

        self.draw_mode = shape_type
        self.draw_dims = dims

        # Open a small dynamic window to tweak dimensions live
        self._open_dims_window()

        # Configure canvas for drawing
        self._canvas.config(cursor="crosshair")
        # Remove any previous preview remains
        self._canvas.delete(self.preview_tag)
        # Bind events
        self._canvas.bind("<Motion>", self._on_draw_move)
        self._canvas.bind("<Button-1>", self._on_draw_click)
        # Bind ESC on the top-level window
        self.frame.winfo_toplevel().bind("<Escape>", self._cancel_draw)

    def _open_dims_window(self):
        """Create a small top-level window with live-update dimension entries."""
        # Destroy previous if exists
        if self._dims_window is not None:
            self._dims_window.destroy()
        self._dims_window = tk.Toplevel(self.frame)
        self._dims_window.title("Dimensions")
        self._dims_window.resizable(False, False)
        self._dims_window.attributes("-topmost", True)
        # Clear vars list
        self._dim_vars.clear()

        def on_var_change(*_):
            """Callback when user edits dimension entries – update draw_dims and preview."""
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
                # Ignore non-numeric inputs while typing
                pass
            self._update_preview()

        if self.draw_mode == "film":
            w, h = self.draw_dims  # type: ignore[misc]
            w_var = tk.StringVar(value=str(w))
            h_var = tk.StringVar(value=str(h))
            self._dim_vars.extend([w_var, h_var])
            tk.Label(self._dims_window, text="Width:").grid(row=0, column=0, padx=4, pady=2)
            tk.Entry(self._dims_window, textvariable=w_var, width=6).grid(row=0, column=1, pady=2)
            tk.Label(self._dims_window, text="Height:").grid(row=1, column=0, padx=4, pady=2)
            tk.Entry(self._dims_window, textvariable=h_var, width=6).grid(row=1, column=1, pady=2)
        elif self.draw_mode == "circle":
            r = self.draw_dims  # type: ignore[assignment]
            r_var = tk.StringVar(value=str(r))
            self._dim_vars.append(r_var)
            tk.Label(self._dims_window, text="Radius:").grid(row=0, column=0, padx=4, pady=2)
            tk.Entry(self._dims_window, textvariable=r_var, width=6).grid(row=0, column=1, pady=2)

        for var in self._dim_vars:
            var.trace_add("write", on_var_change)

    def _update_preview(self):
        """Redraw preview using last known cursor position."""
        if self.draw_mode is None or self._last_cursor is None:
            return
        dummy_evt = type("_e", (), {"x": self._last_cursor[0], "y": self._last_cursor[1]})()
        self._on_draw_move(dummy_evt)

    def _on_draw_move(self, event):
        """Update dynamic preview while the cursor moves."""
        if self.draw_mode is None:
            return

        canvas = self._canvas
        canvas.delete(self.preview_tag)

        # Current canvas coordinates
        x = canvas.canvasx(event.x)
        y = canvas.canvasy(event.y)
        zoom = self.image_processor.zoom or 1.0

        if self.draw_mode == "film":
            w, h = self.draw_dims  # type: ignore[misc]
            dx = (w * zoom) / 2
            dy = (h * zoom) / 2
            canvas.create_rectangle(
                x - dx,
                y - dy,
                x + dx,
                y + dy,
                outline="yellow",
                dash=(4, 2),
                tags=self.preview_tag,
            )
        elif self.draw_mode == "circle":
            r = self.draw_dims  # type: ignore[assignment]
            dr = r * zoom
            canvas.create_oval(
                x - dr,
                y - dr,
                x + dr,
                y + dr,
                outline="yellow",
                dash=(4, 2),
                tags=self.preview_tag,
            )

    def _on_draw_click(self, event):
        """Finalize placement of the current previewed shape on left click."""
        if self.draw_mode is None:
            return

        canvas = self._canvas
        # Canvas coordinates where the user clicked (display coords)
        x_canvas = canvas.canvasx(event.x)
        y_canvas = canvas.canvasy(event.y)
        zoom = self.image_processor.zoom or 1.0

        if self.draw_mode == "film":
            # Rectangle drawn centred at click position
            w, h = self.draw_dims  # type: ignore[misc]
            x_top = int(x_canvas / zoom - w / 2)
            y_top = int(y_canvas / zoom - h / 2)
            self._insert_film(x_top, y_top, w, h)
        elif self.draw_mode == "circle":
            r = self.draw_dims  # type: ignore[assignment]
            cx = int(x_canvas / zoom)
            cy = int(y_canvas / zoom)
            self._insert_circle(cx, cy, r)

        # Clean-up drawing state
        self._cancel_draw()

    def _cancel_draw(self, event=None):
        """Cancel current drawing operation and clean up UI bindings."""
        canvas = getattr(self, "_canvas", None)
        if canvas is not None:
            canvas.delete(self.preview_tag)
            canvas.config(cursor="")
            canvas.unbind("<Motion>")
            canvas.unbind("<Button-1>")

        # Unbind ESC key
        try:
            self.frame.winfo_toplevel().unbind("<Escape>")
        except Exception:
            pass

        # Close dimension window if open
        if getattr(self, "_dims_window", None) is not None:
            self._dims_window.destroy()
            self._dims_window = None
        # Reset helpers
        self._dim_vars.clear()
        self._last_cursor = None
        self.draw_mode = None
        self.draw_dims = None

    # ---------------- Insertion helpers -------------------
    def _insert_film(self, x: int, y: int, w: int, h: int):
        """Insert a film rectangle into overlay and TreeView."""
        global _OVERLAY
        if _OVERLAY is None:
            _OVERLAY = {
                "films": [],
                "circles": [],
                "_shape": self.image_processor.current_image.shape[:2],
            }

        # Determine next index for naming
        film_idx = len([v for v in self.item_to_shape.values() if v[0] == "film"]) + 1
        film_name = f"RC_{film_idx}M"
        film_id = self.tree.insert("", "end", text=film_name, values=("", "", "", ""))
        self.item_to_shape[film_id] = ("film", (x, y, w, h))
        _OVERLAY["films"].append((x, y, w, h))
        self.tree.item(film_id, open=True)
        self.main_window.update_image()

    def _insert_circle(self, cx: int, cy: int, r: int):
        """Insert a circle ROI. Overlap allowed – previous restriction removed."""

        global _OVERLAY
        if _OVERLAY is None:
            _OVERLAY = {
                "films": [],
                "circles": [],
                "_shape": self.image_processor.current_image.shape[:2],
            }

        # Find parent film that contains the circle centre
        parent_id = ""
        for fid, (item_type, coords) in self.item_to_shape.items():
            if item_type != "film":
                continue
            fx, fy, fw, fh = coords
            if (fx <= cx <= fx + fw) and (fy <= cy <= fy + fh):
                parent_id = fid
                break

        circ_idx = len([v for v in self.item_to_shape.values() if v[0] == "circle"]) + 1
        circ_name = f"C{circ_idx}M"
        circ_id = self.tree.insert(parent_id, "end", text=circ_name, values=("", "", "", ""))
        self.item_to_shape[circ_id] = ("circle", (cx, cy, r))
        _OVERLAY["circles"].append((cx, cy, r))
        if parent_id:
            self.tree.item(parent_id, open=True)
        self.main_window.update_image()
        return  # Exit early to skip deprecated manual-add code

    # ---------------------------------------------------------------
    def _autosize_columns(self):
        """Resize TreeView columns so they fit their widest cell contents."""
        # Fonts for measurement
        try:
            heading_font = tkfont.nametofont("TkHeadingFont")
        except Exception:
            heading_font = tkfont.nametofont("TkDefaultFont")
        body_font = tkfont.nametofont("TkDefaultFont")

        # Start with the width of the header texts
        widths: dict[str, int] = {
            "#0": heading_font.measure(self.tree.heading("#0", "text") or "")
        }
        for col in self.tree["columns"]:
            widths[col] = heading_font.measure(self.tree.heading(col, "text"))

        # Recursive traversal to measure each cell
        def measure_item(item_id: str):
            # Tree text
            widths["#0"] = max(widths["#0"], body_font.measure(self.tree.item(item_id, "text")))
            # Other columns
            for col in self.tree["columns"]:
                val = self.tree.set(item_id, col)
                widths[col] = max(widths[col], body_font.measure(val))
            # Children
            for child in self.tree.get_children(item_id):
                measure_item(child)

        for root in self.tree.get_children(""):
            measure_item(root)

        # Apply widths with a bit of padding
        padding = 16  # pixels
        for col, w in widths.items():
            self.tree.column(col, width=w + padding)
