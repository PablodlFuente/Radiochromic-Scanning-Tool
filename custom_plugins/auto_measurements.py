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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3D

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

class AutoMeasurementsTab:
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

    def __init__(self, main_window, notebook, image_processor):
        self.item_to_shape: dict[str, tuple] = {}  # maps TreeView items to shape info
        self.main_window = main_window
        self.image_processor = image_processor
        self.frame = ttk.Frame(notebook)

        # UI
        ttk.Label(self.frame, text="Auto Measurements", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))

        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill=tk.X, padx=10)
        ttk.Button(btn_frame, text="Start", command=self.start_detection).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Clear", command=self._clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Add Film", command=self._add_manual_film).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Add Circle", command=self._add_manual_circle).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Export to CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5)

        # ----------------- Detection Parameter Panels -----------------
        # Bind selection event for highlighting
        # TreeView columns – add avg columns if calibration active
        cols = ["dose", "sigma"]
        if image_processor.calibration_applied:
            cols.extend(["avg", "avg_unc"])

        self.tree = ttk.Treeview(self.frame, columns=tuple(cols), show="tree headings")
        self.tree.heading("dose", text="Dose")
        self.tree.heading("sigma", text="Uncertainty")
        self.tree.column("dose", width=80, anchor=tk.CENTER)
        self.tree.column("sigma", width=100, anchor=tk.CENTER)
        if "avg" in cols:
            self.tree.heading("avg", text="Avg")
            self.tree.heading("avg_unc", text="±Avg")
            self.tree.column("avg", width=80, anchor=tk.CENTER)
            self.tree.column("avg_unc", width=80, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Delete>", self._on_delete_key)
        self.tree.bind("<Button-3>", self._on_right_click)
        self.tree.bind("<Double-1>", self._on_edit_label)

        # Film detection parameters
        self.rc_thresh_var = tk.IntVar(value=180)
        self.rc_min_area_var = tk.IntVar(value=5000)
        film_frame = ttk.LabelFrame(self.frame, text="Film Detection")
        film_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(film_frame, text="Threshold (0-255):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(film_frame, textvariable=self.rc_thresh_var, width=6).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(film_frame, text="Min area (px):").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(film_frame, textvariable=self.rc_min_area_var, width=8).grid(row=1, column=1, sticky=tk.W)

        # Circle detection parameters
        self.min_circle_var = tk.IntVar(value=100)
        self.max_circle_var = tk.IntVar(value=300)
        # Minimum distance between detected circle centers
        self.min_dist_var = tk.IntVar(value=30)
        self.param1_var = tk.IntVar(value=50)  # Canny high threshold
        self.param2_var = tk.IntVar(value=30)  # Accumulator threshold
        circle_frame = ttk.LabelFrame(self.frame, text="Circle Detection")
        circle_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(circle_frame, text="Min radius:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.min_circle_var, width=6).grid(row=0, column=1)
        ttk.Label(circle_frame, text="Max radius:").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.max_circle_var, width=6).grid(row=0, column=3)
        ttk.Label(circle_frame, text="MinDist:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.min_dist_var, width=6).grid(row=1, column=1)
        ttk.Label(circle_frame, text="Param1:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.param1_var, width=6).grid(row=2, column=1)
        ttk.Label(circle_frame, text="Param2:").grid(row=2, column=2, sticky=tk.W)
        ttk.Entry(circle_frame, textvariable=self.param2_var, width=6).grid(row=2, column=3)

        # Enable inline editing of item text (film/circle names)
        # Storage
        self.results = []  # List[Dict]

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

        def save_edit(event=None):
            self.tree.item(item_id, text=entry.get())
            entry.destroy()
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
            film_name = f"Film_{idx}"
            film_id = self.tree.insert("", "end", text=film_name, values=("", ""))
            self.item_to_shape[film_id] = ("film", (x, y, w, h))

            # 2. Detect circles inside film
            circles = self._detect_circles(film_roi)
            for jdx, (cx, cy, r) in enumerate(circles, 1):
                abs_cx = x + cx
                abs_cy = y + cy
                # Use existing measure_area
                res = self.image_processor.measure_area(abs_cx * self.image_processor.zoom, abs_cy * self.image_processor.zoom)
                if res is None:
                    continue
                dose, _, unc, _ = res  # mean(s), std, uncertainty
                circ_name = f"C{jdx}"

                # Format numbers for display; handle RGB tuples or scalars
                if isinstance(dose, tuple):
                    dose_str = ", ".join(f"{v:.2f}" for v in dose)
                else:
                    dose_str = f"{dose:.2f}"

                if isinstance(unc, tuple):
                    unc_str = ", ".join(f"±{v:.2f}" for v in unc)
                else:
                    unc_str = f"±{unc:.2f}"

                if self.image_processor.calibration_applied and isinstance(dose, tuple):
                    avg_val = np.mean(dose)
                    avg_unc = np.mean(unc) if isinstance(unc, tuple) else unc
                    avg_str = f"{avg_val:.2f}"
                    avg_unc_str = f"±{avg_unc:.2f}"
                    val_tuple = (dose_str, unc_str, avg_str, avg_unc_str)
                else:
                    val_tuple = (dose_str, unc_str)

                circle_id = self.tree.insert(film_id, "end", text=circ_name, values=val_tuple)
                self.item_to_shape[circle_id] = ("circle", (abs_cx, abs_cy, r))
                self.results.append({"film": film_name, "circle": circ_name, "dose": dose_str, "unc": unc_str})
                _OVERLAY["circles"].append((abs_cx, abs_cy, r))

        # Open all film nodes
        for child in self.tree.get_children():
            self.tree.item(child, open=True)

        # Force refresh of image to show overlays
        self.main_window.update_image()

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
            # Remove overlaps similar to reference code
            circles = sorted(circles, key=lambda c: c[2], reverse=True)
            valid = []
            for cx, cy, r in circles:
                overlap = False
                for vx, vy, vr in valid:
                    if np.hypot(cx - vx, cy - vy) < (r + vr):
                        overlap = True
                        break
                if not overlap:
                    valid.append((cx, cy, r))
            result = valid
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
                writer.writerow(["Film", "Circle", "Dose", "Uncertainty"])
                for rec in self.results:
                    writer.writerow([rec["film"], rec["circle"], rec["dose"], rec["unc"]])
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
        """Open matplotlib window with 3-D surface of circle RGB intensities (or dose)."""
        img = (
            self.image_processor.original_image
            if getattr(self.image_processor, "original_image", None) is not None
            else self.image_processor.current_image
        )
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

        if roi.ndim == 3 and roi.shape[2] == 3:
            # Average intensity or dose if calibrated
            Z = np.mean(roi, axis=2)
        else:
            Z = roi.copy()

        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
        ax.set_title("3D View Circle")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Value")

        plt.show()

    # Manual addition ------------------------------------------------------
    def _add_manual_film(self):
        from tkinter import simpledialog
        ans = simpledialog.askstring("Add Film", "Enter rectangle x,y,w,h:")
        if not ans:
            return
        try:
            x, y, w, h = map(int, ans.replace(" ", "").split(","))
        except Exception:
            messagebox.showerror("Add Film", "Invalid input. Use: x,y,w,h")
            return

        global _OVERLAY
        if _OVERLAY is None:
            _OVERLAY = {"films": [], "circles": [], "_shape": self.image_processor.current_image.shape[:2]}

        film_name = f"Film_{len(_OVERLAY['films'])+1}M"
        film_id = self.tree.insert("", "end", text=film_name, values=("", ""))
        self.item_to_shape[film_id] = ("film", (x, y, w, h))
        _OVERLAY["films"].append((x, y, w, h))
        self.tree.item(film_id, open=True)
        self.main_window.update_image()

    def _add_manual_circle(self):
        from tkinter import simpledialog
        ans = simpledialog.askstring("Add Circle", "Enter circle cx,cy,r:")
        if not ans:
            return
        try:
            cx, cy, r = map(int, ans.replace(" ", "").split(","))
        except Exception:
            messagebox.showerror("Add Circle", "Invalid input. Use: cx,cy,r")
            return

        # Ensure circle does not overlap existing ones
        for _, (scx, scy, sr) in [v for k, v in self.item_to_shape.items() if v[0] == "circle"]:
            if np.hypot(cx - scx, cy - scy) < (r + sr):
                messagebox.showwarning("Add Circle", "Circle overlaps with existing circle.")
                return

        global _OVERLAY
        if _OVERLAY is None:
            _OVERLAY = {"films": [], "circles": [], "_shape": self.image_processor.current_image.shape[:2]}

        # Link to nearest film if any
        parent_id = ""
        for fid, (ftype, (fx, fy, fw, fh)) in self.item_to_shape.items():
            if ftype == "film" and (fx <= cx <= fx + fw) and (fy <= cy <= fy + fh):
                parent_id = fid
                break

        circ_idx = len([v for v in self.item_to_shape.values() if v[0] == "circle"]) + 1
        circ_name = f"C{circ_idx}M"
        circ_id = self.tree.insert(parent_id, "end", text=circ_name, values=("", ""))
        self.item_to_shape[circ_id] = ("circle", (cx, cy, r))
        _OVERLAY["circles"].append((cx, cy, r))
        self.tree.item(parent_id, open=True)
        self.main_window.update_image()
