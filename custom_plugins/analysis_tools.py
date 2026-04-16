"""
Analysis Tools plugin – companion tab for AutoMeasurements.

Provides:
1. Gaussian centroid detection: compares geometric centroid vs dose-weighted
   centroid for each detected circle, reporting the offset.
   Four detection methods: Weighted Centroid, Gaussian Power, Radius Equation,
   Phase Correlation.
   Draws a filled dot for geometric centroid and an X for dose-weighted centroid.
   Optional isobar (isodose contour) overlay inside each circle.
2. Dose vs Introduced Value: associate a known value to each detected circle,
   then plot measured dose vs introduced value with error bars, a linear fit
   forced through the origin, and R².
"""

from __future__ import annotations

import sys
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2

logger = logging.getLogger(__name__)

TAB_TITLE = "Analysis"

# Module-level storage for centroid overlay markers
# List of (geo_x, geo_y, dose_x, dose_y) in original image coordinates
_CENTROID_MARKERS: list[tuple[float, float, float, float]] = []

# Module-level storage for isobar contours (smooth polylines in absolute coords)
# List of (contour_array,) — each contour_array is Nx1x2 int32 in absolute image coords
_ISOBAR_CONTOURS: list = []
_SHOW_ISOBARS: bool = False


# ---------------------------------------------------------------------------
# Helpers to reach the AutoMeasurements plugin
# ---------------------------------------------------------------------------

def _get_auto_measurements():
    """Return the AutoMeasurementsTab instance (or None)."""
    mod = sys.modules.get("auto_measurements") or sys.modules.get("custom_plugins.auto_measurements")
    if mod is None:
        return None
    return getattr(mod, "_AUTO_MEASUREMENTS_INSTANCE", None)


def _get_overlay():
    mod = sys.modules.get("auto_measurements") or sys.modules.get("custom_plugins.auto_measurements")
    if mod is None:
        return None
    return getattr(mod, "_OVERLAY", None)


# ---------------------------------------------------------------------------
# Plugin interface
# ---------------------------------------------------------------------------

_INSTANCE = None


def setup(main_window, notebook, image_processor):
    global _INSTANCE
    _INSTANCE = AnalysisTab(main_window, notebook, image_processor)
    return _INSTANCE.frame


def process(image):
    """Draw centroid markers and optional isobars on the image overlay."""
    global _CENTROID_MARKERS, _ISOBAR_CONTOURS, _SHOW_ISOBARS

    has_markers = bool(_CENTROID_MARKERS)
    has_isobars = _SHOW_ISOBARS and bool(_ISOBAR_CONTOURS)

    if not has_markers and not has_isobars:
        return image

    overlay = _get_overlay()
    if overlay is None:
        return image

    orig_h, orig_w = overlay.get("_shape", (0, 0))
    if orig_h == 0 or orig_w == 0:
        return image

    h, w = image.shape[:2]
    sx, sy = w / orig_w, h / orig_h

    # Determine color scale
    if np.issubdtype(image.dtype, np.floating):
        color_scale = 1.0 / 255.0
    elif np.issubdtype(image.dtype, np.integer):
        actual_max = np.max(image)
        dtype_info = np.iinfo(image.dtype)
        theoretical_max = dtype_info.max
        if actual_max > theoretical_max * 0.1:
            max_val = theoretical_max
        else:
            max_val = max(actual_max, 255)
        color_scale = max_val / 255.0
    else:
        color_scale = 1

    def scale_color(bgr):
        if np.issubdtype(image.dtype, np.floating):
            return tuple(c * color_scale for c in bgr)
        else:
            dtype_max = np.iinfo(image.dtype).max
            return tuple(min(int(c * color_scale), dtype_max) for c in bgr)

    # Ensure color output
    if len(image.shape) == 2 or image.shape[2] == 1:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    # --- Draw isobars (smooth contour lines) ---
    if has_isobars:
        isobar_color = scale_color((200, 200, 200))  # light gray
        for contour in _ISOBAR_CONTOURS:
            if contour is None or len(contour) < 3:
                continue
            # Scale contour points to display coordinates
            scaled = contour.copy().astype(np.float64)
            scaled[:, :, 0] *= sx
            scaled[:, :, 1] *= sy
            cv2.polylines(out, [scaled.astype(np.int32)], isClosed=True,
                          color=isobar_color, thickness=1, lineType=cv2.LINE_AA)

    # --- Draw centroid markers ---
    if has_markers:
        cyan = scale_color((255, 255, 0))     # Geometric centroid dot (cyan)
        magenta = scale_color((255, 0, 255))  # Dose centroid X (magenta)

        marker_size = max(4, int(min(w, h) * 0.004))

        for (geo_x, geo_y, dose_x, dose_y) in _CENTROID_MARKERS:
            gx, gy = int(geo_x * sx), int(geo_y * sy)
            dx, dy = int(dose_x * sx), int(dose_y * sy)

            # Geometric centroid: filled circle (dot)
            cv2.circle(out, (gx, gy), marker_size, cyan, -1, lineType=cv2.LINE_AA)

            # Dose-weighted centroid: X mark
            s = marker_size + 1
            cv2.line(out, (dx - s, dy - s), (dx + s, dy + s), magenta, 2, lineType=cv2.LINE_AA)
            cv2.line(out, (dx - s, dy + s), (dx + s, dy - s), magenta, 2, lineType=cv2.LINE_AA)

    return out


def on_config_change(config, image_processor):
    pass


# ---------------------------------------------------------------------------
# Main tab class
# ---------------------------------------------------------------------------

class AnalysisTab:
    def __init__(self, main_window, notebook, image_processor):
        self.main_window = main_window
        self.image_processor = image_processor
        self.frame = ttk.Frame(notebook)

        # Introduced values per circle key: (film, circle) -> float
        self._introduced_values: dict[tuple[str, str], float] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- Top: Centroid analysis ----
        centroid_frame = ttk.LabelFrame(self.frame, text="Centroid Analysis (Geometric vs Dose-Weighted)")
        centroid_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        # Row 1: Method selector + isobars
        method_row = ttk.Frame(centroid_frame)
        method_row.pack(fill=tk.X, padx=6, pady=(4, 2))

        ttk.Label(method_row, text="Method:").pack(side=tk.LEFT, padx=(0, 4))
        self._centroid_method = tk.StringVar(value="weighted_centroid")
        method_combo = ttk.Combobox(method_row, textvariable=self._centroid_method,
                                    values=["weighted_centroid", "gaussian_power",
                                            "radius_equation", "phase_correlation"],
                                    state="readonly", width=18)
        method_combo.pack(side=tk.LEFT, padx=(0, 12))

        # Isobars checkbox
        self._show_isobars_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(method_row, text="Show Isobars",
                        variable=self._show_isobars_var,
                        command=self._on_isobars_toggled).pack(side=tk.RIGHT, padx=4)

        # Row 2: Buttons + legend
        btn_row = ttk.Frame(centroid_frame)
        btn_row.pack(fill=tk.X, padx=6, pady=2)

        ttk.Button(btn_row, text="Compute Centroid Offsets",
                   command=self._compute_centroids).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Clear", command=self._clear_centroid_table).pack(side=tk.LEFT, padx=4)

        # Legend
        legend = ttk.Frame(btn_row)
        legend.pack(side=tk.RIGHT, padx=8)
        ttk.Label(legend, text="\u25CF Geometric", foreground="#00cccc",
                  font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(legend, text="\u2716 Dose-Weighted", foreground="#cc00cc",
                  font=("Arial", 9, "bold")).pack(side=tk.LEFT)

        # Treeview for centroid results (tree structure: Film > Circle)
        cols = ("geo_x", "geo_y", "dose_x", "dose_y", "dx", "dy", "dist")
        self.centroid_tree = ttk.Treeview(centroid_frame, columns=cols, show="tree headings", height=8)
        self.centroid_tree.heading("#0", text="Film / Circle", anchor=tk.W)
        self.centroid_tree.column("#0", width=110, minwidth=80)
        for c, w, text in [
            ("geo_x", 65, "Geo X"), ("geo_y", 65, "Geo Y"),
            ("dose_x", 65, "Dose X"), ("dose_y", 65, "Dose Y"),
            ("dx", 55, "ΔX"), ("dy", 55, "ΔY"), ("dist", 60, "Dist (px)")
        ]:
            self.centroid_tree.heading(c, text=text)
            self.centroid_tree.column(c, width=w, anchor=tk.CENTER)

        self.centroid_tree.pack(fill=tk.BOTH, expand=False, padx=6, pady=(0, 6))

        # ---- Bottom: Dose vs Introduced Value ----
        dose_frame = ttk.LabelFrame(self.frame, text="Dose vs Introduced Value")
        dose_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        # Instructions
        ttk.Label(dose_frame,
                  text="Load detections from AutoMeasurements, assign a known value to each circle, then plot.",
                  foreground="gray", font=("Arial", 9)).pack(anchor=tk.W, padx=6, pady=(4, 2))

        btn_row2 = ttk.Frame(dose_frame)
        btn_row2.pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(btn_row2, text="Load Circles",
                   command=self._load_circles).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row2, text="Plot Dose vs Value",
                   command=self._plot_dose_vs_value).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row2, text="Clear", command=self._clear_dose_table).pack(side=tk.LEFT, padx=4)

        # Editable treeview
        cols2 = ("film", "circle", "dose", "se", "value")
        self.dose_tree = ttk.Treeview(dose_frame, columns=cols2, show="headings", height=10)
        for c, w, text in [
            ("film", 80, "Film"), ("circle", 65, "Circle"),
            ("dose", 90, "Dose (Gy)"), ("se", 80, "SE (Gy)"),
            ("value", 90, "Introduced Value")
        ]:
            self.dose_tree.heading(c, text=text)
            self.dose_tree.column(c, width=w, anchor=tk.CENTER)

        self.dose_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # Double-click to edit the "Introduced Value" column
        self.dose_tree.bind("<Double-1>", self._on_dose_tree_double_click)

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------
    def _on_isobars_toggled(self):
        global _SHOW_ISOBARS
        _SHOW_ISOBARS = self._show_isobars_var.get()
        self.main_window.update_image()

    # ------------------------------------------------------------------
    # Centroid analysis — helpers
    # ------------------------------------------------------------------
    def _extract_circle_roi(self, dose_2d, cx, cy, radius):
        """Extract a circular ROI from the dose image.

        Returns (roi, xx, yy, mask, x_min, y_min) or None if empty.
        roi has NaN outside the circle.
        """
        h, w = dose_2d.shape
        y_min = max(0, int(cy - radius))
        y_max = min(h, int(cy + radius + 1))
        x_min = max(0, int(cx - radius))
        x_max = min(w, int(cx + radius + 1))

        roi = dose_2d[y_min:y_max, x_min:x_max].copy()
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        mask = ((xx - cx)**2 + (yy - cy)**2) <= radius**2
        roi[~mask] = np.nan

        if not np.any(~np.isnan(roi)):
            return None
        return roi, xx, yy, mask, x_min, y_min

    def _find_radius(self, item_to_shape, am, cx, cy):
        """Find circle radius from overlay or original_radii."""
        for sid, (stype, scoords) in item_to_shape.items():
            if stype == "circle":
                sx, sy, sr = scoords
                if abs(sx - cx) < 2 and abs(sy - cy) < 2:
                    return sr
        for rid, rv in getattr(am, 'original_radii', {}).items():
            shape = item_to_shape.get(rid)
            if shape and shape[0] == "circle":
                sx, sy, _ = shape[1]
                if abs(sx - cx) < 2 and abs(sy - cy) < 2:
                    return rv
        return 30  # fallback

    # ------------------------------------------------------------------
    # Method 1: Weighted Centroid (dose-weighted center of mass)
    # ------------------------------------------------------------------
    def _centroid_weighted(self, roi, xx, yy, mask):
        """Simple dose-weighted centroid (center of mass)."""
        valid = mask & ~np.isnan(roi)
        zs = roi[valid].astype(np.float64)
        xs = xx[valid].astype(np.float64)
        ys = yy[valid].astype(np.float64)
        d_min = np.min(zs)
        w = zs - d_min + 1e-9
        total = np.sum(w)
        return np.sum(xs * w) / total, np.sum(ys * w) / total

    # ------------------------------------------------------------------
    # Method 2: Gaussian Power (auto-converging)
    # ------------------------------------------------------------------
    def _centroid_gaussian_power(self, roi, xx, yy, mask):
        """Normalize dose to [0,1] and raise to increasing power n until
        the centroid position converges.  Returns (cx, cy) or None if
        convergence is not achieved.
        """
        valid = mask & ~np.isnan(roi)
        zs = roi[valid].astype(np.float64)
        xs = xx[valid].astype(np.float64)
        ys = yy[valid].astype(np.float64)

        z_min, z_max = np.min(zs), np.max(zs)
        span = z_max - z_min
        if span < 1e-12:
            return None  # flat — cannot determine centroid

        normalized = (zs - z_min) / span  # [0, 1]

        prev_cx, prev_cy = None, None
        tol = 0.05  # convergence tolerance in pixels

        for n in range(2, 101):
            powered = normalized ** n
            total = np.sum(powered)
            if total < 1e-30:
                break  # underflow — stop
            cx = np.sum(xs * powered) / total
            cy = np.sum(ys * powered) / total

            if prev_cx is not None:
                delta = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                if delta < tol:
                    return cx, cy  # converged

            prev_cx, prev_cy = cx, cy

        # Did not converge strictly but return last valid position
        if prev_cx is not None:
            return prev_cx, prev_cy
        return None

    # ------------------------------------------------------------------
    # Method 3: Radius Equation (radial-weighted centroid)
    # ------------------------------------------------------------------
    def _centroid_radius_equation(self, roi, xx, yy, mask, cx, cy):
        """Weight pixels by dose / (radial_distance + 1).

        Pixels far from center with high dose pull the centroid more.
        """
        valid = mask & ~np.isnan(roi)
        xs = xx[valid].astype(np.float64)
        ys = yy[valid].astype(np.float64)
        zs = roi[valid].astype(np.float64)

        d_min = np.min(zs)
        w = zs - d_min + 1e-9
        r = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        rw = w / (r + 1.0)
        total = np.sum(rw)

        return np.sum(xs * rw) / total, np.sum(ys * rw) / total

    # ------------------------------------------------------------------
    # Method 4: Phase Correlation (Fourier shift theorem)
    # ------------------------------------------------------------------
    def _centroid_phase_correlation(self, roi, xx, yy, mask, cx, cy, radius):
        """Use phase correlation against a synthetic centered Gaussian
        to measure the sub-pixel displacement of the dose peak.

        A reference Gaussian is generated at the center of a square patch
        the same size as the ROI.  cv2.phaseCorrelate returns the (dx, dy)
        shift between the reference and the real data.
        """
        valid = mask & ~np.isnan(roi)
        if np.sum(valid) < 10:
            return None

        h_roi, w_roi = roi.shape

        # Prepare the real image: fill NaN with minimum, normalize to [0,1]
        z_min = np.nanmin(roi)
        z_max = np.nanmax(roi)
        span = z_max - z_min
        if span < 1e-12:
            return None

        real_img = np.where(np.isnan(roi), z_min, roi)
        real_img = ((real_img - z_min) / span).astype(np.float64)

        # Generate synthetic Gaussian centered at the patch center
        center_y, center_x = h_roi / 2.0, w_roi / 2.0
        # Estimate sigma from the ROI radius (~radius/2 is a reasonable fit)
        sigma = radius * 0.5

        yy_local = np.arange(h_roi, dtype=np.float64)
        xx_local = np.arange(w_roi, dtype=np.float64)
        yg, xg = np.meshgrid(yy_local, xx_local, indexing='ij')
        ref_img = np.exp(-(((xg - center_x)**2 + (yg - center_y)**2) / (2 * sigma**2)))

        # Apply circular mask to both (same mask shape)
        local_mask = mask[: h_roi, : w_roi] if mask.shape == roi.shape else np.ones_like(roi, dtype=bool)
        # Actually rebuild mask in local coords
        yy_l, xx_l = np.mgrid[0:h_roi, 0:w_roi]
        local_mask = ((xx_l - center_x)**2 + (yy_l - center_y)**2) <= radius**2
        real_img[~local_mask] = 0
        ref_img[~local_mask] = 0

        # Apply Hanning window to reduce edge effects
        hann = cv2.createHanningWindow((w_roi, h_roi), cv2.CV_64F)
        real_windowed = real_img * hann
        ref_windowed = ref_img * hann

        # Phase correlate returns (dx, dy) shift from ref to real
        (shift_x, shift_y), response = cv2.phaseCorrelate(ref_windowed, real_windowed)

        # The geometric center of the ROI in absolute coords
        x_min = int(cx - radius)
        y_min = int(cy - radius)
        # Reference Gaussian is at (center_x, center_y) in local coords
        # = (x_min + center_x, y_min + center_y) in absolute coords
        # The real peak is shifted by (shift_x, shift_y) from the reference
        dose_cx = x_min + center_x + shift_x
        dose_cy = y_min + center_y + shift_y

        return dose_cx, dose_cy

    # ------------------------------------------------------------------
    # Isobar contour extraction (smooth continuous lines)
    # ------------------------------------------------------------------
    def _extract_isobars(self, dose_2d, cx, cy, radius, n_levels=6):
        """Compute smooth isodose contour lines inside a circle.

        Uses matplotlib contour generator for smooth curves, then clips
        to the circular boundary.  Returns list of Nx1x2 int32 arrays
        in absolute image coordinates.
        """
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend (safe for thread)
        import matplotlib.pyplot as plt

        extraction = self._extract_circle_roi(dose_2d, cx, cy, radius)
        if extraction is None:
            return []
        roi, xx, yy, mask, x_min, y_min = extraction

        valid_vals = roi[~np.isnan(roi)]
        if len(valid_vals) < 10:
            return []

        v_min, v_max = float(np.nanmin(valid_vals)), float(np.nanmax(valid_vals))
        if v_max - v_min < 1e-9:
            return []

        # Replace NaN with v_min for smooth contouring
        roi_filled = np.where(np.isnan(roi), v_min, roi)

        levels = np.linspace(v_min, v_max, n_levels + 2)[1:-1]

        # Use matplotlib to get smooth contour paths
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        cs = ax.contour(xx, yy, roi_filled, levels=levels)
        plt.close(fig)

        contours_out = []
        # Use allsegs (works on matplotlib >= 3.8, avoids deprecated collections)
        for level_segs in cs.allsegs:
            for seg in level_segs:
                # seg is Nx2 array of (x, y) in absolute image coords
                if len(seg) < 3:
                    continue
                # Clip to circle boundary
                dx = seg[:, 0] - cx
                dy = seg[:, 1] - cy
                inside = (dx**2 + dy**2) <= (radius * 1.02)**2
                if np.sum(inside) < 3:
                    continue
                pts = seg[inside]
                cnt = pts.reshape(-1, 1, 2).astype(np.int32)
                contours_out.append(cnt)

        return contours_out

    def _compute_centroids(self):
        """Compare geometric vs dose-weighted centroid for each circle."""
        global _CENTROID_MARKERS, _ISOBAR_CONTOURS

        am = _get_auto_measurements()
        if am is None or not getattr(am, 'results', None):
            messagebox.showinfo("Analysis", "No AutoMeasurements results. Run detection first.")
            return

        overlay = _get_overlay()
        if overlay is None:
            messagebox.showinfo("Analysis", "No overlay data available.")
            return

        ip = self.image_processor
        if not ip.calibration_applied or ip.dose_channels is None:
            messagebox.showinfo("Analysis",
                                "Calibration must be applied to compute dose-weighted centroids.")
            return

        dose_img = ip.dose_channels
        if dose_img.ndim == 3:
            dose_2d = np.nanmean(dose_img, axis=2)
        else:
            dose_2d = dose_img

        self._clear_centroid_table_internal()
        _CENTROID_MARKERS = []
        _ISOBAR_CONTOURS = []

        method = self._centroid_method.get()
        item_to_shape = overlay.get("item_to_shape", {})
        non_converged = []

        # Group results by film
        film_results: dict[str, list] = {}
        for result in am.results:
            film = result.get("film", "Unknown")
            film_results.setdefault(film, []).append(result)

        for film_name, results in sorted(film_results.items()):
            film_node = self.centroid_tree.insert("", "end", text=film_name, open=True)

            for result in results:
                circle = result.get("circle", "")
                cx = result.get("x")
                cy = result.get("y")
                if cx is None or cy is None:
                    continue

                radius = self._find_radius(item_to_shape, am, cx, cy)

                extraction = self._extract_circle_roi(dose_2d, cx, cy, radius)
                if extraction is None:
                    continue
                roi, xx, yy, mask, x_min, y_min = extraction

                # Compute centroid with selected method
                centroid_result = None
                if method == "weighted_centroid":
                    centroid_result = self._centroid_weighted(roi, xx, yy, mask)
                elif method == "gaussian_power":
                    centroid_result = self._centroid_gaussian_power(roi, xx, yy, mask)
                elif method == "radius_equation":
                    centroid_result = self._centroid_radius_equation(
                        roi, xx, yy, mask, cx, cy)
                elif method == "phase_correlation":
                    centroid_result = self._centroid_phase_correlation(
                        roi, xx, yy, mask, cx, cy, radius)

                if centroid_result is None:
                    non_converged.append(f"{film_name}/{circle}")
                    self.centroid_tree.insert(film_node, "end", text=circle, values=(
                        f"{cx:.1f}", f"{cy:.1f}",
                        "N/C", "N/C", "—", "—", "—"
                    ))
                    continue

                dose_cx, dose_cy = centroid_result
                dx = dose_cx - cx
                dy = dose_cy - cy
                dist = np.sqrt(dx**2 + dy**2)

                self.centroid_tree.insert(film_node, "end", text=circle, values=(
                    f"{cx:.1f}", f"{cy:.1f}",
                    f"{dose_cx:.1f}", f"{dose_cy:.1f}",
                    f"{dx:.2f}", f"{dy:.2f}", f"{dist:.2f}"
                ))

                _CENTROID_MARKERS.append((float(cx), float(cy), float(dose_cx), float(dose_cy)))

                # Compute isobars for this circle
                isobar_contours = self._extract_isobars(dose_2d, cx, cy, radius)
                _ISOBAR_CONTOURS.extend(isobar_contours)

        if non_converged:
            messagebox.showwarning(
                "Convergence",
                f"Could not estimate centroid for:\n" +
                "\n".join(non_converged) +
                "\n\n(marked as N/C in the table)"
            )

        self.main_window.update_image()

    def _clear_centroid_table_internal(self):
        """Clear the tree without touching markers (used during recompute)."""
        for item in self.centroid_tree.get_children():
            self.centroid_tree.delete(item)

    def _clear_centroid_table(self):
        global _CENTROID_MARKERS, _ISOBAR_CONTOURS
        for item in self.centroid_tree.get_children():
            self.centroid_tree.delete(item)
        _CENTROID_MARKERS = []
        _ISOBAR_CONTOURS = []
        self.main_window.update_image()

    # ------------------------------------------------------------------
    # Dose vs Introduced Value
    # ------------------------------------------------------------------
    def _load_circles(self):
        """Load detected circles from AutoMeasurements results."""
        am = _get_auto_measurements()
        if am is None or not getattr(am, 'results', None):
            messagebox.showinfo("Analysis", "No AutoMeasurements results. Run detection first.")
            return

        self._clear_dose_table()

        for result in am.results:
            film = result.get("film", "")
            circle = result.get("circle", "")
            avg = result.get("avg_numeric", result.get("avg", 0.0))
            se = result.get("avg_unc_numeric", result.get("avg_unc", 0.0))

            try:
                avg = float(avg)
            except (ValueError, TypeError):
                avg = 0.0
            try:
                se = float(se)
            except (ValueError, TypeError):
                se = 0.0

            # Restore previously entered value if any
            key = (film, circle)
            prev = self._introduced_values.get(key, "")

            self.dose_tree.insert("", "end", values=(
                film, circle, f"{avg:.4f}", f"{se:.4f}", str(prev)
            ))

    def _clear_dose_table(self):
        # Save currently entered values before clearing
        for item in self.dose_tree.get_children():
            vals = self.dose_tree.item(item, "values")
            if vals and len(vals) >= 5:
                key = (vals[0], vals[1])
                val_str = vals[4].strip()
                if val_str:
                    try:
                        self._introduced_values[key] = float(val_str)
                    except ValueError:
                        pass
        for item in self.dose_tree.get_children():
            self.dose_tree.delete(item)

    def _on_dose_tree_double_click(self, event):
        """Allow editing the 'Introduced Value' column on double-click."""
        region = self.dose_tree.identify_region(event.x, event.y)
        if region != "cell":
            return

        col = self.dose_tree.identify_column(event.x)
        # col is "#1", "#2", ... — we want #5 (value column)
        if col != "#5":
            return

        item = self.dose_tree.identify_row(event.y)
        if not item:
            return

        # Get cell bounding box
        bbox = self.dose_tree.bbox(item, column=col)
        if not bbox:
            return

        x, y, w, h = bbox
        current_vals = self.dose_tree.item(item, "values")
        current_text = current_vals[4] if len(current_vals) >= 5 else ""

        # Create entry widget on top of the cell
        entry = ttk.Entry(self.dose_tree, width=10)
        entry.place(x=x, y=y, width=w, height=h)
        entry.insert(0, current_text)
        entry.focus_set()
        entry.select_range(0, tk.END)

        def _commit(ev=None):
            new_val = entry.get().strip()
            entry.destroy()
            # Update treeview
            vals = list(self.dose_tree.item(item, "values"))
            vals[4] = new_val
            self.dose_tree.item(item, values=vals)
            # Store
            key = (vals[0], vals[1])
            if new_val:
                try:
                    self._introduced_values[key] = float(new_val)
                except ValueError:
                    pass

        def _cancel(ev=None):
            entry.destroy()

        entry.bind("<Return>", _commit)
        entry.bind("<FocusOut>", _commit)
        entry.bind("<Escape>", _cancel)

    def _plot_dose_vs_value(self):
        """Plot measured dose vs introduced value, per film (RC)."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        # Collect data grouped by film
        film_data: dict[str, list] = {}

        for item in self.dose_tree.get_children():
            vals = self.dose_tree.item(item, "values")
            if not vals or len(vals) < 5:
                continue
            film = vals[0]
            val_str = vals[4].strip()
            if not val_str:
                continue
            try:
                introduced = float(val_str)
                dose = float(vals[2])
                se = float(vals[3])
            except (ValueError, TypeError):
                continue

            film_data.setdefault(film, []).append((introduced, dose, se))

        if not film_data:
            messagebox.showinfo("Analysis",
                                "No data to plot. Load circles and enter introduced values first.")
            return

        n_films = len(film_data)
        fig, axes = plt.subplots(1, n_films, figsize=(5 * n_films, 5), squeeze=False)

        for idx, (film_name, data) in enumerate(sorted(film_data.items())):
            ax = axes[0, idx]
            xs = np.array([d[0] for d in data])
            ys = np.array([d[1] for d in data])
            errs = np.array([d[2] for d in data])

            # Plot data points with error bars
            ax.errorbar(xs, ys, yerr=errs, fmt='o', color='#2980b9',
                        markersize=7, capsize=4, markeredgecolor='white',
                        markeredgewidth=1.2, ecolor='gray', zorder=5)

            # Linear fit forced through origin: y = m * x
            # Weighted least squares: m = Σ(w_i * x_i * y_i) / Σ(w_i * x_i²)
            weights = np.where(errs > 0, 1.0 / errs**2, 1.0)
            denom = np.sum(weights * xs**2)
            if denom > 0:
                m = np.sum(weights * xs * ys) / denom
            else:
                m = 1.0

            # R² calculation
            y_pred = m * xs
            ss_res = np.sum((ys - y_pred)**2)
            ss_tot = np.sum((ys - np.mean(ys))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Plot fit line
            x_fit = np.linspace(0, max(xs) * 1.1, 100)
            y_fit = m * x_fit
            ax.plot(x_fit, y_fit, 'r-', linewidth=1.5, zorder=3)

            # Equation and R² annotation
            ax.text(0.05, 0.92,
                    f"y = {m:.4f}·x\nR² = {r_squared:.4f}",
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

            ax.set_xlabel("Introduced Value", fontsize=11)
            ax.set_ylabel("Measured Dose (Gy)", fontsize=11)
            ax.set_title(film_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Force origin
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        fig.tight_layout()
        plt.show()
