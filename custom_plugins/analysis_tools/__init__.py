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
import csv
import io
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

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
        self._introduced_value_label_var = tk.StringVar(value="Introduced Value")
        self._dose_tree_index: dict[str, dict[str, object]] = {}
        self._highlighted_dose_item: str | None = None
        self._dose_data_rows: list[dict[str, object]] = []
        self._analysis_results: list[dict[str, object]] = []
        self._show_analysis_data_var = tk.BooleanVar(value=False)

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
        ttk.Button(btn_row2, text="Copy as XLSX", command=self._copy_dose_data_for_excel).pack(side=tk.RIGHT, padx=4)

        self._see_data_switch = tk.Canvas(btn_row2, width=44, height=22,
                          highlightthickness=0, bd=0)
        self._see_data_switch.pack(side=tk.LEFT, padx=(12, 4))
        self._see_data_switch.bind("<Button-1>", self._toggle_see_data_switch)
        self._see_data_switch.bind("<Return>", self._toggle_see_data_switch)
        self._see_data_switch.bind("<space>", self._toggle_see_data_switch)
        self._see_data_switch.configure(cursor="hand2", takefocus=1)
        ttk.Label(btn_row2, text="See Data").pack(side=tk.LEFT, padx=(0, 4))

        self._force_origin_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(btn_row2, text="Force fit through 0",
                variable=self._force_origin_var,
                command=self._on_analysis_option_changed).pack(side=tk.RIGHT, padx=4)

        self._separate_rc_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(btn_row2, text="Plot each RC separately",
                variable=self._separate_rc_var,
                command=self._on_analysis_option_changed).pack(side=tk.RIGHT, padx=4)

        label_row = ttk.Frame(dose_frame)
        label_row.pack(fill=tk.X, padx=6, pady=(0, 4))
        ttk.Label(label_row, text="X label / data name:").pack(side=tk.LEFT, padx=(0, 6))
        value_name_entry = ttk.Entry(label_row, textvariable=self._introduced_value_label_var, width=24)
        value_name_entry.pack(side=tk.LEFT)
        self._introduced_value_label_var.trace_add("write", self._on_introduced_value_label_changed)

        # Editable treeview with same hierarchy as AutoMeasurements: Film > Circle
        cols2 = ("dose", "se", "value")
        self.dose_tree = ttk.Treeview(dose_frame, columns=cols2, show="tree headings", height=10)
        self.dose_tree.heading("#0", text="Film / Circle", anchor=tk.W)
        self.dose_tree.column("#0", width=130, minwidth=100)
        for c, w, text in [
            ("dose", 90, "Dose (Gy)"), ("se", 80, "SE (Gy)"),
            ("value", 90, "Introduced Value")
        ]:
            self.dose_tree.heading(c, text=text)
            self.dose_tree.column(c, width=w, anchor=tk.CENTER)

        self.dose_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # Double-click to edit the "Introduced Value" column
        self.dose_tree.bind("<Double-1>", self._on_dose_tree_double_click)
        self.dose_tree.bind("<<TreeviewSelect>>", self._on_dose_tree_select)
        self._render_see_data_switch()
        self._refresh_dose_tree_columns()

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

    def _smooth_dose_map(self, dose_2d: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply light Gaussian smoothing while preserving NaN regions."""
        if sigma <= 0:
            return dose_2d

        valid_mask = np.isfinite(dose_2d)
        if not np.any(valid_mask):
            return dose_2d

        filled = np.where(valid_mask, dose_2d, 0.0).astype(np.float64)
        weights = valid_mask.astype(np.float64)

        blurred_values = gaussian_filter(filled, sigma=sigma, mode="nearest")
        blurred_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")

        smoothed = np.full(dose_2d.shape, np.nan, dtype=np.float64)
        valid_blur = blurred_weights > 1e-12
        smoothed[valid_blur] = blurred_values[valid_blur] / blurred_weights[valid_blur]
        return smoothed

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
    def _extract_isobars(self, dose_2d, cx, cy, radius, n_levels=5):
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

        percentile_count = max(1, min(int(n_levels), 5))
        percentile_levels = np.linspace(15.0, 85.0, percentile_count)
        levels = np.unique(np.percentile(valid_vals, percentile_levels))
        if len(levels) == 0:
            return []
        if len(levels) > 5:
            levels = levels[:5]
        levels = levels[(levels > v_min) & (levels < v_max)]
        if len(levels) == 0:
            return []

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
        dose_2d = self._smooth_dose_map(dose_2d, sigma=1.0)

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
    def _toggle_see_data_switch(self, event=None):
        self._show_analysis_data_var.set(not self._show_analysis_data_var.get())
        self._render_see_data_switch()
        self._refresh_dose_tree_columns()
        return "break"

    def _render_see_data_switch(self):
        canvas = getattr(self, "_see_data_switch", None)
        if canvas is None:
            return

        canvas.delete("all")
        is_on = self._show_analysis_data_var.get()
        track_color = "#2f855a" if is_on else "#a0aec0"
        knob_x = 31 if is_on else 13

        canvas.create_oval(2, 2, 20, 20, fill=track_color, outline=track_color)
        canvas.create_rectangle(11, 2, 33, 20, fill=track_color, outline=track_color)
        canvas.create_oval(24, 2, 42, 20, fill=track_color, outline=track_color)
        canvas.create_oval(knob_x - 8, 3, knob_x + 8, 19, fill="white", outline="#e2e8f0")

    def _get_introduced_value_label(self) -> str:
        label = self._introduced_value_label_var.get().strip()
        return label if label else "Introduced Value"

    def _on_introduced_value_label_changed(self, *_args):
        self._refresh_dose_tree_columns()

    def _on_analysis_option_changed(self):
        self._recompute_analysis_results()
        if self._show_analysis_data_var.get():
            self._refresh_dose_tree()

    def _refresh_dose_tree_columns(self):
        if self._show_analysis_data_var.get():
            cols = ("slope", "slope_se", "intercept", "intercept_se", "r2")
            self.dose_tree.configure(columns=cols)
            self.dose_tree.heading("#0", text="Series", anchor=tk.W)
            self.dose_tree.column("#0", width=150, minwidth=110)
            for col, width, text in [
                ("slope", 90, "Slope"),
                ("slope_se", 95, "Unc slope"),
                ("intercept", 90, "Intercept"),
                ("intercept_se", 95, "Unc intercept"),
                ("r2", 70, "R²"),
            ]:
                self.dose_tree.heading(col, text=text)
                self.dose_tree.column(col, width=width, anchor=tk.CENTER)
        else:
            cols = ("dose", "se", "value")
            self.dose_tree.configure(columns=cols)
            self.dose_tree.heading("#0", text="Film / Circle", anchor=tk.W)
            self.dose_tree.column("#0", width=130, minwidth=100)
            for col, width, text in [
                ("dose", 90, "Dose (Gy)"),
                ("se", 80, "SE (Gy)"),
                ("value", 120, self._get_introduced_value_label()),
            ]:
                self.dose_tree.heading(col, text=text)
                self.dose_tree.column(col, width=width, anchor=tk.CENTER)

        self._refresh_dose_tree()

    def _refresh_dose_tree(self):
        if not hasattr(self, "dose_tree"):
            return

        selected_info = None
        selected = self.dose_tree.selection()
        if selected and selected[0] in self._dose_tree_index:
            selected_info = self._dose_tree_index[selected[0]]

        self.dose_tree.delete(*self.dose_tree.get_children())
        self._dose_tree_index.clear()

        if self._show_analysis_data_var.get():
            self._recompute_analysis_results()
            self._set_dose_overlay_highlight(None)
            if not self._analysis_results:
                self.dose_tree.insert(
                    "",
                    "end",
                    text="No analysis results",
                    values=("", "", "", "", ""),
                )
                return

            for result in self._analysis_results:
                self.dose_tree.insert(
                    "",
                    "end",
                    text=str(result.get("name", "")),
                    values=(
                        str(result.get("slope", "")),
                        str(result.get("slope_se", "")),
                        str(result.get("intercept", "")),
                        str(result.get("intercept_se", "")),
                        str(result.get("r2", "")),
                    ),
                )
            return

        films: dict[str, str] = {}
        reselect_item = None
        for row in self._dose_data_rows:
            film = str(row["film"])
            circle = str(row["circle"])
            film_item = films.get(film)
            if film_item is None:
                film_item = self.dose_tree.insert("", "end", text=film, values=("", "", ""), open=True)
                films[film] = film_item

            circle_item = self.dose_tree.insert(
                film_item,
                "end",
                text=circle,
                values=(
                    f"{float(row['dose']):.4f}",
                    f"{float(row['se']):.4f}",
                    str(row.get("value", "")),
                ),
            )
            self._dose_tree_index[circle_item] = {
                "film": film,
                "circle": circle,
                "coords": row["coords"],
            }

            if selected_info and selected_info["film"] == film and selected_info["circle"] == circle:
                reselect_item = circle_item

        if reselect_item is not None:
            self.dose_tree.selection_set(reselect_item)
            self.dose_tree.focus(reselect_item)
            self._set_dose_overlay_highlight(reselect_item)
        else:
            self._set_dose_overlay_highlight(None)

    def _set_dose_overlay_highlight(self, item_id: str | None):
        overlay = _get_overlay()
        if overlay is None:
            return

        if item_id is None:
            overlay.pop("highlight", None)
            self._highlighted_dose_item = None
            self.main_window.update_image()
            return

        circle_info = self._dose_tree_index.get(item_id)
        if not circle_info:
            overlay.pop("highlight", None)
            self._highlighted_dose_item = None
            self.main_window.update_image()
            return

        overlay["highlight"] = ("circle", circle_info["coords"])
        self._highlighted_dose_item = item_id
        self.main_window.update_image()

    def _on_dose_tree_select(self, event=None):
        if self._show_analysis_data_var.get():
            self._set_dose_overlay_highlight(None)
            return

        selected = self.dose_tree.selection()
        if not selected:
            self._set_dose_overlay_highlight(None)
            return

        item_id = selected[0]
        if item_id not in self._dose_tree_index:
            self._set_dose_overlay_highlight(None)
            return

        self._set_dose_overlay_highlight(item_id)

    def _collect_film_data(self):
        film_data: dict[str, list] = {}
        for row in self._dose_data_rows:
            film = str(row["film"])
            val_str = str(row.get("value", "")).strip()
            if not val_str:
                continue
            try:
                introduced = float(val_str)
                dose = float(row["dose"])
                se = float(row["se"])
            except (ValueError, TypeError):
                continue

            film_data.setdefault(film, []).append((str(row["circle"]), introduced, dose, se))
        return film_data

    def _recompute_analysis_results(self):
        film_data = self._collect_film_data()
        if not film_data:
            self._analysis_results = []
            return

        force_origin = self._force_origin_var.get()
        separate_rc = self._separate_rc_var.get()
        sorted_films = sorted(film_data.items())
        analysis_rows = []

        if separate_rc:
            for film_name, data in sorted_films:
                xs = np.array([d[1] for d in data], dtype=np.float64)
                ys = np.array([d[2] for d in data], dtype=np.float64)
                errs = np.array([d[3] for d in data], dtype=np.float64)
                slope, intercept, slope_se, intercept_se, r_squared = self._fit_line(
                    xs, ys, errs, force_origin
                )
                analysis_rows.append({
                    "name": film_name,
                    "slope": f"{slope:.6f}",
                    "slope_se": f"{slope_se:.6f}",
                    "intercept": f"{intercept:.6f}",
                    "intercept_se": f"{intercept_se:.6f}",
                    "r2": f"{r_squared:.6f}",
                })
        else:
            all_data = [d for _, series_data in sorted_films for d in series_data]
            xs = np.array([d[1] for d in all_data], dtype=np.float64)
            ys = np.array([d[2] for d in all_data], dtype=np.float64)
            errs = np.array([d[3] for d in all_data], dtype=np.float64)
            slope, intercept, slope_se, intercept_se, r_squared = self._fit_line(
                xs, ys, errs, force_origin
            )
            analysis_rows.append({
                "name": "All RC Data",
                "slope": f"{slope:.6f}",
                "slope_se": f"{slope_se:.6f}",
                "intercept": f"{intercept:.6f}",
                "intercept_se": f"{intercept_se:.6f}",
                "r2": f"{r_squared:.6f}",
            })

        self._analysis_results = analysis_rows

    def _editable_dose_items(self):
        return [item_id for item_id in self._dose_tree_index.keys() if self.dose_tree.exists(item_id)]

    def _focus_adjacent_dose_item(self, current_item, step):
        items = self._editable_dose_items()
        if current_item not in items:
            return
        next_index = items.index(current_item) + step
        if next_index < 0 or next_index >= len(items):
            return
        next_item = items[next_index]
        self.dose_tree.selection_set(next_item)
        self.dose_tree.focus(next_item)
        self.dose_tree.see(next_item)
        self._set_dose_overlay_highlight(next_item)
        self._open_dose_value_editor(next_item)

    def _fit_line(self, xs: np.ndarray, ys: np.ndarray, errs: np.ndarray, force_origin: bool):
        weights = np.where(errs > 0, 1.0 / errs**2, 1.0)

        if force_origin:
            denom = np.sum(weights * xs**2)
            slope = np.sum(weights * xs * ys) / denom if denom > 0 else 0.0
            intercept = 0.0
        else:
            s = np.sum(weights)
            sx = np.sum(weights * xs)
            sy = np.sum(weights * ys)
            sxx = np.sum(weights * xs**2)
            sxy = np.sum(weights * xs * ys)
            denom = s * sxx - sx**2
            if abs(denom) < 1e-12:
                slope = 0.0
                intercept = float(np.average(ys, weights=weights)) if len(ys) else 0.0
            else:
                slope = (s * sxy - sx * sy) / denom
                intercept = (sxx * sy - sx * sxy) / denom

        y_pred = slope * xs + intercept
        ss_res = np.sum((ys - y_pred)**2)
        ss_tot = np.sum((ys - np.mean(ys))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if force_origin:
            dof = max(len(xs) - 1, 1)
            sigma2 = np.sum(weights * (ys - y_pred) ** 2) / dof if len(xs) > 1 else 0.0
            denom = np.sum(weights * xs**2)
            slope_se = np.sqrt(sigma2 / denom) if denom > 0 and sigma2 > 0 else 0.0
            intercept_se = 0.0
        else:
            dof = max(len(xs) - 2, 1)
            sigma2 = np.sum(weights * (ys - y_pred) ** 2) / dof if len(xs) > 2 else 0.0
            s = np.sum(weights)
            sx = np.sum(weights * xs)
            sxx = np.sum(weights * xs**2)
            denom = s * sxx - sx**2
            if denom > 0 and sigma2 > 0:
                slope_se = np.sqrt(sigma2 * s / denom)
                intercept_se = np.sqrt(sigma2 * sxx / denom)
            else:
                slope_se = 0.0
                intercept_se = 0.0

        return slope, intercept, slope_se, intercept_se, r_squared

    def _load_circles(self):
        """Load detected circles from AutoMeasurements results."""
        am = _get_auto_measurements()
        if am is None or not getattr(am, 'results', None):
            messagebox.showinfo("Analysis", "No AutoMeasurements results. Run detection first.")
            return

        self._clear_dose_table()
        self._analysis_results = []

        overlay = _get_overlay()
        item_to_shape = overlay.get("item_to_shape", {}) if overlay else {}
        loaded_rows = []

        for result in am.results:
            film = result.get("film", "")
            circle = result.get("circle", "")
            avg = result.get("avg_numeric", result.get("avg", 0.0))
            se = result.get("avg_unc_numeric", result.get("avg_unc", 0.0))
            cx = result.get("x")
            cy = result.get("y")

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

            radius = self._find_radius(item_to_shape, am, cx, cy) if cx is not None and cy is not None else 30
            loaded_rows.append({
                "film": film,
                "circle": circle,
                "dose": avg,
                "se": se,
                "value": str(prev),
                "coords": (cx, cy, radius),
            })

        self._dose_data_rows = loaded_rows
        self._recompute_analysis_results()
        self._refresh_dose_tree()

    def _clear_dose_table(self):
        for row in self._dose_data_rows:
            val_str = str(row.get("value", "")).strip()
            if val_str:
                try:
                    self._introduced_values[(str(row["film"]), str(row["circle"]))] = float(val_str)
                except ValueError:
                    pass

        self._dose_data_rows = []
        self._analysis_results = []
        self.dose_tree.delete(*self.dose_tree.get_children())
        self._dose_tree_index.clear()
        self._set_dose_overlay_highlight(None)

    def _copy_dose_data_for_excel(self):
        """Copy current analysis data to the clipboard in tabular form for Excel."""
        output = io.StringIO()
        writer = csv.writer(output, delimiter='\t', lineterminator='\n')

        if self._show_analysis_data_var.get():
            self._recompute_analysis_results()
            writer.writerow(["Series", "Slope", "Unc slope", "Intercept", "Unc intercept", "R2"])
            for row in self._analysis_results:
                writer.writerow([
                    row.get("name", ""),
                    row.get("slope", ""),
                    row.get("slope_se", ""),
                    row.get("intercept", ""),
                    row.get("intercept_se", ""),
                    row.get("r2", ""),
                ])
        else:
            writer.writerow(["Film", "Circle", "Dose (Gy)", "SE (Gy)", self._get_introduced_value_label()])
            for row in self._dose_data_rows:
                writer.writerow([
                    row.get("film", ""),
                    row.get("circle", ""),
                    f"{float(row.get('dose', 0.0)):.6f}",
                    f"{float(row.get('se', 0.0)):.6f}",
                    row.get("value", ""),
                ])

        clipboard_text = output.getvalue()
        if not clipboard_text.strip():
            messagebox.showinfo("Analysis", "No data available to copy.")
            return

        self.frame.clipboard_clear()
        self.frame.clipboard_append(clipboard_text)
        self.frame.update()

    def _on_dose_tree_double_click(self, event):
        """Allow editing the 'Introduced Value' column on double-click."""
        if self._show_analysis_data_var.get():
            return

        region = self.dose_tree.identify_region(event.x, event.y)
        if region != "cell":
            return

        col = self.dose_tree.identify_column(event.x)
        # Tree column is #0, value column is #3
        if col != "#3":
            return

        item = self.dose_tree.identify_row(event.y)
        if not item or item not in self._dose_tree_index:
            return

        self._open_dose_value_editor(item)

    def _open_dose_value_editor(self, item):
        if item not in self._dose_tree_index:
            return

        col = "#3"

        # Get cell bounding box
        bbox = self.dose_tree.bbox(item, column=col)
        if not bbox:
            return

        x, y, w, h = bbox
        current_vals = self.dose_tree.item(item, "values")
        current_text = current_vals[2] if len(current_vals) >= 3 else ""

        # Create entry widget on top of the cell
        entry = ttk.Entry(self.dose_tree, width=10)
        entry.place(x=x, y=y, width=w, height=h)
        entry.insert(0, current_text)
        entry.focus_set()
        entry.select_range(0, tk.END)

        def _commit():
            new_val = entry.get().strip()
            entry.destroy()
            # Update treeview
            vals = list(self.dose_tree.item(item, "values"))
            vals[2] = new_val
            self.dose_tree.item(item, values=vals)
            # Store
            info = self._dose_tree_index.get(item)
            if info is None:
                return
            key = (info["film"], info["circle"])
            for row in self._dose_data_rows:
                if row["film"] == info["film"] and row["circle"] == info["circle"]:
                    row["value"] = new_val
                    break
            if new_val:
                try:
                    self._introduced_values[key] = float(new_val)
                except ValueError:
                    pass
            elif key in self._introduced_values:
                self._introduced_values.pop(key, None)

            self._recompute_analysis_results()
            if self._show_analysis_data_var.get():
                self._refresh_dose_tree()

        def _commit_and_close(ev=None):
            _commit()

        def _commit_and_move(step):
            _commit()
            self._focus_adjacent_dose_item(item, step)
            return "break"

        def _cancel(ev=None):
            entry.destroy()

        entry.bind("<Return>", _commit_and_close)
        entry.bind("<FocusOut>", _commit_and_close)
        entry.bind("<Tab>", lambda ev: _commit_and_move(1))
        entry.bind("<Shift-Tab>", lambda ev: _commit_and_move(-1))
        entry.bind("<Escape>", _cancel)

    def _plot_dose_vs_value(self):
        """Plot measured dose vs introduced value with configurable fitting and grouping."""
        import matplotlib.pyplot as plt

        # Collect data grouped by film
        film_data = self._collect_film_data()

        if not film_data:
            messagebox.showinfo("Analysis",
                                "No data to plot. Load circles and enter introduced values first.")
            return

        force_origin = self._force_origin_var.get()
        separate_rc = self._separate_rc_var.get()
        sorted_films = sorted(film_data.items())
        self._recompute_analysis_results()

        if separate_rc:
            n_films = len(sorted_films)
            fig, axes = plt.subplots(1, n_films, figsize=(5 * n_films, 5), squeeze=False)
            axes_list = list(axes[0])
            grouped_series = [(film_name, data, axes_list[idx], '#2980b9')
                              for idx, (film_name, data) in enumerate(sorted_films)]
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            palette = plt.cm.get_cmap('tab10', max(len(sorted_films), 1))
            axes_list = [ax]
            grouped_series = [
                (film_name, data, ax, palette(idx))
                for idx, (film_name, data) in enumerate(sorted_films)
            ]

        all_xs = []
        all_ys = []

        for idx, (film_name, data, ax, color) in enumerate(grouped_series):
            xs = np.array([d[1] for d in data], dtype=np.float64)
            ys = np.array([d[2] for d in data], dtype=np.float64)
            errs = np.array([d[3] for d in data], dtype=np.float64)
            all_xs.extend(xs.tolist())
            all_ys.extend(ys.tolist())

            label = film_name if not separate_rc else None
            ax.errorbar(xs, ys, yerr=errs, fmt='o', color=color,
                        markersize=7, capsize=4, markeredgecolor='white',
                        markeredgewidth=1.2, ecolor='gray', zorder=5, label=label)

            if separate_rc or (not separate_rc and idx == len(grouped_series) - 1):
                fit_xs = xs if separate_rc else np.array(all_xs, dtype=np.float64)
                fit_ys = ys if separate_rc else np.array(all_ys, dtype=np.float64)
                fit_errs = errs if separate_rc else np.array([
                    d[3] for _, series_data in sorted_films for d in series_data
                ], dtype=np.float64)

                slope, intercept, slope_se, intercept_se, r_squared = self._fit_line(
                    fit_xs, fit_ys, fit_errs, force_origin
                )

                x_min = 0.0 if force_origin else min(0.0, float(np.min(fit_xs)) * 0.95)
                x_max = float(np.max(fit_xs)) * 1.1 if len(fit_xs) else 1.0
                if x_max <= x_min:
                    x_max = x_min + 1.0
                x_fit = np.linspace(x_min, x_max, 100)
                y_fit = slope * x_fit + intercept
                fit_label = 'Fit' if separate_rc else 'Global fit'
                ax.plot(x_fit, y_fit, '-', color='#c0392b', linewidth=1.5, zorder=3,
                        label=fit_label if not separate_rc else None)

                if force_origin:
                    eq_text = f"y = {slope:.4f}·x\nR² = {r_squared:.4f}"
                else:
                    eq_text = f"y = {slope:.4f}·x + {intercept:.4f}\nR² = {r_squared:.4f}"

                ax.text(0.05, 0.92,
                        eq_text,
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

            ax.set_xlabel("Introduced Value", fontsize=11)
            ax.set_ylabel("Measured Dose (Gy)", fontsize=11)
            ax.set_title(film_name if separate_rc else "All RC Data", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            if force_origin:
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)

        if not separate_rc:
            axes_list[0].legend(loc='best')

        if self._show_analysis_data_var.get():
            self._refresh_dose_tree()

        fig.tight_layout()
        plt.show()
