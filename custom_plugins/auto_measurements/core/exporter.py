"""
CSV export functionality for measurement results.

This module contains the CSVExporter class for exporting
measurement results to CSV format.
"""

from typing import Optional, List
import os
import csv
import logging
import json
import math
from datetime import datetime

from ..models import MeasurementResult
from .formatter import MeasurementFormatter
from app.core.calibration_manifest import MANIFEST_NAME, verify_calibration_manifest


class CSVExporter:
    """Handles CSV export of measurements from single or multiple files.
    
    Exports measurement data with proper formatting, including:
    - Multi-file support
    - CTR-corrected values
    - Full numeric precision
    - Uncertainty calculations
    - Metadata (filename, date, film, circle)
    """
    
    def __init__(self, tree_widget, image_processor, file_manager):
        """Initialize CSV exporter.
        
        Args:
            tree_widget: ttk.Treeview widget for accessing current measurements
            image_processor: ImageProcessor for configuration (uncertainty method)
            file_manager: FileDataManager for multi-file access
        """
        self.tree = tree_widget
        self.image_processor = image_processor
        self.file_manager = file_manager
    
    def format_for_csv(self, value):
        """Format numeric values for CSV export with full precision.
        
        Handles single values, tuples (multiple channels), and strings.
        
        Args:
            value: Value to format (float, int, str, tuple, list, or None)
        
        Returns:
            str: Formatted value(s) as string
        """
        if value is None or value == "":
            return ""
        
        # If it's already a tuple or list (multiple channels)
        if isinstance(value, (tuple, list)):
            formatted_parts = []
            for v in value:
                try:
                    num = float(v)
                    formatted_parts.append(f"{num}")
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Non-numeric export value: {v!r}") from exc
            return ", ".join(formatted_parts)
        
        # If it's a string, try to parse it
        if isinstance(value, str):
            # Remove ± symbols if present
            value = value.replace('±', '').strip()
            
            # Check if it's a comma-separated list (multiple channels as string)
            if ',' in value:
                parts = value.split(',')
                formatted_parts = []
                for part in parts:
                    part = part.strip()
                    try:
                        num = float(part)
                        formatted_parts.append(f"{num}")
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"Non-numeric export value: {part!r}") from exc
                return ", ".join(formatted_parts)
        
        # Single numeric value
        try:
            num = float(value)
            return f"{num}"
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Non-numeric export value: {value!r}") from exc

    @staticmethod
    def _parse_numeric(value, field_name, *, sequence=False):
        if value is None or value == "":
            raise ValueError(f"Missing numeric field '{field_name}'")
        if isinstance(value, str):
            cleaned = value.replace('±', '').strip()
            if sequence or ',' in cleaned:
                try:
                    return tuple(float(part.strip()) for part in cleaned.split(','))
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric field '{field_name}': {value!r}") from exc
            try:
                return float(cleaned)
            except ValueError as exc:
                raise ValueError(f"Invalid numeric field '{field_name}': {value!r}") from exc
        if isinstance(value, (tuple, list)):
            try:
                return tuple(float(part) for part in value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid numeric field '{field_name}': {value!r}") from exc
        try:
            return float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid numeric field '{field_name}': {value!r}") from exc

    def _calibration_export_metadata(self):
        csv_path = self.image_processor._find_fit_parameters_file()
        if csv_path is None:
            return "", "unavailable"
        directory = os.path.dirname(csv_path)
        integrity = verify_calibration_manifest(directory)
        if integrity is None:
            return "", "legacy-unverified"
        manifest_path = os.path.join(directory, MANIFEST_NAME)
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                calibration_id = json.load(handle).get("calibration_id", "")
        except (OSError, json.JSONDecodeError):
            calibration_id = ""
        artifact_results = [value for key, value in integrity.items() if key.endswith((".csv", ".npz"))]
        status = "verified" if integrity.get("manifest") and artifact_results and all(artifact_results) else "failed"
        return calibration_id, status
    
    def get_export_values_for_result(self, result):
        """Get numeric values from result for export.
        
        Values in result are already CTR-corrected if subtraction was applied.
        
        Args:
            result: Result dictionary containing measurement data
        
        Returns:
            dict: Dictionary with 'dose', 'std', 'avg', 'avg_unc' numeric values
        """
        # Get numeric values from result - prefer _numeric keys for consistency
        dose_numeric = result.get('dose_numeric', result.get('dose'))
        std_numeric = result.get('std_numeric', result.get('std_per_channel'))
        
        # Use avg_numeric which contains:
        #   - When NO CTR: Original calculated average (full precision)
        #   - When CTR active: CTR-corrected value (full precision)
        avg_numeric = result.get('avg_numeric', result.get('avg'))
        avg_unc_numeric = result.get('avg_unc_numeric', result.get('avg_unc'))

        dose_numeric = self._parse_numeric(dose_numeric, "dose")
        std_numeric = self._parse_numeric(std_numeric, "std")
        avg_numeric = self._parse_numeric(avg_numeric, "average")
        avg_unc_numeric = self._parse_numeric(avg_unc_numeric, "average_uncertainty")
        
        return {
            'dose': dose_numeric,
            'std': std_numeric,
            'avg': avg_numeric,
            'avg_unc': avg_unc_numeric,
        }
    
    COLUMNS = [
        "schema_version", "Filename", "Date", "Film", "Circle",
        "doses_per_channel", "STD_doses_per_channel", "average",
        "standard_uncertainty_average", "expanded_uncertainty_k1.96",
        "pixel_count", "uncertainty_calculation_method", "channel_weights",
        "calibration_id", "calibration_integrity", "units", "valid_pixel_counts",
        "measurement_status", "x", "y", "radius", "dose_correction_factor",
        "flat_applied", "ctr_context",
    ]

    def result_row(self, result, file_path=""):
        """Serialize one measurement using only its recorded provenance."""
        values = self.get_export_values_for_result(result)
        for field, value in values.items():
            items = value if isinstance(value, (tuple, list)) else (value,)
            if any(math.isinf(item) for item in items):
                raise ValueError(f"Infinite value in {field}")
            if field in ("std", "avg_unc") and any(item < 0 for item in items):
                raise ValueError(f"Negative uncertainty in {field}")
        mean, uncertainty = values["avg"], values["avg_unc"]
        if isinstance(mean, (tuple, list)) or isinstance(uncertainty, (tuple, list)):
            raise ValueError("Average and its uncertainty must be scalars.")
        provenance = result.get("provenance", {})
        status = "valid"
        doses = values["dose"] if isinstance(values["dose"], (tuple, list)) else (values["dose"],)
        if not math.isfinite(mean):
            status = "invalid"
        elif not math.isfinite(uncertainty):
            status = "uncertainty_unavailable"
        elif any(not math.isfinite(value) for value in doses):
            status = "partial_channels"
        counts = result.get("valid_pixel_counts", [])
        if counts and min(counts) < result.get("pixel_count", 0) and status == "valid":
            status = "partial_pixels"
        return [
            3, os.path.basename(provenance.get("source_file") or file_path),
            provenance.get("date", ""), result["film"], result["circle"],
            self.format_for_csv(values["dose"]), self.format_for_csv(values["std"]),
            self.format_for_csv(mean), self.format_for_csv(uncertainty),
            self.format_for_csv(1.96 * uncertainty) if math.isfinite(uncertainty) else "",
            result.get("pixel_count", ""), provenance.get("uncertainty_method", "unknown"),
            json.dumps(result.get("channel_weights") or {}, sort_keys=True),
            provenance.get("calibration_id", ""), provenance.get("calibration_integrity", "unknown"),
            provenance.get("units", "unknown"), json.dumps(counts), status,
            result.get("x", ""), result.get("y", ""), result.get("radius", ""),
            provenance.get("dose_correction_factor", ""), provenance.get("flat_applied", False),
            json.dumps(result.get("ctr_context", {}), sort_keys=True),
        ]

    def write_results(self, filename, datasets):
        """Validate all rows before atomically replacing the destination."""
        from app.utils.atomic_file import atomic_open
        rows = [
            self.result_row(result, path)
            for path, results in datasets
            for result in sorted(results, key=lambda row: (row.get("film", ""), row.get("circle", "")))
        ]
        if not rows:
            raise ValueError("No measurements to export.")
        with atomic_open(filename, newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.COLUMNS)
            writer.writerows(rows)
        return len(rows)

    def export_all_files(self, current_results, ctr_map, date_var, metadata_date,
                         original_measurements=None, original_radii=None, original_values=None):
        """Export measured files without relabelling stored results with current settings."""
        from tkinter import filedialog, messagebox
        self.file_manager.store_current_data(
            lambda: current_results, lambda: ctr_map,
            lambda: original_measurements or {}, lambda: original_radii or {},
            lambda: original_values or {},
        )
        datasets = [
            (path, self.file_manager.file_data[path]["results"])
            for path in self.file_manager.file_list
            if path in self.file_manager.file_data
            and self.file_manager.file_data[path].get("measured")
        ]
        if current_results and (not self.file_manager.file_list or
                self.file_manager.current_file_index < 0 or
                self.file_manager.current_file_index >= len(self.file_manager.file_list)):
            datasets.append((getattr(self.image_processor, "current_file", "") or "", current_results))
        if not any(results for _, results in datasets):
            messagebox.showwarning("Export", "No measurement data to export.")
            return False
        filename = filedialog.asksaveasfilename(
            title="Save CSV (All Files)", filetypes=[("CSV files", "*.csv")], defaultextension=".csv"
        )
        if not filename:
            return False
        try:
            count = self.write_results(filename, datasets)
        except Exception as exc:
            messagebox.showerror("Export Error", f"CSV was not replaced:\n{exc}")
            return False
        messagebox.showinfo("Export Complete", f"CSV exported to:\n{filename}\n\nMeasurements: {count}")
        return True

# ============================================================================
# PLUGIN INTERFACE
# ============================================================================



__all__ = ['CSVExporter']
