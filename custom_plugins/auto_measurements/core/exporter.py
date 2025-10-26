"""
CSV export functionality for measurement results.

This module contains the CSVExporter class for exporting
measurement results to CSV format.
"""

from typing import Optional, List
import os
import csv
import logging
from datetime import datetime

from ..models import MeasurementResult
from .formatter import MeasurementFormatter


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
                except (ValueError, TypeError):
                    formatted_parts.append(str(v))
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
                    except (ValueError, TypeError):
                        formatted_parts.append(part)
                return ", ".join(formatted_parts)
        
        # Single numeric value
        try:
            num = float(value)
            return f"{num}"
        except (ValueError, TypeError):
            return str(value)
    
    def get_export_values_for_result(self, result):
        """Get numeric values from result for export.
        
        Values in result are already CTR-corrected if subtraction was applied.
        
        Args:
            result: Result dictionary containing measurement data
        
        Returns:
            dict: Dictionary with 'dose', 'std', 'avg', 'avg_unc' numeric values
        """
        # Get numeric values from result - prefer _numeric keys for consistency
        dose_numeric = result.get('dose_numeric', result.get('dose', 0.0))
        std_numeric = result.get('std_numeric', result.get('std_per_channel', 0.0))
        
        # Use avg_numeric which contains:
        #   - When NO CTR: Original calculated average (full precision)
        #   - When CTR active: CTR-corrected value (full precision)
        avg_numeric = result.get('avg_numeric', result.get('avg', 0.0))
        avg_unc_numeric = result.get('avg_unc_numeric', result.get('avg_unc', 0.0))
        
        # Defensive parsing: handle edge case where values might still be strings
        if isinstance(dose_numeric, str) and dose_numeric.strip():
            try:
                dose_numeric = tuple(float(x.strip()) for x in dose_numeric.split(',')) if ',' in dose_numeric else float(dose_numeric)
            except (ValueError, AttributeError):
                dose_numeric = 0.0
        
        if isinstance(std_numeric, str) and std_numeric.strip():
            try:
                std_clean = std_numeric.replace('±', '').strip()
                std_numeric = tuple(float(x.strip()) for x in std_clean.split(',')) if ',' in std_clean else float(std_clean)
            except (ValueError, AttributeError):
                std_numeric = 0.0
        
        # Convert to float if strings (defensive programming)
        if isinstance(avg_numeric, str):
            try:
                avg_numeric = float(avg_numeric.replace('±', '').strip())
            except (ValueError, AttributeError):
                avg_numeric = 0.0
        
        if isinstance(avg_unc_numeric, str):
            try:
                avg_unc_numeric = float(avg_unc_numeric.replace('±', '').strip())
            except (ValueError, AttributeError):
                avg_unc_numeric = 0.0
        
        return {
            'dose': dose_numeric,
            'std': std_numeric,
            'avg': avg_numeric,
            'avg_unc': avg_unc_numeric,
        }
    
    def export_all_files(self, current_results, ctr_map, date_var, metadata_date, 
                        original_measurements=None, original_radii=None, original_values=None):
        """Export measurements from all files to CSV.
        
        Args:
            current_results: Current file's results list
            ctr_map: Current file's CTR map {film_name: circle_id}
            date_var: tkinter StringVar containing date
            metadata_date: Date extracted from metadata
            original_measurements: Original measurements dict (optional)
            original_radii: Original radii dict (optional)
            original_values: Original values dict (optional)
        
        Returns:
            bool: True if export successful, False otherwise
        """
        import csv
        import os
        from tkinter import filedialog, messagebox
        
        # Use empty dicts if not provided
        if original_measurements is None:
            original_measurements = {}
        if original_radii is None:
            original_radii = {}
        if original_values is None:
            original_values = {}
        
        # Store current file data before checking
        self.file_manager.store_current_data(
            lambda: current_results,
            lambda: ctr_map,
            lambda: original_measurements,
            lambda: original_radii,
            lambda: original_values
        )
        
        # Check if any files have data
        total_measurements = 0
        unmeasured_files = []
        
        for file_path in self.file_manager.file_list:
            if file_path in self.file_manager.file_data:
                file_results = self.file_manager.file_data[file_path]['results']
                total_measurements += len(file_results)
                if not self.file_manager.file_data[file_path]['measured']:
                    unmeasured_files.append(os.path.basename(file_path))
        
        # Check current file as well
        if current_results:
            total_measurements += len(current_results)
        
        if total_measurements == 0:
            messagebox.showwarning("Export", "No measurement data to export.")
            return False
        
        # Warn about unmeasured files
        if unmeasured_files:
            unmeasured_list = "\n".join(unmeasured_files)
            response = messagebox.askyesno(
                "Unmeasured Files", 
                f"The following files have no measurements:\n\n{unmeasured_list}\n\nDo you want to continue with the export?"
            )
            if not response:
                return False

        # Ask for file location
        filename = filedialog.asksaveasfilename(
            title="Save CSV (All Files)",
            filetypes=[("CSV files", "*.csv")],
            defaultextension=".csv"
        )
        
        if not filename:
            return False
            
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                # Define CSV columns
                cols = [
                    "Filename", "Date", "Film", "Circle", 
                    "doses_per_channel", "STD_doses_per_channel",
                    "average", "SE_average", "95%confident_interval(SE)",
                    "pixel_count", "uncertaty_calculation_method"
                ]
                
                writer = csv.writer(csvfile)
                writer.writerow(cols)
                
                # Get uncertainty calculation method
                uncertainty_method = self.image_processor.config.get("uncertainty_estimation_method", "weighted_average")
                
                # Export data from all files
                total_rows = 0
                
                for file_path in self.file_manager.file_list:
                    if file_path not in self.file_manager.file_data or not self.file_manager.file_data[file_path]['measured']:
                        continue
                        
                    file_data = self.file_manager.file_data[file_path]
                    file_results = file_data['results']
                    
                    if not file_results:
                        continue
                    
                    # Get date for this file
                    date_to_use = date_var.get() or metadata_date or ""
                    
                    # Extract filename without path
                    file_name = os.path.basename(file_path)
                    
                    # Get CTR map for this file
                    ctr_map_by_name = file_data.get('ctr_map_by_name', {})
                    
                    # Sort results by film name
                    sorted_results = sorted(file_results, key=lambda r: r.get('film', ''))
                    
                    # Process results for this file
                    for result in sorted_results:
                        film_name = result['film']
                        circle_name = result['circle']
                        
                        # Get export values
                        export_values = self.get_export_values_for_result(result)
                        
                        # Format values
                        doses_formatted = self.format_for_csv(export_values['dose'])
                        std_formatted = self.format_for_csv(export_values['std'])
                        avg_formatted = self.format_for_csv(export_values['avg'])
                        se_average_formatted = self.format_for_csv(export_values['avg_unc'])
                        
                        # Calculate 95% CI
                        ci95_formatted = ""
                        try:
                            se_numeric = export_values['avg_unc']
                            if isinstance(se_numeric, str):
                                se_numeric = float(se_numeric.replace('±', '').strip())
                            if se_numeric and se_numeric > 0:
                                ci95_val = se_numeric * 1.96
                                ci95_formatted = self.format_for_csv(ci95_val)
                        except (ValueError, TypeError):
                            pass
                        
                        # Create row
                        row_data = [
                            file_name, date_to_use, film_name, circle_name,
                            doses_formatted, std_formatted, avg_formatted, se_average_formatted,
                            ci95_formatted, result.get('pixel_count', ''), uncertainty_method
                        ]
                        
                        writer.writerow(row_data)
                        total_rows += 1
                
                # Also export current file data if it has unsaved measurements
                if current_results and (not self.file_manager.file_list or 
                                       self.file_manager.current_file_index >= len(self.file_manager.file_list)):
                    date_to_use = date_var.get() or metadata_date or ""
                    
                    # Get current file name
                    current_file_name = ""
                    if hasattr(self.image_processor, 'image_path') and self.image_processor.image_path:
                        current_file_name = os.path.basename(self.image_processor.image_path)
                    
                    # Build ctr_map_by_name from current ctr_map
                    current_ctr_map_by_name = {}
                    for film_name, ctr_id in ctr_map.items():
                        if self.tree.exists(ctr_id):
                            ctr_circle_name = self.tree.item(ctr_id, 'text').replace(" (CTR)", "")
                            current_ctr_map_by_name[film_name] = ctr_circle_name
                    
                    # Sort results
                    sorted_current_results = sorted(current_results, key=lambda r: r.get('film', ''))
                    
                    for result in sorted_current_results:
                        film_name = result['film']
                        circle_name = result['circle']
                        
                        # Get export values
                        export_values = self.get_export_values_for_result(result)
                        
                        # Format values
                        doses_formatted = self.format_for_csv(export_values['dose'])
                        std_formatted = self.format_for_csv(export_values['std'])
                        avg_formatted = self.format_for_csv(export_values['avg'])
                        se_average_formatted = self.format_for_csv(export_values['avg_unc'])
                        
                        # Calculate 95% CI
                        ci95_formatted = ""
                        try:
                            se_numeric = export_values['avg_unc']
                            if isinstance(se_numeric, str):
                                se_numeric = float(se_numeric.replace('±', '').strip())
                            if se_numeric and se_numeric > 0:
                                ci95_val = se_numeric * 1.96
                                ci95_formatted = self.format_for_csv(ci95_val)
                        except (ValueError, TypeError):
                            pass
                        
                        row_data = [
                            current_file_name, date_to_use, film_name, circle_name,
                            doses_formatted, std_formatted, avg_formatted, se_average_formatted,
                            ci95_formatted, result.get('pixel_count', ''), uncertainty_method
                        ]
                        
                        writer.writerow(row_data)
                        total_rows += 1
            
            # Show success message
            messagebox.showinfo("Export Complete", 
                              f"CSV successfully exported to:\n{filename}\n\n"
                              f"Total measurements exported: {total_rows}")
            return True
                              
        except Exception as exc:
            messagebox.showerror("Export Error", f"Error exporting CSV:\n{str(exc)}")
            return False


# ============================================================================
# PLUGIN INTERFACE
# ============================================================================



__all__ = ['CSVExporter']
