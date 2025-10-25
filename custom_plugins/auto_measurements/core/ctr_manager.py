"""
CTR (Control) circle management for background subtraction.

This module contains the CTRManager class for managing control regions
in radiochromic film dosimetry for background dose correction.
"""

from typing import Optional, List, Tuple
from math import sqrt
import tkinter as tk
from tkinter import ttk, messagebox
import logging

from ..models import Circle, CTR_DOSE_THRESHOLD
from .formatter import MeasurementFormatter


class CTRManager:
    """Manages CTR (Control) circle functionality for background subtraction."""
    
    def __init__(self, tree_widget, formatter: MeasurementFormatter):
        """Initialize CTR manager.
        
        Args:
            tree_widget: ttk.Treeview widget for displaying measurements
            formatter: MeasurementFormatter instance for numeric operations
        """
        self.tree = tree_widget
        self.formatter = formatter
        self.ctr_map = {}  # {film_name: ctr_item_id}
        self.original_measurements = {}  # {item_id: {dose, std, avg, avg_unc}}
    
    def store_original_measurement(self, item_id: str, dose_str: str, std_str: str, 
                                   avg_str: str, avg_unc_str: str):
        """Store original measurement data before any CTR corrections."""
        self.original_measurements[item_id] = {
            "dose": dose_str,
            "std": std_str,
            "avg": avg_str,
            "avg_unc": avg_unc_str
        }
    
    def toggle_ctr_for_item(self, item_id: str, film_name: str, overlay_dict: dict, 
                           update_callback=None) -> bool:
        """Toggle CTR status for a specific item.
        
        Args:
            item_id: Tree item ID
            film_name: Name of the film containing the circle
            overlay_dict: Global _OVERLAY dictionary
            update_callback: Optional callback to trigger after CTR toggle
            
        Returns:
            True if CTR was added, False if removed
        """
        # Check if this circle is already CTR
        if self.ctr_map.get(film_name) == item_id:
            # Remove CTR status
            self.ctr_map.pop(film_name, None)
            current_text = self.tree.item(item_id, "text")
            if "(CTR)" in current_text:
                new_text = current_text.replace(" (CTR)", "")
                self.tree.item(item_id, text=new_text)
            self.tree.item(item_id, tags=())  # Remove CTR tag
            
            # Restore original values for ALL circles in this film
            film_id = self.tree.parent(item_id)
            if film_id:
                for circle_id in self.tree.get_children(film_id):
                    if circle_id in self.original_measurements:
                        self._restore_item_values(circle_id)
            
            if update_callback:
                update_callback()
            return False
        else:
            # Remove existing CTR if any
            if film_name in self.ctr_map:
                old_ctr_id = self.ctr_map[film_name]
                if self.tree.exists(old_ctr_id):
                    old_text = self.tree.item(old_ctr_id, "text")
                    if "(CTR)" in old_text:
                        self.tree.item(old_ctr_id, text=old_text.replace(" (CTR)", ""))
                    self.tree.item(old_ctr_id, tags=())
                    
                    # Restore original values for the old CTR
                    if old_ctr_id in self.original_measurements:
                        self._restore_item_values(old_ctr_id)
            
            # Set new CTR
            self.ctr_map[film_name] = item_id
            current_text = self.tree.item(item_id, "text")
            if "(CTR)" not in current_text:
                self.tree.item(item_id, text=f"{current_text} (CTR)")
            self.tree.item(item_id, tags=("ctr",))
            
            if update_callback:
                update_callback()
            return True
    
    def _restore_item_values(self, item_id: str):
        """Restore original values for a single item."""
        orig_data = self.original_measurements[item_id]
        try:
            avg_unc_val = self.formatter.clean_numeric_string(orig_data["avg_unc"])
            ci95_value = avg_unc_val * 1.96
            ci95_str = f"±{ci95_value:.4f}"
        except (ValueError, TypeError):
            ci95_str = ""
        
        self.tree.item(item_id, values=(
            orig_data["dose"],
            orig_data["std"],
            orig_data["avg"],
            orig_data["avg_unc"],
            ci95_str
        ))
    
    def detect_ctr_automatically(self, film_name: str, film_circles: list, 
                                threshold: float = CTR_DOSE_THRESHOLD) -> bool:
        """Automatically detect CTR circle based on dose threshold.
        
        Args:
            film_name: Name of the film
            film_circles: List of circle data dictionaries with 'avg_val' and 'circle_id'
            threshold: Dose threshold for CTR detection (default: CTR_DOSE_THRESHOLD)
            
        Returns:
            True if CTR was detected and set, False otherwise
        """
        ctr_candidate = None
        min_dose = float('inf')
        
        for circle_data in film_circles:
            avg_val = circle_data.get("avg_val", float('inf'))
            if avg_val <= threshold and avg_val < min_dose:
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
    
    def apply_ctr_subtraction(self, results_list: list) -> None:
        """Apply CTR subtraction to all circles in films with CTR.
        
        Args:
            results_list: List of result dictionaries to update with CTR-corrected values
        """
        logging.info(f"[CTR] apply_ctr_subtraction called. CTR map: {self.ctr_map}")
        logging.info(f"[CTR] Original measurements count: {len(self.original_measurements)}")
        
        for film_name, ctr_id in self.ctr_map.items():
            if not self.tree.exists(ctr_id):
                continue
            
            # Get CTR measurement data
            ctr_orig_data = self.original_measurements.get(ctr_id)
            if not ctr_orig_data:
                logging.warning(f"[CTR] No original data for CTR {ctr_id}")
                continue
            
            logging.info(f"[CTR] CTR original data: {ctr_orig_data}")
            
            ctr_avg = self.formatter.clean_numeric_string(ctr_orig_data["avg"])
            ctr_unc = self.formatter.clean_numeric_string(ctr_orig_data["avg_unc"])
            
            logging.info(f"[CTR] CTR avg={ctr_avg}, unc={ctr_unc}")
            
            if ctr_avg == 0.0 and ctr_unc == 0.0:
                continue
            
            # Find parent film
            film_id = self.tree.parent(ctr_id)
            if not film_id:
                continue
            
            # Get CTR circle name for matching in results
            ctr_circle_name = self.tree.item(ctr_id, 'text').replace(" (CTR)", "")
            
            # Update all circles in this film
            for circle_id in self.tree.get_children(film_id):
                if not self.tree.exists(circle_id):
                    continue
                
                orig_data = self.original_measurements.get(circle_id)
                if not orig_data:
                    continue
                
                orig_avg = self.formatter.clean_numeric_string(orig_data["avg"])
                orig_unc = self.formatter.clean_numeric_string(orig_data["avg_unc"])
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
                
                # Format for TreeView display
                avg_str = f"{corrected_avg}"
                unc_str = f"±{corrected_unc}"
                ci95_value = corrected_unc * 1.96
                ci95_str = f"±{ci95_value}"
                
                # Update TreeView
                current_values = list(self.tree.item(circle_id, "values"))
                if len(current_values) >= 5:
                    current_values[2] = avg_str
                    current_values[3] = unc_str
                    current_values[4] = ci95_str
                    self.tree.item(circle_id, values=tuple(current_values))
                
                # Update results with FULL PRECISION numeric values
                circle_name = self.tree.item(circle_id, 'text').replace(" (CTR)", "")
                for result in results_list:
                    if result['film'] == film_name and result['circle'].replace(" (CTR)", "") == circle_name:
                        # HYBRID SOLUTION: Save original values BEFORE applying CTR correction
                        # This allows restoration when navigating between files
                        if 'avg_original' not in result:
                            result['avg_original'] = orig_avg
                            result['avg_unc_original'] = orig_unc
                        
                        # Apply CTR correction
                        result['avg_numeric'] = corrected_avg
                        result['avg_unc_numeric'] = corrected_unc
                        break
    
    def restore_original_measurements(self):
        """Restore original measurements without CTR subtraction."""
        for item_id, orig_data in self.original_measurements.items():
            if self.tree.exists(item_id):
                try:
                    avg_unc_val = self.formatter.clean_numeric_string(orig_data["avg_unc"])
                    ci95_value = avg_unc_val * 1.96
                    ci95_str = f"±{ci95_value:.4f}"
                except (ValueError, TypeError):
                    ci95_str = ""
                
                self.tree.item(item_id, values=(
                    orig_data["dose"],
                    orig_data["std"],
                    orig_data["avg"],
                    orig_data["avg_unc"],
                    ci95_str
                ))
    
    def get_ctr_for_film(self, film_name: str) -> Optional[str]:
        """Get the CTR item ID for a given film."""
        return self.ctr_map.get(film_name)
    
    def is_ctr_circle(self, item_id: str) -> bool:
        """Check if an item is marked as CTR."""
        return item_id in self.ctr_map.values()
    
    def clear_all_ctr(self):
        """Clear all CTR mappings."""
        for film_name in list(self.ctr_map.keys()):
            ctr_id = self.ctr_map[film_name]
            if self.tree.exists(ctr_id):
                text = self.tree.item(ctr_id, "text")
                if "(CTR)" in text:
                    self.tree.item(ctr_id, text=text.replace(" (CTR)", ""))
                self.tree.item(ctr_id, tags=())
        self.ctr_map.clear()


# ================================================================
# Phase 7: FileDataManager - Multi-file session management
# ================================================================



__all__ = ['CTRManager']
