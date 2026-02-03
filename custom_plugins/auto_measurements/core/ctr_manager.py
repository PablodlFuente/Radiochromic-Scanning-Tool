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
    """Manages CTR (Control) circle functionality for background subtraction.
    
    Supports multiple CTR circles per film - when multiple are selected,
    their values are averaged with proper uncertainty propagation.
    """
    
    def __init__(self, tree_widget, formatter: MeasurementFormatter):
        """Initialize CTR manager.
        
        Args:
            tree_widget: ttk.Treeview widget for displaying measurements
            formatter: MeasurementFormatter instance for numeric operations
        """
        self.tree = tree_widget
        self.formatter = formatter
        self.ctr_map = {}  # {film_name: [list of ctr_item_ids]} - supports multiple CTRs per film
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
        
        Supports multiple CTR circles per film. When multiple CTRs are selected,
        their values will be averaged with proper uncertainty propagation.
        
        Args:
            item_id: Tree item ID
            film_name: Name of the film containing the circle
            overlay_dict: Global _OVERLAY dictionary
            update_callback: Optional callback to trigger after CTR toggle
            
        Returns:
            True if CTR was added, False if removed
        """
        # Initialize list for this film if not exists
        if film_name not in self.ctr_map:
            self.ctr_map[film_name] = []
        
        # Check if this circle is already CTR
        if item_id in self.ctr_map[film_name]:
            # Remove CTR status
            self.ctr_map[film_name].remove(item_id)
            # Clean up empty lists
            if not self.ctr_map[film_name]:
                del self.ctr_map[film_name]
                
            current_text = self.tree.item(item_id, "text")
            if "(CTR)" in current_text:
                new_text = current_text.replace(" (CTR)", "")
                self.tree.item(item_id, text=new_text)
            self.tree.item(item_id, tags=())  # Remove CTR tag
            
            # CRITICAL: Call update_callback which triggers restore_original_measurements
            # This ensures both TreeView AND results list are properly restored
            # Prevents "ghost CTR" where corrected values persist after unmarking
            if update_callback:
                update_callback()
            return False
        else:
            # Add new CTR to the list (allows multiple CTRs per film)
            self.ctr_map[film_name].append(item_id)
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
            # Initialize list if not exists
            if film_name not in self.ctr_map:
                self.ctr_map[film_name] = []
            # Add to list (only if not already there)
            if circle_id not in self.ctr_map[film_name]:
                self.ctr_map[film_name].append(circle_id)
            current_text = self.tree.item(circle_id, "text")
            self.tree.item(circle_id, text=f"{current_text} (CTR)")
            self.tree.item(circle_id, tags=("ctr",))
            return True
        return False
    
    def _compute_averaged_ctr(self, film_name: str, ctr_ids: list, results_lookup: dict) -> Tuple[float, float]:
        """Compute averaged CTR value and uncertainty from multiple CTR circles.
        
        When multiple CTR circles are selected, computes:
        - Average CTR value (mean of all CTR values)
        - Combined uncertainty using error propagation (std dev of means + propagated uncertainties)
        
        Args:
            film_name: Name of the film
            ctr_ids: List of CTR item IDs
            results_lookup: Dictionary for fast result lookup
            
        Returns:
            Tuple of (averaged_ctr_value, combined_uncertainty)
        """
        ctr_values = []
        ctr_uncertainties = []
        
        for ctr_id in ctr_ids:
            if not self.tree.exists(ctr_id):
                continue
                
            ctr_orig_data = self.original_measurements.get(ctr_id)
            if not ctr_orig_data:
                continue
            
            ctr_circle_name = self.tree.item(ctr_id, 'text').replace(" (CTR)", "").replace(" (GLOBAL CTR)", "")
            ctr_result = results_lookup.get((film_name, ctr_circle_name))
            
            if ctr_result:
                ctr_avg = ctr_result.get('avg_original', ctr_result.get('avg_numeric', ctr_result.get('avg', 0)))
                ctr_unc = ctr_result.get('avg_unc_original', ctr_result.get('avg_unc_numeric', ctr_result.get('avg_unc', 0)))
            else:
                ctr_avg = self.formatter.clean_numeric_string(ctr_orig_data["avg"])
                ctr_unc = self.formatter.clean_numeric_string(ctr_orig_data["avg_unc"])
            
            if ctr_avg is not None:
                ctr_values.append(float(ctr_avg))
                ctr_uncertainties.append(float(ctr_unc) if ctr_unc else 0.0)
        
        if not ctr_values:
            return 0.0, 0.0
        
        if len(ctr_values) == 1:
            return ctr_values[0], ctr_uncertainties[0]
        
        # Multiple CTRs: compute average and combined uncertainty
        import numpy as np
        ctr_mean = np.mean(ctr_values)
        
        # Combined uncertainty: sqrt(std_dev_of_means^2 + mean_of_individual_uncertainties^2)
        # This accounts for both the spread between CTR circles AND their individual uncertainties
        std_of_values = np.std(ctr_values, ddof=1) if len(ctr_values) > 1 else 0.0
        mean_unc = np.mean(ctr_uncertainties)
        
        # Error propagation: combine uncertainty from averaging and individual measurement uncertainties
        # SE of mean = std / sqrt(n) for the spread between CTR values
        se_of_mean = std_of_values / sqrt(len(ctr_values))
        # Propagated uncertainty from individual measurements
        propagated_unc = sqrt(sum(u**2 for u in ctr_uncertainties)) / len(ctr_uncertainties)
        
        # Total uncertainty: quadrature sum
        combined_unc = sqrt(se_of_mean**2 + propagated_unc**2)
        
        return float(ctr_mean), float(combined_unc)
    
    def apply_ctr_subtraction(self, results_list: list) -> None:
        """Apply CTR subtraction to all circles in films with CTR.
        
        Supports multiple CTR circles per film - their values are averaged
        with proper uncertainty propagation.
        
        Args:
            results_list: List of result dictionaries to update with CTR-corrected values
        
        IMPORTANT: Always use ORIGINAL values from self.original_measurements or 
        result['avg_original'] to avoid cumulative error when toggling CTR on/off.
        """
        # Create lookup dictionary for O(1) access instead of O(n) searches
        # Key: (film_name, circle_name) -> result dict
        results_lookup = {}
        if results_list:
            for result in results_list:
                key = (result['film'], result['circle'].replace(" (CTR)", "").replace(" (GLOBAL CTR)", ""))
                results_lookup[key] = result
        
        for film_name, ctr_ids in self.ctr_map.items():
            # ctr_ids is now a list of CTR item IDs
            if not ctr_ids:
                continue
            
            # Compute averaged CTR value from all CTR circles in this film
            ctr_avg, ctr_unc = self._compute_averaged_ctr(film_name, ctr_ids, results_lookup)
            
            if ctr_avg == 0.0 and ctr_unc == 0.0:
                continue
            
            # Find parent film from first valid CTR
            film_id = None
            for ctr_id in ctr_ids:
                if self.tree.exists(ctr_id):
                    film_id = self.tree.parent(ctr_id)
                    break
            
            if not film_id:
                continue
            
            # Update all circles in this film
            for circle_id in self.tree.get_children(film_id):
                if not self.tree.exists(circle_id):
                    continue
                
                orig_data = self.original_measurements.get(circle_id)
                if not orig_data:
                    continue
                
                # Get circle name for matching with results_list
                circle_name = self.tree.item(circle_id, 'text').replace(" (CTR)", "").replace(" (GLOBAL CTR)", "")
                
                # CRITICAL: Get ORIGINAL values, not potentially modified ones
                result = results_lookup.get((film_name, circle_name))
                if result:
                    orig_avg = result.get('avg_original', result.get('avg_numeric', result.get('avg')))
                    orig_unc = result.get('avg_unc_original', result.get('avg_unc_numeric', result.get('avg_unc')))
                else:
                    orig_avg = self.formatter.clean_numeric_string(orig_data["avg"])
                    orig_unc = self.formatter.clean_numeric_string(orig_data["avg_unc"])
                
                if orig_avg == 0.0 and orig_unc == 0.0:
                    continue
                
                # All circles (including CTRs) subtract the averaged CTR value
                # CTR circles will show their deviation from the average CTR
                corrected_avg = orig_avg - ctr_avg  # Can be negative for CTR below average
                corrected_unc = sqrt(orig_unc**2 + ctr_unc**2)
                
                # Get current TreeView values to preserve dose and std columns
                current_values = list(self.tree.item(circle_id, "values"))
                
                dose_current = current_values[0] if len(current_values) > 0 else ""
                std_current = current_values[1] if len(current_values) > 1 else ""
                
                _, _, avg_str, avg_unc_str, ci95_str = self.formatter.format_for_treeview(
                    dose_current,
                    std_current,
                    corrected_avg,
                    corrected_unc,
                    sig=2
                )
                
                if len(current_values) >= 5:
                    current_values[2] = avg_str
                    current_values[3] = avg_unc_str
                    current_values[4] = ci95_str
                    self.tree.item(circle_id, values=tuple(current_values))
                
                result = results_lookup.get((film_name, circle_name))
                if result:
                    if 'avg_original' not in result:
                        result['avg_original'] = result['avg_numeric']
                        result['avg_unc_original'] = result['avg_unc_numeric']
                    
                    result['avg_numeric'] = corrected_avg
                    result['avg_unc_numeric'] = corrected_unc
    
    def restore_original_measurements(self, results_list=None):
        """Restore original measurements without CTR subtraction.
        
        Args:
            results_list: Optional list of result dictionaries to update with original values
        """
        for item_id, orig_data in self.original_measurements.items():
            if self.tree.exists(item_id):
                # The orig_data already contains formatted strings (dose_str, std_str, avg_str, avg_unc_str)
                # These were stored using format_for_treeview, so just restore them directly
                
                # Extract numeric values for TreeView formatting and results update
                try:
                    avg_unc_val = self.formatter.clean_numeric_string(orig_data["avg_unc"])
                    avg_val = self.formatter.clean_numeric_string(orig_data["avg"])
                    
                    # Re-format using format_for_treeview to ensure consistency
                    _, _, avg_str, avg_unc_str, ci95_str = self.formatter.format_for_treeview(
                        orig_data["dose"],
                        orig_data["std"],
                        avg_val,
                        avg_unc_val,
                        sig=2
                    )
                except (ValueError, TypeError):
                    # Fallback: use stored values directly
                    avg_str = orig_data["avg"]
                    avg_unc_str = orig_data["avg_unc"]
                    ci95_str = ""
                    avg_val = 0.0
                    avg_unc_val = 0.0
                
                # Update TreeView
                self.tree.item(item_id, values=(
                    orig_data["dose"],
                    orig_data["std"],
                    avg_str,
                    avg_unc_str,
                    ci95_str
                ))
                
                # Update results list with original numeric values (for CSV export)
                if results_list is not None:
                    # Get film and circle names (strip any CTR markers)
                    circle_name = self.tree.item(item_id, 'text').replace(" (CTR)", "").replace(" (GLOBAL CTR)", "")
                    film_id = self.tree.parent(item_id)
                    if film_id:
                        film_name = self.tree.item(film_id, 'text')
                        
                        # Find and update the result
                        for result in results_list:
                            result_circle = result['circle'].replace(" (CTR)", "").replace(" (GLOBAL CTR)", "")
                            if result['film'] == film_name and result_circle == circle_name:
                                # CRITICAL: Only restore if CTR was actually applied (avg_original exists)
                                # If avg_original doesn't exist, avg_numeric already has correct full-precision value
                                if 'avg_original' in result:
                                    # CTR was applied, restore from saved originals (full precision)
                                    result['avg_numeric'] = result['avg_original']
                                    result['avg_unc_numeric'] = result.get('avg_unc_original', result.get('avg_unc', 0.0))
                                    
                                    # Remove avg_original keys to prevent reuse
                                    del result['avg_original']
                                    if 'avg_unc_original' in result:
                                        del result['avg_unc_original']
                                # If avg_original doesn't exist, do nothing - values are already correct
                                break
    
    def get_ctr_for_film(self, film_name: str) -> Optional[List[str]]:
        """Get the CTR item IDs for a given film.
        
        Returns:
            List of CTR item IDs, or None if no CTRs for this film
        """
        return self.ctr_map.get(film_name)
    
    def is_ctr_circle(self, item_id: str) -> bool:
        """Check if an item is marked as CTR."""
        for ctr_ids in self.ctr_map.values():
            if item_id in ctr_ids:
                return True
        return False
    
    def clear_all_ctr(self):
        """Clear all CTR mappings."""
        for film_name, ctr_ids in list(self.ctr_map.items()):
            for ctr_id in ctr_ids:
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
