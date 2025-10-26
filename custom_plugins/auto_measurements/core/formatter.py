"""
Measurement value formatting utilities.

This module contains the MeasurementFormatter class for formatting
measurement values and uncertainties with proper significant figures.
"""

from typing import Tuple
import numpy as np


class MeasurementFormatter:
    """Utility class for formatting measurement values."""
    
    @staticmethod
    def format_significant(value: float, sig: int = 2) -> str:
        """Format number with the given significant figures."""
        if value == 0 or not np.isfinite(value):
            return "0"
        return f"{value:.{sig}g}"
    
    @staticmethod
    def clean_numeric_string(value_str: str) -> float:
        """Clean and parse numeric string that may contain ± symbols."""
        try:
            # Handle string or convert to string if needed
            if not isinstance(value_str, str):
                value_str = str(value_str)
            
            # Remove ± symbol and whitespace
            clean_str = value_str.replace("±", "").strip()
            
            # If multiple values separated by comma, take first one
            if "," in clean_str:
                clean_str = clean_str.split(",")[0].strip()
            
            return float(clean_str)
        except (ValueError, TypeError, AttributeError):
            return 0.0
    
    @staticmethod
    def format_value_uncertainty(value: float, unc: float, sig: int = 2, 
                                force_decimals: bool = False) -> Tuple[str, str]:
        """Format value and uncertainty with consistent decimal places.
        
        Args:
            value: The measured value
            unc: The uncertainty
            sig: Number of significant figures
            force_decimals: If True, always show 2 decimals
            
        Returns:
            Tuple of (formatted_value, formatted_uncertainty)
        """
        unc_fmt = MeasurementFormatter.format_significant(unc, sig)
        
        # Scientific notation – fall back to significant-figure formatting
        if "e" in unc_fmt or "E" in unc_fmt:
            val_fmt = MeasurementFormatter.format_significant(value, sig)
        else:
            # Count decimals in uncertainty
            if "." in unc_fmt:
                decimals = len(unc_fmt.split(".")[1])
            else:
                decimals = 0
            
            if force_decimals:
                decimals = max(decimals, 2)
            
            val_fmt = f"{value:.{decimals}f}"
            unc_fmt = f"{unc:.{decimals}f}"
            
        return val_fmt, unc_fmt
    
    @staticmethod
    def format_for_treeview(dose_values, sigma_values, avg_value, avg_unc_value, sig: int = 2):
        """Format measurement values for TreeView display with consistent decimals.
        
        Formatting rules (TreeView only, CSV uses full precision):
        - doses_per_channel: Always 3 decimals
        - STD_doses_per_channel: Always 3 decimals
        - SE_average: 2 significant figures
        - average: Same decimal places as SE_average
        - 95%confident_interval(SE): Same decimal places as SE_average
        
        Args:
            dose_values: List of dose values or single value or formatted string
            sigma_values: List of sigma values or single value or formatted string  
            avg_value: Average value (float)
            avg_unc_value: Average uncertainty/SE value (float)
            sig: Number of significant figures for SE (default 2)
            
        Returns:
            Tuple of (dose_str, sigma_str, avg_str, avg_unc_str, ci95_str)
        """
        # Format SE with 2 significant figures
        unc_fmt = MeasurementFormatter.format_significant(avg_unc_value, sig)
        
        # Count decimals in SE to match for avg and CI95
        if "." in unc_fmt:
            decimals = len(unc_fmt.split(".")[1])
        else:
            decimals = 0
        
        # Format avg with same decimals as SE
        avg_str = f"{avg_value:.{decimals}f}"
        avg_unc_str = f"±{unc_fmt}"
        
        # Calculate and format 95% CI with same decimals as SE
        ci95_value = avg_unc_value * 1.96
        ci95_str = f"±{ci95_value:.{decimals}f}"
        
        # Handle dose and sigma - ALWAYS 3 decimals for TreeView
        if isinstance(dose_values, str):
            # Already formatted as string, keep it
            dose_str = dose_values
            sigma_str = sigma_values if isinstance(sigma_values, str) else f"±{sigma_values:.3f}"
        elif isinstance(sigma_values, (list, tuple)) and len(sigma_values) > 1:
            # Multi-channel: format all with 3 decimals
            sigma_parts = []
            dose_parts = []
            
            # Ensure dose_values is a list
            if isinstance(dose_values, (list, tuple)):
                dose_list = list(dose_values)
            else:
                dose_list = [dose_values] * len(sigma_values)
            
            # Format each channel with 3 decimals
            for dose_val, sigma_val in zip(dose_list, sigma_values):
                dose_parts.append(f"{dose_val:.3f}")
                sigma_parts.append(f"±{sigma_val:.3f}")
            
            dose_str = ", ".join(dose_parts)
            sigma_str = ", ".join(sigma_parts)
        elif isinstance(sigma_values, (list, tuple)) and len(sigma_values) == 1:
            # Single channel in list format - 3 decimals
            dose_str = f"{dose_values[0]:.3f}"
            sigma_str = f"±{sigma_values[0]:.3f}"
        else:
            # Single value (not a list) - 3 decimals
            dose_str = f"{dose_values:.3f}"
            sigma_str = f"±{sigma_values:.3f}"
        
        return dose_str, sigma_str, avg_str, avg_unc_str, ci95_str



__all__ = ['MeasurementFormatter']
