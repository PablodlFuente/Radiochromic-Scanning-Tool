"""
Calibration panel UI for the Radiochromic Film Analyzer.

This module contains the calibration panel UI class that displays and manages
the calibration controls and status.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging

logger = logging.getLogger(__name__)

class CalibrationPanel:
    """Calibration panel UI for the Radiochromic Film Analyzer."""
    
    def __init__(self, parent, image_processor):
        """Initialize the calibration panel."""
        self.parent = parent
        self.image_processor = image_processor
        
        # Create frame
        self.frame = ttk.Frame(parent)
        
        # Create UI
        self._create_ui()
        
        logger.info("Calibration panel initialized")
    
    def _create_ui(self):
        """Create the UI components."""
        # Calibration wizard button
        ttk.Button(
            self.frame, 
            text="Start Calibration Wizard", 
            command=self.show_calibration_wizard
        ).pack(padx=10, pady=10)
        
        # Calibration status
        self.status_label = ttk.Label(
            self.frame, 
            text="Status: Not calibrated"
        )
        self.status_label.pack(anchor=tk.W, padx=10, pady=5)
        
        # Update status
        self.update_calibration_status()
        
        # Calibration info
        ttk.Separator(self.frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(self.frame, text="Calibration Information:").pack(anchor=tk.W, padx=10, pady=5)
        
        # Info frame
        self.info_frame = ttk.Frame(self.frame)
        self.info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Info labels
        self.factor_label = ttk.Label(self.info_frame, text="Factor: --")
        self.factor_label.pack(anchor=tk.W, pady=2)
        
        self.offset_label = ttk.Label(self.info_frame, text="Offset: --")
        self.offset_label.pack(anchor=tk.W, pady=2)
        
        self.date_label = ttk.Label(self.info_frame, text="Date: --")
        self.date_label.pack(anchor=tk.W, pady=2)
        
        # Update info
        self.update_calibration_info()
    
    def show_calibration_wizard(self, callback=None):
        """Show the calibration wizard dialog."""
        if not self.image_processor.has_image():
            messagebox.showinfo("Information", "You must open an image first to calibrate.")
            return
        
        # Create wizard window
        wizard = tk.Toplevel(self.parent)
        wizard.title("Calibration Wizard")
        wizard.geometry("600x500")
        wizard.grab_set()  # Modal
        
        # Create wizard interface
        ttk.Label(
            wizard, 
            text="Radiochromic Film Calibration Wizard", 
            font=("Arial", 14, "bold")
        ).pack(pady=10)
        
        ttk.Label(
            wizard, 
            text="This wizard will guide you through the calibration process for radiochromic films."
        ).pack(pady=5)
        
        # Parameters frame
        params_frame = ttk.Frame(wizard)
        params_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Calibration parameters
        ttk.Label(params_frame, text="Known dose (Gy):").grid(row=0, column=0, sticky=tk.W, pady=5)
        dose_entry = ttk.Entry(params_frame)
        dose_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        dose_entry.insert(0, "0")
        
        ttk.Label(params_frame, text="Calibration factor:").grid(row=1, column=0, sticky=tk.W, pady=5)
        factor_entry = ttk.Entry(params_frame)
        factor_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        factor_entry.insert(0, "1.0")
        
        ttk.Label(params_frame, text="Offset:").grid(row=2, column=0, sticky=tk.W, pady=5)
        offset_entry = ttk.Entry(params_frame)
        offset_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        offset_entry.insert(0, "0.0")
        
        # Instructions
        ttk.Label(wizard, text="Instructions:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=20, pady=5)
        ttk.Label(wizard, text="1. Select an irradiated area in the image").pack(anchor=tk.W, padx=30)
        ttk.Label(wizard, text="2. Enter the known dose for that area").pack(anchor=tk.W, padx=30)
        ttk.Label(wizard, text="3. Adjust the calibration parameters").pack(anchor=tk.W, padx=30)
        ttk.Label(wizard, text="4. Save the calibration").pack(anchor=tk.W, padx=30)
        
        # Buttons
        buttons_frame = ttk.Frame(wizard)
        buttons_frame.pack(fill=tk.X, padx=20, pady=20)
        
        def save_calibration():
            try:
                # Get values
                dose = float(dose_entry.get())
                factor = float(factor_entry.get())
                offset = float(offset_entry.get())
                
                # Save calibration
                self.image_processor.set_calibration(dose, factor, offset)
                
                # Update UI
                self.update_calibration_status()
                self.update_calibration_info()
                
                # Apply calibration if callback provided
                if callback:
                    callback()
                
                # Close wizard
                wizard.destroy()
                
                # Show success message
                messagebox.showinfo("Success", "Calibration saved successfully.")
                
                logger.info("Calibration saved")
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numeric values.")
                logger.error("Invalid calibration values")
        
        ttk.Button(buttons_frame, text="Cancel", command=wizard.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttons_frame, text="Save Calibration", command=save_calibration).pack(side=tk.RIGHT, padx=5)
    
    def update_calibration_status(self):
        """Update the calibration status display."""
        if self.image_processor.has_calibration():
            self.status_label.config(text="Status: Calibrated")
        else:
            self.status_label.config(text="Status: Not calibrated")
    
    def update_calibration_info(self):
        """Update the calibration information display."""
        if self.image_processor.has_calibration():
            calibration = self.image_processor.get_calibration()
            self.factor_label.config(text=f"Factor: {calibration['factor']}")
            self.offset_label.config(text=f"Offset: {calibration['offset']}")
            self.date_label.config(text=f"Date: {calibration['date']}")
        else:
            self.factor_label.config(text="Factor: --")
            self.offset_label.config(text="Offset: --")
            self.date_label.config(text="Date: --")
