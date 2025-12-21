"""Scanner Calibration Wizard - Field Flattening & Dose Calibration

This wizard guides the user through:
1. Introduction - Explanation of the calibration process
2. Blank Scan Loading - Load and average blank scans for uniformity analysis
3. Uniformity Analysis - Study scanner anisotropies and save field flattening data
4. Dose Calibration - RGB to dose transformation (existing calibration workflow)

All calibration data is stored in the `calibration_data` directory.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from importlib import import_module
from types import ModuleType
from typing import Optional, List
import numpy as np
from PIL import Image
import logging

# For matplotlib plots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Fully-qualified import path of the internal calibration module
_MODULE_NAME = "app.calibration.calibration_app"


def _load_external_calibration_module() -> ModuleType:
    """Import (or retrieve) the internal *calibration_app* module."""
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    module = import_module(_MODULE_NAME)
    return module


class ScannerCalibrationWizard(tk.Toplevel):
    """Multi-step wizard for scanner calibration including field flattening."""

    def __init__(self, parent: tk.Misc, app_config: Optional[dict] = None):
        """Initialize the scanner calibration wizard.
        
        Args:
            parent: Parent window
            app_config: Optional application config dict. If provided, uses this
                instead of reading from disk, ensuring consistency with the
                current session settings.
        """
        super().__init__(parent)
        self.title("Calibrate Scanner")
        self.geometry("1000x700")
        self.minsize(800, 600)
        
        # Store config reference
        self._app_config = app_config

        # Make modal
        try:
            self.transient(parent)
            self.grab_set()
        except Exception:
            pass

        # Data directory for calibration outputs - use configured folder from settings
        calibration_folder = self._get_calibration_folder()
        base_dir = os.path.join(os.getcwd(), "calibration_data")
        if calibration_folder != "default":
            self._data_dir = os.path.join(base_dir, calibration_folder)
        else:
            self._data_dir = base_dir
        os.makedirs(self._data_dir, exist_ok=True)

        # Wizard state
        self.current_step = 0
        self.blank_images: List[np.ndarray] = []  # Loaded blank scans
        self.blank_filenames: List[str] = []  # Filenames for display
        self.averaged_blank: Optional[np.ndarray] = None  # Averaged blank image
        self.flat_field_data: Optional[dict] = None  # Field flattening data

        # Build UI
        self._create_widgets()
        self._show_step(0)

        # Handle close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _get_calibration_folder(self) -> str:
        """Get the calibration folder from config.
        
        Uses the app_config passed at initialization if available,
        otherwise falls back to reading from disk.
        """
        # Prefer in-memory config for consistency with current session
        if self._app_config is not None:
            return self._app_config.get("calibration_folder", "default")
        
        # Fallback: read from disk
        import json
        config_file = os.path.join(os.getcwd(), "rc_config.json")
        try:
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config = json.load(f)
                return config.get("calibration_folder", "default")
        except Exception:
            pass
        return "default"

    def _create_widgets(self):
        """Create the wizard UI structure."""
        # Main container
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Step indicator at top
        self.step_indicator = ttk.Frame(self.main_frame)
        self.step_indicator.pack(fill=tk.X, pady=(0, 10))

        self.step_labels = []
        step_names = ["Introduction", "Load Blank Scans", "Uniformity Analysis", "Dose Calibration"]
        for i, name in enumerate(step_names):
            lbl = ttk.Label(self.step_indicator, text=f"{i+1}. {name}", font=("Arial", 9))
            lbl.pack(side=tk.LEFT, padx=10)
            self.step_labels.append(lbl)

        ttk.Separator(self.main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Content area (will be replaced for each step)
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # Navigation buttons at bottom
        ttk.Separator(self.main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        self.nav_frame = ttk.Frame(self.main_frame)
        self.nav_frame.pack(fill=tk.X, pady=(10, 0))

        self.back_btn = ttk.Button(self.nav_frame, text="← Back", command=self._go_back)
        self.back_btn.pack(side=tk.LEFT)

        self.next_btn = ttk.Button(self.nav_frame, text="Next →", command=self._go_next)
        self.next_btn.pack(side=tk.RIGHT)

        self.cancel_btn = ttk.Button(self.nav_frame, text="Cancel", command=self._on_close)
        self.cancel_btn.pack(side=tk.RIGHT, padx=(0, 10))

    def _update_step_indicator(self):
        """Highlight the current step in the indicator."""
        for i, lbl in enumerate(self.step_labels):
            if i == self.current_step:
                lbl.configure(font=("Arial", 10, "bold"), foreground="blue")
            elif i < self.current_step:
                lbl.configure(font=("Arial", 9), foreground="green")
            else:
                lbl.configure(font=("Arial", 9), foreground="gray")

    def _clear_content(self):
        """Clear the content frame."""
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def _show_step(self, step: int):
        """Display the specified step."""
        self.current_step = step
        self._clear_content()
        self._update_step_indicator()

        # Update navigation buttons
        self.back_btn.configure(state=tk.NORMAL if step > 0 else tk.DISABLED)

        if step == 0:
            self._show_introduction()
        elif step == 1:
            self._show_blank_scan_loader()
        elif step == 2:
            self._show_uniformity_analysis()
        elif step == 3:
            self._show_dose_calibration()

    def _go_back(self):
        """Go to previous step."""
        if self.current_step > 0:
            self._show_step(self.current_step - 1)

    def _go_next(self):
        """Go to next step with validation."""
        if self.current_step == 0:
            # Introduction - no validation needed
            self._show_step(1)
        elif self.current_step == 1:
            # Blank scan loader - need at least one image
            if not self.blank_images:
                messagebox.showwarning("Calibrate Scanner", 
                    "Please load at least one blank scan image before proceeding.")
                return
            self._compute_averaged_blank()
            self._show_step(2)
        elif self.current_step == 2:
            # Uniformity analysis - save field flattening
            self._save_field_flattening()
            self._show_step(3)
        elif self.current_step == 3:
            # Final step - close wizard
            self._on_close()

    # =========================================================================
    # STEP 0: Introduction
    # =========================================================================

    def _show_introduction(self):
        """Show the introduction screen."""
        self.next_btn.configure(text="Next →")

        frame = ttk.Frame(self.content_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        ttk.Label(frame, text="Scanner Calibration Wizard", 
                 font=("Arial", 16, "bold")).pack(pady=(0, 20))

        # Explanation text
        explanation = """The Scanner Calibration window allows you to study and correct 
optical anisotropies of your scanner, as well as obtain the 
RGB-to-dose transformation required for dosimetry.

This wizard will guide you through the following steps:

1. Load Blank Scans
   Upload one or more images of a blank scan (empty scanner bed).
   If multiple images are provided, they will be stacked and averaged
   to reduce noise.

2. Uniformity Analysis
   Analyze the scanner's uniformity across the imaging area.
   View statistics and 2D uniformity maps for each color channel (R, G, B).
   This data will be used for field flattening correction.

3. Dose Calibration
   Perform the RGB-to-dose calibration using radiochromic film samples
   with known doses. Images will be automatically flattened using the
   data from step 2.

Click "Next" to begin the calibration process."""

        text_widget = tk.Text(frame, wrap=tk.WORD, height=20, width=70, 
                             font=("Arial", 11), bg=self.cget("bg"), 
                             relief=tk.FLAT, padx=10, pady=10)
        text_widget.insert("1.0", explanation)
        text_widget.configure(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)

    # =========================================================================
    # STEP 1: Load Blank Scans
    # =========================================================================

    def _show_blank_scan_loader(self):
        """Show the blank scan loading interface."""
        self.next_btn.configure(text="Next →")

        frame = ttk.Frame(self.content_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Instructions
        ttk.Label(frame, text="Load Blank Scan Images", 
                 font=("Arial", 14, "bold")).pack(pady=(0, 10))

        ttk.Label(frame, text="Select one or more images of a blank scan (empty scanner bed).\n"
                 "Multiple images will be averaged to reduce noise.",
                 font=("Arial", 10)).pack(pady=(0, 15))

        # Buttons frame
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(btn_frame, text="Add Blank Scan(s)...", 
                  command=self._load_blank_scans).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear All", 
                  command=self._clear_blank_scans).pack(side=tk.LEFT, padx=5)

        # Status label
        self.blank_status_var = tk.StringVar(value="No images loaded")
        ttk.Label(btn_frame, textvariable=self.blank_status_var, 
                 font=("Arial", 10, "italic")).pack(side=tk.LEFT, padx=20)

        # Listbox for loaded files
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        ttk.Label(list_frame, text="Loaded images:").pack(anchor=tk.W)

        listbox_container = ttk.Frame(list_frame)
        listbox_container.pack(fill=tk.BOTH, expand=True, pady=5)

        self.blank_listbox = tk.Listbox(listbox_container, height=10)
        self.blank_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(listbox_container, orient=tk.VERTICAL, 
                                  command=self.blank_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.blank_listbox.configure(yscrollcommand=scrollbar.set)

        # Update UI with current state
        self._update_blank_scan_ui()

    def _load_blank_scans(self):
        """Open file dialog to load blank scan images."""
        filetypes = [
            ("TIFF files", "*.tif *.tiff"),
            ("All image files", "*.tif *.tiff *.png *.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        files = filedialog.askopenfilenames(
            title="Select Blank Scan Image(s)",
            filetypes=filetypes
        )

        if not files:
            return

        for filepath in files:
            try:
                # Load image
                img = Image.open(filepath)
                img_array = np.array(img)

                # Validate: must be RGB or grayscale
                if len(img_array.shape) == 2:
                    # Grayscale - convert to RGB
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                    # RGBA - drop alpha
                    img_array = img_array[:, :, :3]
                elif len(img_array.shape) != 3 or img_array.shape[2] != 3:
                    messagebox.showwarning("Load Error", 
                        f"Image format not supported: {os.path.basename(filepath)}")
                    continue

                # Validate: all images must have same shape
                if self.blank_images:
                    if img_array.shape != self.blank_images[0].shape:
                        messagebox.showwarning("Load Error",
                            f"Image '{os.path.basename(filepath)}' has different dimensions.\n"
                            f"Expected: {self.blank_images[0].shape[:2]}, "
                            f"Got: {img_array.shape[:2]}")
                        continue

                self.blank_images.append(img_array)
                self.blank_filenames.append(os.path.basename(filepath))
                self.blank_listbox.insert(tk.END, os.path.basename(filepath))

            except Exception as e:
                messagebox.showerror("Load Error", 
                    f"Failed to load '{os.path.basename(filepath)}':\n{e}")

        self._update_blank_scan_ui()

    def _clear_blank_scans(self):
        """Clear all loaded blank scans."""
        self.blank_images.clear()
        self.blank_filenames.clear()
        self.blank_listbox.delete(0, tk.END)
        self._update_blank_scan_ui()

    def _update_blank_scan_ui(self):
        """Update the UI to reflect current blank scan state."""
        count = len(self.blank_images)
        if count == 0:
            self.blank_status_var.set("No images loaded")
        elif count == 1:
            self.blank_status_var.set("1 image loaded")
        else:
            self.blank_status_var.set(f"{count} images loaded (will be averaged)")

    def _compute_averaged_blank(self):
        """Compute the averaged blank image from loaded scans."""
        if not self.blank_images:
            return

        if len(self.blank_images) == 1:
            self.averaged_blank = self.blank_images[0].astype(np.float64)
        else:
            # Stack and average
            stacked = np.stack(self.blank_images, axis=0).astype(np.float64)
            self.averaged_blank = np.mean(stacked, axis=0)

        logger.info(f"Computed averaged blank from {len(self.blank_images)} images")

    # =========================================================================
    # STEP 2: Uniformity Analysis
    # =========================================================================

    def _show_uniformity_analysis(self):
        """Show the uniformity analysis screen."""
        self.next_btn.configure(text="Apply Field Flattening →")

        if self.averaged_blank is None:
            ttk.Label(self.content_frame, text="Error: No blank image available",
                     foreground="red").pack(pady=20)
            return

        frame = ttk.Frame(self.content_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        ttk.Label(frame, text="Scanner Uniformity Analysis", 
                 font=("Arial", 14, "bold")).pack(pady=(0, 10))

        # Statistics frame
        stats_frame = ttk.LabelFrame(frame, text="Channel Statistics")
        stats_frame.pack(fill=tk.X, pady=10, padx=5)

        # Calculate statistics
        img = self.averaged_blank
        channel_names = ["Red", "Green", "Blue"]
        stats = []

        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(stats_grid, text="Channel", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=10)
        ttk.Label(stats_grid, text="Mean", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=10)
        ttk.Label(stats_grid, text="Std Dev", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=10)
        ttk.Label(stats_grid, text="Min", font=("Arial", 10, "bold")).grid(row=0, column=3, padx=10)
        ttk.Label(stats_grid, text="Max", font=("Arial", 10, "bold")).grid(row=0, column=4, padx=10)
        ttk.Label(stats_grid, text="Uniformity %", font=("Arial", 10, "bold")).grid(row=0, column=5, padx=10)

        for i, name in enumerate(channel_names):
            channel = img[:, :, i]
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            min_val = np.min(channel)
            max_val = np.max(channel)
            uniformity = (1 - std_val / mean_val) * 100 if mean_val > 0 else 0

            stats.append({
                "name": name,
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "uniformity": uniformity
            })

            ttk.Label(stats_grid, text=name).grid(row=i+1, column=0, padx=10)
            ttk.Label(stats_grid, text=f"{mean_val:.1f}").grid(row=i+1, column=1, padx=10)
            ttk.Label(stats_grid, text=f"{std_val:.2f}").grid(row=i+1, column=2, padx=10)
            ttk.Label(stats_grid, text=f"{min_val:.0f}").grid(row=i+1, column=3, padx=10)
            ttk.Label(stats_grid, text=f"{max_val:.0f}").grid(row=i+1, column=4, padx=10)
            ttk.Label(stats_grid, text=f"{uniformity:.2f}%").grid(row=i+1, column=5, padx=10)

        # Store stats for saving
        self._uniformity_stats = stats

        # Create 2D uniformity plots
        plot_frame = ttk.LabelFrame(frame, text="2D Uniformity Maps")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        # Create matplotlib figure
        fig = Figure(figsize=(10, 3), dpi=100)
        colors = ['Reds', 'Greens', 'Blues']

        for i, (name, cmap) in enumerate(zip(channel_names, colors)):
            ax = fig.add_subplot(1, 3, i + 1)
            channel = img[:, :, i]

            # Normalize to show variations
            normalized = channel / np.mean(channel)

            im = ax.imshow(normalized, cmap=cmap, vmin=0.95, vmax=1.05)
            ax.set_title(f"{name} Channel")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            fig.colorbar(im, ax=ax, label="Relative intensity")

        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Info label
        ttk.Label(frame, 
            text="Click 'Apply Field Flattening' to save this data and proceed to dose calibration.\n"
                 "The flattening correction will be applied to all images when 'Apply Calibration' is enabled.",
            font=("Arial", 9, "italic"), foreground="gray").pack(pady=10)

    def _save_field_flattening(self):
        """Save the field flattening data to disk."""
        if self.averaged_blank is None:
            return

        try:
            # Compute normalized flat field
            img = self.averaged_blank
            mean_per_channel = np.mean(img, axis=(0, 1))

            # Normalize: flat_field where mean = 1.0 per channel
            flat_field = img / mean_per_channel

            # Prepare data structure
            from datetime import datetime
            self.flat_field_data = {
                "flat_field": flat_field,
                "mean_per_channel": mean_per_channel.tolist(),
                "std_per_channel": [np.std(img[:, :, i]) for i in range(3)],
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_images_averaged": len(self.blank_images),
                "image_shape": img.shape[:2]
            }

            # Save to file
            save_path = os.path.join(self._data_dir, "field_flattening.npz")
            np.savez(save_path,
                flat_field=flat_field,
                mean_per_channel=mean_per_channel,
                std_per_channel=self.flat_field_data["std_per_channel"],
                date_created=self.flat_field_data["date_created"],
                num_images_averaged=self.flat_field_data["num_images_averaged"],
                image_shape=self.flat_field_data["image_shape"]
            )

            logger.info(f"Field flattening data saved to: {save_path}")
            messagebox.showinfo("Field Flattening", 
                f"Field flattening data saved successfully.\n\n"
                f"Images averaged: {len(self.blank_images)}\n"
                f"File: field_flattening.npz")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save field flattening data:\n{e}")
            logger.error(f"Failed to save field flattening: {e}", exc_info=True)

    # =========================================================================
    # STEP 3: Dose Calibration (existing workflow)
    # =========================================================================

    def _show_dose_calibration(self):
        """Show the dose calibration step using the existing calibration app."""
        self.next_btn.configure(text="Finish")

        frame = ttk.Frame(self.content_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        ttk.Label(frame, text="Dose Calibration (RGB → Dose)", 
                 font=("Arial", 14, "bold")).pack(pady=(0, 15))

        # Info about flattening
        if self.flat_field_data:
            ttk.Label(frame, 
                text="✓ Field flattening data has been saved successfully.\n"
                     "   This correction will be applied automatically when 'Apply Calibration' is enabled.",
                font=("Arial", 10), foreground="green").pack(pady=10, anchor=tk.W)
        
        # Explanation
        explanation = """The dose calibration window will open separately.

In that window you will:
1. Load radiochromic film images with known doses
2. Define ROIs (Regions of Interest) on each film
3. Enter the corresponding dose values
4. Generate the calibration curve (RGB → Dose)

The calibration parameters will be saved to the calibration_data folder
and used when 'Apply Calibration' is enabled in the main application.

Click 'Open Dose Calibration' to launch the calibration tool."""

        text_widget = tk.Text(frame, wrap=tk.WORD, height=14, width=60, 
                             font=("Arial", 10), bg=self.cget("bg"), 
                             relief=tk.FLAT, padx=10, pady=10)
        text_widget.insert("1.0", explanation)
        text_widget.configure(state=tk.DISABLED)
        text_widget.pack(fill=tk.X, pady=10)

        # Button to launch calibration app
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Open Dose Calibration", 
                  command=self._launch_calibration_app).pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.cal_status_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.cal_status_var, 
                 font=("Arial", 10, "italic")).pack(pady=10)

    def _launch_calibration_app(self):
        """Launch the calibration app in a separate window."""
        # Preserve original working directory
        self._old_cwd = os.getcwd()
        os.chdir(self._data_dir)

        try:
            module = _load_external_calibration_module()
            if not hasattr(module, "CalibrationApp"):
                raise AttributeError("The calibration module does not contain 'CalibrationApp' class.")

            CalibrationApp = module.CalibrationApp

            # Create a new Toplevel window for the calibration app
            cal_window = tk.Toplevel(self)
            cal_window.title("Dose Calibration")
            cal_window.geometry("1200x800")
            
            # The CalibrationApp expects a root-like widget
            self._calibration_app = CalibrationApp(cal_window)
            
            self.cal_status_var.set("Dose calibration window opened.")
            
            # When calibration window closes, restore cwd
            def on_cal_close():
                try:
                    os.chdir(self._old_cwd)
                except:
                    pass
                cal_window.destroy()
            
            cal_window.protocol("WM_DELETE_WINDOW", on_cal_close)

        except Exception as exc:
            os.chdir(self._old_cwd)
            self.cal_status_var.set(f"Error: {exc}")
            logger.error(f"Failed to load calibration app: {exc}", exc_info=True)
            messagebox.showerror("Calibration Error", 
                f"Failed to open dose calibration:\n{exc}")

    # =========================================================================
    # Cleanup
    # =========================================================================

    def _on_close(self):
        """Handle wizard close."""
        # Restore working directory if changed
        if hasattr(self, '_old_cwd'):
            try:
                os.chdir(self._old_cwd)
            except Exception:
                pass
        self.destroy()


# Keep the old class name for backward compatibility
CalibrationWizardWindow = ScannerCalibrationWizard


__all__ = ['ScannerCalibrationWizard', 'CalibrationWizardWindow']
