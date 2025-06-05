"""
Settings panel UI for the Radiochromic Film Analyzer.

This module contains the settings panel UI class that displays and manages
the application settings.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
import os

logger = logging.getLogger(__name__)

class SettingsPanel:
    """Settings panel UI for the Radiochromic Film Analyzer."""
    
    def __init__(self, parent, app_config, apply_callback=None):
        """Initialize the settings panel."""
        self.parent = parent
        self.app_config = app_config
        self.apply_callback = apply_callback
        
        # Create frame
        self.frame = ttk.Frame(parent)
        
        # Create UI
        self._create_ui()
        
        logger.info("Settings panel initialized")
    
    def _create_ui(self):
        """Create the UI components."""
        # Settings title
        ttk.Label(
            self.frame, 
            text="Application Settings", 
            font=("Arial", 12, "bold")
        ).pack(anchor=tk.W, padx=10, pady=10)
        
        # Image processing section
        ttk.Label(
            self.frame, 
            text="Image Processing", 
            font=("Arial", 10, "bold")
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # Remove background option
        self.remove_background_var = tk.BooleanVar(
            value=self.app_config.get("remove_background", False)
        )
        
        ttk.Checkbutton(
            self.frame, 
            text="Remove white background when loading images", 
            variable=self.remove_background_var
        ).pack(anchor=tk.W, padx=20, pady=2)
        
        ttk.Label(
            self.frame, 
            text="This option detects and crops the useful area of the image,\n"
                 "removing the white background to reduce size.",
            foreground="gray"
        ).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        # Visualization section
        ttk.Label(
            self.frame, 
            text="Visualization", 
            font=("Arial", 10, "bold")
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # Colormap selection
        ttk.Label(
            self.frame, 
            text="Default colormap:"
        ).pack(anchor=tk.W, padx=20, pady=2)
        
        # Available colormaps
        self.colormap_var = tk.StringVar(
            value=self.app_config.get("colormap", "viridis")
        )
        
        colormap_frame = ttk.Frame(self.frame)
        colormap_frame.pack(fill=tk.X, padx=20, pady=2)
        
        self.colormap_combobox = ttk.Combobox(
            colormap_frame,
            textvariable=self.colormap_var,
            values=["viridis", "plasma", "inferno", "magma", "cividis", "jet", "rainbow", "turbo", "hot", "cool"]
        )
        self.colormap_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Optimization section
        ttk.Label(
            self.frame, 
            text="Performance Optimization", 
            font=("Arial", 10, "bold")
        ).pack(anchor=tk.W, padx=10, pady=(20, 5))
        
        # GPU acceleration option
        self.use_gpu_var = tk.BooleanVar(
            value=self.app_config.get("use_gpu", False)
        )
        
        self.gpu_checkbox = ttk.Checkbutton(
            self.frame, 
            text="Use GPU acceleration (if available)", 
            variable=self.use_gpu_var
        )
        self.gpu_checkbox.pack(anchor=tk.W, padx=20, pady=2)
        
        # Force GPU option
        self.gpu_force_enabled_var = tk.BooleanVar(
            value=self.app_config.get("gpu_force_enabled", False)
        )
        
        self.gpu_force_checkbox = ttk.Checkbutton(
            self.frame, 
            text="Force enable GPU (use when GPU is available but not detected)", 
            variable=self.gpu_force_enabled_var,
            command=self._on_force_gpu_change
        )
        self.gpu_force_checkbox.pack(anchor=tk.W, padx=40, pady=2)
        # Always enable the force GPU checkbox
        self.gpu_force_checkbox.config(state=tk.NORMAL)
        
        # GPU status label
        self.gpu_status_label = ttk.Label(
            self.frame,
            text="GPU status: Checking...",
            foreground="gray"
        )
        self.gpu_status_label.pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # Check GPU availability after UI is created
        self.frame.after(100, self._check_gpu_availability)
        
        # Multi-threading option
        self.use_multithreading_var = tk.BooleanVar(
            value=self.app_config.get("use_multithreading", True)
        )
        
        ttk.Checkbutton(
            self.frame, 
            text="Use multi-threading for image processing", 
            variable=self.use_multithreading_var
        ).pack(anchor=tk.W, padx=20, pady=2)
        
        # Thread count
        thread_frame = ttk.Frame(self.frame)
        thread_frame.pack(fill=tk.X, padx=20, pady=2)
        
        ttk.Label(
            thread_frame,
            text="Number of threads:"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Get CPU count
        cpu_count = os.cpu_count() or 4
        
        # Ensure num_threads doesn't exceed available cores
        initial_threads = min(self.app_config.get("num_threads", cpu_count), cpu_count)
        self.num_threads_var = tk.IntVar(value=initial_threads)
        
        self.thread_spinbox = ttk.Spinbox(
            thread_frame,
            from_=1,
            to=cpu_count,  # Limit to actual CPU count
            textvariable=self.num_threads_var,
            width=5
        )
        self.thread_spinbox.pack(side=tk.LEFT)
        
        ttk.Label(
            thread_frame,
            text=f"(System has {cpu_count} CPU cores)",
            foreground="gray"
        ).pack(side=tk.LEFT, padx=10)

        # Apply button
        ttk.Button(
            self.frame, 
            text="Apply Settings", 
            command=self._apply_settings
        ).pack(anchor=tk.CENTER, pady=20)
    
    def _on_force_gpu_change(self):
        """Handle force GPU checkbox change."""
        if self.gpu_force_enabled_var.get():
            # Enable GPU checkbox
            self.gpu_checkbox.config(state=tk.NORMAL)
            self.use_gpu_var.set(True)
            
            # Update status label
            self.gpu_status_label.config(
                text="GPU status: Force-enabled by user",
                foreground="blue"
            )
        else:
            # Re-check GPU availability
            self._check_gpu_availability()
    
    def _check_gpu_availability(self):
        """Check GPU availability and update UI accordingly."""
        try:
            import cv2
            
            # If force-enabled, don't check actual GPU
            if self.gpu_force_enabled_var.get():
                self.gpu_status_label.config(
                    text="GPU status: Force-enabled by user",
                    foreground="blue"
                )
                self.gpu_checkbox.config(state=tk.NORMAL)
                return
            
            # Try multiple methods to detect CUDA
            cuda_detected = False
            
            # Method 1: Standard OpenCV CUDA detection
            try:
                if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    cuda_detected = True
                    gpu_name = cv2.cuda.getDeviceName(0)
                    self.gpu_status_label.config(
                        text=f"GPU available: {gpu_name}",
                        foreground="green"
                    )
            except Exception as e:
                logger.warning(f"Standard CUDA detection failed: {str(e)}")
            
            # Method 2: Try to create a simple CUDA matrix with better error handling
            if not cuda_detected:
                try:
                    import numpy as np
                    # Create a small test matrix
                    test_array = np.zeros((10, 10), dtype=np.uint8)
                    
                    # Try to create a GpuMat and upload data
                    gpu_mat = cv2.cuda_GpuMat()
                    gpu_mat.upload(test_array)
                    
                    # Try to download to verify it works
                    result = gpu_mat.download()
                    
                    if result is not None:
                        cuda_detected = True
                        self.gpu_status_label.config(
                            text="GPU available: Detected via test matrix",
                            foreground="green"
                        )
                except Exception as e:
                    logger.warning(f"CUDA test matrix creation failed: {str(e)}")
            
            # Method 3: Check for NVIDIA GPU using platform-specific methods
            if not cuda_detected:
                import platform
                if platform.system() == "Windows":
                    try:
                        import subprocess
                        result = subprocess.run(["wmic", "path", "win32_VideoController", "get", "name"], 
                                               capture_output=True, text=True)
                        output = result.stdout.lower()
                        if "nvidia" in output:
                            cuda_detected = True
                            
                            # Extract the actual GPU name from the output
                            lines = [line.strip() for line in output.split('\n') if line.strip() and "name" not in line.lower()]
                            gpu_name = next((line for line in lines if "nvidia" in line.lower()), "NVIDIA GPU")
                            
                            self.gpu_status_label.config(
                                text=f"NVIDIA GPU detected ({gpu_name}) but CUDA not available. Use 'Force enable GPU' option.",
                                foreground="orange"
                            )
                    except Exception as e:
                        logger.warning(f"Windows GPU detection failed: {str(e)}")
                elif platform.system() == "Linux":
                    try:
                        import subprocess
                        result = subprocess.run(["lspci"], capture_output=True, text=True)
                        output = result.stdout.lower()
                        if "nvidia" in output:
                            cuda_detected = True
                            self.gpu_status_label.config(
                                text="NVIDIA GPU detected but CUDA not available. Use 'Force enable GPU' option.",
                                foreground="orange"
                            )
                    except Exception as e:
                        logger.warning(f"Linux GPU detection failed: {str(e)}")
            
            # Update UI based on detection results
            if cuda_detected:
                self.gpu_checkbox.config(state=tk.NORMAL)
            else:
                # No GPU available
                self.gpu_status_label.config(
                    text="No CUDA-capable GPU detected. Use 'Force enable GPU' option if needed.",
                    foreground="red"
                )
                # Only disable the GPU checkbox if Force GPU is not enabled
                if not self.gpu_force_enabled_var.get():
                    self.gpu_checkbox.config(state=tk.DISABLED)
                    self.use_gpu_var.set(False)
        except Exception as e:
            # Error checking GPU
            self.gpu_status_label.config(
                text="Error checking GPU availability",
                foreground="red"
            )
            self.gpu_checkbox.config(state=tk.DISABLED)
            self.use_gpu_var.set(False)
            logger.error(f"Error checking GPU: {str(e)}", exc_info=True)

    def show_settings_dialog(self):
        """Show the settings dialog."""
        # Create settings window
        settings_window = tk.Toplevel(self.parent)
        settings_window.title("Settings")
        settings_window.geometry("500x700")  # 40% taller
        settings_window.grab_set()  # Modal
        
        # Main frame with scrollbar for very large content
        main_container = ttk.Frame(settings_window)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Add a canvas with scrollbar
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Main frame
        main_frame = ttk.Frame(scrollable_frame, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(
            main_frame, 
            text="Settings", 
            font=("Arial", 14, "bold")
        ).pack(anchor=tk.W, pady=(0, 20))
        
        # Image processing section
        ttk.Label(
            main_frame, 
            text="Image Processing", 
            font=("Arial", 11, "bold")
        ).pack(anchor=tk.W, pady=(10, 5))
        
        # Remove background option
        remove_background_var = tk.BooleanVar(
            value=self.remove_background_var.get()
        )
        
        ttk.Checkbutton(
            main_frame, 
            text="Remove white background when loading images", 
            variable=remove_background_var
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        ttk.Label(
            main_frame, 
            text="This option detects and crops the useful area of the image,\n"
             "removing the white background to reduce size."
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        # Optimization section
        ttk.Label(
            main_frame, 
            text="Performance Optimization", 
            font=("Arial", 11, "bold")
        ).pack(anchor=tk.W, pady=(20, 5))
        
        # GPU acceleration option
        use_gpu_var = tk.BooleanVar(
            value=self.use_gpu_var.get()
        )
        
        gpu_checkbox = ttk.Checkbutton(
            main_frame, 
            text="Use GPU acceleration (if available)", 
            variable=use_gpu_var
        )
        gpu_checkbox.pack(anchor=tk.W, padx=10, pady=5)
        
        # Force GPU option
        gpu_force_enabled_var = tk.BooleanVar(
            value=self.gpu_force_enabled_var.get()
        )
        
        force_gpu_checkbox = ttk.Checkbutton(
            main_frame, 
            text="Force enable GPU (use if GPU not detected)", 
            variable=gpu_force_enabled_var
        )
        force_gpu_checkbox.pack(anchor=tk.W, padx=30, pady=2)
        force_gpu_checkbox.config(state=tk.NORMAL)
        
        # Multi-threading option
        use_multithreading_var = tk.BooleanVar(
            value=self.use_multithreading_var.get()
        )
        
        ttk.Checkbutton(
            main_frame, 
            text="Use multi-threading for image processing", 
            variable=use_multithreading_var
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        # Thread count
        thread_frame = ttk.Frame(main_frame)
        thread_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(
            thread_frame,
            text="Number of threads:"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Get CPU count
        cpu_count = os.cpu_count() or 4
        
        num_threads_var = tk.IntVar(
            value=self.num_threads_var.get()
        )
        
        thread_spinbox = ttk.Spinbox(
            thread_frame,
            from_=1,
            to=cpu_count,  # Limit to actual CPU count
            textvariable=num_threads_var,
            width=5
        )
        thread_spinbox.pack(side=tk.LEFT)
        
        ttk.Label(
            thread_frame,
            text=f"(System has {cpu_count} CPU cores)"
        ).pack(side=tk.LEFT, padx=10)
        
        # Visualization section
        ttk.Label(
            main_frame, 
            text="Visualization", 
            font=("Arial", 11, "bold")
        ).pack(anchor=tk.W, pady=(20, 5))
        
        # Colormap selection
        ttk.Label(
            main_frame, 
            text="Default colormap:"
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        # Available colormaps
        colormap_var = tk.StringVar(
            value=self.colormap_var.get()
        )
        
        colormap_combobox = ttk.Combobox(
            main_frame,
            textvariable=colormap_var,
            values=["viridis", "plasma", "inferno", "magma", "cividis", "jet", "rainbow", "turbo", "hot", "cool"],
            width=15
        )
        colormap_combobox.pack(anchor=tk.W, padx=30, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        def save_settings():
            # Update variables
            self.remove_background_var.set(remove_background_var.get())
            self.use_gpu_var.set(use_gpu_var.get())
            self.gpu_force_enabled_var.set(gpu_force_enabled_var.get())
            self.use_multithreading_var.set(use_multithreading_var.get())
            self.num_threads_var.set(num_threads_var.get())
            self.colormap_var.set(colormap_var.get())
        
        # Apply settings
            self._apply_settings()
        
        # Close window
            settings_window.destroy()
        
        # Show success message
            messagebox.showinfo("Settings", "Settings saved and applied successfully.")
    
    ttk.Button(
        button_frame, 
        text="Cancel", 
        command=settings_window.destroy
    ).pack(side=tk.RIGHT, padx=5)
    
    ttk.Button(
        button_frame, 
        text="Save & Apply", 
        command=save_settings
    ).pack(side=tk.RIGHT, padx=5)
    
    def _apply_settings(self):
        """Apply the current settings."""
        # Update config
        self.app_config["remove_background"] = self.remove_background_var.get()
        self.app_config["colormap"] = self.colormap_var.get()
        self.app_config["use_gpu"] = self.use_gpu_var.get()
        self.app_config["gpu_force_enabled"] = self.gpu_force_enabled_var.get()
        self.app_config["use_multithreading"] = self.use_multithreading_var.get()
        self.app_config["num_threads"] = self.num_threads_var.get()
        
        # Call apply callback if provided
        if self.apply_callback:
            self.apply_callback()
        
        logger.info("Settings applied")
    
    def get_settings(self):
        """Get the current settings."""
        return {
            "remove_background": self.remove_background_var.get(),
            "colormap": self.colormap_var.get(),
            "use_gpu": self.use_gpu_var.get(),
            "gpu_force_enabled": self.gpu_force_enabled_var.get(),
            "use_multithreading": self.use_multithreading_var.get(),
            "num_threads": self.num_threads_var.get()
        }
