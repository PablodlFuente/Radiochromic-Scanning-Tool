"""
Main window UI for the Radiochromic Film Analyzer.

This module contains the main window UI class that creates and manages
the main application window and its components.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import os
import threading
from app.ui.measurement_panel import MeasurementPanel
from app.ui.image_settings_panel import ImageSettingsPanel
from app.ui.image_panel import ImagePanel
from app.core.image_processor import ImageProcessor
from app.utils.file_manager import FileManager

logger = logging.getLogger(__name__)

class MainWindow:
    """Main window UI for the Radiochromic Film Analyzer."""
    
    def __init__(self, parent, app_config):
        """Initialize the main window."""
        self.parent = parent
        self.app_config = app_config
        
        # Set window properties
        parent.title("Radiochromic Film Analyzer")
        # Don't set geometry here, it's set in RCAnalyzer class
        parent.minsize(800, 660)  # Increased minimum height by 10%
        
        # Initialize components
        self.image_processor = ImageProcessor(self.app_config)
        self.file_manager = FileManager(self.app_config)
        
        # Loading state
        self.loading_image = False
        
        # Create UI
        self._create_menu()
        self._create_main_layout()
        
        logger.info("Main window initialized")
    
    def _create_menu(self):
        """Create the application menu."""
        self.menu_bar = tk.Menu(self.parent)
        self.parent.config(menu=self.menu_bar)
        
        # File menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open...", command=self.open_image)
        
        # Recent files submenu
        self.recent_menu = tk.Menu(self.file_menu, tearoff=0)
        self.file_menu.add_cascade(label="Open Recent", menu=self.recent_menu)
        self._update_recent_menu()
        
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Settings...", command=self.open_settings)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.parent.on_close)
        
        # View menu
        self.view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="View", menu=self.view_menu)
        self.view_menu.add_command(label="Fit to Screen", command=self.fit_to_screen)
        self.view_menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        self.view_menu.add_separator()
        self.view_menu.add_checkbutton(label="Negative Mode", 
                                      variable=tk.BooleanVar(value=self.image_processor.negative_mode),
                                      command=self.update_image)
        
        # Calibration menu
        self.calibration_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Calibration", menu=self.calibration_menu)
        self.calibration_menu.add_command(label="Start Calibration Wizard", 
                                         command=self.start_calibration_wizard)
        
        # Toggle for dose calibration (like Negative mode)
        self.calibration_var = tk.BooleanVar(value=False)
        self.calibration_menu.add_checkbutton(
            label="Apply Calibration",
            variable=self.calibration_var,
            command=self.apply_calibration
        )
        
        # Plugins menu
        from app.plugins.plugin_manager import plugin_manager
        self.plugins_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Plugins", menu=self.plugins_menu)
        self.plugins_menu.add_command(label="Load Plugin...", command=self._load_plugin)
        self.plugins_menu.add_separator()

        # Dynamic list will be populated later
        self._refresh_plugins_menu()

        # Help menu
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="About", command=self.show_about)
    
    def _create_main_layout(self):
        """Create the main application layout."""
        # Main panel (horizontal split)
        self.main_panel = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        self.main_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (image) with dependency injection
        self.image_panel = ImagePanel(self.main_panel, self.image_processor)
        self.main_panel.add(self.image_panel.frame, weight=3)
        
        # Set callbacks for the image panel
        self.image_panel.set_zoom_callback(self.update_zoom)
        
        # Right panel (controls)

        self.controls_frame = ttk.Frame(self.main_panel)
        self.main_panel.add(self.controls_frame, weight=1)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.controls_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Measurement tab with dependency injection
        self.measurement_panel = MeasurementPanel(self.notebook, self.image_processor)
        self.notebook.add(self.measurement_panel.frame, text="Measurement")
        
        # Set measurement callback
        self.image_panel.set_measurement_callback(self.measurement_panel.update_results)
        
        # Image tab with dependency injection and update callback
        self.image_settings_panel = ImageSettingsPanel(
            self.notebook, 
            self.image_processor, 
            self.update_image_settings
        )
        self.notebook.add(self.image_settings_panel.frame, text="Image")

        # Expose notebook to plugin manager now that it exists
        from app.plugins.plugin_manager import plugin_manager
        plugin_manager.init_ui(self, self.notebook, self.image_processor)
        
        # Status bar
        self.status_bar = ttk.Frame(self.parent)
        self.status_bar.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.coordinates_label = ttk.Label(self.status_bar, text="Coordinates: --")
        self.coordinates_label.pack(side=tk.LEFT, padx=20)
        
        self.rgb_label = ttk.Label(self.status_bar, text="RGB: --")
        self.rgb_label.pack(side=tk.LEFT, padx=20)
        
        self.zoom_label = ttk.Label(self.status_bar, text="Zoom: 100%")
        self.zoom_label.pack(side=tk.RIGHT, padx=5)
    
    def open_image(self):
        """Open an image file."""
        file_path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("TIFF Images", "*.tif *.tiff"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.update_status("Loading image...")
            # Load image in a separate thread
            threading.Thread(target=self._load_image_thread, args=(file_path,), daemon=True).start()
    
    def _load_image_thread(self, file_path):
        """Load image in a background thread with improved error handling."""
        try:
            # Set loading flag
            self.loading_image = True

            # Log the file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"Loading image: {file_path} (Size: {file_size_mb:.2f} MB)")

            # Check image format
            file_ext = os.path.splitext(file_path)[1].lower()
            supported_formats = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
            
            if file_ext not in supported_formats:
                logger.warning(f"Potentially unsupported file format: {file_ext}")
            
            # Load the image with timeout protection
            def load_with_timeout():
                return self.image_processor.load_image(file_path)
            
            # For very large files, increase the expected loading time
            estimated_load_time = 5 + (file_size_mb / 20)  # Base + size-dependent factor
            logger.debug(f"Estimated load time: {estimated_load_time:.1f} seconds")
            
            # Load the image
            load_success = self.image_processor.load_image(file_path)
            
            if not load_success:
                raise Exception("Image processor reported loading failure")
            
            # Schedule UI updates on the main thread
            self.parent.after(0, lambda: self._finish_loading_image(file_path))
        except (IOError, OSError) as e:
            logger.error(f"File access error loading image: {str(e)}", exc_info=True)
            self.parent.after(0, lambda: messagebox.showerror("File Error", 
                                                            f"Could not access the image file:\n{str(e)}"))
            self.parent.after(0, lambda: self.update_status("Error: Could not access image file"))
            self.loading_image = False
        except MemoryError as e:
            logger.error(f"Out of memory loading image: {str(e)}", exc_info=True)
            self.parent.after(0, lambda: messagebox.showerror("Memory Error", 
                                                            "Not enough memory to load this image.\n"
                                                            "Try closing other applications or using a smaller image."))
            self.parent.after(0, lambda: self.update_status("Error: Out of memory"))
            self.loading_image = False
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}", exc_info=True)
            self.parent.after(0, lambda: messagebox.showerror("Error", 
                                                            f"Could not load image:\n{str(e)}"))
            self.parent.after(0, lambda: self.update_status("Error loading image"))
            self.loading_image = False
    
    def _finish_loading_image(self, file_path):
        """Finish loading image on the main thread."""
        try:
            # Fit image to screen by default
            self.image_panel.fit_to_screen()
            
            logger.debug("Calling fit_to_screen after loading image")
            
            # Update UI
            self.update_status(f"Loaded image: {os.path.basename(file_path)}")
            
            # Add to recent files
            self.file_manager.add_recent_file(file_path)
            self._update_recent_menu()
            
            # If calibration parameters exist, pre-enable the toggle so user sees dose
            if self.image_processor.has_calibration():
                self.calibration_var.set(True)
                self.apply_calibration()
            
            # Update window title
            self.parent.title(f"Radiochromic Film Analyzer - {os.path.basename(file_path)}")
            
            logger.info(f"Loaded image: {file_path}")
        except Exception as e:
            logger.error(f"Error updating UI after loading image: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Error displaying image: {str(e)}")
        finally:
            # Clear loading flag
            self.loading_image = False
    
    def load_image(self, file_path):
        """Load an image from the specified path."""
        self.update_status("Loading image...")
        threading.Thread(target=self._load_image_thread, args=(file_path,), daemon=True).start()
    
    def _update_recent_menu(self):
        """Update the recent files menu."""
        # Clear current menu
        self.recent_menu.delete(0, tk.END)
        
        # Get recent files
        recent_files = self.file_manager.get_recent_files()
        
        if not recent_files:
            self.recent_menu.add_command(label="(No recent files)", state=tk.DISABLED)
        else:
            for file_path in recent_files:
                name = os.path.basename(file_path)
                self.recent_menu.add_command(
                    label=name,
                    command=lambda f=file_path: self.load_image(f)
                )
    
    def open_settings(self):
        """Open the settings dialog with scrollbar support."""
        # Create settings window
        settings_window = tk.Toplevel(self.parent)
        settings_window.title("Settings")
        settings_window.geometry("500x600")  # 20% taller
        settings_window.grab_set()  # Modal
        
        # Create a canvas with scrollbar for content
        container = ttk.Frame(settings_window)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas for scrolling
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrollable frame
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create window in canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Main frame for content
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
            value=self.app_config.get("remove_background", False)
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
        
        # Measurement section
        ttk.Label(
            main_frame, 
            text="Measurement", 
            font=("Arial", 11, "bold")
        ).pack(anchor=tk.W, pady=(20, 5))
        
        # Auto measure option
        auto_measure_var = tk.BooleanVar(
            value=self.app_config.get("auto_measure", False)
        )
        
        ttk.Checkbutton(
            main_frame, 
            text="Auto measure (update results on mouse move)", 
            variable=auto_measure_var
        ).pack(anchor=tk.W, padx=10, pady=5)
        
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
            value=self.app_config.get("colormap", "viridis")
        )
        
        colormap_combobox = ttk.Combobox(
            main_frame,
            textvariable=colormap_var,
            values=["viridis", "plasma", "inferno", "magma", "cividis", "jet", "rainbow", "turbo", "hot", "cool"]
        )
        colormap_combobox.pack(anchor=tk.W, padx=10, pady=5, fill=tk.X)
        
        # Optimization section
        ttk.Label(
            main_frame, 
            text="Performance Optimization", 
            font=("Arial", 11, "bold")
        ).pack(anchor=tk.W, pady=(20, 5))
        
        # GPU acceleration option
        use_gpu_var = tk.BooleanVar(
            value=self.app_config.get("use_gpu", False)
        )
        
        gpu_checkbox = ttk.Checkbutton(
            main_frame, 
            text="Use GPU acceleration (if available)", 
            variable=use_gpu_var
        )
        gpu_checkbox.pack(anchor=tk.W, padx=10, pady=5)
        
        # Force enable GPU option
        gpu_force_enabled_var = tk.BooleanVar(
            value=self.app_config.get("gpu_force_enabled", False)
        )
        
        gpu_force_enabled_checkbox = ttk.Checkbutton(
            main_frame,
            text="Force enable GPU (use when GPU is available but not detected)",
            variable=gpu_force_enabled_var
        )
        gpu_force_enabled_checkbox.pack(anchor=tk.W, padx=10, pady=5)
        
        # Check GPU availability
        try:
            import cv2
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # GPU is available
                gpu_status = f"GPU available: {cv2.cuda.getDeviceName(0)}"
                gpu_color = "green"
            else:
                # No GPU available
                gpu_status = "No CUDA-capable GPU detected"
                gpu_color = "red"
                gpu_checkbox.config(state=tk.DISABLED)
                use_gpu_var.set(False)
                # Keep force enable option available
        except Exception:
            # Error checking GPU
            gpu_status = "Error checking GPU availability"
            gpu_color = "red"
            gpu_checkbox.config(state=tk.DISABLED)
            use_gpu_var.set(False)
            # Keep force enable option available
        
        ttk.Label(
            main_frame,
            text=gpu_status,
            foreground=gpu_color
        ).pack(anchor=tk.W, padx=10, pady=(0, 5))
        
        # Multi-threading option
        use_multithreading_var = tk.BooleanVar(
            value=self.app_config.get("use_multithreading", True)
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
        
        # Ensure num_threads doesn't exceed available cores
        initial_threads = min(self.app_config.get("num_threads", cpu_count), cpu_count)
        num_threads_var = tk.IntVar(value=initial_threads)
        
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
            text=f"(System has {cpu_count} CPU cores)",
            foreground="gray"
        ).pack(side=tk.LEFT, padx=10)
        
        # Advanced settings section
        ttk.Label(
            main_frame, 
            text="Advanced Settings", 
            font=("Arial", 11, "bold")
        ).pack(anchor=tk.W, pady=(20, 5))
        
        # Memory usage limit
        memory_frame = ttk.Frame(main_frame)
        memory_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(
            memory_frame,
            text="Maximum memory usage (%):"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Memory slider
        memory_var = tk.IntVar(
            value=self.app_config.get("max_memory_percent", 75)
        )
        
        memory_slider = ttk.Scale(
            memory_frame,
            from_=25,
            to=95,
            orient="horizontal",
            variable=memory_var,
            length=200
        )
        memory_slider.pack(side=tk.LEFT, padx=(0, 10))
        
        memory_label = ttk.Label(memory_frame, text=f"{memory_var.get()}%")
        memory_label.pack(side=tk.LEFT)
        
        # Update label when slider value changes
        def update_memory_label(*args):
            memory_label.config(text=f"{memory_var.get()}%")
        
        memory_var.trace_add("write", update_memory_label)
        
        # Cache settings
        cache_frame = ttk.Frame(main_frame)
        cache_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(
            cache_frame,
            text="Maximum cache size (MB):"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Cache size slider
        cache_var = tk.IntVar(
            value=self.app_config.get("max_cache_mb", 512)
        )
        
        cache_slider = ttk.Scale(
            cache_frame,
            from_=64,
            to=4096,
            orient="horizontal",
            variable=cache_var,
            length=200
        )
        cache_slider.pack(side=tk.LEFT, padx=(0, 10))
        
        cache_label = ttk.Label(cache_frame, text=f"{cache_var.get()} MB")
        cache_label.pack(side=tk.LEFT)
        
        # Update label when slider value changes
        def update_cache_label(*args):
            cache_label.config(text=f"{cache_var.get()} MB")
        
        cache_var.trace_add("write", update_cache_label)
        
        # Logging section
        ttk.Label(
            main_frame, 
            text="Logging", 
            font=("Arial", 11, "bold")
        ).pack(anchor=tk.W, pady=(20, 5))
        
        # Log level
        log_level_frame = ttk.Frame(main_frame)
        log_level_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(
            log_level_frame,
            text="Log level:"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Log level combobox
        log_level_var = tk.StringVar(
            value=self.app_config.get("log_level", "INFO")
        )
        
        log_level_combobox = ttk.Combobox(
            log_level_frame,
            textvariable=log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            width=10,
            state="readonly"
        )
        log_level_combobox.pack(side=tk.LEFT)
        
        # Detailed logging option
        detailed_logging_var = tk.BooleanVar(
            value=self.app_config.get("detailed_logging", False)
        )
        
        ttk.Checkbutton(
            main_frame, 
            text="Enable detailed logging (higher disk usage)", 
            variable=detailed_logging_var
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        def save_settings():
            # Update config
            self.app_config["remove_background"] = remove_background_var.get()
            self.app_config["auto_measure"] = auto_measure_var.get()
            self.app_config["colormap"] = colormap_var.get()
            self.app_config["use_gpu"] = use_gpu_var.get()
            self.app_config["gpu_force_enabled"] = gpu_force_enabled_var.get()
            self.app_config["use_multithreading"] = use_multithreading_var.get()
            self.app_config["num_threads"] = num_threads_var.get()
            self.app_config["max_memory_percent"] = memory_var.get()
            self.app_config["max_cache_mb"] = cache_var.get()
            self.app_config["log_level"] = log_level_var.get()
            self.app_config["detailed_logging"] = detailed_logging_var.get()
            
            # Apply settings
            self.apply_settings()
            
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
        
        # Make canvas scrollable with mouse wheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)  # For Windows and MacOS
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # For Linux
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # For Linux
        
        # Make sure scrollregion is updated properly
        canvas.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
    
    def apply_settings(self):
        """Apply settings changes."""
        # Update image processor settings
        self.image_processor.update_settings(self.app_config)
        
        # Reload current image if available
        if self.image_processor.has_image():
            self.update_status("Applying settings...")
            threading.Thread(target=self._apply_settings_thread, daemon=True).start()
    
    def _apply_settings_thread(self):
        """Apply settings in a background thread."""
        try:
            self.image_processor.reprocess_current_image()
            self.parent.after(0, lambda: self._finish_applying_settings())
        except Exception as e:
            logger.error(f"Error applying settings: {str(e)}", exc_info=True)
            self.parent.after(0, lambda: messagebox.showerror("Error", f"Error applying settings: {str(e)}"))
            self.parent.after(0, lambda: self.update_status("Error applying settings"))
    
    def _finish_applying_settings(self):
        """Finish applying settings on the main thread."""
        self.image_panel.fit_to_screen()
        self.update_status("Applied settings to current image")
    
    def update_image_settings(self):
        """Update image settings (contrast, brightness, negative mode)."""
        if self.image_processor.has_image():
            # Update the image without showing loading indicator
            self.image_panel.display_image(is_adjustment=True)
    
    def fit_to_screen(self):
        """Fit the image to the screen."""
        if self.image_processor.has_image() and not self.loading_image:
            self.image_panel.fit_to_screen()
            self.update_status("Fit image to screen")
    
    def reset_zoom(self):
        """Reset zoom to 100%."""
        if self.image_processor.has_image() and not self.loading_image:
            self.image_panel.reset_zoom()
            self.update_status("Reset zoom to 100%")
    
    def update_image(self):
        """Update the displayed image."""
        if self.image_processor.has_image() and not self.loading_image:
            self.image_panel.display_image(is_adjustment=True)
    
    def start_calibration_wizard(self):
        """Start the calibration wizard."""
        # Defer heavy imports so startup is not affected
        try:
            from app.ui.calibration_wizard import CalibrationWizardWindow
        except Exception as exc:
            messagebox.showerror("Calibration Wizard", f"No se pudo abrir el asistente de calibración:\n{exc}")
            logger.error("Failed to launch calibration wizard", exc_info=True)
            return

        # Create the wizard window (modal)
        CalibrationWizardWindow(self.parent)
        # No further action for now – the wizard handles its own lifecycle
    
    def apply_calibration(self):
        """Toggle calibration on/off depending on menu state."""
        if not self.image_processor.has_image():
            self.calibration_var.set(False)
            return

        # If the checkbox is now ON, apply calibration
        if self.calibration_var.get():
            if self.image_processor.has_calibration():
                success = self.image_processor.apply_calibration()
                if success:
                    self.image_panel.display_image(is_adjustment=True)
                    self.update_status("Applied calibration to current image")
                else:
                    # Revert checkbox on failure
                    self.calibration_var.set(False)
            else:
                messagebox.showwarning("Calibration", "No calibration parameters available.")
                self.calibration_var.set(False)
        else:
            # Checkbox turned OFF – revert to original RGB
            if self.image_processor.calibration_applied:
                self.image_processor.reprocess_current_image()
                self.image_panel.display_image(is_adjustment=True)
                self.update_status("Calibration disabled (RGB view)")
    
    def show_about(self):
        """Show the about dialog."""
        messagebox.showinfo(
            "About Radiochromic Film Analyzer",
            "Radiochromic Film Analyzer\n\n"
            "A tool for analyzing radiochromic films and calculating dose.\n\n"
            "Version 1.0.0"
        )
    
    def update_status(self, message):
        """Update the status bar message."""
        self.status_label.config(text=message)
    
    def update_coordinates(self, x, y):
        """Update the coordinates display."""
        self.coordinates_label.config(text=f"Coordinates: ({x}, {y})")
    
    def update_rgb(self, rgb, std_dev=None):
        """Update the RGB / value display under the cursor.

        When a dose calibration is active, the displayed image represents a
        dose map instead of raw pixel values.  In that situation we show the
        dose with two decimal places (matching the precision used elsewhere)
        instead of the default one-decimal RGB formatting.
        """
        is_dose = getattr(self.image_processor, "calibration_applied", False)

        # Handle placeholder or string values (e.g., "--") early to avoid format errors
        if isinstance(rgb, str) or rgb is None:
            label = "Dose" if is_dose else "Value"
            self.rgb_label.config(text=f"{label}: {rgb}")
            return

        if is_dose:
            # ----------------------------------------------------------
            # Dose display (calibration applied) – use two decimals
            # ----------------------------------------------------------
            if isinstance(rgb, tuple) and len(rgb) == 3:
                # Colour dose image
                if isinstance(rgb[0], tuple) and isinstance(rgb[1], tuple):
                    # Pattern: ((valR,valG,valB), (devR,devG,devB))
                    val, dev = rgb
                    self.rgb_label.config(
                        text=(
                            f"Dose: ("  # Per-channel dose
                            f"{val[0]:.2f}±{dev[0]:.2f}, "
                            f"{val[1]:.2f}±{dev[1]:.2f}, "
                            f"{val[2]:.2f}±{dev[2]:.2f})"
                        )
                    )
                elif std_dev is not None and isinstance(std_dev, tuple) and len(std_dev) == 3:
                    self.rgb_label.config(
                        text=(
                            f"Dose: ("
                            f"{rgb[0]:.2f}±{std_dev[0]:.2f}, "
                            f"{rgb[1]:.2f}±{std_dev[1]:.2f}, "
                            f"{rgb[2]:.2f}±{std_dev[2]:.2f})"
                        )
                    )
                else:
                    self.rgb_label.config(
                        text=f"Dose: ({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})"
                    )
            elif isinstance(rgb, tuple) and len(rgb) == 2 and not isinstance(rgb[0], tuple):
                # Grayscale dose with (value, std_dev)
                val, dev = rgb
                self.rgb_label.config(text=f"Dose: {val:.2f}±{dev:.2f}")
            elif std_dev is not None:
                self.rgb_label.config(text=f"Dose: {rgb:.2f}±{std_dev:.2f}")
            else:
                self.rgb_label.config(text=f"Dose: {rgb:.2f}")
        else:
            # ----------------------------------------------------------
            # Raw pixel display (no calibration) – original behaviour
            # ----------------------------------------------------------
            if isinstance(rgb, tuple) and len(rgb) == 3:
                if std_dev is not None and isinstance(std_dev, tuple) and len(std_dev) == 3:
                    self.rgb_label.config(
                        text=(
                            f"RGB: ("
                            f"{rgb[0]:.1f}±{std_dev[0]:.1f}, "
                            f"{rgb[1]:.1f}±{std_dev[1]:.1f}, "
                            f"{rgb[2]:.1f}±{std_dev[2]:.1f})"
                        )
                    )
                else:
                    self.rgb_label.config(text=f"RGB: ({rgb[0]}, {rgb[1]}, {rgb[2]})")
            elif isinstance(rgb, tuple) and len(rgb) == 2 and isinstance(rgb[0], tuple) and isinstance(rgb[1], tuple):
                val, dev = rgb
                self.rgb_label.config(
                    text=(
                        f"RGB: ("
                        f"{val[0]:.1f}±{dev[0]:.1f}, "
                        f"{val[1]:.1f}±{dev[1]:.1f}, "
                        f"{val[2]:.1f}±{dev[2]:.1f})"
                    )
                )
            elif std_dev is not None:
                self.rgb_label.config(text=f"Value: {rgb:.1f}±{std_dev:.1f}")
            elif isinstance(rgb, tuple) and len(rgb) == 2 and not isinstance(rgb[0], tuple):
                val, dev = rgb
                self.rgb_label.config(text=f"Value: {val:.1f}±{dev:.1f}")
            else:
                self.rgb_label.config(text=f"Value: {rgb}")
    
    def update_zoom(self, zoom):
        """Update the zoom level display."""
        self.zoom_label.config(text=f"Zoom: {int(zoom * 100)}%")
    
    # ------------------------------------------------------------------
    # Plugins helpers
    # ------------------------------------------------------------------
    def _refresh_plugins_menu(self):
        from app.plugins.plugin_manager import plugin_manager
        # Clear everything after the first two items (Load + separator)
        self.plugins_menu.delete(2, tk.END)
        for name in plugin_manager.plugin_names():
            var = tk.BooleanVar(value=plugin_manager.is_active(name))
            def _toggle(n=name, v=var):
                plugin_manager.set_active(n, v.get())
            self.plugins_menu.add_checkbutton(label=name, variable=var, command=_toggle)

    def _load_plugin(self):
        from app.plugins.plugin_manager import plugin_manager
        file_path = filedialog.askopenfilename(
            title="Select Python plugin",
            filetypes=[("Python files", "*.py")])
        if not file_path:
            return
        name = plugin_manager.load_plugin_file(file_path)
        if name:
            messagebox.showinfo("Plugin loaded", f"Plugin '{name}' loaded and enabled.")
            self._refresh_plugins_menu()
        else:
            messagebox.showerror("Plugin error", "Failed to load plugin. Check logs for details.")

    def get_config(self):
        """Get the current configuration."""
        # Update config with any UI changes
        return self.app_config

    def cleanup(self):
        """Clean up resources."""
        # Clean up image panel resources
        if hasattr(self, 'image_panel'):
            self.image_panel.cleanup()
