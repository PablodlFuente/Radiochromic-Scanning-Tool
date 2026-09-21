"""
Main window UI for the Radiochromic Film Analyzer.

This module contains the main window UI class that creates and manages
the main application window and its components.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import os
import sys
import threading
import webbrowser
from app.ui.measurement_panel import MeasurementPanel
from app.ui.image_settings_panel import ImageSettingsPanel
from app.ui.image_panel import ImagePanel
from app.core.image_processor import ImageProcessor
from app.utils.file_manager import FileManager
from app.utils.config_manager import ConfigManager
from app.paths import CALIBRATION_ROOT, PROJECT_ROOT
from app.version import __version__

logger = logging.getLogger(__name__)
AUTHOR_NAME = "Pablo de la Fuente Fernández"
AUTHOR_GITHUB_URL = "https://github.com/PablodlFuente"

class MainWindow:
    """Main window UI for the Radiochromic Film Analyzer."""
    
    def __init__(self, parent, app_config):
        """Initialize the main window."""
        self.parent = parent
        self.app_config = app_config
        self._dragged_tab_index = None
        
        # Set window properties
        parent.title("Radiochromic Film Analyzer")
        # Don't set geometry here, it's set in RCAnalyzer class
        parent.minsize(800, 660)  # Increased minimum height by 10%
        
        # Initialize components
        self.image_processor = ImageProcessor(self.app_config)
        self.file_manager = FileManager(self.app_config)
        
        # Loading state
        self.loading_image = False
        self._image_load_lock = threading.Lock()
        self._image_load_request_id = 0
        self._image_load_callbacks = {}
        
        # Create UI
        self._create_menu()
        self._create_main_layout()
        
        # Initialize calibration menu states (disable if no data available)
        self._update_calibration_menu_states()
        
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
        self.calibration_menu.add_command(label="Choose Calibration...", 
                                         command=self.open_settings)
        self.calibration_menu.add_separator()
        self.calibration_menu.add_command(label="Calibrate Scanner", 
                                         command=self.start_calibration_wizard)
        self.calibration_menu.add_command(label="Update Scanner Flatness", 
                                         command=self.start_flatness_wizard)
        self.calibration_menu.add_command(label="Modify Calibration",
                                         command=self.modify_calibration)
        self.calibration_menu.add_separator()
        
        # Toggle for field flattening (independent)
        # Store the menu index for enabling/disabling later
        self.flat_var = tk.BooleanVar(value=False)
        self.calibration_menu.add_checkbutton(
            label="Apply Flat",
            variable=self.flat_var,
            command=self.apply_flat
        )
        self._flat_menu_index = self.calibration_menu.index(tk.END)
        
        # Toggle for dose conversion (independent, can combine with flat)
        self.calibration_var = tk.BooleanVar(value=False)
        self.calibration_menu.add_checkbutton(
            label="Apply Dose Conversion",
            variable=self.calibration_var,
            command=self.apply_calibration
        )
        self._cal_menu_index = self.calibration_menu.index(tk.END)
        
        # Utilities menu
        self.utilities_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Utilities", menu=self.utilities_menu)
        self.utilities_menu.add_command(label="Measure Flatness of Image...", 
                                        command=self.measure_image_flatness)
        
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
        self.help_menu.add_command(label="Documentation", command=self.open_local_documentation)
        self.help_menu.add_command(label="Check for Updates...", command=self.check_for_updates)
        self.help_menu.add_separator()
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
        self._enable_notebook_tab_reordering()
        
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
        self._restore_notebook_tab_order()
        
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

        self.about_footer = ttk.Label(
            self.status_bar,
            text=f"Version {__version__} · {AUTHOR_NAME}",
            cursor="hand2",
        )
        self.about_footer.pack(side=tk.RIGHT, padx=(14, 5))
        self.about_footer.bind("<Button-1>", lambda _event: self.show_about())

    def _enable_notebook_tab_reordering(self):
        """Allow notebook tabs to be reordered by dragging them."""
        self.notebook.bind("<ButtonPress-1>", self._on_notebook_tab_press, add="+")
        self.notebook.bind("<B1-Motion>", self._on_notebook_tab_drag, add="+")
        self.notebook.bind("<ButtonRelease-1>", self._on_notebook_tab_release, add="+")

    def _get_notebook_tab_order(self):
        """Return the current notebook tab texts in display order."""
        return [self.notebook.tab(tab_id, "text") for tab_id in self.notebook.tabs()]

    def _store_notebook_tab_order(self):
        """Persist current notebook tab order in the application config."""
        self.app_config["notebook_tab_order"] = self._get_notebook_tab_order()

    def _restore_notebook_tab_order(self):
        """Restore notebook tab order from the application config when available."""
        saved_order = self.app_config.get("notebook_tab_order", [])
        if not saved_order:
            return

        tab_by_text = {}
        for tab_id in self.notebook.tabs():
            tab_text = self.notebook.tab(tab_id, "text")
            tab_by_text.setdefault(tab_text, []).append(tab_id)

        insert_index = 0
        for tab_text in saved_order:
            tab_list = tab_by_text.get(tab_text)
            if not tab_list:
                continue
            self.notebook.insert(insert_index, tab_list.pop(0))
            insert_index += 1

        self._store_notebook_tab_order()

    def _get_notebook_tab_index_at(self, x, y):
        """Return the notebook tab index at screen coordinates, or None."""
        try:
            return self.notebook.index(f"@{x},{y}")
        except tk.TclError:
            return None

    def _on_notebook_tab_press(self, event):
        """Remember which tab started a drag operation."""
        self._dragged_tab_index = self._get_notebook_tab_index_at(event.x, event.y)

    def _on_notebook_tab_drag(self, event):
        """Reorder tabs live while dragging across the notebook header."""
        if self._dragged_tab_index is None:
            return

        target_index = self._get_notebook_tab_index_at(event.x, event.y)
        if target_index is None or target_index == self._dragged_tab_index:
            return

        tabs = self.notebook.tabs()
        if self._dragged_tab_index >= len(tabs) or target_index >= len(tabs):
            return

        dragged_tab = tabs[self._dragged_tab_index]
        self.notebook.insert(target_index, dragged_tab)
        self._dragged_tab_index = target_index

    def _on_notebook_tab_release(self, event):
        """Finish tab reordering when the drag ends."""
        self._store_notebook_tab_order()
        self._dragged_tab_index = None

    def get_config(self):
        """Get the current configuration."""
        self._store_notebook_tab_order()
        return self.app_config
    
    def _update_calibration_menu_states(self):
        """Update the enabled/disabled state of calibration menu items based on data availability."""
        # Force a fresh check/load of field flattening to avoid stale state/race conditions
        try:
            has_flat = self.image_processor.load_field_flattening()
        except Exception:
            logger.exception("Error while loading field flattening during menu state update")
            has_flat = False

        # Calibration parameters may be loaded on demand - check availability
        try:
            has_cal = self.image_processor.has_calibration()
        except Exception:
            logger.exception("Error while checking calibration availability during menu state update")
            has_cal = False
        
        # Track if we need to refresh the display
        needs_refresh = False
        
        # Update Apply Flat menu item
        if has_flat:
            self.calibration_menu.entryconfigure(self._flat_menu_index, state=tk.NORMAL)
            # If the checkbox is checked but the processor hasn't applied the flat,
            # ensure the correction is actually applied so the UI and internal state
            # remain in sync. This handles cases where the checkbox was programmatically
            # set (e.g. on load) but the flattening wasn't yet executed.
            if self.flat_var.get():
                try:
                    if self.image_processor.has_image():
                        # Reapply corrections to ensure flat is applied immediately
                        self._reapply_corrections()
                    else:
                        # No image yet; make sure flat data is loaded for future operations
                        self.image_processor.load_field_flattening()
                except Exception:
                    # Don't let menu state update fail due to processor errors
                    logger.exception("Error while attempting to apply flat during menu state update")
        else:
            self.calibration_menu.entryconfigure(self._flat_menu_index, state=tk.DISABLED)
            # If flat was enabled but is no longer available, clean up
            if self.flat_var.get():
                self.flat_var.set(False)
                # Clear flattened image state
                self.image_processor.flat_applied = False
                self.image_processor.flattened_image = None
                needs_refresh = True
        
        # Update Apply Dose Conversion menu item
        if has_cal:
            self.calibration_menu.entryconfigure(self._cal_menu_index, state=tk.NORMAL)
        else:
            self.calibration_menu.entryconfigure(self._cal_menu_index, state=tk.DISABLED)
            # If calibration was enabled but is no longer available, clean up
            if self.calibration_var.get():
                self.calibration_var.set(False)
                self.image_processor.calibration_applied = False
                needs_refresh = True
        
        # Refresh display if corrections were removed
        if needs_refresh and self.image_processor.has_image():
            # Reprocess image without the unavailable corrections
            self.image_processor.reprocess_current_image()
            self.image_panel.display_image()
            self.update_status("Calibration data no longer available - corrections removed")
        
        logger.debug(f"Calibration menu states updated: flat={has_flat}, cal={has_cal}")
    
    def open_image(self):
        """Open an image file."""
        file_path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("TIFF Images", "*.tif *.tiff"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.load_image(file_path)
    
    def _load_image_thread(self, file_path, request_id):
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
            
            # Serialize mutations of ImageProcessor. A newer request supersedes
            # the UI result of an older one, but both cannot write concurrently.
            with self._image_load_lock:
                if request_id != self._image_load_request_id:
                    return
                load_success = self.image_processor.load_image(file_path)
            
            if not load_success:
                raise Exception("Image processor reported loading failure")
            
            # Schedule UI updates on the main thread
            self.parent.after(0, lambda: self._finish_loading_image(file_path, request_id))
        except (IOError, OSError) as e:
            logger.error(f"File access error loading image: {str(e)}", exc_info=True)
            message = str(e)
            self.parent.after(0, lambda: self._finish_image_load_error(
                request_id, "File Error", f"Could not access the image file:\n{message}",
                "Error: Could not access image file",
            ))
        except MemoryError as e:
            logger.error(f"Out of memory loading image: {str(e)}", exc_info=True)
            self.parent.after(0, lambda: self._finish_image_load_error(
                request_id, "Memory Error",
                "Not enough memory to load this image.\nTry closing other applications or using a smaller image.",
                "Error: Out of memory",
            ))
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}", exc_info=True)
            message = str(e)
            self.parent.after(0, lambda: self._finish_image_load_error(
                request_id, "Error", f"Could not load image:\n{message}", "Error loading image"
            ))

    def _finish_image_load_error(self, request_id, title, message, status):
        """Report an image-load failure only if it belongs to the active request."""
        if request_id != self._image_load_request_id:
            return
        self._image_load_callbacks.pop(request_id, None)
        self.loading_image = False
        messagebox.showerror(title, message)
        self.update_status(status)
    
    def _finish_loading_image(self, file_path, request_id):
        """Finish loading image on the main thread."""
        if request_id != self._image_load_request_id:
            return
        try:
            from app.plugins.plugin_manager import plugin_manager
            plugin_manager.notify_image_loaded(file_path)
            # Fit image to screen by default
            self.image_panel.fit_to_screen()
            
            logger.debug("Calling fit_to_screen after loading image")
            
            # Update _OVERLAY shape in plugin (if plugin is loaded)
            if hasattr(self.image_processor, 'original_image') and self.image_processor.original_image is not None:
                try:
                    # Import and call plugin's update function
                    from custom_plugins import auto_measurements
                    if hasattr(auto_measurements, 'update_overlay_shape'):
                        auto_measurements.update_overlay_shape(self.image_processor.original_image.shape)
                        logger.debug(f"Updated plugin overlay shape to {self.image_processor.original_image.shape[:2]}")
                except ImportError:
                    pass  # Plugin not loaded yet
                except Exception as e:
                    logger.warning(f"Could not update plugin overlay shape: {e}")
            
            # Update UI
            self.update_status(f"Loaded image: {os.path.basename(file_path)}")
            self.measurement_panel.update_size_limits()
            
            # Add to recent files
            self.file_manager.add_recent_file(file_path)
            self._update_recent_menu()
            
            # Auto-enable flat and calibration if data is available
            has_flat = self.image_processor.has_field_flattening()
            has_cal = self.image_processor.has_calibration()
            
            if has_flat or has_cal:
                # Set checkboxes
                if has_flat:
                    self.flat_var.set(True)
                if has_cal:
                    self.calibration_var.set(True)
                
                # Apply corrections (flat first, then dose conversion)
                corrections_ok = self._reapply_corrections()
                
                # Update status
                if not corrections_ok:
                    self.update_status("Loaded raw image; requested corrections failed")
                elif has_flat and has_cal:
                    self.update_status(f"Loaded: {os.path.basename(file_path)} (Flat + Dose applied)")
                elif has_flat:
                    self.update_status(f"Loaded: {os.path.basename(file_path)} (Flat applied)")
                elif has_cal:
                    self.update_status(f"Loaded: {os.path.basename(file_path)} (Dose applied)")
            
            # Update window title
            self.parent.title(f"Radiochromic Film Analyzer - {os.path.basename(file_path)}")
            
            # Update calibration menu states (enable/disable based on data availability)
            self._update_calibration_menu_states()

            callback = self._image_load_callbacks.pop(request_id, None)
            if callback:
                callback(file_path)
            
            logger.info(f"Loaded image: {file_path}")
        except Exception as e:
            logger.error(f"Error updating UI after loading image: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Error displaying image: {str(e)}")
        finally:
            # Clear loading flag
            self.loading_image = False
    
    def load_image(self, file_path, on_complete=None):
        """Load an image from the specified path."""
        self.update_status("Loading image...")
        self.loading_image = True
        self._image_load_request_id += 1
        request_id = self._image_load_request_id
        self._image_load_callbacks.clear()
        if on_complete:
            self._image_load_callbacks[request_id] = on_complete
        threading.Thread(
            target=self._load_image_thread,
            args=(file_path, request_id),
            daemon=True,
        ).start()
    
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
        
        # Calibration section
        ttk.Label(
            main_frame, 
            text="Calibration", 
            font=("Arial", 11, "bold")
        ).pack(anchor=tk.W, pady=(10, 5))
        
        # Get available calibration folders
        calibration_base_dir = os.fspath(CALIBRATION_ROOT)
        calibration_folders = ["default"]  # Default uses calibration_data root
        
        try:
            if os.path.exists(calibration_base_dir):
                # List all subdirectories in calibration_data
                for item in os.listdir(calibration_base_dir):
                    item_path = os.path.join(calibration_base_dir, item)
                    if os.path.isdir(item_path):
                        calibration_folders.append(item)
        except Exception as e:
            logger.error(f"Error scanning calibration folders: {e}")
        
        # Calibration folder selection
        calibration_frame = ttk.Frame(main_frame)
        calibration_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(
            calibration_frame,
            text="Calibration folder:"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        calibration_folder_var = tk.StringVar(
            value=self.app_config.get("calibration_folder", "default")
        )
        
        calibration_combobox = ttk.Combobox(
            calibration_frame,
            textvariable=calibration_folder_var,
            values=calibration_folders,
            state="readonly",
            width=20
        )
        calibration_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Description
        ttk.Label(
            main_frame,
            text="Select which calibration data to use.\n'default' uses files in calibration_data root folder.",
            foreground="gray",
            justify=tk.LEFT,
            font=("Arial", 9)
        ).pack(anchor=tk.W, padx=10, pady=(5, 10))

        conversion_frame = ttk.Frame(main_frame)
        conversion_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(conversion_frame, text="Dose conversion:").pack(side=tk.LEFT, padx=(0, 10))
        conversion_method_var = tk.StringVar(
            value=self.app_config.get("calibration_conversion_method", "auto")
        )
        ttk.Combobox(
            conversion_frame,
            textvariable=conversion_method_var,
            values=["auto", "spline", "fit"],
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT)
        ttk.Label(
            main_frame,
            text=(
                "Auto: shape-preserving cubic interpolation inside the calibration range and "
                "rational fit outside it. Spline: calibrated range only. Fit: rational model."
            ),
            foreground="gray",
            justify=tk.LEFT,
            wraplength=430,
            font=("Arial", 9),
        ).pack(anchor=tk.W, padx=10, pady=(0, 10))
        
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
        
        # Uncertainty Estimation section
        ttk.Label(
            main_frame, 
            text="Uncertainty Estimation", 
            font=("Arial", 11, "bold")
        ).pack(anchor=tk.W, pady=(20, 5))
        
        # Uncertainty estimation method selection
        uncertainty_frame = ttk.Frame(main_frame)
        uncertainty_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(
            uncertainty_frame, 
            text="Method for combining channel uncertainties:"
        ).pack(anchor=tk.W, pady=(0, 5))
        
        # Available uncertainty estimation methods
        uncertainty_method_var = tk.StringVar(
            value=self.app_config.get("uncertainty_estimation_method", "dersimonian_laird")
        )
        
        uncertainty_combobox = ttk.Combobox(
            uncertainty_frame,
            textvariable=uncertainty_method_var,
            values=["weighted_average", "birge_factor", "dersimonian_laird", "sensitivity_weighted"],
            state="readonly"
        )
        uncertainty_combobox.pack(fill=tk.X, pady=(0, 5))
        
        # Add explanatory text for each method
        ttk.Label(
            uncertainty_frame, 
            text="• Weighted Average: Standard weighted mean (fastest)\n"
                 "• Birge Factor: Inflates uncertainties when χ²ν > 1\n"
                 "• DerSimonian-Laird: Random effects model with between-channel variance\n"
                 "• Sensitivity Weighted: Physics-based weights from calibration curve (requires calibration)",
            foreground="gray",
            justify=tk.LEFT,
            font=("Arial", 9)
        ).pack(anchor=tk.W, pady=(0, 10))
        
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
        
        # Updates section
        ttk.Label(
            main_frame, 
            text="Updates", 
            font=("Arial", 11, "bold")
        ).pack(anchor=tk.W, pady=(20, 5))
        
        # Check for updates on startup
        check_updates_startup_var = tk.BooleanVar(
            value=self.app_config.get("check_updates_on_startup", True)
        )
        
        ttk.Checkbutton(
            main_frame, 
            text="Check for updates when application starts", 
            variable=check_updates_startup_var
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        ttk.Label(
            main_frame,
            text="When enabled, the application will check GitHub for new versions on startup.",
            foreground="gray",
            justify=tk.LEFT,
            font=("Arial", 9)
        ).pack(anchor=tk.W, padx=10, pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        def save_settings():
            # Update config
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
            self.app_config["uncertainty_estimation_method"] = uncertainty_method_var.get()
            self.app_config["calibration_folder"] = calibration_folder_var.get()
            self.app_config["calibration_conversion_method"] = conversion_method_var.get()
            self.app_config["check_updates_on_startup"] = check_updates_startup_var.get()
            
            # Save configuration to file
            try:
                config_manager = ConfigManager()
                config_manager.save_config(self.app_config)
                logger.info("Settings saved to configuration file")
            except Exception as e:
                logger.error(f"Error saving configuration: {str(e)}", exc_info=True)
                messagebox.showerror("Error", f"Error saving configuration: {str(e)}")
                return
            
            # Apply settings
            self.apply_settings()
            
            # Update calibration menu states (flat/dose availability may have changed)
            self._update_calibration_menu_states()
            
            # Clean up events and close window
            cleanup_events()
            settings_window.destroy()
            
            # Show success message
            messagebox.showinfo("Settings", "Settings saved and applied successfully.")
        
        # Cleanup function to unbind events when window is destroyed
        def cleanup_events():
            try:
                canvas.unbind_all("<MouseWheel>")
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")
            except tk.TclError:
                pass
        
        def cancel_settings():
            cleanup_events()
            settings_window.destroy()
        
        ttk.Button(
            button_frame, 
            text="Cancel", 
            command=cancel_settings
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Save & Apply", 
            command=save_settings
        ).pack(side=tk.RIGHT, padx=5)
        
        # Make canvas scrollable with mouse wheel
        def _on_mousewheel(event):
            try:
                # Check if canvas still exists and is valid
                if canvas.winfo_exists():
                    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except tk.TclError:
                # Canvas has been destroyed, ignore the event
                pass
        
        def _on_scroll_up(event):
            try:
                if canvas.winfo_exists():
                    canvas.yview_scroll(-1, "units")
            except tk.TclError:
                pass
        
        def _on_scroll_down(event):
            try:
                if canvas.winfo_exists():
                    canvas.yview_scroll(1, "units")
            except tk.TclError:
                pass
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)  # For Windows and MacOS
        canvas.bind_all("<Button-4>", _on_scroll_up)  # For Linux
        canvas.bind_all("<Button-5>", _on_scroll_down)   # For Linux
        
        # Bind cleanup to window destruction
        settings_window.protocol("WM_DELETE_WINDOW", lambda: (cleanup_events(), settings_window.destroy()))
        
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
            if self._reapply_corrections():
                self._finish_applying_settings()
    
    def _apply_settings_thread(self):
        """Apply settings in a background thread."""
        try:
            # Reprocess image in background thread
            self.image_processor.reprocess_current_image()

            # After reprocessing, ensure corrections (flat/cal) are applied
            # on the main thread so UI updates and messageboxes are safe.
            self.parent.after(0, lambda: self._reapply_corrections())
            self.parent.after(0, lambda: self._finish_applying_settings())
        except Exception as e:
            logger.error(f"Error applying settings: {str(e)}", exc_info=True)
            message = str(e)
            self.parent.after(0, lambda: messagebox.showerror("Error", f"Error applying settings: {message}"))
            self.parent.after(0, lambda: self.update_status("Error applying settings"))
    
    def _finish_applying_settings(self):
        """Finish applying settings on the main thread."""
        self.image_panel.fit_to_screen()
        self.update_status("Applied settings to current image")
        # Ensure menu states reflect newly loaded calibration/flat data
        try:
            self._update_calibration_menu_states()
        except Exception:
            logger.exception("Failed to update calibration menu states after applying settings")
    
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
        """Start the scanner calibration wizard."""
        # Defer heavy imports so startup is not affected
        try:
            from app.ui.calibration_wizard import ScannerCalibrationWizard
        except Exception as exc:
            messagebox.showerror("Calibrate Scanner", f"Could not open the calibration wizard:\n{exc}")
            logger.error("Failed to launch scanner calibration wizard", exc_info=True)
            return

        # Create the wizard window (modal), passing the current config for consistency
        ScannerCalibrationWizard(self.parent, app_config=self.app_config)
        # No further action for now – the wizard handles its own lifecycle
    
    def start_flatness_wizard(self):
        """Start the scanner flatness update wizard (field flattening only)."""
        try:
            from app.ui.calibration_wizard import ScannerCalibrationWizard
        except Exception as exc:
            messagebox.showerror("Update Scanner Flatness", f"Could not open the wizard:\n{exc}")
            logger.error("Failed to launch flatness wizard", exc_info=True)
            return

        # Create the wizard in flatness_only mode
        ScannerCalibrationWizard(self.parent, app_config=self.app_config, flatness_only=True)

    def modify_calibration(self):
        """Open the selected calibration directly in the fit-editing window."""
        folder_name = str(self.app_config.get("calibration_folder", "default"))
        calibration_dir = (CALIBRATION_ROOT / folder_name).resolve()
        try:
            calibration_dir.relative_to(CALIBRATION_ROOT.resolve())
        except ValueError:
            messagebox.showerror("Modify Calibration", "The selected calibration folder is invalid.")
            return
        if not (calibration_dir / "calibration_data.csv").is_file():
            messagebox.showerror(
                "Modify Calibration",
                "The selected calibration does not contain calibration_data.csv.",
            )
            return
        try:
            from app.calibration.calibration_app import CalibrationApp
            calibration_window = tk.Toplevel(self.parent)
            calibration_window.title("Modify Calibration")
            calibration_window.geometry("1200x800")
            calibration_app = CalibrationApp(
                calibration_window, data_dir=calibration_dir,
                load_existing=True, open_fit=True, fit_only=True,
            )
            calibration_window.calibration_app = calibration_app
        except Exception as exc:
            logger.error("Failed to open calibration editor", exc_info=True)
            messagebox.showerror("Modify Calibration", f"Could not open the calibration editor:\n{exc}")
    
    def _reapply_corrections(self):
        """Reapply flat and/or dose corrections based on current checkbox states.
        
        Optimized to compute integral images only once at the end of the pipeline,
        rather than after each individual operation.
        """
        has_flat = bool(self.flat_var.get())
        has_cal = bool(self.calibration_var.get())
        
        # Start from original image
        # Skip integral compute if we have more operations to apply
        success = self.image_processor.process_corrections(flat=has_flat, calibration=has_cal)
        self.flat_var.set(self.image_processor.flat_applied)
        self.calibration_var.set(self.image_processor.calibration_applied)
        if not success:
            self.update_status("Processing failed; raw image restored")
            messagebox.showerror("Processing failed", self.image_processor.last_processing_error)
        elif self.image_processor.last_processing_warning:
            self.update_status(self.image_processor.last_processing_warning)
            messagebox.showwarning("Processing warning", self.image_processor.last_processing_warning)
        
        # Update display
        self.image_panel.display_image(is_adjustment=True)
        
        # Notify plugins
        from app.plugins.plugin_manager import plugin_manager
        if not self.loading_image:
            plugin_manager.notify_config_change(self.app_config)
        return success
    
    def apply_calibration(self):
        """Toggle dose conversion on/off (independent of flat)."""
        if not self.image_processor.has_image():
            self.calibration_var.set(False)
            return

        if self.calibration_var.get():
            # Turning ON dose conversion
            if not self.image_processor.has_calibration():
                messagebox.showwarning("Calibration", "No calibration parameters available.")
                self.calibration_var.set(False)
                return
            
            # Check bit depth compatibility
            image_bits = self.image_processor.get_image_bit_depth()
            calibration_bits = self.image_processor.get_calibration_bit_depth()
            
            if image_bits != calibration_bits:
                response = messagebox.askyesno(
                    "Bit Depth Mismatch",
                    f"The current image is {image_bits}-bit, but the calibration curve "
                    f"was created with {calibration_bits}-bit values.\n\n"
                    f"To apply the calibration, the image must be rescaled to {calibration_bits}-bit.\n\n"
                    f"Do you want to rescale the image to {calibration_bits}-bit and proceed?",
                    icon='warning'
                )
                
                if not response:
                    self.calibration_var.set(False)
                    self.update_status("Calibration cancelled - bit depth mismatch")
                    return
                
                # Rescale the image to match calibration bit depth
                if calibration_bits == 8:
                    success_rescale = self.image_processor.rescale_to_8bit()
                else:
                    success_rescale = self.image_processor.rescale_to_16bit()
                
                if not success_rescale:
                    messagebox.showerror("Error", f"Failed to rescale image to {calibration_bits}-bit.")
                    self.calibration_var.set(False)
                    return
        
        # Reapply all corrections based on current states
        if not self._reapply_corrections():
            return
        
        # Update status
        flat_on = self.flat_var.get()
        cal_on = self.calibration_var.get()
        if cal_on and flat_on:
            self.update_status("Applied: Flat + Dose Conversion")
        elif cal_on:
            self.update_status("Applied: Dose Conversion")
        elif flat_on:
            self.update_status("Applied: Flat")
        else:
            self.update_status("Original RGB view")
    
    def apply_flat(self):
        """Toggle field flattening on/off (independent of dose conversion)."""
        if not self.image_processor.has_image():
            self.flat_var.set(False)
            return

        if self.flat_var.get():
            # Turning ON flat
            if not self.image_processor.has_field_flattening():
                messagebox.showwarning("Field Flattening", 
                    "No field flattening data available.\n\n"
                    "Use 'Calibrate Scanner' to create field flattening data first.")
                self.flat_var.set(False)
                return
        
        # Reapply all corrections based on current states
        if not self._reapply_corrections():
            return
        
        # Update status
        flat_on = self.flat_var.get()
        cal_on = self.calibration_var.get()
        if cal_on and flat_on:
            self.update_status("Applied: Flat + Dose Conversion")
        elif cal_on:
            self.update_status("Applied: Dose Conversion")
        elif flat_on:
            self.update_status("Applied: Flat")
        else:
            self.update_status("Original RGB view")
    
    def measure_image_flatness(self):
        """Analyze and display flatness/uniformity statistics of the current image."""
        if not self.image_processor.has_image():
            messagebox.showwarning("Measure Flatness", "No image loaded.\n\nPlease load an image first.")
            return
        
        # Determine which image to analyze based on flat_var state
        if self.flat_var.get() and self.image_processor.flattened_image is not None:
            # Use the flattened image (flat correction applied)
            image = self.image_processor.flattened_image
            image_source = "Flattened Image (Apply Flat ON)"
        else:
            # Use original image (no flat correction)
            image = self.image_processor.original_image
            image_source = "Original Image (Apply Flat OFF)"
        
        if image is None:
            messagebox.showerror("Error", "Could not access image data.")
            return
        
        # Create analysis window
        flatness_window = tk.Toplevel(self.parent)
        flatness_window.title("Image Flatness Analysis")
        flatness_window.geometry("1200x750")
        flatness_window.transient(self.parent)
        
        # Main frame with scrollbar
        canvas_outer = tk.Canvas(flatness_window)
        scrollbar = ttk.Scrollbar(flatness_window, orient="vertical", command=canvas_outer.yview)
        scrollable_frame = ttk.Frame(canvas_outer)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_outer.configure(scrollregion=canvas_outer.bbox("all"))
        )
        
        canvas_outer.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_outer.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas_outer.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas_outer.bind_all("<MouseWheel>", _on_mousewheel)
        flatness_window.protocol("WM_DELETE_WINDOW", lambda: (canvas_outer.unbind_all("<MouseWheel>"), flatness_window.destroy()))
        
        main_frame = scrollable_frame
        
        # Header frame with title and info button
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        ttk.Label(header_frame, text="Image Flatness / Uniformity Analysis", 
                  font=("Arial", 14, "bold")).pack(side=tk.LEFT)
        
        # CV Info button
        def show_cv_info():
            cv_explanation = (
                "What is CV (Coefficient of Variation)?\n\n"
                "The Coefficient of Variation (CV) is a standardized measure of dispersion:\n\n"
                "    CV (%) = (Standard Deviation / Mean) × 100\n\n"
                "Interpretation:\n"
                "• CV < 1%: Excellent uniformity - scanner is very well calibrated\n"
                "• CV 1-2%: Good uniformity - acceptable for most applications\n"
                "• CV 2-5%: Acceptable - may need recalibration for high-precision work\n"
                "• CV > 5%: Poor uniformity - scanner calibration recommended\n\n"
                "Lower CV values indicate better spatial uniformity across the scanned image.\n"
                "For radiochromic film dosimetry, CV < 2% is typically desired."
            )
            messagebox.showinfo("CV Information", cv_explanation)
        
        info_btn = ttk.Button(header_frame, text="ℹ CV Info", command=show_cv_info, width=10)
        info_btn.pack(side=tk.RIGHT, padx=10)
        
        # Source indication
        ttk.Label(main_frame, text=f"Analyzing: {image_source}", 
                  font=("Arial", 10, "italic")).pack(pady=(0, 10))
        
        # Calculate statistics per channel
        import numpy as np
        
        if len(image.shape) == 3:
            channels = ['Red', 'Green', 'Blue']
            channel_colors = ['#FF4444', '#44AA44', '#4444FF']
            channel_data = [image[:, :, i] for i in range(3)]
        else:
            channels = ['Grayscale']
            channel_colors = ['#888888']
            channel_data = [image]
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Channel Statistics")
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Headers
        headers = ["Channel", "Mean", "Std Dev", "CV (%)", "Min", "Max", "Range", "Assessment"]
        for col, header in enumerate(headers):
            ttk.Label(stats_frame, text=header, font=("Arial", 10, "bold")).grid(
                row=0, column=col, padx=8, pady=5, sticky="w")
        
        stats_data = []
        for i, (ch_name, ch_data, ch_color) in enumerate(zip(channels, channel_data, channel_colors)):
            # Convert to float for accurate statistics
            ch_float = ch_data.astype(np.float64)
            mean_val = np.mean(ch_float)
            std_val = np.std(ch_float)
            cv = (std_val / mean_val * 100) if mean_val > 0 else 0
            min_val = np.min(ch_float)
            max_val = np.max(ch_float)
            range_val = max_val - min_val
            
            # Individual channel assessment
            if cv < 1.0:
                ch_assessment = "Excellent"
            elif cv < 2.0:
                ch_assessment = "Good"
            elif cv < 5.0:
                ch_assessment = "Acceptable"
            else:
                ch_assessment = "Poor"
            
            stats_data.append({
                'channel': ch_name,
                'mean': mean_val,
                'std': std_val,
                'cv': cv,
                'min': min_val,
                'max': max_val,
                'range': range_val,
                'data': ch_float,
                'color': ch_color,
                'assessment': ch_assessment
            })
            
            row_data = [ch_name, f"{mean_val:.2f}", f"{std_val:.2f}", f"{cv:.2f}%",
                       f"{min_val:.0f}", f"{max_val:.0f}", f"{range_val:.0f}", ch_assessment]
            for col, val in enumerate(row_data):
                lbl = ttk.Label(stats_frame, text=val)
                lbl.grid(row=i+1, column=col, padx=8, pady=2, sticky="w")
        
        # Overall Uniformity assessment
        assessment_frame = ttk.LabelFrame(main_frame, text="Overall Uniformity Assessment")
        assessment_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Calculate overall uniformity score (lower CV = better uniformity)
        avg_cv = np.mean([s['cv'] for s in stats_data])
        max_cv = np.max([s['cv'] for s in stats_data])
        worst_channel = [s['channel'] for s in stats_data if s['cv'] == max_cv][0]
        
        if avg_cv < 1.0:
            uniformity_grade = "Excellent"
        elif avg_cv < 2.0:
            uniformity_grade = "Good"
        elif avg_cv < 5.0:
            uniformity_grade = "Acceptable"
        else:
            uniformity_grade = "Poor - Consider recalibrating scanner"
        
        summary_text = (
            f"Average CV across all channels: {avg_cv:.2f}%  |  "
            f"Worst channel: {worst_channel} (CV = {max_cv:.2f}%)  |  "
            f"Overall Grade: {uniformity_grade}"
        )
        ttk.Label(assessment_frame, text=summary_text, font=("Arial", 10)).pack(anchor="w", padx=10, pady=8)
        
        # Matplotlib plots - 3 columns layout
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            
            n_channels = len(stats_data)
            
            # Create figure: 2 rows x 3 columns (or fewer for grayscale)
            # Row 1: Deviation maps, Row 2: Histograms
            fig = Figure(figsize=(12, 7), dpi=100)
            
            for idx, s in enumerate(stats_data):
                ch_data = s['data']
                
                # Downsample for display if image is large
                h, w = ch_data.shape
                step = max(1, max(h, w) // 400)
                display_downsampled = ch_data[::step, ::step]
                
                # Calculate normalized deviation from mean for uniformity visualization
                mean_val = s['mean']
                deviation_percent = ((display_downsampled - mean_val) / mean_val) * 100
                
                # Row 1: Deviation maps (3 columns)
                ax1 = fig.add_subplot(2, n_channels, idx + 1)
                
                # Use symmetric color scale centered on 0
                vmax = max(abs(deviation_percent.min()), abs(deviation_percent.max()))
                vmax = min(vmax, 10)  # Cap at ±10% for better visualization
                
                im = ax1.imshow(deviation_percent, cmap='RdYlGn_r', aspect='auto',
                               vmin=-vmax, vmax=vmax)
                ax1.set_title(f"{s['channel']} - Deviation from Mean\nCV = {s['cv']:.2f}%", 
                             fontsize=10, fontweight='bold')
                ax1.set_xlabel("X (pixels)", fontsize=8)
                ax1.set_ylabel("Y (pixels)", fontsize=8)
                ax1.tick_params(labelsize=7)
                fig.colorbar(im, ax=ax1, label="Deviation (%)", shrink=0.8)
                
                # Row 2: Histograms (3 columns)
                ax2 = fig.add_subplot(2, n_channels, n_channels + idx + 1)
                flat_data = ch_data.flatten()
                if len(flat_data) > 100000:
                    flat_data = np.random.choice(flat_data, 100000, replace=False)
                
                ax2.hist(flat_data, bins=100, alpha=0.7, color=s['color'], edgecolor='black', linewidth=0.3)
                ax2.axvline(s['mean'], color='black', linestyle='--', linewidth=1.5, label=f"Mean: {s['mean']:.0f}")
                ax2.axvline(s['mean'] - s['std'], color='gray', linestyle=':', linewidth=1.2)
                ax2.axvline(s['mean'] + s['std'], color='gray', linestyle=':', linewidth=1.2, label=f"±1σ: {s['std']:.1f}")
                ax2.set_title(f"{s['channel']} - Pixel Distribution", fontsize=10, fontweight='bold')
                ax2.set_xlabel("Pixel Value", fontsize=8)
                ax2.set_ylabel("Frequency", fontsize=8)
                ax2.tick_params(labelsize=7)
                ax2.legend(fontsize=7, loc='upper right')
            
            fig.tight_layout(pad=1.5)
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, master=main_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
        except ImportError as e:
            ttk.Label(main_frame, text=f"Matplotlib not available for plots: {e}").pack(pady=10)
        
        # Close button
        ttk.Button(main_frame, text="Close", command=lambda: (canvas_outer.unbind_all("<MouseWheel>"), flatness_window.destroy())).pack(pady=10)
        
        logger.info(f"Flatness analysis ({image_source}): avg CV={avg_cv:.2f}%, grade={uniformity_grade}")
    
    def show_about(self):
        """Show application information with a link to the author's profile."""
        dialog = tk.Toplevel(self.parent)
        dialog.title("About Radiochromic Film Analyzer")
        dialog.transient(self.parent)
        dialog.resizable(False, False)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=24)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            frame,
            text="Radiochromic Film Analyzer",
            font=("TkDefaultFont", 14, "bold"),
        ).pack(pady=(0, 12))
        ttk.Label(
            frame,
            text="A tool for analyzing radiochromic films and calculating dose.",
            wraplength=380,
            justify=tk.CENTER,
        ).pack(pady=(0, 16))
        ttk.Label(frame, text="Created by").pack()
        author_link = ttk.Label(
            frame,
            text=AUTHOR_NAME,
            foreground="#0563C1",
            cursor="hand2",
            font=("TkDefaultFont", 10, "underline"),
        )
        author_link.pack(pady=(2, 14))
        author_link.bind("<Button-1>", lambda _event: webbrowser.open_new_tab(AUTHOR_GITHUB_URL))
        ttk.Label(frame, text=f"Version {__version__}\nLicensed under the GNU GPL v3", justify=tk.CENTER).pack()
        ttk.Button(frame, text="Close", command=dialog.destroy).pack(pady=(18, 0))

        dialog.update_idletasks()
        x = self.parent.winfo_rootx() + max((self.parent.winfo_width() - dialog.winfo_width()) // 2, 0)
        y = self.parent.winfo_rooty() + max((self.parent.winfo_height() - dialog.winfo_height()) // 2, 0)
        dialog.geometry(f"+{x}+{y}")

    def open_local_documentation(self):
        """Open the installed documentation index in the default browser."""
        documentation = PROJECT_ROOT / "docs" / "Home.md"
        if not documentation.is_file():
            messagebox.showerror(
                "Documentation unavailable",
                "The local documentation index is missing. Reinstall the application to restore it.",
                parent=self.parent,
            )
            return
        try:
            webbrowser.open_new_tab(documentation.resolve().as_uri())
        except Exception as exc:
            logger.error("Could not open local documentation", exc_info=True)
            messagebox.showerror("Documentation unavailable", str(exc), parent=self.parent)
    
    def check_for_updates(self):
        """Check for updates from GitHub and offer to update if available."""
        from app.utils.updater import UpdateChecker
        
        # Create update checker
        updater = UpdateChecker()
        
        # Create a progress dialog
        update_window = tk.Toplevel(self.parent)
        update_window.title("Check for Updates")
        update_window.geometry("450x300")
        update_window.resizable(False, False)
        update_window.grab_set()  # Modal
        
        # Center the window
        update_window.update_idletasks()
        x = (update_window.winfo_screenwidth() - 450) // 2
        y = (update_window.winfo_screenheight() - 300) // 2
        update_window.geometry(f"+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(update_window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(
            main_frame, 
            text="Software Update", 
            font=("Arial", 14, "bold")
        ).pack(anchor=tk.W, pady=(0, 15))
        
        # Status text widget
        status_text = tk.Text(main_frame, height=8, width=50, state=tk.DISABLED, wrap=tk.WORD)
        status_text.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        update_btn = ttk.Button(button_frame, text="Update Now", state=tk.DISABLED)
        update_btn.pack(side=tk.RIGHT, padx=5)
        
        close_btn = ttk.Button(button_frame, text="Close", command=update_window.destroy)
        close_btn.pack(side=tk.RIGHT, padx=5)
        
        def append_status(text):
            """Append text to the status widget."""
            status_text.config(state=tk.NORMAL)
            status_text.insert(tk.END, text + "\n")
            status_text.see(tk.END)
            status_text.config(state=tk.DISABLED)
            update_window.update()
        
        def check_updates_thread():
            """Background thread to check for updates."""
            try:
                self.parent.after(0, lambda: append_status("Checking for updates..."))
                
                # Use the updater module to check for updates
                result = updater.check_for_updates()
                
                if not result['success']:
                    self.parent.after(0, lambda: append_status(f"❌ Error: {result['error']}"))
                    return
                
                current_version = result['current_version']
                latest_version = result['latest_version']
                self.parent.after(0, lambda: append_status(f"Installed version: {current_version}"))
                self.parent.after(0, lambda: append_status(f"Latest release: {latest_version}"))
                
                if not result['has_updates']:
                    self.parent.after(0, lambda: append_status("\n✅ You are running the latest version!"))
                else:
                    self.parent.after(0, lambda: append_status("\nA newer published release is available."))
                    if not getattr(sys, "frozen", False):
                        self.parent.after(0, lambda: append_status(
                            "Automatic installation is available in the installed application."
                        ))
                        return
                    
                    # Enable update button
                    def enable_update():
                        update_btn.config(state=tk.NORMAL, command=do_update)
                    self.parent.after(0, enable_update)
                    
            except Exception as e:
                message = str(e)
                self.parent.after(0, lambda: append_status(f"❌ Error: {message}"))
        
        def do_update():
            """Perform the update."""
            update_btn.config(state=tk.DISABLED)
            threading.Thread(target=perform_update_thread, daemon=True).start()
        
        def perform_update_thread():
            """Background thread to perform the update."""
            try:
                self.parent.after(0, lambda: append_status("\nDownloading the release installer..."))
                release, error = updater.get_latest_release()
                result = ({"success": False, "error": error} if release is None
                          else updater.download_release_installer(release))
                if result['success']:
                    result = updater.prepare_installer_update(result['path'])
                
                if result['success']:
                    self.parent.after(0, lambda: append_status("✅ Update downloaded and verified."))
                    self.parent.after(0, lambda: append_status("The application will close while the release installer updates program files and then restart."))
                    
                    # Show restart prompt
                    def prompt_restart():
                        if messagebox.askyesno(
                            "Update Complete",
                            "The release installer is ready.\n\n"
                            "Close the application now to install the update and restart?"
                        ):
                            update_window.destroy()
                            self.parent.destroy()
                    
                    self.parent.after(0, prompt_restart)
                else:
                    self.parent.after(0, lambda: append_status(f"❌ Error during update:\n{result['error']}"))
                    
            except Exception as e:
                message = str(e)
                self.parent.after(0, lambda: append_status(f"❌ Error: {message}"))
        
        # Start checking in background
        threading.Thread(target=check_updates_thread, daemon=True).start()

    def open_downgrade_dialog(self):
        """Show a dialog that lets the user choose an older version by date."""
        from app.utils.updater import UpdateChecker

        updater = UpdateChecker()
        versions_by_item = {}

        dialog = tk.Toplevel(self.parent)
        dialog.title("Downgrade Version")
        dialog.geometry("760x500")
        dialog.minsize(700, 420)
        dialog.grab_set()

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - 760) // 2
        y = (dialog.winfo_screenheight() - 500) // 2
        dialog.geometry(f"+{x}+{y}")

        main_frame = ttk.Frame(dialog, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main_frame,
            text="Choose Version by Date",
            font=("Arial", 14, "bold")
        ).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(
            main_frame,
            text=(
                "Select a commit from the remote history. The application will switch to that version "
                "and restart in detached HEAD mode."
            ),
            wraplength=700,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 10))

        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("date", "commit", "message")
        version_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=14)
        version_tree.heading("date", text="Date")
        version_tree.heading("commit", text="Commit")
        version_tree.heading("message", text="Message")
        version_tree.column("date", width=100, anchor=tk.CENTER)
        version_tree.column("commit", width=90, anchor=tk.CENTER)
        version_tree.column("message", width=480, anchor=tk.W)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=version_tree.yview)
        version_tree.configure(yscrollcommand=scrollbar.set)
        version_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        status_text = tk.Text(main_frame, height=6, state=tk.DISABLED, wrap=tk.WORD)
        status_text.pack(fill=tk.BOTH, expand=False, pady=(12, 10))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        downgrade_btn = ttk.Button(button_frame, text="Downgrade to Selected Version", state=tk.DISABLED)
        downgrade_btn.pack(side=tk.RIGHT, padx=5)

        refresh_btn = ttk.Button(button_frame, text="Refresh List")
        refresh_btn.pack(side=tk.RIGHT, padx=5)

        close_btn = ttk.Button(button_frame, text="Close", command=dialog.destroy)
        close_btn.pack(side=tk.RIGHT, padx=5)

        def append_status(text):
            status_text.config(state=tk.NORMAL)
            status_text.insert(tk.END, text + "\n")
            status_text.see(tk.END)
            status_text.config(state=tk.DISABLED)
            dialog.update()

        def on_version_select(_event=None):
            selection = version_tree.selection()
            if not selection:
                downgrade_btn.config(state=tk.DISABLED)
                return

            version_info = versions_by_item.get(selection[0])
            if version_info is None or version_info.get("is_current"):
                downgrade_btn.config(state=tk.DISABLED)
                return

            downgrade_btn.config(state=tk.NORMAL)

        version_tree.bind("<<TreeviewSelect>>", on_version_select)

        def populate_versions(result):
            version_tree.delete(*version_tree.get_children())
            versions_by_item.clear()

            if not result['success']:
                append_status(f"❌ Error: {result['error']}")
                return

            current_commit = result.get('current_commit')
            if current_commit:
                append_status(f"Current version: {current_commit}")

            for version in result['versions']:
                message = version['subject']
                if version.get('is_current'):
                    message = f"[current] {message}"
                item_id = version_tree.insert(
                    "",
                    "end",
                    values=(version['date'], version['short_commit'], message),
                )
                versions_by_item[item_id] = version

            append_status(f"Loaded {len(result['versions'])} remote version(s).")
            on_version_select()

        def load_versions_thread():
            self.parent.after(0, lambda: append_status("Fetching remote version history..."))
            result = updater.list_remote_versions()
            self.parent.after(0, lambda: populate_versions(result))

        def perform_downgrade_thread(commit_hash, commit_date, subject):
            try:
                self.parent.after(0, lambda: append_status(
                    f"Switching to {commit_hash[:8]} from {commit_date}: {subject}"
                ))

                if updater.has_local_changes():
                    self.parent.after(0, lambda: append_status("Stashing local changes..."))

                result = updater.checkout_version(commit_hash)
                if result['success']:
                    self.parent.after(0, lambda: append_status("✅ Downgrade successful!"))
                    self.parent.after(0, lambda: append_status(
                        "\n⚠️ The application needs to restart to run the selected version."
                    ))

                    def prompt_restart():
                        if messagebox.askyesno(
                            "Downgrade Complete",
                            "The selected version has been checked out successfully.\n\n"
                            "Would you like to restart now to apply the change?",
                            parent=dialog,
                        ):
                            dialog.destroy()
                            updater.restart_application()

                    self.parent.after(0, prompt_restart)
                else:
                    self.parent.after(0, lambda: append_status(
                        f"❌ Error during downgrade:\n{result['error']}"
                    ))
            except Exception as exc:
                message = str(exc)
                self.parent.after(0, lambda: append_status(f"❌ Error: {message}"))
            finally:
                self.parent.after(0, on_version_select)

        def do_downgrade():
            selection = version_tree.selection()
            if not selection:
                return

            version_info = versions_by_item.get(selection[0])
            if version_info is None:
                return

            if not messagebox.askyesno(
                "Confirm Downgrade",
                f"Downgrade to version from {version_info['date']}\n"
                f"Commit: {version_info['short_commit']}\n\n"
                f"{version_info['subject']}\n\n"
                "The application will restart after changing version.",
                parent=dialog,
            ):
                return

            downgrade_btn.config(state=tk.DISABLED)
            threading.Thread(
                target=perform_downgrade_thread,
                args=(version_info['commit'], version_info['date'], version_info['subject']),
                daemon=True,
            ).start()

        downgrade_btn.config(command=do_downgrade)
        refresh_btn.config(command=lambda: threading.Thread(target=load_versions_thread, daemon=True).start())

        threading.Thread(target=load_versions_thread, daemon=True).start()
    
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

    def cleanup(self):
        """Clean up resources."""
        # Clean up image panel resources
        if hasattr(self, 'image_panel'):
            self.image_panel.cleanup()
        from app.plugins.plugin_manager import plugin_manager
        plugin_manager.shutdown()
