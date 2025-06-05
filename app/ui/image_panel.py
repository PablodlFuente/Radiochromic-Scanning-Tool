"""
Image panel UI for the Radiochromic Film Analyzer.

This module contains the image panel UI class that displays and manages
the image view.
"""

import tkinter as tk
from tkinter import ttk
import logging
import threading
import queue
import time
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class ImagePanel:
    """Image panel UI for the Radiochromic Film Analyzer."""
    
    def __init__(self, parent, image_processor):
        """Initialize the image panel."""
        self.parent = parent
        self.image_processor = image_processor
        
        # Create frame
        self.frame = ttk.Frame(parent)
        
        # Initialize callbacks
        self.zoom_callback = None
        self.measurement_callback = None
        
        # Loading state
        self.loading = False
        self.current_image_tk = None
        self.is_adjustment = False  # Flag to indicate if this is an adjustment or initial load
        
        # Measurement display
        self.measurement_visible = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Measured area (from click)
        self.measured_area_visible = False
        self.measured_x = 0
        self.measured_y = 0
        self.last_measurement_results = None
        
        # Store image coordinates (not canvas coordinates) for measurements
        self.measured_img_x = 0
        self.measured_img_y = 0
        
        # Auto-measure results (separate from clicked measurement)
        self.auto_measure_results = None
        
        # Measurement threading
        self.measurement_queue = queue.Queue()
        self.measurement_thread = None
        self.measurement_active = False
        self.last_queued_position = None
        self.measurement_task_id = 0  # Task ID counter for prioritization
        self.current_task_id = 0  # Current task being processed
        
        # Debounce for auto-measure to reduce lag
        self.last_auto_measure_time = time.time()
        self.auto_measure_debounce_ms = 100  # Reduced from 250ms to 100ms with optimizations
        self.pending_auto_measure = None
        
        # Binning status
        self.binning_in_progress = False
        
        # Precise movement settings
        self.precise_mode = True
        self.precise_step = 3  # Fixed step size (3 pixels)
        
        # Create UI
        self._create_ui()
        
        # Set progress callback
        self.image_processor.set_progress_callback(self._on_progress_update)
        
        # Start measurement worker thread
        self._start_measurement_thread()
        
        logger.info("Image panel initialized")
    
    def _start_measurement_thread(self):
        """Start the measurement worker thread."""
        self.measurement_active = True
        self.measurement_thread = threading.Thread(target=self._measurement_worker, daemon=True)
        self.measurement_thread.start()
        logger.debug("Started measurement worker thread")
    
    def _stop_measurement_thread(self):
        """Stop the measurement worker thread."""
        self.measurement_active = False
        if self.measurement_thread and self.measurement_thread.is_alive():
            self.measurement_queue.put(None)  # Signal to exit
            self.measurement_thread.join(timeout=1.0)
            logger.debug("Stopped measurement worker thread")
    
    def _measurement_worker(self):
        """Worker thread for processing measurements."""
        while self.measurement_active:
            try:
                # Get the next measurement task from the queue
                task = self.measurement_queue.get(timeout=0.5)
                
                # Check for exit signal
                if task is None:
                    break
                
                # Unpack the task
                task_id, x, y = task
                
                # Skip if this task is obsolete (a newer task has been queued)
                if task_id < self.current_task_id:
                    logger.debug(f"Skipping obsolete measurement task {task_id} (current: {self.current_task_id})")
                    self.measurement_queue.task_done()
                    continue
                
                # Update current task ID
                self.current_task_id = task_id
                
                # Perform the measurement
                results = self.image_processor.measure_area(x, y)
                
                # Update the UI on the main thread
                if results:
                    # Crear un hilo separado para el procesamiento del histograma
                    if self.measurement_callback:
                        self.canvas.after(0, lambda r=results: self._update_measurement_results(r))
                        
                        # Procesar el histograma en un hilo separado si tenemos datos crudos
                        if hasattr(self.image_processor, 'last_measurement_raw_data') and self.image_processor.last_measurement_raw_data is not None:
                            threading.Thread(
                                target=self._process_histogram_data,
                                args=(self.image_processor.last_measurement_raw_data, self.image_processor.last_measurement_coordinates),
                                daemon=True
                            ).start()
            
                # Mark task as done
                self.measurement_queue.task_done()
            except queue.Empty:
                # No tasks in the queue, continue waiting
                continue
            except Exception as e:
                logger.error(f"Error in measurement worker: {str(e)}", exc_info=True)

    
    def _update_measurement_results(self, results):
        """Update measurement results on the main thread."""
        if self.measurement_callback and results:
            # Store auto-measure results
            self.auto_measure_results = results
            # Update the measurement panel with auto-measure results
            if hasattr(self.parent.winfo_toplevel(), 'main_window') and hasattr(self.parent.winfo_toplevel().main_window, 'measurement_panel'):
                self.parent.winfo_toplevel().main_window.measurement_panel.update_auto_measure_results(results)
            else:
                # Fallback to regular callback
                self.measurement_callback(results)

    def _process_histogram_data(self, raw_data, coordinates):
        """Process histogram data in a separate thread."""
        try:
            # Aquí se realizaría cualquier procesamiento pesado relacionado con el histograma
            # Cuando termine, programamos la actualización del histograma en el hilo principal
            self.canvas.after(0, lambda: self._update_histogram_from_data(raw_data))
        except Exception as e:
            logger.error(f"Error processing histogram data: {str(e)}", exc_info=True)

    def _update_histogram_from_data(self, raw_data):
        """Update the histogram UI with processed data."""
        # Buscar el panel de medición para actualizar el histograma
        if hasattr(self.parent.winfo_toplevel(), 'main_window') and hasattr(self.parent.winfo_toplevel().main_window, 'measurement_panel'):
            measurement_panel = self.parent.winfo_toplevel().main_window.measurement_panel
            measurement_panel._update_histogram()

    def _create_ui(self):
        """Create the UI components."""
        # Frame for canvas and scrollbars
        self.canvas_frame = ttk.Frame(self.frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas to display the image
        self.canvas = tk.Canvas(self.canvas_frame, bg="lightgray", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbars
        self.scrollbar_y = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.scrollbar_y.grid(row=0, column=1, sticky="ns")
        
        self.scrollbar_x = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        # Configure scrollbars
        self.scrollbar_y.config(command=self.canvas.yview)
        self.scrollbar_x.config(command=self.canvas.xview)
        self.canvas.config(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)
        
        # Configure grid
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Loading indicator
        self.loading_frame = ttk.Frame(self.canvas, padding=10)
        self.loading_label = ttk.Label(self.loading_frame, text="Loading image...", font=("Arial", 12))
        self.loading_label.pack(pady=5)
        self.loading_progress = ttk.Progressbar(self.loading_frame, mode="indeterminate", length=200)
        self.loading_progress.pack(pady=5)
        
        # Controls frame
        self.controls_frame = ttk.Frame(self.frame)
        self.controls_frame.pack(fill=tk.X, pady=5)

        # Zoom percentage label
        self.zoom_percentage = ttk.Label(self.controls_frame, text="100%")
        self.zoom_percentage.pack(side=tk.LEFT, padx=5)
        
        # Binning control
        ttk.Label(self.controls_frame, text="Binning:").pack(side=tk.LEFT, padx=(20, 5))
        
        # Binning dropdown
        self.binning_var = tk.StringVar(value="1x1")
        
        self.binning_options = ["1x1", "2x2", "4x4", "6x6"]
        self.binning_combobox = ttk.Combobox(
            self.controls_frame,
            textvariable=self.binning_var,
            values=self.binning_options,
            width=6,
            state="readonly"
        )
        self.binning_combobox.pack(side=tk.LEFT, padx=5)
        self.binning_combobox.bind("<<ComboboxSelected>>", self._on_binning_change)
        
        # Position indicator
        self.position_label = ttk.Label(self.controls_frame, text="Pos: --,--")
        self.position_label.pack(side=tk.LEFT, padx=10)

        # Status frame with progress bar
        self.status_frame = ttk.Frame(self.frame)
        self.status_frame.pack(fill=tk.X, pady=2)
        
        # Binning status label and progress bar
        self.status_label = ttk.Label(
            self.status_frame, 
            text="", 
            foreground="blue",
            font=("Arial", 10, "bold")
        )
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Progress percentage label
        self.progress_percentage = ttk.Label(
            self.status_frame,
            text="",
            width=8,
            anchor=tk.E
        )
        self.progress_percentage.pack(side=tk.RIGHT, padx=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self.status_frame, 
            mode="determinate", 
            length=200
        )
        self.progress_bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Cancel button
        self.cancel_button = ttk.Button(
            self.status_frame,
            text="Cancel",
            command=self._cancel_operation
        )
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # Hide status frame initially
        self.status_frame.pack_forget()

        # Fit to screen button
        ttk.Button(
            self.controls_frame, 
            text="Fit to Screen", 
            command=self.fit_to_screen
        ).pack(side=tk.RIGHT, padx=5)

        # Reset zoom button
        ttk.Button(
            self.controls_frame, 
            text="Reset Zoom", 
            command=self.reset_zoom
        ).pack(side=tk.RIGHT, padx=5)
        
        # Initialize callbacks
        self.zoom_callback = None
        
        # Bind events
        self.canvas.bind("<ButtonPress-2>", self._start_pan)  # Middle button
        self.canvas.bind("<B2-Motion>", self._pan_image)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Leave>", self._on_mouse_leave)
        
        # Bind mouse wheel for zooming
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", lambda e: self._on_mouse_wheel(e, 1))  # Linux scroll up
        self.canvas.bind("<Button-5>", lambda e: self._on_mouse_wheel(e, -1))  # Linux scroll down
        
        # Bind keyboard events for precise movement
        self.canvas.bind("<KeyPress>", self._on_key_press)
        # Ensure canvas can receive keyboard focus
        self.canvas.config(takefocus=1)
        # Bind to root window for global keyboard handling
        self.frame.winfo_toplevel().bind("<KeyPress>", self._on_key_press)
        self.canvas.bind("<FocusIn>", lambda e: self._show_focus_indicator())
        self.canvas.bind("<FocusOut>", lambda e: self._hide_focus_indicator())
        
        # Make canvas focusable
        self.canvas.config(takefocus=1)
    
    def _toggle_precise_mode(self):
        """Toggle precise movement mode."""
        # Always enable precise mode
        self.precise_mode = True
        
        logger.debug(f"Precise mode enabled with step size {self.precise_step}")
    
    def _show_precise_mode_instructions(self):
        """Show instructions for precise mode."""
        # Create instruction text on canvas
        self.canvas.delete("instructions")
        
        if self.measured_area_visible:
            instructions = "Use arrow keys to move measurement area. Press Shift+arrows for larger steps."
            self.canvas.create_text(
                10, 10, 
                text=instructions,
                anchor=tk.NW,
                fill="blue",
                tags="instructions"
            )
    
    def _hide_precise_mode_instructions(self):
        """Hide instructions for precise mode."""
        self.canvas.delete("instructions")
    
    def _show_focus_indicator(self):
        """Show a visual indicator that the canvas has focus."""
        if self.precise_mode:
            self.canvas.config(highlightthickness=2, highlightbackground="blue")
    
    def _hide_focus_indicator(self):
        """Hide the focus indicator."""
        self.canvas.config(highlightthickness=0)
    
    def _on_key_press(self, event):
        """Handle key press events for precise movement."""
        if not self.measured_area_visible:
            return
        
        # Get current step size
        step = self.precise_step
        
        # Increase step size if Shift is pressed
        if event.state & 0x1:  # Shift key
            step *= 5
        
        # Handle arrow keys
        moved = False
        
        if event.keysym == "Up":
            self.measured_y -= step
            moved = True
        elif event.keysym == "Down":
            self.measured_y += step
            moved = True
        elif event.keysym == "Left":
            self.measured_x -= step
            moved = True
        elif event.keysym == "Right":
            self.measured_x += step
            moved = True
        
        if moved:
            # Update image coordinates
            current_zoom = self.image_processor.get_zoom()
            self.measured_img_x = int(self.measured_x / current_zoom)
            self.measured_img_y = int(self.measured_y / current_zoom)
            
            # Update position label
            self.position_label.config(text=f"Pos: {self.measured_img_x},{self.measured_img_y}")
            
            # Redraw measured area
            self._draw_measured_area(self.measured_x, self.measured_y)
            
            # Update measurement results
            results = self.image_processor.measure_area(self.measured_x, self.measured_y)
            if results and self.measurement_callback:
                self.last_measurement_results = results
                self.measurement_callback(results)
            
            logger.debug(f"Moved measurement area to ({self.measured_img_x}, {self.measured_img_y})")
    
    def _on_progress_update(self, operation, progress, status=None):
        """Handle progress updates from the image processor."""
        # Schedule UI update on the main thread
        self.canvas.after(0, lambda: self._update_progress_ui(operation, progress, status))
    
    def _update_progress_ui(self, operation, progress, status=None):
        """Update the progress UI on the main thread."""
        if operation == "binning":
            # Show status frame if not already visible
            if not self.binning_in_progress and progress < 100:
                self.binning_in_progress = True
                self.status_frame.pack(fill=tk.X, pady=2, before=self.canvas_frame)
                self.frame.update_idletasks()  # Force UI update
            
            # Update progress bar
            self.progress_bar["value"] = progress
            
            # Update percentage label
            self.progress_percentage.config(text=f"{int(progress)}%")
            
            # Update status label
            if status:
                self.status_label.config(text=status)
            else:
                self.status_label.config(text="Applying binning...")
            
            # Make sure the cancel button is visible
            self.cancel_button.pack(side=tk.RIGHT, padx=5)
            
            # Hide status frame when complete
            if progress >= 100:
                # Wait a moment before hiding to show completion
                self.canvas.after(1000, self._hide_status_frame)
        
        elif operation == "loading":
            # Show status frame if not already visible
            if not self.binning_in_progress and progress < 100:
                self.binning_in_progress = True
                self.status_frame.pack(fill=tk.X, pady=2, before=self.canvas_frame)
                self.frame.update_idletasks()  # Force UI update
            
            # Update progress bar
            self.progress_bar["value"] = progress
            
            # Update percentage label
            self.progress_percentage.config(text=f"{int(progress)}%")
            
            # Update status label
            if status:
                self.status_label.config(text=status)
            else:
                self.status_label.config(text="Loading image...")
            
            # Hide status frame when complete
            if progress >= 100:
                # Wait a moment before hiding to show completion
                self.canvas.after(500, self._hide_status_frame)
    
    def _hide_status_frame(self):
        """Hide the status frame."""
        if self.binning_in_progress:
            self.binning_in_progress = False
            self.status_frame.pack_forget()
            self.frame.update_idletasks()  # Force UI update
            logger.debug("Hiding status frame")
    
    def _cancel_operation(self):
        """Cancel the current operation."""
        self.image_processor.cancel_current_operation()
        self.status_label.config(text="Cancelling operation...")
        logger.debug("Cancelling operation")
    
    def _on_binning_change(self, event):
        """Handle binning change."""
        binning_str = self.binning_var.get()
        binning = int(binning_str.split('x')[0])
        
        # Use a separate thread to apply binning to keep UI responsive
        threading.Thread(target=self._apply_binning, args=(binning,), daemon=True).start()
        
        logger.debug(f"Binning change initiated to {binning_str}")
    
    def _apply_binning(self, binning):
        """Apply binning in a background thread."""
        try:
            # Update image processor
            self.image_processor.set_binning(binning)
            
            # Schedule UI update on the main thread
            self.canvas.after(0, lambda: self._after_binning_applied())
        except Exception as e:
            logger.error(f"Error applying binning: {str(e)}", exc_info=True)
            # Hide binning status on error
            self.canvas.after(0, self._hide_status_frame)
    
    def _after_binning_applied(self):
        """Handle UI updates after binning is applied."""
        # Redisplay the image
        self.display_image(is_adjustment=True)
        
        logger.debug("Binning applied successfully")
    
    def display_image(self, is_adjustment=False):
        """Display the current image."""
        if not self.image_processor.has_image():
            return
        
        # Set adjustment flag
        self.is_adjustment = is_adjustment
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas doesn't have dimensions yet, schedule update
            self.canvas.after(100, lambda: self.display_image(is_adjustment))
            return
        
        # Show loading indicator only for initial loads, not adjustments
        if not is_adjustment:
            self._show_loading()
        
        # Process image in a separate thread
        threading.Thread(target=self._process_and_display_image, daemon=True).start()
    
    def _process_and_display_image(self):
        """Process and display the image in a background thread."""
        try:
            # Get the processed image
            logger.debug("Getting display image from image processor")
            image_tk, width, height = self.image_processor.get_display_image()
            
            if image_tk is None:
                logger.error("Image processor returned None for display image")
            else:
                logger.debug(f"Got display image with dimensions: {width}x{height}")
            
            # Schedule UI update on the main thread
            self.canvas.after(0, lambda: self._update_canvas(image_tk, width, height))
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            # Hide loading indicator on error
            self.canvas.after(0, self._hide_loading)
    
    def _update_canvas(self, image_tk, width, height):
        """Update the canvas with the processed image."""
        if image_tk:
            # Update canvas scrollregion
            self.canvas.config(scrollregion=(0, 0, width, height))
        
            # Display on canvas
            self.canvas.delete("all")
            logger.debug(f"Updating canvas with image dimensions: {width}x{height}")
            self.canvas.create_image(
                width//2, 
                height//2, 
                image=image_tk, 
                anchor=tk.CENTER,
                tags="image"
            )
        
            # Store reference to prevent garbage collection
            self.current_image_tk = image_tk
        
            # Update zoom percentage
            current_zoom = self.image_processor.get_zoom()
            self.zoom_percentage.config(text=f"{int(current_zoom * 100)}%")
        
            # Update zoom callback
            if self.zoom_callback:
                self.zoom_callback(current_zoom)
        
            # Redraw measurement shape if needed
            if self.measurement_visible:
                self._draw_measurement_shape(self.last_mouse_x, self.last_mouse_y)
            
            # Update measured area position if needed
            if self.measured_area_visible:
                self.update_measured_area_position()
                
                # Show precise mode instructions if enabled
                if self.precise_mode:
                    self._show_precise_mode_instructions()
        
            logger.debug("Image displayed successfully")
        else:
            logger.error("Failed to display image: image_tk is None")
    
        # Hide loading indicator
        self._hide_loading()
    
    def _show_loading(self):
        """Show the loading indicator."""
        if not self.loading and not self.is_adjustment:
            self.loading = True
            self.loading_window = self.canvas.create_window(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                window=self.loading_frame,
                tags="loading"
            )
            self.loading_progress.start(10)
    
    def _hide_loading(self):
        """Hide the loading indicator."""
        if self.loading:
            self.loading = False
            self.loading_progress.stop()
            self.canvas.delete("loading")
    
    def fit_to_screen(self):
        """Fit the image to the screen."""
        if not self.image_processor.has_image():
            return
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas doesn't have dimensions yet, schedule update
            self.canvas.after(100, self.fit_to_screen)
            return
        
        logger.debug(f"Fitting image to screen: canvas dimensions {canvas_width}x{canvas_height}")
        
        # Calculate zoom to fit
        self.image_processor.fit_to_screen(canvas_width, canvas_height)
        
        # Reset scroll position
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        
        # Update display
        self.display_image(is_adjustment=True)
        
        logger.debug("Image fit to screen")
    
    def reset_zoom(self):
        """Reset zoom to 100%."""
        if not self.image_processor.has_image():
            return
        
        # Reset zoom to exactly 1.0
        self.image_processor.set_zoom(1.0)
        
        # Reset scroll position
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        
        # Update display
        self.display_image(is_adjustment=True)
        
        logger.debug("Zoom reset to 100%")
    
    def _start_pan(self, event):
        """Start panning the image."""
        self.canvas.scan_mark(event.x, event.y)
    
    def _pan_image(self, event):
        """Pan the image."""
        self.canvas.scan_dragto(event.x, event.y, gain=1)
    
    def _debounce_auto_measure(self, x, y):
        """Debounce auto-measure to reduce UI lag."""
        # Cancel any pending auto-measure
        if self.pending_auto_measure:
            self.canvas.after_cancel(self.pending_auto_measure)
            self.pending_auto_measure = None
    
        # Usar un tiempo de debounce más corto para mejorar la respuesta
        debounce_time = 20  # ms
    
        # Schedule a new auto-measure
        self.pending_auto_measure = self.canvas.after(
            debounce_time, 
            lambda: self._queue_auto_measure(x, y)
        )
    
    def _queue_auto_measure(self, x, y):
        """Queue an auto-measure after debounce."""
        # Clear pending flag
        self.pending_auto_measure = None
        
        # Clear the measurement queue to avoid processing obsolete measurements
        # This is a key optimization to prevent lag when moving the mouse quickly
        try:
            # Clear the queue without blocking
            while not self.measurement_queue.empty():
                try:
                    self.measurement_queue.get_nowait()
                    self.measurement_queue.task_done()
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"Error clearing measurement queue: {str(e)}", exc_info=True)
        
        # Increment task ID for prioritization
        self.measurement_task_id += 1
        current_task_id = self.measurement_task_id
        
        # Store the position we're queuing
        self.last_queued_position = (x, y)
        
        # Queue the measurement task with its ID
        self.measurement_queue.put((current_task_id, x, y))
        
        logger.debug(f"Queued measurement task {current_task_id} at ({x}, {y})")
    
    def _on_mouse_move(self, event):
        """Handle mouse movement over the image."""
        if not self.image_processor.has_image() or self.loading:
            return
    
        # Get canvas position
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Store last mouse position
        self.last_mouse_x = x
        self.last_mouse_y = y
        
        # Update cursor info
        img_x, img_y, rgb = self.image_processor.get_pixel_info(x, y)

        # Find the main window by traversing up the widget hierarchy
        root = self.frame.winfo_toplevel()
        if hasattr(root, 'main_window'):
            main_window = root.main_window
        
            if img_x is not None:
                # Update coordinates and RGB in main window
                main_window.update_coordinates(img_x, img_y)
            
                # Check if we have binned data with standard deviation
                if isinstance(rgb, tuple) and len(rgb) == 2:
                    # Tenemos un tuple (valor, std_dev)
                    main_window.update_rgb(rgb[0], std_dev=rgb[1])
                else:
                    main_window.update_rgb(rgb)
            
                # Draw measurement shape immediately (lightweight operation)
                self._draw_measurement_shape(x, y)
                self.measurement_visible = True

                # Automatically calculate and update measurement results if enabled
                if hasattr(main_window, 'measurement_panel') and main_window.measurement_panel.is_auto_measure_enabled():
                    # Use debouncing to reduce frequency of measurements
                    self._debounce_auto_measure(x, y)
            else:
                # Clear info
                main_window.update_coordinates("--", "--")
                main_window.update_rgb("--")
            
                # Clear measurement shape
                self.canvas.delete("measurement_shape")
                self.measurement_visible = False
    
    def _on_mouse_leave(self, event):
        """Handle mouse leaving the canvas."""
        # Clear the temporary measurement shape (red outline)
        self.canvas.delete("measurement_shape")
        self.measurement_visible = False
        
        # Cancel any pending auto-measure
        if self.pending_auto_measure:
            self.canvas.after_cancel(self.pending_auto_measure)
            self.pending_auto_measure = None
        
        # Find the main window
        root = self.frame.winfo_toplevel()
        if hasattr(root, 'main_window') and hasattr(root.main_window, 'measurement_panel'):
            measurement_panel = root.main_window.measurement_panel
            
            # If we have a fixed measurement (green outline), show its results
            if self.measured_area_visible and self.last_measurement_results:
                logger.debug("Mouse left image: Showing fixed measurement results")
                # Force update with the fixed measurement results
                measurement_panel.update_results(self.last_measurement_results)
            else:
                # If no fixed measurement, clear the results
                logger.debug("Mouse left image: Clearing measurement results (no fixed measurement)")
                measurement_panel.update_results(None)
    
    def _on_click(self, event):
        """Handle mouse click on the image."""
        if not self.image_processor.has_image() or self.loading:
            return
        
        # Get canvas position
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Calculate position in the original image
        img_x = int(x / self.image_processor.get_zoom())
        img_y = int(y / self.image_processor.get_zoom())
        
        # Store the image coordinates (not canvas coordinates)
        self.measured_img_x = img_x
        self.measured_img_y = img_y
        
        # Update position label
        self.position_label.config(text=f"Pos: {img_x},{img_y}")
        
        # Perform measurement
        results = self.image_processor.measure_area(x, y)
        
        if results:
            # Store the measured position and results
            self.measured_x = x
            self.measured_y = y
            self.measured_area_visible = True
            self.last_measurement_results = results
            
            # Focus the canvas to receive keyboard events
            self.canvas.focus_set()
            
            # Show instructions
            self._show_precise_mode_instructions()
        
        # Draw measured area
        self._draw_measured_area(x, y)

        # Update measurement panel via callback
        if self.measurement_callback:
            self.measurement_callback(results)
    
    def _on_mouse_wheel(self, event, delta=None):
        """Handle mouse wheel for zooming."""
        if not self.image_processor.has_image() or self.loading:
            return
    
        # Check if Ctrl key is pressed
        if event.state & 0x4:  # Ctrl key
            # Let the global Ctrl+wheel handler take care of it
            return
    
        # Determine zoom factor
        if delta is None:  # For Windows
            delta = event.delta / 120
    
        # If trying to zoom in (delta > 0), check if image already fills the visible area
        if delta > 0:
            # Get current image dimensions
            current_width = self.image_processor.current_image.shape[1] * self.image_processor.zoom
            current_height = self.image_processor.current_image.shape[0] * self.image_processor.zoom
        
            # Get visible canvas area
            visible_width = self.canvas.winfo_width()
            visible_height = self.canvas.winfo_height()
        
            # If image already fills both dimensions, don't zoom in further
            if current_width >= visible_width and current_height >= visible_height:
                return "break"
    
        # Get canvas position
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
    
        # Zoom at cursor position
        zoom_changed = self.image_processor.zoom_at_point(x, y, delta)

        # Update display only if zoom changed
        if zoom_changed:
            # Update zoom percentage
            current_zoom = self.image_processor.get_zoom()
            self.zoom_percentage.config(text=f"{int(current_zoom * 100)}%")
            self.display_image(is_adjustment=True)
    
        logger.debug(f"Zoomed to {self.image_processor.get_zoom():.2f}x at ({x}, {y})")
    
        # Prevent event from propagating
        return "break"
    
    def _on_zoom_change(self, zoom):
        """Handle zoom change."""
        if not self.image_processor.has_image() or self.loading:
            return
    
        # Update zoom percentage display
        self.zoom_percentage.config(text=f"{int(zoom * 100)}%")
    
        # Update display
        self.display_image(is_adjustment=True)
    
    def _draw_measurement_shape(self, x, y):
        """Draw the measurement shape at the specified position."""
        # Get measurement settings
        shape, size = self.image_processor.get_measurement_settings()
        
        # Apply zoom
        size_px = size * self.image_processor.get_zoom()
        
        # Draw shape
        self.canvas.delete("measurement_shape")
        
        if shape == "circular":
            self.canvas.create_oval(
                x - size_px, y - size_px,
                x + size_px, y + size_px,
                outline="red", tags="measurement_shape"
            )
        else:  # square
            self.canvas.create_rectangle(
                x - size_px, y - size_px,
                x + size_px, y + size_px,
                outline="red", tags="measurement_shape"
            )
    
    def _draw_measured_area(self, x, y):
        """Draw the measured area at the specified position."""
        # Get measurement settings
        shape, size = self.image_processor.get_measurement_settings()
        
        # Apply zoom
        size_px = size * self.image_processor.get_zoom()
        
        # Draw shape
        self.canvas.delete("measured_area")
        
        if shape == "circular":
            # Fixed: Corrected the coordinates for the circular shape
            self.canvas.create_oval(
                x - size_px, y - size_px,
                x + size_px, y + size_px,
                outline="green", width=2, tags="measured_area"
            )
        else:  # square
            self.canvas.create_rectangle(
                x - size_px, y - size_px,
                x + size_px, y + size_px,
                outline="green", width=2, tags="measured_area"
            )

    def set_zoom_callback(self, callback):
        """Set the callback for zoom changes."""
        self.zoom_callback = callback
    
    def set_measurement_callback(self, callback):
        """Set the callback for measurement updates."""
        self.measurement_callback = callback

    def update_measured_area_position(self):
        """Update the measured area position based on current zoom."""
        if self.measured_area_visible and self.measured_img_x and self.measured_img_y:
            # Calculate new canvas position based on image coordinates and current zoom
            current_zoom = self.image_processor.get_zoom()
            new_x = self.measured_img_x * current_zoom
            new_y = self.measured_img_y * current_zoom
        
            # Update stored canvas coordinates
            self.measured_x = new_x
            self.measured_y = new_y
            
            # Update position label
            self.position_label.config(text=f"Pos: {self.measured_img_x},{self.measured_img_y}")
        
            # Redraw the measured area
            self._draw_measured_area(new_x, new_y)
        
            # Update measurement results
            if self.measurement_callback:
                # Recalculate measurement at the same image position with new zoom
                results = self.image_processor.measure_area(new_x, new_y)
                if results:
                    self.last_measurement_results = results
                    self.measurement_callback(results)

    def cleanup(self):
        """Clean up resources."""
        self._stop_measurement_thread()
    
    def _clear_measurement(self):
        """Clear the current measurement."""
        if not self.image_processor.has_image() or self.loading:
            return
        
        # Clear measured area
        self.canvas.delete("measured_area")
        self.measured_area_visible = False
        self.last_measurement_results = None
        self.auto_measure_results = None
        
        # Clear instructions
        self.canvas.delete("instructions")
        
        # Find the main window
        root = self.frame.winfo_toplevel()
        if hasattr(root, 'main_window') and hasattr(root.main_window, 'measurement_panel'):
            measurement_panel = root.main_window.measurement_panel
            
            # Clear results
            measurement_panel.update_results(None)

    def measure_area(self, canvas_x, canvas_y):
        """Measure the area at the specified canvas position."""
        if self.image_processor.current_image is None:
            return None
    
        try:
            # Calculate position in the current image
            img_x = int(canvas_x / self.image_processor.zoom)
            img_y = int(canvas_y / self.image_processor.zoom)
        
            # Check if position is within image bounds
            if (0 <= img_x < self.image_processor.current_image.shape[1] and 
                0 <= img_y < self.image_processor.current_image.shape[0]):
            
                # For auto-measure, use a smaller measurement size to improve performance
                is_auto_measure = self.image_processor.auto_measure and not hasattr(self, 'measured_area_visible')
                actual_size = self.image_processor.measurement_size // 2 if is_auto_measure else self.image_processor.measurement_size
            
                # Create mask for the region of interest
                if self.image_processor.measurement_shape == "circular":
                    # Create circular mask using OpenCV
                    mask = np.zeros(self.image_processor.current_image.shape[:2], dtype=np.uint8)
                    cv2.circle(mask, (img_x, img_y), actual_size, 255, -1)
                    
                    # Get coordinates of all pixels in the mask
                    y_coords, x_coords = np.where(mask > 0)
                    
                    # Store coordinates relative to the center of the measurement
                    rel_coords = np.column_stack((x_coords - img_x, y_coords - img_y))
                    
                    # Count pixels in the mask
                    pixel_count = len(x_coords)
                    
                    # For circular regions, we can't use integral images directly
                    # So we'll use the traditional approach
                    if len(self.image_processor.current_image.shape) == 3:  # Color image
                        # Use OpenCV's built-in functions for faster processing
                        means, std_devs = cv2.meanStdDev(self.image_processor.current_image, mask=mask)
                        means = means.flatten()
                        std_devs = std_devs.flatten()
                        
                        # Calculate uncertainties
                        uncertainties = std_devs / np.sqrt(pixel_count) if pixel_count > 0 else np.zeros_like(std_devs)
                        
                        # Collect raw data for histogram (sample if too many pixels)
                        if len(x_coords) > 1000:
                            # Take a random sample of 1000 pixels
                            indices = np.random.choice(len(x_coords), 1000, replace=False)
                            sample_x = x_coords[indices]
                            sample_y = y_coords[indices]
                            sample_rel = rel_coords[indices]
                        else:
                            sample_x = x_coords
                            sample_y = y_coords
                            sample_rel = rel_coords
                        
                        # Collect raw data for histogram
                        masked_data = np.zeros((len(sample_x), self.image_processor.current_image.shape[2]), dtype=np.uint8)
                        for c in range(self.image_processor.current_image.shape[2]):
                            masked_data[:, c] = np.array([self.image_processor.current_image[y, x, c] for x, y in zip(sample_y, sample_x)])
                        
                        # Store the raw data and coordinates for later use
                        self.image_processor.last_measurement_raw_data = masked_data
                        self.image_processor.last_measurement_coordinates = sample_rel
                        self.image_processor.last_auto_measure_time = time.time()
                    
                        return (
                            tuple(means),
                            tuple(std_devs),
                            tuple(uncertainties),
                            pixel_count
                        )
                    else:  # Grayscale image
                        # For grayscale, use a more efficient approach
                        mean, std_dev = cv2.meanStdDev(self.image_processor.current_image, mask=mask)
                        mean = float(mean[0][0])
                        std_dev = float(std_dev[0][0])
                        
                        # Calculate uncertainty
                        uncertainty = std_dev / np.sqrt(pixel_count) if pixel_count > 0 else 0
                        
                        # Collect raw data for histogram (sample if too many pixels)
                        if len(x_coords) > 1000:
                            # Take a random sample of 1000 pixels
                            indices = np.random.choice(len(x_coords), 1000, replace=False)
                            sample_x = x_coords[indices]
                            sample_y = y_coords[indices]
                            sample_rel = rel_coords[indices]
                        else:
                            sample_x = x_coords
                            sample_y = y_coords
                            sample_rel = rel_coords
                        
                        # Collect raw data for histogram
                        masked_data = np.zeros(len(sample_x), dtype=np.uint8)
                        for i, (x, y) in enumerate(zip(sample_x, sample_y)):
                            masked_data[i] = self.image_processor.current_image[y, x]
                        
                        # Store the raw data and coordinates for later use
                        self.image_processor.last_measurement_raw_data = masked_data
                        self.image_processor.last_measurement_coordinates = sample_rel
                        self.image_processor.last_auto_measure_time = time.time()
                    
                        return mean, std_dev, uncertainty, pixel_count
                else:  # Square measurement
                    # For square regions, we can use integral images for fast calculation
                    # Calculate the square boundaries
                    x1 = max(0, img_x - actual_size)
                    y1 = max(0, img_y - actual_size)
                    x2 = min(self.image_processor.current_image.shape[1] - 1, img_x + actual_size)
                    y2 = min(self.image_processor.current_image.shape[0] - 1, img_y + actual_size)
                    
                    # Count pixels in the square
                    pixel_count = (x2 - x1 + 1) * (y2 - y1 + 1)
                    
                    # Get coordinates for raw data collection
                    x_coords = np.arange(x1, x2 + 1)
                    y_coords = np.arange(y1, y2 + 1)
                    X, Y = np.meshgrid(x_coords, y_coords)
                    x_coords = X.flatten()
                    y_coords = Y.flatten()
                    
                    # Store coordinates relative to the center of the measurement
                    rel_coords = np.column_stack((x_coords - img_x, y_coords - img_y))
                    
                    if len(self.image_processor.current_image.shape) == 3:  # Color image
                        # Use integral images for each channel if available
                        if hasattr(self.image_processor, 'integral_images') and self.image_processor.integral_images is not None:
                            means = []
                            std_devs = []
                            
                            for c in range(self.image_processor.current_image.shape[2]):
                                mean, std_dev = self.image_processor._calculate_stats_from_integral(x1, y1, x2, y2, channel=c)
                                means.append(mean)
                                std_devs.append(std_dev)
                            
                            # Calculate uncertainties
                            uncertainties = [std_dev / np.sqrt(pixel_count) if pixel_count > 0 else 0 for std_dev in std_devs]
                        else:
                            # Fallback to traditional method if integral images not available
                            region = self.image_processor.current_image[y1:y2+1, x1:x2+1]
                            means = [np.mean(region[:,:,c]) for c in range(region.shape[2])]
                            std_devs = [np.std(region[:,:,c]) for c in range(region.shape[2])]
                            uncertainties = [std_dev / np.sqrt(pixel_count) if pixel_count > 0 else 0 for std_dev in std_devs]
                        
                        # Collect raw data for histogram (sample if too many pixels)
                        if len(x_coords) > 1000:
                            # Take a random sample of 1000 pixels
                            indices = np.random.choice(len(x_coords), 1000, replace=False)
                            sample_x = x_coords[indices]
                            sample_y = y_coords[indices]
                            sample_rel = rel_coords[indices]
                        else:
                            sample_x = x_coords
                            sample_y = y_coords
                            sample_rel = rel_coords
                        
                        # Collect raw data for histogram
                        masked_data = np.zeros((len(sample_x), self.image_processor.current_image.shape[2]), dtype=np.uint8)
                        for c in range(self.image_processor.current_image.shape[2]):
                            masked_data[:, c] = np.array([self.image_processor.current_image[y, x, c] for x, y in zip(sample_y, sample_x)])
                        
                        # Store the raw data and coordinates for later use
                        self.image_processor.last_measurement_raw_data = masked_data
                        self.image_processor.last_measurement_coordinates = sample_rel
                        self.image_processor.last_auto_measure_time = time.time()
                        
                        return (
                            tuple(means),
                            tuple(std_devs),
                            tuple(uncertainties),
                            pixel_count
                        )
                    else:  # Grayscale image
                        # Use integral image for fast calculation if available
                        if hasattr(self.image_processor, 'integral_images') and self.image_processor.integral_images is not None:
                            mean, std_dev = self.image_processor._calculate_stats_from_integral(x1, y1, x2, y2)
                        else:
                            # Fallback to traditional method if integral images not available
                            region = self.image_processor.current_image[y1:y2+1, x1:x2+1]
                            mean = np.mean(region)
                            std_dev = np.std(region)
                        
                        # Calculate uncertainty
                        uncertainty = std_dev / np.sqrt(pixel_count) if pixel_count > 0 else 0
                        
                        # Collect raw data for histogram (sample if too many pixels)
                        if len(x_coords) > 1000:
                            # Take a random sample of 1000 pixels
                            indices = np.random.choice(len(x_coords), 1000, replace=False)
                            sample_x = x_coords[indices]
                            sample_y = y_coords[indices]
                            sample_rel = rel_coords[indices]
                        else:
                            sample_x = x_coords
                            sample_y = y_coords
                            sample_rel = rel_coords
                        
                        # Collect raw data for histogram
                        masked_data = np.zeros(len(sample_x), dtype=np.uint8)
                        for i, (x, y) in enumerate(zip(sample_y, sample_x)):
                            masked_data[i] = self.image_processor.current_image[y, x]
                        
                        # Store the raw data and coordinates for later use
                        self.image_processor.last_measurement_raw_data = masked_data
                        self.image_processor.last_measurement_coordinates = sample_rel
                        self.image_processor.last_auto_measure_time = time.time()
                        
                        return mean, std_dev, uncertainty, pixel_count
            
            return None
        except Exception as e:
            logger.error(f"Error measuring area: {str(e)}", exc_info=True)
            return None
