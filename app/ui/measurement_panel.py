"""
Measurement panel UI for the Radiochromic Film Analyzer.

This module contains the measurement panel UI class that displays and manages
the measurement controls and results.
"""

import tkinter as tk
from tkinter import ttk
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)

class MeasurementPanel:
    """Measurement panel UI for the Radiochromic Film Analyzer."""
    
    def __init__(self, parent, image_processor):
        """Initialize the measurement panel."""
        self.parent = parent
        self.image_processor = image_processor
        
        # Create frame
        self.frame = ttk.Frame(parent)
        
        # Tooltip variables
        self.tooltip_window = None
        self.tooltip_timer = None
        
        # Measurement data
        self.current_measurement_data = None
        self.has_valid_measurement = False  # Flag to track if we have a valid measurement
        
        # Manual line selection state
        self.manual_line_points = []  # Will store [(x1, y1), (x2, y2)]
        self.selecting_manual_line = False
        
        # Create UI
        self._create_ui()
        
        logger.info("Measurement panel initialized")
    
    def _create_ui(self):
        """Create the UI components."""
        # Shape options
        ttk.Label(self.frame, text="Measurement shape:").pack(anchor=tk.W, padx=10, pady=5)
        
        # Shape radio buttons in HORIZONTAL layout
        self.shape_var = tk.StringVar(value="circular")
        shape_buttons_frame = ttk.Frame(self.frame)
        shape_buttons_frame.pack(anchor=tk.W, padx=20, pady=2)
        
        ttk.Radiobutton(
            shape_buttons_frame, 
            text="Circular", 
            variable=self.shape_var, 
            value="circular",
            command=self._on_shape_change
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(
            shape_buttons_frame, 
            text="Rectangular", 
            variable=self.shape_var, 
            value="rectangular",
            command=self._on_shape_change
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(
            shape_buttons_frame, 
            text="Line", 
            variable=self.shape_var, 
            value="line",
            command=self._on_shape_change
        ).pack(side=tk.LEFT, padx=5)
        
        # Size control section
        self.size_label = ttk.Label(self.frame, text="Size (pixels):")
        self.size_label.pack(anchor=tk.W, padx=10, pady=10)
        
        # Frame for size controls (will show/hide based on shape)
        self.size_frame = ttk.Frame(self.frame)
        self.size_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Circular/Square: Single size spinbox
        self.single_size_frame = ttk.Frame(self.size_frame)
        self.size_var = tk.IntVar(value=20)
        # Add trace to auto-update when value changes
        self.size_var.trace_add('write', lambda *args: self._on_size_change())
        
        self.size_spinbox = ttk.Spinbox(
            self.single_size_frame, 
            from_=5, 
            to=500, 
            textvariable=self.size_var, 
            width=10,
            command=self._on_size_change
        )
        self.size_spinbox.pack(side=tk.LEFT)
        
        # Bind events for auto-update and tooltip
        self.size_spinbox.bind("<Return>", lambda e: self._on_size_change())
        self.size_spinbox.bind("<FocusOut>", lambda e: self._on_size_change())
        self.size_spinbox.bind("<Enter>", self._show_tooltip)
        self.size_spinbox.bind("<Leave>", self._hide_tooltip)
        
        # Rectangular: Size X and Size Y in HORIZONTAL layout
        self.rect_size_frame = ttk.Frame(self.size_frame)
        
        ttk.Label(self.rect_size_frame, text="Width:").pack(side=tk.LEFT, padx=5)
        self.size_x_var = tk.IntVar(value=40)
        # Add trace to auto-update when value changes
        self.size_x_var.trace_add('write', lambda *args: self._on_size_change())
        
        self.size_x_spinbox = ttk.Spinbox(
            self.rect_size_frame,
            from_=5,
            to=500,
            textvariable=self.size_x_var,
            width=10,
            command=self._on_size_change
        )
        self.size_x_spinbox.pack(side=tk.LEFT, padx=5)
        # Bind events for auto-update
        self.size_x_spinbox.bind("<Return>", lambda e: self._on_size_change())
        self.size_x_spinbox.bind("<FocusOut>", lambda e: self._on_size_change())
        
        ttk.Label(self.rect_size_frame, text="Height:").pack(side=tk.LEFT, padx=10)
        self.size_y_var = tk.IntVar(value=40)
        # Add trace to auto-update when value changes
        self.size_y_var.trace_add('write', lambda *args: self._on_size_change())
        
        self.size_y_spinbox = ttk.Spinbox(
            self.rect_size_frame,
            from_=5,
            to=500,
            textvariable=self.size_y_var,
            width=10,
            command=self._on_size_change
        )
        self.size_y_spinbox.pack(side=tk.LEFT, padx=5)
        # Bind events for auto-update
        self.size_y_spinbox.bind("<Return>", lambda e: self._on_size_change())
        self.size_y_spinbox.bind("<FocusOut>", lambda e: self._on_size_change())
        
        # Line: Orientation selection in HORIZONTAL layout
        self.line_orientation_frame = ttk.Frame(self.size_frame)
        
        ttk.Label(self.line_orientation_frame, text="Orientation:").pack(side=tk.LEFT, padx=5)
        self.line_orientation_var = tk.StringVar(value="horizontal")
        
        ttk.Radiobutton(
            self.line_orientation_frame,
            text="Horizontal",
            variable=self.line_orientation_var,
            value="horizontal",
            command=self._on_line_orientation_change
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(
            self.line_orientation_frame,
            text="Vertical",
            variable=self.line_orientation_var,
            value="vertical",
            command=self._on_line_orientation_change
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(
            self.line_orientation_frame,
            text="Manual",
            variable=self.line_orientation_var,
            value="manual",
            command=self._on_line_orientation_change
        ).pack(side=tk.LEFT, padx=5)
        
        # Button to start manual line selection (hidden, kept for compatibility)
        self.select_line_button = ttk.Button(
            self.line_orientation_frame,
            text="Select Line Points",
            command=self._start_manual_line_selection,
            state=tk.DISABLED
        )
        # Don't pack it - it's auto-started now
        
        # Initially show single size frame (for circular)
        self.single_size_frame.pack(fill=tk.X)
        
        # Auto-measurement option
        ttk.Separator(self.frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)
        
        # Fix: Access the auto_measure attribute directly instead of calling get_auto_measure()
        self.auto_measure_var = tk.BooleanVar(value=self.image_processor.auto_measure)
        
        ttk.Checkbutton(
            self.frame,
            text="Auto measure (update results on mouse move)",
            variable=self.auto_measure_var,
            command=self._on_auto_measure_change
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        # Clear measurement button
        ttk.Button(
            self.frame,
            text="Clear Measurement",
            command=self._clear_measurement
        ).pack(anchor=tk.W, padx=10, pady=5)
        
        # Results section
        ttk.Separator(self.frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(self.frame, text="Results:").pack(anchor=tk.W, padx=10, pady=5)
        
        # Results frame
        self.results_frame = ttk.Frame(self.frame)
        self.results_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        
        # Results labels
        self.average_label = ttk.Label(self.results_frame, text="Average RGB: --")
        self.average_label.pack(anchor=tk.W, pady=2)
        
        self.std_dev_label = ttk.Label(self.results_frame, text="Standard deviation: --")
        self.std_dev_label.pack(anchor=tk.W, pady=2)
        
        self.std_err_label = ttk.Label(self.results_frame, text="STD err: --")
        self.std_err_label.pack(anchor=tk.W, pady=2)
        
        # Pixel count label
        self.pixel_count_label = ttk.Label(self.results_frame, text="Pixels: --")
        self.pixel_count_label.pack(anchor=tk.W, pady=2)
        
        # Visualization section (Histogram or Line Profile)
        ttk.Separator(self.frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)
        self.viz_title_label = ttk.Label(self.frame, text="Histogram:")
        self.viz_title_label.pack(anchor=tk.W, padx=10, pady=5)
        
        # Create a frame for the visualization (histogram or line profile)
        self.visualization_frame = ttk.Frame(self.frame)
        self.visualization_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create matplotlib figure for visualization
        self.viz_fig = Figure(figsize=(4, 2), dpi=100)
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, master=self.visualization_frame)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Visualization buttons frame
        self.viz_buttons_frame = ttk.Frame(self.frame)
        self.viz_buttons_frame.pack(fill=tk.X, pady=10)
        
        # 3D View button (only for non-line measurements)
        self.view_3d_button = ttk.Button(
            self.viz_buttons_frame,
            text="3D View",
            command=self._show_3d_view,
            state=tk.DISABLED  # Initially disabled
        )
        self.view_3d_button.pack(side=tk.LEFT, padx=(10, 5))
        
        # 2D View button (only for non-line measurements)
        self.view_2d_button = ttk.Button(
            self.viz_buttons_frame,
            text="2D View",
            command=self._show_2d_view,
            state=tk.DISABLED  # Initially disabled
        )
        self.view_2d_button.pack(side=tk.LEFT, padx=5)
        
        # Interactive Graph button (only for line measurements)
        self.interactive_graph_button = ttk.Button(
            self.viz_buttons_frame,
            text="Interactive Graph",
            command=self._show_interactive_graph,
            state=tk.DISABLED  # Initially disabled
        )
        # Don't pack yet - will be shown/hidden based on shape
        
        # Set initial values in image processor
        self._on_shape_change()
        self._on_size_change()
        self._on_auto_measure_change()
        
        # Register global event handlers for Ctrl+wheel
        self._register_global_handlers()
    
    def _show_tooltip(self, event):
        """Show tooltip when mouse enters the spinbox."""
        # Cancel any existing timer
        if self.tooltip_timer:
            self.parent.after_cancel(self.tooltip_timer)
            self.tooltip_timer = None
        
        # Create tooltip if it doesn't exist
        if not self.tooltip_window:
            x, y, _, _ = self.size_spinbox.bbox("insert")
            x += self.size_spinbox.winfo_rootx() + 25
            y += self.size_spinbox.winfo_rooty() + 20
            
            # Create a toplevel window
            self.tooltip_window = tk.Toplevel(self.size_spinbox)
            self.tooltip_window.wm_overrideredirect(True)
            self.tooltip_window.wm_geometry(f"+{x}+{y}")
            
            # Create tooltip content
            tooltip_frame = ttk.Frame(self.tooltip_window, relief="solid", borderwidth=1)
            tooltip_frame.pack(ipadx=5, ipady=5)
            
            ttk.Label(
                tooltip_frame, 
                text="Ctrl+Wheel to change",
                background="#ffffe0",
                relief="flat",
                borderwidth=0
            ).pack()
        
        # Set timer to hide tooltip after 3 seconds
        self.tooltip_timer = self.parent.after(3000, self._hide_tooltip)
    
    def _hide_tooltip(self, event=None):
        """Hide tooltip when mouse leaves the spinbox or after timeout."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
        
        if self.tooltip_timer:
            self.parent.after_cancel(self.tooltip_timer)
            self.tooltip_timer = None
    
    def _register_global_handlers(self):
        """Register global event handlers."""
        # Get the root window
        root = self.frame.winfo_toplevel()
        
        # Bind Ctrl+wheel events globally
        root.bind_all("<Control-MouseWheel>", self._on_ctrl_wheel)
        root.bind_all("<Control-Button-4>", lambda e: self._on_ctrl_wheel(e, 1))  # Linux scroll up
        root.bind_all("<Control-Button-5>", lambda e: self._on_ctrl_wheel(e, -1))  # Linux scroll down
    
    def _on_shape_change(self):
        """Handle shape change."""
        shape = self.shape_var.get()
        self.image_processor.set_measurement_shape(shape)
        
        # If changing away from line shape, cancel manual line selection if active
        if shape != "line" and self.selecting_manual_line:
            self._cancel_manual_line_selection()
        
        # Clear manual line markers when changing shape
        root = self.frame.winfo_toplevel()
        if hasattr(root, 'main_window') and hasattr(root.main_window, 'image_panel'):
            image_panel = root.main_window.image_panel
            image_panel.canvas.delete('manual_line_marker')
        
        # Show/hide size controls based on shape
        self.single_size_frame.pack_forget()
        self.rect_size_frame.pack_forget()
        self.line_orientation_frame.pack_forget()
        
        if shape == "circular":
            # Show single size control
            self.single_size_frame.pack(fill=tk.X)
            self.size_label.config(text="Size (pixels):")
            # Update visualization title
            self.viz_title_label.config(text="Histogram:")
            # Show 3D/2D buttons, hide Interactive Graph
            self.view_3d_button.pack(side=tk.LEFT, padx=5)
            self.view_2d_button.pack(side=tk.LEFT, padx=5)
            self.interactive_graph_button.pack_forget()
            # Enable 3D/2D buttons if we have a measurement
            if self.has_valid_measurement:
                self.view_3d_button.config(state=tk.NORMAL)
                self.view_2d_button.config(state=tk.NORMAL)
            
        elif shape == "rectangular":
            # Show width and height controls
            self.rect_size_frame.pack(fill=tk.X)
            self.size_label.config(text="Size:")
            # Update visualization title
            self.viz_title_label.config(text="Histogram:")
            # Show 3D/2D buttons, hide Interactive Graph
            self.view_3d_button.pack(side=tk.LEFT, padx=5)
            self.view_2d_button.pack(side=tk.LEFT, padx=5)
            self.interactive_graph_button.pack_forget()
            # Enable 3D/2D buttons if we have a measurement
            if self.has_valid_measurement:
                self.view_3d_button.config(state=tk.NORMAL)
                self.view_2d_button.config(state=tk.NORMAL)
            
        elif shape == "line":
            # Show orientation selection, no size control
            self.line_orientation_frame.pack(fill=tk.X)
            self.size_label.config(text="")
            # Update visualization title
            orientation = self.line_orientation_var.get()
            if orientation == "manual":
                axis_label = "Custom"
            else:
                axis_label = "X" if orientation == "horizontal" else "Y"
            self.viz_title_label.config(text=f"Line Profile (vs {axis_label}):")
            # Hide 3D/2D buttons, show Interactive Graph
            self.view_3d_button.pack_forget()
            self.view_2d_button.pack_forget()
            self.interactive_graph_button.pack(side=tk.LEFT, padx=5)
            # Enable Interactive Graph button if we have a measurement
            if self.has_valid_measurement:
                self.interactive_graph_button.config(state=tk.NORMAL)
            else:
                self.interactive_graph_button.config(state=tk.DISABLED)
        
        # Find the main window to update the measurement display
        root = self.frame.winfo_toplevel()
        if hasattr(root, 'main_window') and hasattr(root.main_window, 'image_panel'):
            # Force redraw of measurement shape at current position
            image_panel = root.main_window.image_panel
            if image_panel.measurement_visible:
                image_panel._draw_measurement_shape(image_panel.last_mouse_x, image_panel.last_mouse_y)
            
            # Redraw measured area if visible
            if image_panel.measured_area_visible:
                image_panel._draw_measured_area(image_panel.measured_x, image_panel.measured_y)
                
                # Update measurement results
                results = self.image_processor.measure_area(image_panel.measured_x, image_panel.measured_y)
                if results:
                    image_panel.last_measurement_results = results
                    self.update_results(results)
        
        logger.debug(f"Measurement shape changed to {shape}")
    
    def _on_line_orientation_change(self):
        """Handle line orientation change."""
        orientation = self.line_orientation_var.get()
        # Store orientation in image processor (we'll add this attribute)
        self.image_processor.line_orientation = orientation
        
        # Clear any previous manual line visualization
        root = self.frame.winfo_toplevel()
        if hasattr(root, 'main_window') and hasattr(root.main_window, 'image_panel'):
            image_panel = root.main_window.image_panel
            image_panel.canvas.delete('manual_line_marker')
        
        # Handle manual orientation
        if orientation == "manual":
            # Hide the Select Line Points button - auto-start instead
            self.select_line_button.pack_forget()
            # Auto-start manual line selection
            self._start_manual_line_selection()
        else:
            # Make sure button is hidden for non-manual modes
            self.select_line_button.pack_forget()
            # Restore normal canvas bindings if we were in manual mode
            if self.selecting_manual_line:
                self._cancel_manual_line_selection()
        
        # Update visualization title
        if orientation == "manual":
            axis_label = "Custom"
        else:
            axis_label = "X" if orientation == "horizontal" else "Y"
        self.viz_title_label.config(text=f"Line Profile (vs {axis_label}):")
        
        # Redraw if measurement is visible (only for non-manual orientations)
        if orientation != "manual":
            root = self.frame.winfo_toplevel()
            if hasattr(root, 'main_window') and hasattr(root.main_window, 'image_panel'):
                image_panel = root.main_window.image_panel
                if image_panel.measured_area_visible:
                    image_panel._draw_measured_area(image_panel.measured_x, image_panel.measured_y)
                    results = self.image_processor.measure_area(image_panel.measured_x, image_panel.measured_y)
                    if results:
                        image_panel.last_measurement_results = results
                        self.update_results(results)
        
        logger.debug(f"Line orientation changed to {orientation}")
    
    def _on_size_change(self):
        """Handle size change."""
        shape = self.shape_var.get()
        
        try:
            if shape == "circular":
                size = self.size_var.get()
                if size < 5 or size > 500:  # Validate range
                    return
                self.image_processor.set_measurement_size(size)
                logger.debug(f"Circular measurement size changed to {size}")
            elif shape == "rectangular":
                # For rectangular, we need to store both width and height
                width = self.size_x_var.get()
                height = self.size_y_var.get()
                if width < 5 or width > 500 or height < 5 or height > 500:  # Validate range
                    return
                # Store as tuple in image processor
                self.image_processor.measurement_size_rect = (width, height)
                # Also update the main size for compatibility
                self.image_processor.set_measurement_size(max(width, height))
                logger.debug(f"Rectangular measurement size changed to {width}x{height}")
            # Line shape doesn't use size
        except tk.TclError:
            # Value is invalid (e.g., empty string while typing)
            return
        
        # Find the main window to update the measurement display
        root = self.frame.winfo_toplevel()
        if hasattr(root, 'main_window') and hasattr(root.main_window, 'image_panel'):
            # Force redraw of measurement shape at current position
            image_panel = root.main_window.image_panel
            if image_panel.measurement_visible:
                image_panel._draw_measurement_shape(image_panel.last_mouse_x, image_panel.last_mouse_y)
            
            # Redraw measured area if visible and recalculate measurement
            if image_panel.measured_area_visible:
                image_panel._draw_measured_area(image_panel.measured_x, image_panel.measured_y)
                
                # Update measurement results
                results = self.image_processor.measure_area(image_panel.measured_x, image_panel.measured_y)
                if results:
                    image_panel.last_measurement_results = results
                    self.update_results(results)
    
    def _on_auto_measure_change(self):
        """Handle auto measure change."""
        auto_measure = self.auto_measure_var.get()
        # Fix: Use set_auto_measure method to update the auto_measure attribute
        self.image_processor.set_auto_measure(auto_measure)
        logger.debug(f"Auto measure changed to {auto_measure}")
    
    def _clear_measurement(self):
        """Clear the current measurement."""
        # Find the main window to clear the measurement
        root = self.frame.winfo_toplevel()
        if hasattr(root, 'main_window') and hasattr(root.main_window, 'image_panel'):
            image_panel = root.main_window.image_panel
            
            # Clear measured area
            image_panel.canvas.delete("measured_area")
            # Also clear instructions
            image_panel.canvas.delete("instructions")
            
            image_panel.measured_area_visible = False
            image_panel.last_measurement_results = None
            image_panel.auto_measure_results = None
            
            # Clear results
            self.update_results(None)
            
            # Clear histogram
            self._clear_histogram()
            
            # Disable view buttons
            self.view_3d_button.config(state=tk.DISABLED)
            self.view_2d_button.config(state=tk.DISABLED)
            
            # Clear measurement data
            self.current_measurement_data = None
            self.has_valid_measurement = False
        
        logger.debug("Measurement cleared")
    
    def _on_ctrl_wheel(self, event, delta=None):
        """Handle Ctrl+wheel for size change."""
        # Determine size change
        if delta is None:  # For Windows
            delta = event.delta / 120
        
        # Calculate new size
        new_size = self.size_var.get() + (5 if delta > 0 else -5)
        
        # Limit between 5 and 500
        if 5 <= new_size <= 500:
            self.size_var.set(new_size)
            self._on_size_change()  # This will now redraw the measurement area
        
        # Log the change
        logger.debug(f"Measurement size changed to {new_size} via Ctrl+wheel")
    
        # Prevent event from propagating
        return "break"
    
    def update_results(self, results):
        """Update the measurement results display."""
        if not results:
            self.average_label.config(text="Average RGB: --")
            self.std_dev_label.config(text="Standard deviation: --")
            self.std_err_label.config(text="STD err: --")
            self.pixel_count_label.config(text="Pixels: --")
            self.view_3d_button.config(state=tk.DISABLED)
            self.view_2d_button.config(state=tk.DISABLED)
            self.interactive_graph_button.config(state=tk.DISABLED)
            self._clear_histogram()
            self.current_measurement_data = None
            self.has_valid_measurement = False
            return
        
        # Store the measurement data for histogram and 3D view
        self.current_measurement_data = results
        self.has_valid_measurement = True
        
        # Unpack results (6-tuple format)
        average, std_dev, std_err, rgb_mean, rgb_mean_std, pixel_count = results
        
        
        # Update labels
        if isinstance(average, tuple) and len(average) == 3:
            # RGB values
            # Determine if dose calibration is active so we can tailor the label
            if self.image_processor.calibration_applied:
                self.average_label.config(
                    text=(
                        f"Average dose Gy (RGB): ({average[0]:.2f}, {average[1]:.2f}, {average[2]:.2f}) | "
                        f"Mean: {rgb_mean:.4f}"
                    )
                )
            else:
                # No calibration – behave as before
                self.average_label.config(
                    text=f"Average RGB: ({average[0]:.2f}, {average[1]:.2f}, {average[2]:.2f})"
                )
            # Get the uncertainty method name for display
            method = self.image_processor.config.get("uncertainty_estimation_method", "weighted_average")
            method_display = {
                "weighted_average": "Weighted",
                "birge_factor": "Birge", 
                "dersimonian_laird": "D-L"
            }.get(method, method)
            
            self.std_dev_label.config(
                text=f"Standard deviation: ({std_dev[0]:.2f}, {std_dev[1]:.2f}, {std_dev[2]:.2f}) | Combined ({method_display}): {rgb_mean_std:.4f}"
            )
            self.std_err_label.config(
                text=f"STD err: ({std_err[0]:.4f}, {std_err[1]:.4f}, {std_err[2]:.4f})"
            )
            
            # Enable view buttons for RGB data based on shape
            shape = self.shape_var.get()
            if shape == "line":
                self.interactive_graph_button.config(state=tk.NORMAL)
            else:
                self.view_3d_button.config(state=tk.NORMAL)
                self.view_2d_button.config(state=tk.NORMAL)
        else:
            # Grayscale value
            self.average_label.config(text=f"Average: {average:.2f}")
            self.std_dev_label.config(text=f"Standard deviation: {std_dev:.2f}")
            self.std_err_label.config(text=f"STD err: {std_err:.4f}")
            
            # Enable buttons based on shape
            shape = self.shape_var.get()
            if shape == "line":
                self.interactive_graph_button.config(state=tk.NORMAL)
            else:
                # Enable only 2D View button for grayscale data
                self.view_3d_button.config(state=tk.DISABLED)
                self.view_2d_button.config(state=tk.NORMAL)
        
        # Update pixel count
        self.pixel_count_label.config(text=f"Pixels: {pixel_count}")
        
        # Update histogram
        self._update_histogram()
        
        logger.debug("Measurement results updated")
    
    def update_auto_measure_results(self, results):
        """Update the measurement results display with auto-measure results."""
        if not results:
            return
        
        # Update the results display
        self.update_results(results)
        
        # Update the histogram if we have raw data
        if hasattr(self.image_processor, 'last_measurement_raw_data') and self.image_processor.last_measurement_raw_data is not None:
            self._update_histogram()
    
    def _update_histogram(self):
        """Update the visualization (histogram or line profile) with measurement data."""
        # Clear the previous plot
        self.viz_fig.clear()
        
        # Check if this is a line measurement
        shape = self.shape_var.get()
        is_line = (shape == "line")
        
        if is_line:
            self._update_line_profile()
        else:
            self._update_histogram_plot()
    
    def _update_line_profile(self):
        """Update line profile plot for line measurements."""
        # Get the raw pixel data from the image processor
        raw_data = self.image_processor.get_last_measurement_raw_data()
        coords = self.image_processor.last_measurement_coordinates
        
        if raw_data is None or len(raw_data) == 0:
            # No data to display
            self.viz_canvas.draw()
            return
        
        ax = self.viz_fig.add_subplot(111)
        orientation = self.line_orientation_var.get()
        
        # Calculate positions for x-axis
        if coords is not None and len(coords) > 0:
            if orientation == "manual":
                # For manual lines, calculate distance along the line
                distances = np.sqrt(
                    (coords[:, 1] - coords[0, 1])**2 + 
                    (coords[:, 0] - coords[0, 0])**2
                )
                positions = distances
                axis_label = "Distance (pixels)"
            elif orientation == "horizontal":
                # Horizontal line - use column (X) coordinates
                positions = coords[:, 1]
                axis_label = "X Position (pixels)"
            else:  # vertical
                # Vertical line - use row (Y) coordinates
                positions = coords[:, 0]
                axis_label = "Y Position (pixels)"
        else:
            # Fallback to simple indexing
            positions = np.arange(len(raw_data))
            axis_label = "Position (pixels)"
        
        # Check if calibration is applied
        if self.image_processor.calibration_applied:
            # Show single dose series
            if len(raw_data.shape) > 1 and raw_data.shape[1] == 3:
                # RGB - average the channels for dose
                dose_profile = np.mean(raw_data, axis=1)
            else:
                # Grayscale dose
                dose_profile = raw_data
            
            ax.plot(positions, dose_profile, 'k-', linewidth=1.5, label='Dose')
            ax.set_ylabel('Dose (Gy)')
            title = f'Dose Profile - {orientation.capitalize()} Line'
            ax.set_title(title)
            ax.legend()
            
        else:
            # No calibration - show RGB channels separately
            # Get max pixel value for Y-axis limit
            max_pixel_value = self.image_processor.get_max_pixel_value()
            
            if len(raw_data.shape) > 1 and raw_data.shape[1] == 3:
                # RGB data - plot 3 series
                colors = ['r', 'g', 'b']
                labels = ['Red', 'Green', 'Blue']
                for i in range(3):
                    ax.plot(positions, raw_data[:, i], color=colors[i], linewidth=1.5, alpha=0.7, label=labels[i])
                ax.set_ylabel('Pixel Value')
                title = f'RGB Profile - {orientation.capitalize()} Line'
                ax.set_title(title)
                ax.set_ylim(0, max_pixel_value)
                ax.legend()
            else:
                # Grayscale data
                ax.plot(positions, raw_data, 'k-', linewidth=1.5, label='Intensity')
                ax.set_ylabel('Pixel Value')
                title = f'Intensity Profile - {orientation.capitalize()} Line'
                ax.set_title(title)
                ax.set_ylim(0, max_pixel_value)
                ax.legend()
        
        ax.set_xlabel(axis_label)
        ax.grid(True, alpha=0.3)
        
        # Adjust layout and draw
        self.viz_fig.tight_layout()
        self.viz_canvas.draw()
    
    def _update_histogram_plot(self):
        """Update histogram plot for circular/rectangular measurements."""
        # Get the raw pixel data from the image processor
        raw_data = self.image_processor.get_last_measurement_raw_data()
        
        if raw_data is None or len(raw_data) == 0:
            # No data to display
            self.viz_canvas.draw()
            return
        
        # Get the max value based on actual image bit depth (supports any bit depth)
        max_pixel_value = self.image_processor.get_max_pixel_value()
        
        # Check if we have RGB or grayscale data
        if len(raw_data.shape) > 1 and raw_data.shape[1] == 3:
            if self.image_processor.calibration_applied:
                # Dose-calibrated RGB: plot histogram of the per-pixel average dose only
                averaged = np.mean(raw_data, axis=1)
                ax = self.viz_fig.add_subplot(111)

                cleaned = averaged[~np.isnan(averaged)]
                if cleaned.size == 0:
                    self.viz_canvas.draw()
                    return

                data_min = float(np.min(cleaned))
                data_max = float(np.max(cleaned))
                if np.isclose(data_min, data_max):
                    data_min -= 0.5
                    data_max += 0.5

                ax.hist(cleaned, bins=50, color='gray', range=(data_min, data_max))
                ax.set_title('Average Dose Histogram')
                ax.set_xlim(data_min, data_max)
                ax.set_xlabel('Dose')
                ax.set_ylabel('Frequency')
            else:
                # Standard RGB image – plot per-channel histograms
                ax = self.viz_fig.add_subplot(111)
                colors = ['r', 'g', 'b']
                labels = ['Red', 'Green', 'Blue']
                for i in range(3):
                    # Use bins=50 for a good balance between detail and performance
                    ax.hist(raw_data[:, i], bins=50, alpha=0.5, color=colors[i], label=labels[i], range=(0, max_pixel_value))
                ax.set_title('RGB Histogram')
                ax.legend(loc='upper right')
                ax.set_xlim(0, max_pixel_value)
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Frequency')
        else:
            # Single-channel data: could be standard grayscale or dose-calibrated values
            ax = self.viz_fig.add_subplot(111)

            if self.image_processor.calibration_applied:
                # Dose data – adjust axis to data range
                cleaned = raw_data[~np.isnan(raw_data)]
                if cleaned.size == 0:
                    self.viz_canvas.draw()
                    return
                data_min = float(np.min(cleaned))
                data_max = float(np.max(cleaned))
                # Avoid zero range
                if np.isclose(data_min, data_max):
                    data_min -= 0.5
                    data_max += 0.5
                ax.hist(cleaned, bins=50, color='gray', range=(data_min, data_max))
                ax.set_title('Dose Histogram')
                ax.set_xlim(data_min, data_max)
                ax.set_xlabel('Dose')
            else:
                # Standard grayscale data - use appropriate range for bit depth
                ax.hist(raw_data[~np.isnan(raw_data)], bins=50, color='gray', range=(0, max_pixel_value))
                ax.set_title('Grayscale Histogram')
                ax.set_xlim(0, max_pixel_value)
                ax.set_xlabel('Pixel Value')

            ax.set_ylabel('Frequency')

        # Adjust layout and draw
        self.viz_fig.tight_layout()
        self.viz_canvas.draw()
    
    def _clear_histogram(self):
        """Clear the visualization (histogram or line profile)."""
        self.viz_fig.clear()
        self.viz_canvas.draw()
    
    def _show_3d_view(self):
        """Show 3D view of RGB channels."""
        # Check if we have valid measurement data
        if not self.has_valid_measurement:
            return
        
        # Get the raw pixel data and coordinates from the image processor
        raw_data = self.image_processor.get_last_measurement_raw_data()
        coordinates = self.image_processor.get_last_measurement_coordinates()
        
        if (raw_data is None or len(raw_data) == 0 or raw_data.shape[1] != 3 or 
            coordinates is None or len(coordinates) == 0):
            # No RGB data or coordinates to display
            return
        
        # Import scipy for grid interpolation
        try:
            from scipy.interpolate import griddata
        except ImportError:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Error", "The scipy package is required for 3D visualization. Please install it with 'pip install scipy'.")
            return
        
        # Create a new window for 3D view
        view_window = tk.Toplevel(self.frame)
        view_window.title("3D RGB Channel View")
        view_window.geometry("900x700")
        
        # Prepare figure
        fig = Figure(figsize=(9, 7), dpi=100)

        # Coordinates (swap indices to correct orientation)
        x = coordinates[:, 1]
        y = coordinates[:, 0]

        # Regular grid for surface plot
        grid_size = 50
        xi = np.linspace(min(x), max(x), grid_size)
        yi = np.linspace(min(y), max(y), grid_size)
        X, Y = np.meshgrid(xi, yi)

        # Colormap
        root = self.frame.winfo_toplevel()
        default_cmap = "viridis"
        if hasattr(root, 'main_window'):
            default_cmap = root.main_window.app_config.get("colormap", "viridis")

        axes = []
        if self.image_processor.calibration_applied:
            # Single subplot for average dose
            titles = ['Average Dose']
            ax = fig.add_subplot(111, projection='3d')
            axes.append(ax)

            z_full = np.mean(raw_data, axis=1)
            Z = griddata((x, y), z_full, (X, Y), method='cubic', fill_value=0)
            surf = ax.plot_surface(X, Y, Z, cmap=default_cmap,
                                   linewidth=0, antialiased=True, alpha=1.0)
            ax.set_title('Average Dose')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_zlabel('Dose')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        else:
            # Three subplots for RGB channels
            titles = ['Red Channel', 'Green Channel', 'Blue Channel']
            for i in range(3):
                ax = fig.add_subplot(1, 3, i+1, projection='3d')
                axes.append(ax)
                
                # Get intensity values for this channel
                z = raw_data[:, i]
                
                # Interpolate scattered data to a grid
                Z = griddata((x, y), z, (X, Y), method='cubic', fill_value=0)
                
                # Create a surface plot using the selected colormap with full opacity (alpha=1.0)
                surf = ax.plot_surface(X, Y, Z, cmap=default_cmap, 
                              linewidth=0, antialiased=True, alpha=1.0)
                
                ax.set_title(titles[i])
                ax.set_xlabel('X (pixels)')
                ax.set_ylabel('Y (pixels)')
                ax.set_zlabel('Intensity')
                
                # Add a color bar
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
    
        # Adjust layout
        fig.tight_layout()
    
        # Create canvas and add to window
        canvas = FigureCanvasTkAgg(fig, master=view_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
        # Add status bar for cursor position and value
        status_frame = ttk.Frame(view_window)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, before=canvas.get_tk_widget())
    
        status_label = ttk.Label(status_frame, text="Position: -- Value: --")
        status_label.pack(side=tk.LEFT, padx=10, pady=5)
    
        # Function to update cursor info
        def motion_notify_callback(event):
            if event.inaxes is not None:
                ax_index = axes.index(event.inaxes) if event.inaxes in axes else -1
                if ax_index >= 0:
                    # Get the x, y coordinates in data space
                    x_pos, y_pos = event.xdata, event.ydata
                
                    # Find the closest grid point
                    x_idx = np.abs(xi - x_pos).argmin()
                    y_idx = np.abs(yi - y_pos).argmin()
                
                    # Get the z value at that point
                    z_val = griddata((x, y), raw_data[:, ax_index], (x_pos, y_pos), method='cubic')
                
                    # Update status bar - display coordinates in correct order
                    status_label.config(text=f"Channel: {titles[ax_index]} | Position: ({y_pos:.1f}, {x_pos:.1f}) | Value: {float(z_val):.1f}")
    
        # Connect the motion_notify_event
        canvas.mpl_connect('motion_notify_event', motion_notify_callback)
    
        # Add a close button
        ttk.Button(
            view_window,
            text="Close",
            command=view_window.destroy
        ).pack(pady=10)
    
    def _show_2d_view(self):
        """Show 2D heat map view of measurement data."""
        # Check if we have valid measurement data
        if not self.has_valid_measurement:
            return
        
        # Get the raw pixel data and coordinates from the image processor
        raw_data = self.image_processor.get_last_measurement_raw_data()
        coordinates = self.image_processor.get_last_measurement_coordinates()
        
        if (raw_data is None or len(raw_data) == 0 or 
            coordinates is None or len(coordinates) == 0):
            # No data or coordinates to display
            return
        
        # Import scipy for grid interpolation
        try:
            from scipy.interpolate import griddata
            from scipy.ndimage import gaussian_filter
        except ImportError:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Error", "The scipy package is required for 2D visualization. Please install it with 'pip install scipy'.")
            return
        
        # Create a new window for 2D view
        view_window = tk.Toplevel(self.frame)
        view_window.title("2D Heat Map View")
        view_window.geometry("800x600")
        
        # Create a figure
        fig = Figure(figsize=(8, 6), dpi=100)
        
        # Get the scattered data points
        # Get the scattered data points - swap indices to correct coordinate orientation
        x = coordinates[:, 1]  # X coordinates relative to center
        y = coordinates[:, 0]  # Y coordinates relative to center
        
        # Create a regular grid for the heat map with higher resolution for better circular representation
        grid_size = 300  # Increased from 200 for even higher resolution
        xi = np.linspace(min(x), max(x), grid_size)
        yi = np.linspace(min(y), max(y), grid_size)
        X, Y = np.meshgrid(xi, yi)
        
        # Get the selected colormap from config
        root = self.frame.winfo_toplevel()
        default_cmap = "viridis"
        if hasattr(root, 'main_window'):
            default_cmap = root.main_window.app_config.get("colormap", "viridis")
        
        # Store all axes and their corresponding data for cursor tracking
        axes = []
        grid_data = []
        titles = []
        
        # Determine visualization mode based on calibration status
        if len(raw_data.shape) > 1 and raw_data.shape[1] == 3 and not self.image_processor.calibration_applied:
            # RGB data (no dose calibration) - create 3 subplots
            channel_titles = ['Red Channel', 'Green Channel', 'Blue Channel']
            
            for i in range(3):
                ax = fig.add_subplot(1, 3, i+1)
                axes.append(ax)
                titles.append(channel_titles[i])
                
                # Get intensity values for this channel
                z = raw_data[:, i]
                
                # Improved interpolation for better circular representation
                # Using 'cubic' method with higher resolution grid
                Z = griddata((x, y), z, (X, Y), method='cubic', fill_value=0)
                
                # Apply a circular mask to ensure perfect circular shape
                # Create a mask for circular area
                center_x, center_y = 0, 0  # Assuming coordinates are centered at origin
                radius = max(abs(min(x)), abs(max(x)), abs(min(y)), abs(max(y)))
                
                # Create distance matrix from center
                xx, yy = np.meshgrid(xi, yi)
                distances = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
                
                # Create circular mask
                mask = distances <= radius
                
                # Apply mask
                Z_masked = Z.copy()
                Z_masked[~mask] = np.nan
                
                # Apply slight Gaussian smoothing to edges
                Z_smooth = gaussian_filter(Z_masked, sigma=1.0)
                
                grid_data.append(Z_smooth)
                
                # Create a heat map using the selected colormap
                im = ax.pcolormesh(X, Y, Z_smooth, cmap=default_cmap, shading='auto')
                ax.set_title(channel_titles[i])
                ax.set_xlabel('X (pixels)')
                ax.set_ylabel('Y (pixels)')
                
                # Set aspect ratio to be equal (square) for proper circular display
                ax.set_aspect('equal')
                
                # Add a color bar
                fig.colorbar(im, ax=ax, shrink=0.8)
        else:
            # Single channel visualization (grayscale or average dose)
            ax = fig.add_subplot(111)
            axes.append(ax)
            if self.image_processor.calibration_applied:
                titles.append('Average Dose Heat Map')
                # Compute average dose across channels if needed
                if len(raw_data.shape) > 1 and raw_data.shape[1] == 3:
                    raw_single = np.mean(raw_data, axis=1)
                else:
                    raw_single = raw_data
            else:
                titles.append('Intensity Heat Map')
                raw_single = raw_data
            
            # Improved interpolation for better circular representation
            Z = griddata((x, y), raw_single, (X, Y), method='cubic', fill_value=0)
            
            # Apply a circular mask to ensure perfect circular shape
            # Create a mask for circular area
            center_x, center_y = 0, 0  # Assuming coordinates are centered at origin
            radius = max(abs(min(x)), abs(max(x)), abs(min(y)), abs(max(y)))
            
            # Create distance matrix from center
            xx, yy = np.meshgrid(xi, yi)
            distances = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
            
            # Create circular mask
            mask = distances <= radius
            
            # Apply mask
            Z_masked = Z.copy()
            Z_masked[~mask] = np.nan
            
            # Apply slight Gaussian smoothing to edges
            Z_smooth = gaussian_filter(Z_masked, sigma=1.0)
            
            grid_data.append(Z_smooth)
            
            # Create a heat map using the selected colormap with improved shading
            im = ax.pcolormesh(X, Y, Z_smooth, cmap=default_cmap, shading='auto')
            ax.set_title('Intensity Heat Map')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            
            # Set aspect ratio to be equal (square) for proper circular display
            ax.set_aspect('equal')
            
            # Add a color bar
            fig.colorbar(im, ax=ax)
        
        # Adjust layout
        fig.tight_layout()
        
        # Create canvas and add to window
        canvas = FigureCanvasTkAgg(fig, master=view_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add status bar for cursor position and value
        status_frame = ttk.Frame(view_window)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, before=canvas.get_tk_widget())
        
        status_label = ttk.Label(status_frame, text="Position: -- Value: --")
        status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Function to update cursor info
        def motion_notify_callback(event):
            if event.inaxes is not None:
                ax_index = axes.index(event.inaxes) if event.inaxes in axes else -1
                if ax_index >= 0:
                    # Get the x, y coordinates in data space
                    x_pos, y_pos = event.xdata, event.ydata
                    
                    # Find the closest grid point
                    x_idx = np.abs(xi - x_pos).argmin()
                    y_idx = np.abs(yi - y_pos).argmin()
                    
                    # Get the z value at that point (if within bounds)
                    if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                        z_val = grid_data[ax_index][y_idx, x_idx]
                        if not np.isnan(z_val):  # Only update if value is not NaN (inside the circle)
                            # Update status bar - display coordinates in correct order
                            status_label.config(text=f"{titles[ax_index]} | Position: ({y_pos:.1f}, {x_pos:.1f}) | Value: {z_val:.1f}")
        
        # Connect the motion_notify_event
        canvas.mpl_connect('motion_notify_event', motion_notify_callback)
        
        # Add a close button
        ttk.Button(
            view_window,
            text="Close",
            command=view_window.destroy
        ).pack(pady=10)
    
    def _start_manual_line_selection(self):
        """Start manual line selection mode - user clicks 2 points on canvas."""
        # Clear previous points
        self.manual_line_points = []
        self.selecting_manual_line = True
        
        # Get the image panel from main window
        root = self.frame.winfo_toplevel()
        if not hasattr(root, 'main_window'):
            return
        
        image_panel = root.main_window.image_panel
        
        # Clear any previous markers
        image_panel.canvas.delete('manual_line_marker')
        
        # Bind click event to image panel canvas
        # Save original binding if exists
        if not hasattr(self, '_original_canvas_click_binding'):
            original_binding = image_panel.canvas.bind("<Button-1>")
            self._original_canvas_click_binding = original_binding
        
        # Set our custom click handler
        image_panel.canvas.bind("<Button-1>", self._on_manual_line_click)
    
    def _on_manual_line_click(self, event):
        """Handle click events during manual line selection."""
        # Get the image panel
        root = self.frame.winfo_toplevel()
        if not hasattr(root, 'main_window'):
            return
        
        image_panel = root.main_window.image_panel
        
        # Convert canvas coordinates to image coordinates
        canvas_x = image_panel.canvas.canvasx(event.x)
        canvas_y = image_panel.canvas.canvasy(event.y)
        
        # Get image coordinates using image_processor
        img_x, img_y, _ = self.image_processor.get_pixel_info(canvas_x, canvas_y)
        
        if img_x is None or img_y is None:
            return  # Click outside image
        
        # Add point
        self.manual_line_points.append((img_x, img_y))
        
        # Draw X marker on canvas - convert image coords back to canvas coords
        zoom = self.image_processor.get_zoom()
        marker_canvas_x = img_x * zoom
        marker_canvas_y = img_y * zoom
        
        marker_size = 6
        # Draw X with two diagonal lines
        image_panel.canvas.create_line(
            marker_canvas_x - marker_size, marker_canvas_y - marker_size,
            marker_canvas_x + marker_size, marker_canvas_y + marker_size,
            fill='green', width=2, tags='manual_line_marker'
        )
        image_panel.canvas.create_line(
            marker_canvas_x - marker_size, marker_canvas_y + marker_size,
            marker_canvas_x + marker_size, marker_canvas_y - marker_size,
            fill='green', width=2, tags='manual_line_marker'
        )
        
        # If we have 2 points, complete the selection
        if len(self.manual_line_points) >= 2:
            # Draw line between points
            x1, y1 = self.manual_line_points[0]
            x2, y2 = self.manual_line_points[1]
            
            # Convert to canvas coordinates
            canvas_x1 = x1 * zoom
            canvas_y1 = y1 * zoom
            canvas_x2 = x2 * zoom
            canvas_y2 = y2 * zoom
            
            image_panel.canvas.create_line(
                canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                fill='green', width=2, tags='manual_line_marker'
            )
            
            # Complete selection
            self._finish_manual_line_selection()
    
    def _finish_manual_line_selection(self):
        """Complete manual line selection and perform measurement."""
        # Get the image panel
        root = self.frame.winfo_toplevel()
        if not hasattr(root, 'main_window'):
            return
        
        image_panel = root.main_window.image_panel
        
        # Update image processor with manual line points
        if len(self.manual_line_points) >= 2:
            self.image_processor.manual_line_points = self.manual_line_points[:2]
            
            # Remove temporary markers (will be redrawn as measured_area)
            image_panel.canvas.delete('manual_line_marker')
            
            # Calculate midpoint in image coordinates
            x1, y1 = self.manual_line_points[0]
            x2, y2 = self.manual_line_points[1]
            mid_img_x = (x1 + x2) // 2
            mid_img_y = (y1 + y2) // 2
            
            # Convert to canvas coordinates for measurement
            zoom = self.image_processor.get_zoom()
            mid_canvas_x = mid_img_x * zoom
            mid_canvas_y = mid_img_y * zoom
            
            # Store measured position
            image_panel.measured_img_x = mid_img_x
            image_panel.measured_img_y = mid_img_y
            image_panel.measured_x = mid_canvas_x
            image_panel.measured_y = mid_canvas_y
            image_panel.measured_area_visible = True
            
            # Update position label
            image_panel.position_label.config(text=f"Pos: {mid_img_x},{mid_img_y}")
            
            # Perform measurement at midpoint (using canvas coordinates)
            results = self.image_processor.measure_area(mid_canvas_x, mid_canvas_y)
            
            if results:
                image_panel.last_measurement_results = results
                self.update_results(results)
            
            # Draw measured area (this will draw the green line)
            image_panel._draw_measured_area(mid_canvas_x, mid_canvas_y)
        
        # Check if we're still in manual mode
        if self.line_orientation_var.get() == "manual":
            # Clear points for next selection but keep selecting mode active
            self.manual_line_points = []
            self.selecting_manual_line = True
            # Don't restore bindings - keep custom click handler active for next line
        else:
            # Not in manual mode anymore, restore original bindings
            self._restore_original_bindings()
            self.manual_line_points = []
            self.selecting_manual_line = False
    
    def _restore_original_bindings(self):
        """Restore original canvas click bindings."""
        root = self.frame.winfo_toplevel()
        if not hasattr(root, 'main_window'):
            return
        
        image_panel = root.main_window.image_panel
        
        # Restore original click binding
        if hasattr(self, '_original_canvas_click_binding'):
            if self._original_canvas_click_binding:
                image_panel.canvas.bind("<Button-1>", self._original_canvas_click_binding)
            else:
                image_panel.canvas.unbind("<Button-1>")
            delattr(self, '_original_canvas_click_binding')
    
    def _cancel_manual_line_selection(self):
        """Cancel manual line selection without performing measurement."""
        # Restore original bindings
        self._restore_original_bindings()
        
        # Clear state
        self.selecting_manual_line = False
        self.manual_line_points = []
        
        # Clear visual markers
        root = self.frame.winfo_toplevel()
        if hasattr(root, 'main_window') and hasattr(root.main_window, 'image_panel'):
            image_panel = root.main_window.image_panel
            image_panel.canvas.delete('manual_line_marker')
    
    def _show_interactive_graph(self):
        """Open an interactive matplotlib window for line profile visualization."""
        if not self.has_valid_measurement or self.current_measurement_data is None:
            return
        
        # Get the line profile data from the image processor
        raw_data = self.image_processor.get_last_measurement_raw_data()
        coordinates = self.image_processor.get_last_measurement_coordinates()
        
        if raw_data is None or coordinates is None or len(raw_data) == 0:
            return
        
        # Import matplotlib.pyplot for interactive window
        import matplotlib.pyplot as plt
        
        # Create new figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate position along line
        if len(coordinates) > 0:
            distances = np.sqrt(
                (coordinates[:, 1] - coordinates[0, 1])**2 + 
                (coordinates[:, 0] - coordinates[0, 0])**2
            )
        else:
            distances = np.arange(len(raw_data))
        
        # Plot based on calibration state
        if self.image_processor.calibration_applied:
            # Plot dose values
            dose_values = np.mean(raw_data, axis=1)
            ax.plot(distances, dose_values, 'b-', linewidth=2, label='Dose')
            ax.set_ylabel('Dose (Gy)', fontsize=12)
            ax.set_title('Line Profile - Dose', fontsize=14, fontweight='bold')
        else:
            # Plot RGB channels
            ax.plot(distances, raw_data[:, 0], 'r-', linewidth=2, label='Red', alpha=0.7)
            ax.plot(distances, raw_data[:, 1], 'g-', linewidth=2, label='Green', alpha=0.7)
            ax.plot(distances, raw_data[:, 2], 'b-', linewidth=2, label='Blue', alpha=0.7)
            ax.set_ylabel('Pixel Value', fontsize=12)
            ax.set_title('Line Profile - RGB Channels', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Position (pixels)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Enable toolbar for zoom, pan, save
        plt.tight_layout()
        plt.show()
    
    def is_auto_measure_enabled(self):
        """Check if auto-measurement is enabled."""
        return self.auto_measure_var.get()
