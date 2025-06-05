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
        
        # Create UI
        self._create_ui()
        
        logger.info("Measurement panel initialized")
    
    def _create_ui(self):
        """Create the UI components."""
        # Shape options
        ttk.Label(self.frame, text="Measurement shape:").pack(anchor=tk.W, padx=10, pady=5)
        
        # Shape radio buttons
        self.shape_var = tk.StringVar(value="circular")
        
        ttk.Radiobutton(
            self.frame, 
            text="Circular", 
            variable=self.shape_var, 
            value="circular",
            command=self._on_shape_change
        ).pack(anchor=tk.W, padx=20, pady=2)
        
        ttk.Radiobutton(
            self.frame, 
            text="Square", 
            variable=self.shape_var, 
            value="square",
            command=self._on_shape_change
        ).pack(anchor=tk.W, padx=20, pady=2)
        
        # Size control
        ttk.Label(self.frame, text="Size (pixels):").pack(anchor=tk.W, padx=10, pady=10)
        
        # Frame for size control
        size_frame = ttk.Frame(self.frame)
        size_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Size spinbox
        self.size_var = tk.IntVar(value=20)
        
        self.size_spinbox = ttk.Spinbox(
            size_frame, 
            from_=5, 
            to=500, 
            textvariable=self.size_var, 
            width=10,
            command=self._on_size_change
        )
        self.size_spinbox.pack(side=tk.LEFT)
        
        # Bind events for tooltip
        self.size_spinbox.bind("<Enter>", self._show_tooltip)
        self.size_spinbox.bind("<Leave>", self._hide_tooltip)
        
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
        
        self.uncertainty_label = ttk.Label(self.results_frame, text="Uncertainty: --")
        self.uncertainty_label.pack(anchor=tk.W, pady=2)
        
        # Pixel count label
        self.pixel_count_label = ttk.Label(self.results_frame, text="Pixels: --")
        self.pixel_count_label.pack(anchor=tk.W, pady=2)
        
        # Histogram section
        ttk.Separator(self.frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(self.frame, text="Histogram:").pack(anchor=tk.W, padx=10, pady=5)
        
        # Create a frame for the histogram
        self.histogram_frame = ttk.Frame(self.frame)
        self.histogram_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create matplotlib figure for histogram
        self.histogram_fig = Figure(figsize=(4, 2), dpi=100)
        self.histogram_canvas = FigureCanvasTkAgg(self.histogram_fig, master=self.histogram_frame)
        self.histogram_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Visualization buttons frame
        viz_buttons_frame = ttk.Frame(self.frame)
        viz_buttons_frame.pack(fill=tk.X, pady=10)
        
        # 3D View button
        self.view_3d_button = ttk.Button(
            viz_buttons_frame,
            text="3D View",
            command=self._show_3d_view,
            state=tk.DISABLED  # Initially disabled
        )
        self.view_3d_button.pack(side=tk.LEFT, padx=(10, 5))
        
        # 2D View button
        self.view_2d_button = ttk.Button(
            viz_buttons_frame,
            text="2D View",
            command=self._show_2d_view,
            state=tk.DISABLED  # Initially disabled
        )
        self.view_2d_button.pack(side=tk.LEFT, padx=5)
        
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
    
    def _on_size_change(self):
        """Handle size change."""
        size = self.size_var.get()
        self.image_processor.set_measurement_size(size)
        
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
        
        logger.debug(f"Measurement size changed to {size}")
    
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
            self.uncertainty_label.config(text="Uncertainty: --")
            self.pixel_count_label.config(text="Pixels: --")
            self.view_3d_button.config(state=tk.DISABLED)
            self.view_2d_button.config(state=tk.DISABLED)
            self._clear_histogram()
            self.current_measurement_data = None
            self.has_valid_measurement = False
            return
        
        # Store the measurement data for histogram and 3D view
        self.current_measurement_data = results
        self.has_valid_measurement = True
        
        # Unpack results (now includes pixel count)
        if len(results) == 4:  # New format with pixel count
            average, std_dev, uncertainty, pixel_count = results
        else:  # Old format without pixel count (for compatibility)
            average, std_dev, uncertainty = results
            pixel_count = 0
        
        # Update labels
        if isinstance(average, tuple) and len(average) == 3:
            # RGB values
            self.average_label.config(
                text=f"Average RGB: ({average[0]:.2f}, {average[1]:.2f}, {average[2]:.2f})"
            )
            self.std_dev_label.config(
                text=f"Standard deviation: ({std_dev[0]:.2f}, {std_dev[1]:.2f}, {std_dev[2]:.2f})"
            )
            self.uncertainty_label.config(
                text=f"Uncertainty: ({uncertainty[0]:.4f}, {uncertainty[1]:.4f}, {uncertainty[2]:.4f})"
            )
            
            # Enable view buttons for RGB data
            self.view_3d_button.config(state=tk.NORMAL)
            self.view_2d_button.config(state=tk.NORMAL)
        else:
            # Grayscale value
            self.average_label.config(text=f"Average: {average:.2f}")
            self.std_dev_label.config(text=f"Standard deviation: {std_dev:.2f}")
            self.uncertainty_label.config(text=f"Uncertainty: {uncertainty:.4f}")
            
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
        """Update the histogram with measurement data."""
        # Clear the previous histogram
        self.histogram_fig.clear()
        
        # Get the raw pixel data from the image processor
        raw_data = self.image_processor.get_last_measurement_raw_data()
        
        if raw_data is None or len(raw_data) == 0:
            # No data to display
            self.histogram_canvas.draw()
            return
        
        # Check if we have RGB or grayscale data
        if len(raw_data.shape) > 1 and raw_data.shape[1] == 3:
            # RGB data
            ax = self.histogram_fig.add_subplot(111)
        
            # Plot histograms for each channel
            colors = ['r', 'g', 'b']
            labels = ['Red', 'Green', 'Blue']
        
            for i in range(3):
                # Use bins=50 for a good balance between detail and performance
                ax.hist(raw_data[:, i], bins=50, alpha=0.5, color=colors[i], label=labels[i], range=(0, 255))
    
            ax.set_title('RGB Histogram')
            ax.legend(loc='upper right')
    
            # Set x-axis limits to 0-255 (8-bit RGB)
            ax.set_xlim(0, 255)
        else:
            # Grayscale data
            ax = self.histogram_fig.add_subplot(111)
            # Use bins=50 for a good balance between detail and performance
            ax.hist(raw_data, bins=50, color='gray', range=(0, 255))
            ax.set_title('Grayscale Histogram')
    
            # Set x-axis limits to 0-255 (8-bit grayscale)
            ax.set_xlim(0, 255)

        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')

        # Adjust layout and draw
        self.histogram_fig.tight_layout()
        self.histogram_canvas.draw()
    
    def _clear_histogram(self):
        """Clear the histogram."""
        self.histogram_fig.clear()
        self.histogram_canvas.draw()
    
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
        
        # Create a figure with 3 subplots (one for each channel)
        fig = Figure(figsize=(9, 7), dpi=100)
        
        # Get the scattered data points
        # Get the scattered data points - swap indices to correct coordinate orientation
        x = coordinates[:, 1]  # X coordinates relative to center
        y = coordinates[:, 0]  # Y coordinates relative to center
        
        # Create a regular grid for the surface plot
        grid_size = 50
        xi = np.linspace(min(x), max(x), grid_size)
        yi = np.linspace(min(y), max(y), grid_size)
        X, Y = np.meshgrid(xi, yi)
        
        # Get the selected colormap from config
        root = self.frame.winfo_toplevel()
        default_cmap = "viridis"
        if hasattr(root, 'main_window'):
            default_cmap = root.main_window.app_config.get("colormap", "viridis")
        
        # Create 3D plots for each channel
        titles = ['Red Channel', 'Green Channel', 'Blue Channel']
        axes = []
        
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
        
        # Check if we have RGB or grayscale data
        if len(raw_data.shape) > 1 and raw_data.shape[1] == 3:
            # RGB data - create 3 subplots
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
                im = ax.pcolormesh(X, Y, Z_smooth, cmap=default_cmap, shading='gouraud')
                ax.set_title(channel_titles[i])
                ax.set_xlabel('X (pixels)')
                ax.set_ylabel('Y (pixels)')
                
                # Set aspect ratio to be equal (square) for proper circular display
                ax.set_aspect('equal')
                
                # Add a color bar
                fig.colorbar(im, ax=ax, shrink=0.8)
        else:
            # Grayscale data - create a single plot
            ax = fig.add_subplot(111)
            axes.append(ax)
            titles.append('Intensity Heat Map')
            
            # Improved interpolation for better circular representation
            Z = griddata((x, y), raw_data, (X, Y), method='cubic', fill_value=0)
            
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
            im = ax.pcolormesh(X, Y, Z_smooth, cmap=default_cmap, shading='gouraud')
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
    
    def is_auto_measure_enabled(self):
        """Check if auto-measurement is enabled."""
        return self.auto_measure_var.get()
