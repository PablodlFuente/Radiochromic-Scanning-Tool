# Plugin Development Guide - Radiochromic Scanning Tool

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Plugin Structure](#basic-plugin-structure)
3. [Required Interface](#required-interface)
4. [Step-by-Step Examples](#step-by-step-examples)
5. [Accessing Resources](#accessing-resources)
6. [Advanced Plugins (Packages)](#advanced-plugins-packages)
7. [Best Practices](#best-practices)
8. [Debugging and Testing](#debugging-and-testing)
9. [API Reference](#api-reference)

---

## Introduction

Plugins allow extending the Radiochromic Scanning Tool functionality without modifying the base code. Each plugin appears as an independent tab in the main interface.

### What can a plugin do?

- **Process images**: Apply filters, transformations, analysis
- **Add custom UI**: Buttons, sliders, charts, tables
- **Perform measurements**: ROI analysis, automatic detection
- **Export data**: CSV, JSON, processed images
- **Interact with canvas**: Draw overlays, detect clicks

---

## Basic Plugin Structure

### Simple Plugin (Single File)

Create a `.py` file in the `custom_plugins/` folder:

```python
"""
my_plugin.py - Brief plugin description
"""

import tkinter as tk
from tkinter import ttk
import numpy as np

# Title that will appear in the tab
TAB_TITLE = "My Plugin"

def setup(main_window, notebook, image_processor):
    """
    Setup the plugin UI.
    
    Args:
        main_window: Main application window
        notebook: ttk.Notebook where the tab will be added
        image_processor: Image processor (ImageProcessor)
    
    Returns:
        ttk.Frame: Frame with the plugin UI
    """
    # Create main frame
    frame = ttk.Frame(notebook)
    
    # Add widgets
    ttk.Label(frame, text="Hello from my plugin!").pack(pady=10)
    ttk.Button(frame, text="Process", command=lambda: print("Processing...")).pack()
    
    return frame

def process(image: np.ndarray) -> np.ndarray:
    """
    Process the image displayed on the canvas.
    
    This function is called every time the image is updated.
    
    Args:
        image: Image as NumPy array (H, W, 3) in RGB format
    
    Returns:
        np.ndarray: Processed image (same format)
    """
    # Example: return image unchanged
    return image

# OPTIONAL: Function to react to configuration changes
def on_config_change(config, image_processor):
    """
    Called when application configuration changes.
    
    Args:
        config: Dictionary with current configuration
        image_processor: Image processor
    """
    print(f"Configuration updated: {config}")
```

### Advanced Plugin (Package)

For complex plugins, create a package structure:

```
custom_plugins/
└── my_advanced_plugin/
    ├── __init__.py          # Entry point (defines TAB_TITLE, setup, process)
    ├── ui/
    │   ├── __init__.py
    │   └── main_tab.py      # Main UI
    ├── core/
    │   ├── __init__.py
    │   ├── processor.py     # Processing logic
    │   └── detector.py      # Detection algorithms
    └── models/
        ├── __init__.py
        └── data_models.py   # Data structures
```

**`__init__.py`** (entry point):
```python
"""My Advanced Plugin - Complex image analysis."""

from .ui.main_tab import setup
from .core.processor import process

TAB_TITLE = "Advanced Plugin"

# setup and process are imported from corresponding modules
```

---

## Required Interface

### Mandatory

#### 1. Variable `TAB_TITLE`
```python
TAB_TITLE = "Plugin Name"  # String, max 20 characters recommended
```

#### 2. Function `setup(main_window, notebook, image_processor)`
```python
def setup(main_window, notebook, image_processor):
    """Setup and return the UI frame."""
    frame = ttk.Frame(notebook)
    # ... configure UI ...
    return frame
```

**Parameters:**
- `main_window`: Instance of `MainWindow`, provides access to:
  - `main_window.image_panel.canvas`: Canvas for drawing overlays
  - `main_window.image_panel`: Image panel
  - `main_window.update_image()`: Force image update
- `notebook`: `ttk.Notebook` where the tab is added
- `image_processor`: Instance of `ImageProcessor`, provides:
  - `image_processor.current_image`: Loaded image (NumPy array)
  - `image_processor.measure_area(x, y)`: Measure region
  - `image_processor.zoom`: Current zoom factor
  - `image_processor.has_image()`: Check if image is loaded

#### 3. Function `process(image)`
```python
def process(image: np.ndarray) -> np.ndarray:
    """Process the image for canvas display."""
    # Called on each image update
    # Do NOT do intensive logging here!
    return image  # Or modified image
```

### Optional

#### Function `on_config_change(config, image_processor)`
```python
def on_config_change(config, image_processor):
    """React to configuration changes."""
    # Example: update parameters according to config
    use_gpu = config.get("use_gpu", False)
```

---

## Step-by-Step Examples

### Example 1: Color Inversion Plugin

```python
"""color_inverter.py - Inverts image colors."""

import tkinter as tk
from tkinter import ttk
import numpy as np

TAB_TITLE = "Inverter"

# Global variable to control if active
_enabled = False

def setup(main_window, notebook, image_processor):
    """Setup the UI."""
    frame = ttk.Frame(notebook)
    
    def toggle_inversion():
        global _enabled
        _enabled = not _enabled
        btn.config(text="Disable" if _enabled else "Enable")
        main_window.update_image()  # Force update
    
    ttk.Label(frame, text="Inverts image colors").pack(pady=10)
    btn = ttk.Button(frame, text="Enable", command=toggle_inversion)
    btn.pack(pady=5)
    
    return frame

def process(image: np.ndarray) -> np.ndarray:
    """Invert colors if enabled."""
    if _enabled:
        return 255 - image  # Invert RGB values
    return image
```

### Example 2: Gaussian Blur Plugin

```python
"""gaussian_blur.py - Applies gaussian blur."""

import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2

TAB_TITLE = "Blur"

_kernel_size = 5
_enabled = False

def setup(main_window, notebook, image_processor):
    """Setup UI with slider for kernel size."""
    frame = ttk.Frame(notebook)
    
    # Checkbox to enable/disable
    enabled_var = tk.BooleanVar(value=False)
    
    def toggle_blur():
        global _enabled
        _enabled = enabled_var.get()
        main_window.update_image()
    
    ttk.Checkbutton(
        frame, 
        text="Enable gaussian blur",
        variable=enabled_var,
        command=toggle_blur
    ).pack(pady=10)
    
    # Slider for kernel size
    ttk.Label(frame, text="Kernel size:").pack()
    
    def on_slider_change(val):
        global _kernel_size
        # Ensure it's odd
        _kernel_size = int(float(val)) | 1
        size_label.config(text=f"{_kernel_size}x{_kernel_size}")
        if _enabled:
            main_window.update_image()
    
    slider = ttk.Scale(
        frame,
        from_=3,
        to=51,
        orient=tk.HORIZONTAL,
        command=on_slider_change
    )
    slider.set(_kernel_size)
    slider.pack(fill=tk.X, padx=20, pady=5)
    
    size_label = ttk.Label(frame, text=f"{_kernel_size}x{_kernel_size}")
    size_label.pack()
    
    return frame

def process(image: np.ndarray) -> np.ndarray:
    """Apply gaussian filter if active."""
    if _enabled and _kernel_size > 0:
        return cv2.GaussianBlur(image, (_kernel_size, _kernel_size), 0)
    return image
```

### Example 3: Edge Detection Plugin

```python
"""edge_detector.py - Edge detection with Canny."""

import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2

TAB_TITLE = "Edges"

_enabled = False
_threshold1 = 50
_threshold2 = 150

def setup(main_window, notebook, image_processor):
    """UI for edge detection control."""
    frame = ttk.Frame(notebook)
    
    # Variables
    enabled_var = tk.BooleanVar(value=False)
    
    def toggle():
        global _enabled
        _enabled = enabled_var.get()
        main_window.update_image()
    
    def update_thresholds(*args):
        global _threshold1, _threshold2
        _threshold1 = int(t1_var.get())
        _threshold2 = int(t2_var.get())
        if _enabled:
            main_window.update_image()
    
    # Checkbox
    ttk.Checkbutton(
        frame,
        text="Enable edge detection (Canny)",
        variable=enabled_var,
        command=toggle
    ).pack(pady=10)
    
    # Threshold 1
    ttk.Label(frame, text="Lower threshold:").pack()
    t1_var = tk.IntVar(value=_threshold1)
    ttk.Scale(
        frame,
        from_=0,
        to=255,
        variable=t1_var,
        orient=tk.HORIZONTAL,
        command=update_thresholds
    ).pack(fill=tk.X, padx=20)
    ttk.Label(frame, textvariable=t1_var).pack()
    
    # Threshold 2
    ttk.Label(frame, text="Upper threshold:").pack()
    t2_var = tk.IntVar(value=_threshold2)
    ttk.Scale(
        frame,
        from_=0,
        to=255,
        variable=t2_var,
        orient=tk.HORIZONTAL,
        command=update_thresholds
    ).pack(fill=tk.X, padx=20)
    ttk.Label(frame, textvariable=t2_var).pack()
    
    return frame

def process(image: np.ndarray) -> np.ndarray:
    """Detect edges with Canny algorithm."""
    if _enabled:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Detect edges
        edges = cv2.Canny(gray, _threshold1, _threshold2)
        # Convert to RGB for display
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return image
```

### Example 4: Canvas Overlay Plugin

```python
"""circle_overlay.py - Draws detected circles on image."""

import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2

TAB_TITLE = "Circles"

_circles = []  # List of detected circles
_show_overlay = False

def setup(main_window, notebook, image_processor):
    """UI for circle detection."""
    frame = ttk.Frame(notebook)
    
    canvas = main_window.image_panel.canvas
    overlay_tag = "circle_overlay"
    
    def detect_circles():
        global _circles
        if not image_processor.has_image():
            return
        
        # Get grayscale image
        img = image_processor.current_image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=100
        )
        
        _circles = circles[0] if circles is not None else []
        result_label.config(text=f"Circles detected: {len(_circles)}")
        
        if _show_overlay:
            draw_overlay()
    
    def draw_overlay():
        """Draw circles on canvas."""
        canvas.delete(overlay_tag)
        
        if not _circles or not _show_overlay:
            return
        
        zoom = image_processor.zoom or 1.0
        
        for (x, y, r) in _circles:
            x_canvas = x * zoom
            y_canvas = y * zoom
            r_canvas = r * zoom
            
            canvas.create_oval(
                x_canvas - r_canvas,
                y_canvas - r_canvas,
                x_canvas + r_canvas,
                y_canvas + r_canvas,
                outline="lime",
                width=2,
                tags=overlay_tag
            )
    
    def toggle_overlay():
        global _show_overlay
        _show_overlay = overlay_var.get()
        draw_overlay()
    
    # Button to detect
    ttk.Button(
        frame,
        text="Detect Circles",
        command=detect_circles
    ).pack(pady=10)
    
    # Results label
    result_label = ttk.Label(frame, text="No detection")
    result_label.pack(pady=5)
    
    # Checkbox to show overlay
    overlay_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        frame,
        text="Show overlay",
        variable=overlay_var,
        command=toggle_overlay
    ).pack(pady=5)
    
    return frame

def process(image: np.ndarray) -> np.ndarray:
    """Does not modify image, only uses overlays."""
    return image
```

---

## Accessing Resources

### ImageProcessor

```python
# In the setup function:
def setup(main_window, notebook, image_processor):
    # Check if image exists
    if image_processor.has_image():
        img = image_processor.current_image  # NumPy array (H, W, 3)
        height, width = img.shape[:2]
    
    # Current zoom factor
    zoom = image_processor.zoom
    
    # Measure circular region
    result = image_processor.measure_area(x, y)
    # Returns: (dose, std, unc, rgb_mean, rgb_std, pixel_count)
    
    # Current measurement size
    size = image_processor.measurement_size
```

### Canvas (for overlays)

```python
# In the setup function:
def setup(main_window, notebook, image_processor):
    canvas = main_window.image_panel.canvas
    
    # Draw circle
    canvas.create_oval(x1, y1, x2, y2, outline="red", tags="my_overlay")
    
    # Draw rectangle
    canvas.create_rectangle(x1, y1, x2, y2, fill="blue", tags="my_overlay")
    
    # Draw text
    canvas.create_text(x, y, text="Hello", fill="white", tags="my_overlay")
    
    # Clear overlays
    canvas.delete("my_overlay")
    
    # Canvas coordinates with zoom
    def canvas_coords(x_img, y_img):
        zoom = image_processor.zoom or 1.0
        return x_img * zoom, y_img * zoom
```

### Global Configuration

```python
# In on_config_change:
def on_config_change(config, image_processor):
    # Read configuration values
    use_gpu = config.get("use_gpu", False)
    colormap = config.get("colormap", "gray")
    num_threads = config.get("num_threads", 4)
    
    # Update plugin state according to config
    print(f"GPU: {use_gpu}, Threads: {num_threads}")
```

---

## Advanced Plugins (Packages)

For complex functionalities, use package structure. See complete example: `custom_plugins/auto_measurements/`

### Recommended Structure

```
my_plugin/
├── __init__.py              # TAB_TITLE, setup, process, on_config_change
├── ui/
│   ├── __init__.py
│   └── main_tab.py          # Main UI class
├── core/
│   ├── __init__.py
│   ├── detector.py          # Detection algorithms
│   ├── processor.py         # Image processing
│   └── exporter.py          # Data export
└── models/
    ├── __init__.py
    └── data_models.py       # Data classes
```

### `__init__.py` (Entry Point)

```python
"""My Complex Plugin - Description."""

from .ui.main_tab import MainTabUI
from .core.processor import ImageProcessorExtension

TAB_TITLE = "Complex Plugin"

# Global instance (if needed to share state)
_plugin_instance = None

def setup(main_window, notebook, image_processor):
    """Setup delegated to UI class."""
    global _plugin_instance
    _plugin_instance = MainTabUI(main_window, notebook, image_processor)
    return _plugin_instance.frame

def process(image):
    """Process delegated to instance."""
    if _plugin_instance:
        return _plugin_instance.process(image)
    return image

def on_config_change(config, image_processor):
    """Config change delegated to instance."""
    if _plugin_instance:
        _plugin_instance.on_config_change(config)
```

---

## Best Practices

### ✅ Do

1. **Use module-level global variables for state**
   ```python
   _enabled = False
   _parameters = {}
   ```

2. **Minimize logging in `process()`**
   ```python
   # DON'T do this in process()!
   # logging.debug(f"Processing image {image.shape}")  # Called every frame!
   
   # Only in specific events:
   def on_button_click():
       logging.info("Button clicked")
   ```

3. **Handle None images**
   ```python
   def process(image):
       if image is None:
           return None
       # ... process ...
   ```

4. **Use unique tags for overlays**
   ```python
   OVERLAY_TAG = "my_plugin_overlay_12345"
   canvas.create_oval(..., tags=OVERLAY_TAG)
   ```

5. **Clean resources when changing images**
   ```python
   def on_config_change(config, image_processor):
       # Clean old overlays
       canvas.delete(OVERLAY_TAG)
   ```

6. **Validate data before processing**
   ```python
   if not image_processor.has_image():
       messagebox.showwarning("Error", "No image loaded")
       return
   ```

### ❌ Avoid

1. **Don't block the main thread**
   ```python
   # ❌ BAD
   time.sleep(10)  # Freezes UI
   
   # ✅ GOOD
   threading.Thread(target=long_operation, daemon=True).start()
   ```

2. **Don't modify global application objects**
   ```python
   # ❌ BAD
   image_processor.current_image = my_image
   
   # ✅ GOOD
   # Return modified image from process()
   ```

3. **Don't forget to return the frame in setup()**
   ```python
   def setup(main_window, notebook, image_processor):
       frame = ttk.Frame(notebook)
       # ... configure UI ...
       return frame  # IMPORTANT!
   ```

4. **Don't assume plugin loading order**
   - Don't depend on other plugins
   - Each plugin must be independent

5. **Don't use absolute paths**
   ```python
   # ❌ BAD
   with open("C:/my_data/config.json")
   
   # ✅ GOOD
   import os
   plugin_dir = os.path.dirname(__file__)
   config_path = os.path.join(plugin_dir, "config.json")
   ```

---

## Debugging and Testing

### Logging

```python
import logging

logger = logging.getLogger(__name__)

def setup(main_window, notebook, image_processor):
    logger.info("Plugin initialized")
    
    def on_button_click():
        logger.debug("Button pressed")
        try:
            # ... code ...
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
```

Logs appear in `logs/rc_analyzer_YYYYMMDD_HHMMSS.log`

### Manual Testing

1. Place your plugin `.py` in `custom_plugins/`
2. Restart the application
3. The plugin will appear as a new tab
4. Check logs in the `logs/` folder

### Hot-Reload (Without Restarting)

If you modify the plugin and want to reload it:
1. Go to menu **Tools → Load Plugin** (if available)
2. Or restart the application

### Error Handling

```python
def process(image):
    try:
        # ... processing ...
        return processed_image
    except Exception as e:
        logger.error(f"Error in process(): {e}", exc_info=True)
        return image  # Return original image if fails
```

---

## API Reference

### ImageProcessor

| Method/Property | Description | Return |
|-----------------|-------------|--------|
| `has_image()` | Check if image is loaded | `bool` |
| `current_image` | Current image | `np.ndarray` (H, W, 3) RGB |
| `zoom` | Current zoom factor | `float` |
| `measurement_size` | ROI size for measurements | `int` (pixels) |
| `measure_area(x, y)` | Measure circular region at (x, y) | `tuple` (6 elements) |
| `calibration` | Active calibration model | `CalibrationModel` or `None` |

### MainWindow

| Method/Property | Description |
|-----------------|-------------|
| `image_panel` | Image display panel |
| `image_panel.canvas` | Tkinter canvas for drawing |
| `update_image()` | Force image display update |

### Canvas (tkinter)

| Method | Description |
|--------|-------------|
| `create_oval(x1, y1, x2, y2, **opts)` | Draw circle/ellipse |
| `create_rectangle(x1, y1, x2, y2, **opts)` | Draw rectangle |
| `create_text(x, y, text="...", **opts)` | Draw text |
| `create_line(x1, y1, x2, y2, **opts)` | Draw line |
| `delete(tag)` | Delete elements with specific tag |

## Support and Contributions

### Need help?

1. Check logs in `logs/`
2. Consult the `auto_measurements` code as reference
3. Open an issue on GitHub

### Contributing Your Plugin

If you create a useful plugin:
1. Document it well
2. Add usage examples
3. Create a Pull Request
4. Include `README.md` in your plugin folder


