# Radiochromic-Scanning-Tool

## Overview

The Radiochromic-Scanning-Tool is a specialized application designed for the analysis and measurement of radiochromic films used in medical physics and radiation dosimetry. This tool provides precise measurements, visualization, and analysis capabilities for researchers and medical physicists.

## Features

- **Image Processing**: Open, view, and analyse radiochromic film images.
- **Measurement Tools**: Circular, rectangular and line-profile regions with adjustable size; instant numeric preview under the cursor.
- **Auto-Measurements**: Built-in plugin that automatically detects circular ROIs, computes dose & uncertainty and exports results to CSV in one click.
- **Statistical Analysis**: Average pixel / dose values, standard deviation and fully propagated uncertainty, both per-channel and averaged.
- **Consistent Formatting**: Values and uncertainties share matching precision (two decimals for calibrated dose) across the UI and CSV export.
- **Visualization**:
  - Real-time RGB / dose histograms
  - 2-D heat maps of intensity or dose distribution
  - 3-D surface plots of RGB channels or dose
- **Calibration**: Non-linear calibration against reference doses with full uncertainty propagation; toggle on/off instantly.
- **Performance**: Multithreading, optional GPU acceleration (CuPy) and on-the-fly binning (2×, 4×…) for large images.
- **Plugins & Extensibility**: Load/unload external Python plugins directly from the GUI for custom workflows.
- **Image Enhancement**: Contrast, brightness, saturation and negative mode.
- **Configuration**: All settings are saved automatically between sessions.

## Installation

### Prerequisites

- Python 3.7 or higher
- Required Python packages:
  - tkinter (included with standard Python on Windows/macOS)
  - numpy
  - matplotlib
  - pillow (PIL)
  - scipy (for 2-D/3-D visualisation)
  - opencv-python (CV2 image processing & fast statistics)
  - scikit-image (required for auto-measurements plugin)
  - cupy-cuda** (optional, for GPU acceleration – choose the build matching your CUDA version)

### Installation Steps

1. Clone the repository or download the source code:
   ```
   git clone https://github.com/yourusername/radiochromic-film-analyzer.git
   cd radiochromic-film-analyzer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python main.py
   ```

## Usage Guide

### Opening an Image

1. Click on "File" → "Open" 
2. Select a radiochromic film image file (supported formats: TIFF)
3. The image will be displayed in the main window

### Taking Measurements

1. Select the measurement shape (circular or square) in the Measurement panel
2. Adjust the measurement size using the spinbox or Ctrl+Mouse Wheel
3. Move your cursor over the image to see the measurement region
4. Click on the area you want to measure
5. View the measurement results in the Results section

### Visualizing Data

- **Histogram**: Automatically updates with each measurement
- **2D View**: Click the "2D View" button to see a heat map of intensity values
- **3D View**: Click the "3D View" button to see a 3D surface plot of RGB channels

### Adjusting Image Settings

1. Use the sliders in the Image Settings panel to adjust:
   - Contrast
   - Brightness
   - Saturation
2. Toggle "Negative mode" to invert image colors

### Calibration

1. Click on "Tools" → "Calibration Wizard"
2. Follow the on-screen instructions to calibrate your measurements

### Saving Configuration

- The application automatically saves your settings when you exit
- Settings are stored in `rc_config.json` in the application directory

## Keyboard Shortcuts

- **Ctrl+Mouse Wheel**: Adjust measurement size
- **Mouse Wheel**: Zoom in/out

### Error Logs

- Error logs are stored in `rc_analyzer.log` in the application directory
- Check this file for detailed error information if you encounter issues

## License

This project is licensed under the GNU GPL v3 - see the LICENSE file for details.

## Acknowledgments

- This application was developed for research purposes in medical physics
- Special thanks to contributors and testers
