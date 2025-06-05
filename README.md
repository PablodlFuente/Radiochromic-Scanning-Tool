# Radiochromic Film Analyzer

## Overview

The Radiochromic Film Analyzer is a specialized application designed for the analysis and measurement of radiochromic films used in medical physics and radiation dosimetry. This tool provides precise measurements, visualization, and analysis capabilities for researchers and medical physicists.

## Features

- **Image Processing**: Open, view, and analyze radiochromic film images
- **Measurement Tools**: Circular and square measurement regions with adjustable size
- **Statistical Analysis**: Calculate average RGB values, standard deviation, and uncertainty
- **Visualization**: 
  - Real-time RGB histograms
  - 2D heat maps of intensity distribution
  - 3D surface plots of RGB channels
- **Calibration**: Calibrate measurements against known dose values
- **Image Enhancement**: Adjust contrast, brightness, saturation, and negative mode
- **Configuration**: Save and load application settings

## Installation

### Prerequisites

- Python 3.7 or higher
- Required Python packages:
  - tkinter
  - numpy
  - matplotlib
  - pillow (PIL)
  - scipy (for 2D/3D visualization)

### Installation Steps

1. Clone the repository or download the source code:
   \`\`\`
   git clone https://github.com/yourusername/radiochromic-film-analyzer.git
   cd radiochromic-film-analyzer
   \`\`\`

2. Install the required dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

3. Run the application:
   \`\`\`
   python main.py
   \`\`\`

## Usage Guide

### Opening an Image

1. Click on "File" → "Open" or press Ctrl+O
2. Select a radiochromic film image file (supported formats: PNG, JPEG, TIFF)
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

- **Ctrl+O**: Open image
- **Ctrl+S**: Save image
- **Ctrl+Z**: Undo
- **Ctrl+Y**: Redo
- **Ctrl+Mouse Wheel**: Adjust measurement size
- **+/-**: Zoom in/out

## Troubleshooting

### Common Issues

- **Application fails to start**: Ensure all dependencies are installed correctly
- **Image doesn't load**: Check if the image format is supported
- **Visualization is slow**: Reduce the measurement size for faster performance

### Error Logs

- Error logs are stored in `rc_analyzer.log` in the application directory
- Check this file for detailed error information if you encounter issues

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This application was developed for research purposes in medical physics
- Special thanks to contributors and testers
