"""
Image processor for the Radiochromic Film Analyzer.

This module contains the image processor class that handles image loading,
processing, and analysis.
"""

import numpy as np
from PIL import Image, ImageTk, ImageOps, ImageEnhance
import cv2
import logging
import os
import threading
import shutil
from datetime import datetime
import tempfile
import pickle
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import math
import platform
import psutil
from scipy import ndimage
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Image processor for the Radiochromic Film Analyzer."""
    
    def __init__(self, config):
        """Initialize the image processor."""
        self.config = config
        
        # Image data
        self.current_image = None
        self.original_image = None
        self.flattened_image = None  # Store image with flat applied (for display when calibration active)
        self.current_file = None
        self.binned_image = None  # Store the binned image
        # Track whether dose calibration is currently applied
        self.calibration_applied = False
        # Track whether field flattening is currently applied
        self.flat_applied = False
        
        # Display settings
        self.zoom = 1.0
        self.negative_mode = config.get("negative_mode", False)
        self.contrast = 1.0
        self.brightness = 1.0
        self.saturation = 1.0
        self.binning = 1  # Default binning is 1x1 (no binning)
        self.binned_image_file = None
        
        # Image bit depth (detected on load)
        self.image_bit_depth = 8  # Actual bit depth (8, 12, 14, 16, etc.)
        self.image_max_value = 255  # Maximum possible value for current bit depth
        
        # Measurement settings
        self.measurement_shape = "circular"
        self.measurement_size = 20
        self.measurement_size_rect = (40, 40)  # (width, height) for rectangular measurements
        self.line_orientation = "horizontal"  # "horizontal", "vertical", or "manual" for line measurements
        self.manual_line_points = None  # [(x1, y1), (x2, y2)] for manual line measurements
        self.auto_measure = config.get("auto_measure", False)
        
        # Measurement data
        self.last_measurement_raw_data = None
        self.last_measurement_coordinates = None
        self.last_auto_measure_time = time.time()
        
        # Integral image cache
        self.integral_images = None
        self.integral_images_squared = None
        
        # Calibration data
        self.calibration = None
        self.calibration_bit_depth = 8  # Default, updated when loading calibration params
        
        # Field flattening data (loaded from calibration_data/field_flattening.npz)
        self.flat_field = None  # Normalized flat field array (H, W, 3)
        self.flat_field_info = None  # Metadata about the flat field
        
        # Processing flags
        self.processing_lock = threading.Lock()
        self.max_display_size = 4096
        
        # Progress tracking
        self.progress_callback = None
        self.cancel_operation = False
        
        # System information
        self.system_info = self._get_system_info()
        
        # GPU acceleration
        self.cuda_available = self._check_cuda_support()  # Add CUDA availability flag
        self.gpu_available = False
        self.gpu_enabled = False
        self.gpu_devices = {}
        self.gpu_force_enabled = config.get("gpu_force_enabled", False)
        if self.cuda_available:
            self._init_gpu()
        else:
            logger.warning("CUDA not available, skipping GPU initialization")
        
        # Multi-threading
        self.use_multithreading = config.get("use_multithreading", True)
        self.num_threads = config.get("num_threads", os.cpu_count() or 4)
        
        # Set up temp directory
        self.temp_dir = self._setup_temp_dir()
        
        # Resource tracking
        self._temp_files = []  # Track temporary files for better cleanup
        
        # Performance optimization
        self.last_process_time = 0
        self.processing_threshold = 16.67  # ~60fps in milliseconds
        
        logger.info("Image processor initialized")
    
    def _get_system_info(self):
        """Get system information for optimization decisions."""
        info = {
            "platform": platform.system(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024 ** 3),
            "python_version": platform.python_version(),
            "opencv_version": cv2.__version__
        }
        
        logger.info(f"System info: {info}")
        return info
    
    def _check_cuda_support(self):
        """Check if OpenCV was built with CUDA support."""
        try:
            # Check CUDA device count
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logger.info("CUDA support detected")
                return True
        except Exception as e:
            logger.warning(f"CUDA check failed: {str(e)}")
        return False
    
    def set_progress_callback(self, callback):
        """Set the progress callback function."""
        self.progress_callback = callback
    
    def _report_progress(self, operation, progress, status=None):
        """Report progress to the callback function."""
        if self.progress_callback:
            self.progress_callback(operation, progress, status)
    
    def cancel_current_operation(self):
        """Cancel the current operation."""
        self.cancel_operation = True
    
    def _setup_temp_dir(self):
        """Set up the temporary directory for the application."""
        # Create a temp directory in the application folder
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp")
        
        # Create the directory if it doesn't exist
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            logger.info(f"Created temporary directory: {temp_dir}")
        
        return temp_dir
    
    def _init_gpu(self):
        """Initialize GPU acceleration if available."""
        try:
            # First check if GPU is force-enabled in config
            if self.gpu_force_enabled:
                logger.info("GPU acceleration force-enabled by user")
                self.gpu_available = True
                self.gpu_enabled = True
                self.gpu_devices = {"0": {"name": "Force-enabled GPU", "memory": "Unknown"}}
                return True
            
            # Try multiple methods to detect CUDA capabilities
            cuda_detected = False
            
            # Method 1: Standard OpenCV CUDA detection
            try:
                if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    cuda_detected = True
                    logger.info("CUDA detected using standard OpenCV method")
                    
                    # Get GPU device info
                    device_info = {}
                    for i in range(cv2.cuda.getCudaEnabledDeviceCount()):
                        device_info[i] = {
                            "name": cv2.cuda.getDeviceName(i),
                            "memory": "Unknown"  # OpenCV doesn't provide memory info easily
                        }
                    self.gpu_devices = device_info
            except Exception as e:
                logger.warning(f"Standard CUDA detection failed: {str(e)}")
            
            # Method 2: Try to create a simple CUDA matrix with better error handling
            if not cuda_detected:
                try:
                    # Create a small test matrix
                    test_array = np.zeros((10, 10), dtype=np.uint8)
                    
                    # Try to create a GpuMat and upload data
                    gpu_mat = cv2.cuda_GpuMat()
                    gpu_mat.upload(test_array)
                    
                    # Try to download to verify it works
                    result = gpu_mat.download()
                    
                    if result is not None:
                        cuda_detected = True
                        logger.info("CUDA detected using test matrix method")
                        
                        # Set basic device info
                        self.gpu_devices = {"0": {"name": "Detected GPU", "memory": "Unknown"}}
                except Exception as e:
                    logger.warning(f"CUDA test matrix creation failed: {str(e)}")
                    # Try alternative method for older OpenCV versions
                    try:
                        # Some versions use cv2.cuda.createGpuMatFromCudaMemory instead
                        test_array = np.zeros((10, 10), dtype=np.uint8)
                        gpu_mat = cv2.cuda.createGpuMatFromCudaMemory(test_array.shape, cv2.CV_8UC1)
                        cuda_detected = True
                        logger.info("CUDA detected using alternative test matrix method")
                        self.gpu_devices = {"0": {"name": "Detected GPU", "memory": "Unknown"}}
                    except Exception as e2:
                        logger.warning(f"Alternative CUDA test failed: {str(e2)}")
            
            # Method 3: Check for NVIDIA GPU using platform-specific methods
            if not cuda_detected:
                if self.system_info["platform"] == "Windows":
                    try:
                        import subprocess
                        result = subprocess.run(["wmic", "path", "win32_VideoController", "get", "name"], 
                                               capture_output=True, text=True)
                        output = result.stdout.lower()
                        if "nvidia" in output:
                            # Extract the actual GPU name from the output
                            lines = [line.strip() for line in output.split('\n') if line.strip() and "name" not in line.lower()]
                            gpu_name = next((line for line in lines if "nvidia" in line.lower()), "NVIDIA GPU")
                            
                            logger.info(f"NVIDIA GPU detected: {gpu_name}")
                            logger.info("CUDA initialization failed. Setting GPU as available but with warning.")
                            
                            # Set as available but with warning
                            cuda_detected = True
                            self.gpu_devices = {"0": {"name": gpu_name, "memory": "Unknown"}}
                            
                            # Log additional information about CUDA environment
                            logger.info("To fix CUDA issues, try:")
                            logger.info("1. Install the latest NVIDIA drivers")
                            logger.info("2. Install CUDA toolkit compatible with your GPU")
                            logger.info("3. Use 'Force enable GPU' option in settings")
                    except Exception as e:
                        logger.warning(f"Windows GPU detection failed: {str(e)}")
                elif self.system_info["platform"] == "Linux":
                    try:
                        import subprocess
                        result = subprocess.run(["lspci"], capture_output=True, text=True)
                        output = result.stdout.lower()
                        if "nvidia" in output:
                            logger.info("NVIDIA GPU detected using Linux-specific method")
                            self.gpu_devices = {"0": {"name": "NVIDIA GPU (detected by system)", "memory": "Unknown"}}
                            cuda_detected = True
                    except Exception as e:
                        logger.warning(f"Linux GPU detection failed: {str(e)}")
            
            # Set GPU availability and enabled state
            self.gpu_available = cuda_detected
            self.gpu_enabled = cuda_detected and self.config.get("use_gpu", False)
            
            if cuda_detected:
                logger.info(f"GPU acceleration available. Enabled: {self.gpu_enabled}")
                return True
            else:
                logger.info("No CUDA-capable GPU detected")
                return False

            # If we detected an NVIDIA GPU but CUDA failed, suggest using force enable
            if cuda_detected and "nvidia" in str(self.gpu_devices).lower():
                logger.info("NVIDIA GPU detected but CUDA may not be fully functional.")
                logger.info("You can try using the 'Force enable GPU' option in settings.")
                
                # Enable GPU if force-enabled is set
                if self.gpu_force_enabled:
                    self.gpu_enabled = self.config.get("use_gpu", False)
                    logger.info(f"Force-enabled GPU. GPU acceleration enabled: {self.gpu_enabled}")
        except Exception as e:
            logger.error(f"Error initializing GPU: {str(e)}", exc_info=True)
            self.gpu_available = False
            self.gpu_enabled = False
            return False
    
    def set_gpu_enabled(self, enabled):
        """Enable or disable GPU acceleration."""
        if enabled and not self.gpu_available and not self.gpu_force_enabled:
            logger.warning("Cannot enable GPU acceleration: No GPU available")
            return False
        
        self.gpu_enabled = enabled
        self.config["use_gpu"] = enabled
        logger.info(f"GPU acceleration set to: {enabled}")
        return True
    
    def set_gpu_force_enabled(self, force_enabled):
        """Force enable GPU acceleration even if no GPU is detected."""
        self.gpu_force_enabled = force_enabled
        self.config["gpu_force_enabled"] = force_enabled
        
        if force_enabled:
            self.gpu_available = True
            self.gpu_enabled = True
            logger.info("GPU acceleration force-enabled by user")
        else:
            # Re-initialize GPU detection
            self._init_gpu()
        
        return True

    def _sanitize_filename(self, file_path: str) -> tuple[str, bool]:
        """Check if filename has problematic Unicode characters and rename if needed.
        
        OpenCV's imread cannot handle certain Unicode characters in file paths.
        This method detects such characters and renames the file.
        
        Returns:
            tuple: (new_file_path, was_renamed)
        """
        import unicodedata
        
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # Map of accented characters to their base form
        accent_map = {
            'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ã': 'a',
            'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e',
            'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i',
            'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'õ': 'o',
            'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u',
            'ñ': 'n', 'ç': 'c',
            'Á': 'A', 'À': 'A', 'Ä': 'A', 'Â': 'A', 'Ã': 'A',
            'É': 'E', 'È': 'E', 'Ë': 'E', 'Ê': 'E',
            'Í': 'I', 'Ì': 'I', 'Ï': 'I', 'Î': 'I',
            'Ó': 'O', 'Ò': 'O', 'Ö': 'O', 'Ô': 'O', 'Õ': 'O',
            'Ú': 'U', 'Ù': 'U', 'Ü': 'U', 'Û': 'U',
            'Ñ': 'N', 'Ç': 'C',
        }
        
        new_filename = []
        has_problematic_chars = False
        
        for char in filename:
            if char in accent_map:
                new_filename.append(accent_map[char])
                has_problematic_chars = True
            elif ord(char) > 127:
                # Other non-ASCII characters -> underscore
                new_filename.append('_')
                has_problematic_chars = True
            else:
                new_filename.append(char)
        
        if not has_problematic_chars:
            return file_path, False
        
        new_filename = ''.join(new_filename)
        new_path = os.path.join(directory, new_filename)
        
        # Check for collision: if new_path already exists and is different from original
        if os.path.exists(new_path) and os.path.normpath(new_path) != os.path.normpath(file_path):
            # Add a unique suffix to avoid overwriting
            base, ext = os.path.splitext(new_filename)
            counter = 1
            while os.path.exists(new_path):
                new_filename = f"{base}_{counter}{ext}"
                new_path = os.path.join(directory, new_filename)
                counter += 1
            logger.info(f"Collision detected, using unique filename: {new_filename}")
        
        # Rename the file
        try:
            os.rename(file_path, new_path)
            logger.info(f"Renamed file with Unicode characters: '{filename}' -> '{new_filename}'")
            return new_path, True
        except Exception as e:
            logger.error(f"Could not rename file: {e}")
            return file_path, False

    def load_image(self, file_path):
        """Load an image from the specified path."""
        try:
            # Reset cancel flag
            self.cancel_operation = False
            
            # Calibration state resets when loading a new image
            self.calibration_applied = False
            
            # Check for problematic Unicode characters in filename
            original_path = file_path
            file_path, was_renamed = self._sanitize_filename(file_path)
            if was_renamed:
                # Store info for UI notification
                self._renamed_file_info = (os.path.basename(original_path), os.path.basename(file_path))
            else:
                self._renamed_file_info = None
            
            # Report progress
            self._report_progress("loading", 0, "Starting image load")
            
            # Clear any existing binned image files for the previous image
            if hasattr(self, 'binned_image_file') and self.binned_image_file and os.path.exists(self.binned_image_file):
                try:
                    os.remove(self.binned_image_file)
                    logger.debug(f"Removed previous binned image file: {self.binned_image_file}")
                except Exception as e:
                    logger.warning(f"Could not remove previous binned image file: {str(e)}")
                
                self.binned_image_file = None
            
            # Report progress
            self._report_progress("loading", 10, "Reading image file")
            
            # Load image with OpenCV for better performance
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            # Explicit check: cv2.imread returns None if it cannot read the file
            if image is None:
                logger.error(f"cv2.imread failed to read file: {file_path}")
                logger.error("Possible causes: file path contains special characters, file is corrupted, or unsupported format")
                raise IOError(f"Could not read image file. The file may be corrupted, in an unsupported format, or the path contains special characters that weren't properly sanitized.")
            
            # Check if operation was cancelled
            if self.cancel_operation:
                logger.info("Image loading cancelled")
                return False
            
            # Report progress
            self._report_progress("loading", 30, "Processing image")
            
            # Convert BGR/BGRA to RGB
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    # RGBA image: convert to RGB by removing alpha channel
                    logger.info("Converting RGBA image to RGB")
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                elif image.shape[2] == 3:
                    # BGR image: convert to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Check image size and downsample if necessary for memory efficiency
            height, width = image.shape[:2]
            logger.info(f"Loading image: {file_path}, size: {width}x{height}")
            
            # For very large images, downsample immediately to save memory
            if width > 4000 or height > 4000:
                self._report_progress("loading", 40, "Downsampling large image")
                scale_factor = min(4000 / width, 4000 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                logger.info(f"Downsampling large image to {new_width}x{new_height}")
                
                # Use GPU for downsampling if available
                if self.cuda_available:
                    try:
                        # Upload image to GPU
                        gpu_image = cv2.cuda_GpuMat()
                        gpu_image.upload(image)
                        
                        # Resize on GPU
                        resized = cv2.cuda.resize(gpu_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        
                        # Download result
                        image = resized.download()
                    except Exception as e:
                        logger.error(f"GPU downsampling error: {str(e)}, falling back to CPU", exc_info=True)
                        # Fall back to CPU
                        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                else:
                    # CPU downsampling
                    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Check if operation was cancelled
            if self.cancel_operation:
                logger.info("Image loading cancelled")
                return False
            
            # Report progress
            self._report_progress("loading", 60, "Processing background")
            
            # Remove white background if option is enabled
            if self.config.get("remove_background", False):
                image = self.detect_useful_area(image)
            
            # Detect and store image bit depth based on dtype and actual values
            self.image_bit_depth, self.image_max_value = self._detect_bit_depth(image)
            logger.info(f"Detected {self.image_bit_depth}-bit image (max value: {self.image_max_value})")
            
            # Store images - ensure we're preserving the original data type
            self.original_image = image
            self.current_image = self.original_image.copy()
            self.current_file = file_path
            
            # Report progress
            self._report_progress("loading", 70, "Computing integral images")
            
            # Compute integral images for fast statistics
            self._compute_integral_images()
            
            # Report progress
            self._report_progress("loading", 80, "Applying settings")
            
            # Apply binning if needed
            if self.binning > 1:
                self.apply_binning()
            else:
                self.binned_image = None
            
            # Reset display settings
            self.zoom = 1.0
            self.contrast = 1.0
            self.brightness = 1.0
            self.saturation = 1.0
            
            # Clear measurement data
            self.last_measurement_raw_data = None
            self.last_measurement_coordinates = None
            
            # Report progress
            self._report_progress("loading", 100, "Image loaded successfully")
            
            logger.info(f"Loaded image: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}", exc_info=True)
            self._report_progress("loading", 100, f"Error: {str(e)}")
            raise
    
    def _compute_integral_images(self):
        """Compute integral images for fast statistics calculation."""
        try:
            if self.current_image is None:
                return
            
            logger.info("Computing integral images for fast statistics")
            
            # For very large images, use a downsampled version for integral images
            height, width = self.current_image.shape[:2]
            very_large = width > 8000 or height > 8000
            
            if very_large:
                logger.info(f"Using downsampled integral images for large image ({width}x{height})")
                downsample_factor = 2
                if len(self.current_image.shape) == 3:
                    downsampled = self.current_image[::downsample_factor, ::downsample_factor, :]
                else:
                    downsampled = self.current_image[::downsample_factor, ::downsample_factor]
                
                # Store the downsample factor for later use
                self._integral_downsample_factor = downsample_factor
                image_for_integral = downsampled
            else:
                self._integral_downsample_factor = 1
                image_for_integral = self.current_image
            
            # For color images, compute integral images for each channel
            if len(image_for_integral.shape) == 3:
                height, width, channels = image_for_integral.shape
                
                # Initialize integral images for each channel
                self.integral_images = []
                self.integral_images_squared = []
                
                for c in range(channels):
                    # Extract channel
                    channel = image_for_integral[:, :, c].astype(np.float32)
                    
                    # Compute integral image
                    integral = cv2.integral(channel)
                    
                    # Compute integral of squared values for variance/std calculation
                    integral_squared = cv2.integral(channel * channel)
                    
                    self.integral_images.append(integral)
                    self.integral_images_squared.append(integral_squared)
            else:
                # For grayscale images
                # Convert to float32 for precision
                img_float = image_for_integral.astype(np.float32)
                
                # Compute integral image
                self.integral_images = cv2.integral(img_float)
                
                # Compute integral of squared values for variance/std calculation
                self.integral_images_squared = cv2.integral(img_float * img_float)
            
            logger.info("Integral images computed successfully")
        except Exception as e:
            logger.error(f"Error computing integral images: {str(e)}", exc_info=True)
            # Reset integral images on error
            self.integral_images = None
            self.integral_images_squared = None
    
    def detect_useful_area(self, image):
        """Detect the useful area of the image and crop the white background."""
        try:
            # Convert to grayscale for processing if it's a color image
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply threshold to separate foreground from background
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the bounding rectangle that contains all contours
                x_min, y_min, width, height = cv2.boundingRect(np.vstack([cnt for cnt in contours]))
                x_max = x_min + width
                y_max = y_min + height
                
                # Add a margin
                margin = 10
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(image.shape[1], x_max + margin)
                y_max = min(image.shape[0], y_max + margin)
                
                # Crop the image
                return image[y_min:y_max, x_min:x_max]
            
            return image
                
        except Exception as e:
            logger.error(f"Error detecting useful area: {str(e)}", exc_info=True)
            # In case of error, return the original image
            return image
    
    def _calculate_optimal_tile_size(self, width, height, channels):
        """Calculate optimal tile size based on image dimensions and system resources."""
        # Base tile size on available memory
        memory_gb = self.system_info["memory_gb"]
        
        # For GPU processing, use larger tiles to reduce overhead
        if self.cuda_available and self.gpu_enabled:
            # GPU processing benefits from larger tiles due to transfer overhead
            if memory_gb < 4:
                max_tile_size = 1024
            elif memory_gb < 8:
                max_tile_size = 2048
            else:
                max_tile_size = 4096
            
            # For GPU, we want fewer, larger tiles
            num_tiles = 4  # Fixed number of tiles for GPU
        else:
            # For CPU processing, use smaller tiles for better parallelism
            if memory_gb < 4 or (width > 4000 and height > 4000):
                max_tile_size = 512
            elif memory_gb < 8 or (width > 2000 and height > 2000):
                max_tile_size = 1024
            else:
                max_tile_size = 2048
        
        # For CPU, use one tile per core for optimal parallelism
        cpu_count = self.system_info["cpu_count"] or 4
        num_tiles = cpu_count
    
        # Calculate tile dimensions to get approximately num_tiles total
        tiles_per_dimension = int(math.sqrt(num_tiles))
    
        # Calculate tile width and height
        tile_width = min(max_tile_size, width // tiles_per_dimension)
        tile_height = min(max_tile_size, height // tiles_per_dimension)
    
        # Ensure tile dimensions are multiples of binning size
        tile_width = ((tile_width + self.binning - 1) // self.binning) * self.binning
        tile_height = ((tile_height + self.binning - 1) // self.binning) * self.binning
    
        # Ensure minimum tile size
        tile_width = max(tile_width, self.binning * 10)
        tile_height = max(tile_height, self.binning * 10)
    
        logger.debug(f"Using tile size: {tile_width}x{tile_height} for {'GPU' if self.gpu_enabled else 'CPU'} processing")
        return tile_width, tile_height
    
    def _process_bin_region(self, y, x, y_start, y_end, x_start, x_end, channels):
        """Process a single bin region for multithreaded binning."""
        if channels == 1:
            # Extract region
            region = self.original_image[y_start:y_end, x_start:x_end]
            
            # Calculate mean and standard deviation
            return np.mean(region), np.std(region)
        else:
            # Process each channel
            means = []
            stds = []
            for c in range(channels):
                # Extract region for this channel
                region = self.original_image[y_start:y_end, x_start:x_end, c]
                
                # Calculate mean and standard deviation
                means.append(np.mean(region))
                stds.append(np.std(region))
            
            return means, stds
    
    def apply_binning(self):
        """Apply binning to the original image and save the result."""
        if self.original_image is None or self.binning <= 1:
            self.binned_image = None
            return
        
        try:
            # Reset cancel flag
            self.cancel_operation = False
            
            # Get image dimensions
            if len(self.original_image.shape) == 3:
                height, width, channels = self.original_image.shape
            else:
                height, width = self.original_image.shape
                channels = 1
        
            # Report start of binning
            logger.info(f"Starting binning operation: {self.binning}x{self.binning} on image {width}x{height}")
            self._report_progress("binning", 0, f"Starting {self.binning}x{self.binning} binning on {width}x{height} image")
        
            # Calculate new dimensions (ensure they're multiples of binning factor)
            new_height = (height // self.binning) * self.binning
            new_width = (width // self.binning) * self.binning
        
            # Crop image to ensure dimensions are multiples of binning factor
            img = self.original_image[:new_height, :new_width]
        
            # Estimate memory requirement
            memory_required = img.nbytes * 3  # Original + mean + std_dev
            available_memory = psutil.virtual_memory().available
        
            # Choose method based on available memory
            if memory_required < available_memory * 0.7:  # Use 70% of available memory as threshold
                # Method 1: Direct vectorization (when image fits in memory)
                self._report_progress("binning", 10, "Using direct vectorization method")
                binned, std_dev = self._bin_image_vectorized(img, self.binning)
            else:
                # Method 2: Stripe processing for large images
                self._report_progress("binning", 10, "Using stripe processing method for large image")
                stripe_height = self.binning * 1000  # Process 1000 output rows at a time
                binned, std_dev = self._bin_image_stripes(img, self.binning, stripe_height)
        
            # Store the binned image and standard deviation
            self.binned_image = {
                'data': binned.astype(np.uint8),
                'std_dev': std_dev
            }
        
            # Update current image to use binned image
            self.current_image = binned.astype(np.uint8)
        
            # Save the binned image to a temporary file
            self._save_binned_image()
            
            # Recompute integral images for the binned image
            self._compute_integral_images()
        
        # Report completion
            logger.info(f"Binning complete: {self.binning}x{self.binning} on {width}x{height} image")
            self._report_progress("binning", 100, f"Binning complete: {self.binning}x{self.binning}")

        except Exception as e:
            logger.error(f"Error applying binning: {str(e)}", exc_info=True)
            self._report_progress("binning", 100, f"Error: {str(e)}")
            self.binned_image = None

    def _bin_image_vectorized(self, img, binning):
        """Bin image using vectorized NumPy operations."""
        self._report_progress("binning", 20, "Preparing image for vectorized binning")
        
        # Get dimensions
        if len(img.shape) == 2:
            h, w = img.shape
            # Reshape to blocks
            blocks = img.reshape(h//binning, binning, w//binning, binning)
        
            # Calculate mean and std_dev
            self._report_progress("binning", 50, "Calculating mean and standard deviation")
            mean = blocks.mean(axis=(1,3))
            # Calcular correctamente la desviación estándar
            std_dev = blocks.std(axis=(1,3))
        else:
            h, w, c = img.shape
            # Reshape to blocks
            blocks = img.reshape(h//binning, binning, w//binning, binning, c)
        
            # Calculate mean and std_dev
            self._report_progress("binning", 50, "Calculating mean and standard deviation")
            mean = blocks.mean(axis=(1,3))
            # Calcular correctamente la desviación estándar
            std_dev = blocks.std(axis=(1,3))
        
        self._report_progress("binning", 95, "Vectorized binning complete")
        return mean, std_dev

    def _bin_image_stripes(self, img, binning, stripe_height):
        """Bin image using stripe processing for large images."""
        # Get dimensions
        if len(img.shape) == 2:
            h, w = img.shape
            c = 1
        else:
            h, w, c = img.shape
        
        # Ensure stripe_height is a multiple of binning
        stripe_height = (stripe_height // binning) * binning
        
        # Calculate output dimensions
        out_h = h // binning
        out_w = w // binning
        
        # Create output arrays
        if c == 1:
            mean_out = np.zeros((out_h, out_w), dtype=np.float32)
            std_out = np.zeros((out_h, out_w), dtype=np.float32)
        else:
            mean_out = np.zeros((out_h, out_w, c), dtype=np.float32)
            std_out = np.zeros((out_h, out_w, c), dtype=np.float32)
        
        # Process image in stripes
        total_stripes = math.ceil(h / stripe_height)
        for i, y0 in enumerate(range(0, h, stripe_height)):
            # Check if operation was cancelled
            if self.cancel_operation:
                logger.info("Binning operation cancelled")
                return None, None
            
            # Calculate stripe boundaries
            y1 = min(y0 + stripe_height, h)
            stripe_h = y1 - y0
            
            # Extract stripe
            if len(img.shape) == 2:
                stripe = img[y0:y1, :]
            else:
                stripe = img[y0:y1, :, :]
            
            # Process stripe
            if len(img.shape) == 2:
                # Ensure stripe dimensions are multiples of binning
                stripe_h_valid = (stripe_h // binning) * binning
                if stripe_h_valid == 0:
                    continue  # Skip if stripe is too small
            
                stripe = stripe[:stripe_h_valid, :]
            
                # Reshape to blocks
                blocks = stripe.reshape(stripe_h_valid//binning, binning, w//binning, binning)
            
                # Calculate mean and std_dev
                mean = blocks.mean(axis=(1,3))
                # Usar std directamente en lugar de calcular desde mean2
                std = blocks.std(axis=(1,3))
            else:
                # Ensure stripe dimensions are multiples of binning
                stripe_h_valid = (stripe_h // binning) * binning
                if stripe_h_valid == 0:
                    continue  # Skip if stripe is too small
                
                stripe = stripe[:stripe_h_valid, :, :]
                
                # Reshape to blocks
                blocks = stripe.reshape(stripe_h_valid//binning, binning, w//binning, binning, c)
                
                # Calculate mean and std_dev
                mean = blocks.mean(axis=(1,3))
                # Usar std directamente en lugar de calcular desde mean2
                std = blocks.std(axis=(1,3))
        
            # Copy results to output arrays
            out_y0 = y0 // binning
            out_y1 = out_y0 + mean.shape[0]
            mean_out[out_y0:out_y1, :] = mean
            std_out[out_y0:out_y1, :] = std
        
            # Report progress
            progress = 20 + (i + 1) / total_stripes * 70
            self._report_progress("binning", progress, f"Processing stripe {i+1}/{total_stripes}")
    
        return mean_out, std_out
    
    def get_auto_measure(self):
        """Get the auto measure setting."""
        return self.auto_measure

    def set_measurement_shape(self, shape):
        """Set the measurement shape."""
        self.measurement_shape = shape
        logger.debug(f"Measurement shape set to {shape}")
        return True
    
    def set_measurement_size(self, size):
        """Set the measurement size."""
        self.measurement_size = size
        logger.debug(f"Measurement size set to {size}")
        return True
    
    def set_auto_measure(self, auto_measure):
        """Set the auto measure flag."""
        self.auto_measure = auto_measure
        logger.debug(f"Auto measure set to {auto_measure}")
        return True
    
    def get_measurement_settings(self):
        """Get the current measurement settings."""
        return self.measurement_shape, self.measurement_size
    
    def get_last_measurement_raw_data(self):
        """Get the raw data from the last measurement."""
        return self.last_measurement_raw_data
    
    def get_last_measurement_coordinates(self):
        """Get the coordinates from the last measurement."""
        return self.last_measurement_coordinates
    
    def set_binning(self, binning):
        """Set the binning factor."""
        if binning == self.binning:
            return False
        
        self.binning = binning
        logger.info(f"Binning set to {binning}x{binning}")
        
        # Apply binning if we have an image
        if self.original_image is not None:
            if binning > 1:
                self.apply_binning()
            else:
                # For 1x1 binning, just use the original image
                self.current_image = self.original_image.copy()
                self.binned_image = None
                # Recompute integral images
                self._compute_integral_images()
                # Report completion
                logger.info("Restored original image (1x1 binning)")
                self._report_progress("binning", 100, "Restored original image (1x1 binning)")
        
        return True
    
    def _save_binned_image(self):
        """Save the binned image to a temporary file."""
        if self.binned_image is None:
            return
        
        try:
            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = os.path.join(self.temp_dir, f"binned_{timestamp}.pkl")
            
            # Save the binned image data
            with open(filename, "wb") as f:
                pickle.dump(self.binned_image, f)
            
            # Store the filename and track it for cleanup
            self.binned_image_file = filename
            self._temp_files.append(filename)
            
            logger.debug(f"Saved binned image to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving binned image: {str(e)}", exc_info=True)
            return False
    
    def get_zoom(self):
        """Get the current zoom level."""
        return self.zoom

    # ------------------------------------------------------------------
    # Metadata helpers expected by external plugins (e.g. auto_measurements)
    # ------------------------------------------------------------------
    def get_all_metadata(self):
        """Return a dictionary with all metadata that can be extracted from the current image file.

        The *auto_measurements* plugin expects this method to exist. We try a
        best-effort extraction of metadata using Pillow so that the plugin can
        access DPI information and any EXIF fields available. If no image is
        loaded or metadata cannot be retrieved, an empty dict is returned so
        that the caller can safely proceed to other fallback methods.
        """
        if self.current_file is None or not os.path.exists(self.current_file):
            return {}

        try:
            from PIL import Image, ExifTags

            metadata: dict[str, object] = {}
            with Image.open(self.current_file) as img:
                # Basic info dictionary (may include dpi amongst others)
                if img.info:
                    metadata.update({f"PIL_{k}": v for k, v in img.info.items()})

                # Explicitly store DPI if available (convenience key)
                if "dpi" in img.info:
                    metadata["PIL_dpi_direct"] = img.info["dpi"]

                # Try to read EXIF data (JPEG/TIFF mainly)
                exif = img.getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        metadata[f"EXIF_{tag}"] = value
            return metadata
        except Exception as e:
            logger.warning(f"Could not extract metadata from image: {e}")
            return {}

    def get_dpi(self):
        """Return the DPI (dots-per-inch) of the currently loaded image if available.

        Several fallback strategies are attempted:
        1. Use Pillow's *info["dpi"]* entry (most common for PNG/TIFF).
        2. Inspect EXIF XResolution/YResolution tags.
        3. Resort to the configured default DPI (if set) or *None*.
        """
        if self.current_file is None or not os.path.exists(self.current_file):
            return None

        try:
            from PIL import Image, ExifTags

            with Image.open(self.current_file) as img:
                # 1. Direct dpi tuple (horizontal, vertical)
                dpi_tuple = img.info.get("dpi")
                if dpi_tuple and dpi_tuple[0] > 0:
                    return float(dpi_tuple[0])  # assume square pixels → use X dpi

                # 2. EXIF tags
                exif = img.getexif()
                if exif:
                    x_res_tag = None
                    y_res_tag = None
                    for tag_id, name in ExifTags.TAGS.items():
                        if name == "XResolution":
                            x_res_tag = tag_id
                        elif name == "YResolution":
                            y_res_tag = tag_id
                    if x_res_tag in exif and exif[x_res_tag]:
                        # EXIF resolutions can be (num, den) tuples – convert to float
                        x_val = exif[x_res_tag]
                        if isinstance(x_val, tuple) and len(x_val) == 2 and x_val[1] != 0:
                            return float(x_val[0]) / float(x_val[1])
                        elif isinstance(x_val, (int, float)):
                            return float(x_val)
                # 3. Config default
                default_dpi = self.config.get("default_dpi")
                return float(default_dpi) if default_dpi else None
        except Exception as e:
            logger.warning(f"Could not determine DPI for image: {e}")
            return None

    # ------------------------------------------------------------------
    # Path aliases for backward compatibility with older plugins
    # ------------------------------------------------------------------
    @property
    def original_image_path(self):
        """Alias for the path to the originally loaded image file."""
        return self.current_file

    @property
    def image_path(self):
        """Alias preserved for backward compatibility (same as original_image_path)."""
        return self.current_file

    @property
    def current_file_path(self):
        """Another alias for external plugins that expect this attribute."""
        return self.current_file
    
    def set_zoom(self, zoom):
        """Set the zoom level."""
        self.zoom = max(0.1, min(10.0, zoom))
        return True
    
    def zoom_at_point(self, x, y, delta):
        """Zoom at a specific point."""
        # Calculate zoom factor
        factor = 1.1 if delta > 0 else 0.9
        
        # Calculate new zoom
        new_zoom = self.zoom * factor
        
        # Limit zoom range
        new_zoom = max(0.1, min(10.0, new_zoom))
        
        # Check if zoom actually changed
        if abs(new_zoom - self.zoom) < 0.001:
            return False
        
        # Update zoom
        self.zoom = new_zoom
        
        return True
    
    def fit_to_screen(self, canvas_width, canvas_height):
        """Fit the image to the screen."""
        if self.current_image is None:
            return False
        
        # Get image dimensions
        img_height, img_width = self.current_image.shape[:2]
        
        # Calculate zoom to fit
        zoom_x = canvas_width / img_width
        zoom_y = canvas_height / img_width
        
        # Use the smaller zoom to ensure the entire image fits
        self.zoom = min(zoom_x, zoom_y) * 0.95  # 5% margin
        
        return True
    
    def get_pixel_info(self, canvas_x, canvas_y):
        """Get pixel information at the specified canvas position."""
        if self.current_image is None:
            return None, None, None
        
        try:
            # Calculate position in the current image
            img_x = int(canvas_x / self.zoom)
            img_y = int(canvas_y / self.zoom)
            
            # Check if position is within image bounds
            if (0 <= img_x < self.current_image.shape[1] and 
                0 <= img_y < self.current_image.shape[0]):
                
                # Get pixel value
                if len(self.current_image.shape) == 3:
                    # Color image
                    if self.calibration_applied:
                        rgb = tuple(map(float, self.current_image[img_y, img_x]))
                    else:
                        rgb = tuple(map(int, self.current_image[img_y, img_x]))
                    
                    # If we have binned data with standard deviation, include it
                    if self.binned_image is not None and 'std_dev' in self.binned_image:
                        std_dev = tuple(map(float, self.binned_image['std_dev'][img_y, img_x]))
                        return img_x, img_y, (rgb, std_dev)
                    
                    return img_x, img_y, rgb
                else:
                    # Grayscale image
                    if self.calibration_applied:
                        value = float(self.current_image[img_y, img_x])
                    else:
                        value = int(self.current_image[img_y, img_x])
                    
                    # If we have binned data with standard deviation, include it
                    if self.binned_image is not None and 'std_dev' in self.binned_image:
                        std_dev = float(self.binned_image['std_dev'][img_y, img_x])
                        return img_x, img_y, (value, std_dev)
                    
                    return img_x, img_y, value
            
            return None, None, None
        except Exception as e:
            logger.error(f"Error getting pixel info: {str(e)}", exc_info=True)
            return None, None, None
    
    def _calculate_stats_from_integral(self, x1, y1, x2, y2, channel=None):
        """Calculate statistics from integral image for a rectangular region."""
        if self.integral_images is None or self.integral_images_squared is None:
            return None, None
        
        # Ensure coordinates are within bounds
        height, width = self.current_image.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # Ensure x1 <= x2 and y1 <= y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Calculate area
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        if area <= 0:
            return 0, 0
        
        # For color images
        if len(self.current_image.shape) == 3 and channel is not None:
            # Get integral images for this channel
            integral = self.integral_images[channel]
            integral_squared = self.integral_images_squared[channel]
            
            # Calculate sum using integral image
            # Note: OpenCV's integral images have an extra row and column
            sum_val = (integral[y2+1, x2+1] - integral[y2+1, x1] - 
                      integral[y1, x2+1] + integral[y1, x1])
            
            # Calculate sum of squares using integral image
            sum_sq = (integral_squared[y2+1, x2+1] - integral_squared[y2+1, x1] - 
                     integral_squared[y1, x2+1] + integral_squared[y1, x1])
            
            # Calculate mean and standard deviation
            mean = sum_val / area
            variance = (sum_sq / area) - (mean * mean)
            std_dev = np.sqrt(max(0, variance))  # Avoid negative values due to precision
            
            return mean, std_dev
        
        # For grayscale images
        elif len(self.current_image.shape) == 2:
            # Calculate sum using integral image
            sum_val = (self.integral_images[y2+1, x2+1] - self.integral_images[y2+1, x1] - 
                      self.integral_images[y1, x2+1] + self.integral_images[y1, x1])
            
            # Calculate sum of squares using integral image
            sum_sq = (self.integral_images_squared[y2+1, x2+1] - self.integral_images_squared[y2+1, x1] - 
                     self.integral_images_squared[y1, x2+1] + self.integral_images_squared[y1, x1])
            
            # Calculate mean and standard deviation
            mean = sum_val / area
            variance = (sum_sq / area) - (mean * mean)
            std_dev = np.sqrt(max(0, variance))  # Avoid negative values due to precision
            
            return mean, std_dev
    
    def _bresenham_line(self, x0, y0, x1, y1):
        """
        Generate coordinates of pixels along a line using Bresenham's algorithm.
        
        Args:
            x0, y0: Starting point coordinates
            x1, y1: Ending point coordinates
            
        Returns:
            list: List of (x, y) tuples representing pixels along the line
        """
        coordinates = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            coordinates.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return coordinates
    
    def _calculate_combined_uncertainty(self, means, std_errors):
        """
        Calculate combined uncertainty using the configured estimation method.
        
        Args:
            means: Array of mean values for each channel
            std_errors: Array of standard errors for each channel
            
        Returns:
            tuple: (combined_mean, combined_uncertainty)
        """
        method = self.config.get("uncertainty_estimation_method", "weighted_average")
        
        if method == "birge_factor":
            return self._birge_factor_method(means, std_errors)
        elif method == "dersimonian_laird":
            return self._dersimonian_laird_method(means, std_errors)
        else:  # Default: weighted_average
            return self._weighted_average_method(means, std_errors)
    
    def _weighted_average_method(self, means, std_errors):
        """
        Calculate combined uncertainty using weighted average (original method).
        
        Args:
            means: Array of mean values for each channel
            std_errors: Array of standard errors for each channel
            
        Returns:
            tuple: (combined_mean, combined_uncertainty)
        """
        # Avoid division by zero
        valid_indices = std_errors > 0
        if not np.any(valid_indices):
            return np.mean(means), 0.0
        
        valid_means = means[valid_indices]
        valid_std_errors = std_errors[valid_indices]
        
        weights = 1 / (valid_std_errors**2)
        combined_mean = np.average(valid_means, weights=weights)
        combined_uncertainty = 1 / np.sqrt(np.sum(weights))
        
        return combined_mean, combined_uncertainty
    
    def _birge_factor_method(self, means, std_errors):
        """
        Calculate combined uncertainty using Birge factor method.
        
        The Birge factor inflates uncertainties globally when χ²_ν > 1.
        
        Args:
            means: Array of mean values for each channel
            std_errors: Array of standard errors for each channel
            
        Returns:
            tuple: (combined_mean, combined_uncertainty)
        """
        # Avoid division by zero
        valid_indices = std_errors > 0
        if not np.any(valid_indices):
            return np.mean(means), 0.0
        
        valid_means = means[valid_indices]
        valid_std_errors = std_errors[valid_indices]
        n = len(valid_means)
        
        if n <= 1:
            return valid_means[0] if n == 1 else 0.0, valid_std_errors[0] if n == 1 else 0.0
        
        # Calculate initial weighted average
        weights = 1 / (valid_std_errors**2)
        weighted_mean = np.average(valid_means, weights=weights)
        
        # Calculate χ² statistic
        chi_squared = np.sum(weights * (valid_means - weighted_mean)**2)
        chi_squared_nu = chi_squared / (n - 1)  # Reduced chi-squared
        
        # Apply Birge factor if χ²_ν > 1
        if chi_squared_nu > 1:
            birge_factor = np.sqrt(chi_squared_nu)
            inflated_std_errors = birge_factor * valid_std_errors
            
            # Recalculate with inflated uncertainties
            new_weights = 1 / (inflated_std_errors**2)
            combined_mean = np.average(valid_means, weights=new_weights)
            combined_uncertainty = 1 / np.sqrt(np.sum(new_weights))
        else:
            # Use original weighted average
            combined_mean = weighted_mean
            combined_uncertainty = 1 / np.sqrt(np.sum(weights))
        
        return combined_mean, combined_uncertainty
    
    def _dersimonian_laird_method(self, means, std_errors):
        """
        Calculate combined uncertainty using DerSimonian-Laird random effects model.
        
        This method incorporates between-channel heterogeneity via τ².
        
        Args:
            means: Array of mean values for each channel
            std_errors: Array of standard errors for each channel
            
        Returns:
            tuple: (combined_mean, combined_uncertainty)
        """
        # Avoid division by zero
        valid_indices = std_errors > 0
        if not np.any(valid_indices):
            return np.mean(means), 0.0
        
        valid_means = means[valid_indices]
        valid_std_errors = std_errors[valid_indices]
        n = len(valid_means)
        
        if n <= 1:
            return valid_means[0] if n == 1 else 0.0, valid_std_errors[0] if n == 1 else 0.0
        
        # Initial weights
        w_i = 1 / (valid_std_errors**2)
        
        # Calculate fixed effects estimate
        D_fixed = np.sum(w_i * valid_means) / np.sum(w_i)
        
        # Calculate Q statistic 
        Q = np.sum(w_i * (valid_means - D_fixed)**2)
        
        # Estimate between-channel variance τ²
        sum_w_i = np.sum(w_i)
        sum_w_i_squared = np.sum(w_i**2)
        
        denominator = sum_w_i - (sum_w_i_squared / sum_w_i)
        if denominator > 0:
            tau_squared = max(0, (Q - (n - 1)) / denominator)
        else:
            tau_squared = 0
        
        # New weights incorporating τ²
        w_i_star = 1 / (valid_std_errors**2 + tau_squared)
        
        # Final combined estimate
        combined_mean = np.sum(w_i_star * valid_means) / np.sum(w_i_star)
        combined_uncertainty = 1 / np.sqrt(np.sum(w_i_star))
        
        return combined_mean, combined_uncertainty

    def measure_area(self, canvas_x, canvas_y):
        """Measure the area at the specified canvas position."""
        if self.current_image is None:
            return None

        try:
            # Calculate position in the current image
            img_x = int(canvas_x / self.zoom)
            img_y = int(canvas_y / self.zoom)
        
            # Check if position is within image bounds
            if (0 <= img_x < self.current_image.shape[1] and
                0 <= img_y < self.current_image.shape[0]):
        
                actual_size = self.measurement_size
        
                # Create mask for the region of interest
                if self.measurement_shape == "circular":
                    # Create circular mask using OpenCV
                    mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
                    cv2.circle(mask, (img_x, img_y), actual_size, 255, -1)
                
                    # Get coordinates of all pixels in the mask
                    y_coords, x_coords = np.where(mask > 0)
                
                    # Store coordinates relative to the center of the measurement
                    rel_coords = np.column_stack((x_coords - img_x, y_coords - img_y))
                
                    # Count pixels in the mask
                    pixel_count = len(x_coords)
                
                    # For circular regions, we can't use integral images directly
                    # So we'll use the traditional approach
                    if len(self.current_image.shape) == 3:  # Color image
                        # Use OpenCV's built-in functions for faster processing
                        means, std_devs = cv2.meanStdDev(self.current_image, mask=mask)
                        means = means.flatten()
                        std_devs = std_devs.flatten()
                        std_err = std_devs / np.sqrt(pixel_count) if pixel_count > 0 else np.zeros_like(std_devs)
                        
                        # Calculate combined uncertainty using the configured method
                        rgb_mean, rgb_mean_std = self._calculate_combined_uncertainty(means, std_err)
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
                    
                        # Collect raw data for histogram (preserve original dtype for 16-bit support)
                        masked_data = np.zeros((len(sample_x), self.current_image.shape[2]), dtype=self.current_image.dtype)
                        for c in range(self.current_image.shape[2]):
                            masked_data[:, c] = np.array([self.current_image[x, y, c] for x, y in zip(sample_y, sample_x)])
                    
                        # Store the raw data and coordinates for later use
                        self.last_measurement_raw_data = masked_data
                        self.last_measurement_coordinates = sample_rel
                        self.last_auto_measure_time = time.time()

                        return (
                            tuple(means),
                            tuple(std_devs),
                            tuple(std_err),
                            rgb_mean,
                            rgb_mean_std,
                            pixel_count
                        )
                    else:  # Grayscale image
                        # For grayscale, use a more efficient approach
                        mean, std_dev = cv2.meanStdDev(self.current_image, mask=mask)
                        mean = float(mean[0][0])
                        std_dev = float(std_dev[0][0])
                    
                        # Calculate std_err
                        std_err = std_dev / np.sqrt(pixel_count) if pixel_count > 0 else 0
                    
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
                    
                        # Collect raw data for histogram (preserve original dtype for 16-bit support)
                        masked_data = np.zeros(len(sample_x), dtype=self.current_image.dtype)
                        for i, (x, y) in enumerate(zip(sample_x, sample_y)):
                            masked_data[i] = self.current_image[y, x] # Note: Grayscale indexing was [y,x], seems correct as per typical image coord. conventions if x=col, y=row
                    
                        # Store the raw data and coordinates for later use
                        self.last_measurement_raw_data = masked_data
                        self.last_measurement_coordinates = sample_rel
                        self.last_auto_measure_time = time.time()
                    


                        # For grayscale, we don't need combined uncertainty (only one channel)
                        return mean, std_dev, std_err, mean, std_err, pixel_count

                elif self.measurement_shape == "rectangular":
                    # For rectangular regions, use width and height from measurement_size_rect
                    width, height = self.measurement_size_rect
                    
                    # Calculate region boundaries
                    x1 = max(0, img_x - width // 2)
                    y1 = max(0, img_y - height // 2)
                    x2 = min(self.current_image.shape[1] - 1, img_x + width // 2)
                    y2 = min(self.current_image.shape[0] - 1, img_y + height // 2)
                
                    # Ensure x1 < x2 and y1 < y2
                    if x1 >= x2 or y1 >= y2:
                        logger.warning("Invalid region for rectangular measurement")
                        return None
                
                    pixel_count = (x2 - x1 + 1) * (y2 - y1 + 1)
                
                    if len(self.current_image.shape) == 3:  # Color image
                        means = []
                        std_devs = []
                        std_err = []
                    
                        for c in range(self.current_image.shape[2]):
                            mean, std_dev = self._calculate_stats_from_integral(x1, y1, x2, y2, channel=c)
                            means.append(mean)
                            std_devs.append(std_dev)
                            std_err.append(std_dev / np.sqrt(pixel_count) if pixel_count > 0 else 0)
                    
                        # Calculate combined uncertainty using the configured method
                        means_array = np.array(means)
                        std_err_array = np.array(std_err)
                        rgb_mean, rgb_mean_std = self._calculate_combined_uncertainty(means_array, std_err_array)
                    
                        # Collect raw data for histogram
                        masked_data = self.current_image[y1:y2+1, x1:x2+1, :].reshape(-1, self.current_image.shape[2])
                    
                        # Store the raw data and coordinates for later use
                        self.last_measurement_raw_data = masked_data
                        # For rectangular, relative coordinates are just the grid within the rectangle
                        rel_x, rel_y = np.meshgrid(np.arange(x1 - img_x, x2 - img_x + 1), np.arange(y1 - img_y, y2 - img_y + 1))
                        self.last_measurement_coordinates = np.column_stack((rel_x.ravel(), rel_y.ravel()))
                        self.last_auto_measure_time = time.time()
                    
                        return (
                            tuple(means),
                            tuple(std_devs),
                            tuple(std_err),
                            rgb_mean,
                            rgb_mean_std,
                            pixel_count
                        )
                    else:  # Grayscale image
                        mean, std_dev = self._calculate_stats_from_integral(x1, y1, x2, y2)
                        std_err = std_dev / np.sqrt(pixel_count) if pixel_count > 0 else 0
                    
                        # Collect raw data for histogram
                        masked_data = self.current_image[y1:y2+1, x1:x2+1].ravel()
                    
                        # Store the raw data and coordinates for later use
                        self.last_measurement_raw_data = masked_data
                        rel_x, rel_y = np.meshgrid(np.arange(x1 - img_x, x2 - img_x + 1), np.arange(y1 - img_y, y2 - img_y + 1))
                        self.last_measurement_coordinates = np.column_stack((rel_x.ravel(), rel_y.ravel()))
                        self.last_auto_measure_time = time.time()
                    
                        # For grayscale, we don't need combined uncertainty (only one channel)
                        return mean, std_dev, std_err, mean, std_err, pixel_count
                elif self.measurement_shape == "line":
                    # For line profiles, get pixels along the line based on orientation
                    orientation = getattr(self, 'line_orientation', 'horizontal')
                    
                    if orientation == "horizontal":
                        # Horizontal line - vary X, keep Y fixed
                        # Use full width of image as line length
                        x_start = 0
                        x_end = self.current_image.shape[1] - 1
                        y_fixed = img_y
                        
                        if y_fixed < 0 or y_fixed >= self.current_image.shape[0]:
                            logger.warning("Invalid Y position for horizontal line measurement")
                            return None
                        
                        if len(self.current_image.shape) == 3:  # Color image
                            line_pixels = self.current_image[y_fixed, x_start:x_end+1, :]
                        else:  # Grayscale
                            line_pixels = self.current_image[y_fixed, x_start:x_end+1]
                        
                        # Coordinates are (row, col) pairs for each pixel
                        coordinates = np.column_stack([
                            np.full(x_end - x_start + 1, y_fixed),
                            np.arange(x_start, x_end + 1)
                        ])
                        
                    elif orientation == "vertical":
                        # Vertical line - vary Y, keep X fixed
                        # Use full height of image as line length
                        y_start = 0
                        y_end = self.current_image.shape[0] - 1
                        x_fixed = img_x
                        
                        if x_fixed < 0 or x_fixed >= self.current_image.shape[1]:
                            logger.warning("Invalid X position for vertical line measurement")
                            return None
                        
                        if len(self.current_image.shape) == 3:  # Color image
                            line_pixels = self.current_image[y_start:y_end+1, x_fixed, :]
                        else:  # Grayscale
                            line_pixels = self.current_image[y_start:y_end+1, x_fixed]
                        
                        # Coordinates are (row, col) pairs for each pixel
                        coordinates = np.column_stack([
                            np.arange(y_start, y_end + 1),
                            np.full(y_end - y_start + 1, x_fixed)
                        ])
                        
                    elif orientation == "manual":
                        # Manual line - use Bresenham's algorithm to get pixels between two points
                        if self.manual_line_points is None or len(self.manual_line_points) < 2:
                            logger.warning("Manual line orientation selected but no points defined")
                            return None
                        
                        x1, y1 = self.manual_line_points[0]
                        x2, y2 = self.manual_line_points[1]
                        
                        # Get line pixels using Bresenham's algorithm
                        line_coords = self._bresenham_line(x1, y1, x2, y2)
                        
                        # Filter out coordinates outside image bounds
                        valid_coords = []
                        for x, y in line_coords:
                            if 0 <= y < self.current_image.shape[0] and 0 <= x < self.current_image.shape[1]:
                                valid_coords.append((y, x))  # (row, col)
                        
                        if len(valid_coords) == 0:
                            logger.warning("Manual line is completely outside image bounds")
                            return None
                        
                        coordinates = np.array(valid_coords)
                        
                        # Extract pixel values
                        if len(self.current_image.shape) == 3:  # Color image
                            line_pixels = self.current_image[coordinates[:, 0], coordinates[:, 1], :]
                        else:  # Grayscale
                            line_pixels = self.current_image[coordinates[:, 0], coordinates[:, 1]]
                    
                    else:
                        logger.warning(f"Unknown line orientation: {orientation}")
                        return None
                    
                    pixel_count = len(line_pixels)
                
                    if len(self.current_image.shape) == 3:  # Color image
                        means = np.mean(line_pixels, axis=0)
                        std_devs = np.std(line_pixels, axis=0)
                        std_err = std_devs / np.sqrt(pixel_count) if pixel_count > 0 else np.zeros_like(std_devs)
                    
                        # Calculate combined uncertainty using the configured method
                        rgb_mean, rgb_mean_std = self._calculate_combined_uncertainty(means, std_err)
                    
                        self.last_measurement_raw_data = line_pixels
                        self.last_measurement_coordinates = coordinates
                        self.last_auto_measure_time = time.time()
                    
                        return (
                            tuple(means),
                            tuple(std_devs),
                            tuple(std_err),
                            rgb_mean,
                            rgb_mean_std,
                            pixel_count
                        )
                    else:  # Grayscale image
                        mean = np.mean(line_pixels)
                        std_dev = np.std(line_pixels)
                        std_err = std_dev / np.sqrt(pixel_count) if pixel_count > 0 else 0
                    
                        self.last_measurement_raw_data = line_pixels
                        self.last_measurement_coordinates = coordinates
                        self.last_auto_measure_time = time.time()
                    
                        # For grayscale, we don't need combined uncertainty (only one channel)
                        return mean, std_dev, std_err, mean, std_err, pixel_count
                else:
                    logger.warning(f"Unknown measurement shape: {self.measurement_shape}")
                    return None
            else:
                logger.debug(f"Measurement position ({img_x}, {img_y}) is outside image bounds")
                return None
        except Exception as e:
            logger.error(f"Error measuring area: {str(e)}", exc_info=True)
            return None
    
    def has_image(self):
        """Check if an image is loaded."""
        return self.current_image is not None
    
    def has_calibration(self):
        """Check if calibration is available.

        A calibration is considered present if either:
        1. The deprecated *self.calibration* (linear model) is set, **or**
        2. The *calibration_data/fit_parameters.csv* file exists produced
           by the new nonlinear calibration workflow.
        """
        if self.calibration is not None:
            return True

        return self._find_fit_parameters_file() is not None
    
    def get_calibration(self):
        """Get the current calibration."""
        if self.calibration is None:
            return None
    
        return {
            "factor": self.calibration.factor,
            "offset": self.calibration.offset,
            "date": self.calibration.date
        }
    
    def set_calibration(self, dose, factor, offset):
        """Set the calibration parameters."""
        from app.models.calibration_model import CalibrationModel
    
        self.calibration = CalibrationModel(dose, factor, offset)
        self.calibration.save_to_file()
    
        return True
    
    def apply_calibration(self):
        """Apply nonlinear calibration to the **current_image**.

        The calibration parameters (a, b, c) for the inverse response
        y = a + b / (x - c)   with *y* being the stored pixel value and *x*
        the absorbed dose, are stored in a CSV file named
        ``calibration_data/fit_parameters.csv`` generated by the external
        *Calibration Wizard*.

        The dose is obtained by inverting the model:

            dose = c + b / (pixel - a)

        On success ``self.current_image`` becomes a **float32 single-channel**
        array containing the dose for each pixel (NaN where the inversion is
        undefined).  Integral images are recomputed to keep measurement logic
        working.
        
        NOTE: Field flattening is now applied independently via apply_flat().
        Call apply_flat() before apply_calibration() if both are desired.
        """
        # Require an image to work on.
        if self.current_image is None:
            logger.warning("apply_calibration called with no image loaded")
            return False

        # Work on current_image (may already have flat applied)
        image_to_calibrate = self.current_image.copy()

        # ------------------------------------------------------------------
        # Locate *fit_parameters.csv* written by the calibration wizard
        # ------------------------------------------------------------------
        csv_path = self._find_fit_parameters_file()

        if csv_path is None:
            logger.error("Calibration parameters file 'fit_parameters.csv' not found in calibration_data directory")
            return False

        # Import here to avoid top-level dependency cycles
        import csv

        # ------------------------------------------------------------------
        # Parse CSV and retrieve (a, b, c) and their sigmas for each RGB channel
        # ------------------------------------------------------------------
        params: dict[str, tuple[float, float, float]] = {}
        param_sigmas: dict[str, tuple[float, float, float]] = {}
        calibration_bit_depth_from_file = None
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ch = row.get("Channel", "").strip().upper()
                    if ch not in ("R", "G", "B"):
                        continue
                    try:
                        a = float(row.get("a", "nan"))
                        b = float(row.get("b", "nan"))
                        c = float(row.get("c", "nan"))
                    except ValueError:
                        continue
                    params[ch] = (a, b, c)

                    # Read calibration bit depth if present
                    if calibration_bit_depth_from_file is None:
                        bit_depth_str = row.get("bit_depth")
                        if bit_depth_str:
                            try:
                                calibration_bit_depth_from_file = int(bit_depth_str)
                            except ValueError:
                                pass

                    # Optional std_err columns – tolerate multiple naming conventions
                    try:
                        sa = float(row.get("sigma_a") or row.get("sa") or row.get("σa") or 0.0)
                        sb = float(row.get("sigma_b") or row.get("sb") or row.get("σb") or 0.0)
                        sc = float(row.get("sigma_c") or row.get("sc") or row.get("σc") or 0.0)
                    except ValueError:
                        sa = sb = sc = 0.0
                    param_sigmas[ch] = (sa, sb, sc)
            
            # Update calibration bit depth
            if calibration_bit_depth_from_file:
                self.calibration_bit_depth = calibration_bit_depth_from_file
                logger.info(f"Calibration bit depth from file: {self.calibration_bit_depth}-bit")
            else:
                self.calibration_bit_depth = 8  # Legacy calibration
                logger.info("No bit_depth in calibration file, assuming 8-bit")
                
        except Exception as exc:
            logger.error("Failed to read calibration parameters: %s", exc, exc_info=True)
            return False

        # Ensure we have the three colour channels
        if not all(ch in params for ch in ("R", "G", "B")):
            logger.error("Incomplete calibration parameters – need R, G and B channels")
            return False

        # ------------------------------------------------------------------
        # Apply inverse calibration and compute std_err propagation
        # ------------------------------------------------------------------
        try:
            # Work on ORIGINAL pixel values (uint8) for derivative computation
            if self.original_image is None:
                orig_img = self.current_image.copy()
            else:
                orig_img = self.original_image
            img_uint = orig_img.astype(np.float32)

            if img_uint.ndim == 2:
                # Single-channel image – treat as green channel
                a, b, c = params["G"]
                sa, sb, sc = param_sigmas.get("G", (0.0, 0.0, 0.0))
                denom = img_uint - a
                dose = np.full_like(img_uint, np.nan, dtype=np.float32)
                
                # Valid domain: denom != 0 AND resulting dose must be physically meaningful
                # For inverse model dose = c + b/denom, we need denom to have same sign as b
                # to get positive contribution, and final dose should be >= 0
                valid = denom != 0
                dose[valid] = c + b / denom[valid]
                
                # Mark negative doses as invalid (physically impossible)
                invalid_dose = dose < 0
                if np.any(invalid_dose):
                    n_invalid = np.sum(invalid_dose)
                    logger.warning(f"G channel: {n_invalid} pixels ({100*n_invalid/dose.size:.2f}%) have negative dose, marking as NaN")
                    dose[invalid_dose] = np.nan

                # std_err per-pixel
                var = np.full_like(img_uint, np.nan, dtype=np.float32)
                valid_final = ~np.isnan(dose)
                if np.any(valid_final):
                    denom_valid = denom[valid_final]
                    term_a = (-b / (denom_valid ** 2)) ** 2 * sa ** 2
                    term_b = (1.0 / denom_valid) ** 2 * sb ** 2
                    term_c = sc ** 2
                    var[valid_final] = term_a + term_b + term_c
                sigma = np.sqrt(var)

                # Store helper arrays
                self.dose_channels = np.expand_dims(dose, axis=-1)
                self.dose_std_err_channels = np.expand_dims(sigma, axis=-1)
                dose_for_integral = dose  # single-channel
            else:
                # Colour image – compute dose and variance per channel
                h, w, _ = img_uint.shape
                doses = []
                vars_ = []
                for idx, ch in enumerate(("R", "G", "B")):
                    a, b, c = params[ch]
                    sa, sb, sc = param_sigmas.get(ch, (0.0, 0.0, 0.0))
                    pix = img_uint[:, :, idx]
                    denom = pix - a
                    
                    dose_ch = np.full((h, w), np.nan, dtype=np.float32)
                    var_ch = np.full((h, w), np.nan, dtype=np.float32)
                    
                    # Valid domain check
                    valid = denom != 0
                    dose_ch[valid] = c + b / denom[valid]
                    
                    # Mark negative doses as invalid (physically impossible)
                    invalid_dose = dose_ch < 0
                    if np.any(invalid_dose):
                        n_invalid = np.sum(invalid_dose)
                        logger.warning(f"{ch} channel: {n_invalid} pixels ({100*n_invalid/dose_ch.size:.2f}%) have negative dose, marking as NaN")
                        dose_ch[invalid_dose] = np.nan
                    
                    # Variance via error propagation (only for valid pixels)
                    valid_final = ~np.isnan(dose_ch)
                    if np.any(valid_final):
                        denom_valid = denom[valid_final]
                        term_a = (-b / (denom_valid ** 2)) ** 2 * sa ** 2
                        term_b = (1.0 / denom_valid) ** 2 * sb ** 2
                        term_c = sc ** 2
                        var_ch[valid_final] = term_a + term_b + term_c

                    doses.append(dose_ch)
                    vars_.append(var_ch)

                dose_stack = np.stack(doses, axis=-1)  # HxWx3
                var_stack = np.stack(vars_, axis=-1)

                self.dose_channels = dose_stack
                self.dose_std_err_channels = np.sqrt(var_stack)
                dose_for_integral = np.nanmean(dose_stack, axis=-1)  # averaged dose for integral images

            # Replace current_image with 3-channel dose to keep RGB workflow alive
            if self.dose_channels.ndim == 2:
                self.current_image = np.expand_dims(self.dose_channels, axis=-1)
            else:
                self.current_image = self.dose_channels.astype(np.float32)

            # Recompute integral images on average dose to speed up statistics
            self._compute_integral_images()
            self.calibration_applied = True
            self.calibration_param_sigmas = param_sigmas  # store for later use
            logger.info("Calibration applied successfully – dose image ready (3-channel float32)")
            return True

        except Exception as exc:
            logger.error("Error applying calibration: %s", exc, exc_info=True)
            return False


    
    def get_image_bit_depth(self):
        """Return the bit depth of the current image (8 or 16)."""
        return self.image_bit_depth
    
    def get_calibration_bit_depth(self):
        """Return the bit depth expected by the calibration curve.
        
        This is read from the fit_parameters.csv file. If not found,
        defaults to 8-bit for backwards compatibility with older calibrations.
        
        Always reads from file to get the most up-to-date value,
        since the user may have created a new calibration during the session.
        """
        # Always reload from file to ensure we have the current calibration
        self._load_calibration_bit_depth()
        return self.calibration_bit_depth if self.calibration_bit_depth else 8
    
    def _load_calibration_bit_depth(self):
        """Load calibration bit depth from fit_parameters.csv."""
        csv_path = self._find_fit_parameters_file()
        if csv_path is None:
            self.calibration_bit_depth = 8
            return
        
        try:
            import csv
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bit_depth = row.get("bit_depth")
                    if bit_depth:
                        self.calibration_bit_depth = int(bit_depth)
                        logger.info(f"Loaded calibration bit depth: {self.calibration_bit_depth}-bit")
                        return
            # If bit_depth column not found, assume old calibration (8-bit)
            self.calibration_bit_depth = 8
            logger.info("No bit_depth in calibration file, assuming 8-bit (legacy calibration)")
        except Exception as e:
            logger.warning(f"Could not read calibration bit depth: {e}")
            self.calibration_bit_depth = 8
    
    def _detect_bit_depth(self, image):
        """Detect the actual bit depth of an image.
        
        Determines bit depth based on dtype and actual maximum value.
        Supports 8, 10, 12, 14, 16, 24, and 32-bit images.
        
        Args:
            image: numpy array of the image
            
        Returns:
            tuple: (bit_depth, max_possible_value)
        """
        dtype = image.dtype
        actual_max = np.max(image)
        
        # Map dtypes to their theoretical bit depths
        dtype_bits = {
            np.uint8: 8,
            np.uint16: 16,
            np.uint32: 32,
            np.int8: 8,
            np.int16: 16,
            np.int32: 32,
            np.float32: 32,
            np.float64: 64,
        }
        
        # Get theoretical max for dtype
        theoretical_bits = dtype_bits.get(dtype.type, 8)
        
        if dtype in [np.float32, np.float64]:
            # For float images, assume normalized [0, 1] or actual range
            if actual_max <= 1.0:
                return 8, 1.0  # Normalized float
            else:
                # Determine based on actual max value
                if actual_max <= 255:
                    return 8, 255
                elif actual_max <= 4095:
                    return 12, 4095
                elif actual_max <= 16383:
                    return 14, 16383
                elif actual_max <= 65535:
                    return 16, 65535
                else:
                    return 32, actual_max
        
        # For integer types, try to detect actual bit depth from values
        if theoretical_bits == 16:
            # Could be 10, 12, 14, or 16-bit stored in uint16
            if actual_max <= 1023:
                return 10, 1023
            elif actual_max <= 4095:
                return 12, 4095
            elif actual_max <= 16383:
                return 14, 16383
            else:
                return 16, 65535
        elif theoretical_bits == 32:
            # Could be various depths stored in uint32
            if actual_max <= 255:
                return 8, 255
            elif actual_max <= 65535:
                return 16, 65535
            elif actual_max <= 16777215:
                return 24, 16777215
            else:
                return 32, 4294967295
        else:
            # 8-bit
            return 8, 255
    
    def get_max_pixel_value(self):
        """Return the maximum possible pixel value for the current image bit depth."""
        return self.image_max_value
    
    def rescale_to_bit_depth(self, target_bit_depth):
        """Rescale the image to a target bit depth.
        
        This performs linear scaling from the current bit depth range to
        the target bit depth range.
        
        Args:
            target_bit_depth: Target bit depth (8, 10, 12, 14, 16, etc.)
            
        Returns:
            bool: True if rescaling was successful, False otherwise.
        """
        if self.current_image is None:
            logger.warning("rescale_to_bit_depth called with no image loaded")
            return False
        
        if self.image_bit_depth == target_bit_depth:
            logger.info(f"Image is already {target_bit_depth}-bit, no rescaling needed")
            return True
        
        # Calculate max values for source and target
        source_max = self.image_max_value
        target_max = (2 ** target_bit_depth) - 1
        
        # Determine target dtype
        if target_bit_depth <= 8:
            target_dtype = np.uint8
        elif target_bit_depth <= 16:
            target_dtype = np.uint16
        else:
            target_dtype = np.uint32
        
        try:
            # Linear rescaling: new_value = old_value * (target_max / source_max)
            scale_factor = target_max / source_max
            
            self.original_image = (self.original_image.astype(np.float64) * scale_factor).astype(target_dtype)
            self.current_image = (self.current_image.astype(np.float64) * scale_factor).astype(target_dtype)
            
            old_bit_depth = self.image_bit_depth
            self.image_bit_depth = target_bit_depth
            self.image_max_value = target_max
            
            # Recompute integral images with new values
            self._compute_integral_images()
            
            logger.info(f"Image rescaled from {old_bit_depth}-bit to {target_bit_depth}-bit successfully")
            return True
        except Exception as e:
            logger.error(f"Error rescaling image to {target_bit_depth}-bit: {e}", exc_info=True)
            return False
    
    def rescale_to_8bit(self):
        """Rescale image to 8-bit. Convenience wrapper for rescale_to_bit_depth."""
        return self.rescale_to_bit_depth(8)
    
    def rescale_to_16bit(self):
        """Rescale image to 16-bit. Convenience wrapper for rescale_to_bit_depth."""
        return self.rescale_to_bit_depth(16)
    
    def update_settings(self, config):
        """Update settings from config."""
        # Check if calibration folder changed
        old_cal_folder = self.config.get("calibration_folder", "") if self.config else ""
        new_cal_folder = config.get("calibration_folder", "")
        cal_folder_changed = old_cal_folder != new_cal_folder
        
        self.config = config
    
        # Update settings
        self.negative_mode = config.get("negative_mode", False)
        self.auto_measure = config.get("auto_measure", False)
    
        # Update GPU settings
        self.gpu_force_enabled = config.get("gpu_force_enabled", False)
        if self.cuda_available:
            self._init_gpu()
        else:
            logger.warning("CUDA not available, skipping GPU initialization")
        self.gpu_enabled = config.get("use_gpu", False) and (self.gpu_available or self.gpu_force_enabled)
    
        # Update threading settings
        self.use_multithreading = config.get("use_multithreading", True)
        self.num_threads = config.get("num_threads", os.cpu_count() or 4)
    
        # If calibration folder changed, reload calibration data
        if cal_folder_changed:
            logger.info(f"Calibration folder changed from '{old_cal_folder}' to '{new_cal_folder}', reloading data")
            # Reset flat field and calibration state
            self.flat_field = None
            self.flat_field_info = None
            self.calibration = None
            self.calibration_applied = False
            self.flat_applied = False
            self.flattened_image = None
            # Try to load new flat field (calibration is loaded on demand)
            self.load_field_flattening()
    
        logger.info("Settings updated")
        return True
    
    def reprocess_current_image(self, skip_integral_compute: bool = False):
        """Reprocess the current image with updated settings.
        
        Args:
            skip_integral_compute: If True, skip computing integral images.
                Use this when calling multiple processing steps in sequence
                and computing integrals only at the end for better performance.
        """
        if self.original_image is None:
            return False
    
        # Reset the current image to the original
        self.current_image = self.original_image.copy()
        # Calibration no longer applied
        self.calibration_applied = False
        self.flat_applied = False
        self.flattened_image = None
    
        # Apply binning if needed
        if self.binning > 1:
            self.apply_binning()
        elif not skip_integral_compute:
            # Recompute integral images
            self._compute_integral_images()
    
        return True
    
    def get_display_image(self):
        """Get the image for display with current settings applied."""
        if self.current_image is None:
            return None, 0, 0
    
        try:
            # Apply display settings
            # When dose calibration is active we still want to show an RGB
            # image to the user so that the visual appearance remains unchanged
            # while all internal calculations work on the calibrated dose image.
            # If flat was also applied, show the flattened RGB; otherwise show original.
            if self.calibration_applied and self.original_image is not None:
                if self.flat_applied and self.flattened_image is not None:
                    image = self.flattened_image.copy()
                else:
                    image = self.original_image.copy()
            else:
                image = self.current_image.copy()
        
            # Apply negative mode
            if self.negative_mode:
                image = self.image_max_value - image
        
            # Apply contrast, brightness, and saturation
            if self.contrast != 1.0 or self.brightness != 1.0 or self.saturation != 1.0:
                # Convert to PIL Image for easier adjustments
                pil_ready = self._prepare_image_for_display(image)
                if len(pil_ready.shape) == 3:
                    pil_image = Image.fromarray(pil_ready)
                else:
                    pil_image = Image.fromarray(pil_ready).convert("L")
            
                # Apply contrast
                if self.contrast != 1.0:
                    enhancer = ImageEnhance.Contrast(pil_image)
                    pil_image = enhancer.enhance(self.contrast)
            
                # Apply brightness
                if self.brightness != 1.0:
                    enhancer = ImageEnhance.Brightness(pil_image)
                    pil_image = enhancer.enhance(self.brightness)
            
                # Apply saturation (only for color images)
                if self.saturation != 1.0 and len(image.shape) == 3:
                    enhancer = ImageEnhance.Color(pil_image)
                    pil_image = enhancer.enhance(self.saturation)
            
                # Convert back to numpy array
                image = np.array(pil_image)
        
            # Apply zoom
            if self.zoom != 1.0:
                # Calculate new dimensions
                height, width = image.shape[:2]
                new_height = int(height * self.zoom)
                new_width = int(width * self.zoom)
            
                # Use GPU for resizing if available
                if self.cuda_available:
                    try:
                        # Upload image to GPU
                        gpu_image = cv2.cuda_GpuMat()
                        gpu_image.upload(image)
                    
                        # Resize on GPU
                        resized = cv2.cuda.resize(gpu_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    
                        # Download result
                        image = resized.download()
                    except Exception as e:
                        logger.error(f"GPU resizing error: {str(e)}, falling back to CPU", exc_info=True)
                        # Fall back to CPU
                        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                else:
                    # CPU resizing
                    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
            # Apply user plugins (if any)
            from app.plugins.plugin_manager import plugin_manager
            image = plugin_manager.apply_plugins(image)
            image = self._prepare_image_for_display(image)

            # Convert to PIL Image for display
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image).convert("L")

            # Convert to PhotoImage for Tkinter
            image_tk = ImageTk.PhotoImage(pil_image)
        
            # Return the image and its dimensions
            return image_tk, pil_image.width, pil_image.height
        except Exception as e:
            logger.error(f"Error preparing display image: {str(e)}", exc_info=True)
            return None, 0, 0
    
    def cleanup(self):
        """Clean up resources."""
        # Clean up any temporary files
        for file_path in self._temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {file_path}: {str(e)}")
        
        # Clear the list
        self._temp_files = []
    
        # Clean up any other temporary files
        try:
            for file in glob.glob(os.path.join(self.temp_dir, "tile_*.npy")):
                try:
                    os.remove(file)
                    logger.debug(f"Removed temporary tile file: {file}")
                except Exception as e:
                    logger.warning(f"Could not remove temporary tile file: {str(e)}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_image_for_display(self, image):
        """Convert numpy arrays to uint8 for safe Pillow/Tk display.
        
        Handles any bit depth by scaling to 8-bit range using the actual
        maximum value of the image for proper display.
        """
        if image is None:
            return None

        np_image = np.asarray(image)

        # Common case: already uint8
        if np_image.dtype == np.uint8:
            return np_image

        # Boolean masks become 0/255
        if np.issubdtype(np_image.dtype, np.bool_):
            return np_image.astype(np.uint8) * 255

        # For integer images (16-bit, 32-bit, etc.), scale using the known max value
        if np.issubdtype(np_image.dtype, np.integer):
            # Use the image_max_value if available, otherwise calculate from dtype
            if hasattr(self, 'image_max_value') and self.image_max_value > 0:
                max_val = self.image_max_value
            else:
                # Fallback: use theoretical max for dtype
                if np_image.dtype == np.uint16:
                    max_val = 65535
                elif np_image.dtype == np.int16:
                    # Shift signed to unsigned range
                    np_image = np_image.astype(np.int32) + 32768
                    max_val = 65535
                elif np_image.dtype == np.uint32:
                    max_val = np.max(np_image) if np.max(np_image) > 0 else 1
                else:
                    max_val = np.max(np_image) if np.max(np_image) > 0 else 255
            
            # Scale to 8-bit
            scale_factor = 255.0 / max_val
            return (np_image * scale_factor).astype(np.uint8)

        # Float images: assume 0-1 range or normalize if outside
        if np.issubdtype(np_image.dtype, np.floating):
            float_image = np.nan_to_num(np_image, nan=0.0, posinf=1.0, neginf=0.0)
            max_val = float_image.max()
            # If values are in 0-1 range, scale directly
            if max_val <= 1.0:
                return (np.clip(float_image, 0.0, 1.0) * 255.0).astype(np.uint8)
            # Otherwise normalize
            min_val = float_image.min()
            if max_val == min_val:
                return np.zeros_like(float_image, dtype=np.uint8)
            scaled = (float_image - min_val) / (max_val - min_val)
            return (scaled * 255.0).astype(np.uint8)

        # Other types: clip to 0-255
        return np.clip(np_image, 0, 255).astype(np.uint8)

    def _find_fit_parameters_file(self):
        """Return absolute path to *fit_parameters.csv* if it exists, else None.
        
        When a specific calibration folder is selected (not 'default'),
        search ONLY in that folder - no fallback to root calibration_data.
        """
        calibration_folder = self.config.get("calibration_folder", "default")
        
        if calibration_folder != "default":
            # Strict search: only in the selected subfolder
            candidate_dirs = [
                os.path.join(os.getcwd(), "calibration_data", calibration_folder),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "calibration_data", calibration_folder),
            ]
        else:
            # Default: search in root calibration_data folder
            candidate_dirs = [
                os.path.join(os.getcwd(), "calibration_data"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "calibration_data"),
            ]

        for d in candidate_dirs:
            path = os.path.join(d, "fit_parameters.csv")
            if os.path.isfile(path):
                logger.info(f"Using calibration file: {path}")
                return path

        return None

    # =========================================================================
    # Field Flattening Methods
    # =========================================================================

    def _find_field_flattening_file(self):
        """Return absolute path to *field_flattening.npz* if it exists, else None.
        
        When a specific calibration folder is selected (not 'default'),
        search ONLY in that folder - no fallback to root calibration_data.
        """
        calibration_folder = self.config.get("calibration_folder", "default")
        
        if calibration_folder != "default":
            # Strict search: only in the selected subfolder
            candidate_dirs = [
                os.path.join(os.getcwd(), "calibration_data", calibration_folder),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "calibration_data", calibration_folder),
            ]
        else:
            # Default: search in root calibration_data folder
            candidate_dirs = [
                os.path.join(os.getcwd(), "calibration_data"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "calibration_data"),
            ]

        for d in candidate_dirs:
            path = os.path.join(d, "field_flattening.npz")
            if os.path.isfile(path):
                logger.info(f"Found field flattening file: {path}")
                return path

        return None

    def load_field_flattening(self):
        """Load field flattening data from disk if available.
        
        Returns:
            bool: True if field flattening was loaded successfully, False otherwise.
        """
        ff_path = self._find_field_flattening_file()
        if ff_path is None:
            logger.debug("No field flattening file found")
            self.flat_field = None
            self.flat_field_info = None
            return False

        try:
            data = np.load(ff_path, allow_pickle=True)
            self.flat_field = data['flat_field']
            self.flat_field_info = {
                'mean_per_channel': data['mean_per_channel'].tolist() if 'mean_per_channel' in data else None,
                'std_per_channel': data['std_per_channel'].tolist() if 'std_per_channel' in data else None,
                'date_created': str(data['date_created']) if 'date_created' in data else None,
                'num_images_averaged': int(data['num_images_averaged']) if 'num_images_averaged' in data else None,
                'image_shape': tuple(data['image_shape']) if 'image_shape' in data else None,
            }
            logger.info(f"Loaded field flattening data: shape={self.flat_field.shape}, "
                       f"images_averaged={self.flat_field_info.get('num_images_averaged')}")
            return True
        except Exception as e:
            logger.error(f"Failed to load field flattening data: {e}")
            self.flat_field = None
            self.flat_field_info = None
            return False

    def has_field_flattening(self) -> bool:
        """Check if field flattening data is available."""
        if self.flat_field is not None:
            return True
        # Try to load if not already loaded
        return self.load_field_flattening()

    def apply_field_flattening(self, image: np.ndarray) -> np.ndarray:
        """Apply field flattening correction to an image.
        
        The correction formula is:
            corrected = image * mean(flat_field) / flat_field
        
        This normalizes the scanner response so that a uniform input
        produces a uniform output.
        
        Args:
            image: Input image array (H, W, 3) or (H, W)
            
        Returns:
            Corrected image array with same dtype as input.
        """
        if self.flat_field is None:
            logger.warning("apply_field_flattening called but no flat field loaded")
            return image

        # Handle grayscale images
        if len(image.shape) == 2:
            # Convert to 3-channel for processing
            image_3ch = np.stack([image] * 3, axis=-1)
            result = self._apply_flattening_3ch(image_3ch)
            # Return first channel
            return result[:, :, 0]
        
        return self._apply_flattening_3ch(image)

    def _apply_flattening_3ch(self, image: np.ndarray) -> np.ndarray:
        """Apply field flattening to a 3-channel image.
        
        For maximum scientific fidelity:
        - Warns if resize is needed (reduces correction accuracy)
        - Preserves full dynamic range
        - Handles both integer and float images correctly
        """
        flat = self.flat_field
        
        # Check if flat field needs to be resized to match image
        if flat.shape[:2] != image.shape[:2]:
            logger.warning(
                f"FIDELITY WARNING: Flat field shape {flat.shape[:2]} differs from image {image.shape[:2]}. "
                f"Resizing flat field will reduce correction accuracy. "
                f"For maximum fidelity, use images at the same resolution as the flat field calibration."
            )
            # Use bilinear interpolation to resize flat field
            flat = cv2.resize(flat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # When resizing, we should also recalculate mean from resized flat
            # to maintain mathematical consistency (image * mean / flat = mean when image == flat)
            mean_per_channel = np.mean(flat, axis=(0, 1), keepdims=True)
            logger.info(f"Recalculated mean from resized flat field: {mean_per_channel.flatten()}")
        else:
            # Use the mean from when the flat field was created (stored in npz)
            # This ensures that applying flat to the same image gives a perfectly uniform result
            if self.flat_field_info and self.flat_field_info.get('mean_per_channel'):
                mean_per_channel = np.array(self.flat_field_info['mean_per_channel']).reshape(1, 1, 3)
            else:
                # Fallback: compute mean from flat field
                mean_per_channel = np.mean(flat, axis=(0, 1), keepdims=True)

        # Store original dtype and determine value range
        orig_dtype = image.dtype
        
        # Determine the actual max value for clipping based on dtype AND image content
        if np.issubdtype(orig_dtype, np.integer):
            max_val = np.iinfo(orig_dtype).max
        else:
            # For float images, determine if normalized (0-1) or not
            actual_max = np.max(image)
            if actual_max <= 1.0:
                max_val = 1.0
            elif actual_max <= 255:
                max_val = 255.0
            elif actual_max <= 65535:
                max_val = 65535.0
            else:
                max_val = actual_max * 1.1  # Allow some headroom
            logger.debug(f"Float image detected, using max_val={max_val} for clipping")
        
        # Prevent division by zero with epsilon relative to data range
        epsilon = max_val * 1e-9 if max_val > 0 else 1e-9
        flat_safe = np.maximum(flat.astype(np.float64), epsilon)
        
        # Apply correction: corrected = image * mean / flat
        corrected = image.astype(np.float64) * mean_per_channel / flat_safe
        
        # Clip to valid range
        corrected = np.clip(corrected, 0, max_val)
        
        return corrected.astype(orig_dtype)

    def apply_flat(self, skip_integral_compute: bool = False) -> bool:
        """Apply field flattening correction (independent of dose conversion).
        
        This applies the scanner uniformity correction to the current image.
        Can be combined with dose conversion, but MUST be applied BEFORE dose
        conversion for physically meaningful results (flat corrects scanner
        response in signal space, not dose space).
        
        Args:
            skip_integral_compute: If True, skip computing integral images.
                Use this when chaining with other operations that will compute
                integrals at the end.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.current_image is None:
            logger.warning("apply_flat called with no image loaded")
            return False
        
        # Safety check: flat should not be applied after calibration (dose space)
        if self.calibration_applied:
            logger.error(
                "apply_flat called when calibration is already applied. "
                "Field flattening corrects scanner response in signal space, not dose space. "
                "Please reprocess the image and apply flat BEFORE dose calibration."
            )
            return False
        
        if not self.has_field_flattening():
            logger.error("apply_flat called but no field flattening data available")
            return False
        
        try:
            # Apply field flattening to current image
            self.current_image = self.apply_field_flattening(self.current_image)
            
            # Store flattened image for display when calibration is also active
            self.flattened_image = self.current_image.copy()
            
            # Recompute integral images for measurement (unless skipped for pipeline optimization)
            if not skip_integral_compute:
                self._compute_integral_images()
            
            # Mark as flat applied
            self.flat_applied = True
            
            logger.info("Applied field flattening")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply flat: {e}", exc_info=True)
            return False

