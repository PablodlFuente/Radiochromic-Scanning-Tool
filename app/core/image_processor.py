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
        self.current_file = None
        self.binned_image = None  # Store the binned image
        # Track whether dose calibration is currently applied
        self.calibration_applied = False
        
        # Display settings
        self.zoom = 1.0
        self.negative_mode = config.get("negative_mode", False)
        self.contrast = 1.0
        self.brightness = 1.0
        self.saturation = 1.0
        self.binning = 1  # Default binning is 1x1 (no binning)
        self.binned_image_file = None
        
        # Measurement settings
        self.measurement_shape = "circular"
        self.measurement_size = 20
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

    def load_image(self, file_path):
        """Load an image from the specified path."""
        try:
            # Reset cancel flag
            self.cancel_operation = False
            
            # Calibration state resets when loading a new image
            self.calibration_applied = False
            
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
            
            # Check if operation was cancelled
            if self.cancel_operation:
                logger.info("Image loading cancelled")
                return False
            
            # Report progress
            self._report_progress("loading", 30, "Processing image")
            
            # Convert BGR to RGB if color image
            if len(image.shape) == 3 and image.shape[2] == 3:
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
    
    def get_auto_measure():
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
                    
                        # Collect raw data for histogram
                        masked_data = np.zeros((len(sample_x), self.current_image.shape[2]), dtype=np.uint8)
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
                    
                        # Collect raw data for histogram
                        masked_data = np.zeros(len(sample_x), dtype=np.uint8)
                        for i, (x, y) in enumerate(zip(sample_x, sample_y)):
                            masked_data[i] = self.current_image[y, x] # Note: Grayscale indexing was [y,x], seems correct as per typical image coord. conventions if x=col, y=row
                    
                        # Store the raw data and coordinates for later use
                        self.last_measurement_raw_data = masked_data
                        self.last_measurement_coordinates = sample_rel
                        self.last_auto_measure_time = time.time()
                    


                        # For grayscale, we don't need combined uncertainty (only one channel)
                        return mean, std_dev, std_err, mean, std_err, pixel_count

                elif self.measurement_shape == "rectangular":
                    # For rectangular regions, we can use integral images for faster calculation
                    # Calculate region boundaries
                    x1 = max(0, img_x - actual_size // 2)
                    y1 = max(0, img_y - actual_size // 2)
                    x2 = min(self.current_image.shape[1] - 1, img_x + actual_size // 2)
                    y2 = min(self.current_image.shape[0] - 1, img_y + actual_size // 2)
                
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
                    # For line profiles, we need to get pixels along the line
                    # For simplicity, we'll take a horizontal line of 'actual_size' length
                    x_start = max(0, img_x - actual_size // 2)
                    x_end = min(self.current_image.shape[1] - 1, img_x + actual_size // 2)
                    y_fixed = img_y
                
                    if x_start >= x_end:
                        logger.warning("Invalid region for line measurement")
                        return None
                
                    line_pixels = self.current_image[y_fixed, x_start:x_end+1]
                    pixel_count = len(line_pixels)
                
                    if len(self.current_image.shape) == 3:  # Color image
                        means = np.mean(line_pixels, axis=0)
                        std_devs = np.std(line_pixels, axis=0)
                        std_err = std_devs / np.sqrt(pixel_count) if pixel_count > 0 else np.zeros_like(std_devs)
                    
                        # Calculate combined uncertainty using the configured method
                        rgb_mean, rgb_mean_std = self._calculate_combined_uncertainty(means, std_err)
                    
                        self.last_measurement_raw_data = line_pixels
                        self.last_measurement_coordinates = np.arange(x_start - img_x, x_end - img_x + 1)
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
                        self.last_measurement_coordinates = np.arange(x_start - img_x, x_end - img_x + 1)
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
        """
        # Require an image to work on.
        if self.current_image is None:
            logger.warning("apply_calibration called with no image loaded")
            return False

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

                    # Optional std_err columns – tolerate multiple naming conventions
                    try:
                        sa = float(row.get("sigma_a") or row.get("sa") or row.get("σa") or 0.0)
                        sb = float(row.get("sigma_b") or row.get("sb") or row.get("σb") or 0.0)
                        sc = float(row.get("sigma_c") or row.get("sc") or row.get("σc") or 0.0)
                    except ValueError:
                        sa = sb = sc = 0.0
                    param_sigmas[ch] = (sa, sb, sc)
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
                valid = denom != 0
                dose[valid] = c + b / denom[valid]

                # std_err per-pixel
                var = np.full_like(img_uint, np.nan, dtype=np.float32)
                if np.any(valid):
                    term_a = (-b / (denom[valid] ** 2)) ** 2 * sa ** 2
                    term_b = (1.0 / denom[valid]) ** 2 * sb ** 2
                    term_c = sc ** 2
                    var[valid] = term_a + term_b + term_c
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
                    valid = denom != 0

                    dose_ch = np.full((h, w), np.nan, dtype=np.float32)
                    var_ch = np.full((h, w), np.nan, dtype=np.float32)
                    # Dose
                    dose_ch[valid] = c + b / denom[valid]
                    # Variance via error propagation
                    term_a = (-b / (denom[valid] ** 2)) ** 2 * sa ** 2
                    term_b = (1.0 / denom[valid]) ** 2 * sb ** 2
                    term_c = sc ** 2
                    var_ch[valid] = term_a + term_b + term_c

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


    
    def update_settings(self, config):
        """Update settings from config."""
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
    
        logger.info("Settings updated")
        return True
    
    def reprocess_current_image(self):
        """Reprocess the current image with updated settings."""
        if self.original_image is None:
            return False
    
        # Reset the current image to the original
        self.current_image = self.original_image.copy()
        # Calibration no longer applied
        self.calibration_applied = False
    
        # Apply binning if needed
        if self.binning > 1:
            self.apply_binning()
        else:
            # Recompute integral images
            self._compute_integral_images()
    
        return True
    
    def get_display_image(self):
        """Get the image for display with current settings applied."""
        if self.current_image is None:
            return None, 0, 0
    
        try:
            # Apply display settings
            # When dose calibration is active we still want to show the *original* (pre-calibration)
            # image to the user so that the visual appearance remains unchanged while all
            # internal calculations work on the calibrated single-channel dose image.
            if self.calibration_applied and self.original_image is not None:
                image = self.original_image.copy()
            else:
                image = self.current_image.copy()
        
            # Apply negative mode
            if self.negative_mode:
                image = 255 - image
        
            # Apply contrast, brightness, and saturation
            if self.contrast != 1.0 or self.brightness != 1.0 or self.saturation != 1.0:
                # Convert to PIL Image for easier adjustments
                if len(image.shape) == 3:
                    pil_image = Image.fromarray(image)
                else:
                    pil_image = Image.fromarray(image).convert("L")
            
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

    def _find_fit_parameters_file(self):
        """Return absolute path to *fit_parameters.csv* if it exists, else None."""
        # Get calibration folder from config
        calibration_folder = self.config.get("calibration_folder", "default")
        
        candidate_dirs = [
            os.path.join(os.getcwd(), "calibration_data"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "calibration_data"),
        ]
        
        # If a specific calibration folder is selected, check subdirectory first
        if calibration_folder != "default":
            priority_dirs = [
                os.path.join(os.getcwd(), "calibration_data", calibration_folder),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "calibration_data", calibration_folder),
            ]
            candidate_dirs = priority_dirs + candidate_dirs

        for d in candidate_dirs:
            path = os.path.join(d, "fit_parameters.csv")
            if os.path.isfile(path):
                logger.info(f"Using calibration file: {path}")
                return path

        return None
