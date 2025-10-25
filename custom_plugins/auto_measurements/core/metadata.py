"""
Image metadata extraction and resolution detection.

This module contains the MetadataExtractor class for extracting
metadata from radiochromic film images including DPI/resolution and date.
"""

from typing import Optional
import os
import logging


class MetadataExtractor:
    """Handles image metadata extraction and resolution detection."""
    
    def __init__(self, image_processor):
        self.image_processor = image_processor
        self.stored_resolution = None
        self.session_metadata_key = None
        self.manual_dpi_value = None
    
    def get_all_metadata(self) -> dict:
        """Get all metadata from the current image using multiple methods."""
        all_metadata = {}
        
        try:
            # Method 1: Try to get metadata from image processor
            if hasattr(self.image_processor, 'get_all_metadata'):
                processor_metadata = self.image_processor.get_all_metadata()
                if processor_metadata:
                    all_metadata.update(processor_metadata)
            
            # Method 2: Get image path
            img_path = self._get_image_path()
            
            if img_path and os.path.exists(img_path):
                # Method 3: PIL/Pillow - EXIF and basic info
                all_metadata.update(self._extract_pil_metadata(img_path))
                
                # Method 4: Try exifread library (if available)
                all_metadata.update(self._extract_exifread_metadata(img_path))
                
                # Method 5: Try to read Windows properties (Windows only)
                all_metadata.update(self._extract_windows_metadata(img_path))
                
                # Method 6: Manual EXIF parsing
                all_metadata.update(self._extract_manual_dpi(img_path))
            
            # Add debug information
            all_metadata['_debug_total_keys'] = len(all_metadata)
            all_metadata['_debug_image_path'] = img_path if img_path else "No path found"
            
            return all_metadata
                        
        except Exception as e:
            logging.error(f"Error getting metadata: {e}")
            return {}
    
    def _get_image_path(self) -> Optional[str]:
        """Get the current image file path."""
        for attr in ['original_image_path', 'image_path', 'current_file_path']:
            if hasattr(self.image_processor, attr):
                path = getattr(self.image_processor, attr)
                if path:
                    return path
        return None
    
    def _extract_pil_metadata(self, img_path: str) -> dict:
        """Extract metadata using PIL/Pillow."""
        metadata = {}
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            with Image.open(img_path) as img:
                # Get EXIF data
                exifdata = img.getexif()
                for tag_id, value in exifdata.items():
                    tag = TAGS.get(tag_id, f"Unknown_EXIF_{tag_id}")
                    metadata[f"EXIF_{tag}"] = value
                
                # Get image info (including DPI)
                if hasattr(img, 'info') and img.info:
                    for key, value in img.info.items():
                        metadata[f"PIL_{key}"] = value
                
                # Try to get DPI directly
                if hasattr(img, 'info') and 'dpi' in img.info:
                    metadata['PIL_dpi_direct'] = img.info['dpi']
                
                # Get format-specific info
                if hasattr(img, 'format'):
                    metadata['Format'] = img.format
                if hasattr(img, 'mode'):
                    metadata['Mode'] = img.mode
                if hasattr(img, 'size'):
                    metadata['Size'] = img.size
        
        except Exception as e:
            logging.debug(f"PIL metadata extraction failed: {e}")
        
        return metadata
    
    def _extract_exifread_metadata(self, img_path: str) -> dict:
        """Extract metadata using exifread library."""
        metadata = {}
        try:
            import exifread
            with open(img_path, 'rb') as f:
                tags = exifread.process_file(f, details=True)
                for tag, value in tags.items():
                    if tag not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']:
                        metadata[f"ExifRead_{tag}"] = str(value)
        except ImportError:
            metadata['_info_exifread'] = "ExifRead not installed - may provide more metadata"
        except Exception as e:
            logging.debug(f"ExifRead metadata extraction failed: {e}")
        
        return metadata
    
    def _extract_windows_metadata(self, img_path: str) -> dict:
        """Extract metadata using Windows properties (Windows only)."""
        metadata = {}
        try:
            import platform
            import subprocess
            import json
            
            if platform.system() == "Windows":
                ps_command = f'''
                $file = Get-Item "{img_path}"
                $props = @{{}}
                $file.PSObject.Properties | ForEach-Object {{
                    if ($_.Value -ne $null) {{
                        $props[$_.Name] = $_.Value.ToString()
                    }}
                }}
                $props | ConvertTo-Json
                '''
                
                result = subprocess.run(['powershell', '-Command', ps_command], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        windows_props = json.loads(result.stdout)
                        for key, value in windows_props.items():
                            metadata[f"Windows_{key}"] = value
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logging.debug(f"Windows properties extraction failed: {e}")
        
        return metadata
    
    def _extract_manual_dpi(self, img_path: str) -> dict:
        """Manually parse file for DPI patterns."""
        metadata = {}
        try:
            with open(img_path, 'rb') as f:
                content = f.read(8192)  # Read first 8KB
                
                import re
                dpi_patterns = [
                    rb'(\d+)\s*dpi',
                    rb'(\d+)\s*dots per inch',
                    rb'resolution[^\d]*(\d+)',
                ]
                
                for pattern in dpi_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        metadata['Manual_DPI_Detection'] = [int(m) for m in matches if m.isdigit()]
                        break
        except Exception as e:
            logging.debug(f"Manual EXIF parsing failed: {e}")
        
        return metadata
    
    def extract_resolution(self, metadata_key: str, metadata_value) -> Optional[float]:
        """Extract resolution/DPI value from a metadata entry."""
        try:
            # Handle different metadata formats
            if isinstance(metadata_value, (tuple, list)) and len(metadata_value) >= 1:
                return float(metadata_value[0])  # Use X resolution
            elif isinstance(metadata_value, (int, float)):
                return float(metadata_value)
            elif isinstance(metadata_value, str):
                return self._parse_resolution_string(metadata_value)
            
            # Handle manual DPI detection results
            if metadata_key == 'Manual_DPI_Detection' and isinstance(metadata_value, list):
                for val in metadata_value:
                    if 10 <= val <= 10000:
                        return float(val)
            
            return None
        except (ValueError, TypeError, IndexError, ZeroDivisionError):
            return None
    
    def _parse_resolution_string(self, value_str: str) -> Optional[float]:
        """Parse resolution from string with various formats."""
        import re
        
        cleaned = value_str.replace(',', '.').lower()
        
        # Look for common DPI/resolution patterns
        patterns = [
            r'(\d+\.?\d*)\s*dpi',
            r'(\d+\.?\d*)\s*dots per inch',
            r'(\d+\.?\d*)\s*pixels per inch',
            r'(\d+\.?\d*)\s*ppi',
            r'(\d+\.?\d*)\s*/\s*1',  # Rational format like "300/1"
            r'(\d+\.?\d*)\s*x\s*\d+',  # Format like "300 x 300"
            r'(\d+\.?\d*)',  # Just extract first number
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, cleaned)
            if matches:
                value = float(matches[0])
                if 10 <= value <= 10000:  # Reasonable DPI range
                    return value
        
        # Special handling for fractions
        if '/' in cleaned:
            parts = cleaned.split('/')
            if len(parts) == 2:
                try:
                    numerator = float(parts[0].strip())
                    denominator = float(parts[1].strip())
                    if denominator != 0:
                        value = numerator / denominator
                        if 10 <= value <= 10000:
                            return value
                except ValueError:
                    pass
        
        return None
    
    def get_resolution_auto(self) -> Optional[float]:
        """Automatically detect resolution from metadata."""
        try:
            # Check if manual DPI was set
            if hasattr(self, 'manual_dpi_value') and self.manual_dpi_value:
                return self.manual_dpi_value
            
            # Check if we have stored resolution from session
            if self.stored_resolution:
                return self.stored_resolution
            
            # Get all metadata
            metadata = self.get_all_metadata()
            
            # If session key is set, use it directly
            if self.session_metadata_key and self.session_metadata_key in metadata:
                resolution = self.extract_resolution(self.session_metadata_key, metadata[self.session_metadata_key])
                if resolution:
                    self.stored_resolution = resolution
                    return resolution
            
            # Try common DPI/resolution keys first
            priority_keys = [
                'PIL_dpi', 'PIL_dpi_direct', 'EXIF_XResolution', 'EXIF_YResolution',
                'PIL_resolution', 'ExifRead_Image XResolution', 'ExifRead_Image YResolution',
                'Manual_DPI_Detection'
            ]
            
            for key in priority_keys:
                if key in metadata:
                    resolution = self.extract_resolution(key, metadata[key])
                    if resolution:
                        self.stored_resolution = resolution
                        return resolution
            
            # Try all other keys
            for key, value in metadata.items():
                if 'dpi' in key.lower() or 'resolution' in key.lower():
                    resolution = self.extract_resolution(key, value)
                    if resolution:
                        self.stored_resolution = resolution
                        return resolution
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting resolution from metadata: {e}")
            return None
    
    def extract_date(self) -> Optional[str]:
        """Extract date from metadata or file modification date."""
        try:
            metadata = self.get_all_metadata()
            
            # Try EXIF date tags first
            date_keys = [
                'EXIF_DateTime', 'EXIF_DateTimeOriginal', 'EXIF_DateTimeDigitized',
                'ExifRead_EXIF DateTimeOriginal', 'ExifRead_Image DateTime'
            ]
            
            for key in date_keys:
                if key in metadata:
                    date_str = str(metadata[key])
                    # Parse common date formats
                    try:
                        import datetime as dt
                        # Try EXIF format: "YYYY:MM:DD HH:MM:SS"
                        if ':' in date_str and len(date_str) >= 10:
                            date_part = date_str.split()[0]  # Get date part
                            year, month, day = date_part.split(':')
                            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    except:
                        pass
            
            # Fallback to file modification time
            img_path = self._get_image_path()
            if img_path and os.path.exists(img_path):
                import datetime as dt
                mtime = os.path.getmtime(img_path)
                date = dt.datetime.fromtimestamp(mtime)
                return date.strftime("%Y-%m-%d")
            
            return None
            
        except Exception as e:
            logging.error(f"Error extracting date from metadata: {e}")
            return None




__all__ = ['MetadataExtractor']
