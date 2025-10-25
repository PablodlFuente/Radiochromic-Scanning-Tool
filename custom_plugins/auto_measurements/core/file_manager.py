"""
File data management and persistence.

This module contains the FileDataManager class for storing and loading
detection results, including circles, films, and measurements.
"""

from typing import Optional, List, Dict, Any
import os
import json
import logging
from datetime import datetime

from ..models import DetectionParams, Circle, Film


# Direct access to parent module for _OVERLAY (proxy doesn't work for globals)
def _get_parent_module():
    """Get parent module reference."""
    import sys
    return sys.modules.get('auto_measurements') or sys.modules.get('custom_plugins.auto_measurements')


class FileDataManager:
    """Manages multiple image files in a single analysis session.
    
    Handles:
    - File list management (add, remove, navigate)
    - Per-file data persistence (results, CTR maps, overlays)
    - File navigation with automatic data save/restore
    - Data structure consistency across file switches
    """
    
    def __init__(self, tree_widget, image_processor, main_window):
        """Initialize file data manager.
        
        Args:
            tree_widget: ttk.Treeview widget for displaying measurements
            image_processor: ImageProcessor instance for image operations
            main_window: MainWindow instance for loading images
        """
        self.tree = tree_widget
        self.image_processor = image_processor
        self.main_window = main_window
        
        # File management
        self.file_list = []
        self.current_file_index = 0
        self.file_data = {}  # {file_path: {results, ctr_map_by_name, overlay_state, etc.}}
        
        # UI controls (to be set by parent)
        self.prev_button = None
        self.next_button = None
        self.file_counter_label = None
        self.current_file_label = None
    
    def set_ui_controls(self, prev_button, next_button, file_counter_label, current_file_label):
        """Set UI control references for navigation.
        
        Args:
            prev_button: Button for previous file
            next_button: Button for next file  
            file_counter_label: Label showing current/total files
            current_file_label: Label showing current filename
        """
        self.prev_button = prev_button
        self.next_button = next_button
        self.file_counter_label = file_counter_label
        self.current_file_label = current_file_label
    
    def add_files(self, store_current_callback, update_treeview_callback):
        """Allow user to select multiple image files for analysis.
        
        Args:
            store_current_callback: Function to store current file data before changes
            update_treeview_callback: Function to update TreeView after loading file
        
        Returns:
            bool: True if files were added, False otherwise
        """
        from tkinter import filedialog, messagebox
        
        # If files are already loaded, ask user what to do
        if self.file_list:
            response = messagebox.askyesnocancel(
                "Add Files",
                "Files are already loaded. What would you like to do?\n\n"
                "Yes = Add to current set\n"
                "No = Start new set (clear current files)\n"
                "Cancel = Don't add files"
            )
            
            if response is None:  # Cancel
                return False
            elif not response:  # No - start new set
                self.clear_all()
        
        filetypes = [
            ("Image files", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"),
            ("TIFF files", "*.tif *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select image files for analysis",
            filetypes=filetypes
        )
        
        if files:
            # Store current file data if any exists
            store_current_callback()
            
            # Add new files to the list
            new_files_count = 0
            for file_path in files:
                if file_path not in self.file_list:
                    self.file_list.append(file_path)
                    new_files_count += 1
                    # Initialize empty data for this file
                    self.file_data[file_path] = {
                        'results': [],
                        'ctr_map': {},
                        'original_measurements': {},
                        'original_radii': {},
                        'original_values': {},
                        'measured': False,
                        'overlay_state': {
                            'films': [],
                            'circles': [],
                            'item_to_shape': {},
                            'ctr_map': {},
                            '_shape': (0, 0),
                            'scale': 1.0
                        }
                    }
            
            # Update UI
            self.update_navigation_controls()
            
            # Load the first file if no image is currently loaded or if this is the first file added
            should_load_first = False
            if len(self.file_list) == new_files_count:  # All files are new (empty list or cleared)
                should_load_first = True
            elif not self.image_processor.has_image():  # No image currently loaded
                should_load_first = True
            
            if should_load_first and self.file_list:
                self.current_file_index = 0
                self.load_file(self.file_list[0], update_treeview_callback)
            
            if new_files_count > 0:
                messagebox.showinfo("Files Added", f"Added {new_files_count} new file(s). Total: {len(self.file_list)} files.")
            else:
                messagebox.showinfo("Files Added", "All selected files were already in the list.")
            
            return True
        return False
    
    def store_current_data(self, get_results_callback, get_ctr_map_callback, 
                          get_original_measurements_callback, get_original_radii_callback,
                          get_original_values_callback):
        """Store current measurements and data for the current file.
        
        Args:
            get_results_callback: Function that returns current results list
            get_ctr_map_callback: Function that returns current CTR map {film_name: circle_id}
            get_original_measurements_callback: Function that returns original measurements dict
            get_original_radii_callback: Function that returns original radii dict
            get_original_values_callback: Function that returns original values dict
        """
        if self.file_list and 0 <= self.current_file_index < len(self.file_list):
            current_file = self.file_list[self.current_file_index]
            
            # Get parent module for direct _OVERLAY access
            parent_module = _get_parent_module()
            
            # Initialize _OVERLAY if None (shape will be updated when image loads)
            if parent_module._OVERLAY is None:
                parent_module._OVERLAY = {
                    'films': [],
                    'circles': [],
                    'item_to_shape': {},
                    'ctr_map': {},
                    '_shape': (0, 0),  # Will be updated by update_overlay_shape()
                    'scale': 1.0
                }
            
            # Build a shape mapping by name (film/circle names) instead of TreeView IDs
            # This way we can restore it even when TreeView IDs change
            shapes_by_name = {}
            radii_by_name = {}
            ctr_map_by_name = {}  # Store CTR using names instead of TreeView IDs
            
            for item_id, (shape_type, coords) in parent_module._OVERLAY.get('item_to_shape', {}).items():
                item_name = self.tree.item(item_id, 'text') if self.tree.exists(item_id) else None
                if item_name:
                    # Remove " (CTR)" suffix for consistent naming
                    item_name_clean = item_name.replace(" (CTR)", "")
                    if shape_type == 'film':
                        shapes_by_name[('film', item_name_clean)] = coords
                    elif shape_type == 'circle':
                        # Get parent film name
                        parent_id = self.tree.parent(item_id)
                        parent_name = self.tree.item(parent_id, 'text') if parent_id else None
                        if parent_name:
                            shapes_by_name[('circle', parent_name, item_name_clean)] = coords
                            # Also store the original radius with clean name
                            original_radii = get_original_radii_callback()
                            if item_id in original_radii:
                                radii_by_name[(parent_name, item_name_clean)] = original_radii[item_id]
            
            # Convert ctr_map from {film_name: circle_id} to {film_name: circle_name}
            ctr_map = get_ctr_map_callback()
            for film_name, circle_id in ctr_map.items():
                if self.tree.exists(circle_id):
                    circle_name = self.tree.item(circle_id, 'text')
                    # Remove " (CTR)" suffix if present
                    circle_name = circle_name.replace(" (CTR)", "")
                    ctr_map_by_name[film_name] = circle_name
            
            overlay_state = {
                'films': parent_module._OVERLAY.get('films', []).copy(),
                'circles': parent_module._OVERLAY.get('circles', []).copy(),
                'shapes_by_name': shapes_by_name,
                'radii_by_name': radii_by_name,
                'ctr_map': parent_module._OVERLAY.get('ctr_map', {}).copy(),
                '_shape': parent_module._OVERLAY.get('_shape', (0, 0)),
                'scale': parent_module._OVERLAY.get('scale', 1.0)
            }
            
            results = get_results_callback()
            self.file_data[current_file] = {
                'results': results.copy(),
                'ctr_map_by_name': ctr_map_by_name,  # Store by name instead of ID
                'original_measurements': get_original_measurements_callback().copy(),
                'radii_by_name': radii_by_name.copy(),
                'original_values': get_original_values_callback().copy(),
                'measured': len(results) > 0,
                'overlay_state': overlay_state
            }
    
    def load_file_data(self, file_path, set_results_callback, set_ctr_map_callback,
                      set_original_measurements_callback, set_original_radii_callback,
                      set_original_values_callback):
        """Load data for a specific file.
        
        Args:
            file_path: Path to file to load data for
            set_results_callback: Function to set results list
            set_ctr_map_callback: Function to set CTR map
            set_original_measurements_callback: Function to set original measurements
            set_original_radii_callback: Function to set original radii
            set_original_values_callback: Function to set original values
        
        Returns:
            dict: CTR map by name (temporary storage for TreeView rebuild)
        """
        # Get parent module for direct _OVERLAY access
        parent_module = _get_parent_module()
        
        # Initialize _OVERLAY if None (shape will be updated by update_overlay_shape())
        if parent_module._OVERLAY is None:
            parent_module._OVERLAY = {
                'films': [],
                'circles': [],
                'item_to_shape': {},
                'ctr_map': {},
                '_shape': (0, 0),
                'scale': 1.0
            }
        
        ctr_map_by_name = {}
        
        if file_path in self.file_data:
            data = self.file_data[file_path]
            set_results_callback(data['results'].copy())
            
            # Load ctr_map_by_name (will be converted to IDs in _update_treeview_from_data)
            # For now, just store it temporarily
            ctr_map_by_name = data.get('ctr_map_by_name', {}).copy()
            # Backward compatibility: if old format exists, use it
            if not ctr_map_by_name and 'ctr_map' in data:
                set_ctr_map_callback(data['ctr_map'].copy())
            else:
                set_ctr_map_callback({})  # Will be rebuilt in _update_treeview_from_data
            
            set_original_measurements_callback(data['original_measurements'].copy())
            
            # Handle backward compatibility for original_radii
            # Old format: TreeView IDs as keys -> won't work after rebuild
            # New format: stored in radii_by_name with (film, circle) tuples
            if 'radii_by_name' in data:
                # New format - will be restored in _update_treeview_from_data
                set_original_radii_callback({})
            elif 'original_radii' in data:
                # Old format - try to preserve it
                set_original_radii_callback(data['original_radii'].copy())
            else:
                set_original_radii_callback({})
            
            set_original_values_callback(data.get('original_values', {}).copy())
            
            # Don't restore overlay_state here - it will be rebuilt in _update_treeview_from_data
            # Just preserve the _shape and scale
            if 'overlay_state' in data:
                overlay_state = data['overlay_state']
                parent_module._OVERLAY['_shape'] = overlay_state.get('_shape', (0, 0))
                parent_module._OVERLAY['scale'] = overlay_state.get('scale', 1.0)
                # Clear the old data that will be rebuilt
                parent_module._OVERLAY['films'] = []
                parent_module._OVERLAY['circles'] = []
                parent_module._OVERLAY['item_to_shape'] = {}
        else:
            # No data for this file yet
            set_results_callback([])
            set_ctr_map_callback({})
            set_original_measurements_callback({})
            set_original_radii_callback({})
            set_original_values_callback({})
        
        return ctr_map_by_name
    
    def load_file(self, file_path, update_treeview_callback):
        """Load a specific image file and its associated data.
        
        Args:
            file_path: Path to file to load
            update_treeview_callback: Function to update TreeView after loading
        """
        from tkinter import messagebox
        import os
        
        try:
            # Load the image through the main window
            self.main_window.load_image(file_path)
            
            # Update TreeView with loaded data (will call load_file_data internally)
            update_treeview_callback()
            
            # Update image display to show overlay for this file
            self.main_window.update_image()
            
            # Update UI
            filename = os.path.basename(file_path)
            if self.current_file_label:
                self.current_file_label.config(text=f"Current: {filename}", foreground="black")
            
        except Exception as e:
            # Clean error message to avoid Unicode encoding issues in messagebox
            error_msg = str(e).encode('ascii', 'replace').decode('ascii')
            # Get just the filename, not the full path
            filename = os.path.basename(file_path)
            messagebox.showerror("Load Error", f"Error loading file {filename}:\n{error_msg}")
            logging.error(f"Error loading file {file_path}: {e}", exc_info=True)
    
    def navigate_previous(self, store_current_callback, update_treeview_callback):
        """Navigate to previous file.
        
        Args:
            store_current_callback: Function to store current file data
            update_treeview_callback: Function to update TreeView after navigation
        """
        if self.current_file_index > 0:
            store_current_callback()
            self.current_file_index -= 1
            self.load_file(self.file_list[self.current_file_index], update_treeview_callback)
            self.update_navigation_controls()
    
    def navigate_next(self, store_current_callback, update_treeview_callback):
        """Navigate to next file.
        
        Args:
            store_current_callback: Function to store current file data
            update_treeview_callback: Function to update TreeView after navigation
        """
        if self.current_file_index < len(self.file_list) - 1:
            store_current_callback()
            self.current_file_index += 1
            self.load_file(self.file_list[self.current_file_index], update_treeview_callback)
            self.update_navigation_controls()
    
    def update_navigation_controls(self):
        """Update navigation button states and counter label."""
        if not self.prev_button or not self.next_button or not self.file_counter_label:
            return  # UI controls not set yet
        
        if not self.file_list:
            self.prev_button.config(state="disabled")
            self.next_button.config(state="disabled")
            self.file_counter_label.config(text="0/0")
            return
        
        total_files = len(self.file_list)
        current_num = self.current_file_index + 1
        
        # Update counter
        self.file_counter_label.config(text=f"{current_num}/{total_files}")
        
        # Update button states
        self.prev_button.config(state="normal" if self.current_file_index > 0 else "disabled")
        self.next_button.config(state="normal" if self.current_file_index < total_files - 1 else "disabled")
    
    def clear_all(self):
        """Clear all files and data completely.
        
        Returns:
            bool: True if data was cleared, False if no data to clear
        """
        if not self.file_list:
            return False
        
        self.file_list.clear()
        self.file_data.clear()
        self.current_file_index = 0
        
        # Update navigation controls
        self.update_navigation_controls()
        if self.current_file_label:
            self.current_file_label.config(text="No files loaded", foreground="gray")
        
        return True
    
    def get_current_file(self):
        """Get current file path.
        
        Returns:
            str: Current file path or None if no files loaded
        """
        if self.file_list and 0 <= self.current_file_index < len(self.file_list):
            return self.file_list[self.current_file_index]
        return None
    
    def has_files(self):
        """Check if any files are loaded.
        
        Returns:
            bool: True if files are loaded, False otherwise
        """
        return len(self.file_list) > 0
    
    def get_file_count(self):
        """Get total number of files.
        
        Returns:
            int: Number of files in the list
        """
        return len(self.file_list)
    
    def get_shapes_and_radii_for_current_file(self):
        """Get shapes_by_name and radii_by_name for current file.
        
        Returns:
            tuple: (shapes_by_name dict, radii_by_name dict)
        """
        shapes_by_name = {}
        radii_by_name = {}
        
        if self.file_list and 0 <= self.current_file_index < len(self.file_list):
            current_file = self.file_list[self.current_file_index]
            if current_file in self.file_data:
                file_data = self.file_data[current_file]
                overlay_state = file_data.get('overlay_state', {})
                shapes_by_name = overlay_state.get('shapes_by_name', {})
                radii_by_name = file_data.get('radii_by_name', overlay_state.get('radii_by_name', {}))
                
                # Backward compatibility: extract radii from shapes_by_name if radii_by_name is empty
                if not radii_by_name and shapes_by_name:
                    for key, coords in shapes_by_name.items():
                        if key[0] == 'circle' and len(coords) == 3:  # (cx, cy, r)
                            film_name, circle_name = key[1], key[2]
                            radii_by_name[(film_name, circle_name)] = coords[2]
        
        return shapes_by_name, radii_by_name


# ================================================================
# Phase 8: CSVExporter - CSV export functionality
# ================================================================



__all__ = ['FileDataManager']
