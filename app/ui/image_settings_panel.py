"""
Image settings panel UI for the Radiochromic Film Analyzer.

This module contains the image settings panel UI class that displays and manages
the image display settings.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
import time

logger = logging.getLogger(__name__)

class ImageSettingsPanel:
  """Image settings panel UI for the Radiochromic Film Analyzer."""
  
  def __init__(self, parent, image_processor, apply_callback=None):
      """Initialize the image settings panel."""
      self.parent = parent
      self.image_processor = image_processor
      self.apply_callback = apply_callback
      
      # Create frame
      self.frame = ttk.Frame(parent)
      
      # Debounce variables for sliders
      self.slider_update_delay = 100  # ms
      self.contrast_timer_id = None
      self.brightness_timer_id = None
      self.saturation_timer_id = None
      self.last_update_time = 0
      self.update_threshold = 50  # ms
      
      # Create UI
      self._create_ui()
      
      logger.info("Image settings panel initialized")
  
  def _create_ui(self):
      """Create the UI components."""
      # Settings title
      ttk.Label(
          self.frame, 
          text="Image Settings", 
          font=("Arial", 12, "bold")
      ).pack(anchor=tk.W, padx=10, pady=10)
      
      # Display options section
      ttk.Label(
          self.frame, 
          text="Display Options", 
          font=("Arial", 10, "bold")
      ).pack(anchor=tk.W, padx=10, pady=(10, 5))
      
      # Negative mode option
      self.negative_mode_var = tk.BooleanVar(
          value=self.image_processor.negative_mode
      )
      
      ttk.Checkbutton(
          self.frame, 
          text="Negative mode (invert image colors)", 
          variable=self.negative_mode_var,
          command=self._apply_settings
      ).pack(anchor=tk.W, padx=20, pady=5)
      
      # Contrast control
      ttk.Label(
          self.frame, 
          text="Contrast:", 
          font=("Arial", 10)
      ).pack(anchor=tk.W, padx=10, pady=(10, 5))
      
      # Contrast slider and value frame
      contrast_frame = ttk.Frame(self.frame)
      contrast_frame.pack(fill=tk.X, padx=20, pady=5)
      
      self.contrast_var = tk.DoubleVar(value=self.image_processor.contrast)
      self.contrast_slider = ttk.Scale(
          contrast_frame,
          from_=0.0,
          to=2.0,
          orient=tk.HORIZONTAL,
          variable=self.contrast_var,
          command=self._on_contrast_change
      )
      self.contrast_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
      
      # Contrast value entry
      self.contrast_entry = ttk.Entry(contrast_frame, width=6)
      self.contrast_entry.pack(side=tk.RIGHT)
      self.contrast_entry.insert(0, f"{self.contrast_var.get():.2f}")
      
      # Bind events for the entry
      self.contrast_entry.bind("<Return>", self._on_contrast_entry_change)
      self.contrast_entry.bind("<FocusOut>", self._on_contrast_entry_change)
      
      # Bind slider release event for final update
      self.contrast_slider.bind("<ButtonRelease-1>", self._on_contrast_slider_release)
      
      # Brightness control
      ttk.Label(
          self.frame, 
          text="Brightness:", 
          font=("Arial", 10)
      ).pack(anchor=tk.W, padx=10, pady=(10, 5))
      
      # Brightness slider and value frame
      brightness_frame = ttk.Frame(self.frame)
      brightness_frame.pack(fill=tk.X, padx=20, pady=5)
      
      self.brightness_var = tk.DoubleVar(value=self.image_processor.brightness)
      self.brightness_slider = ttk.Scale(
          brightness_frame,
          from_=0.0,
          to=2.0,
          orient=tk.HORIZONTAL,
          variable=self.brightness_var,
          command=self._on_brightness_change
      )
      self.brightness_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
      
      # Brightness value entry
      self.brightness_entry = ttk.Entry(brightness_frame, width=6)
      self.brightness_entry.pack(side=tk.RIGHT)
      self.brightness_entry.insert(0, f"{self.brightness_var.get():.2f}")
      
      # Bind events for the entry
      self.brightness_entry.bind("<Return>", self._on_brightness_entry_change)
      self.brightness_entry.bind("<FocusOut>", self._on_brightness_entry_change)
      
      # Bind slider release event for final update
      self.brightness_slider.bind("<ButtonRelease-1>", self._on_brightness_slider_release)
      
      # Saturation control
      ttk.Label(
          self.frame, 
          text="Saturation:", 
          font=("Arial", 10)
      ).pack(anchor=tk.W, padx=10, pady=(10, 5))
      
      # Saturation slider and value frame
      saturation_frame = ttk.Frame(self.frame)
      saturation_frame.pack(fill=tk.X, padx=20, pady=5)
      
      self.saturation_var = tk.DoubleVar(value=self.image_processor.saturation)
      self.saturation_slider = ttk.Scale(
          saturation_frame,
          from_=0.0,
          to=2.0,
          orient=tk.HORIZONTAL,
          variable=self.saturation_var,
          command=self._on_saturation_change
      )
      self.saturation_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
      
      # Saturation value entry
      self.saturation_entry = ttk.Entry(saturation_frame, width=6)
      self.saturation_entry.pack(side=tk.RIGHT)
      self.saturation_entry.insert(0, f"{self.saturation_var.get():.2f}")
      
      # Bind events for the entry
      self.saturation_entry.bind("<Return>", self._on_saturation_entry_change)
      self.saturation_entry.bind("<FocusOut>", self._on_saturation_entry_change)
      
      # Bind slider release event for final update
      self.saturation_slider.bind("<ButtonRelease-1>", self._on_saturation_slider_release)
      
      # Reset button
      ttk.Button(
          self.frame,
          text="Reset to Default",
          command=self._reset_settings
      ).pack(anchor=tk.CENTER, pady=20)
  
  def _on_contrast_change(self, value):
      """Handle contrast slider change with debouncing."""
      value = float(value)
      # Update entry without triggering its change event
      self.contrast_entry.delete(0, tk.END)
      self.contrast_entry.insert(0, f"{value:.2f}")
      
      # Store the value but don't apply immediately
      self.image_processor.contrast = value
      
      # Cancel any existing timer
      if self.contrast_timer_id:
          self.frame.after_cancel(self.contrast_timer_id)
      
      # Check if we should update now based on time since last update
      current_time = time.time() * 1000  # Convert to ms
      if current_time - self.last_update_time > self.update_threshold:
          self._apply_settings()
          self.last_update_time = current_time
      else:
          # Schedule a delayed update
          self.contrast_timer_id = self.frame.after(self.slider_update_delay, self._apply_settings)
  
  def _on_contrast_slider_release(self, event):
      """Handle contrast slider release - final update."""
      # Cancel any pending timer
      if self.contrast_timer_id:
          self.frame.after_cancel(self.contrast_timer_id)
          self.contrast_timer_id = None
      
      # Apply settings immediately
      self._apply_settings()
  
  def _on_brightness_change(self, value):
      """Handle brightness slider change with debouncing."""
      value = float(value)
      # Update entry without triggering its change event
      self.brightness_entry.delete(0, tk.END)
      self.brightness_entry.insert(0, f"{value:.2f}")
      
      # Store the value but don't apply immediately
      self.image_processor.brightness = value
      
      # Cancel any existing timer
      if self.brightness_timer_id:
          self.frame.after_cancel(self.brightness_timer_id)
      
      # Check if we should update now based on time since last update
      current_time = time.time() * 1000  # Convert to ms
      if current_time - self.last_update_time > self.update_threshold:
          self._apply_settings()
          self.last_update_time = current_time
      else:
          # Schedule a delayed update
          self.brightness_timer_id = self.frame.after(self.slider_update_delay, self._apply_settings)
  
  def _on_brightness_slider_release(self, event):
      """Handle brightness slider release - final update."""
      # Cancel any pending timer
      if self.brightness_timer_id:
          self.frame.after_cancel(self.brightness_timer_id)
          self.brightness_timer_id = None
      
      # Apply settings immediately
      self._apply_settings()
  
  def _on_saturation_change(self, value):
      """Handle saturation slider change with debouncing."""
      value = float(value)
      # Update entry without triggering its change event
      self.saturation_entry.delete(0, tk.END)
      self.saturation_entry.insert(0, f"{value:.2f}")
      
      # Store the value but don't apply immediately
      self.image_processor.saturation = value
      
      # Cancel any existing timer
      if self.saturation_timer_id:
          self.frame.after_cancel(self.saturation_timer_id)
      
      # Check if we should update now based on time since last update
      current_time = time.time() * 1000  # Convert to ms
      if current_time - self.last_update_time > self.update_threshold:
          self._apply_settings()
          self.last_update_time = current_time
      else:
          # Schedule a delayed update
          self.saturation_timer_id = self.frame.after(self.slider_update_delay, self._apply_settings)
  
  def _on_saturation_slider_release(self, event):
      """Handle saturation slider release - final update."""
      # Cancel any pending timer
      if self.saturation_timer_id:
          self.frame.after_cancel(self.saturation_timer_id)
          self.saturation_timer_id = None
      
      # Apply settings immediately
      self._apply_settings()
  
  def _on_contrast_entry_change(self, event=None):
      """Handle contrast entry change."""
      try:
          value = float(self.contrast_entry.get())
          if 0.0 <= value <= 2.0:
              self.contrast_var.set(value)
              self.image_processor.contrast = value
              self._apply_settings()
          else:
              # Reset to valid value
              self.contrast_entry.delete(0, tk.END)
              self.contrast_entry.insert(0, f"{self.contrast_var.get():.2f}")
              messagebox.showwarning("Invalid Value", "Contrast must be between 0.0 and 2.0")
      except ValueError:
          # Reset to valid value
          self.contrast_entry.delete(0, tk.END)
          self.contrast_entry.insert(0, f"{self.contrast_var.get():.2f}")
          messagebox.showwarning("Invalid Value", "Please enter a valid number")
  
  def _on_brightness_entry_change(self, event=None):
      """Handle brightness entry change."""
      try:
          value = float(self.brightness_entry.get())
          if 0.0 <= value <= 2.0:
              self.brightness_var.set(value)
              self.image_processor.brightness = value
              self._apply_settings()
          else:
              # Reset to valid value
              self.brightness_entry.delete(0, tk.END)
              self.brightness_entry.insert(0, f"{self.brightness_var.get():.2f}")
              messagebox.showwarning("Invalid Value", "Brightness must be between 0.0 and 2.0")
      except ValueError:
          # Reset to valid value
          self.brightness_entry.delete(0, tk.END)
          self.brightness_entry.insert(0, f"{self.brightness_var.get():.2f}")
          messagebox.showwarning("Invalid Value", "Please enter a valid number")
  
  def _on_saturation_entry_change(self, event=None):
      """Handle saturation entry change."""
      try:
          value = float(self.saturation_entry.get())
          if 0.0 <= value <= 2.0:
              self.saturation_var.set(value)
              self.image_processor.saturation = value
              self._apply_settings()
          else:
              # Reset to valid value
              self.saturation_entry.delete(0, tk.END)
              self.saturation_entry.insert(0, f"{self.saturation_var.get():.2f}")
              messagebox.showwarning("Invalid Value", "Saturation must be between 0.0 and 2.0")
      except ValueError:
          # Reset to valid value
          self.saturation_entry.delete(0, tk.END)
          self.saturation_entry.insert(0, f"{self.saturation_var.get():.2f}")
          messagebox.showwarning("Invalid Value", "Please enter a valid number")
  
  def _reset_settings(self):
      """Reset settings to default values."""
      self.contrast_var.set(1.0)
      self.brightness_var.set(1.0)
      self.saturation_var.set(1.0)
      self.negative_mode_var.set(False)
      
      self.image_processor.contrast = 1.0
      self.image_processor.brightness = 1.0
      self.image_processor.saturation = 1.0
      self.image_processor.negative_mode = False
      
      # Reset entry boxes
      self.contrast_entry.delete(0, tk.END)
      self.contrast_entry.insert(0, "1.00")
      self.brightness_entry.delete(0, tk.END)
      self.brightness_entry.insert(0, "1.00")
      self.saturation_entry.delete(0, tk.END)
      self.saturation_entry.insert(0, "1.00")
      
      self._apply_settings()
  
  def _apply_settings(self):
      """Apply the current settings."""
      # Update image processor settings
      self.image_processor.negative_mode = self.negative_mode_var.get()
      
      # Call apply callback if provided
      if self.apply_callback:
          self.apply_callback()
  
  def get_settings(self):
      """Get the current settings."""
      return {
          "negative_mode": self.negative_mode_var.get(),
          "contrast": self.contrast_var.get(),
          "brightness": self.brightness_var.get(),
          "saturation": self.saturation_var.get()
      }
