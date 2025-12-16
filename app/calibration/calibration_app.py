import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog, Frame, Label, Entry, StringVar
import matplotlib
matplotlib.use('TkAgg') # Ensure TkAgg backend is used for matplotlib with Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk
import cv2
import os
import re
import numpy as np
import csv
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Radiomic Film Calibration")
        self.root.geometry("1200x800")
        
        # Create main frames
        self.image_frame = Frame(root, width=800, height=800)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.list_frame = Frame(root, width=400, height=800)
        self.list_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create a frame to hold the image canvas and its scrollbars
        self.image_canvas_container = Frame(self.image_frame)
        self.image_canvas_container.pack(fill=tk.BOTH, expand=True)

        self.h_scrollbar = tk.Scrollbar(self.image_canvas_container, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scrollbar = tk.Scrollbar(self.image_canvas_container, orient=tk.VERTICAL)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_canvas = tk.Canvas(self.image_canvas_container, bg="lightgray",
                                       xscrollcommand=self.h_scrollbar.set,
                                       yscrollcommand=self.v_scrollbar.set)
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.h_scrollbar.config(command=self.image_canvas.xview)
        self.v_scrollbar.config(command=self.image_canvas.yview)

        self.image_canvas.bind("<Configure>", self._on_image_canvas_configure)
        self.image_canvas.bind("<MouseWheel>", self.zoom_image) # For Windows/Linux
        self.image_canvas.bind("<Button-4>", self.zoom_image) # For macOS
        self.image_canvas.bind("<Button-5>", self.zoom_image) # For macOS

        # Bind mouse events for panning
        # Pan/ROI drawing router
        self.image_canvas.bind("<ButtonPress-1>", self.on_canvas_press_router)
        self.image_canvas.bind("<B1-Motion>", self.on_canvas_drag_router)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_canvas_release_router)
        self.image_canvas.bind("<Motion>", self.preview_roi_on_motion)
        self.image_canvas.bind("<Leave>", self.on_mouse_leave_canvas) # Handles ROI preview and data display on leave

        self.original_pil_image = None
        self.current_tk_image = None
        self.zoom_factor = 1.0
        self.image_on_canvas = None
        self.zoom_timer = None # To manage delayed high-quality rendering
        self.preview_roi_id = None # To store the ID of the temporary ROI preview
        self.image_rois = {} # Stores ROIs for each image: {image_path: [{'shape': 'rect', 'size': 50, 'x_img': 100, 'y_img': 150, 'id': None}, ...]}
        self.plot_window = None # To keep track of the plot window
        self.current_image_path = None
        self.is_panning = False # Flag to indicate if panning is active
        # self.canvas_mode_var = tk.StringVar(value="pan") # Removed, ROI drawing is always active
        self.measured_data_labels = {} # To hold labels for displaying ROI stats
        self.csv_filename = "calibration_data.csv"
        # Remove ALL CSV files in the working directory to start with a clean run
        for fname in os.listdir('.'):
            if fname.lower().endswith('.csv'):
                try:
                    os.remove(fname)
                except Exception as e:
                    print(f"Warning: could not remove {fname}: {e}")
        self._initialize_csv_file()  # Create fresh calibration CSV
        self.excluded_points: set[tuple[str,int]] = set() # Set to keep track of excluded calibration points as (channel, index) tuples
        self._manual_override: dict[str, tuple[float,float,float]] = {} # store manual parameters keyed by channel when user overrides
        self.calibration_bit_depth = 8  # Default to 8-bit, updated when images are loaded

        # Initial default text. It will be updated by _on_image_canvas_configure
        self.image_canvas.create_text(1, 1, text="No image selected", font=("Arial", 24), fill="gray", tags="default_text")
        
        # Create scrollable frame for list items
        self.scroll_frame = Frame(self.list_frame)
        self.scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self.scroll_frame)
        self.scrollbar = tk.Scrollbar(self.scroll_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # ROI Controls
        self.roi_frame = tk.LabelFrame(self.list_frame, text="ROI Selection", padx=5, pady=5)
        self.roi_frame.pack(pady=10, padx=5, fill=tk.X)

        self.roi_shape_var = StringVar(value="rectangle") # Default to rectangle

        tk.Radiobutton(self.roi_frame, text="Rectangle", variable=self.roi_shape_var, value="rectangle").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(self.roi_frame, text="Circle", variable=self.roi_shape_var, value="circle").pack(side=tk.LEFT, padx=5)

        tk.Label(self.roi_frame, text="Size (px):").pack(side=tk.LEFT, padx=5)
        self.roi_size_entry = Entry(self.roi_frame, width=8)
        self.roi_size_entry.insert(0, "50")

        self.roi_size_entry.pack(side=tk.LEFT, padx=5)

        # Auto-advance checkbox now sits with ROI controls
        self.auto_advance_var = tk.BooleanVar(value=True)
        self.auto_advance_checkbox = tk.Checkbutton(self.roi_frame, text="Auto-advance", variable=self.auto_advance_var)
        self.auto_advance_checkbox.pack(side=tk.LEFT, padx=5)

        # Frame for displaying ROI measurement data
        self.roi_data_frame = tk.LabelFrame(self.list_frame, text="ROI Data")
        self.roi_data_frame.pack(pady=10, padx=10, fill="x")
        self._create_roi_data_labels(self.roi_data_frame)
        self._clear_roi_data_labels() # Initialize with empty data
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Load button
        self.load_button = tk.Button(self.list_frame, text="Load Images", command=self.load_images)
        self.load_button.pack(pady=10)

        self.delete_button = tk.Button(self.list_frame, text="Delete Selected Image", command=self.delete_selected_image)
        self.delete_button.pack(pady=5) # Reduced pady for spacing

        self.add_images_button = tk.Button(self.list_frame, text="Add More Images", command=self.add_images)
        self.add_images_button.pack(pady=5)

        self.calibrate_button = tk.Button(self.list_frame, text="Calibrate", command=self.perform_calibration, state=tk.DISABLED)
        self.calibrate_button.pack(pady=5)

        # Initialize variables
        self.image_files = []
        self.current_image = None
        self.gray_values = []
        self.list_item_widgets = []
        self.selected_idx = -1  # Track currently selected index
        self.dose_entries = []
        
    def load_images(self):
        files = filedialog.askopenfilenames(
            title="Select TIFF Images",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if files:
            self.image_files = []  # Clear existing files
            self.gray_values = []  # Clear existing gray values
            for f_path in files:
                self.image_files.append(f_path)
                base_name = os.path.basename(f_path)
                parsed_dose = self._parse_dose_from_filename(base_name)
                self.gray_values.append(parsed_dose if parsed_dose is not None else "0.0")
            
            self._populate_image_list() # This will create StringVars and UI
            if self.image_files: # If images were loaded
                self.select_image(0)
            else: # No images loaded or all were filtered out, clear selection and display
                self.current_image_path = None
                # self.image_canvas.delete("all") # select_image(-1) or similar should handle this
                self.selected_idx = -1
                self._clear_roi_data_labels()
                # Ensure canvas is cleared if no images are loaded
                if hasattr(self, 'image_canvas'): self.image_canvas.delete("all")

    def add_images(self):
        """Allows adding more images to the existing list."""
        files = filedialog.askopenfilenames(
            title="Select Additional TIFF Images",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if files:
            newly_added_count = 0
            first_new_idx = -1 # To select the first successfully added image
            
            for f_path in files:
                if f_path not in self.image_files: # Avoid adding exact same path duplicates
                    if first_new_idx == -1:
                        first_new_idx = len(self.image_files) # Index where the first new image will be added
                    self.image_files.append(f_path)
                    base_name = os.path.basename(f_path)
                    parsed_dose = self._parse_dose_from_filename(base_name)
                    self.gray_values.append(parsed_dose if parsed_dose is not None else "0.0")
                    newly_added_count += 1
                else:
                    print(f"Image {f_path} already in list. Skipping.")
            
            if newly_added_count > 0: 
                self._populate_image_list() # This will update UI and dose_entries
                if first_new_idx != -1: # If new images were added
                    self.select_image(first_new_idx)
                # If no new images were added but files were selected (all duplicates), current selection is maintained.
            # If no files were selected from dialog, also do nothing to current selection.

    def _parse_dose_from_filename(self, filename):
        # Regex to find a number (integer or float) followed by 'Gy', case-insensitive
        # It captures the number part. Examples: "file_2.5Gy.tif" -> "2.5", "image_10Gy_scan.dcm" -> "10"
        match = re.search(r"(\d+(?:\.\d+)?)Gy", filename, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def delete_selected_image(self):
        if self.selected_idx < 0 or self.selected_idx >= len(self.image_files):
            messagebox.showwarning("Delete Image", "No image selected to delete.")
            return

        image_path_to_delete = self.image_files[self.selected_idx]
        image_name_to_delete = os.path.basename(image_path_to_delete)

        confirm = messagebox.askyesno("Confirm Delete", 
                                      f"Are you sure you want to delete '{image_name_to_delete}'?\n"
                                      f"This will remove it from the list and the CSV file.")
        if not confirm:
            return

        original_selected_idx = self.selected_idx # Store for intelligent re-selection
        
        # Remove from internal data structures
        del self.image_files[self.selected_idx]
        if image_path_to_delete in self.image_rois:
            del self.image_rois[image_path_to_delete]

        # Remove from CSV file
        if os.path.exists(self.csv_filename):
            all_data_rows = []
            header = [] # To store the header row if found
            try:
                with open(self.csv_filename, 'r', newline='') as f_read:
                    reader = csv.reader(f_read)
                    try:
                        header = next(reader) # Attempt to read the first line as header
                    except StopIteration: # File is empty
                        pass # header remains empty, no data rows to read

                    for row in reader: # Read remaining lines as data rows
                        if row and len(row) > 0 and row[0] != image_name_to_delete:
                            all_data_rows.append(row)
                
                # Write back to CSV: header (if it existed) then filtered data rows
                with open(self.csv_filename, 'w', newline='') as f_write:
                    writer = csv.writer(f_write)
                    if header: # If a header was read (even if it was an empty list from an empty file initially)
                        writer.writerow(header)
                    if all_data_rows:
                        writer.writerows(all_data_rows)
                    # If header was empty and all_data_rows is empty, an empty file is written.
                    # If header was present but all_data_rows is empty, only header is written.
                print(f"CSV updated after attempting to remove '{image_name_to_delete}'.")
            except Exception as e:
                messagebox.showerror("Delete Image Error", f"Error updating CSV: {e}")
        
        # Refresh the image list in the GUI
        self._populate_image_list()
        
        # Attempt to re-select an appropriate item
        if not self.image_files: # List is now empty
            # _populate_image_list handles clearing display if selected_idx becomes -1
            pass
        else: # List is not empty, try to select a new item
            new_selection_idx = min(original_selected_idx, len(self.image_files) - 1)
            # Ensure new_selection_idx is valid (e.g., 0 if original_selected_idx was 0 and list still has items)
            if new_selection_idx < 0: new_selection_idx = 0 
            self.select_image(new_selection_idx)

        print(f"Image '{image_name_to_delete}' deleted.")

    def _populate_image_list(self):
        # 1. Clear existing widgets from the scrollable frame
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # 2. Re-initialize lists that are populated by this method
        self.dose_entries = []  # Stores StringVars for dose entries
        self.list_item_widgets = [] # Stores the item_frame widgets for selection highlighting

        # 3. Populate the list based on the current self.image_files
        for idx, file_path in enumerate(self.image_files):
            image_name = os.path.basename(file_path)
            
            item_frame = tk.Frame(self.scrollable_frame, borderwidth=1, relief="raised")
            item_frame.pack(fill=tk.X, pady=1, padx=1)
            
            item_frame.bind("<Button-1>", lambda e, index=idx: self.select_image(index))

            label = tk.Label(item_frame, text=image_name, anchor="w", width=30)
            label.pack(side=tk.LEFT, padx=5, pady=2)
            label.bind("<Button-1>", lambda e, index=idx: self.select_image(index))

            base_name = os.path.basename(file_path)
            parsed_dose = self._parse_dose_from_filename(base_name)
            initial_dose_value = parsed_dose if parsed_dose is not None else "0.0"
            
            dose_var = tk.StringVar(value=initial_dose_value)

            entry = tk.Entry(item_frame, textvariable=dose_var, width=8)
            entry.pack(side=tk.RIGHT, padx=5, pady=2)
            entry.bind("<FocusIn>", lambda e, index=idx: self.select_image(index, set_focus_to_canvas=False))

            # Use the current loop index 'idx' for on_dose_changed
            dose_var.trace_add("write", lambda name, index_lambda, mode, var=dose_var, current_idx=idx: self.on_dose_changed(var, current_idx))
            self.dose_entries.append(dose_var)
            self.list_item_widgets.append(item_frame)

        # 4. Handle selection state and UI updates after repopulating
        if self.selected_idx >= len(self.image_files): 
            self.selected_idx = -1 
            if self.current_image_path: 
                self._clear_roi_data_labels()
                self.image_canvas.delete("all")
                self.original_pil_image = None
                self.current_image_path = None
                self.display_default_canvas_text()
        
        self.canvas.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all")) 
        self._highlight_selected_item()
        self._check_calibration_readiness() # Check after repopulating list

    def on_dose_changed(self, dose_var, img_idx):
        try:
            if not (0 <= img_idx < len(self.image_files)):
                print(f"on_dose_changed: Invalid img_idx {img_idx}")
                return

            new_dose_value = dose_var.get()
            # Ensure image_files[img_idx] is valid before accessing
            if img_idx >= len(self.image_files):
                 print(f"Error: img_idx {img_idx} out of range for self.image_files in on_dose_changed")
                 return
            image_name = os.path.basename(self.image_files[img_idx])
            print(f"Dose changed for {image_name} (index {img_idx}) to: '{new_dose_value}'")

            # Update the internal gray_values list
            if img_idx < len(self.gray_values):
                self.gray_values[img_idx] = new_dose_value
            else:
                print(f"Warning: img_idx {img_idx} out of bounds for gray_values in on_dose_changed.")

            # Update CSV with the new dose value
            self._update_dose_in_csv(image_name, new_dose_value)
            # print(f"CSV update for {image_name} (idx {img_idx}) was called in on_dose_changed.") # Optional: keep for debugging
        except Exception as e:
            img_name_for_log = image_name if 'image_name' in locals() else f'unknown (img_idx: {img_idx})'
            print(f"Error in on_dose_changed for image {img_name_for_log}: {e}")
            import traceback
            traceback.print_exc()

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
        
        # For float images
        if np.issubdtype(dtype, np.floating):
            if actual_max <= 1.0:
                return 8, 1.0  # Normalized float
            elif actual_max <= 255:
                return 8, 255
            elif actual_max <= 4095:
                return 12, 4095
            elif actual_max <= 16383:
                return 14, 16383
            elif actual_max <= 65535:
                return 16, 65535
            else:
                return 32, actual_max
        
        # For 16-bit integer types, detect actual bit depth from values
        if dtype == np.uint16:
            if actual_max <= 1023:
                return 10, 1023
            elif actual_max <= 4095:
                return 12, 4095
            elif actual_max <= 16383:
                return 14, 16383
            else:
                return 16, 65535
        
        # For 32-bit integer types
        if dtype == np.uint32:
            if actual_max <= 255:
                return 8, 255
            elif actual_max <= 65535:
                return 16, 65535
            elif actual_max <= 16777215:
                return 24, 16777215
            else:
                return 32, 4294967295
        
        # Default: 8-bit
        return 8, 255

    def _highlight_selected_item(self):
        """Updates the visual highlight of the selected item in the list."""
        for i, item_frame in enumerate(self.list_item_widgets):
            if i == self.selected_idx:
                item_frame.config(relief="raised", bg="lightblue") # Highlight with relief and background
                for child in item_frame.winfo_children():
                    if not isinstance(child, tk.Entry): # Don't change background of Entry widgets
                        child.config(bg="lightblue")
            else:
                # Use the default background of the scrollable_frame's master or a standard color
                default_bg = self.scrollable_frame.cget("bg") 
                item_frame.config(relief="sunken", bg=default_bg)
                for child in item_frame.winfo_children():
                    if not isinstance(child, tk.Entry):
                        child.config(bg=default_bg)
        
    def select_image(self, idx, set_focus_to_canvas=True):
        if not (0 <= idx < len(self.image_files)):
            print(f"Error: Attempted to select invalid image index {idx}")
            return

        self.selected_idx = idx
        self._highlight_selected_item()
            
        # Set focus to the image canvas so it can receive mouse events
        if set_focus_to_canvas:
            self.image_canvas.focus_set()

        image_path = self.image_files[idx]
        self.current_image_path = image_path # Set current image path here
        try:
            # Use OpenCV to load image preserving 16-bit depth
            cv_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if cv_img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            if len(cv_img.shape) == 3:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            # Store the original numpy array for measurements (preserves original bit depth)
            self.original_image_array = cv_img.copy()
            
            # Detect bit depth from numpy dtype and actual values
            self.calibration_bit_depth, self.calibration_max_value = self._detect_bit_depth(cv_img)
            print(f"Detected {self.calibration_bit_depth}-bit image (dtype={cv_img.dtype}): {image_path}")
            print(f"  Value range: min={cv_img.min()}, max={cv_img.max()}, max_possible={self.calibration_max_value}")
            
            # Convert to 8-bit for PIL display only
            if cv_img.dtype != np.uint8:
                scale_factor = 255.0 / self.calibration_max_value
                display_img = (cv_img * scale_factor).astype(np.uint8)
            else:
                display_img = cv_img
            
            # Create PIL image for display
            pil_img = Image.fromarray(display_img)
            
            pil_img = self._crop_white_border(pil_img)
            self.original_pil_image = pil_img  # Store processed image for display
            
            # Also crop the numpy array to match
            if hasattr(self, 'last_crop_bbox') and self.last_crop_bbox:
                left, top, right, bottom = self.last_crop_bbox
                self.original_image_array = self.original_image_array[top:bottom, left:right]

            # Calculate initial zoom_factor to fit the image
            # Ensure canvas has correct dimensions before calculating fit
            self.image_canvas.update_idletasks()
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            img_width, img_height = self.original_pil_image.size

            if img_width > 0 and img_height > 0 and canvas_width > 0 and canvas_height > 0:
                scale_w = canvas_width / img_width
                scale_h = canvas_height / img_height
                self.zoom_factor = min(scale_w, scale_h)
            else:
                self.zoom_factor = 1.0 # Fallback if dimensions are zero

            self.image_canvas.delete("roi") # Clear any ROIs from a previous image
            self.display_image_on_canvas(filter=Image.LANCZOS)

            # Redraw stored ROI for this image if it exists, and update display
            if self.current_image_path in self.image_rois and self.image_rois[self.current_image_path]:
                stored_roi_data = self.image_rois[self.current_image_path][0]
                self._draw_single_roi(stored_roi_data) # Redraw it on the canvas
                self._update_roi_data_display(stored_roi_data) # Update the data panel
            else:
                self._clear_roi_data_labels() # Clear data panel if no ROI for this image

        except Exception as e:
            self.image_canvas.delete("all")
            self.image_canvas.create_text(self.image_canvas.winfo_width()/2, self.image_canvas.winfo_height()/2,
                                          text=f"Error loading image: {e}", font=("Arial", 16), fill="red")

    def _on_image_canvas_configure(self, event):
        if self.original_pil_image:
            self.display_image_on_canvas()
        else:
            # Update default text position if no image is loaded
            self.image_canvas.coords("default_text", event.width/2, event.height/2)

    def display_image_on_canvas(self, filter=Image.BILINEAR):
        if self.original_pil_image is None:
            return

        self.image_canvas.delete("all") # Clear previous image and text

        # Calculate scaled dimensions
        width, height = self.original_pil_image.size
        new_width = int(width * self.zoom_factor)
        new_height = int(height * self.zoom_factor)

        # Resize image using specified filter
        resized_image = self.original_pil_image.resize((new_width, new_height), filter)
        self.current_tk_image = ImageTk.PhotoImage(resized_image)

        # Place image at the center of the canvas initially, or maintain current view
        # For simplicity, let's place it at (0,0) and let scrollbars handle the view
        self.image_on_canvas = self.image_canvas.create_image(0, 0, anchor="nw", image=self.current_tk_image)

        # Update scroll region to match image size
        self.image_canvas.config(scrollregion=self.image_canvas.bbox(tk.ALL))

    def zoom_image(self, event):
        if self.original_pil_image is None:
            return

        # Clear any pending high-quality render
        if self.zoom_timer:
            self.root.after_cancel(self.zoom_timer)

        # Determine zoom direction
        # event.delta is for Windows/Linux, event.num is for macOS
        if event.delta > 0 or event.num == 4: # Zoom in
            self.zoom_factor *= 1.02 # Even smaller increment for smoother zoom
        elif event.delta < 0 or event.num == 5: # Zoom out
            self.zoom_factor /= 1.02 # Even smaller increment for smoother zoom

        # Limit zoom factor
        self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))

        # Render with a faster filter during active zooming
        self.display_image_on_canvas(filter=Image.BILINEAR)

        # Schedule a high-quality render after a short delay (e.g., 50ms)
        self.zoom_timer = self.root.after(50, lambda: self.display_image_on_canvas(filter=Image.LANCZOS))

    def start_pan(self, event):
        self.image_canvas.scan_mark(event.x, event.y)

    def pan_image(self, event):
        self.image_canvas.scan_dragto(event.x, event.y, gain=1)

    def on_canvas_press_router(self, event):
        # Check for Ctrl key (state mask 0x0004 for Ctrl)
        if event.state & 0x0004:
            self.start_pan(event)
        else:
            self.draw_roi_on_click(event)

    def on_canvas_drag_router(self, event):
        if self.is_panning:
            self.pan_image(event)
        # If not panning, <Motion> binding to preview_roi_on_motion will handle ROI preview

    def on_canvas_release_router(self, event):
        if self.is_panning:
            self.stop_pan(event)

    # --- Panning methods ---
    def start_pan(self, event):
        self.image_canvas.scan_mark(event.x, event.y)
        self.is_panning = True
        self.clear_roi_preview() # Clear ROI preview when starting pan

    def pan_image(self, event):
        self.image_canvas.scan_dragto(event.x, event.y, gain=1)

    def stop_pan(self, event):
        self.is_panning = False
    # --- End Panning methods ---

    def preview_roi_on_motion(self, event):
        if self.is_panning: # Don't show ROI preview if panning
            self.clear_roi_preview()
            return
            
        # ROI preview is always active now (if not panning)
        if self.original_pil_image is None:
            self.clear_roi_preview() # Clear if no image
            self._clear_roi_data_labels() # Clear data display too
            return

        self.clear_roi_preview()

        canvas_x = self.image_canvas.canvasx(event.x)
        canvas_y = self.image_canvas.canvasy(event.y)

        try:
            roi_size = int(self.roi_size_entry.get())
        except ValueError:
            self._clear_roi_data_labels() # Clear data if size is invalid
            return # Don't draw preview if size is invalid

        roi_shape = self.roi_shape_var.get()

        # Calculate preview box in canvas coordinates
        preview_x1_canvas = canvas_x - roi_size / 2
        preview_y1_canvas = canvas_y - roi_size / 2
        preview_x2_canvas = canvas_x + roi_size / 2
        preview_y2_canvas = canvas_y + roi_size / 2

        if roi_shape == "rectangle":
            self.preview_roi_id = self.image_canvas.create_rectangle(preview_x1_canvas, preview_y1_canvas, preview_x2_canvas, preview_y2_canvas, outline="red", width=2, dash=(2, 2), tags="roi_preview")
        elif roi_shape == "circle":
            self.preview_roi_id = self.image_canvas.create_oval(preview_x1_canvas, preview_y1_canvas, preview_x2_canvas, preview_y2_canvas, outline="red", width=2, dash=(2, 2), tags="roi_preview")

        # Dynamically update ROI data display with preview info
        if self.zoom_factor != 0:
            preview_x_img_orig = canvas_x / self.zoom_factor
            preview_y_img_orig = canvas_y / self.zoom_factor
            temp_roi_data = {
                'shape': roi_shape,
                'size': roi_size,
                'x_img': preview_x_img_orig,
                'y_img': preview_y_img_orig,
                # Actual measurements (meanR, etc.) are not done for live preview yet
            }
            self._update_roi_data_display(temp_roi_data)
        else:
            self._clear_roi_data_labels()

    def clear_roi_preview(self, event=None):
        if self.preview_roi_id:
            self.image_canvas.delete(self.preview_roi_id)
            self.preview_roi_id = None

    def _draw_single_roi(self, roi_data):
        # Helper to draw a single ROI on the canvas based on stored data
        shape = roi_data['shape']
        size = roi_data['size']
        x_img = roi_data['x_img']
        y_img = roi_data['y_img']

        # Convert image coordinates to current canvas coordinates (considering zoom)
        # This assumes x_img, y_img are relative to the original image (0,0)
        # and not already scaled by zoom_factor
        current_x = x_img * self.zoom_factor
        current_y = y_img * self.zoom_factor

        x1 = current_x - size / 2
        y1 = current_y - size / 2
        x2 = current_x + size / 2
        y2 = current_y + size / 2

        if shape == "rectangle":
            roi_data['canvas_id'] = self.image_canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=3, tags="roi")
        elif shape == "circle":
            roi_data['canvas_id'] = self.image_canvas.create_oval(x1, y1, x2, y2, outline="green", width=3, tags="roi")

    def _initialize_csv_file(self):
        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ImageName", "Dose", "ROIShape", "ROISize",
                                 "ROIX_orig", "ROIY_orig", "MeanR", "MeanG", "MeanB",
                                 "StdDevR", "StdDevG", "StdDevB", "NumPixels"])

    def draw_roi_on_click(self, event):
        if self.original_pil_image is None or self.current_image_path is None:
            print("No image selected or current_image_path is not set.")
            return

        canvas_x = self.image_canvas.canvasx(event.x)
        canvas_y = self.image_canvas.canvasy(event.y)
        
        if self.zoom_factor == 0: # Avoid division by zero if zoom_factor is somehow 0
            print("Error: Zoom factor is zero.")
            return
            
        x_img_orig = canvas_x / self.zoom_factor
        y_img_orig = canvas_y / self.zoom_factor

        roi_shape = self.roi_shape_var.get()
        try:
            roi_size = int(self.roi_size_entry.get())
        except ValueError:
            print("Invalid ROI size. Please enter an integer.")
            return

        new_roi_data = {
            'shape': roi_shape,
            'size': roi_size,
            'x_img': x_img_orig,
            'y_img': y_img_orig,
            'canvas_id': None # Will be set by _draw_single_roi
        }

        # Enforce single ROI per image:
        # Clear any existing ROI for the current image from data and canvas
        if self.current_image_path in self.image_rois:
            for old_roi in self.image_rois[self.current_image_path]: # Should be a list with 0 or 1 item
                if old_roi.get('canvas_id'):
                    self.image_canvas.delete(old_roi['canvas_id'])
        
        # Store the new ROI (as a list containing the single ROI dictionary)
        self.image_rois[self.current_image_path] = [new_roi_data]

        # Draw the new ROI
        self._draw_single_roi(new_roi_data)
        
        # Update the displayed data (basic info for now)
        self._update_roi_data_display(new_roi_data)
        
        # Perform measurement and save data (to be fully implemented)
        self._measure_and_save_roi_data(new_roi_data)

    def _measure_and_save_roi_data(self, roi_data):
        if self.original_pil_image is None or self.current_image_path is None or self.selected_idx < 0:
            print("Cannot measure ROI: No image, path, or selection index.")
            return

        try:
            dose_value_str = self.dose_entries[self.selected_idx].get()
            # Attempt to convert to float, handle if it's already a float or int, or invalid
            try:
                dose_value = float(dose_value_str)
            except ValueError:
                if isinstance(dose_value_str, (int, float)):
                    dose_value = dose_value_str # Already a number
                else:
                    print(f"Invalid dose value: {dose_value_str}")
                    dose_value = "N/A" # Or some other placeholder
        except IndexError:
            print("Selected index for dose is out of bounds.")
            dose_value = "N/A"

        pil_image = self.original_pil_image
        # Use the numpy array that preserves original bit depth (16-bit if applicable)
        if hasattr(self, 'original_image_array') and self.original_image_array is not None:
            img_array = self.original_image_array
            print(f"DEBUG ROI: Using original_image_array with dtype={img_array.dtype}, shape={img_array.shape}")
            if img_array.dtype == np.uint16:
                print(f"DEBUG ROI: 16-bit array, value range: min={img_array.min()}, max={img_array.max()}")
        else:
            img_array = np.array(pil_image)
            print(f"DEBUG ROI: Fallback to pil_image array with dtype={img_array.dtype}")
        img_h, img_w = img_array.shape[:2]

        x_center_orig = roi_data['x_img']
        y_center_orig = roi_data['y_img']
        roi_size = roi_data['size']
        roi_shape = roi_data['shape']

        # Define ROI boundaries in original image coordinates
        x_min_orig = int(x_center_orig - roi_size / 2)
        y_min_orig = int(y_center_orig - roi_size / 2)
        x_max_orig = int(x_center_orig + roi_size / 2)
        y_max_orig = int(y_center_orig + roi_size / 2)

        # Clip to image dimensions
        x_min_clipped = max(0, x_min_orig)
        y_min_clipped = max(0, y_min_orig)
        x_max_clipped = min(img_w, x_max_orig)
        y_max_clipped = min(img_h, y_max_orig)

        if x_min_clipped >= x_max_clipped or y_min_clipped >= y_max_clipped:
            print("ROI is outside image bounds or has zero area after clipping.")
            # Update roi_data with N/A and update display
            roi_data.update({'mean_r': 'N/A', 'mean_g': 'N/A', 'mean_b': 'N/A',
                             'std_r': 'N/A', 'std_g': 'N/A', 'std_b': 'N/A',
                             'num_pixels': 0})
            self._update_roi_data_display(roi_data)
            # Optionally write N/A to CSV or skip
            return

        # Extract the rectangular region containing the ROI
        region_pixels = img_array[y_min_clipped:y_max_clipped, x_min_clipped:x_max_clipped]

        pixels_in_roi = []
        if roi_shape == 'rectangle':
            if region_pixels.size > 0:
                pixels_in_roi = region_pixels.reshape(-1, region_pixels.shape[-1]) # Assuming last dim is color channels
        elif roi_shape == 'circle':
            radius_sq = (roi_size / 2)**2
            # Center of the ROI in original image coordinates
            # Iterate through the clipped rectangular bounding box
            for r_idx in range(y_min_clipped, y_max_clipped):
                for c_idx in range(x_min_clipped, x_max_clipped):
                    # Distance squared from current pixel (r_idx, c_idx) to ROI center (y_center_orig, x_center_orig)
                    dist_sq = (r_idx - y_center_orig)**2 + (c_idx - x_center_orig)**2
                    if dist_sq <= radius_sq:
                        pixels_in_roi.append(img_array[r_idx, c_idx])
            if pixels_in_roi:
                pixels_in_roi = np.array(pixels_in_roi)
            else:
                pixels_in_roi = np.empty((0, img_array.shape[-1])) # Ensure correct shape for empty array

        if not pixels_in_roi.any() or pixels_in_roi.shape[0] == 0:
            print("No pixels found in ROI after shape processing.")
            roi_data.update({'mean_r': 'N/A', 'mean_g': 'N/A', 'mean_b': 'N/A',
                             'std_r': 'N/A', 'std_g': 'N/A', 'std_b': 'N/A',
                             'num_pixels': 0})
        else:
            num_pixels = pixels_in_roi.shape[0]
            mean_rgb = np.mean(pixels_in_roi, axis=0)
            std_dev_rgb = np.std(pixels_in_roi, axis=0)
            print(f"DEBUG ROI: pixels_in_roi dtype={pixels_in_roi.dtype}, mean_rgb={mean_rgb}")
            roi_data.update({
                'mean_r': mean_rgb[0] if len(mean_rgb) > 0 else 'N/A',
                'mean_g': mean_rgb[1] if len(mean_rgb) > 1 else 'N/A',
                'mean_b': mean_rgb[2] if len(mean_rgb) > 2 else 'N/A',
                'std_r': std_dev_rgb[0] if len(std_dev_rgb) > 0 else 'N/A',
                'std_g': std_dev_rgb[1] if len(std_dev_rgb) > 1 else 'N/A',
                'std_b': std_dev_rgb[2] if len(std_dev_rgb) > 2 else 'N/A',
                'num_pixels': num_pixels
            })

        # Update the UI display with the new measurements
        self._update_roi_data_display(roi_data)

        # Save to CSV (Read-Modify-Write to handle overwrites)
        image_name = os.path.basename(self.current_image_path)
        new_csv_row_values = [
            image_name,
            dose_value,
            roi_data.get('shape', 'N/A'),
            roi_data.get('size', 'N/A'),
            f"{roi_data.get('x_img', 0):.2f}",
            f"{roi_data.get('y_img', 0):.2f}",
            f"{roi_data.get('mean_r', 'N/A'):.2f}" if isinstance(roi_data.get('mean_r'), (int, float)) else roi_data.get('mean_r', 'N/A'),
            f"{roi_data.get('mean_g', 'N/A'):.2f}" if isinstance(roi_data.get('mean_g'), (int, float)) else roi_data.get('mean_g', 'N/A'),
            f"{roi_data.get('mean_b', 'N/A'):.2f}" if isinstance(roi_data.get('mean_b'), (int, float)) else roi_data.get('mean_b', 'N/A'),
            f"{roi_data.get('std_r', 'N/A'):.2f}" if isinstance(roi_data.get('std_r'), (int, float)) else roi_data.get('std_r', 'N/A'),
            f"{roi_data.get('std_g', 'N/A'):.2f}" if isinstance(roi_data.get('std_g'), (int, float)) else roi_data.get('std_g', 'N/A'),
            f"{roi_data.get('std_b', 'N/A'):.2f}" if isinstance(roi_data.get('std_b'), (int, float)) else roi_data.get('std_b', 'N/A'),
            roi_data.get('num_pixels', 'N/A')
        ]

        updated_rows = []
        header = ["ImageName", "Dose", "ROIShape", "ROISize", "ROIX_orig", "ROIY_orig", 
                  "MeanR", "MeanG", "MeanB", "StdDevR", "StdDevG", "StdDevB", "NumPixels"]
        found = False

        if os.path.exists(self.csv_filename):
            try:
                with open(self.csv_filename, 'r', newline='') as f_read:
                    reader = csv.reader(f_read)
                    try:
                        header = next(reader) # Read existing header
                    except StopIteration: # Empty file
                        pass # Use default header
                    updated_rows.append(header)
                    for row in reader: # Read remaining lines as data rows
                        if row and row[0] == image_name: # Check if image name matches
                            updated_rows.append(new_csv_row_values) # Replace with new data
                            found = True
                        elif row: # Keep other rows
                            updated_rows.append(row)
            except Exception as e:
                print(f"Error reading CSV for update: {e}")
                # Fallback to simple append or error out, here we'll try to ensure header is there
                if not updated_rows: updated_rows.append(header)
        else:
            updated_rows.append(header) # CSV doesn't exist, start with header

        if not found:
            updated_rows.append(new_csv_row_values) # Append if image_name was not found

        try:
            with open(self.csv_filename, 'w', newline='') as f_write:
                writer = csv.writer(f_write)
                writer.writerows(updated_rows)
            print(f"ROI data saved/updated in {self.csv_filename}")
        except Exception as e:
            print(f"Error writing updated ROI data to CSV: {e}")
        finally:
            self._check_calibration_readiness() # Update calibrate button state

        # Auto-advance logic
        if self.auto_advance_var.get():
            if self.selected_idx is not None and (self.selected_idx + 1) < len(self.image_files):
                next_image_index = self.selected_idx + 1
                # Schedule the selection of the next image to allow current event processing to complete
                self.root.after_idle(lambda idx=next_image_index: self.select_image(idx))
            elif self.selected_idx is not None: # If it was the last image or only one image
                # Optionally, provide feedback or loop to the first image
                print("Auto-advance: Reached end of image list or no further image to advance to.")

    def _create_roi_data_labels(self, parent_frame):
        self.measured_data_labels = {}
        labels_info = [
            ("Shape:", "N/A"), ("Size:", "N/A"), ("X:", "N/A"), ("Y:", "N/A"),
            ("Mean R:", "N/A"), ("Mean G:", "N/A"), ("Mean B:", "N/A"),
            ("Std R:", "N/A"), ("Std G:", "N/A"), ("Std B:", "N/A"),
            ("Pixels:", "N/A")
        ]
        for i, (text, val) in enumerate(labels_info):
            Label(parent_frame, text=text).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            data_label = Label(parent_frame, text=val)
            data_label.grid(row=i, column=1, sticky='w', padx=5, pady=2)
            self.measured_data_labels[text.replace(":", "").lower().replace(" ", "")] = data_label

    def _check_calibration_readiness(self):
        if not self.image_files: # No images loaded
            self.calibrate_button.config(state=tk.DISABLED)
            return

        all_measured = True
        for img_path in self.image_files:
            if img_path not in self.image_rois or not self.image_rois[img_path]:
                all_measured = False
                break
            # Further check if ROI data actually contains mean values
            roi_data = self.image_rois[img_path][0]
            if not all(k in roi_data for k in ("mean_r", "mean_g", "mean_b")):
                all_measured = False
                break
            # Check if values are numeric (not 'N/A')
            if not all(isinstance(roi_data.get(k), (int, float)) for k in ("mean_r", "mean_g", "mean_b")):
                all_measured = False
                break

        if all_measured:
            self.calibrate_button.config(state=tk.NORMAL)
        else:
            self.calibrate_button.config(state=tk.DISABLED)

    def perform_calibration(self):
        """Open the fit calibration window directly."""
        # Check if we have enough data
        doses = []
        for idx, img_path in enumerate(self.image_files):
            try:
                dose_str = self.gray_values[idx]
                dose = float(dose_str)
            except (ValueError, IndexError):
                continue

            if img_path not in self.image_rois or not self.image_rois[img_path]:
                continue
            
            roi_data = self.image_rois[img_path][0]
            try:
                float(roi_data['mean_r'])
                float(roi_data['mean_g'])
                float(roi_data['mean_b'])
            except (KeyError, ValueError, TypeError):
                continue
            
            doses.append(dose)

        if len(doses) < 2:
            messagebox.showinfo("Calibration", "Not enough valid data points (minimum 2 required) with both dose and measured RGB values.")
            return

        # Open fit window directly
        self.open_fit_window()

    def _on_close_plot_window(self):
        if self.plot_window:
            self.plot_window.destroy()
            self.plot_window = None

    def _clear_roi_data_labels(self):
        if not self.measured_data_labels: # If called before labels are created
            return
        default_values = {
            'shape': "N/A", 'size': "N/A", 'x': "N/A", 'y': "N/A",
            'meanr': "N/A", 'meang': "N/A", 'meanb': "N/A",
            'stdr': "N/A", 'stdg': "N/A", 'stdb': "N/A",
            'pixels': "N/A"
        }
        for key, val in default_values.items():
            if key in self.measured_data_labels:
                 self.measured_data_labels[key].config(text=val)

    def _update_roi_data_display(self, roi_data):
        if not self.measured_data_labels: # If called before labels are created
            return
        self.measured_data_labels['shape'].config(text=str(roi_data.get('shape', 'N/A')))
        self.measured_data_labels['size'].config(text=str(roi_data.get('size', 'N/A')))
        self.measured_data_labels['x'].config(text=f"{roi_data.get('x_img', 0):.2f}")
        self.measured_data_labels['y'].config(text=f"{roi_data.get('y_img', 0):.2f}")
        self.measured_data_labels['meanr'].config(text=f"{roi_data.get('mean_r', 'N/A'):.2f}" if isinstance(roi_data.get('mean_r'), (int, float)) else "N/A")
        self.measured_data_labels['meang'].config(text=f"{roi_data.get('mean_g', 'N/A'):.2f}" if isinstance(roi_data.get('mean_g'), (int, float)) else "N/A")
        self.measured_data_labels['meanb'].config(text=f"{roi_data.get('mean_b', 'N/A'):.2f}" if isinstance(roi_data.get('mean_b'), (int, float)) else "N/A")
        self.measured_data_labels['stdr'].config(text=f"{roi_data.get('std_r', 'N/A'):.2f}" if isinstance(roi_data.get('std_r'), (int, float)) else "N/A")
        self.measured_data_labels['stdg'].config(text=f"{roi_data.get('std_g', 'N/A'):.2f}" if isinstance(roi_data.get('std_g'), (int, float)) else "N/A")
        self.measured_data_labels['stdb'].config(text=f"{roi_data.get('std_b', 'N/A'):.2f}" if isinstance(roi_data.get('std_b'), (int, float)) else "N/A")
        self.measured_data_labels['pixels'].config(text=str(roi_data.get('num_pixels', 'N/A')))

    def on_mouse_leave_canvas(self, event):
        """Handles mouse leaving the canvas: clears visual preview and shows measured ROI data or clears display."""
        self.image_canvas.delete("roi_preview_shape") # Clear the visual preview rectangle

        if self.current_image_path and self.current_image_path in self.image_rois and self.image_rois[self.current_image_path]:
            # A measurement exists for this image, display its data
            final_roi_data = self.image_rois[self.current_image_path][0]
            self._update_roi_data_display(final_roi_data)
        elif self.current_image_path:
             # Current image exists, but no finalized ROI for it. Clear data labels.
             self._clear_roi_data_labels()
        else:
            # No current image selected at all. Clear data labels.
            self._clear_roi_data_labels()

    # ------------------ CSV helper ------------------ #
    def _update_dose_in_csv(self, image_name: str, new_dose):
        """Update the dose value for a given image in calibration_data.csv."""
        fname = self.csv_filename
        if not os.path.exists(fname):
            return  # Nothing to update yet

        try:
            rows = []
            with open(fname, newline='') as f:
                rdr = csv.reader(f)
                for row in rdr:
                    if row and row[0] == image_name:
                        row[1] = str(new_dose)
                    rows.append(row)

            with open(fname, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerows(rows)
        except Exception as e:
            print(f"Error updating CSV for {image_name}: {e}")

    # ---------------- Fit Data window ---------------- #
    def _get_calibration_data(self):
        """Return doses, means, and std dev arrays."""
        doses, r, g, b, sr, sg, sb = [], [], [], [], [], [], []
        for idx, img_path in enumerate(self.image_files):
            if img_path in self.image_rois and self.image_rois[img_path]:
                roi = self.image_rois[img_path][0]
                try:
                    dose_val = float(self.gray_values[idx])
                except (ValueError, IndexError):
                    continue
                if all(k in roi for k in ("mean_r", "mean_g", "mean_b")):
                    doses.append(dose_val)
                    r.append(roi["mean_r"])
                    g.append(roi["mean_g"])
                    b.append(roi["mean_b"])
                    sr.append(roi.get("std_r",0))
                    sg.append(roi.get("std_g",0))
                    sb.append(roi.get("std_b",0))
        return (np.array(doses), np.array(r), np.array(g), np.array(b),
                np.array(sr), np.array(sg), np.array(sb))

    def open_fit_window(self):
        if hasattr(self, "fit_window") and self.fit_window and self.fit_window.winfo_exists():
            self.fit_window.lift()
            return

        self.fit_window = tk.Toplevel(self.root)
        self.fit_window.title("Fit Calibration Data")
        self.fit_window.geometry("1000x750")

        # ----- PLOT -----
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        self.fit_fig, self.fit_ax = plt.subplots(figsize=(8,5))
        canvas = FigureCanvasTkAgg(self.fit_fig, master=self.fit_window)
        toolbar = NavigationToolbar2Tk(canvas, self.fit_window)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fit_canvas = canvas

        # ----- CONTROLS -----
        controls_frame = tk.Frame(self.fit_window)
        controls_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.fit_type_var = tk.StringVar(value="standard")
        for text,val in (("Standard", "standard"),("Cubic spline","spline")):
            tk.Radiobutton(controls_frame, text=text, variable=self.fit_type_var, value=val, command=self._on_fit_type_changed).pack(side=tk.LEFT, padx=5)

        # Checkbox to allow fitting standard model to spline instead of raw points
        self.fit_to_spline_var = tk.BooleanVar(value=False)
        tk.Checkbutton(controls_frame, text="Fit std→spline", variable=self.fit_to_spline_var, command=self._auto_fit).pack(side=tk.LEFT, padx=10)

        # ----- PARAMETER TABLE -----
        param_frame = tk.LabelFrame(self.fit_window, text="Fit Parameters (editable)")
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        headers = ("Channel","a","σa","b","σb","c","σc","R²")
        for col,h in enumerate(headers):
            tk.Label(param_frame, text=h, font=("Arial",10,"bold")).grid(row=0,column=col,padx=4)

        self.param_entries = {}
        for row,channel in enumerate(("R","G","B"), start=1):
            tk.Label(param_frame, text=channel).grid(row=row,column=0)
            self.param_entries[channel] = {}
            for col,p_name in enumerate(("a","σa","b","σb","c","σc"), start=1):
                e = tk.Entry(param_frame, width=10)
                e.grid(row=row,column=col,padx=2)
                self.param_entries[channel][p_name]=e
                if p_name.startswith('σ'):
                    e.configure(state='readonly')
                else:
                    e.bind('<FocusOut>', self._on_param_entry_changed)
                self.param_entries[channel][p_name]=e
            r2_lbl = tk.Label(param_frame, text="")
            r2_lbl.grid(row=row,column=7)
            self.param_entries[channel]["r2_label"] = r2_lbl

        # ----- BUTTONS -----
        btn_frame = tk.Frame(self.fit_window)
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Auto Fit", command=self._auto_fit).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Apply Fit", command=self._apply_fit).pack(side=tk.RIGHT, padx=10)
        tk.Button(controls_frame, text="Save Spline", command=self._save_spline_csv).pack(side=tk.RIGHT, padx=5)

        # prepare results dict before plotting
        self.latest_fit_results = {}

        self.fit_fig.canvas.mpl_connect('button_press_event', self._on_fit_click)

        self._update_fit_plot()
        self._manual_override.clear() # clear manual overrides when opening fit window

        self.fit_canvas.draw()

    def _on_fit_type_changed(self):
        """Clear params when switching type and refresh plot."""
        for ch in ("R","G","B"):
            for pn in ("a","σa","b","σb","c","σc"):
                self.param_entries[ch][pn].delete(0,tk.END)
            self.param_entries[ch]["r2_label"].config(text="")
        self._auto_fit()

    def _auto_fit(self):
       """Perform automatic fitting and update plot."""
       self._manual_override.clear() # Limpia los overrides manuales
       self._update_fit_plot()

    def _apply_fit(self):
        """Save current entries to CSV and close window."""
        fname = "fit_parameters.csv"
        try:
            with open(fname, 'w', newline='', encoding='utf-8') as f:  # Added encoding='utf-8'
                w = csv.writer(f)
                # Include bit_depth in header to indicate calibration bit depth
                w.writerow(["Channel", "a", "σa", "b", "σb", "c", "σc", "R2", "bit_depth"])
                for ch in ("R", "G", "B"):
                    a = self.param_entries[ch]['a'].get().strip()
                    σa = self.param_entries[ch]['σa'].get().strip()
                    b = self.param_entries[ch]['b'].get().strip()
                    σb = self.param_entries[ch]['σb'].get().strip()
                    c = self.param_entries[ch]['c'].get().strip()
                    σc = self.param_entries[ch]['σc'].get().strip()
                    r2text = self.param_entries[ch]['r2_label'].cget('text')
                    # Include calibration bit depth so apply_calibration knows what to expect
                    w.writerow([ch, a, σa, b, σb, c, σc, r2text, self.calibration_bit_depth])
            messagebox.showinfo("Fit saved", f"Fit parameters saved to {os.path.abspath(fname)}\\nCalibration bit depth: {self.calibration_bit_depth}-bit")
        except Exception as e:
            messagebox.showerror("Save error", str(e))
        self.fit_window.destroy()

    def _update_fit_plot(self):
        """Redraw scatter and chosen fit."""
        doses, r, g, b, sr, sg, sb = self._get_calibration_data()
        self.fit_ax.clear()
        if len(doses) == 0:
            self.fit_ax.set_title("No measured data to fit")
            self.fit_canvas.draw()
            return

        # Helper to split included/excluded indices per channel
        npts = len(doses)
        idxs = np.arange(npts)

        def split_points(channel, data, errs, color, label):
            included_mask = np.array([(channel, i) not in self.excluded_points for i in idxs])
            excluded_mask = ~included_mask

            # Plot included points (filled)
            self.fit_ax.errorbar(doses[included_mask], data[included_mask], yerr=errs[included_mask], fmt='o', color=color, label=label)
            # Plot excluded points (hollow)
            if excluded_mask.any():
                self.fit_ax.errorbar(doses[excluded_mask], data[excluded_mask], yerr=errs[excluded_mask], fmt='o', markerfacecolor='none', ecolor=color, markeredgecolor=color, color=color, label=f'{label} (excluded)')
            return included_mask

        mask_r = split_points('R', r, sr, 'red', 'Mean R')
        mask_g = split_points('G', g, sg, 'green', 'Mean G')
        mask_b = split_points('B', b, sb, 'blue', 'Mean B')

        x_line = np.linspace(min(doses), max(doses), 300)

        fit_type = self.fit_type_var.get()
        try:
            if fit_type == "standard":
                def model(x, a, b, c):
                    return a + b/(x - c)

                for ch_data, mask, color, label, ch_name in ((r, mask_r, "red", "Fit R", "R"), 
                                                              (g, mask_g, "green", "Fit G", "G"), 
                                                              (b, mask_b, "blue", "Fit B", "B")):
                    
                    current_popt = np.array([]) # Parámetros del ajuste actual
                    current_perr = np.array([]) # Errores de los parámetros del ajuste actual

                    # Prepara datos para el ajuste (excluyendo puntos y dosis cero)
                    fit_mask = mask & (doses > 0)
                    
                    # Si no hay suficientes puntos después de filtrar, salta este canal
                    if np.sum(fit_mask) < 3: # Necesitas al menos 3 puntos para 3 parámetros
                        print(f"Fit warning for {label}: Not enough data points ({np.sum(fit_mask)}) after filtering for fitting.")
                        # Limpia UI para este canal
                        for pn_ui_clear, epn_ui_clear in zip(("a","b","c"), ("σa","σb","σc")):
                            self.param_entries[ch_name][pn_ui_clear].delete(0,tk.END)
                            self.param_entries[ch_name][epn_ui_clear].configure(state='normal')
                            self.param_entries[ch_name][epn_ui_clear].delete(0,tk.END)
                            self.param_entries[ch_name][epn_ui_clear].configure(state='readonly')
                        self.param_entries[ch_name]['r2_label'].config(text="N/A")
                        if label in self.latest_fit_results: self.latest_fit_results.pop(label)
                        continue

                    # Estimaciones iniciales
                    a0 = np.mean(ch_data[fit_mask])
                    b0 = (np.max(ch_data[fit_mask]) - np.min(ch_data[fit_mask])) * (np.ptp(doses[fit_mask]) + 1e-6)
                    c0 = 0.0
                    
                    y_target_for_fit = ch_data[fit_mask] # Datos Y para el ajuste
                    
                    # Opción de ajustar a la spline
                    if self.fit_to_spline_var.get():
                        try:
                            # Asegúrate de que hay suficientes puntos para la spline también
                            if np.sum(fit_mask) >= 4: # CubicSpline necesita al menos 4 puntos
                                cs_tmp = CubicSpline(doses[fit_mask], ch_data[fit_mask])
                                y_target_for_fit = cs_tmp(doses[fit_mask])
                            else:
                                print(f"Fit warning for {label}: Not enough points ({np.sum(fit_mask)}) for spline, using raw data.")
                        except Exception as e_spline:
                            print(f"Fit warning for {label}: Spline creation failed ({e_spline}), using raw data.")
                            pass # fallback to raw data if spline fails

                    # Comprueba si hay valores manuales
                    if ch_name in self._manual_override:
                        current_popt = np.array(self._manual_override[ch_name])
                        # current_perr se queda vacío para ajustes manuales
                    else: # Intenta el ajuste automático
                        try:
                            fit_params, fit_cov_matrix = curve_fit(model, doses[fit_mask], y_target_for_fit, p0=(a0, b0, c0), maxfev=20000)
                            current_popt = fit_params
                            
                            # Calcula errores solo si la matriz de covarianza es válida
                            if fit_cov_matrix is not None and np.all(np.isfinite(fit_cov_matrix)) and fit_cov_matrix.shape == (3,3):
                                print(f"DEBUG {label}: fit_cov_matrix IS considered valid by initial check.")
                                diag_pcov = np.diag(fit_cov_matrix)
                                print(f"DEBUG {label}: diag_pcov = {diag_pcov}")
                                
                                # Intentamos calcular la raíz cuadrada. Puede producir NaN si hay negativos en diag_pcov.
                                diag_sqrt = np.empty_like(diag_pcov)
                                for i_err, val_err in enumerate(diag_pcov):
                                    if val_err >= 0:
                                        diag_sqrt[i_err] = np.sqrt(val_err)
                                    else:
                                        diag_sqrt[i_err] = np.nan
                                        print(f"DEBUG {label}: Negative value encountered in diag_pcov at index {i_err}: {val_err}")

                                print(f"DEBUG {label}: diag_sqrt (after np.sqrt attempt) = {diag_sqrt}")
                                
                                # ASIGNACIÓN CORRECTA A current_perr:
                                if np.all(np.isfinite(diag_sqrt)):
                                    current_perr = diag_sqrt
                                    print(f"DEBUG {label}: current_perr (all finite) = {current_perr}") 
                                else:
                                    current_perr = np.array([np.nan, np.nan, np.nan])
                                    print(f"DEBUG {label}: current_perr (some non-finite, set to nan) = {current_perr}")
                            else: 
                                # Este 'else' es para el 'if fit_cov_matrix is not None...'
                                current_perr = np.array([np.nan, np.nan, np.nan])
                                print(f"Fit warning for {label}: Covariance matrix unusable (failed initial check).")
                        except (RuntimeError, ValueError) as fit_err:
                            print(f"Fit warning for {label}: {fit_err}")
                            # current_popt y current_perr se quedan como np.array([])

                    # Comprueba si el ajuste fue exitoso o si hay valores manuales válidos
                    if current_popt.size != 3:
                        # El ajuste falló o los valores manuales no eran válidos
                        print(f"Fit info for {label}: No valid parameters obtained.")
                        # Limpia UI
                        for pn_ui_clear, epn_ui_clear in zip(("a","b","c"), ("σa","σb","σc")):
                            self.param_entries[ch_name][pn_ui_clear].delete(0,tk.END)
                            self.param_entries[ch_name][epn_ui_clear].configure(state='normal')
                            self.param_entries[ch_name][epn_ui_clear].delete(0,tk.END)
                            self.param_entries[ch_name][epn_ui_clear].configure(state='readonly')
                        self.param_entries[ch_name]['r2_label'].config(text="N/A")
                        if label in self.latest_fit_results: self.latest_fit_results.pop(label)
                        continue # Al siguiente canal

                    # Si llegamos aquí, current_popt es válido. Procedemos.
                    
                    # Para R^2 y la línea de ajuste, usa los puntos apropiados
                    # Si es manual, usa todos los puntos (doses). Si es auto, usa fit_mask.
                    points_for_eval = doses if ch_name in self._manual_override else doses[fit_mask]
                    actual_y_for_r2 = ch_data if ch_name in self._manual_override else ch_data[fit_mask] # Y reales correspondientes
                    
                    # Asegúrate de que haya datos para evaluar
                    if points_for_eval.size == 0:
                        print(f"Fit info for {label}: No points to evaluate model.")
                        r2 = np.nan
                    else:
                        fitted_y_values = model(points_for_eval, *current_popt)
                        
                        # Cálculo de R^2
                        mean_actual_y = np.mean(actual_y_for_r2)
                        ss_tot = np.sum((actual_y_for_r2 - mean_actual_y)**2)
                        ss_res = np.sum((actual_y_for_r2 - fitted_y_values)**2)
                        
                        if ss_tot > 1e-9: # Evita división por cero
                            r2 = 1 - (ss_res / ss_tot)
                        else:
                            r2 = np.nan if ss_res > 1e-9 else 1.0 # Perfecto ajuste si ambos son cero

                    self.latest_fit_results[label] = {"params": current_popt, "errors": current_perr, "r2": r2}

                    # Dibuja la línea de ajuste
                    line_style = "-." if ch_name in self._manual_override else "--"
                    plot_label_text = f"Manual {ch_name}" if ch_name in self._manual_override else label
                    self.fit_ax.plot(x_line, model(x_line, *current_popt), color=color, linestyle=line_style, label=plot_label_text)

                    # Actualiza los campos de la UI
                    for i, param_name_short_ui in enumerate(("a","b","c")):
                        param_entry_ui = self.param_entries[ch_name][param_name_short_ui]
                        error_entry_ui = self.param_entries[ch_name][f"σ{param_name_short_ui}"]

                        param_entry_ui.delete(0, tk.END)
                        param_entry_ui.insert(0, f"{current_popt[i]:.5g}")

                        error_entry_ui.configure(state='normal')
                        error_entry_ui.delete(0, tk.END)
                        if current_perr.size == 3 and not np.isnan(current_perr[i]):
                            error_entry_ui.insert(0, str(current_perr[i]))
                        error_entry_ui.configure(state='readonly')
                    
                    self.param_entries[ch_name]['r2_label'].config(text=f"{r2:.4f}" if not np.isnan(r2) else "N/A")

            elif fit_type == "spline":
                for ch_data, mask, color, label in ((r, mask_r, "red", "Spline R"), (g, mask_g, "green", "Spline G"), (b, mask_b, "blue", "Spline B")):
                    cs = CubicSpline(doses[mask], ch_data[mask])
                    self.fit_ax.plot(x_line, cs(x_line), color=color, linestyle="--", label=label)
                    self.latest_fit_results[label] = {"params": [], "errors": [], "r2": None, "x": x_line.tolist(), "y": cs(x_line).tolist()}
        except Exception as e:
            print(f"Fit error: {e}")

        self.fit_ax.set_xlabel("Dose")
        self.fit_ax.set_ylabel("Channel mean value")
        self.fit_ax.legend()
        self.fit_ax.grid(True)
        self.fit_canvas.draw()

        # fill entry boxes & r2 labels
        for ch in ("R","G","B"):
            res = self.latest_fit_results.get(f"Fit {ch}") or self.latest_fit_results.get(f"Spline {ch}") or {}
            params = res.get("params")
            if params is not None and hasattr(params, '__len__') and len(params) == 3:
                a, b, c = params
                for val, pname in zip((a, b, c), ("a", "b", "c")):
                    self.param_entries[ch][pname].delete(0,tk.END)
                    self.param_entries[ch][pname].insert(0, f"{val:.5g}")
                self.param_entries[ch]["r2_label"].config(text=f"{res['r2']:.4f}")
                # Rellenar errores si están disponibles y son finitos
                errors = res.get("errors")
                if errors is not None and hasattr(errors, '__len__') and len(errors) == 3:
                    for err_val, pname in zip(errors, ("a", "b", "c")):
                        err_entry = self.param_entries[ch][f"σ{pname}"]
                        err_entry.configure(state='normal')
                        err_entry.delete(0, tk.END)
                        if not (err_val is None or np.isnan(err_val)):
                            err_entry.insert(0, str(err_val))
                        err_entry.configure(state='readonly')
            else:
                for pname in ("a", "b", "c"):
                    self.param_entries[ch][pname].delete(0,tk.END)
                self.param_entries[ch]["r2_label"].config(text="")
                # limpiar errores
                for pname in ("σa","σb","σc"):
                    err_entry = self.param_entries[ch][pname]
                    err_entry.configure(state='normal')
                    err_entry.delete(0, tk.END)
                    err_entry.configure(state='readonly')

    def _on_fit_click(self, event):
        """Handle mouse clicks on the fit plot to toggle data point inclusion/exclusion."""
        if event.button != 1 or event.inaxes != self.fit_ax:
            return  # Only respond to left-click inside axes

        # Retrieve current data
        doses, r, g, b, sr, sg, sb = self._get_calibration_data()
        if len(doses) == 0:
            return

        # Build list of points with channels
        points = []  # tuples: (channel, idx, x, y)
        for idx in range(len(doses)):
            points.append(('R', idx, doses[idx], r[idx]))
            points.append(('G', idx, doses[idx], g[idx]))
            points.append(('B', idx, doses[idx], b[idx]))

        # Determine nearest point in data coordinates
        x_click, y_click = event.xdata, event.ydata
        if x_click is None or y_click is None:
            return

        distances = [ (abs(px - x_click)**2 + abs(py - y_click)**2, ch, idx) for ch, idx, px, py in points ]
        dist, ch_sel, idx_sel = min(distances, key=lambda t: t[0])

        # Define tolerance as 2% of x-range and y-range
        x_range = self.fit_ax.get_xlim()
        y_range = self.fit_ax.get_ylim()
        tol = 0.02 * ((x_range[1]-x_range[0])**2 + (y_range[1]-y_range[0])**2)
        if dist > tol:
            return  # Click too far from any point

        key = (ch_sel, idx_sel)
        if key in self.excluded_points:
            self.excluded_points.remove(key)
        else:
            self.excluded_points.add(key)

        # Refresh plot
        self._update_fit_plot()

    def _on_param_entry_changed(self, event=None):
        """Called on manual parameter change."""
        # read entries and if three valid numbers, store as manual override
        widget = event.widget if event else None
        if widget:
            # determine which channel
            for ch in ("R","G","B"):
                if widget in self.param_entries[ch].values():
                    try:
                        vals = [float(self.param_entries[ch][pn].get()) for pn in ("a","b","c")]
                        self._manual_override[ch] = tuple(vals)
                    except ValueError:
                        if ch in self._manual_override:
                            self._manual_override.pop(ch)
                    break
        self._update_fit_plot()

    def _save_spline_csv(self):
        """Save current spline fit curves to CSV file."""
        if self.fit_type_var.get() != 'spline':
            messagebox.showwarning("Save Spline", "Switch to 'Cubic spline' fit type first.")
            return

        # Ensure plot up to date
        self._update_fit_plot()

        # Expect latest spline results stored
        x_vals = None
        rows = []
        for ch in ('R','G','B'):
            key = f"Spline {ch}"
            res = self.latest_fit_results.get(key)
            if not res or 'x' not in res:
                messagebox.showerror("Save Spline", "Spline data not available. Run Auto Fit first.")
                return
            if x_vals is None:
                x_vals = res['x']
            rows.append(res['y'])

        # transpose rows to columns
        fname = 'spline_points.csv'
        try:
            with open(fname,'w',newline='') as f:
                w=csv.writer(f)
                w.writerow(['Dose','SplineR','SplineG','SplineB'])
                for i, dose in enumerate(x_vals):
                    w.writerow([dose, rows[0][i], rows[1][i], rows[2][i]])
            messagebox.showinfo("Save Spline", f"Spline points saved to {os.path.abspath(fname)}")
        except Exception as e:
            messagebox.showerror("Save Spline", str(e))

    # -------- helper to crop white border -------- #
    def _crop_white_border(self, pil_img, thresh: int = 240):
        """Detect white border (scanner background) and crop it out and auto-zoom."""
        self.last_crop_bbox = None  # Reset crop bbox
        try:
            # Convert to grayscale for detection (this is just for mask, not affecting original)
            # For 16-bit images, need to scale threshold
            temp_arr = np.array(pil_img)
            if temp_arr.dtype == np.uint16:
                # Scale threshold from 8-bit (0-255) to 16-bit (0-65535)
                thresh_scaled = int(thresh * 256)
                gray_arr = np.mean(temp_arr, axis=-1) if temp_arr.ndim == 3 else temp_arr
                bw = gray_arr < thresh_scaled
            else:
                gray = pil_img.convert('L')
                bw_img = gray.point(lambda p: 255 if p < thresh else 0)
                bw = np.array(bw_img) > 0
            
            # Find bounding box of non-white pixels
            rows = np.any(bw, axis=1)
            cols = np.any(bw, axis=0)
            if not np.any(rows) or not np.any(cols):
                return pil_img  # Could not detect film region
            
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            pad = 2  # small padding
            left = max(0, cmin - pad)
            top = max(0, rmin - pad)
            right = min(pil_img.width, cmax + pad + 1)
            bottom = min(pil_img.height, rmax + pad + 1)
            
            self.last_crop_bbox = (left, top, right, bottom)
            return pil_img.crop((left, top, right, bottom))
        except Exception as e:
            print(f"Crop error: {e}")
            return pil_img

if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationApp(root)
    root.mainloop()
