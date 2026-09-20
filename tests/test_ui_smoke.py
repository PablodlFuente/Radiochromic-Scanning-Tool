"""Real Tk event-loop integration; no scanner or user configuration required."""
import copy
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.models.config_model import DEFAULT_CONFIG
from app.ui.main_window import MainWindow
from app.utils.image_io import write_tiff_unchanged


class TkWorkflowTests(unittest.TestCase):
    def test_loading_another_image_clears_results_and_keeps_native_precision(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        root.withdraw()
        root.on_close = root.destroy
        failures = []
        completed = []
        root.report_callback_exception = lambda *exc: failures.append(str(exc[1]))
        config = copy.deepcopy(DEFAULT_CONFIG)
        config.update(calibration_folder="__test_no_calibration__", check_updates_on_startup=False)
        window = None
        with tempfile.TemporaryDirectory() as directory, patch(
            "app.utils.config_manager.ConfigManager.save_config", return_value=True
        ), patch("app.utils.file_manager.FileManager.add_recent_file"), patch(
            "app.ui.main_window.messagebox.showerror", side_effect=lambda *args: failures.append(str(args))
        ):
            try:
                window = MainWindow(root, config)
                root.main_window = window
                from custom_plugins import auto_measurements
                tab = auto_measurements._AUTO_MEASUREMENTS_INSTANCE
                self.assertIsNotNone(tab)
                first = str(Path(directory) / "primera.tif")
                second = str(Path(directory) / "segunda.tif")
                write_tiff_unchanged(first, np.full((16, 16, 3), 1000, dtype=np.uint16))
                write_tiff_unchanged(second, np.full((16, 16, 3), 2000, dtype=np.uint16))

                def finish(_):
                    completed.append(True)
                    root.after(100, root.quit)

                def next_image(_):
                    tab.results = [{"film": "previous"}]
                    tab.tree.insert("", "end", text="previous")
                    root.after(10, lambda: window.load_image(second, on_complete=finish))

                root.after(10, lambda: window.load_image(first, on_complete=next_image))
                timeout = root.after(10000, root.quit)
                root.mainloop()
                root.after_cancel(timeout)
                self.assertEqual(failures, [])
                self.assertEqual(completed, [True], "Image workflow timed out")
                self.assertEqual(tab.results, [])
                self.assertEqual(tab.tree.get_children(), ())
                self.assertEqual(window.image_processor.current_image.dtype, np.uint16)
                self.assertTrue(np.all(window.image_processor.current_image == 2000))
            finally:
                if window is not None:
                    window.cleanup()
                    window.image_processor.cleanup()
                root.destroy()
