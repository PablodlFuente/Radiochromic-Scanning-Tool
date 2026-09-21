"""Real Tk event-loop integration; no scanner or user configuration required."""
import copy
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import matplotlib.pyplot as plt

from app.models.config_model import DEFAULT_CONFIG
from app.ui.main_window import MainWindow
from app.utils.image_io import write_tiff_unchanged
from app.ui.plot_window import show_figure


class TkWorkflowTests(unittest.TestCase):
    def test_plot_window_is_owned_by_tk_and_can_close_cleanly(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        root.withdraw()
        figure = plt.figure()
        window = show_figure(root, figure, "Test plot")
        root.update_idletasks()
        self.assertTrue(window.winfo_exists())
        window.close_figure()
        root.update_idletasks()
        root.destroy()

    def test_plot_window_reuses_the_same_logical_view(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        root.withdraw()
        first = show_figure(root, plt.figure(), "Reusable plot")
        second = show_figure(root, plt.figure(), "Reusable plot")
        root.update_idletasks()
        self.assertIs(first, second)
        self.assertEqual(len(root._plot_windows), 1)
        first.close_figure()
        root.update_idletasks()
        root.destroy()

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
        config.update(calibration_folder="__test_no_calibration__", automatic_updates=False)
        window = None
        with tempfile.TemporaryDirectory() as directory, patch(
            "app.utils.config_manager.ConfigManager.save_config", return_value=True
        ), patch("app.utils.file_manager.FileManager.add_recent_file"), patch(
            "app.ui.main_window.messagebox.showerror",
            side_effect=lambda *args, **kwargs: failures.append(str(args)),
        ):
            try:
                window = MainWindow(root, config)
                root.main_window = window
                self.assertEqual(
                    window.about_footer.cget("text"),
                    "Version 2.1.2 · Pablo de la Fuente Fernández",
                )
                with patch("app.ui.main_window.webbrowser.open_new_tab") as open_browser:
                    window.open_local_documentation()
                self.assertTrue(open_browser.called)
                self.assertIn("docs/Home.md", open_browser.call_args.args[0])
                from custom_plugins import auto_measurements
                tab = auto_measurements._AUTO_MEASUREMENTS_INSTANCE
                self.assertIsNotNone(tab)
                button_labels = {
                    child.cget("text")
                    for container in tab.frame.winfo_children()
                    for child in container.winfo_children()
                    if child.winfo_class() == "TButton"
                }
                self.assertIn("Add Measurement Area", button_labels)
                self.assertIn("Copy as XLSX", button_labels)
                self.assertNotIn("Add Circle", button_labels)
                tab.global_ctr = {"item_id": "missing"}
                tab._update_ctr_column_headings(True)
                self.assertEqual(tab.tree.heading("dose", "text"), "Channel dose (raw)")
                self.assertEqual(tab.tree.heading("avg", "text"), "Average - CTR")
                tab.global_ctr = None
                tab._update_ctr_column_headings(False)
                tab.add_area_button.invoke()
                root.update()
                popup = tab._shape_picker_window
                self.assertIsNotNone(popup)
                root.after(100, root.quit)
                root.mainloop()
                self.assertTrue(popup.winfo_exists())
                picker = popup.winfo_children()[0]
                menu_buttons = picker.winfo_children()
                self.assertEqual(
                    [button.cget("text") for button in menu_buttons],
                    ["Draw rectangle", "Draw circle", "Draw custom"],
                )
                self.assertTrue(all(
                    button.pack_info().get("side", "top") == "top"
                    for button in menu_buttons
                ))
                popup.destroy()
                root.update()
                self.assertIsNone(tab._shape_picker_window)
                tab.add_rc_button.invoke()
                root.update()
                self.assertIsNotNone(tab._shape_picker_window)
                self.assertEqual(tab.draw_target, "film")
                tab._shape_picker_window.destroy()
                root.update()
                tab._show_detection_settings()
                root.update()
                self.assertIsNotNone(tab._detection_settings_window)
                self.assertEqual(
                    tab.detection_mode_var.get(), "Circles and squares"
                )
                tab._show_dose_correction_tooltip()
                root.update_idletasks()
                self.assertIsNotNone(tab._dose_correction_tooltip)
                tab._hide_dose_correction_tooltip()
                tab._detection_settings_window.destroy()
                tab._detection_settings_window = None
                first = str(Path(directory) / "primera.tif")
                second = str(Path(directory) / "segunda.tif")
                write_tiff_unchanged(first, np.full((16, 16, 3), 1000, dtype=np.uint16))
                write_tiff_unchanged(second, np.full((16, 16, 3), 2000, dtype=np.uint16))

                def finish(_):
                    completed.append(True)
                    root.after(100, root.quit)

                def next_image(_):
                    tab._clear_overlay()
                    tab._insert_film_shape("rectangle", (1, 1, 14, 14))
                    film_id = tab.tree.get_children()[0]
                    tab._insert_measurement_shape("rectangle", (3, 3, 5, 5))
                    self.assertEqual(len(tab.tree.get_children(film_id)), 1)
                    self.assertEqual(tab.results[0]["shape"], "rectangle")
                    tab.file_manager.file_list = [first]
                    tab.file_manager.current_file_index = 0
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
                self.assertEqual(tab.file_manager.current_file_index, -1)
                self.assertEqual(tab.tree.get_children(), ())
                self.assertEqual(window.image_processor.current_image.dtype, np.uint16)
                self.assertTrue(np.all(window.image_processor.current_image == 2000))
            finally:
                if window is not None:
                    window.cleanup()
                    window.image_processor.cleanup()
                root.destroy()
