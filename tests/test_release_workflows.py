"""Regression cases spanning numerical state, UI entry points and durable outputs."""
import csv
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from matplotlib.figure import Figure

from app.core.image_processor import ImageProcessor
from app.calibration.calibration_app import CalibrationApp
from app.core.spline_calibration import save_spline_calibration
from custom_plugins.auto_measurements.core.exporter import CSVExporter
from custom_plugins.auto_measurements.core.ctr_manager import CTRManager
from custom_plugins.auto_measurements.core.formatter import MeasurementFormatter
from custom_plugins.auto_measurements.ui.main_tab import AutoMeasurementsTab


def processor(image):
    obj = ImageProcessor.__new__(ImageProcessor)
    obj.original_image = image.copy()
    obj.current_image = image.copy()
    obj.processing_lock = threading.RLock()
    obj.binning = 1
    obj.zoom = 1.0
    obj.calibration_applied = False
    obj.flat_applied = False
    obj.flattened_image = None
    obj.calibration_source_image = None
    obj.calibration_provenance = {}
    obj.measurement_shape = "circular"
    obj.measurement_size = 1
    obj.measurement_size_rect = (4, 4)
    obj.config = {}
    obj._report_progress = lambda *args: None
    obj._compute_integral_images = lambda: None
    return obj


def measurement(value=1.0, **provenance):
    return dict(film="F", circle="C", dose=(value,) * 3, std_per_channel=(.1,) * 3,
                avg=value, avg_unc=.1, pixel_count=5, provenance=provenance)


class ProcessingWorkflowTests(unittest.TestCase):
    def test_auto_measurement_3d_view_uses_tk_plot_window(self):
        tab = AutoMeasurementsTab.__new__(AutoMeasurementsTab)
        tab.frame = object()
        tab.image_processor = SimpleNamespace(
            calibration_applied=False,
            original_image=np.arange(9 * 9 * 3, dtype=float).reshape(9, 9, 3),
            current_image=None,
        )
        with patch("app.ui.plot_window.show_figure") as show:
            tab._show_circle_3d(4, 4, 2)
        show.assert_called_once()

    def test_auto_conversion_uses_spline_inside_and_fit_outside_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit_path = root / "fit_parameters.csv"
            with fit_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Channel", "a", "b", "c", "bit_depth", "dose_min", "dose_max"])
                for channel in "RGB":
                    writer.writerow([channel, 0, 100, -1, 16, 0, 2])
            knots = {
                channel: (np.array([0., 1., 2.]), np.array([100., 50., 100. / 3.]))
                for channel in "RGB"
            }
            save_spline_calibration(root / "spline_calibration.npz", knots, 16)
            image = np.array([[[50, 50, 50], [25, 25, 25]]], dtype=np.uint16)
            p = processor(image)
            p.image_max_value = 65535
            p.config = {"calibration_conversion_method": "auto"}
            p._find_fit_parameters_file = lambda: str(fit_path)
            self.assertTrue(p.apply_calibration())
            np.testing.assert_allclose(p.dose_channels[0, 0], [1, 1, 1])
            np.testing.assert_allclose(p.dose_channels[0, 1], [3, 3, 3])
            self.assertTrue(p.dose_extrapolated_mask_channels[0, 1].all())

    def test_spline_conversion_does_not_extrapolate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit_path = root / "fit_parameters.csv"
            with fit_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Channel", "a", "b", "c", "bit_depth", "dose_min", "dose_max"])
                for channel in "RGB":
                    writer.writerow([channel, 0, 100, -1, 16, 0, 2])
            knots = {
                channel: (np.array([0., 1., 2.]), np.array([100., 50., 100. / 3.]))
                for channel in "RGB"
            }
            save_spline_calibration(root / "spline_calibration.npz", knots, 16)
            p = processor(np.array([[[50, 50, 50], [25, 25, 25]]], dtype=np.uint16))
            p.image_max_value = 65535
            p.config = {"calibration_conversion_method": "spline"}
            p._find_fit_parameters_file = lambda: str(fit_path)
            self.assertTrue(p.apply_calibration())
            np.testing.assert_allclose(p.dose_channels[0, 0], [1, 1, 1])
            self.assertTrue(np.isnan(p.dose_channels[0, 1]).all())
            self.assertFalse(p.dose_extrapolated_mask_channels.any())

    def test_fit_conversion_ignores_an_unrelated_invalid_spline_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit_path = root / "fit_parameters.csv"
            with fit_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Channel", "a", "b", "c", "bit_depth", "dose_min", "dose_max"])
                for channel in "RGB":
                    writer.writerow([channel, 0, 100, -1, 16, 0, 2])
            (root / "spline_calibration.npz").write_bytes(b"not an npz file")
            p = processor(np.array([[[50, 50, 50]]], dtype=np.uint16))
            p.image_max_value = 65535
            p.config = {"calibration_conversion_method": "fit"}
            p._find_fit_parameters_file = lambda: str(fit_path)
            self.assertTrue(p.apply_calibration())
            np.testing.assert_allclose(p.dose_channels[0, 0], [1, 1, 1])
            self.assertIsNone(getattr(p, "last_processing_warning", None))

    def test_rectangular_roi_has_exact_requested_even_dimensions(self):
        p = processor(np.ones((9, 9)))
        p.measurement_shape = "rectangular"
        p.measurement_size_rect = (4, 2)
        self.assertEqual(p.measure_area(4, 4)[-1], 8)

    def test_preview_binning_preserves_uint16_analysis_and_calibration(self):
        p = processor(np.full((4, 4, 3), 1000, dtype=np.uint16))
        p.current_image = np.full((4, 4, 3), 2.5, dtype=np.float64)
        p.calibration_applied = True
        p.set_binning(2)
        self.assertEqual(p.binned_image["data"][0, 0, 0], 1000)
        np.testing.assert_array_equal(p.current_image, np.full((4, 4, 3), 2.5))
        self.assertTrue(p.calibration_applied)
        p.set_binning(1)
        self.assertEqual(p.current_image[0, 0, 0], 2.5)

    def test_binned_pixel_information_maps_full_resolution_coordinates(self):
        p = processor(np.arange(8 * 8 * 3, dtype=np.uint16).reshape(8, 8, 3))
        p.set_binning(2)
        x, y, value = p.get_pixel_info(7, 7)
        self.assertEqual((x, y), (7, 7))
        expected_std = tuple(p.binned_image["std_dev"][3, 3])
        np.testing.assert_allclose(value[1], expected_std)

    def test_circle_ignores_manual_tool_and_restores_it(self):
        p = processor(np.arange(49).reshape(7, 7))
        p.measurement_shape = "rectangular"
        result = p.measure_circle(3, 3, 1)
        self.assertEqual(result[-1], 5)
        self.assertEqual(p.measurement_shape, "rectangular")

    def test_roi_covariance_uses_the_same_valid_pixels_as_mean(self):
        p = processor(np.ones((2, 2)))
        p.calibration_applied = True
        p.calibration_fit_params = {"G": (0., 1., 0.)}
        p.calibration_param_covariances = {"G": np.diag([1., 0., 0.])}
        mean, _, uncertainty = p._summarize_roi_pixels(
            np.array([[1.], [np.nan]]), np.array([[1.], [.1]]))
        self.assertEqual(mean[0], 1.)
        self.assertEqual(uncertainty[0], 1.)

    def test_failed_flat_aborts_dose_pipeline_and_restores_raw_image(self):
        p = processor(np.full((2, 2, 3), 1000, dtype=np.uint16))
        p.apply_flat = Mock(return_value=False)
        p.apply_calibration = Mock(return_value=True)
        self.assertFalse(p.process_corrections(flat=True, calibration=True))
        p.apply_calibration.assert_not_called()
        self.assertFalse(p.calibration_applied)
        self.assertFalse(p.flat_applied)
        np.testing.assert_array_equal(p.current_image, p.original_image)

    def test_invalid_flat_cannot_produce_saturated_scientific_data(self):
        p = processor(np.ones((2, 2, 3), dtype=np.uint16))
        p.flat_field = np.zeros((2, 2, 3))
        with self.assertRaises(ValueError):
            p._apply_flattening_3ch(p.current_image)

    def test_completely_invalid_roi_is_not_a_measurement(self):
        p = processor(np.full((3, 3), np.nan))
        self.assertIsNone(p.measure_circle(1, 1, 1))


class ExportWorkflowTests(unittest.TestCase):
    def test_locked_destination_preserves_csv_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "results.csv"
            target.write_text("previous", encoding="utf-8")
            exporter = CSVExporter(None, None, None)
            with patch("app.utils.atomic_file.os.replace", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    exporter.write_results(target, [("image.tif", [measurement()])])
            self.assertEqual(target.read_text(encoding="utf-8"), "previous")
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_invalid_late_row_preserves_existing_csv(self):
        exporter = CSVExporter(None, None, None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            path.write_text("existing results", encoding="utf-8")
            with self.assertRaises(ValueError):
                exporter.write_results(path, [("scan.tif", [measurement(), measurement("bad")])])
            self.assertEqual(path.read_text(), "existing results")
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_batch_keeps_calibration_date_and_units_per_measurement(self):
        exporter = CSVExporter(None, SimpleNamespace(config={"calibration_folder": "C"}), None)
        rows = [measurement(calibration_id="A", date="2026-01-01", units="Gy", conversion_method="auto"),
                measurement(calibration_id="B", date="2026-02-01", units="scanner_intensity")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            self.assertEqual(exporter.write_results(path, [("first.tif", rows[:1]), ("second.tif", rows[1:])]), 2)
            with path.open(encoding="utf-8", newline="") as handle:
                exported = list(csv.DictReader(handle))
        self.assertEqual([row["calibration_id"] for row in exported], ["A", "B"])
        self.assertEqual([row["Date"] for row in exported], ["2026-01-01", "2026-02-01"])
        self.assertEqual(exported[1]["units"], "scanner_intensity")
        self.assertEqual([row["conversion_method"] for row in exported], ["auto", "not_applied"])

    def test_unknown_provenance_is_never_labelled_current_calibration(self):
        exporter = CSVExporter(None, None, None)
        row = dict(zip(exporter.COLUMNS, exporter.result_row(measurement())))
        self.assertEqual(row["calibration_integrity"], "unknown")
        self.assertEqual(row["calibration_id"], "")


class CalibrationPersistenceTests(unittest.TestCase):
    def test_closing_modified_calibration_prompts_before_discarding(self):
        app = CalibrationApp.__new__(CalibrationApp)
        app._fit_dirty = True
        app.fit_window = SimpleNamespace(destroy=Mock())
        app._apply_fit = Mock()
        with patch(
            "app.calibration.calibration_app.messagebox.askyesnocancel",
            return_value=False,
        ) as prompt:
            app._close_fit_window()
        prompt.assert_called_once()
        app._apply_fit.assert_not_called()
        app.fit_window.destroy.assert_called_once()

    def test_closing_modified_calibration_can_save(self):
        app = CalibrationApp.__new__(CalibrationApp)
        app._fit_dirty = True
        app.fit_window = SimpleNamespace(destroy=Mock())
        app._apply_fit = Mock()
        with patch(
            "app.calibration.calibration_app.messagebox.askyesnocancel",
            return_value=True,
        ):
            app._close_fit_window()
        app._apply_fit.assert_called_once()
        app.fit_window.destroy.assert_not_called()

    def test_fit_point_selection_requires_a_near_screen_click(self):
        app = CalibrationApp.__new__(CalibrationApp)
        figure = Figure()
        app.fit_ax = figure.add_subplot()
        app.fit_ax.set_xlim(0, 10)
        app.fit_ax.set_ylim(0, 100)
        figure.canvas.draw()
        app._get_calibration_data = lambda: (
            np.array([1.0]), np.array([20.0]), np.array([40.0]), np.array([60.0]),
            np.array([0.0]), np.array([0.0]), np.array([0.0]),
        )
        point_x, point_y = app.fit_ax.transData.transform((1.0, 20.0))
        far_event = SimpleNamespace(inaxes=app.fit_ax, x=point_x + 100, y=point_y + 100)
        near_event = SimpleNamespace(inaxes=app.fit_ax, x=point_x + 2, y=point_y + 2)
        self.assertIsNone(app._nearest_fit_point(far_event))
        self.assertEqual(app._nearest_fit_point(near_event)[:3], ("R", 0, 1.0))

    def test_saved_exclusions_are_restored_for_modify_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "calibration_manifest.json").write_text(json.dumps({
                "dose_calibration": {"excluded_points": [
                    {"channel": "R", "index": 1}, {"channel": "B", "index": 3},
                    {"channel": "invalid", "index": 0},
                ]}
            }), encoding="utf-8")
            app = SimpleNamespace(data_dir=root, image_files=["a", "b", "c"], excluded_points=set())
            CalibrationApp._restore_saved_exclusions(app)
        self.assertEqual(app.excluded_points, {("R", 1)})

    def test_calibration_dialog_rejects_nonpositive_flat(self):
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(Path(tmp) / "field_flattening.npz", flat_field=np.zeros((2, 2, 3)))
            app = SimpleNamespace(data_dir=Path(tmp), flat_field=None)
            with self.assertLogs("app.calibration.calibration_app", level="ERROR"):
                CalibrationApp._load_field_flattening(app)
            self.assertIsNone(app.flat_field)
            with self.assertRaises(ValueError):
                CalibrationApp._apply_field_flattening(app, np.ones((2, 2, 3)))

    def test_saved_parameters_keep_precision_and_channel_fit_domains(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = {
                f"Fit {ch}": dict(params=np.array([1.234567890123, 100.123456789, -1.]),
                    errors=np.array([.1, .2, .3]), covariance=np.diag([.01, .04, .09]),
                    r2=.987654321, n_points=4, dose_range=[0., float(i + 3)],
                    fit_method="weighted_nonlinear_least_squares")
                for i, ch in enumerate("RGB")
            }
            app = SimpleNamespace(data_dir=Path(tmp), latest_fit_results=results,
                fit_type_var=SimpleNamespace(get=lambda: "standard"), calibration_bit_depth=16,
                excluded_points=[], image_files=[], fit_window=SimpleNamespace(destroy=lambda:None),
                _spline_knots_from_data=lambda: {
                    channel: (np.array([0., 1., 2.]), np.array([100., 50., 100. / 3.]))
                    for channel in "RGB"
                })
            with patch("app.calibration.calibration_app.messagebox.showinfo"), patch(
                "app.calibration.calibration_app.messagebox.showerror") as error:
                CalibrationApp._apply_fit(app)
                error.assert_not_called()
            with (Path(tmp) / "fit_parameters.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(float(rows[0]["a"]), 1.234567890123)
            self.assertEqual([float(row["dose_max"]) for row in rows], [3., 4., 5.])
            self.assertTrue((Path(tmp) / "spline_calibration.npz").is_file())
            manifest = json.loads(
                (Path(tmp) / "calibration_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn("spline_calibration.npz", manifest["artifacts"])


class ControlWorkflowTests(unittest.TestCase):
    def test_global_control_uses_remeasured_numeric_value(self):
        tab = SimpleNamespace(
            global_ctr={"item_id": "control", "film_name": "F", "circle_data": {"avg_original": 99}},
            tree=SimpleNamespace(exists=lambda item: True, item=lambda *args: "C (GLOBAL CTR)",
                                 get_children=lambda: ()),
            results=[{"film": "F", "circle": "C", "avg_numeric": .25, "avg_unc_numeric": .02}],
        )
        AutoMeasurementsTab._apply_global_ctr_subtraction(tab)
        self.assertEqual(tab.global_ctr["circle_data"]["avg_original"], .25)
        self.assertEqual(tab.global_ctr["circle_data"]["avg_unc_numeric"], .02)

    def test_invalid_control_is_excluded_from_covariance_membership(self):
        tree = SimpleNamespace(exists=lambda item:True, item=lambda item, field:item)
        manager = CTRManager(tree, MeasurementFormatter)
        manager.original_measurements = {"good": {"avg":".2", "avg_unc":".1"},
                                         "bad": {"avg":"nan", "avg_unc":".1"}}
        mean, uncertainty, ids = manager._compute_averaged_ctr("F", ["good", "bad"], {}, return_ids=True)
        self.assertEqual(ids, ["good"])
        self.assertEqual(mean, .2)
        self.assertEqual(uncertainty, .1)


if __name__ == "__main__":
    unittest.main()
