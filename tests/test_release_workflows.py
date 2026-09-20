"""Regression cases spanning numerical state, UI entry points and durable outputs."""
import csv
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.core.image_processor import ImageProcessor
from app.calibration.calibration_app import CalibrationApp
from custom_plugins.auto_measurements.core.exporter import CSVExporter
from custom_plugins.auto_measurements.core.ctr_manager import CTRManager
from custom_plugins.auto_measurements.core.formatter import MeasurementFormatter


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
        rows = [measurement(calibration_id="A", date="2026-01-01", units="Gy"),
                measurement(calibration_id="B", date="2026-02-01", units="scanner_intensity")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            self.assertEqual(exporter.write_results(path, [("first.tif", rows[:1]), ("second.tif", rows[1:])]), 2)
            with path.open(encoding="utf-8", newline="") as handle:
                exported = list(csv.DictReader(handle))
        self.assertEqual([row["calibration_id"] for row in exported], ["A", "B"])
        self.assertEqual([row["Date"] for row in exported], ["2026-01-01", "2026-02-01"])
        self.assertEqual(exported[1]["units"], "scanner_intensity")

    def test_unknown_provenance_is_never_labelled_current_calibration(self):
        exporter = CSVExporter(None, None, None)
        row = dict(zip(exporter.COLUMNS, exporter.result_row(measurement())))
        self.assertEqual(row["calibration_integrity"], "unknown")
        self.assertEqual(row["calibration_id"], "")


class CalibrationPersistenceTests(unittest.TestCase):
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
                fit_to_spline_var=SimpleNamespace(get=lambda:False), excluded_points=[],
                image_files=[], fit_window=SimpleNamespace(destroy=lambda:None))
            with patch("app.calibration.calibration_app.messagebox.showinfo"), patch(
                "app.calibration.calibration_app.messagebox.showerror") as error:
                CalibrationApp._apply_fit(app)
                error.assert_not_called()
            with (Path(tmp) / "fit_parameters.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(float(rows[0]["a"]), 1.234567890123)
            self.assertEqual([float(row["dose_max"]) for row in rows], [3., 4., 5.])


class ControlWorkflowTests(unittest.TestCase):
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
