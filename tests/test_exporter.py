import unittest

from custom_plugins.auto_measurements.core.exporter import CSVExporter


class CSVExporterTests(unittest.TestCase):
    def setUp(self):
        self.exporter = CSVExporter(None, None, None)

    def test_numeric_strings_are_parsed_without_losing_precision(self):
        values = self.exporter.get_export_values_for_result({
            "dose_numeric": "1.23456789, 2.5, 3.0",
            "std_numeric": "±0.1, ±0.2, ±0.3",
            "avg_numeric": "1.23456789",
            "avg_unc_numeric": "±0.012345",
        })
        self.assertEqual(values["dose"][0], 1.23456789)
        self.assertEqual(values["avg"], 1.23456789)
        self.assertEqual(values["avg_unc"], 0.012345)

    def test_invalid_numeric_data_is_not_silently_exported_as_zero(self):
        with self.assertRaisesRegex(ValueError, "average"):
            self.exporter.get_export_values_for_result({
                "dose": 1.0,
                "std_per_channel": 0.1,
                "avg": "not measured",
                "avg_unc": 0.1,
            })

    def test_format_for_csv_rejects_non_numeric_channel(self):
        with self.assertRaises(ValueError):
            self.exporter.format_for_csv([1.0, "invalid", 3.0])


if __name__ == "__main__":
    unittest.main()
