import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from app.models.config_model import DEFAULT_CONFIG
from app.plugins.plugin_manager import PluginManager
from app.utils.config_manager import ConfigManager
from custom_plugins.auto_measurements.core.metadata import MetadataExtractor


class ConfigurationTests(unittest.TestCase):
    def test_loading_partial_config_merges_all_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"negative_mode": True}), encoding="utf-8")
            config = ConfigManager(path).load_config()
        self.assertTrue(config["negative_mode"])
        self.assertEqual(config["calibration_folder"], DEFAULT_CONFIG["calibration_folder"])
        self.assertEqual(config["calibration_conversion_method"], "auto")

    def test_save_replaces_config_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            manager = ConfigManager(path)
            self.assertTrue(manager.save_config({"negative_mode": True}))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"negative_mode": True})
            self.assertEqual(list(Path(directory).glob("rc_config_*.json")), [])


class MetadataSafetyTests(unittest.TestCase):
    def test_windows_path_is_passed_via_environment_not_interpolated(self):
        extractor = MetadataExtractor(SimpleNamespace())
        hostile_path = 'C:\\scan\\film"; Write-Output injected; #.tif'
        completed = SimpleNamespace(returncode=0, stdout="{}")
        with patch("platform.system", return_value="Windows"), patch(
            "subprocess.run", return_value=completed
        ) as run:
            extractor._extract_windows_metadata(hostile_path)
        command = run.call_args.args[0]
        environment = run.call_args.kwargs["env"]
        self.assertNotIn(hostile_path, command[-1])
        self.assertEqual(environment["RADIOCHROMIC_METADATA_PATH"], str(Path(hostile_path).absolute()))
        self.assertIn("-LiteralPath", command[-1])


class PluginLifecycleTests(unittest.TestCase):
    def test_disabling_plugin_calls_teardown(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = PluginManager(directory)
            called = []
            module = ModuleType("test_plugin")
            module.teardown = lambda: called.append(True)
            manager._plugins["test"] = module
            manager._active["test"] = True
            manager._tabs["test"] = object()
            manager._notebook = SimpleNamespace(index=lambda frame: 0, forget=lambda index: None)
            manager.set_active("test", False)
        self.assertEqual(called, [True])


if __name__ == "__main__":
    unittest.main()
