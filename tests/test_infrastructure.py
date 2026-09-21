import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from app.models.config_model import DEFAULT_CONFIG
from app.plugins.plugin_manager import PluginManager
from app.utils.config_manager import ConfigManager
from app.version import __version__
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

    def test_frozen_application_imports_bundled_plugins(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "sys.frozen", True, create=True
        ):
            manager = PluginManager(directory)
        self.assertIn("analysis_tools", manager.plugin_names())
        self.assertIn("auto_measurements", manager.plugin_names())

    def test_user_plugin_package_loads_independently_of_bundled_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "my_plugin"
            package.mkdir()
            (package / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")
            (package / "__init__.py").write_text(
                "from .helper import VALUE\n"
                "def process(image):\n"
                "    return image + VALUE\n",
                encoding="utf-8",
            )
            manager = PluginManager(directory)
        self.assertIn("my_plugin", manager.plugin_names())


class ReleaseMetadataTests(unittest.TestCase):
    def test_wheel_and_application_versions_match(self):
        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["project"]["version"], __version__)

    def test_installer_is_per_user_and_preserves_user_data_directories(self):
        installer = Path("installer/RadiochromicFilmAnalyzer.iss").read_text(encoding="utf-8")
        self.assertIn("DefaultDirName={localappdata}\\Programs", installer)
        self.assertIn("PrivilegesRequired=lowest", installer)
        self.assertIn("RadiochromicFilmAnalyzer-Setup", installer)
        for directory in ("logs", "temp", "custom_plugins", "calibration_data"):
            self.assertIn(f'Name: "{{app}}\\{directory}"; Flags: uninsneveruninstall', installer)

    def test_packaging_uses_the_application_icon(self):
        spec = Path("radiochromic_scanning_tool.spec").read_text(encoding="utf-8")
        self.assertTrue(Path("resources/radiochromic_film_analyzer.ico").is_file())
        self.assertIn('icon="resources/radiochromic_film_analyzer.ico"', spec)


if __name__ == "__main__":
    unittest.main()
