import json
import tempfile
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from app.models.config_model import DEFAULT_CONFIG
from app.paths import ApplicationPermissionError, ensure_writable_directories
from app.plugins.plugin_manager import PluginManager
from app.ui.main_window import MainWindow
from app.utils.config_manager import ConfigManager
from app.version import __version__
from custom_plugins.auto_measurements.core.metadata import MetadataExtractor
from custom_plugins.auto_measurements.core.file_manager import FileDataManager


class ConfigurationTests(unittest.TestCase):
    def test_loading_partial_config_merges_all_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"negative_mode": True}), encoding="utf-8")
            config = ConfigManager(path).load_config()
        self.assertTrue(config["negative_mode"])
        self.assertEqual(config["calibration_folder"], DEFAULT_CONFIG["calibration_folder"])
        self.assertEqual(config["calibration_conversion_method"], "auto")
        self.assertEqual(config["auto_measurement_dose_correction_factor"], 1.0)

    def test_save_replaces_config_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            manager = ConfigManager(path)
            self.assertTrue(manager.save_config({"negative_mode": True}))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"negative_mode": True})
            self.assertEqual(list(Path(directory).glob("rc_config_*.json")), [])

    def test_legacy_update_preference_is_migrated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"check_updates_on_startup": False}), encoding="utf-8")
            config = ConfigManager(path).load_config()
        self.assertFalse(config["automatic_updates"])

    def test_writable_directory_check_reports_a_controlled_permission_error(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "app.paths.PROJECT_ROOT", Path(directory)
        ), patch("app.paths.CALIBRATION_ROOT", Path(directory) / "calibration_data"), patch(
            "app.paths.tempfile.mkstemp", side_effect=PermissionError("denied")
        ):
            with self.assertRaises(ApplicationPermissionError):
                ensure_writable_directories()


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
        self.assertEqual(run.call_args.kwargs["creationflags"], getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0))


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


class AutoMeasurementNavigationTests(unittest.TestCase):
    def test_loaded_batch_path_updates_counter_and_both_navigation_buttons(self):
        class Control:
            def __init__(self):
                self.values = {}

            def config(self, **kwargs):
                self.values.update(kwargs)

        manager = FileDataManager.__new__(FileDataManager)
        manager.file_list = [r"C:\\batch\\first.tif", r"C:\\batch\\second.tif"]
        manager.current_file_index = 0
        manager.prev_button = Control()
        manager.next_button = Control()
        manager.file_counter_label = Control()
        manager.current_file_label = None

        self.assertTrue(manager.set_current_file_from_loaded_path(r"C:\\batch\\second.tif"))
        self.assertEqual(manager.current_file_index, 1)
        self.assertEqual(manager.file_counter_label.values["text"], "2/2")
        self.assertEqual(manager.prev_button.values["state"], "normal")
        self.assertEqual(manager.next_button.values["state"], "disabled")


class CalibrationSelectionTests(unittest.TestCase):
    def test_reselecting_a_calibration_restores_requested_dose_conversion(self):
        class Variable:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        class Processor:
            calibration_available = False

            def update_settings(self, _config):
                return True

            def has_field_flattening(self):
                return False

            def has_calibration(self):
                return self.calibration_available

            def has_image(self):
                return False

        window = MainWindow.__new__(MainWindow)
        window.flat_var = Variable(False)
        window.calibration_var = Variable(True)
        window._requested_flat_correction = False
        window._requested_calibration_correction = True
        window.app_config = {"calibration_folder": "default"}
        window.image_processor = Processor()

        window.apply_settings()
        self.assertFalse(window.calibration_var.get())
        self.assertTrue(window._requested_calibration_correction)

        window.image_processor.calibration_available = True
        window.app_config["calibration_folder"] = "validated_calibration"
        window.apply_settings()
        self.assertTrue(window.calibration_var.get())


class ReleaseMetadataTests(unittest.TestCase):
    def test_wheel_and_application_versions_match(self):
        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["project"]["version"], __version__)

    def test_installer_is_per_user_and_preserves_user_data_directories(self):
        installer = Path("installer/RadiochromicFilmAnalyzer.iss").read_text(encoding="utf-8")
        self.assertIn("DefaultDirName={localappdata}\\Programs", installer)
        self.assertIn("PrivilegesRequired=lowest", installer)
        self.assertIn("RadiochromicFilmAnalyzer-Setup", installer)
        self.assertIn("DisableDirPage=no", installer)
        for directory in ("logs", "temp", "custom_plugins", "calibration_data"):
            self.assertIn(f'Name: "{{app}}\\{directory}"; Flags: uninsneveruninstall', installer)

    def test_one_file_build_includes_a_startup_splash(self):
        spec = Path("radiochromic_scanning_tool.spec").read_text(encoding="utf-8")
        self.assertIn("Splash(", spec)
        self.assertIn("splash.binaries", spec)

    def test_one_file_build_explicitly_includes_analysis_plugin(self):
        spec = Path("radiochromic_scanning_tool.spec").read_text(encoding="utf-8")
        self.assertIn('"custom_plugins.analysis_tools"', spec)

    def test_packaging_uses_the_application_icon(self):
        spec = Path("radiochromic_scanning_tool.spec").read_text(encoding="utf-8")
        self.assertTrue(Path("resources/radiochromic_film_analyzer.ico").is_file())
        self.assertIn('icon="resources/radiochromic_film_analyzer.ico"', spec)

    def test_local_documentation_and_wiki_have_a_navigable_home_page(self):
        local_home = Path("docs/Home.md").read_text(encoding="utf-8")
        wiki_home = Path("wiki/Home.md").read_text(encoding="utf-8")
        sidebar = Path("wiki/_Sidebar.md").read_text(encoding="utf-8")
        for page in ("MATHEMATICAL_MODEL", "WORKFLOWS", "DATA_FORMATS", "ARCHITECTURE", "RELEASES"):
            self.assertTrue(Path("wiki", f"{page}.md").is_file())
            self.assertIn(page, local_home)
            self.assertIn(page, wiki_home)
            self.assertIn(page, sidebar)


if __name__ == "__main__":
    unittest.main()
