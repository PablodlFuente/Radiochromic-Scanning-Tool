"""Stable application paths independent of the process working directory."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "rc_config.json"
CALIBRATION_ROOT = PROJECT_ROOT / "calibration_data"
PLUGINS_ROOT = PROJECT_ROOT / "custom_plugins"
RECENT_FILES_FILE = PROJECT_ROOT / "recent_files.json"
