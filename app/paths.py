"""Stable bundled-resource and writable-data paths."""

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", SOURCE_ROOT))
PROJECT_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else SOURCE_ROOT
CONFIG_FILE = PROJECT_ROOT / "rc_config.json"
CALIBRATION_ROOT = PROJECT_ROOT / "calibration_data"
PLUGINS_ROOT = BUNDLE_ROOT / "custom_plugins"
RECENT_FILES_FILE = PROJECT_ROOT / "recent_files.json"
APPLICATION_ICON = BUNDLE_ROOT / "resources" / "radiochromic_film_analyzer.ico"


def ensure_writable_directories() -> None:
    """Create the application-owned directories beside the executable.

    The installer creates these folders during setup.  Keeping this function
    also makes a manually copied executable usable and never overwrites user
    data during a release update.
    """
    for directory in (
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "temp",
        PROJECT_ROOT / "custom_plugins",
        CALIBRATION_ROOT,
        PROJECT_ROOT / "docs",
    ):
        directory.mkdir(parents=True, exist_ok=True)
