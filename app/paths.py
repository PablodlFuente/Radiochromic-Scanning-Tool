"""Stable bundled-resource and writable-data paths."""

import os
import sys
import tempfile
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", SOURCE_ROOT))
PROJECT_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else SOURCE_ROOT
CONFIG_FILE = PROJECT_ROOT / "rc_config.json"
CALIBRATION_ROOT = PROJECT_ROOT / "calibration_data"
PLUGINS_ROOT = BUNDLE_ROOT / "custom_plugins"
RECENT_FILES_FILE = PROJECT_ROOT / "recent_files.json"
APPLICATION_ICON = BUNDLE_ROOT / "resources" / "radiochromic_film_analyzer.ico"


class ApplicationPermissionError(PermissionError):
    """Raised when the selected application directory is not writable."""


def ensure_writable_directories() -> None:
    """Create and verify the application-owned writable directories.

    A user may deliberately install the application in a protected location
    such as ``Program Files``.  Directory creation alone is not sufficient in
    that case, because existing folders can still reject log writes.  Probe the
    log directory before the logging subsystem is configured so startup can
    fail with a controlled privilege message instead of an unhandled exception.
    """
    writable_directories = (
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "temp",
        PROJECT_ROOT / "custom_plugins",
        CALIBRATION_ROOT,
    )
    try:
        for directory in writable_directories:
            directory.mkdir(parents=True, exist_ok=True)
        descriptor, probe_path = tempfile.mkstemp(prefix=".write_probe_", dir=PROJECT_ROOT / "logs")
        os.close(descriptor)
        Path(probe_path).unlink(missing_ok=True)
    except OSError as exc:
        raise ApplicationPermissionError(
            f"The installation directory is not writable: {PROJECT_ROOT}"
        ) from exc
