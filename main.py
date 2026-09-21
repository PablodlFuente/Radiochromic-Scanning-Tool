#!/usr/bin/env python3
"""
Radiochromic Film Analyzer - Main Entry Point

This script initializes and runs the Radiochromic Film Analyzer application.
"""

import sys
import os
import logging
import datetime
from app.rc_analyzer import RCAnalyzer
from app.paths import ApplicationPermissionError, PROJECT_ROOT, ensure_writable_directories
from app.utils.config_manager import ConfigManager
from app.utils.updater import UpdateChecker

def _show_permission_error(error):
    """Report a protected installation directory before Tk/logging starts."""
    message = (
        f"{error}\n\n"
        "Run the installer or application with administrator privileges, or choose a folder "
        "where you have write permission. The application will now close."
    )
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                None, message, "Radiochromic Film Analyzer - Permission required", 0x10
            )
            return
        except Exception:
            pass
    print(message, file=sys.stderr)


# Configure logging only after proving that the selected installation directory
# accepts writes. This avoids PyInstaller's unhandled-exception dialog.
try:
    ensure_writable_directories()
except ApplicationPermissionError as exc:
    _show_permission_error(exc)
    raise SystemExit(1)
logs_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create timestamped log filename, e.g. rc_analyzer_20250614_193835.log
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(logs_dir, f"rc_analyzer_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def _update_splash(message):
    """Update the PyInstaller splash when running from the packaged app."""
    try:
        import pyi_splash
        pyi_splash.update_text(message)
    except Exception:
        pass


def _close_splash():
    try:
        import pyi_splash
        pyi_splash.close()
    except Exception:
        pass


def _apply_startup_update() -> bool:
    """Install a newer release before constructing the main application window."""
    if not getattr(sys, "frozen", False):
        return False
    config = ConfigManager().load_config()
    if not config.get("automatic_updates", True):
        return False
    _update_splash("Checking for published updates…")
    try:
        result = UpdateChecker().install_latest_published_release()
    except Exception:
        logger.exception("Startup update check failed")
        return False
    if result.get("update_started"):
        logger.info("A newer release is installing before application startup")
        return True
    if not result.get("success"):
        logger.warning("Startup update failed: %s", result.get("error"))
    return False

def main():
    """Main entry point for the application."""
    try:
        logger.info("Starting Radiochromic Film Analyzer")
        if _apply_startup_update():
            return
        _update_splash("Starting Radiochromic Film Analyzer…")
        app = RCAnalyzer()
        _close_splash()
        app.mainloop()
        logger.info("Application closed normally")
    except Exception as e:
        _close_splash()
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
