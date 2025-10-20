"""Calibration Wizard that integrates the built-in `calibration_app` module
within the Radiochromic Film Analyzer.

All CSV files produced by the calibration workflow are redirected to a
`calibration_data` directory (created on-demand) to keep the project
root clean.
"""

from __future__ import annotations

# Built-ins & stdlib
import os
import sys
import tkinter as tk
from importlib import import_module
from types import ModuleType
from typing import Optional
from tkinter import messagebox

# Import ConfigManager to access calibration folder setting
from app.utils.config_manager import ConfigManager

# Fully-qualified import path of the internal calibration module copied
# into this repository.
_MODULE_NAME = "app.calibration.calibration_app"


def _load_external_calibration_module() -> ModuleType:
    """Import (or retrieve) the internal *calibration_app* module."""
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]

    # Standard import once the file is part of the package
    module = import_module(_MODULE_NAME)
    return module


class CalibrationWizardWindow(tk.Toplevel):
    """Top-level window that hosts the external *CalibrationApp*."""

    def __init__(self, parent: tk.Misc):  # parent is the main Tk or any widget
        super().__init__(parent)
        self.title("Calibration Wizard")
        # Provide ample default size; the embedded app can tweak it later.
        self.geometry("1200x800")

        # Place window relative to parent for nicer UX
        try:
            self.transient(parent)
            self.grab_set()  # Make it modal with respect to the main window
        except Exception:
            pass

        # Get calibration folder from config
        config_manager = ConfigManager()
        config = config_manager.load_config()
        calibration_folder = config.get("calibration_folder", "default")
        
        # Directory where all CSV outputs will be stored
        if calibration_folder == "default":
            self._data_dir = os.path.join(os.getcwd(), "calibration_data")
        else:
            self._data_dir = os.path.join(os.getcwd(), "calibration_data", calibration_folder)
        
        os.makedirs(self._data_dir, exist_ok=True)

        # Preserve original working directory so we can restore it when the
        # wizard is closed.  Redirect CWD so that the legacy script writes its
        # CSV files inside *calibration_data* transparently.
        self._old_cwd: str = os.getcwd()
        os.chdir(self._data_dir)

        # Import and instantiate the external application.
        try:
            module = _load_external_calibration_module()
            if not hasattr(module, "CalibrationApp"):
                raise AttributeError("El módulo externo no contiene la clase 'CalibrationApp'.")

            CalibrationApp = module.CalibrationApp  # type: ignore[attr-defined]
            # Instantiate the external calibration GUI, passing this toplevel as
            # its *root* so it integrates into the existing Tk loop.
            self._calibration_app = CalibrationApp(self)  # noqa: attribute-defined-outside-init
        except Exception as exc:
            # Restore cwd even if instantiation fails
            os.chdir(self._old_cwd)
            messagebox.showerror("Calibration Wizard", f"Error al cargar el asistente de calibración:\n{exc}")
            # Close this window since the wizard failed to load
            self.destroy()
            raise

        # Override close protocol to ensure CWD restoration
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_close(self):  # noqa: D401 – simple method
        """Handle the closing of the wizard window."""
        # Restore previous working directory so other parts of the
        # application remain unaffected.
        try:
            os.chdir(self._old_cwd)
        except Exception:
            pass
        self.destroy()

    # Public API (could be expanded later) ------------------------------

    def get_results_path(self) -> str:
        """Return the directory where the wizard saved its CSV results."""
        return self._data_dir
