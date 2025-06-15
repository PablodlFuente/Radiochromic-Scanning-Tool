"""Plugin management for Radiochromic Film Analyzer.

This module provides a singleton ``plugin_manager`` which is responsible for
loading, enabling/disabling and executing user-supplied image-processing
plugins.  Plugins live as ``.py`` files in the *custom_plugins* directory at the
project root.  Each plugin must expose a callable named ``process`` with the
signature::

    def process(image: numpy.ndarray) -> numpy.ndarray

The function receives the current *image* (either RGB uint8 or single-channel
float32 dose image) and must return a numpy array with the same shape expected
by the application.  Plugins are executed sequentially in load order.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from types import ModuleType
from typing import Dict, List

import logging

logger = logging.getLogger(__name__)


class PluginManager:
    """Simple runtime plugin loader/executor."""

    def __init__(self, plugins_dir: str):
        self.plugins_dir = plugins_dir
        os.makedirs(self.plugins_dir, exist_ok=True)

        # name -> module
        self._plugins: Dict[str, ModuleType] = {}
        # name -> active flag
        self._active: Dict[str, bool] = {}

        # UI context
        self._main_window = None
        self._notebook = None
        self._image_processor = None

        # UI tabs per plugin name
        self._tabs: Dict[str, object] = {}

        self.scan_plugins()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scan_plugins(self) -> None:
        """Discover and import all .py files in *plugins_dir*."""
        self._plugins.clear()
        self._active.clear()

        for file in sorted(os.listdir(self.plugins_dir)):
            if not file.endswith(".py"):
                continue
            path = os.path.join(self.plugins_dir, file)
            name = os.path.splitext(file)[0]
            try:
                module = self._load_module(name, path)
                if module is not None and hasattr(module, "process"):
                    self._plugins[name] = module
                    self._active[name] = True  # default enabled
                    logger.info("Loaded plugin '%s'", name)
                else:
                    logger.warning("Plugin '%s' ignored – no 'process' function", name)
            except Exception as exc:
                logger.error("Failed to load plugin '%s': %s", name, exc, exc_info=True)

    def load_plugin_file(self, source_path: str) -> str | None:
        """Copy *source_path* into *plugins_dir* and load it.

        Returns the plugin name if successful, *None* otherwise.
        """
        if not source_path.lower().endswith(".py"):
            logger.warning("Plugin file must be a .py: %s", source_path)
            return None

        dest_name = os.path.basename(source_path)
        dest_path = os.path.join(self.plugins_dir, dest_name)
        try:
            shutil.copy2(source_path, dest_path)
            logger.info("Copied plugin to %s", dest_path)
        except Exception as exc:
            logger.error("Failed to copy plugin: %s", exc, exc_info=True)
            return None

        # (Re)load
        name = os.path.splitext(dest_name)[0]
        try:
            module = self._load_module(name, dest_path, force_reload=True)
            if module and hasattr(module, "process"):
                self._plugins[name] = module
                self._active[name] = True
                # If UI context already available, immediately add tab
                if self._notebook is not None:
                    self._enable_plugin_ui(name)
                return name
            logger.warning("File '%s' is not a valid plugin (missing 'process')", dest_name)
        except Exception as exc:
            logger.error("Error loading plugin '%s': %s", name, exc, exc_info=True)
        return None

    def init_ui(self, main_window, notebook, image_processor):
        """Provide UI context so plugins can create tabs."""
        self._main_window = main_window
        self._notebook = notebook
        self._image_processor = image_processor

        # Activate tabs for already-enabled plugins
        for name, act in self._active.items():
            if act:
                self._enable_plugin_ui(name)

    def _enable_plugin_ui(self, name: str):
        if name in self._tabs or self._notebook is None:
            return
        mod = self._plugins[name]
        if hasattr(mod, "setup"):
            try:
                frame = mod.setup(self._main_window, self._notebook, self._image_processor)
                if frame is not None:
                    self._notebook.add(frame, text=getattr(mod, "TAB_TITLE", name))
                    self._tabs[name] = frame
            except Exception as exc:
                logger.error("Plugin '%s' setup failed: %s", name, exc, exc_info=True)

    def _disable_plugin_ui(self, name: str):
        if name not in self._tabs or self._notebook is None:
            return
        frame = self._tabs.pop(name)
        try:
            index = self._notebook.index(frame)
            self._notebook.forget(index)
        except Exception:
            pass

    def set_active(self, name: str, active: bool) -> None:
        if name in self._active:
            self._active[name] = active
            logger.info("Plugin '%s' active=%s", name, active)
            # UI handling
            if active:
                self._enable_plugin_ui(name)
            else:
                self._disable_plugin_ui(name)

    def is_active(self, name: str) -> bool:
        return self._active.get(name, False)

    def plugin_names(self) -> List[str]:
        return list(self._plugins.keys())

    def has_active_plugins(self) -> bool:
        return any(self._active.values())

    def apply_plugins(self, image):
        """Sequentially apply all active plugins to *image*.

        Each plugin receives the image returned by the previous one.  Any
        exceptions inside a plugin are caught and logged – the chain continues
        with the last valid image.
        """
        if not self.has_active_plugins():
            return image

        for name in self.plugin_names():
            if not self.is_active(name):
                continue
            mod = self._plugins[name]
            try:
                image = mod.process(image)
            except Exception as exc:
                logger.error("Plugin '%s' raised an exception: %s", name, exc, exc_info=True)
        return image

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_module(self, mod_name: str, path: str, *, force_reload: bool = False) -> ModuleType | None:
        if mod_name in sys.modules and not force_reload:
            return sys.modules[mod_name]

        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None:
            raise ImportError(f"Cannot create spec for {path}")
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        assert loader is not None  # for mypy
        loader.exec_module(module)
        sys.modules[mod_name] = module
        return module


# ----------------------------------------------------------------------
# Singleton accessible from the rest of the application
# ----------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_plugins_root = os.path.join(_project_root, "custom_plugins")
plugin_manager = PluginManager(_plugins_root)
