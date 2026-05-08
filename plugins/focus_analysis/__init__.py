"""Focus Analysis Plugin: field curvature / depth-of-field 분석.

Migrated from: focus_analysis_gui/core/
Core logic: focus_analyzer.py, result_plotter.py (no Qt dependency)
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from image_source import ImageSource
from plugins.plugin_base import PluginBase


class FocusAnalysisPlugin(PluginBase):

    @property
    def name(self) -> str:
        return "Focus Analysis"

    def run(self, source: ImageSource, frame_idx: int, parent) -> None:
        if not source.is_loaded:
            QMessageBox.warning(parent, "Warning", "No image loaded.")
            return
        from .dialog import FocusAnalysisDialog
        dlg = FocusAnalysisDialog(source, frame_idx, parent)
        dlg.exec()
