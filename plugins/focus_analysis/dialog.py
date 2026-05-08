"""Focus Analysis Dialog: 설정 UI + 백그라운드 분석 워커."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .focus_analyzer import FocusAnalyzer
from .result_plotter import ResultPlotter

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from image_source import ImageSource


class _AnalysisWorker(QThread):

    progress_updated = Signal(int)
    status_updated = Signal(str)
    analysis_completed = Signal(dict)
    analysis_failed = Signal(str)

    def __init__(self, images, z_positions, settings):
        super().__init__()
        self.images = images
        self.z_positions = z_positions
        self.settings = settings

    def run(self):
        try:
            metric = self.settings["selected_metric"]
            grid_count = self.settings["grid_count"]
            output_path = self.settings["output_path"]

            self.status_updated.emit("Focus analysis starting...")
            self.progress_updated.emit(0)

            analyzer = FocusAnalyzer(grid_count)
            blob_params = self.settings.get("blob_params")
            if blob_params is not None:
                analyzer.set_blob_detector_params(blob_params)

            def focus_cb(progress):
                self.progress_updated.emit(int(progress * 0.7))

            df = analyzer.analyze_focus_for_images(
                self.images,
                self.z_positions,
                metric,
                focus_cb,
                output_path=output_path,
            )

            self.status_updated.emit(f"Analysis done: {len(df)} data points. Saving...")

            minimize = metric == "edge_spread_function"
            plotter = ResultPlotter(output_path)

            def result_cb(progress):
                self.progress_updated.emit(70 + int(progress * 0.3))

            files = plotter.process_and_save_results(
                df,
                metric,
                use_normalized_z=True,
                minimize_focus=minimize,
                show_diagonal_only=True,
                progress_callback=result_cb,
                focus_analyzer=analyzer,
                auto_depth_calculation=self.settings["auto_depth_calculation"],
                depth_threshold=self.settings.get("depth_threshold"),
            )

            self.progress_updated.emit(100)
            self.status_updated.emit("Complete.")
            self.analysis_completed.emit(files)

        except Exception as e:
            self.analysis_failed.emit(str(e))


class FocusAnalysisDialog(QDialog):

    def __init__(self, source: ImageSource, frame_idx: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Focus Analysis")
        self.setMinimumWidth(440)
        self._source = source
        self._worker: _AnalysisWorker | None = None
        self._blob_params: cv2.SimpleBlobDetector_Params | None = None

        self._build_ui()
        self._connect()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        first = self._source.get_frame(0, copy=False)
        n = self._source.frame_count
        if first is not None:
            h, w = first.shape[:2]
            root.addWidget(QLabel(f"Loaded: {n} frames ({w}×{h}, {first.dtype})"))
        else:
            root.addWidget(QLabel(f"Loaded: {n} frames"))

        grp = QGroupBox("Settings")
        form = QGridLayout(grp)

        form.addWidget(QLabel("Z step (mm):"), 0, 0)
        self._z_step = QDoubleSpinBox()
        self._z_step.setRange(0.001, 100.0)
        self._z_step.setValue(0.5)
        self._z_step.setDecimals(3)
        form.addWidget(self._z_step, 0, 1)

        form.addWidget(QLabel("Grid count:"), 1, 0)
        self._grid_count = QSpinBox()
        self._grid_count.setRange(3, 15)
        self._grid_count.setValue(9)
        self._grid_count.setSingleStep(2)
        form.addWidget(self._grid_count, 1, 1)

        form.addWidget(QLabel("Focus metric:"), 2, 0)
        self._metric_combo = QComboBox()
        self._metric_combo.addItems([
            "edge_spread_function",
            "laplacian_variance",
            "energy_laplacian",
            "curvature_measure",
        ])
        form.addWidget(self._metric_combo, 2, 1)

        self._btn_blob = QPushButton("Blob Parameters...")
        form.addWidget(self._btn_blob, 3, 0, 1, 2)

        form.addWidget(QLabel("Output folder:"), 4, 0)
        out_row = QHBoxLayout()
        self._output_edit = QLabel("")
        self._output_edit.setStyleSheet("color: gray; font-style: italic;")
        out_row.addWidget(self._output_edit, stretch=1)
        self._btn_browse = QPushButton("Browse")
        out_row.addWidget(self._btn_browse)
        form.addLayout(out_row, 4, 1)

        root.addWidget(grp)

        depth_grp = QGroupBox("Depth of Field")
        depth_layout = QGridLayout(depth_grp)

        self._chk_auto_depth = QCheckBox("Auto depth calculation")
        self._chk_auto_depth.setChecked(True)
        depth_layout.addWidget(self._chk_auto_depth, 0, 0, 1, 2)

        depth_layout.addWidget(QLabel("Depth threshold:"), 1, 0)
        self._depth_threshold = QDoubleSpinBox()
        self._depth_threshold.setRange(0.0, 10000.0)
        self._depth_threshold.setValue(10.0)
        self._depth_threshold.setDecimals(3)
        self._depth_threshold.setEnabled(False)
        depth_layout.addWidget(self._depth_threshold, 1, 1)

        root.addWidget(depth_grp)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        root.addWidget(self._progress)

        self._status = QLabel("Ready")
        self._status.setStyleSheet("color: gray;")
        root.addWidget(self._status)

        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("Run")
        self._btn_run.setDefault(True)
        self._btn_close = QPushButton("Close")
        btn_row.addStretch()
        btn_row.addWidget(self._btn_run)
        btn_row.addWidget(self._btn_close)
        root.addLayout(btn_row)

    def _connect(self) -> None:
        self._btn_browse.clicked.connect(self._browse_output)
        self._btn_blob.clicked.connect(self._open_blob_dialog)
        self._metric_combo.currentTextChanged.connect(self._on_metric_changed)
        self._chk_auto_depth.toggled.connect(
            lambda checked: self._depth_threshold.setEnabled(not checked)
        )
        self._btn_run.clicked.connect(self._on_run)
        self._btn_close.clicked.connect(self.close)

    def _on_metric_changed(self, metric: str) -> None:
        self._btn_blob.setEnabled(metric == "edge_spread_function")

    def _open_blob_dialog(self) -> None:
        from .blob_dialog import BlobDetectorDialog
        dlg = BlobDetectorDialog(
            self._source, self._grid_count.value(), self
        )
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.applied_params is not None:
            self._blob_params = dlg.applied_params
            self._status.setText("Blob parameters applied.")

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d:
            self._output_edit.setText(d)

    def _on_run(self) -> None:
        output = self._output_edit.text().strip()
        if not output:
            QMessageBox.warning(self, "Warning", "Select an output folder first.")
            return
        if self._source.frame_count < 3:
            QMessageBox.warning(self, "Warning", "Need at least 3 frames.")
            return

        images = self._prepare_images()
        z_step = self._z_step.value()
        z_positions = [i * z_step for i in range(len(images))]

        settings = {
            "selected_metric": self._metric_combo.currentText(),
            "grid_count": self._grid_count.value(),
            "output_path": output,
            "auto_depth_calculation": self._chk_auto_depth.isChecked(),
            "depth_threshold": self._depth_threshold.value(),
            "blob_params": self._blob_params,
        }

        self._btn_run.setEnabled(False)
        self._worker = _AnalysisWorker(images, z_positions, settings)
        self._worker.progress_updated.connect(self._progress.setValue)
        self._worker.status_updated.connect(self._status.setText)
        self._worker.analysis_completed.connect(self._on_completed)
        self._worker.analysis_failed.connect(self._on_failed)
        self._worker.start()

    def _prepare_images(self) -> list:
        images = []
        for i in range(self._source.frame_count):
            img = self._source.get_frame(i, copy=False)
            if img is None:
                continue
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.dtype != np.uint8:
                mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
                if mx > mn:
                    img = ((img.astype(np.float64) - mn) / (mx - mn) * 255).astype(
                        np.uint8
                    )
                else:
                    img = np.zeros(img.shape[:2], dtype=np.uint8)
            images.append(img)
        return images

    def _on_completed(self, files: dict) -> None:
        self._btn_run.setEnabled(True)
        self._worker = None
        msg = "Generated files:\n"
        for key, path in files.items():
            if path:
                msg += f"  {key}: {path}\n"
        QMessageBox.information(self, "Analysis Complete", msg)

    def _on_failed(self, error: str) -> None:
        self._btn_run.setEnabled(True)
        self._worker = None
        QMessageBox.critical(self, "Analysis Failed", error)

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Warning", "Analysis is still running.")
            event.ignore()
            return
        super().closeEvent(event)
