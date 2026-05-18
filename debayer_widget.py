"""DebayerWidget: Demosaic 설정 위젯. QDockWidget 내부에 배치."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from image_source import ImageSource
from viewer import ImageViewer

DEBAYER_MAP = {
    "bayer_gr": cv2.COLOR_BayerGR2BGR,
    "bayer_rg": cv2.COLOR_BayerRG2BGR,
    "bayer_bg": cv2.COLOR_BayerBG2BGR,
    "bayer_gb": cv2.COLOR_BayerGB2BGR,
}


class DebayerWidget(QWidget):

    def __init__(
        self,
        viewer: ImageViewer,
        source: ImageSource,
        frame_idx: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(360)
        self._viewer = viewer
        self._source = source
        self._frame_idx = frame_idx
        self._previewing = False

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(50)
        self._debounce.timeout.connect(self._flush_preview)

        self._build_ui()
        self._connect()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        row_pat = QHBoxLayout()
        row_pat.addWidget(QLabel("Pattern:"))
        self._pattern_combo = QComboBox()
        self._pattern_combo.addItems(list(DEBAYER_MAP.keys()))
        row_pat.addWidget(self._pattern_combo)
        row_pat.addStretch()
        root.addLayout(row_pat)

        self._r_slider, self._r_spin = self._make_slider_spin(200)
        self._g_slider, self._g_spin = self._make_slider_spin(100)
        self._b_slider, self._b_spin = self._make_slider_spin(200)
        root.addLayout(self._wrap_row("R weight (%):", self._r_slider, self._r_spin))
        root.addLayout(self._wrap_row("G weight (%):", self._g_slider, self._g_spin))
        root.addLayout(self._wrap_row("B weight (%):", self._b_slider, self._b_spin))

        self._chk_grayscale = QCheckBox("Grayscale output")
        root.addWidget(self._chk_grayscale)

        self._chk_preview = QCheckBox("Preview")
        root.addWidget(self._chk_preview)

        self._chk_propagate = QCheckBox("Propagate to all frames")
        self._chk_propagate.setChecked(True)
        root.addWidget(self._chk_propagate)

        btn_row = QHBoxLayout()
        self._btn_reset = QPushButton("Reset")
        self._btn_apply = QPushButton("Apply")
        self._btn_apply.setDefault(True)
        btn_row.addWidget(self._btn_reset)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_apply)
        root.addLayout(btn_row)

        root.addStretch()

    def _make_slider_spin(self, default: int):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 300)
        slider.setValue(default)
        slider.setSingleStep(10)
        slider.setPageStep(10)
        spin = QSpinBox()
        spin.setRange(0, 300)
        spin.setValue(default)
        spin.setSuffix(" %")
        spin.setMinimumWidth(70)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        return slider, spin

    @staticmethod
    def _wrap_row(label: str, slider, spin):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(slider, stretch=1)
        row.addWidget(spin)
        return row

    # ------------------------------------------------------------------ signals
    def _connect(self) -> None:
        self._pattern_combo.currentIndexChanged.connect(self._on_param_changed)
        for s in (self._r_spin, self._g_spin, self._b_spin):
            s.valueChanged.connect(self._on_param_changed)
        self._chk_grayscale.toggled.connect(self._on_grayscale_toggled)
        self._chk_preview.toggled.connect(self._on_preview_toggled)
        self._btn_reset.clicked.connect(self._on_reset)
        self._btn_apply.clicked.connect(self._on_apply)

    # ------------------------------------------------------------------ debayer
    def _debayer_frame(self, img: np.ndarray) -> np.ndarray:
        method = self._pattern_combo.currentText()
        code = DEBAYER_MAP[method]
        gray = img
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, code).astype(np.float32)
        r = self._r_spin.value() / 100.0
        g = self._g_spin.value() / 100.0
        b = self._b_spin.value() / 100.0
        result *= np.array([b, g, r], dtype=np.float32)
        if self._chk_grayscale.isChecked():
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        np.clip(result, 0, 255, out=result)
        return result.astype(np.uint8)

    def _get_weights(self) -> tuple[float, float, float]:
        if self._chk_grayscale.isChecked():
            return (1.0, 1.0, 1.0)
        return (
            self._r_spin.value() / 100.0,
            self._g_spin.value() / 100.0,
            self._b_spin.value() / 100.0,
        )

    # ------------------------------------------------------------------ preview
    def _on_param_changed(self) -> None:
        if self._chk_preview.isChecked():
            self._debounce.start()

    def _on_grayscale_toggled(self, checked: bool) -> None:
        enabled = not checked
        for w in (self._r_slider, self._r_spin, self._g_slider, self._g_spin,
                  self._b_slider, self._b_spin):
            w.setEnabled(enabled)
        if checked:
            self._r_slider.setValue(100)
            self._g_slider.setValue(100)
            self._b_slider.setValue(100)
        self._on_param_changed()

    def _on_preview_toggled(self, checked: bool) -> None:
        if checked:
            self._flush_preview()
        else:
            self._restore_original()

    def _flush_preview(self) -> None:
        if not self._chk_preview.isChecked():
            return
        img = self._source.get_frame(self._frame_idx, copy=False)
        if img is None:
            return
        result = self._debayer_frame(img)
        self._previewing = True
        self._viewer.set_image(result)

    def _restore_original(self) -> None:
        if self._previewing:
            img = self._source.get_frame(self._frame_idx, copy=False)
            if img is not None:
                self._viewer.set_image(img)
            self._previewing = False

    # ------------------------------------------------------------------ actions
    def _on_reset(self) -> None:
        self._pattern_combo.setCurrentIndex(0)
        self._r_slider.setValue(200)
        self._g_slider.setValue(100)
        self._b_slider.setValue(200)
        self._chk_grayscale.setChecked(False)

    def _on_apply(self) -> None:
        propagate = self._chk_propagate.isChecked()
        if propagate:
            indices = list(range(self._source.frame_count))
        else:
            indices = [self._frame_idx]

        n = len(indices)
        if n > 1:
            progress = QProgressDialog("Applying demosaic...", "Cancel", 0, n, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            for count, i in enumerate(indices):
                if progress.wasCanceled():
                    break
                img = self._source.get_frame(i, copy=False)
                if img is not None:
                    self._source.set_frame(i, self._debayer_frame(img))
                progress.setValue(count + 1)
                QApplication.processEvents()
            progress.close()
        else:
            img = self._source.get_frame(indices[0], copy=False)
            if img is not None:
                self._source.set_frame(indices[0], self._debayer_frame(img))

        self._chk_preview.setChecked(False)
        self._previewing = False
        new_img = self._source.get_frame(self._frame_idx, copy=False)
        if new_img is not None:
            self._viewer._auto_min = None
            self._viewer._auto_max = None
            self._viewer.set_image(new_img)

    def set_frame_idx(self, idx: int) -> None:
        self._frame_idx = idx
        if self._chk_preview.isChecked():
            self._debounce.start()

    def cleanup(self) -> None:
        self._debounce.stop()
        self._restore_original()
