"""FrameOffsetWidget: shift a frame by x/y pixel offset with live preview."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from image_source import ImageSource
from viewer import ImageViewer


class FrameOffsetWidget(QWidget):

    def __init__(
        self,
        viewer: ImageViewer,
        source: ImageSource,
        frame_idx: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._source = source
        self._frame_idx = frame_idx
        self._original = source.get_frame(frame_idx, copy=True)
        self._applied = False

        self._build_ui()
        self._connect()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self._frame_label = QLabel()
        self._update_frame_label()
        root.addWidget(self._frame_label)

        row_x = QHBoxLayout()
        row_x.addWidget(QLabel("X offset (px):"))
        self._spin_x = QSpinBox()
        self._spin_x.setRange(-9999, 9999)
        self._spin_x.setValue(0)
        row_x.addWidget(self._spin_x)
        root.addLayout(row_x)

        row_y = QHBoxLayout()
        row_y.addWidget(QLabel("Y offset (px):"))
        self._spin_y = QSpinBox()
        self._spin_y.setRange(-9999, 9999)
        self._spin_y.setValue(0)
        row_y.addWidget(self._spin_y)
        root.addLayout(row_y)

        self._status = QLabel("")
        self._status.setStyleSheet("color: gray;")
        root.addWidget(self._status)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_reset = QPushButton("Reset")
        self._btn_apply = QPushButton("Apply")
        btn_row.addWidget(self._btn_reset)
        btn_row.addWidget(self._btn_apply)
        root.addLayout(btn_row)

    def _connect(self) -> None:
        self._spin_x.valueChanged.connect(self._update_preview)
        self._spin_y.valueChanged.connect(self._update_preview)
        self._btn_reset.clicked.connect(self._on_reset)
        self._btn_apply.clicked.connect(self._on_apply)

    @staticmethod
    def _shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
        if dx == 0 and dy == 0:
            return img.copy()
        h, w = img.shape[:2]
        result = np.zeros_like(img)
        sx0 = max(0, -dx)
        sx1 = min(w, w - dx)
        sy0 = max(0, -dy)
        sy1 = min(h, h - dy)
        dx0 = max(0, dx)
        dx1 = min(w, w + dx)
        dy0 = max(0, dy)
        dy1 = min(h, h + dy)
        if sx1 > sx0 and sy1 > sy0 and dx1 > dx0 and dy1 > dy0:
            result[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
        return result

    def _update_preview(self) -> None:
        dx = self._spin_x.value()
        dy = self._spin_y.value()
        preview = self._shift_image(self._original, dx, dy)
        self._viewer.set_image(preview)
        if dx != 0 or dy != 0:
            self._status.setText(f"Preview: dx={dx}, dy={dy}")
        else:
            self._status.setText("")

    def _on_reset(self) -> None:
        self._spin_x.setValue(0)
        self._spin_y.setValue(0)

    def _on_apply(self) -> None:
        dx = self._spin_x.value()
        dy = self._spin_y.value()
        shifted = self._shift_image(self._original, dx, dy)
        self._source.set_frame(self._frame_idx, shifted)
        self._original = shifted
        self._spin_x.setValue(0)
        self._spin_y.setValue(0)
        self._applied = True
        name = self._source.frame_name(self._frame_idx)
        self._status.setText(f"Applied offset ({dx}, {dy}) to {name}")

    def set_frame_idx(self, idx: int) -> None:
        self._restore_if_dirty()
        self._frame_idx = idx
        self._original = self._source.get_frame(idx, copy=True)
        self._applied = False
        self._spin_x.blockSignals(True)
        self._spin_y.blockSignals(True)
        self._spin_x.setValue(0)
        self._spin_y.setValue(0)
        self._spin_x.blockSignals(False)
        self._spin_y.blockSignals(False)
        self._update_frame_label()
        self._status.setText("")

    def _update_frame_label(self) -> None:
        name = self._source.frame_name(self._frame_idx)
        n = self._source.frame_count
        self._frame_label.setText(f"Frame: {name}  ({self._frame_idx + 1}/{n})")

    def _restore_if_dirty(self) -> None:
        if not self._applied:
            dx = self._spin_x.value()
            dy = self._spin_y.value()
            if dx != 0 or dy != 0:
                self._viewer.set_image(self._original)

    def cleanup(self) -> None:
        self._restore_if_dirty()
