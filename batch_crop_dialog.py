"""BatchCropDialog: ROI + frame offset crop 설정 다이얼로그 (이미지 프리뷰 포함)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from viewer import _normalize_to_u8

if TYPE_CHECKING:
    from image_source import ImageSource


class _PreviewView(QGraphicsView):
    """Ctrl+wheel=zoom, wheel=frame scroll, drag=pan."""

    frame_scroll = Signal(int)

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHints(self.renderHints())

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if delta > 0:
                self.scale(1.25, 1.25)
            elif delta < 0:
                self.scale(0.8, 0.8)
        else:
            if delta > 0:
                self.frame_scroll.emit(-1)
            elif delta < 0:
                self.frame_scroll.emit(1)
        event.accept()


class BatchCropDialog(QDialog):

    def __init__(
        self,
        roi: tuple[int, int, int, int],
        frame_count: int,
        img_w: int,
        img_h: int,
        source: ImageSource,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Crop")
        self.setMinimumSize(700, 600)

        self._roi = roi
        self._frame_count = frame_count
        self._img_w = img_w
        self._img_h = img_h
        self._source = source

        x0, y0, x1, y1 = roi
        rw, rh = x1 - x0, y1 - y0

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            f"ROI: ({x0}, {y0})  {rw} × {rh}    |    "
            f"Image: {img_w} × {img_h}  |  {frame_count} frames"
        ))

        # -- Offset controls --
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Offset X:"))
        self._spin_ox = QSpinBox()
        self._spin_ox.setRange(-img_w, img_w)
        self._spin_ox.setValue(0)
        ctrl_row.addWidget(self._spin_ox)
        ctrl_row.addWidget(QLabel("Offset Y:"))
        self._spin_oy = QSpinBox()
        self._spin_oy.setRange(-img_h, img_h)
        self._spin_oy.setValue(0)
        ctrl_row.addWidget(self._spin_oy)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # -- Image preview --
        self._scene = QGraphicsScene(self)
        self._view = _PreviewView(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._rect_item = None
        self._auto_min: float | None = None
        self._auto_max: float | None = None
        layout.addWidget(self._view, stretch=1)

        # -- Frame slider + spinbox --
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Frame:"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(1, max(frame_count, 1))
        self._frame_spin.setValue(1)
        slider_row.addWidget(self._frame_spin)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(frame_count - 1, 0))
        self._slider.setTracking(True)
        slider_row.addWidget(self._slider, stretch=1)
        self._frame_label = QLabel(f"/ {frame_count}")
        slider_row.addWidget(self._frame_label)
        layout.addLayout(slider_row)

        # -- Info + warning --
        self._info_label = QLabel()
        self._info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self._info_label)

        self._warning = QLabel()
        self._warning.setStyleSheet("color: orange;")
        layout.addWidget(self._warning)

        # -- Buttons --
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_execute = QPushButton("Execute")
        btn_execute.setDefault(True)
        btn_execute.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_execute)
        layout.addLayout(btn_row)

        # -- Signals --
        self._spin_ox.valueChanged.connect(self._on_param_changed)
        self._spin_oy.valueChanged.connect(self._on_param_changed)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._frame_spin.valueChanged.connect(self._on_spin_changed)
        self._view.frame_scroll.connect(self._on_view_scroll)

        self._show_frame(0)
        self._update_warning()

    # ------------------------------------------------------------------ preview
    def _show_frame(self, idx: int) -> None:
        img = self._source.get_frame(idx, copy=False)
        if img is None:
            return
        display = self._to_display_u8(img)
        h, w = display.shape[:2]

        if display.ndim == 2:
            qimg = QImage(display.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            qimg = QImage(display.data, w, h, 3 * w, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pixmap)
            self._scene.setSceneRect(self._pixmap_item.boundingRect())
            self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self._pixmap_item.setPixmap(pixmap)

        self._draw_roi_rect(idx)

    def _draw_roi_rect(self, idx: int) -> None:
        if self._rect_item is not None:
            self._scene.removeItem(self._rect_item)
            self._rect_item = None

        x0, y0, x1, y1 = self._roi
        rw, rh = x1 - x0, y1 - y0
        ox = self._spin_ox.value()
        oy = self._spin_oy.value()

        cx = x0 + idx * ox
        cy = y0 + idx * oy

        pen = QPen(QColor(0, 255, 0), 2)
        pen.setCosmetic(True)
        self._rect_item = self._scene.addRect(
            QRectF(cx, cy, rw, rh), pen
        )

        self._info_label.setText(f"Frame {idx + 1}: ROI at ({cx}, {cy})")

    def _to_display_u8(self, img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            if img.ndim == 3 and img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        if self._auto_min is None or self._auto_max is None:
            self._auto_min = float(np.nanmin(img))
            self._auto_max = float(np.nanmax(img))
        out = _normalize_to_u8(img, self._auto_min, self._auto_max)
        if out.ndim == 3 and out.shape[2] == 3:
            return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out

    # ------------------------------------------------------------------ slots
    def _on_slider_changed(self, idx: int) -> None:
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(idx + 1)
        self._frame_spin.blockSignals(False)
        self._show_frame(idx)

    def _on_spin_changed(self, display_idx: int) -> None:
        idx = display_idx - 1
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)
        self._show_frame(idx)

    def _on_view_scroll(self, delta: int) -> None:
        new_val = self._slider.value() + delta
        new_val = max(0, min(new_val, self._slider.maximum()))
        self._slider.setValue(new_val)

    def _on_param_changed(self) -> None:
        idx = self._slider.value()
        self._draw_roi_rect(idx)
        self._update_warning()

    def _update_warning(self) -> None:
        x0, y0, x1, y1 = self._roi
        rw, rh = x1 - x0, y1 - y0
        ox = self._spin_ox.value()
        oy = self._spin_oy.value()
        n = self._frame_count

        oob = 0
        for i in range(n):
            cx = x0 + i * ox
            cy = y0 + i * oy
            if cx < 0 or cy < 0 or cx + rw > self._img_w or cy + rh > self._img_h:
                oob += 1

        if oob > 0:
            self._warning.setText(
                f"Warning: {oob} frame(s) partially out of bounds (zero-padded)")
        else:
            self._warning.setText("")

    # ------------------------------------------------------------------ result
    def get_config(self) -> dict:
        return {
            "offset_x": self._spin_ox.value(),
            "offset_y": self._spin_oy.value(),
        }
