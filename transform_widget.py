"""TransformWidget: 이미지 회전 + XY 이동 통합 패널 (슬라이더 + 스핀박스)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGraphicsLineItem,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from image_source import ImageSource
    from viewer import ImageViewer


class TransformWidget(QWidget):

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
        self._original: np.ndarray | None = source.get_frame(frame_idx, copy=True)
        self._grid_items: list[QGraphicsLineItem] = []
        self._applied = False

        self._build_ui()
        self._connect_signals()
        self._update_slider_ranges()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # -- Frame label --
        self._frame_label = QLabel()
        self._update_frame_label()
        root.addWidget(self._frame_label)

        # -- Rotate group --
        rotate_grp = QGroupBox("Rotate")
        rotate_lay = QHBoxLayout(rotate_grp)
        rotate_lay.addWidget(QLabel("Angle (°):"))
        self._slider_angle = QSlider(Qt.Orientation.Horizontal)
        self._slider_angle.setRange(-180, 180)
        self._slider_angle.setValue(0)
        rotate_lay.addWidget(self._slider_angle, stretch=1)
        self._spin_angle = QDoubleSpinBox()
        self._spin_angle.setRange(-360.0, 360.0)
        self._spin_angle.setDecimals(2)
        self._spin_angle.setSingleStep(0.1)
        self._spin_angle.setValue(0.0)
        self._spin_angle.setFixedWidth(80)
        rotate_lay.addWidget(self._spin_angle)
        root.addWidget(rotate_grp)

        # -- Shift group --
        shift_grp = QGroupBox("Shift")
        shift_lay = QVBoxLayout(shift_grp)

        row_x = QHBoxLayout()
        row_x.addWidget(QLabel("X (px):"))
        self._slider_x = QSlider(Qt.Orientation.Horizontal)
        self._slider_x.setRange(-500, 500)
        self._slider_x.setValue(0)
        row_x.addWidget(self._slider_x, stretch=1)
        self._spin_x = QSpinBox()
        self._spin_x.setRange(-9999, 9999)
        self._spin_x.setValue(0)
        self._spin_x.setFixedWidth(80)
        row_x.addWidget(self._spin_x)
        shift_lay.addLayout(row_x)

        row_y = QHBoxLayout()
        row_y.addWidget(QLabel("Y (px):"))
        self._slider_y = QSlider(Qt.Orientation.Horizontal)
        self._slider_y.setRange(-500, 500)
        self._slider_y.setValue(0)
        row_y.addWidget(self._slider_y, stretch=1)
        self._spin_y = QSpinBox()
        self._spin_y.setRange(-9999, 9999)
        self._spin_y.setValue(0)
        self._spin_y.setFixedWidth(80)
        row_y.addWidget(self._spin_y)
        shift_lay.addLayout(row_y)

        root.addWidget(shift_grp)

        # -- Options --
        opt_row = QHBoxLayout()
        self._chk_preview = QCheckBox("Preview")
        self._chk_preview.setChecked(True)
        opt_row.addWidget(self._chk_preview)
        self._chk_grid = QCheckBox("Grid")
        opt_row.addWidget(self._chk_grid)
        opt_row.addWidget(QLabel("Count:"))
        self._spin_grid = QSpinBox()
        self._spin_grid.setRange(2, 100)
        self._spin_grid.setValue(10)
        self._spin_grid.setEnabled(False)
        opt_row.addWidget(self._spin_grid)
        opt_row.addStretch()
        root.addLayout(opt_row)

        # -- Target --
        target_row = QHBoxLayout()
        self._rb_current = QRadioButton("Current frame")
        self._rb_all = QRadioButton("All frames")
        self._rb_current.setChecked(True)
        target_row.addWidget(self._rb_current)
        target_row.addWidget(self._rb_all)
        target_row.addStretch()
        root.addLayout(target_row)

        # -- Buttons --
        btn_row = QHBoxLayout()
        self._btn_apply = QPushButton("Apply")
        btn_row.addWidget(self._btn_apply)
        self._btn_reset = QPushButton("Reset")
        btn_row.addWidget(self._btn_reset)
        root.addLayout(btn_row)

        # -- Status --
        self._status = QLabel("")
        self._status.setStyleSheet("color: gray; font-style: italic;")
        root.addWidget(self._status)

        root.addStretch()

    def _connect_signals(self) -> None:
        self._slider_angle.valueChanged.connect(self._on_slider_angle)
        self._spin_angle.valueChanged.connect(self._on_spin_angle)
        self._slider_x.valueChanged.connect(self._on_slider_x)
        self._spin_x.valueChanged.connect(self._on_spin_x)
        self._slider_y.valueChanged.connect(self._on_slider_y)
        self._spin_y.valueChanged.connect(self._on_spin_y)
        self._chk_preview.toggled.connect(self._on_preview_toggled)
        self._chk_grid.toggled.connect(self._on_grid_toggled)
        self._spin_grid.valueChanged.connect(self._on_grid_changed)
        self._btn_apply.clicked.connect(self._on_apply)
        self._btn_reset.clicked.connect(self._on_reset)

    # ------------------------------------------------------------------ slider <-> spin sync
    def _on_slider_angle(self, val: int) -> None:
        self._spin_angle.blockSignals(True)
        self._spin_angle.setValue(float(val))
        self._spin_angle.blockSignals(False)
        self._on_param_changed()

    def _on_spin_angle(self, val: float) -> None:
        clamped = max(-180, min(180, int(round(val))))
        self._slider_angle.blockSignals(True)
        self._slider_angle.setValue(clamped)
        self._slider_angle.blockSignals(False)
        self._on_param_changed()

    def _on_slider_x(self, val: int) -> None:
        self._spin_x.blockSignals(True)
        self._spin_x.setValue(val)
        self._spin_x.blockSignals(False)
        self._on_param_changed()

    def _on_spin_x(self, val: int) -> None:
        clamped = max(self._slider_x.minimum(), min(self._slider_x.maximum(), val))
        self._slider_x.blockSignals(True)
        self._slider_x.setValue(clamped)
        self._slider_x.blockSignals(False)
        self._on_param_changed()

    def _on_slider_y(self, val: int) -> None:
        self._spin_y.blockSignals(True)
        self._spin_y.setValue(val)
        self._spin_y.blockSignals(False)
        self._on_param_changed()

    def _on_spin_y(self, val: int) -> None:
        clamped = max(self._slider_y.minimum(), min(self._slider_y.maximum(), val))
        self._slider_y.blockSignals(True)
        self._slider_y.setValue(clamped)
        self._slider_y.blockSignals(False)
        self._on_param_changed()

    # ------------------------------------------------------------------ public
    def set_frame_idx(self, idx: int) -> None:
        self._restore_if_dirty()
        self._frame_idx = idx
        self._original = self._source.get_frame(idx, copy=True)
        self._applied = False
        self._block_all_signals(True)
        self._slider_angle.setValue(0)
        self._spin_angle.setValue(0.0)
        self._slider_x.setValue(0)
        self._spin_x.setValue(0)
        self._slider_y.setValue(0)
        self._spin_y.setValue(0)
        self._block_all_signals(False)
        self._update_frame_label()
        self._update_slider_ranges()
        self._status.setText("")

    def cleanup(self) -> None:
        self._remove_grid()
        self._restore_if_dirty()

    # ------------------------------------------------------------------ transform
    @staticmethod
    def _shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
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

    @staticmethod
    def _transform_image(
        img: np.ndarray, angle: float, dx: int, dy: int,
    ) -> np.ndarray:
        no_rotate = abs(angle) < 1e-6
        no_shift = dx == 0 and dy == 0
        if no_rotate and no_shift:
            return img.copy()
        if no_rotate:
            return TransformWidget._shift_image(img, dx, dy)
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        return cv2.warpAffine(
            img, M, (w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    # ------------------------------------------------------------------ preview
    def _on_param_changed(self) -> None:
        if self._chk_preview.isChecked():
            self._show_preview()

    def _show_preview(self) -> None:
        if self._original is None:
            return
        angle = self._spin_angle.value()
        dx = self._spin_x.value()
        dy = self._spin_y.value()
        result = self._transform_image(self._original, angle, dx, dy)
        self._viewer.set_image(result)
        self._update_grid()

    def _on_preview_toggled(self, checked: bool) -> None:
        if checked:
            self._show_preview()
        else:
            if self._original is not None:
                self._viewer.set_image(self._original)

    # ------------------------------------------------------------------ grid
    def _grid_pen(self) -> QPen:
        pen = QPen(QColor(0, 200, 255, 120), 1)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        return pen

    def _update_grid(self) -> None:
        self._remove_grid()
        if not self._chk_grid.isChecked():
            return
        img = self._viewer.raw_image
        if img is None:
            return
        h, w = img.shape[:2]
        n = self._spin_grid.value()
        pen = self._grid_pen()
        scene = self._viewer.scene_ref
        for i in range(1, n):
            x = w * i / n
            item = scene.addLine(x, 0, x, h, pen)
            item.setZValue(50)
            self._grid_items.append(item)
        for i in range(1, n):
            y = h * i / n
            item = scene.addLine(0, y, w, y, pen)
            item.setZValue(50)
            self._grid_items.append(item)

    def _remove_grid(self) -> None:
        scene = self._viewer.scene_ref
        for item in self._grid_items:
            scene.removeItem(item)
        self._grid_items.clear()

    def _on_grid_toggled(self, checked: bool) -> None:
        self._spin_grid.setEnabled(checked)
        self._update_grid()

    def _on_grid_changed(self) -> None:
        if self._chk_grid.isChecked():
            self._update_grid()

    # ------------------------------------------------------------------ apply / reset
    def _on_apply(self) -> None:
        angle = self._spin_angle.value()
        dx = self._spin_x.value()
        dy = self._spin_y.value()
        if abs(angle) < 1e-6 and dx == 0 and dy == 0:
            self._status.setText("No transform to apply.")
            return
        if self._rb_all.isChecked():
            n = self._source.frame_count
            for i in range(n):
                img = self._source.get_frame(i, copy=True)
                if img is None:
                    continue
                self._source.set_frame(i, self._transform_image(img, angle, dx, dy))
            msg = f"Applied to {n} frames."
        else:
            img = self._source.get_frame(self._frame_idx, copy=True)
            if img is None:
                return
            self._source.set_frame(
                self._frame_idx, self._transform_image(img, angle, dx, dy),
            )
            msg = f"Applied to frame {self._frame_idx + 1}."
        self._original = self._source.get_frame(self._frame_idx, copy=True)
        self._applied = True
        self._block_all_signals(True)
        self._slider_angle.setValue(0)
        self._spin_angle.setValue(0.0)
        self._slider_x.setValue(0)
        self._spin_x.setValue(0)
        self._slider_y.setValue(0)
        self._spin_y.setValue(0)
        self._block_all_signals(False)
        img = self._source.get_frame(self._frame_idx, copy=False)
        if img is not None:
            self._viewer.set_image(img)
        self._update_grid()
        parts = []
        if abs(angle) >= 1e-6:
            parts.append(f"rotate {angle:.2f}°")
        if dx != 0 or dy != 0:
            parts.append(f"shift ({dx}, {dy})")
        self._status.setText(f"{', '.join(parts)} — {msg}")

    def _on_reset(self) -> None:
        self._block_all_signals(True)
        self._slider_angle.setValue(0)
        self._spin_angle.setValue(0.0)
        self._slider_x.setValue(0)
        self._spin_x.setValue(0)
        self._slider_y.setValue(0)
        self._spin_y.setValue(0)
        self._block_all_signals(False)
        if self._original is not None and self._chk_preview.isChecked():
            self._viewer.set_image(self._original)
        self._update_grid()
        self._status.setText("")

    # ------------------------------------------------------------------ helpers
    def _is_dirty(self) -> bool:
        if self._applied:
            return False
        angle = self._spin_angle.value()
        dx = self._spin_x.value()
        dy = self._spin_y.value()
        return abs(angle) >= 1e-6 or dx != 0 or dy != 0

    def _restore_if_dirty(self) -> None:
        if self._is_dirty() and self._original is not None:
            self._viewer.set_image(self._original)

    def _update_frame_label(self) -> None:
        name = self._source.frame_name(self._frame_idx)
        n = self._source.frame_count
        self._frame_label.setText(f"Frame: {name}  ({self._frame_idx + 1}/{n})")

    def _update_slider_ranges(self) -> None:
        if self._original is None:
            return
        h, w = self._original.shape[:2]
        self._slider_x.setRange(-w, w)
        self._slider_y.setRange(-h, h)

    def _block_all_signals(self, block: bool) -> None:
        for w in (
            self._slider_angle, self._spin_angle,
            self._slider_x, self._spin_x,
            self._slider_y, self._spin_y,
        ):
            w.blockSignals(block)
