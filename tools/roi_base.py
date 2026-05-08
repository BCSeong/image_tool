"""ROI Tool 공통 베이스: Rectangle/Ellipse 공유 로직."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPen, QColor
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tool_base import BaseTool
from viewer import ImageViewer


class RoiToolBase(BaseTool):
    """Rectangle/Ellipse 공통: 드래그 → 도형 → 통계 계산."""

    def __init__(self, viewer: ImageViewer) -> None:
        self._viewer = viewer
        self._anchor_x = self._anchor_y = 0
        self._x0 = self._y0 = self._x1 = self._y1 = 0
        self._drawing = False
        self._moving = False
        self._move_ox = self._move_oy = 0
        self._has_roi = False
        self._shape_item = None
        self._panel: QWidget | None = None
        self._labels: dict[str, QLabel] = {}

    def _pen(self) -> QPen:
        pen = QPen(QColor(0, 255, 0), 2)
        pen.setCosmetic(True)
        return pen

    # -- 서브클래스가 구현 --
    @abstractmethod
    def _create_or_update_shape(self) -> None:
        """scene 위에 도형을 그리거나 갱신."""

    @abstractmethod
    def _make_mask(self, h: int, w: int) -> np.ndarray:
        """ROI 영역의 bool 마스크 반환 (shape = (h, w))."""

    # -- 좌표 계산 (Shift/Ctrl 반영) --
    def _update_coords(self, x: int, y: int, event) -> None:
        ax, ay = self._anchor_x, self._anchor_y
        dx, dy = x - ax, y - ay

        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

        if shift:
            side = max(abs(dx), abs(dy))
            dx = side * (1 if dx >= 0 else -1)
            dy = side * (1 if dy >= 0 else -1)

        if ctrl:
            self._x0 = ax - abs(dx)
            self._y0 = ay - abs(dy)
            self._x1 = ax + abs(dx)
            self._y1 = ay + abs(dy)
        else:
            self._x0 = ax
            self._y0 = ay
            self._x1 = ax + dx
            self._y1 = ay + dy

    # -- hit test (서브클래스가 오버라이드 가능) --
    def _hit_test(self, x: int, y: int) -> bool:
        x0, y0, x1, y1 = self._roi_rect()
        return x0 <= x <= x1 and y0 <= y <= y1

    def _shift_all(self, dx: int, dy: int) -> None:
        self._x0 += dx
        self._y0 += dy
        self._x1 += dx
        self._y1 += dy
        self._anchor_x += dx
        self._anchor_y += dy

    # -- 마우스 핸들러 --
    def on_mouse_press(self, x: int, y: int, event) -> bool:
        if self._has_roi and self._hit_test(x, y):
            self._moving = True
            self._move_ox, self._move_oy = x, y
            return True
        self._moving = False
        self._anchor_x, self._anchor_y = x, y
        self._x0, self._y0 = x, y
        self._x1, self._y1 = x, y
        self._drawing = True
        if self._shape_item is not None:
            self._viewer.scene_ref.removeItem(self._shape_item)
            self._shape_item = None
        return True

    def on_mouse_move(self, x: int, y: int, event) -> None:
        if self._moving:
            dx, dy = x - self._move_ox, y - self._move_oy
            self._shift_all(dx, dy)
            self._move_ox, self._move_oy = x, y
            self._create_or_update_shape()
            return
        if not self._drawing:
            return
        self._update_coords(x, y, event)
        self._create_or_update_shape()

    def on_mouse_release(self, x: int, y: int, event) -> None:
        if self._moving:
            self._moving = False
            self._update_roi_info()
            return
        if not self._drawing:
            return
        self._update_coords(x, y, event)
        self._drawing = False
        self._has_roi = True
        self._create_or_update_shape()
        self._update_roi_info()

    # -- 키보드: 화살표로 ROI 1px 이동 --
    def on_key_press(self, key: int, event) -> bool:
        if not self._has_roi:
            return False
        dx, dy = 0, 0
        if key == Qt.Key.Key_Left:
            dx = -1
        elif key == Qt.Key.Key_Right:
            dx = 1
        elif key == Qt.Key.Key_Up:
            dy = -1
        elif key == Qt.Key.Key_Down:
            dy = 1
        else:
            return False
        self._shift_all(dx, dy)
        self._create_or_update_shape()
        self._update_roi_info()
        return True

    def on_frame_changed(self, idx: int, image: np.ndarray | None) -> None:
        pass

    def select_all(self, w: int, h: int) -> None:
        """이미지 전체를 ROI로 설정."""
        self._anchor_x, self._anchor_y = 0, 0
        self._x0, self._y0 = 0, 0
        self._x1, self._y1 = w, h
        self._has_roi = True
        if self._shape_item is not None:
            self._viewer.scene_ref.removeItem(self._shape_item)
            self._shape_item = None
        self._create_or_update_shape()
        self._update_roi_info()

    def deactivate(self) -> None:
        self._shape_item = None
        self._has_roi = False

    # -- 통계 계산 --
    def _roi_rect(self) -> tuple[int, int, int, int]:
        """정규화된 (x0, y0, x1, y1) 반환."""
        x0, x1 = sorted((self._x0, self._x1))
        y0, y1 = sorted((self._y0, self._y1))
        return x0, y0, x1, y1

    # -- public API (MeasureWidget에서 사용) --
    @property
    def has_roi(self) -> bool:
        return self._has_roi

    def get_roi_rect(self) -> tuple[int, int, int, int] | None:
        """정규화된 (x0, y0, x1, y1) 반환. ROI 없으면 None."""
        if not self._has_roi:
            return None
        return self._roi_rect()

    def get_mask(self, h: int, w: int) -> np.ndarray | None:
        if not self._has_roi:
            return None
        return self._make_mask(h, w)

    # -- ROI 좌표 표시 --
    def _update_roi_info(self) -> None:
        if not self._labels:
            return
        x0, y0, x1, y1 = self._roi_rect()
        self._labels["x"].setText(str(x0))
        self._labels["y"].setText(str(y0))
        self._labels["w"].setText(str(x1 - x0))
        self._labels["h"].setText(str(y1 - y0))

    # -- 패널 --
    def build_panel(self) -> QWidget | None:
        self._panel = QWidget()
        layout = QVBoxLayout(self._panel)

        form = QFormLayout()
        for key in ("x", "y", "w", "h"):
            lbl = QLabel("—")
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._labels[key] = lbl
            form.addRow(f"{key.upper()}:", lbl)
        layout.addLayout(form)

        layout.addStretch()
        return self._panel
