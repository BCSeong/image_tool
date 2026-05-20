"""ROI Tool 공통 베이스: Rectangle/Ellipse 공유 로직."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QPen, QColor
from PySide6.QtWidgets import (
    QFormLayout,
    QGraphicsRectItem,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tool_base import BaseTool
from viewer import ImageViewer


class RoiToolBase(BaseTool):
    """Rectangle/Ellipse 공통: 드래그 → 도형 → 통계 계산."""

    _HANDLE_HALF = 4  # 핸들 반폭 (화면 픽셀)

    def __init__(self, viewer: ImageViewer) -> None:
        self._viewer = viewer
        self._anchor_x = self._anchor_y = 0
        self._x0 = self._y0 = self._x1 = self._y1 = 0
        self._drawing = False
        self._moving = False
        self._move_ox = self._move_oy = 0
        self._has_roi = False
        self._shape_item = None
        self._handle_items: list[QGraphicsRectItem] = []
        self._resizing = False
        self._resize_handle = -1
        self._resize_ratio: float | None = None
        self._panel: QWidget | None = None
        self._spins: dict[str, QSpinBox] = {}

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

    @abstractmethod
    def _handle_positions(self) -> list[tuple[float, float]]:
        """핸들 중심 좌표 리스트 (scene 좌표)."""

    @abstractmethod
    def _apply_resize(self, handle: int, x: int, y: int, shift: bool) -> None:
        """핸들 드래그에 따라 ROI 좌표 업데이트."""

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

    # -- 핸들 관리 --
    def _handle_pen(self) -> QPen:
        pen = QPen(QColor(0, 255, 0), 1)
        pen.setCosmetic(True)
        return pen

    def _scene_pixel_size(self) -> float:
        t = self._viewer.transform()
        return 1.0 / t.m11() if t.m11() != 0 else 1.0

    def _update_handles(self) -> None:
        if not self._has_roi:
            self._remove_handles()
            return
        positions = self._handle_positions()
        ps = self._scene_pixel_size()
        half = self._HANDLE_HALF * ps
        scene = self._viewer.scene_ref
        pen = self._handle_pen()
        brush = QBrush(QColor(255, 255, 255))
        while len(self._handle_items) < len(positions):
            item = QGraphicsRectItem()
            item.setPen(pen)
            item.setBrush(brush)
            item.setZValue(100)
            scene.addItem(item)
            self._handle_items.append(item)
        for i, (cx, cy) in enumerate(positions):
            self._handle_items[i].setRect(cx - half, cy - half, half * 2, half * 2)
            self._handle_items[i].setVisible(True)
        for i in range(len(positions), len(self._handle_items)):
            self._handle_items[i].setVisible(False)

    def _remove_handles(self) -> None:
        scene = self._viewer.scene_ref
        for item in self._handle_items:
            scene.removeItem(item)
        self._handle_items.clear()

    def _handle_hit_test(self, x: int, y: int) -> int:
        if not self._has_roi:
            return -1
        ps = self._scene_pixel_size()
        radius = (self._HANDLE_HALF + 2) * ps
        for i, (cx, cy) in enumerate(self._handle_positions()):
            if abs(x - cx) <= radius and abs(y - cy) <= radius:
                return i
        return -1

    # -- hit test (서브클래스가 오버라이드 가능) --
    def _hit_test(self, x: int, y: int) -> bool:
        x0, y0, x1, y1 = self._roi_rect()
        return x0 <= x <= x1 and y0 <= y <= y1

    def _normalize_coords(self) -> None:
        x0, y0, x1, y1 = self._roi_rect()
        self._x0, self._y0, self._x1, self._y1 = x0, y0, x1, y1
        self._anchor_x, self._anchor_y = x0, y0

    def _shift_all(self, dx: int, dy: int) -> None:
        self._x0 += dx
        self._y0 += dy
        self._x1 += dx
        self._y1 += dy
        self._anchor_x += dx
        self._anchor_y += dy

    # -- 마우스 핸들러 --
    def on_mouse_press(self, x: int, y: int, event) -> bool:
        # 1) 핸들 리사이즈
        h_idx = self._handle_hit_test(x, y)
        if h_idx >= 0:
            self._resizing = True
            self._resize_handle = h_idx
            x0, y0, x1, y1 = self._roi_rect()
            w, h = x1 - x0, y1 - y0
            self._resize_ratio = w / h if h > 0 else None
            return True
        # 2) ROI 이동
        if self._has_roi and self._hit_test(x, y):
            self._moving = True
            self._move_ox, self._move_oy = x, y
            return True
        # 3) 새 ROI 그리기
        self._moving = False
        self._anchor_x, self._anchor_y = x, y
        self._x0, self._y0 = x, y
        self._x1, self._y1 = x, y
        self._drawing = True
        if self._shape_item is not None:
            self._viewer.scene_ref.removeItem(self._shape_item)
            self._shape_item = None
        self._remove_handles()
        return True

    def on_mouse_move(self, x: int, y: int, event) -> None:
        if self._resizing:
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self._apply_resize(self._resize_handle, x, y, shift)
            self._create_or_update_shape()
            self._update_handles()
            self._update_roi_info()
            return
        if self._moving:
            dx, dy = x - self._move_ox, y - self._move_oy
            self._shift_all(dx, dy)
            self._move_ox, self._move_oy = x, y
            self._create_or_update_shape()
            self._update_handles()
            self._update_roi_info()
            return
        if not self._drawing:
            return
        self._update_coords(x, y, event)
        self._create_or_update_shape()
        self._update_roi_info()

    def on_mouse_release(self, x: int, y: int, event) -> None:
        if self._resizing:
            self._resizing = False
            self._resize_handle = -1
            self._normalize_coords()
            self._create_or_update_shape()
            self._update_handles()
            self._update_roi_info()
            return
        if self._moving:
            self._moving = False
            self._update_handles()
            self._update_roi_info()
            return
        if not self._drawing:
            return
        self._update_coords(x, y, event)
        self._drawing = False
        self._has_roi = True
        self._create_or_update_shape()
        self._update_handles()
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
        self._update_handles()
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
        self._update_handles()
        self._update_roi_info()

    def deactivate(self) -> None:
        self._remove_handles()
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
        if not self._spins:
            return
        x0, y0, x1, y1 = self._roi_rect()
        for sp in self._spins.values():
            sp.blockSignals(True)
        self._spins["x"].setValue(x0)
        self._spins["y"].setValue(y0)
        self._spins["w"].setValue(x1 - x0)
        self._spins["h"].setValue(y1 - y0)
        for sp in self._spins.values():
            sp.blockSignals(False)

    def _on_spin_changed(self) -> None:
        if not self._has_roi:
            return
        x = self._spins["x"].value()
        y = self._spins["y"].value()
        w = self._spins["w"].value()
        h = self._spins["h"].value()
        self._x0, self._y0 = x, y
        self._x1, self._y1 = x + w, y + h
        self._anchor_x, self._anchor_y = x, y
        self._create_or_update_shape()
        self._update_handles()

    # -- 패널 --
    def build_panel(self) -> QWidget | None:
        self._panel = QWidget()
        layout = QVBoxLayout(self._panel)

        form = QFormLayout()
        for key in ("x", "y", "w", "h"):
            sp = QSpinBox()
            sp.setRange(-99999, 99999)
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(self._on_spin_changed)
            self._spins[key] = sp
            form.addRow(f"{key.upper()}:", sp)
        layout.addLayout(form)

        layout.addStretch()
        return self._panel
