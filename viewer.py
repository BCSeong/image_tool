"""ImageViewer: QGraphicsView 기반 이미지 뷰어. tool 이벤트 위임 포함."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QPen, QColor
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

if TYPE_CHECKING:
    from tool_base import BaseTool


def _normalize_to_u8(img: np.ndarray, mn: float, mx: float) -> np.ndarray:
    """min/max 범위를 0-255로 정규화. 정수형은 LUT, float는 vectorized."""
    span = mx - mn
    if span < 1e-12:
        return np.zeros(img.shape[:2], dtype=np.uint8)

    if img.dtype == np.uint8:
        lut = np.arange(256, dtype=np.float32)
        lut = ((lut - mn) / span * 255).clip(0, 255).astype(np.uint8)
        if img.ndim == 3:
            return cv2.LUT(img, lut)
        return cv2.LUT(img, lut)

    if img.dtype == np.uint16:
        lut = np.arange(65536, dtype=np.float32)
        lut = ((lut - mn) / span * 255).clip(0, 255).astype(np.uint8)
        if img.ndim == 2:
            return lut[img]
        return lut[img[:, :, 0]] if img.shape[2] == 1 else np.stack(
            [lut[img[:, :, c]] for c in range(img.shape[2])], axis=2
        )

    alpha = 255.0 / span
    beta = -mn * alpha
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


class ImageViewer(QGraphicsView):
    """줌/팬/좌표 추적을 지원하는 이미지 뷰어. 활성 tool에 마우스 이벤트 위임."""

    mouse_moved = Signal(int, int, object)  # (img_x, img_y, pixel_value)
    frame_scroll = Signal(int)  # +1 or -1 (스크롤로 프레임 전환)
    right_clicked = Signal(int, int, object)  # (img_x, img_y, QMouseEvent)
    escape_pressed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._raw_image: np.ndarray | None = None
        self._active_tool: BaseTool | None = None
        self._display_min: float | None = None
        self._display_max: float | None = None
        self._auto_min: float | None = None
        self._auto_max: float | None = None
        self._img_size: tuple[int, int] = (0, 0)
        self._panning = False
        self._pan_start = None

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setRenderHints(self.renderHints())
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    @property
    def raw_image(self) -> np.ndarray | None:
        return self._raw_image

    def set_active_tool(self, tool: BaseTool | None) -> None:
        self._active_tool = tool
        if tool is not None:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def set_image(self, img: np.ndarray) -> None:
        """numpy 배열을 QPixmap으로 변환하여 표시. 8/16bit gray, BGR 지원."""
        self._raw_image = img
        self._auto_min = None
        self._auto_max = None
        display = self._to_display(img)
        h, w = display.shape[:2]

        if display.ndim == 2:
            qimg = QImage(display.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            qimg = QImage(display.data, w, h, 3 * w, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pixmap)
        else:
            self._pixmap_item.setPixmap(pixmap)

        new_size = (w, h)
        if new_size != self._img_size:
            self._img_size = new_size
            self._scene.setSceneRect(self._pixmap_item.boundingRect())

    def set_display_range(self, min_val: float, max_val: float) -> None:
        self._display_min = min_val
        self._display_max = max_val
        self.refresh_display()

    def get_display_range(self) -> tuple[float, float] | None:
        mn = self._display_min if self._display_min is not None else self._auto_min
        mx = self._display_max if self._display_max is not None else self._auto_max
        if mn is not None and mx is not None:
            return mn, mx
        return None

    def clear_display_range(self) -> None:
        self._display_min = None
        self._display_max = None
        self._auto_min = None
        self._auto_max = None
        self.refresh_display()

    def refresh_display(self) -> None:
        if self._raw_image is not None:
            display = self._to_display(self._raw_image)
            h, w = display.shape[:2]
            if display.ndim == 2:
                qimg = QImage(display.data, w, h, w, QImage.Format.Format_Grayscale8)
            else:
                qimg = QImage(display.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            if self._pixmap_item is not None:
                self._pixmap_item.setPixmap(pixmap)

    def fit_view(self) -> None:
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def clear_overlays(self) -> None:
        """pixmap 이외의 모든 scene 아이템 제거."""
        for item in self._scene.items():
            if item is not self._pixmap_item:
                self._scene.removeItem(item)

    @property
    def scene_ref(self) -> QGraphicsScene:
        return self._scene

    # -- 내부 변환 --
    def _to_display(self, img: np.ndarray) -> np.ndarray:
        """표시용 8bit 이미지 변환. LUT 기반 고속 처리."""
        mn = self._display_min
        mx = self._display_max

        if mn is not None and mx is not None:
            normalized = _normalize_to_u8(img, mn, mx)
            if normalized.ndim == 3 and normalized.shape[2] == 3:
                return cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
            return normalized

        if img.dtype == np.uint8:
            if img.ndim == 3 and img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        if img.dtype in (np.uint16, np.float32, np.float64):
            auto_mn = getattr(self, "_auto_min", None)
            auto_mx = getattr(self, "_auto_max", None)
            if auto_mn is None or auto_mx is None:
                auto_mn = float(np.nanmin(img))
                auto_mx = float(np.nanmax(img))
                self._auto_min = auto_mn
                self._auto_max = auto_mx
            normalized = _normalize_to_u8(img, auto_mn, auto_mx)
            if normalized.ndim == 3 and normalized.shape[2] == 3:
                return cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
            return normalized

        return img.astype(np.uint8) if img.ndim == 2 else img

    def _scene_to_image(self, scene_pos) -> tuple[int, int] | None:
        """scene 좌표를 이미지 픽셀 좌표로 변환. 범위 밖이면 None."""
        if self._pixmap_item is None:
            return None
        ix = int(scene_pos.x())
        iy = int(scene_pos.y())
        rect = self._pixmap_item.boundingRect()
        if 0 <= ix < rect.width() and 0 <= iy < rect.height():
            return ix, iy
        return None

    def _pixel_value_at(self, x: int, y: int):
        """(x, y) 위치의 raw 픽셀 값 반환."""
        if self._raw_image is None:
            return None
        h, w = self._raw_image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return self._raw_image[y, x]
        return None

    # -- 마우스 이벤트 --
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            pos = self._scene_to_image(self.mapToScene(event.pos()))
            if pos:
                self.right_clicked.emit(pos[0], pos[1], event)
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._active_tool:
            pos = self._scene_to_image(self.mapToScene(event.pos()))
            if pos:
                consumed = self._active_tool.on_mouse_press(pos[0], pos[1], event)
                if consumed:
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            return
        scene_pos = self.mapToScene(event.pos())
        pos = self._scene_to_image(scene_pos)
        if pos:
            val = self._pixel_value_at(pos[0], pos[1])
            self.mouse_moved.emit(pos[0], pos[1], val)
            if self._active_tool:
                self._active_tool.on_mouse_move(pos[0], pos[1], event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.viewport().unsetCursor()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._active_tool:
            pos = self._scene_to_image(self.mapToScene(event.pos()))
            if pos:
                self._active_tool.on_mouse_release(pos[0], pos[1], event)
        super().mouseReleaseEvent(event)

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

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.escape_pressed.emit()
            return
        if self._active_tool and self._active_tool.on_key_press(key, event):
            return
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.scale(1.25, 1.25)
        elif key == Qt.Key.Key_Minus:
            self.scale(0.8, 0.8)
        else:
            super().keyPressEvent(event)
