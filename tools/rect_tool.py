"""Rectangle ROI Tool."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QGraphicsRectItem

from tools.roi_base import RoiToolBase
from viewer import ImageViewer


class RectTool(RoiToolBase):

    def __init__(self, viewer: ImageViewer) -> None:
        super().__init__(viewer)

    @property
    def name(self) -> str:
        return "Rectangle"

    def _create_or_update_shape(self) -> None:
        x0, y0, x1, y1 = self._roi_rect()
        w, h = x1 - x0, y1 - y0
        if self._shape_item is None:
            self._shape_item = self._viewer.scene_ref.addRect(
                x0, y0, w, h, self._pen()
            )
        else:
            self._shape_item.setRect(x0, y0, w, h)

    def _handle_positions(self) -> list[tuple[float, float]]:
        x0, y0, x1, y1 = self._roi_rect()
        return [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]

    def _apply_resize(self, handle: int, x: int, y: int, shift: bool) -> None:
        # TL=0, TR=1, BL=2, BR=3
        if handle == 0:
            self._x0, self._y0 = x, y
        elif handle == 1:
            self._x1, self._y0 = x, y
        elif handle == 2:
            self._x0, self._y1 = x, y
        else:
            self._x1, self._y1 = x, y

        if shift and self._resize_ratio is not None:
            x0, y0, x1, y1 = self._roi_rect()
            w, h = x1 - x0, y1 - y0
            if h == 0:
                return
            cur_ratio = w / h
            if cur_ratio > self._resize_ratio:
                new_w = int(round(h * self._resize_ratio))
                if handle in (0, 2):  # left side moves
                    self._x0 = self._x1 - new_w if self._x0 < self._x1 else self._x1 + new_w
                else:
                    self._x1 = self._x0 + new_w if self._x1 > self._x0 else self._x0 - new_w
            else:
                new_h = int(round(w / self._resize_ratio))
                if handle in (0, 1):  # top side moves
                    self._y0 = self._y1 - new_h if self._y0 < self._y1 else self._y1 + new_h
                else:
                    self._y1 = self._y0 + new_h if self._y1 > self._y0 else self._y0 - new_h

    def _make_mask(self, h: int, w: int) -> np.ndarray:
        x0, y0, x1, y1 = self._roi_rect()
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask
