"""Ellipse ROI Tool."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtWidgets import QGraphicsEllipseItem

from tools.roi_base import RoiToolBase
from viewer import ImageViewer


class EllipseTool(RoiToolBase):

    def __init__(self, viewer: ImageViewer) -> None:
        super().__init__(viewer)

    @property
    def name(self) -> str:
        return "Ellipse"

    def _hit_test(self, x: int, y: int) -> bool:
        x0, y0, x1, y1 = self._roi_rect()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
        if rx < 1 or ry < 1:
            return False
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1

    def _create_or_update_shape(self) -> None:
        x0, y0, x1, y1 = self._roi_rect()
        w, h = x1 - x0, y1 - y0
        if self._shape_item is None:
            self._shape_item = self._viewer.scene_ref.addEllipse(
                x0, y0, w, h, self._pen()
            )
        else:
            self._shape_item.setRect(x0, y0, w, h)

    def _make_mask(self, h: int, w: int) -> np.ndarray:
        x0, y0, x1, y1 = self._roi_rect()
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        rx = (x1 - x0) / 2
        ry = (y1 - y0) / 2
        if rx < 1 or ry < 1:
            return np.zeros((h, w), dtype=bool)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)),
                    0, 0, 360, 255, -1)
        return mask > 0
