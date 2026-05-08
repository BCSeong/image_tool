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

    def _make_mask(self, h: int, w: int) -> np.ndarray:
        x0, y0, x1, y1 = self._roi_rect()
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask
