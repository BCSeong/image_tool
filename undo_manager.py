"""UndoManager: 프레임 수정 액션에 대한 실행취소 스택."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from image_source import ImageSource


class UndoManager:

    def __init__(self, source: ImageSource, max_stack: int = 50) -> None:
        self._source = source
        self._max_stack = max_stack
        self._stack: list[tuple[int, np.ndarray, str]] = []

    def push(self, frame_idx: int, old_img: np.ndarray, description: str) -> None:
        self._stack.append((frame_idx, old_img.copy(), description))
        if len(self._stack) > self._max_stack:
            self._stack.pop(0)

    def undo(self) -> tuple[int, np.ndarray] | None:
        if not self._stack:
            return None
        frame_idx, img, _ = self._stack.pop()
        self._source.set_frame(frame_idx, img)
        return frame_idx, img

    def can_undo(self) -> bool:
        return len(self._stack) > 0

    def last_description(self) -> str:
        if self._stack:
            return self._stack[-1][2]
        return ""

    def clear(self) -> None:
        self._stack.clear()
