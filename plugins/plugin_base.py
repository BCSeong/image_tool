"""PluginBase: 모든 plugin의 추상 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from image_source import ImageSource
    from PySide6.QtWidgets import QMainWindow


class PluginBase(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugins 메뉴에 표시할 이름."""

    @abstractmethod
    def run(self, source: "ImageSource", frame_idx: int, parent: "QMainWindow") -> None:
        """Plugin 실행."""
