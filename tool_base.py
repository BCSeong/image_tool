"""BaseTool: 모든 분석 tool의 추상 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PySide6.QtWidgets import QWidget


class BaseTool(ABC):
    """tool 플러그인이 구현해야 할 인터페이스."""

    @property
    @abstractmethod
    def name(self) -> str:
        """툴바에 표시할 이름."""

    @property
    def icon(self) -> str | None:
        """아이콘 경로 또는 None (텍스트 버튼)."""
        return None

    def activate(self) -> None:
        """tool이 선택될 때 호출."""

    def deactivate(self) -> None:
        """다른 tool로 전환될 때 호출."""

    def on_mouse_press(self, x: int, y: int, event) -> bool:
        """이미지 좌표에서 마우스 클릭. True 반환 시 이벤트 소비."""
        return False

    def on_mouse_move(self, x: int, y: int, event) -> None:
        """이미지 좌표에서 마우스 이동."""

    def on_mouse_release(self, x: int, y: int, event) -> None:
        """이미지 좌표에서 마우스 릴리즈."""

    def on_key_press(self, key: int, event) -> bool:
        """키보드 입력. True 반환 시 이벤트 소비."""
        return False

    def on_frame_changed(self, idx: int, image: np.ndarray | None) -> None:
        """프레임이 변경될 때 호출."""

    def build_panel(self) -> QWidget | None:
        """우측 도킹 영역에 표시할 위젯. None이면 패널 없음."""
        return None
