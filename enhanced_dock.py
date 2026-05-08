"""EnhancedDockWidget: custom titlebar with float/maximize/close buttons."""

from __future__ import annotations

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWidget,
)

_BTN_STYLE = "font-size: 14px; padding: 0px 4px;"


class EnhancedDockWidget(QDockWidget):

    def __init__(self, title: str, parent=None, *, closable: bool = True):
        super().__init__(title, parent)
        self._closable = closable
        self._maximized = False
        self._build_titlebar(title)
        self.topLevelChanged.connect(self._on_top_level_changed)

    def _build_titlebar(self, title: str) -> None:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(self._title_label)
        layout.addStretch()

        self._btn_float = self._make_btn("↗", "Float")
        self._btn_float.clicked.connect(self._toggle_float)
        layout.addWidget(self._btn_float)

        self._btn_maximize = self._make_btn("□", "Maximize")
        self._btn_maximize.setVisible(False)
        self._btn_maximize.clicked.connect(self._toggle_maximize)
        layout.addWidget(self._btn_maximize)

        self._btn_close = self._make_btn("×", "Close")
        self._btn_close.clicked.connect(self.close)
        self._btn_close.setVisible(self._closable)
        layout.addWidget(self._btn_close)

        self.setTitleBarWidget(bar)

    @staticmethod
    def _make_btn(text: str, tooltip: str) -> QToolButton:
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setAutoRaise(True)
        btn.setStyleSheet(_BTN_STYLE)
        btn.setFixedSize(QSize(24, 20))
        return btn

    def setWindowTitle(self, title: str) -> None:
        super().setWindowTitle(title)
        if hasattr(self, "_title_label"):
            self._title_label.setText(title)

    def _toggle_float(self) -> None:
        if self.isFloating():
            self._maximized = False
            self.setFloating(False)
        else:
            self.setFloating(True)

    def _toggle_maximize(self) -> None:
        if self._maximized:
            self._maximized = False
            self.showNormal()
        else:
            self._maximized = True
            self.showMaximized()
        self._update_maximize_btn()

    def _on_top_level_changed(self, floating: bool) -> None:
        self._btn_maximize.setVisible(floating)
        if floating:
            self._btn_float.setText("↙")
            self._btn_float.setToolTip("Dock")
        else:
            self._maximized = False
            self._btn_float.setText("↗")
            self._btn_float.setToolTip("Float")
        self._update_maximize_btn()

    def _update_maximize_btn(self) -> None:
        if self._maximized:
            self._btn_maximize.setText("⧉")
            self._btn_maximize.setToolTip("Restore")
        else:
            self._btn_maximize.setText("□")
            self._btn_maximize.setToolTip("Maximize")
