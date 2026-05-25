"""CropDialog: ROI crop 설정 다이얼로그."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)


def parse_indices(text: str, max_idx: int) -> list[int] | None:
    """쉼표/공백 구분 인덱스 및 범위(start-end) 파싱. 유효하지 않으면 None."""
    indices: list[int] = []
    for token in text.replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        if "-" in token and not token.startswith("-"):
            parts = token.split("-", 1)
            try:
                a, b = int(parts[0]), int(parts[1])
            except ValueError:
                return None
            if a > b or a < 0 or b > max_idx:
                return None
            indices.extend(range(a, b + 1))
        else:
            try:
                v = int(token)
            except ValueError:
                return None
            if v < 0 or v > max_idx:
                return None
            indices.append(v)
    return indices if indices else None


class CropDialog(QDialog):

    def __init__(
        self,
        roi: tuple[int, int, int, int],
        current_frame: int,
        frame_count: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Crop")
        self.setMinimumWidth(360)

        x0, y0, x1, y1 = roi
        rw, rh = x1 - x0, y1 - y0

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"ROI: ({x0}, {y0})  {rw} × {rh}"))

        self._current_frame = current_frame
        self._frame_count = frame_count

        # -- Frame selection mode --
        self._radio_current = QRadioButton("Current frame only")
        self._radio_range = QRadioButton("Frame range")
        self._radio_custom = QRadioButton("Custom indices")
        self._radio_range.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._radio_current, 0)
        self._mode_group.addButton(self._radio_range, 1)
        self._mode_group.addButton(self._radio_custom, 2)
        layout.addWidget(self._radio_current)

        # Range row
        range_row = QHBoxLayout()
        range_row.addWidget(self._radio_range)
        self._spin_start = QSpinBox()
        self._spin_start.setRange(0, max(frame_count - 1, 0))
        self._spin_start.setValue(0)
        range_row.addWidget(self._spin_start)
        range_row.addWidget(QLabel("to"))
        self._spin_end = QSpinBox()
        self._spin_end.setRange(0, max(frame_count - 1, 0))
        self._spin_end.setValue(max(frame_count - 1, 0))
        range_row.addWidget(self._spin_end)
        self._range_label = QLabel(f"({frame_count} frames)")
        range_row.addWidget(self._range_label)
        layout.addLayout(range_row)

        # Custom row
        custom_row = QHBoxLayout()
        custom_row.addWidget(self._radio_custom)
        self._edit_indices = QLineEdit()
        self._edit_indices.setPlaceholderText("e.g. 0,1,3,5-8,10")
        custom_row.addWidget(self._edit_indices, stretch=1)
        self._custom_label = QLabel("")
        custom_row.addWidget(self._custom_label)
        layout.addLayout(custom_row)

        # -- Signals --
        self._mode_group.idToggled.connect(self._on_mode_changed)
        self._spin_start.valueChanged.connect(self._on_range_changed)
        self._spin_end.valueChanged.connect(self._on_range_changed)
        self._edit_indices.textChanged.connect(self._on_custom_changed)
        self._on_mode_changed(1, True)

        # -- Buttons --
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        self._btn_ok = QPushButton("Crop")
        self._btn_ok.setDefault(True)
        self._btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(self._btn_ok)
        layout.addLayout(btn_row)

    def _on_mode_changed(self, id_: int, checked: bool) -> None:
        if not checked:
            return
        self._spin_start.setEnabled(id_ == 1)
        self._spin_end.setEnabled(id_ == 1)
        self._edit_indices.setEnabled(id_ == 2)
        if id_ == 0:
            self._range_label.setText("")
            self._custom_label.setText("")
        elif id_ == 1:
            self._on_range_changed()
            self._custom_label.setText("")
        else:
            self._range_label.setText("")
            self._on_custom_changed()

    def _on_range_changed(self) -> None:
        s = self._spin_start.value()
        e = self._spin_end.value()
        if e < s:
            self._spin_end.setValue(s)
            e = s
        n = e - s + 1
        self._range_label.setText(f"({n} frames)")

    def _on_custom_changed(self) -> None:
        text = self._edit_indices.text().strip()
        if not text:
            self._custom_label.setText("")
            return
        indices = parse_indices(text, self._frame_count - 1)
        if indices is None:
            self._custom_label.setText("(invalid)")
            self._custom_label.setStyleSheet("color: red;")
        else:
            self._custom_label.setText(f"({len(indices)} frames)")
            self._custom_label.setStyleSheet("")

    def get_config(self) -> dict:
        mode_id = self._mode_group.checkedId()
        if mode_id == 0:
            return {
                "mode": "current",
                "frame": self._current_frame,
            }
        if mode_id == 2:
            indices = parse_indices(
                self._edit_indices.text(), self._frame_count - 1
            )
            if indices:
                return {
                    "mode": "custom",
                    "indices": indices,
                }
            return {
                "mode": "current",
                "frame": self._current_frame,
            }
        return {
            "mode": "range",
            "start": self._spin_start.value(),
            "end": self._spin_end.value(),
        }
