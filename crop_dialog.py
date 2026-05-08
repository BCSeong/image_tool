"""CropDialog: ROI crop 설정 다이얼로그."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


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
        self.setMinimumWidth(320)

        x0, y0, x1, y1 = roi
        rw, rh = x1 - x0, y1 - y0

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"ROI: ({x0}, {y0})  {rw} × {rh}"))

        self._chk_all = QCheckBox("Crop all frames")
        self._chk_all.setChecked(True)
        layout.addWidget(self._chk_all)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("From:"))
        self._spin_start = QSpinBox()
        self._spin_start.setRange(0, max(frame_count - 1, 0))
        self._spin_start.setValue(0)
        range_row.addWidget(self._spin_start)
        range_row.addWidget(QLabel("To:"))
        self._spin_end = QSpinBox()
        self._spin_end.setRange(0, max(frame_count - 1, 0))
        self._spin_end.setValue(max(frame_count - 1, 0))
        range_row.addWidget(self._spin_end)
        self._range_label = QLabel(f"({frame_count} frames)")
        range_row.addWidget(self._range_label)
        layout.addLayout(range_row)

        self._current_frame = current_frame
        self._frame_count = frame_count

        self._chk_all.toggled.connect(self._on_toggle)
        self._spin_start.valueChanged.connect(self._on_range_changed)
        self._spin_end.valueChanged.connect(self._on_range_changed)
        self._on_toggle(self._chk_all.isChecked())

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Crop")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

    def _on_toggle(self, checked: bool) -> None:
        self._spin_start.setEnabled(checked)
        self._spin_end.setEnabled(checked)
        if checked:
            self._on_range_changed()
        else:
            self._range_label.setText("(current frame only)")

    def _on_range_changed(self) -> None:
        s = self._spin_start.value()
        e = self._spin_end.value()
        if e < s:
            self._spin_end.setValue(s)
            e = s
        n = e - s + 1
        self._range_label.setText(f"({n} frames)")

    def get_config(self) -> dict:
        if self._chk_all.isChecked():
            return {
                "mode": "range",
                "start": self._spin_start.value(),
                "end": self._spin_end.value(),
            }
        return {
            "mode": "current",
            "frame": self._current_frame,
        }
