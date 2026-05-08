"""SaveSequenceDialog: 프레임별 개별 이미지 저장 다이얼로그."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

_FORMATS = {
    "TIFF": ".tif",
    "PNG": ".png",
    "BMP": ".bmp",
}


class SaveSequenceDialog(QDialog):

    def __init__(
        self,
        frame_count: int,
        has_frame_names: bool,
        sample_frame_names: list[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save as Sequence")
        self.setMinimumWidth(420)

        self._count = frame_count
        self._sample_names = sample_frame_names or []

        layout = QVBoxLayout(self)

        # -- Directory --
        row_dir = QHBoxLayout()
        row_dir.addWidget(QLabel("Directory:"))
        self._dir_edit = QLineEdit()
        row_dir.addWidget(self._dir_edit, stretch=1)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._browse)
        row_dir.addWidget(btn_browse)
        layout.addLayout(row_dir)

        # -- Format --
        row_fmt = QHBoxLayout()
        row_fmt.addWidget(QLabel("Format:"))
        self._combo_fmt = QComboBox()
        self._combo_fmt.addItems(list(_FORMATS.keys()))
        row_fmt.addWidget(self._combo_fmt)
        row_fmt.addStretch()
        layout.addLayout(row_fmt)

        # -- Use frame names --
        self._chk_frame_names = QCheckBox("Use original frame names")
        self._chk_frame_names.setEnabled(has_frame_names)
        layout.addWidget(self._chk_frame_names)

        # -- Naming controls --
        form = QVBoxLayout()

        def add_row(label, widget):
            r = QHBoxLayout()
            r.addWidget(QLabel(label))
            r.addWidget(widget, stretch=1)
            form.addLayout(r)

        self._prefix = QLineEdit("image")
        add_row("Prefix:", self._prefix)

        self._postfix = QLineEdit("")
        add_row("Postfix:", self._postfix)

        self._start_idx = QSpinBox()
        self._start_idx.setRange(0, 99999)
        self._start_idx.setValue(0)
        add_row("Start index:", self._start_idx)

        self._zero_pad = QSpinBox()
        self._zero_pad.setRange(1, 6)
        self._zero_pad.setValue(3)
        add_row("Zero padding:", self._zero_pad)

        layout.addLayout(form)

        self._naming_widgets = [
            self._prefix, self._postfix, self._start_idx, self._zero_pad,
        ]

        # -- Preview --
        self._preview = QLabel()
        self._preview.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self._preview)

        # -- Buttons --
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_save = QPushButton("Save")
        btn_save.setDefault(True)
        btn_save.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        layout.addLayout(btn_row)

        # -- Signals --
        self._chk_frame_names.toggled.connect(self._on_toggle_frame_names)
        self._combo_fmt.currentIndexChanged.connect(self._refresh_preview)
        self._prefix.textChanged.connect(self._refresh_preview)
        self._postfix.textChanged.connect(self._refresh_preview)
        self._start_idx.valueChanged.connect(self._refresh_preview)
        self._zero_pad.valueChanged.connect(self._refresh_preview)
        self._refresh_preview()

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            self._dir_edit.setText(d)

    def _on_toggle_frame_names(self, checked: bool) -> None:
        for w in self._naming_widgets:
            w.setEnabled(not checked)
        self._refresh_preview()

    def _ext(self) -> str:
        return _FORMATS[self._combo_fmt.currentText()]

    def _make_custom_name(self, idx: int) -> str:
        pad = self._zero_pad.value()
        pre = self._prefix.text()
        post = self._postfix.text()
        num = f"{idx:0{pad}d}"
        sep = "_" if pre else ""
        return f"{pre}{sep}{num}{post}{self._ext()}"

    def _frame_name_with_ext(self, name: str) -> str:
        p = Path(name)
        return p.stem + self._ext()

    def _refresh_preview(self) -> None:
        if self._chk_frame_names.isChecked() and self._sample_names:
            first = self._frame_name_with_ext(self._sample_names[0])
            if len(self._sample_names) > 1:
                last = self._frame_name_with_ext(self._sample_names[-1])
                self._preview.setText(f"Preview: {first} … {last}")
            else:
                self._preview.setText(f"Preview: {first}")
        else:
            start = self._start_idx.value()
            first = self._make_custom_name(start)
            if self._count > 1:
                last = self._make_custom_name(start + self._count - 1)
                self._preview.setText(f"Preview: {first} … {last}")
            else:
                self._preview.setText(f"Preview: {first}")

    def get_config(self) -> dict | None:
        d = self._dir_edit.text().strip()
        if not d:
            return None
        return {
            "directory": Path(d),
            "format": self._combo_fmt.currentText(),
            "use_frame_names": self._chk_frame_names.isChecked(),
            "prefix": self._prefix.text(),
            "postfix": self._postfix.text(),
            "start": self._start_idx.value(),
            "pad": self._zero_pad.value(),
        }

    def make_path(self, cfg: dict, idx: int, frame_name: str = "") -> Path:
        if cfg["use_frame_names"] and frame_name:
            return cfg["directory"] / self._frame_name_with_ext(frame_name)
        num_idx = cfg["start"] + idx
        return cfg["directory"] / self._make_custom_name(num_idx)
