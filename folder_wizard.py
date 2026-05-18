"""FolderWizard: 폴더 로딩 설정 다이얼로그.

파일 필터(확장자/텍스트 패턴), 정렬, 범위 지정, 매칭 미리보기 제공.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)


@dataclass
class FolderLoadConfig:
    """마법사 결과."""
    folder: Path
    extensions: list[str]
    text_filter: str
    sort_by: str  # "name" | "natural" | "date"
    start: int  # 0-based
    end: int  # inclusive, -1 = 마지막까지
    step: int = 1
    matched_paths: list[Path] = field(default_factory=list)


_ALL_EXTENSIONS = [".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"]


def _natural_sort_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', p.name)]


class FolderWizard(QDialog):
    """폴더 로딩 마법사 다이얼로그."""

    def __init__(self, parent=None, initial_folder: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Open Folder")
        self.setMinimumSize(700, 560)
        self._matched: list[Path] = []
        self._build_ui()
        self._connect()
        if initial_folder:
            self._folder_edit.setText(initial_folder)
            self._refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # --- 폴더 선택 ---
        grp_folder = QGroupBox("Folder")
        hf = QHBoxLayout(grp_folder)
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Select folder or type path...")
        self._btn_browse = QPushButton("Browse...")
        hf.addWidget(self._folder_edit, stretch=1)
        hf.addWidget(self._btn_browse)
        root.addWidget(grp_folder)

        # --- 필터 ---
        grp_filter = QGroupBox("Filter")
        vf = QVBoxLayout(grp_filter)

        # 확장자 체크박스
        ext_row = QHBoxLayout()
        ext_row.addWidget(QLabel("Extensions:"))
        self._ext_checks: dict[str, QCheckBox] = {}
        for ext in _ALL_EXTENSIONS:
            cb = QCheckBox(ext)
            cb.setChecked(True)
            self._ext_checks[ext] = cb
            ext_row.addWidget(cb)
        ext_row.addStretch()
        vf.addLayout(ext_row)

        # 텍스트 필터
        text_row = QHBoxLayout()
        text_row.addWidget(QLabel("Name contains:"))
        self._text_filter = QLineEdit()
        self._text_filter.setPlaceholderText("(optional) e.g. frame, GD*002 (* = wildcard)")
        text_row.addWidget(self._text_filter, stretch=1)
        vf.addLayout(text_row)

        # 정렬
        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort by:"))
        self._sort_group = QButtonGroup(self)
        self._rb_name = QRadioButton("Name")
        self._rb_natural = QRadioButton("Name (natural)")
        self._rb_date = QRadioButton("Date modified")
        self._rb_name.setChecked(True)
        self._sort_group.addButton(self._rb_name, 0)
        self._sort_group.addButton(self._rb_natural, 1)
        self._sort_group.addButton(self._rb_date, 2)
        sort_row.addWidget(self._rb_name)
        sort_row.addWidget(self._rb_natural)
        sort_row.addWidget(self._rb_date)
        sort_row.addStretch()
        vf.addLayout(sort_row)

        root.addWidget(grp_filter)

        # --- 범위 ---
        grp_range = QGroupBox("Frame Range")
        hr = QHBoxLayout(grp_range)
        hr.addWidget(QLabel("Start:"))
        self._start_spin = QSpinBox()
        self._start_spin.setMinimum(0)
        self._start_spin.setMaximum(0)
        hr.addWidget(self._start_spin)
        hr.addWidget(QLabel("End:"))
        self._end_spin = QSpinBox()
        self._end_spin.setMinimum(0)
        self._end_spin.setMaximum(0)
        hr.addWidget(self._end_spin)
        hr.addWidget(QLabel("Step:"))
        self._step_spin = QSpinBox()
        self._step_spin.setMinimum(1)
        self._step_spin.setMaximum(9999)
        self._step_spin.setValue(1)
        hr.addWidget(self._step_spin)
        self._range_label = QLabel("(0 files)")
        hr.addWidget(self._range_label)
        hr.addStretch()
        root.addWidget(grp_range)

        # --- 미리보기 ---
        grp_preview = QGroupBox("Preview")
        vp = QVBoxLayout(grp_preview)

        self._preview_check = QCheckBox("Show image thumbnail")
        self._preview_check.setChecked(False)
        vp.addWidget(self._preview_check)

        hp = QHBoxLayout()
        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(180)
        hp.addWidget(self._file_list, stretch=1)

        self._thumbnail = QLabel()
        self._thumbnail.setFixedSize(180, 180)
        self._thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumbnail.setStyleSheet("border: 1px solid #ccc; background: #222;")
        self._thumbnail.hide()
        hp.addWidget(self._thumbnail)

        vp.addLayout(hp)
        root.addWidget(grp_preview)

        # --- 버튼 ---
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        root.addWidget(self._buttons)

    def _connect(self) -> None:
        self._btn_browse.clicked.connect(self._browse)
        self._folder_edit.editingFinished.connect(self._refresh)
        self._text_filter.textChanged.connect(self._refresh)
        self._sort_group.idClicked.connect(lambda _: self._refresh())
        for cb in self._ext_checks.values():
            cb.stateChanged.connect(self._refresh)
        self._start_spin.valueChanged.connect(self._on_range_changed)
        self._end_spin.valueChanged.connect(self._on_range_changed)
        self._step_spin.valueChanged.connect(self._on_range_changed)
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        self._preview_check.toggled.connect(self._on_preview_toggled)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

    # ------------------------------------------------------------------ actions
    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if folder:
            self._folder_edit.setText(folder)
            self._refresh()

    def _refresh(self) -> None:
        folder = self._folder_edit.text()
        if not folder or not Path(folder).is_dir():
            self._matched = []
            self._update_preview()
            return

        exts = {ext for ext, cb in self._ext_checks.items() if cb.isChecked()}
        text = self._text_filter.text().strip().lower()
        sort_idx = self._sort_group.checkedId()

        root = Path(folder)
        all_files = [
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]
        if text:
            if '*' in text or '?' in text:
                pattern = f"*{text}*" if not text.startswith('*') else text
                pattern = f"{pattern}*" if not pattern.endswith('*') else pattern
                all_files = [p for p in all_files
                             if fnmatch.fnmatch(p.stem.lower(), pattern)]
            else:
                all_files = [p for p in all_files if text in p.stem.lower()]

        if sort_idx == 0:
            all_files.sort(key=lambda p: p.name)
        elif sort_idx == 1:
            all_files.sort(key=_natural_sort_key)
        else:
            all_files.sort(key=lambda p: p.stat().st_mtime)

        self._matched = all_files
        n = len(all_files)

        self._start_spin.setMaximum(max(n - 1, 0))
        self._end_spin.setMaximum(max(n - 1, 0))
        self._start_spin.setValue(0)
        self._end_spin.setValue(max(n - 1, 0))

        self._update_preview()

    def _on_range_changed(self) -> None:
        s = self._start_spin.value()
        e = self._end_spin.value()
        if e < s:
            self._end_spin.setValue(s)
        self._update_preview()

    def _update_preview(self) -> None:
        s = self._start_spin.value()
        e = self._end_spin.value()
        step = self._step_spin.value()
        subset = self._matched[s:e + 1:step] if self._matched else []

        self._file_list.clear()
        for p in subset:
            self._file_list.addItem(p.name)
        n = len(subset)
        self._range_label.setText(f"({n} files)")
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(n > 0)

        if n > 0:
            self._file_list.setCurrentRow(0)
        else:
            self._thumbnail.clear()
            self._thumbnail.setText("No match")

    def _on_preview_toggled(self, checked: bool) -> None:
        if checked:
            self._thumbnail.show()
            self._on_file_selected(self._file_list.currentRow())
        else:
            self._thumbnail.hide()

    def _on_file_selected(self, row: int) -> None:
        if not self._preview_check.isChecked():
            return
        s = self._start_spin.value()
        e = self._end_spin.value()
        step = self._step_spin.value()
        subset = self._matched[s:e + 1:step]
        if row < 0 or row >= len(subset):
            self._thumbnail.clear()
            return
        path = subset[row]
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            self._thumbnail.setText("Cannot read")
            return
        self._show_thumbnail(img)

    def _show_thumbnail(self, img: np.ndarray) -> None:
        if img.dtype != np.uint8:
            fimg = img.astype(np.float64)
            mn, mx = np.nanmin(fimg), np.nanmax(fimg)
            if mx - mn < 1e-12:
                display = np.zeros(img.shape[:2], dtype=np.uint8)
            else:
                display = ((fimg - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)
        else:
            display = img

        if display.ndim == 3 and display.shape[2] == 3:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            h, w, ch = display.shape
            qimg = QImage(display.data, w, h, ch * w, QImage.Format.Format_RGB888)
        elif display.ndim == 2:
            h, w = display.shape
            qimg = QImage(display.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            self._thumbnail.setText("Unsupported")
            return

        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            QSize(176, 176),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumbnail.setPixmap(scaled)

    # ------------------------------------------------------------------ result
    def get_config(self) -> FolderLoadConfig | None:
        if not self._matched:
            return None
        s = self._start_spin.value()
        e = self._end_spin.value()
        step = self._step_spin.value()
        subset = self._matched[s:e + 1:step]
        sort_map = {0: "name", 1: "natural", 2: "date"}
        return FolderLoadConfig(
            folder=Path(self._folder_edit.text()),
            extensions=[ext for ext, cb in self._ext_checks.items() if cb.isChecked()],
            text_filter=self._text_filter.text().strip(),
            sort_by=sort_map.get(self._sort_group.checkedId(), "name"),
            start=s,
            end=e,
            step=step,
            matched_paths=subset,
        )
