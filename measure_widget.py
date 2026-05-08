"""Measure 위젯: Ctrl+M으로 ROI 통계를 테이블에 누적."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import median_filter


# ── 측정 속성 정의 ──────────────────────────────────────────
AVAILABLE_PROPS = [
    ("Mean", True),
    ("StdDev", True),
    ("Min", True),
    ("Max", True),
    ("Median", False),
    ("Area", True),
    ("RobustStdDev", False),
]

FIXED_COLUMNS = ["#", "Frame", "Tool", "ROI"]


# ── RobustStdDev 계산 함수 ──────────────────────────────────
def _detrend_2d(roi: np.ndarray, order: int = 2) -> np.ndarray:
    ny, nx = roi.shape
    yy, xx = np.mgrid[:ny, :nx]
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    z = roi.ravel()
    terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            terms.append(coords[:, 0] ** i * coords[:, 1] ** j)
    A = np.column_stack(terms)
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    surface = (A @ coeffs).reshape(ny, nx)
    return roi - surface


def _robust_std(roi_2d: np.ndarray, order: int, mad_n: float,
                use_median: bool, median_ks: int) -> float:
    if use_median:
        roi_2d = median_filter(roi_2d, size=median_ks)
    detrended = _detrend_2d(roi_2d, order)
    flat = detrended.ravel()
    med_val = np.median(flat)
    mad_val = np.median(np.abs(flat - med_val))
    if mad_val < 1e-12:
        mad_val = 1e-12
    inlier = np.abs(flat - med_val) <= mad_n * mad_val
    if np.count_nonzero(inlier) == 0:
        return 0.0
    return float(np.std(flat[inlier]))


# ── 단일 채널 측정 ──────────────────────────────────────────
def _compute_channel(vals: np.ndarray, area: int, enabled: set,
                     roi_2d: np.ndarray | None, robust_cfg: dict) -> dict:
    result = {}
    if "Mean" in enabled:
        result["Mean"] = float(np.mean(vals))
    if "StdDev" in enabled:
        result["StdDev"] = float(np.std(vals))
    if "Min" in enabled:
        result["Min"] = float(np.min(vals))
    if "Max" in enabled:
        result["Max"] = float(np.max(vals))
    if "Median" in enabled:
        result["Median"] = float(np.median(vals))
    if "Area" in enabled:
        result["Area"] = area
    if "RobustStdDev" in enabled and roi_2d is not None:
        result["RobustStdDev"] = _robust_std(
            roi_2d.astype(np.float64),
            robust_cfg["order"], robust_cfg["mad_n"],
            robust_cfg["use_median"], robust_cfg["median_ks"],
        )
    return result


# ── 설정 다이얼로그 ─────────────────────────────────────────
class MeasureConfigDialog(QDialog):
    def __init__(self, enabled: set, robust_cfg: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Measure Configuration")
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        self._checks: dict[str, QCheckBox] = {}
        for i, (name, _) in enumerate(AVAILABLE_PROPS):
            chk = QCheckBox(name)
            chk.setChecked(name in enabled)
            self._checks[name] = chk
            grid.addWidget(chk, i // 4, i % 4)
        layout.addLayout(grid)

        grp = QGroupBox("RobustStdDev Parameters")
        gv = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Detrend order:"))
        self._spin_order = QSpinBox()
        self._spin_order.setRange(0, 4)
        self._spin_order.setValue(robust_cfg.get("order", 2))
        row1.addWidget(self._spin_order)
        gv.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("MAD multiplier:"))
        self._spin_mad = QDoubleSpinBox()
        self._spin_mad.setRange(0.5, 10.0)
        self._spin_mad.setDecimals(1)
        self._spin_mad.setSingleStep(0.5)
        self._spin_mad.setValue(robust_cfg.get("mad_n", 3.0))
        row2.addWidget(self._spin_mad)
        gv.addLayout(row2)

        row3 = QHBoxLayout()
        self._chk_med = QCheckBox("Median filter")
        self._chk_med.setChecked(robust_cfg.get("use_median", False))
        row3.addWidget(self._chk_med)
        row3.addWidget(QLabel("Kernel:"))
        self._spin_ks = QSpinBox()
        self._spin_ks.setRange(3, 15)
        self._spin_ks.setSingleStep(2)
        self._spin_ks.setValue(robust_cfg.get("median_ks", 3))
        row3.addWidget(self._spin_ks)
        gv.addLayout(row3)

        layout.addWidget(grp)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("OK")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

    def get_enabled(self) -> set:
        return {name for name, chk in self._checks.items() if chk.isChecked()}

    def get_robust_cfg(self) -> dict:
        return {
            "order": self._spin_order.value(),
            "mad_n": self._spin_mad.value(),
            "use_median": self._chk_med.isChecked(),
            "median_ks": self._spin_ks.value(),
        }


# ── Measure 위젯 ────────────────────────────────────────────
class MeasureWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._enabled: set = {name for name, default in AVAILABLE_PROPS if default}
        self._robust_cfg: dict = {
            "order": 2, "mad_n": 3.0, "use_median": False, "median_ks": 3,
        }
        self._row_count = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        btn_row = QHBoxLayout()
        btn_cfg = QPushButton("Configure")
        btn_cfg.clicked.connect(self._on_configure)
        btn_row.addWidget(btn_cfg)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._on_clear)
        btn_row.addWidget(btn_clear)
        btn_csv = QPushButton("Export CSV")
        btn_csv.clicked.connect(self._on_export_csv)
        btn_row.addWidget(btn_csv)
        btn_copy = QPushButton("Copy")
        btn_copy.setToolTip("Copy table to clipboard")
        btn_copy.clicked.connect(self._on_copy_clipboard)
        btn_row.addWidget(btn_copy)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._table = QTableWidget(0, 0)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, stretch=1)

        self._rebuild_columns()

    def _column_names(self) -> list[str]:
        cols = list(FIXED_COLUMNS)
        for name, _ in AVAILABLE_PROPS:
            if name in self._enabled:
                cols.append(name)
        return cols

    def _rebuild_columns(self) -> None:
        cols = self._column_names()
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        header = self._table.horizontalHeader()
        for ci in range(min(len(cols), len(FIXED_COLUMNS))):
            header.setSectionResizeMode(ci, QHeaderView.ResizeMode.ResizeToContents)

    def add_measurement(self, tool_name: str,
                        roi_rect: tuple[int, int, int, int],
                        mask: np.ndarray, image: np.ndarray,
                        frame_name: str = "") -> None:
        x0, y0, x1, y1 = roi_rect
        area = int(np.count_nonzero(mask))
        if area == 0:
            return

        is_rgb = image.ndim == 3 and image.shape[2] == 3

        if is_rgb:
            channels = []
            for ch in range(3):
                ch_data = image[:, :, ch]
                vals = ch_data[mask].astype(np.float64)
                roi_2d = ch_data[y0:y1, x0:x1].astype(np.float64) if "RobustStdDev" in self._enabled else None
                channels.append(_compute_channel(vals, area, self._enabled, roi_2d, self._robust_cfg))
            self._add_row_rgb(tool_name, roi_rect, channels, frame_name)
        else:
            data = image if image.ndim == 2 else image[:, :, 0]
            vals = data[mask].astype(np.float64)
            roi_2d = data[y0:y1, x0:x1].astype(np.float64) if "RobustStdDev" in self._enabled else None
            props = _compute_channel(vals, area, self._enabled, roi_2d, self._robust_cfg)
            self._add_row_gray(tool_name, roi_rect, props, frame_name)

    def _add_row_gray(self, tool_name: str,
                      roi_rect: tuple[int, int, int, int],
                      props: dict, frame_name: str = "") -> None:
        self._row_count += 1
        row = self._table.rowCount()
        self._table.insertRow(row)

        x0, y0, x1, y1 = roi_rect
        cols = self._column_names()
        values = {
            "#": str(self._row_count),
            "Frame": frame_name,
            "Tool": tool_name,
            "ROI": f"({x0},{y0}) {x1 - x0}×{y1 - y0}",
        }
        for key, val in props.items():
            if key == "Area":
                values[key] = str(val)
            else:
                values[key] = f"{val:.4f}"

        for ci, col in enumerate(cols):
            item = QTableWidgetItem(values.get(col, ""))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, ci, item)

        self._table.scrollToBottom()

    def _add_row_rgb(self, tool_name: str,
                     roi_rect: tuple[int, int, int, int],
                     channels: list[dict], frame_name: str = "") -> None:
        self._row_count += 1
        row = self._table.rowCount()
        self._table.insertRow(row)

        x0, y0, x1, y1 = roi_rect
        cols = self._column_names()
        ch_names = ["B", "G", "R"]

        values = {
            "#": str(self._row_count),
            "Frame": frame_name,
            "Tool": tool_name,
            "ROI": f"({x0},{y0}) {x1 - x0}×{y1 - y0}",
        }
        for prop_name, _ in AVAILABLE_PROPS:
            if prop_name not in self._enabled:
                continue
            parts = []
            for ci, ch_name in enumerate(ch_names):
                v = channels[ci].get(prop_name)
                if v is None:
                    continue
                if prop_name == "Area":
                    parts.append(str(v))
                    break
                else:
                    parts.append(f"{ch_name}:{v:.2f}")
            values[prop_name] = " ".join(parts)

        for ci, col in enumerate(cols):
            item = QTableWidgetItem(values.get(col, ""))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, ci, item)

        self._table.scrollToBottom()

    def _on_configure(self) -> None:
        dlg = MeasureConfigDialog(self._enabled, self._robust_cfg, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._enabled = dlg.get_enabled()
        self._robust_cfg = dlg.get_robust_cfg()
        self._rebuild_columns()

    def _on_clear(self) -> None:
        self._table.setRowCount(0)
        self._row_count = 0

    def _on_export_csv(self) -> None:
        if self._table.rowCount() == 0:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Measure Table", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        with open(path, "w") as f:
            f.write(self._table_to_text(","))

    def _on_copy_clipboard(self) -> None:
        if self._table.rowCount() == 0:
            return
        text = self._table_to_text("\t")
        QApplication.clipboard().setText(text)

    def _table_to_text(self, sep: str) -> str:
        cols = self._column_names()
        lines = [sep.join(cols)]
        for row in range(self._table.rowCount()):
            cells = []
            for ci in range(len(cols)):
                item = self._table.item(row, ci)
                cells.append(item.text() if item else "")
            lines.append(sep.join(cells))
        return "\n".join(lines) + "\n"

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Delete:
            rows = sorted({idx.row() for idx in self._table.selectedIndexes()},
                          reverse=True)
            for row in rows:
                self._table.removeRow(row)
            return
        super().keyPressEvent(event)
