"""Brightness/Contrast 조절 다이얼로그.

히스토그램 + min/max/brightness/contrast 슬라이더 + Apply/Reset.
성능: LUT 기반 viewer 업데이트, QTimer 디바운스.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressDialog,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from image_source import ImageSource
from viewer import ImageViewer

_SLIDER_STEPS = 10000


class HistogramCanvas(FigureCanvasQTAgg):
    """히스토그램 + min/max 수직선 표시."""

    def __init__(self, parent=None) -> None:
        self._fig = Figure(figsize=(4.5, 2.0), dpi=100)
        self._fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.15)
        super().__init__(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._line_min = None
        self._line_max = None
        self.setMinimumHeight(160)

    def plot_histogram(self, img: np.ndarray, bins: int = 256) -> None:
        self._ax.clear()
        is_rgb = img.ndim == 3 and img.shape[2] == 3
        if is_rgb:
            for ch, color in enumerate(["red", "green", "blue"]):
                data = img[:, :, ch].ravel()
                finite = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.floating) else data
                if len(finite) == 0:
                    continue
                self._ax.hist(finite, bins=bins, color=color, alpha=0.4,
                              histtype="stepfilled", linewidth=0.5)
        else:
            data = img.ravel()
            finite = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.floating) else data
            if len(finite) > 0:
                self._ax.hist(finite, bins=bins, color="gray", alpha=0.7,
                              histtype="stepfilled", linewidth=0.5)
        self._ax.set_ylabel("")
        self._ax.set_xlabel("")
        self._ax.tick_params(labelsize=7)
        self._line_min = self._ax.axvline(0, color="red", linewidth=1.2, linestyle="--")
        self._line_max = self._ax.axvline(1, color="red", linewidth=1.2, linestyle="--")
        self.draw()

    def update_lines(self, mn: float, mx: float) -> None:
        if self._line_min is None:
            return
        self._line_min.set_xdata([mn, mn])
        self._line_max.set_xdata([mx, mx])
        margin = (mx - mn) * 0.05 if mx > mn else 1.0
        self._ax.set_xlim(mn - margin, mx + margin)
        self.draw_idle()


class _SliderRow:
    """슬라이더 + 수치 입력 한 쌍. value_changed 시그널 하나만 외부에 노출."""

    def __init__(self, label: str, parent_layout: QVBoxLayout,
                 val_min: float, val_max: float, value: float, decimals: int) -> None:
        self._val_min = val_min
        self._val_max = val_max
        self._decimals = decimals
        self._syncing = False

        row = QHBoxLayout()
        row.addWidget(QLabel(label))

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, _SLIDER_STEPS)
        row.addWidget(self.slider, stretch=1)

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(val_min, val_max)
        self.spin.setValue(value)
        self.spin.setFixedWidth(100)
        row.addWidget(self.spin)

        parent_layout.addLayout(row)

        self._set_slider_from_value(value)
        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

    def _val_to_slider(self, v: float) -> int:
        if self._val_max - self._val_min < 1e-12:
            return 0
        return int((v - self._val_min) / (self._val_max - self._val_min) * _SLIDER_STEPS)

    def _slider_to_val(self, s: int) -> float:
        return self._val_min + s / _SLIDER_STEPS * (self._val_max - self._val_min)

    def _set_slider_from_value(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._val_to_slider(v))
        self.slider.blockSignals(False)

    def _on_slider(self, s: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.spin.setValue(self._slider_to_val(s))
        self._syncing = False

    def _on_spin(self, val: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        self._set_slider_from_value(val)
        self._syncing = False

    @property
    def value(self) -> float:
        return self.spin.value()

    @value.setter
    def value(self, v: float) -> None:
        self._syncing = True
        self.spin.setValue(v)
        self._set_slider_from_value(v)
        self._syncing = False

    def set_range(self, lo: float, hi: float) -> None:
        self._val_min = lo
        self._val_max = hi
        self.spin.setRange(lo, hi)
        self._set_slider_from_value(self.spin.value())


class BCDialog(QWidget):
    """Brightness / Contrast 조절 위젯. QDockWidget 내부에 배치."""

    display_changed = Signal(float, float)

    def __init__(self, viewer: ImageViewer, source: ImageSource,
                 frame_idx: int, undo_mgr=None, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(420)
        self._viewer = viewer
        self._source = source
        self._frame_idx = frame_idx
        self._undo_mgr = undo_mgr

        img = source.get_frame(frame_idx)
        self._img = img
        self._data_min = float(np.nanmin(img))
        self._data_max = float(np.nanmax(img))
        self._init_min = self._data_min
        self._init_max = self._data_max
        self._syncing = False

        existing = viewer.get_display_range()
        if existing is not None:
            self._data_min, self._data_max = existing

        self._pending_mn: float | None = None
        self._pending_mx: float | None = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(30)
        self._debounce.timeout.connect(self._flush_preview)

        self._build_ui()
        self._connect()
        self._update_histogram()
        self._viewer.set_display_range(self._data_min, self._data_max)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self._canvas = HistogramCanvas()
        root.addWidget(self._canvas)

        grp = QGroupBox("Display Range")
        vg = QVBoxLayout(grp)

        d = self._decimals()
        rng_lo = self._data_min - (self._data_max - self._data_min) * 0.5
        rng_hi = self._data_max + (self._data_max - self._data_min) * 0.5

        self._row_min = _SliderRow("Min:", vg, rng_lo, rng_hi, self._data_min, d)
        self._row_max = _SliderRow("Max:", vg, rng_lo, rng_hi, self._data_max, d)

        center = (self._data_min + self._data_max) / 2
        width = self._data_max - self._data_min
        self._row_bright = _SliderRow("Brightness:", vg, rng_lo, rng_hi, center, d)
        self._row_contrast = _SliderRow("Contrast:", vg, 0, (rng_hi - rng_lo) * 2, width, d)

        root.addWidget(grp)

        self._lock_check = QCheckBox("Lock display range")
        self._lock_check.setChecked(True)
        root.addWidget(self._lock_check)

        self._propagate_check = QCheckBox("Propagate to all slices")
        self._propagate_check.setChecked(True)
        root.addWidget(self._propagate_check)

        btn_row = QHBoxLayout()
        self._btn_auto = QPushButton("Auto")
        self._btn_reset = QPushButton("Reset")
        self._btn_apply = QPushButton("Apply")
        self._btn_apply.setDefault(True)
        btn_row.addWidget(self._btn_auto)
        btn_row.addWidget(self._btn_reset)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_apply)
        root.addLayout(btn_row)

    def _decimals(self) -> int:
        if self._img is None:
            return 1
        if self._img.dtype in (np.float32, np.float64):
            return 4
        if self._img.dtype == np.uint16:
            return 1
        return 0

    def _connect(self) -> None:
        self._row_min.spin.valueChanged.connect(self._on_minmax_changed)
        self._row_max.spin.valueChanged.connect(self._on_minmax_changed)

        self._row_bright.spin.valueChanged.connect(self._on_bc_changed)
        self._row_contrast.spin.valueChanged.connect(self._on_bc_changed)

        self._btn_auto.clicked.connect(self._on_auto)
        self._btn_reset.clicked.connect(self._on_reset)
        self._btn_apply.clicked.connect(self._on_apply)

    def _on_minmax_changed(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        mn = self._row_min.value
        mx = self._row_max.value
        self._row_bright.value = (mn + mx) / 2
        self._row_contrast.value = mx - mn
        self._syncing = False
        self._schedule_preview(mn, mx)

    def _on_bc_changed(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        b = self._row_bright.value
        c = self._row_contrast.value
        mn = b - c / 2
        mx = b + c / 2
        self._row_min.value = mn
        self._row_max.value = mx
        self._syncing = False
        self._schedule_preview(mn, mx)

    def _schedule_preview(self, mn: float, mx: float) -> None:
        """디바운스: 30ms 내 마지막 값만 실제 처리."""
        self._pending_mn = mn
        self._pending_mx = mx
        self._canvas.update_lines(mn, mx)
        self._debounce.start()

    def _cancel_pending(self) -> None:
        """디바운스 타이머를 중단하고 pending 값 초기화."""
        self._debounce.stop()
        self._pending_mn = None
        self._pending_mx = None

    def _flush_preview(self) -> None:
        if self._pending_mn is not None and self._pending_mx is not None:
            self._viewer.set_display_range(self._pending_mn, self._pending_mx)

    def _update_histogram(self) -> None:
        if self._img is not None:
            nbins = 256 if self._img.dtype == np.uint8 else 512
            self._canvas.plot_histogram(self._img, bins=nbins)
        self._update_lines()

    def _update_lines(self) -> None:
        self._canvas.update_lines(self._row_min.value, self._row_max.value)

    def _on_auto(self) -> None:
        self._cancel_pending()
        img = self._source.get_frame(self._frame_idx)
        if img is None:
            return
        self._img = img
        self._data_min = float(np.nanmin(img))
        self._data_max = float(np.nanmax(img))
        self._set_all(self._data_min, self._data_max)
        self._update_histogram()
        self._viewer.set_display_range(self._data_min, self._data_max)

    def _on_reset(self) -> None:
        self._cancel_pending()
        self._set_all(self._init_min, self._init_max)
        self._update_histogram()
        self._viewer.set_display_range(self._init_min, self._init_max)

    def _set_all(self, mn: float, mx: float) -> None:
        self._syncing = True
        self._row_min.value = mn
        self._row_max.value = mx
        self._row_bright.value = (mn + mx) / 2
        self._row_contrast.value = mx - mn
        self._syncing = False
        self._canvas.update_lines(mn, mx)

    def _on_apply(self) -> None:
        self._cancel_pending()
        mn = self._row_min.value
        mx = self._row_max.value
        propagate = self._propagate_check.isChecked()

        if propagate:
            indices = list(range(self._source.frame_count))
        else:
            indices = [self._frame_idx]

        n = len(indices)
        cancelled = False

        if n > 1:
            progress = QProgressDialog("Applying B/C...", "Cancel", 0, n, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            for count, i in enumerate(indices):
                if progress.wasCanceled():
                    cancelled = True
                    break
                img = self._source.get_frame(i)
                if img is not None:
                    if self._undo_mgr is not None:
                        self._undo_mgr.push(i, img, "B/C")
                    self._source.set_frame(i, self._remap(img, mn, mx))
                progress.setValue(count + 1)
                QApplication.processEvents()

            progress.close()
        else:
            img = self._source.get_frame(indices[0])
            if img is not None:
                if self._undo_mgr is not None:
                    self._undo_mgr.push(indices[0], img, "B/C")
                self._source.set_frame(indices[0], self._remap(img, mn, mx))

        self._viewer._display_min = None
        self._viewer._display_max = None
        self._viewer._auto_min = None
        self._viewer._auto_max = None

        new_img = self._source.get_frame(self._frame_idx)
        if new_img is not None:
            # 새 이미지를 먼저 viewer에 반영 (auto display range 적용)
            self._viewer.set_image(new_img)

            self._img = new_img
            self._data_min = float(np.nanmin(new_img))
            self._data_max = float(np.nanmax(new_img))
            self._init_min = self._data_min
            self._init_max = self._data_max

            rng_lo = self._data_min - (self._data_max - self._data_min) * 0.5
            rng_hi = self._data_max + (self._data_max - self._data_min) * 0.5
            self._syncing = True
            self._row_min.set_range(rng_lo, rng_hi)
            self._row_max.set_range(rng_lo, rng_hi)
            self._row_bright.set_range(rng_lo, rng_hi)
            self._row_contrast.set_range(0, (rng_hi - rng_lo) * 2)
            self._set_all(self._data_min, self._data_max)
            self._syncing = False
            self._cancel_pending()
            self._update_histogram()

    @staticmethod
    def _remap(img: np.ndarray, mn: float, mx: float) -> np.ndarray:
        """display_min..display_max -> 0..type_max."""
        dtype = img.dtype
        span = mx - mn
        if span < 1e-12:
            return np.zeros_like(img)

        if dtype == np.uint8:
            lut = np.arange(256, dtype=np.float32)
            lut = ((lut - mn) / span * 255).clip(0, 255).astype(np.uint8)
            return cv2.LUT(img, lut) if img.ndim <= 3 else lut[img]

        if dtype == np.uint16:
            lut = np.arange(65536, dtype=np.float32)
            lut_u16 = ((lut - mn) / span * 65535).clip(0, 65535).astype(np.uint16)
            if img.ndim == 2:
                return lut_u16[img]
            return np.stack([lut_u16[img[:, :, c]] for c in range(img.shape[2])], axis=2)

        out = ((img.astype(np.float64) - mn) / span).clip(0, 1)
        return out.astype(dtype)

    def set_frame_idx(self, idx: int) -> None:
        self._frame_idx = idx
        if self._lock_check.isChecked():
            return
        img = self._source.get_frame(idx)
        if img is None:
            return
        self._img = img
        self._data_min = float(np.nanmin(img))
        self._data_max = float(np.nanmax(img))
        self._set_all(self._data_min, self._data_max)
        self._update_histogram()
        self._viewer.set_display_range(self._data_min, self._data_max)

    def cleanup(self) -> None:
        """Dock 닫힐 때 호출: 타이머 정리. display range는 유지."""
        self._cancel_pending()
        self._debounce.stop()
