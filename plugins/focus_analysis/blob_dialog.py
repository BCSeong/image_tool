"""Blob Detector 파라미터 설정 다이얼로그.

edge_spread_function metric 사용 시 SimpleBlobDetector 파라미터를 조정/테스트.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .focus_analyzer import FocusAnalyzer

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from image_source import ImageSource


class BlobDetectorDialog(QDialog):
    """SimpleBlobDetector 파라미터 설정 및 테스트 다이얼로그."""

    def __init__(self, source: ImageSource, grid_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Blob Detector Parameters")
        self.setMinimumSize(900, 700)
        self._source = source
        self._analyzer = FocusAnalyzer(grid_count)
        self._grids: list[np.ndarray] = []
        self.applied_params: cv2.SimpleBlobDetector_Params | None = None

        self._build_ui()
        self._connect()
        self._update_image_combo()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        main_h = QHBoxLayout()

        # -- Left: parameters --
        left_grp = QGroupBox("Parameters")
        left = QGridLayout(left_grp)
        row = 0

        left.addWidget(QLabel("minThreshold:"), row, 0)
        self._min_threshold = QSpinBox()
        self._min_threshold.setRange(0, 255)
        self._min_threshold.setValue(0)
        left.addWidget(self._min_threshold, row, 1)

        left.addWidget(QLabel("maxThreshold:"), row, 2)
        self._max_threshold = QSpinBox()
        self._max_threshold.setRange(0, 255)
        self._max_threshold.setValue(255)
        left.addWidget(self._max_threshold, row, 3)
        row += 1

        self._chk_area = QCheckBox("filterByArea")
        self._chk_area.setChecked(True)
        left.addWidget(self._chk_area, row, 0, 1, 2)
        left.addWidget(QLabel("minArea:"), row, 2)
        self._min_area = QSpinBox()
        self._min_area.setRange(1, 100000)
        self._min_area.setValue(100)
        left.addWidget(self._min_area, row, 3)
        row += 1

        left.addWidget(QLabel("maxArea:"), row, 2)
        self._max_area = QSpinBox()
        self._max_area.setRange(1, 1000000)
        self._max_area.setValue(10000)
        left.addWidget(self._max_area, row, 3)
        row += 1

        self._chk_circularity = QCheckBox("filterByCircularity")
        self._chk_circularity.setChecked(True)
        left.addWidget(self._chk_circularity, row, 0, 1, 2)
        left.addWidget(QLabel("minCircularity:"), row, 2)
        self._min_circularity = QDoubleSpinBox()
        self._min_circularity.setRange(0.0, 1.0)
        self._min_circularity.setValue(0.7)
        self._min_circularity.setDecimals(2)
        self._min_circularity.setSingleStep(0.1)
        left.addWidget(self._min_circularity, row, 3)
        row += 1

        self._chk_inertia = QCheckBox("filterByInertia")
        self._chk_inertia.setChecked(True)
        left.addWidget(self._chk_inertia, row, 0, 1, 2)
        left.addWidget(QLabel("minInertiaRatio:"), row, 2)
        self._min_inertia = QDoubleSpinBox()
        self._min_inertia.setRange(0.0, 1.0)
        self._min_inertia.setValue(0.7)
        self._min_inertia.setDecimals(2)
        self._min_inertia.setSingleStep(0.1)
        left.addWidget(self._min_inertia, row, 3)
        row += 1

        self._chk_convexity = QCheckBox("filterByConvexity")
        self._chk_convexity.setChecked(True)
        left.addWidget(self._chk_convexity, row, 0, 1, 2)
        left.addWidget(QLabel("minConvexity:"), row, 2)
        self._min_convexity = QDoubleSpinBox()
        self._min_convexity.setRange(0.0, 1.0)
        self._min_convexity.setValue(0.7)
        self._min_convexity.setDecimals(2)
        self._min_convexity.setSingleStep(0.1)
        left.addWidget(self._min_convexity, row, 3)
        row += 1

        left.addWidget(QLabel("Image:"), row, 0)
        self._image_combo = QComboBox()
        left.addWidget(self._image_combo, row, 1, 1, 3)
        row += 1

        left.addWidget(QLabel("Grid:"), row, 0)
        self._grid_combo = QComboBox()
        left.addWidget(self._grid_combo, row, 1, 1, 3)
        row += 1

        self._btn_test = QPushButton("Test")
        left.addWidget(self._btn_test, row, 0, 1, 4)
        row += 1

        self._result_label = QLabel("Click Test to detect blobs.")
        self._result_label.setWordWrap(True)
        left.addWidget(self._result_label, row, 0, 1, 4)

        main_h.addWidget(left_grp, 1)

        # -- Right: preview --
        right_grp = QGroupBox("Preview")
        right = QVBoxLayout(right_grp)
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(400, 400)
        self._preview.setStyleSheet("border: 1px solid #ccc; background: #222;")
        self._preview.setText("No preview")
        right.addWidget(self._preview)
        main_h.addWidget(right_grp, 2)

        layout.addLayout(main_h)

        # -- Buttons --
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_apply = QPushButton("Apply")
        self._btn_cancel = QPushButton("Cancel")
        btn_row.addWidget(self._btn_apply)
        btn_row.addWidget(self._btn_cancel)
        layout.addLayout(btn_row)

    def _connect(self) -> None:
        self._image_combo.currentIndexChanged.connect(self._on_image_changed)
        self._btn_test.clicked.connect(self._on_test)
        self._btn_apply.clicked.connect(self._on_apply)
        self._btn_cancel.clicked.connect(self.reject)

    def _update_image_combo(self) -> None:
        self._image_combo.clear()
        for i in range(self._source.frame_count):
            self._image_combo.addItem(f"Frame {i}")
        if self._source.frame_count > 0:
            self._on_image_changed(0)

    def _on_image_changed(self, idx: int) -> None:
        if idx < 0 or idx >= self._source.frame_count:
            return
        img = self._source.get_frame(idx, copy=False)
        if img is None:
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
            if mx > mn:
                img = ((img.astype(np.float64) - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                img = np.zeros(img.shape[:2], dtype=np.uint8)

        self._grids, _ = self._analyzer.divide_into_grids(img)
        gc = self._analyzer.grid_count
        self._grid_combo.clear()
        for i in range(len(self._grids)):
            gy, gx = divmod(i, gc)
            self._grid_combo.addItem(f"Grid {i} (Y:{gy}, X:{gx})")

    def _create_params(self) -> cv2.SimpleBlobDetector_Params:
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = self._min_threshold.value()
        params.maxThreshold = self._max_threshold.value()
        params.filterByArea = self._chk_area.isChecked()
        if params.filterByArea:
            params.minArea = self._min_area.value()
            params.maxArea = self._max_area.value()
        params.filterByCircularity = self._chk_circularity.isChecked()
        if params.filterByCircularity:
            params.minCircularity = self._min_circularity.value()
        params.filterByInertia = self._chk_inertia.isChecked()
        if params.filterByInertia:
            params.minInertiaRatio = self._min_inertia.value()
        params.filterByConvexity = self._chk_convexity.isChecked()
        if params.filterByConvexity:
            params.minConvexity = self._min_convexity.value()
        return params

    def _on_test(self) -> None:
        grid_idx = self._grid_combo.currentIndex()
        if grid_idx < 0 or grid_idx >= len(self._grids):
            QMessageBox.warning(self, "Warning", "Select a valid grid.")
            return

        test_img = self._grids[grid_idx].copy()
        params = self._create_params()

        try:
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(test_img)

            result_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                r = int(kp.size / 2)
                cv2.circle(result_img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(result_img, (x, y), 2, (0, 0, 255), -1)

            self._show_preview(result_img)
            self._result_label.setText(f"Detected blobs: {len(keypoints)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self._result_label.setText(f"Error: {e}")

    def _show_preview(self, img: np.ndarray) -> None:
        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        preview_size = self._preview.size()
        scaled = pixmap.scaled(
            preview_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(scaled)

    def _on_apply(self) -> None:
        self.applied_params = self._create_params()
        self.accept()
