"""ImageMatchingWidget: 이미지 스택 프레임 간 offset/rotation 자동 추정 및 보정."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from image_source import ImageSource
    from viewer import ImageViewer


class ImageMatchingWidget(QWidget):

    def __init__(
        self,
        viewer: ImageViewer,
        source: ImageSource,
        frame_idx: int,
        open_stack_cb: Callable[[np.ndarray, str, list[str] | None], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._source = source
        self._frame_idx = frame_idx
        self._open_stack_cb = open_stack_cb
        self._results: list[dict] | None = None

        layout = QVBoxLayout(self)

        # -- Method --
        layout.addWidget(QLabel("Method:"))
        method_row = QHBoxLayout()
        self._radio_phase = QRadioButton("Phase Correlation")
        self._radio_feature = QRadioButton("Feature Matching")
        self._radio_feature.setChecked(True)
        self._method_group = QButtonGroup(self)
        self._method_group.addButton(self._radio_phase, 0)
        self._method_group.addButton(self._radio_feature, 1)
        method_row.addWidget(self._radio_phase)
        method_row.addWidget(self._radio_feature)
        layout.addLayout(method_row)

        # -- Reference frame --
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference frame:"))
        self._spin_ref = QSpinBox()
        self._spin_ref.setRange(0, max(source.frame_count - 1, 0))
        self._spin_ref.setValue(0)
        ref_row.addWidget(self._spin_ref)
        ref_row.addStretch()
        layout.addLayout(ref_row)

        # -- Rotation --
        self._chk_rotation = QCheckBox("Estimate rotation")
        self._chk_rotation.setEnabled(self._radio_feature.isChecked())
        layout.addWidget(self._chk_rotation)

        # -- Output --
        layout.addWidget(QLabel("Output:"))
        out_row = QHBoxLayout()
        self._radio_new = QRadioButton("New window")
        self._radio_apply = QRadioButton("Apply to current")
        self._radio_new.setChecked(True)
        self._output_group = QButtonGroup(self)
        self._output_group.addButton(self._radio_new, 0)
        self._output_group.addButton(self._radio_apply, 1)
        out_row.addWidget(self._radio_new)
        out_row.addWidget(self._radio_apply)
        layout.addLayout(out_row)

        # -- Estimate button --
        self._btn_estimate = QPushButton("Estimate")
        self._btn_estimate.clicked.connect(self._estimate)
        layout.addWidget(self._btn_estimate)

        # -- Results table --
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Frame", "dx", "dy", "angle"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table, stretch=1)

        # -- Apply button --
        self._btn_apply = QPushButton("Apply")
        self._btn_apply.setEnabled(False)
        self._btn_apply.clicked.connect(self._apply)
        layout.addWidget(self._btn_apply)

        # -- Status --
        self._status = QLabel("")
        self._status.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self._status)

        # -- Signals --
        self._method_group.idToggled.connect(self._on_method_changed)

    def set_frame_idx(self, idx: int) -> None:
        self._frame_idx = idx
        if self._results is not None and 0 <= idx < len(self._results):
            self._table.selectRow(idx)

    def cleanup(self) -> None:
        self._results = None

    # ------------------------------------------------------------------ slots
    def _on_method_changed(self, id_: int, checked: bool) -> None:
        if not checked:
            return
        self._chk_rotation.setEnabled(id_ == 1)
        if id_ == 0:
            self._chk_rotation.setChecked(False)

    # ------------------------------------------------------------------ estimate
    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _to_u8(self, gray: np.ndarray) -> np.ndarray:
        if gray.dtype == np.uint8:
            return gray
        mn, mx = float(gray.min()), float(gray.max())
        if mx - mn < 1e-12:
            return np.zeros_like(gray, dtype=np.uint8)
        return ((gray.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)

    def _phase_correlate(self, ref_gray, tgt_gray):
        ref_f = ref_gray.astype(np.float32)
        tgt_f = tgt_gray.astype(np.float32)
        (dx, dy), response = cv2.phaseCorrelate(ref_f, tgt_f)
        return dx, dy, 0.0, response

    def _feature_match(self, ref_gray, tgt_gray, estimate_rotation):
        ref_u8 = self._to_u8(ref_gray)
        tgt_u8 = self._to_u8(tgt_gray)
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(ref_u8, None)
        kp2, des2 = orb.detectAndCompute(tgt_u8, None)
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 4:
            return None
        matches = sorted(matches, key=lambda m: m.distance)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        if estimate_rotation:
            M, inliers = cv2.estimateAffinePartial2D(pts2, pts1)
            if M is None:
                return None
            angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
            dx, dy = M[0, 2], M[1, 2]
        else:
            dx = float(np.median(pts1[:, 0] - pts2[:, 0]))
            dy = float(np.median(pts1[:, 1] - pts2[:, 1]))
            angle = 0.0
        return dx, dy, angle, 1.0

    def _estimate(self) -> None:
        n = self._source.frame_count
        ref_idx = self._spin_ref.value()
        ref_img = self._source.get_frame(ref_idx, copy=False)
        if ref_img is None:
            return
        ref_gray = self._to_gray(ref_img)
        use_phase = self._radio_phase.isChecked()
        estimate_rotation = self._chk_rotation.isChecked()

        progress = QProgressDialog("Estimating...", "Cancel", 0, n, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        results = []
        for i in range(n):
            if progress.wasCanceled():
                break
            if i == ref_idx:
                results.append({"dx": 0.0, "dy": 0.0, "angle": 0.0, "ok": True})
                progress.setValue(i + 1)
                continue
            img = self._source.get_frame(i, copy=False)
            if img is None:
                results.append({"dx": 0.0, "dy": 0.0, "angle": 0.0, "ok": False})
                progress.setValue(i + 1)
                continue
            tgt_gray = self._to_gray(img)
            if use_phase:
                r = self._phase_correlate(ref_gray, tgt_gray)
            else:
                r = self._feature_match(ref_gray, tgt_gray, estimate_rotation)
            if r is None:
                results.append({"dx": 0.0, "dy": 0.0, "angle": 0.0, "ok": False})
            else:
                dx, dy, angle, _ = r
                results.append({"dx": dx, "dy": dy, "angle": angle, "ok": True})
            progress.setValue(i + 1)
        progress.close()

        self._results = results
        self._populate_table()
        ok_count = sum(1 for r in results if r["ok"])
        self._status.setText(f"Estimated: {ok_count}/{len(results)} frames OK")
        self._btn_apply.setEnabled(ok_count > 0)

    def _populate_table(self) -> None:
        if self._results is None:
            return
        self._table.setRowCount(len(self._results))
        for i, r in enumerate(self._results):
            name = self._source.frame_name(i)
            self._table.setItem(i, 0, QTableWidgetItem(name))
            self._table.setItem(i, 1, QTableWidgetItem(f"{r['dx']:.2f}"))
            self._table.setItem(i, 2, QTableWidgetItem(f"{r['dy']:.2f}"))
            self._table.setItem(i, 3, QTableWidgetItem(f"{r['angle']:.3f}"))
            if not r["ok"]:
                for c in range(4):
                    item = self._table.item(i, c)
                    item.setForeground(Qt.GlobalColor.red)

    # ------------------------------------------------------------------ apply
    def _correct_frame(self, img, dx, dy, angle):
        h, w = img.shape[:2]
        if abs(angle) < 1e-6:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
        else:
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            M[0, 2] += dx
            M[1, 2] += dy
        return cv2.warpAffine(
            img, M, (w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    def _apply(self) -> None:
        if self._results is None:
            return
        n = self._source.frame_count
        new_window = self._radio_new.isChecked()

        progress = QProgressDialog("Applying correction...", "Cancel", 0, n, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        corrected = []
        names = []
        for i in range(n):
            if progress.wasCanceled():
                break
            img = self._source.get_frame(i, copy=True)
            if img is None:
                progress.setValue(i + 1)
                continue
            r = self._results[i] if i < len(self._results) else None
            if r and r["ok"] and (abs(r["dx"]) > 0.01 or abs(r["dy"]) > 0.01 or abs(r["angle"]) > 0.001):
                img = self._correct_frame(img, r["dx"], r["dy"], r["angle"])
            if new_window:
                corrected.append(img)
                names.append(self._source.frame_name(i))
            else:
                self._source.set_frame(i, img)
            progress.setValue(i + 1)
        progress.close()

        if new_window and corrected:
            stack = np.stack(corrected)
            self._open_stack_cb(stack, "Image Matching", names)
            self._status.setText("Corrected stack opened in new window.")
        elif not new_window:
            idx = self._frame_idx
            from viewer import ImageViewer
            self._viewer.set_image(self._source.get_frame(idx, copy=False))
            self._status.setText("Correction applied to current source.")
