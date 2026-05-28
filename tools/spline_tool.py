"""Spline Profile Tool: cubic-spline 곡선 경로를 따른 프로파일 추출."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QBrush, QPainterPath, QPen, QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tool_base import BaseTool
from viewer import ImageViewer


class SplineTool(BaseTool):

    _HANDLE_HALF = 5
    _HIT_TOLERANCE = 15

    def __init__(self, viewer: ImageViewer, source=None) -> None:
        self._viewer = viewer
        self._source = source
        self._control_points: list[list[int]] = []
        self._finalized = False
        self._handle_items: list[QGraphicsEllipseItem] = []
        self._path_item: QGraphicsPathItem | None = None
        self._dragging_handle: int = -1
        self._moving = False
        self._move_ox = self._move_oy = 0
        self._frame_idx = 0
        self._panel: QWidget | None = None
        self._canvas: FigureCanvasQTAgg | None = None
        self._length_label: QLabel | None = None
        self._points_label: QLabel | None = None
        self._held_figures: list[dict] = []
        self._held_layout: QVBoxLayout | None = None
        self._superpose_entry: dict | None = None
        self._spline_xs: np.ndarray | None = None
        self._spline_ys: np.ndarray | None = None
        self._peak_items: list[QGraphicsEllipseItem] = []
        self._chk_peaks: QCheckBox | None = None
        self._spin_prominence: QDoubleSpinBox | None = None
        self._cp_spins: list[tuple[QSpinBox, QSpinBox]] = []
        self._cp_widgets: list[QWidget] = []
        self._cp_layout: QVBoxLayout | None = None
        self._syncing_spins = False

    @property
    def name(self) -> str:
        return "Spline"

    def _pen(self) -> QPen:
        pen = QPen(QColor(0, 200, 255), 2)
        pen.setCosmetic(True)
        return pen

    # ---------------------------------------------------------------- spline
    def _compute_spline_coords(self) -> bool:
        pts = self._control_points
        n = len(pts)
        if n < 2:
            self._spline_xs = None
            self._spline_ys = None
            return False
        t = np.arange(n, dtype=float)
        px = np.array([p[0] for p in pts], dtype=float)
        py = np.array([p[1] for p in pts], dtype=float)
        if n == 2:
            num = max(int(np.hypot(px[1] - px[0], py[1] - py[0])), 2)
            self._spline_xs = np.linspace(px[0], px[1], num)
            self._spline_ys = np.linspace(py[0], py[1], num)
            return True
        cs_x = CubicSpline(t, px, bc_type="natural")
        cs_y = CubicSpline(t, py, bc_type="natural")
        seg_lengths = np.array([
            np.hypot(px[i + 1] - px[i], py[i + 1] - py[i]) for i in range(n - 1)
        ])
        total = seg_lengths.sum()
        num = max(int(total), 2)
        t_fine = np.linspace(0, n - 1, num)
        self._spline_xs = cs_x(t_fine)
        self._spline_ys = cs_y(t_fine)
        return True

    # ---------------------------------------------------------------- drawing
    def _update_path(self) -> None:
        if self._spline_xs is None or len(self._spline_xs) < 2:
            if self._path_item is not None:
                self._viewer.scene_ref.removeItem(self._path_item)
                self._path_item = None
            return
        pp = QPainterPath()
        pp.moveTo(QPointF(self._spline_xs[0], self._spline_ys[0]))
        for i in range(1, len(self._spline_xs)):
            pp.lineTo(QPointF(self._spline_xs[i], self._spline_ys[i]))
        if self._path_item is None:
            self._path_item = QGraphicsPathItem()
            self._path_item.setPen(self._pen())
            self._path_item.setZValue(10)
            self._viewer.scene_ref.addItem(self._path_item)
        self._path_item.setPath(pp)
        self._update_handles()

    def _scene_pixel_size(self) -> float:
        t = self._viewer.transform()
        return 1.0 / t.m11() if t.m11() != 0 else 1.0

    def _update_handles(self) -> None:
        ps = self._scene_pixel_size()
        half = self._HANDLE_HALF * ps
        scene = self._viewer.scene_ref
        pen = QPen(QColor(0, 200, 255), 1)
        pen.setCosmetic(True)
        brush = QBrush(QColor(255, 255, 255))
        while len(self._handle_items) < len(self._control_points):
            item = QGraphicsEllipseItem()
            item.setPen(pen)
            item.setBrush(brush)
            item.setZValue(100)
            scene.addItem(item)
            self._handle_items.append(item)
        for i, pt in enumerate(self._control_points):
            self._handle_items[i].setRect(
                pt[0] - half, pt[1] - half, half * 2, half * 2,
            )
            self._handle_items[i].setVisible(True)
        for i in range(len(self._control_points), len(self._handle_items)):
            self._handle_items[i].setVisible(False)

    def _remove_handles(self) -> None:
        scene = self._viewer.scene_ref
        for item in self._handle_items:
            scene.removeItem(item)
        self._handle_items.clear()

    def _handle_hit_test(self, x: int, y: int) -> int:
        ps = self._scene_pixel_size()
        radius = (self._HANDLE_HALF + 2) * ps
        for i, pt in enumerate(self._control_points):
            if math.hypot(x - pt[0], y - pt[1]) <= radius:
                return i
        return -1

    def _spline_hit_test(self, x: int, y: int) -> bool:
        if self._spline_xs is None:
            return False
        dx = self._spline_xs - x
        dy = self._spline_ys - y
        return float(np.min(dx * dx + dy * dy)) <= self._HIT_TOLERANCE ** 2

    # ---------------------------------------------------------------- mouse
    def on_mouse_press(self, x: int, y: int, event) -> bool:
        if self._control_points:
            h_idx = self._handle_hit_test(x, y)
            if h_idx >= 0:
                self._dragging_handle = h_idx
                return True
            if self._spline_hit_test(x, y):
                self._moving = True
                self._move_ox, self._move_oy = x, y
                return True
        if self._finalized:
            self._clear_spline()
        self._control_points.append([x, y])
        self._compute_spline_coords()
        self._update_path()
        if self._points_label:
            self._points_label.setText(f"Points: {len(self._control_points)}")
        self._sync_cp_spinboxes()
        if len(self._control_points) >= 2:
            self._compute_and_display()
        return True

    def on_mouse_move(self, x: int, y: int, event) -> None:
        if self._dragging_handle >= 0:
            self._control_points[self._dragging_handle] = [x, y]
            self._compute_spline_coords()
            self._update_path()
            self._sync_cp_spinboxes()
            self._compute_and_display()
            return
        if self._moving:
            dx, dy = x - self._move_ox, y - self._move_oy
            for pt in self._control_points:
                pt[0] += dx
                pt[1] += dy
            self._move_ox, self._move_oy = x, y
            self._compute_spline_coords()
            self._update_path()
            self._sync_cp_spinboxes()
            self._compute_and_display()

    def on_mouse_release(self, x: int, y: int, event) -> None:
        if self._dragging_handle >= 0:
            self._dragging_handle = -1
            self._compute_and_display()
            return
        if self._moving:
            self._moving = False
            self._compute_and_display()

    def _undo_last_point(self) -> None:
        if not self._finalized and self._control_points:
            self._control_points.pop()
            self._compute_spline_coords()
            self._update_path()
            if self._points_label:
                self._points_label.setText(
                    f"Points: {len(self._control_points)}")
            self._sync_cp_spinboxes()
            if len(self._control_points) >= 2:
                self._compute_and_display()

    def on_key_press(self, key: int, event) -> bool:
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if len(self._control_points) >= 2:
                self._finalized = True
                self._compute_spline_coords()
                self._update_path()
                self._compute_and_display()
            return True
        if key == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._undo_last_point()
            return True
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self._undo_last_point()
            return True
        if self._control_points and key in (
            Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down,
        ):
            dx = -1 if key == Qt.Key.Key_Left else 1 if key == Qt.Key.Key_Right else 0
            dy = -1 if key == Qt.Key.Key_Up else 1 if key == Qt.Key.Key_Down else 0
            for pt in self._control_points:
                pt[0] += dx
                pt[1] += dy
            self._compute_spline_coords()
            self._update_path()
            self._sync_cp_spinboxes()
            self._compute_and_display()
            return True
        return False

    def on_frame_changed(self, idx: int, image: np.ndarray | None) -> None:
        self._frame_idx = idx
        if self._finalized and len(self._control_points) >= 2:
            self._compute_and_display()

    def deactivate(self) -> None:
        self._remove_handles()
        self._remove_peak_markers()
        if self._path_item is not None:
            self._viewer.scene_ref.removeItem(self._path_item)
            self._path_item = None
        self._control_points.clear()
        self._finalized = False
        self._spline_xs = None
        self._spline_ys = None

    def _clear_spline(self) -> None:
        self._remove_handles()
        self._remove_peak_markers()
        if self._path_item is not None:
            self._viewer.scene_ref.removeItem(self._path_item)
            self._path_item = None
        self._control_points.clear()
        self._finalized = False
        self._spline_xs = None
        self._spline_ys = None
        if self._points_label:
            self._points_label.setText("Points: 0")
        self._sync_cp_spinboxes()

    # ---------------------------------------------------------------- profile
    def _get_profile(self) -> tuple[np.ndarray, np.ndarray] | None:
        img = self._viewer.raw_image
        if img is None:
            return None
        return self._get_profile_from(img)

    def _get_profile_from(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if self._spline_xs is None or len(self._spline_xs) < 2:
            return None
        xs = np.clip(self._spline_xs.astype(int), 0, img.shape[1] - 1)
        ys = np.clip(self._spline_ys.astype(int), 0, img.shape[0] - 1)
        dx = np.diff(self._spline_xs)
        dy = np.diff(self._spline_ys)
        seg = np.sqrt(dx * dx + dy * dy)
        dist = np.concatenate([[0], np.cumsum(seg)])
        values = img[ys, xs]
        return dist, values

    def _plot_profile(self, ax, dist, values) -> None:
        if values.ndim == 2 and values.shape[1] == 3:
            for ch, color in enumerate(["blue", "green", "red"]):
                ax.plot(dist, values[:, ch], color=color, linewidth=0.8, alpha=0.8)
        else:
            ax.plot(dist, values, color="gray", linewidth=0.8)
        ax.set_xlabel("Distance (px)", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.tick_params(labelsize=7)

    def _update_live_plot(self, ax, dist, values) -> None:
        is_rgb = values.ndim == 2 and values.shape[1] == 3
        lines = ax.get_lines()
        expected = 3 if is_rgb else 1
        if len(lines) != expected:
            ax.clear()
            self._plot_profile(ax, dist, values)
            return
        if is_rgb:
            for ch in range(3):
                lines[ch].set_data(dist, values[:, ch])
        else:
            lines[0].set_data(dist, values)
        ax.relim()
        ax.autoscale_view()

    def _compute_and_display(self) -> None:
        result = self._get_profile()
        if result is None or self._canvas is None:
            return
        dist, values = result
        fig = self._canvas.figure
        axes = fig.get_axes()
        if axes:
            ax = axes[0]
            self._update_live_plot(ax, dist, values)
        else:
            ax = fig.add_subplot(111)
            self._plot_profile(ax, dist, values)
        self._apply_axis_settings(ax)
        fig.tight_layout()
        self._canvas.draw_idle()
        if self._length_label:
            self._length_label.setText(f"Length: {dist[-1]:.1f} px")
        self._update_peak_markers()

    # ---------------------------------------------------------------- peaks
    def _update_peak_markers(self) -> None:
        self._remove_peak_markers()
        if self._chk_peaks is None or not self._chk_peaks.isChecked():
            return
        if self._spline_xs is None or len(self._spline_xs) < 2:
            return
        result = self._get_profile()
        if result is None:
            return
        _, values = result
        if values.ndim == 2 and values.shape[1] == 3:
            vals_1d = values.astype(float).mean(axis=1)
        else:
            vals_1d = values.astype(float)
        prom = self._spin_prominence.value() if self._spin_prominence else 10.0
        peaks, _ = find_peaks(vals_1d, prominence=prom)
        ps = self._scene_pixel_size()
        half = self._HANDLE_HALF * ps
        scene = self._viewer.scene_ref
        pen = QPen(QColor(255, 50, 50), 1)
        pen.setCosmetic(True)
        brush = QBrush(QColor(255, 50, 50, 160))
        for pi in peaks:
            if pi >= len(self._spline_xs):
                continue
            x, y = float(self._spline_xs[pi]), float(self._spline_ys[pi])
            item = QGraphicsEllipseItem(x - half, y - half, half * 2, half * 2)
            item.setPen(pen)
            item.setBrush(brush)
            item.setZValue(90)
            scene.addItem(item)
            self._peak_items.append(item)

    def _remove_peak_markers(self) -> None:
        scene = self._viewer.scene_ref
        for item in self._peak_items:
            scene.removeItem(item)
        self._peak_items.clear()

    # ---------------------------------------------------------------- control point spinboxes
    def _sync_cp_spinboxes(self) -> None:
        if self._cp_layout is None:
            return
        self._syncing_spins = True
        n = len(self._control_points)
        while len(self._cp_spins) < n:
            self._add_cp_spin_row(len(self._cp_spins))
        for i in range(n):
            self._cp_spins[i][0].setValue(self._control_points[i][0])
            self._cp_spins[i][1].setValue(self._control_points[i][1])
            self._cp_widgets[i].setVisible(True)
        for i in range(n, len(self._cp_spins)):
            self._cp_widgets[i].setVisible(False)
        self._syncing_spins = False

    def _add_cp_spin_row(self, idx: int) -> None:
        row_w = QWidget()
        row = QHBoxLayout(row_w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        lbl = QLabel(f"#{idx + 1}")
        lbl.setFixedWidth(24)
        row.addWidget(lbl)
        row.addWidget(QLabel("X:"))
        sx = QSpinBox()
        sx.setRange(0, 99999)
        sx.setFixedWidth(70)
        row.addWidget(sx)
        row.addWidget(QLabel("Y:"))
        sy = QSpinBox()
        sy.setRange(0, 99999)
        sy.setFixedWidth(70)
        row.addWidget(sy)
        row.addStretch()
        sx.valueChanged.connect(lambda v, i=idx: self._on_cp_spin_changed(i, 0, v))
        sy.valueChanged.connect(lambda v, i=idx: self._on_cp_spin_changed(i, 1, v))
        self._cp_spins.append((sx, sy))
        self._cp_widgets.append(row_w)
        self._cp_layout.addWidget(row_w)

    def _on_cp_spin_changed(self, idx: int, axis: int, val: int) -> None:
        if self._syncing_spins:
            return
        if idx >= len(self._control_points):
            return
        self._control_points[idx][axis] = val
        self._compute_spline_coords()
        self._update_path()
        if self._finalized:
            self._compute_and_display()

    def _on_peak_toggled(self, *_args) -> None:
        if len(self._control_points) >= 2:
            self._update_peak_markers()

    # ---------------------------------------------------------------- axis
    def _apply_axis_settings(self, ax) -> None:
        if self._chk_auto_x.isChecked():
            ax.set_xlim(auto=True)
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=False)
        else:
            ax.set_xlim(self._spin_xmin.value(), self._spin_xmax.value())
        if self._chk_auto_y.isChecked():
            ax.set_ylim(auto=True)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
        else:
            ax.set_ylim(self._spin_ymin.value(), self._spin_ymax.value())
        xlo, xhi = ax.get_xlim()
        if self._chk_flip_x.isChecked() and xlo < xhi:
            ax.set_xlim(xhi, xlo)
        elif not self._chk_flip_x.isChecked() and xlo > xhi:
            ax.set_xlim(xhi, xlo)
        ylo, yhi = ax.get_ylim()
        if self._chk_flip_y.isChecked() and ylo < yhi:
            ax.set_ylim(yhi, ylo)
        elif not self._chk_flip_y.isChecked() and ylo > yhi:
            ax.set_ylim(yhi, ylo)

    def _on_setting_changed(self) -> None:
        self._spin_xmin.setEnabled(not self._chk_auto_x.isChecked())
        self._spin_xmax.setEnabled(not self._chk_auto_x.isChecked())
        self._spin_ymin.setEnabled(not self._chk_auto_y.isChecked())
        self._spin_ymax.setEnabled(not self._chk_auto_y.isChecked())
        if len(self._control_points) >= 2:
            self._compute_and_display()

    def _on_figsize_changed(self) -> None:
        w = self._spin_fig_w.value()
        h = self._spin_fig_h.value()
        self._canvas.figure.set_size_inches(w, h, forward=False)
        self._canvas.setFixedSize(int(w * 100), int(h * 100))
        self._update_export_label()
        if len(self._control_points) >= 2:
            self._compute_and_display()

    def _update_export_label(self) -> None:
        w = self._spin_fig_w.value()
        h = self._spin_fig_h.value()
        dpi = self._spin_dpi.value()
        self._export_px_label.setText(f"Export: {int(w * dpi)} × {int(h * dpi)} px")

    def _frame_label_text(self, idx: int) -> str:
        if self._source is not None:
            return self._source.frame_name(idx)
        return f"frame{idx}"

    # ---------------------------------------------------------------- hold
    def _plot_profile_labeled(self, ax, dist, values, label: str) -> None:
        if values.ndim == 2 and values.shape[1] == 3:
            c = ax.plot(dist, values[:, 0], linewidth=0.8, alpha=0.8, label=label)[0].get_color()
            ax.plot(dist, values[:, 1], linewidth=0.8, alpha=0.8, color=c, linestyle="--")
            ax.plot(dist, values[:, 2], linewidth=0.8, alpha=0.8, color=c, linestyle=":")
        else:
            ax.plot(dist, values, linewidth=0.8, label=label)
        ax.set_xlabel("Distance (px)", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.tick_params(labelsize=7)

    def _on_hold_superpose(self, dist, values, label) -> None:
        if self._superpose_entry is None:
            fw = self._spin_fig_w.value()
            fh = self._spin_fig_h.value()
            fig = Figure(figsize=(fw, fh), dpi=100)
            ax = fig.add_subplot(111)
            self._plot_profile_labeled(ax, dist, values, label)
            self._apply_axis_settings(ax)
            ax.legend(fontsize=6, loc="best")
            fig.tight_layout()
            canvas = FigureCanvasQTAgg(fig)
            canvas.setFixedSize(int(fw * 100), int(fh * 100))
            canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            entry = {
                "canvas": canvas, "fig": fig, "label": "Superposition",
                "dist": dist, "values": values,
                "_profiles": [(dist, values, label)],
                "_all_profiles": [(dist, values, label)],
                "_is_superpose": True,
            }
            self._superpose_entry = entry
            self._held_figures.append(entry)
            canvas.customContextMenuRequested.connect(
                lambda pos, e=entry: self._held_context_menu(e, canvas.mapToGlobal(pos))
            )
            if self._held_layout is not None:
                self._held_layout.addWidget(canvas)
            canvas.draw_idle()
        else:
            entry = self._superpose_entry
            ax = entry["fig"].get_axes()[0]
            self._plot_profile_labeled(ax, dist, values, label)
            self._apply_axis_settings(ax)
            ax.legend(fontsize=6, loc="best")
            entry["fig"].tight_layout()
            entry["_profiles"].append((dist, values, label))
            entry["_all_profiles"] = entry["_profiles"]
            entry["canvas"].draw_idle()

    def _on_hold(self) -> None:
        result = self._get_profile()
        if result is None:
            return
        dist, values = result
        label = self._frame_label_text(self._frame_idx)
        if self._chk_superpose.isChecked():
            self._on_hold_superpose(dist, values, label)
            return
        fw = self._spin_fig_w.value()
        fh = self._spin_fig_h.value()
        fig = Figure(figsize=(fw, fh), dpi=100)
        ax = fig.add_subplot(111)
        self._plot_profile(ax, dist, values)
        ax.set_title(label, fontsize=7)
        self._apply_axis_settings(ax)
        fig.tight_layout()
        canvas = FigureCanvasQTAgg(fig)
        canvas.setFixedSize(int(fw * 100), int(fh * 100))
        canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        entry = {"canvas": canvas, "dist": dist, "values": values,
                 "label": label, "fig": fig}
        self._held_figures.append(entry)
        canvas.customContextMenuRequested.connect(
            lambda pos, e=entry: self._held_context_menu(e, canvas.mapToGlobal(pos))
        )
        if self._held_layout is not None:
            self._held_layout.addWidget(canvas)
        canvas.draw_idle()

    def _held_context_menu(self, entry: dict, global_pos) -> None:
        menu = QMenu()
        act_edit = menu.addAction("Edit Properties...")
        menu.addSeparator()
        act_save = menu.addAction("Save PNG")
        act_csv = menu.addAction("Export CSV")
        menu.addSeparator()
        act_delete = menu.addAction("Delete")
        chosen = menu.exec(global_pos)
        if chosen == act_edit:
            from tools.figure_props_dialog import FigurePropsDialog
            dlg = FigurePropsDialog(entry, self._spin_dpi.value(), self._panel)
            dlg.exec()
        elif chosen == act_save:
            dpi = entry.get("_dpi", self._spin_dpi.value())
            path, _ = QFileDialog.getSaveFileName(
                self._panel, "Save Figure", f"{entry['label']}.png",
                "PNG Files (*.png)")
            if path:
                entry["fig"].savefig(path, dpi=dpi)
        elif chosen == act_csv:
            if "_all_profiles" in entry:
                path, _ = QFileDialog.getSaveFileName(
                    self._panel, "Export Profiles", "", "CSV Files (*.csv)")
                if path:
                    self._write_csv(path, entry["_all_profiles"])
            else:
                self._save_single_csv(entry["dist"], entry["values"])
        elif chosen == act_delete:
            if entry is self._superpose_entry:
                self._superpose_entry = None
            self._held_figures.remove(entry)
            canvas = entry["canvas"]
            canvas.customContextMenuRequested.disconnect()
            canvas.setParent(None)
            canvas.deleteLater()

    def _live_context_menu(self, pos) -> None:
        if not self._finalized or self._canvas is None:
            return
        menu = QMenu()
        act_save = menu.addAction("Save PNG")
        act_csv = menu.addAction("Export CSV")
        chosen = menu.exec(self._canvas.mapToGlobal(pos))
        if chosen == act_save:
            path, _ = QFileDialog.getSaveFileName(
                self._panel, "Save Figure", "spline_profile.png",
                "PNG Files (*.png)")
            if path:
                self._canvas.figure.savefig(path, dpi=self._spin_dpi.value())
        elif chosen == act_csv:
            result = self._get_profile()
            if result is not None:
                self._save_single_csv(result[0], result[1])

    # ---------------------------------------------------------------- all frames
    def _on_plot_all_frames(self) -> None:
        if not self._finalized or self._source is None:
            return
        n = self._source.frame_count
        if n == 0:
            return
        superpose = self._chk_superpose.isChecked()
        progress = QProgressDialog("Plotting all frames...", "Cancel", 0, n)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        for i in range(n):
            if progress.wasCanceled():
                break
            img = self._source.get_frame(i, copy=False)
            if img is None:
                progress.setValue(i + 1)
                continue
            result = self._get_profile_from(img)
            if result is None:
                progress.setValue(i + 1)
                continue
            dist, values = result
            label = self._frame_label_text(i)
            if superpose:
                self._on_hold_superpose(dist, values, label)
            else:
                fw = self._spin_fig_w.value()
                fh = self._spin_fig_h.value()
                fig = Figure(figsize=(fw, fh), dpi=100)
                ax = fig.add_subplot(111)
                self._plot_profile(ax, dist, values)
                ax.set_title(label, fontsize=7)
                self._apply_axis_settings(ax)
                fig.tight_layout()
                canvas = FigureCanvasQTAgg(fig)
                canvas.setFixedSize(int(fw * 100), int(fh * 100))
                canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                entry = {"canvas": canvas, "fig": fig, "label": label,
                         "dist": dist, "values": values}
                self._held_figures.append(entry)
                canvas.customContextMenuRequested.connect(
                    lambda pos, e=entry: self._held_context_menu(e, canvas.mapToGlobal(pos))
                )
                if self._held_layout is not None:
                    self._held_layout.addWidget(canvas)
                canvas.draw_idle()
            progress.setValue(i + 1)
            QApplication.processEvents()
        progress.close()

    # ---------------------------------------------------------------- save / clear
    def _on_save_all(self) -> None:
        if not self._held_figures:
            return
        from tools.line_tool import _SaveAllDialog
        dlg = _SaveAllDialog(len(self._held_figures), self._panel)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cfg = dlg.get_config()
        if cfg is None:
            return
        cfg["directory"].mkdir(parents=True, exist_ok=True)
        for i, entry in enumerate(self._held_figures):
            idx = cfg["start"] + i
            path = dlg.make_path(cfg, idx)
            entry["fig"].savefig(str(path), dpi=self._spin_dpi.value())
        if dlg.export_csv:
            csv_name = f"{cfg['prefix']}_all.csv" if cfg["prefix"] else "all.csv"
            self._save_multi_csv(cfg["directory"] / csv_name)

    def _on_clear_all(self) -> None:
        for entry in self._held_figures:
            canvas = entry["canvas"]
            canvas.customContextMenuRequested.disconnect()
            canvas.setParent(None)
            canvas.deleteLater()
        self._held_figures.clear()
        self._superpose_entry = None

    # ---------------------------------------------------------------- CSV
    def _save_single_csv(self, dist: np.ndarray, values: np.ndarray) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self._panel, "Export Spline Profile", "", "CSV Files (*.csv)")
        if not path:
            return
        self._write_csv(path, [(dist, values, None)])

    def _save_multi_csv(self, path: Path) -> None:
        profiles = []
        for entry in self._held_figures:
            if "_all_profiles" in entry:
                profiles.extend(entry["_all_profiles"])
            else:
                profiles.append((entry["dist"], entry["values"], entry["label"]))
        if profiles:
            self._write_csv(str(path), profiles)

    @staticmethod
    def _write_csv(path: str, profiles: list) -> None:
        if len(profiles) == 1:
            dist, values, _ = profiles[0]
            is_rgb = values.ndim == 2 and values.shape[1] == 3
            header = "distance,B,G,R" if is_rgb else "distance,value"
            with open(path, "w") as f:
                f.write(header + "\n")
                for j in range(len(dist)):
                    if is_rgb:
                        f.write(f"{dist[j]:.2f},{values[j, 0]},{values[j, 1]},{values[j, 2]}\n")
                    else:
                        val = values[j] if values.ndim == 1 else values[j, 0]
                        f.write(f"{dist[j]:.2f},{val}\n")
            return
        max_len = max(len(p[0]) for p in profiles)
        is_rgb = profiles[0][1].ndim == 2 and profiles[0][1].shape[1] == 3
        with open(path, "w") as f:
            cols = ["distance"]
            for k, (_, _, lbl) in enumerate(profiles):
                tag = lbl or f"profile_{k}"
                if is_rgb:
                    cols.extend([f"{tag}_B", f"{tag}_G", f"{tag}_R"])
                else:
                    cols.append(tag)
            f.write(",".join(cols) + "\n")
            for j in range(max_len):
                row = [f"{profiles[0][0][j]:.2f}" if j < len(profiles[0][0]) else ""]
                for dist, values, _ in profiles:
                    if j < len(dist):
                        if is_rgb:
                            row.extend([str(values[j, c]) for c in range(3)])
                        else:
                            row.append(str(values[j] if values.ndim == 1 else values[j, 0]))
                    else:
                        row.extend([""] * (3 if is_rgb else 1))
                f.write(",".join(row) + "\n")

    # ---------------------------------------------------------------- panel
    def build_panel(self) -> QWidget | None:
        from PySide6.QtWidgets import QSplitter

        self._panel = QWidget()
        root = QHBoxLayout(self._panel)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ======== Left ========
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)

        fig = Figure(figsize=(3.5, 2.0), dpi=100)
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setFixedSize(350, 200)
        self._canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._canvas.customContextMenuRequested.connect(self._live_context_menu)
        ll.addWidget(self._canvas)

        self._length_label = QLabel("Length: —")
        ll.addWidget(self._length_label)

        self._points_label = QLabel("Points: 0")
        self._points_label.setStyleSheet("color: gray;")
        ll.addWidget(self._points_label)

        info = QLabel("Click to add points. Enter to finalize. Backspace to undo.")
        info.setWordWrap(True)
        info.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        ll.addWidget(info)

        # -- Control Points --
        cp_grp = QGroupBox("Control Points")
        cp_grp.setCheckable(True)
        cp_grp.setChecked(False)
        cp_inner = QVBoxLayout(cp_grp)
        cp_scroll = QScrollArea()
        cp_scroll.setWidgetResizable(True)
        cp_scroll.setMaximumHeight(120)
        cp_scroll_w = QWidget()
        self._cp_layout = QVBoxLayout(cp_scroll_w)
        self._cp_layout.setContentsMargins(2, 2, 2, 2)
        self._cp_layout.setSpacing(2)
        cp_scroll.setWidget(cp_scroll_w)
        cp_inner.addWidget(cp_scroll)
        ll.addWidget(cp_grp)

        # -- Peak Markers --
        peak_row = QHBoxLayout()
        self._chk_peaks = QCheckBox("Show Peaks")
        peak_row.addWidget(self._chk_peaks)
        peak_row.addWidget(QLabel("Prominence:"))
        self._spin_prominence = QDoubleSpinBox()
        self._spin_prominence.setRange(0.1, 100000.0)
        self._spin_prominence.setValue(10.0)
        self._spin_prominence.setDecimals(1)
        self._spin_prominence.setSingleStep(1.0)
        self._spin_prominence.setFixedWidth(80)
        peak_row.addWidget(self._spin_prominence)
        peak_row.addStretch()
        ll.addLayout(peak_row)
        self._chk_peaks.toggled.connect(self._on_peak_toggled)
        self._spin_prominence.valueChanged.connect(self._on_peak_toggled)

        # -- Axis Limits --
        grp_axis = QGroupBox("Axis Limits")
        gv = QVBoxLayout(grp_axis)

        def make_limit_row(label, auto_default=True):
            chk = QCheckBox("Auto")
            chk.setChecked(auto_default)
            sp_min = QDoubleSpinBox()
            sp_min.setRange(-1e9, 1e9)
            sp_min.setDecimals(2)
            sp_min.setEnabled(not auto_default)
            sp_max = QDoubleSpinBox()
            sp_max.setRange(-1e9, 1e9)
            sp_max.setDecimals(2)
            sp_max.setValue(255)
            sp_max.setEnabled(not auto_default)
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(chk)
            row.addWidget(sp_min)
            row.addWidget(QLabel("–"))
            row.addWidget(sp_max)
            gv.addLayout(row)
            return chk, sp_min, sp_max

        self._chk_auto_x, self._spin_xmin, self._spin_xmax = make_limit_row("X:")
        self._chk_auto_y, self._spin_ymin, self._spin_ymax = make_limit_row("Y:")

        flip_row = QHBoxLayout()
        self._chk_flip_x = QCheckBox("Flip X")
        self._chk_flip_y = QCheckBox("Flip Y")
        flip_row.addWidget(self._chk_flip_x)
        flip_row.addWidget(self._chk_flip_y)
        flip_row.addStretch()
        gv.addLayout(flip_row)

        for w in (self._chk_auto_x, self._chk_auto_y, self._chk_flip_x, self._chk_flip_y):
            w.toggled.connect(self._on_setting_changed)
        for w in (self._spin_xmin, self._spin_xmax, self._spin_ymin, self._spin_ymax):
            w.valueChanged.connect(self._on_setting_changed)
        ll.addWidget(grp_axis)

        # -- Export Figure --
        grp_fig = QGroupBox("Export Figure")
        fig_lay = QVBoxLayout(grp_fig)
        fig_row = QHBoxLayout()
        fig_row.addWidget(QLabel("W:"))
        self._spin_fig_w = QDoubleSpinBox()
        self._spin_fig_w.setRange(2.0, 12.0)
        self._spin_fig_w.setValue(3.5)
        self._spin_fig_w.setSingleStep(0.5)
        self._spin_fig_w.setSuffix(" in")
        fig_row.addWidget(self._spin_fig_w)
        fig_row.addWidget(QLabel("H:"))
        self._spin_fig_h = QDoubleSpinBox()
        self._spin_fig_h.setRange(1.0, 8.0)
        self._spin_fig_h.setValue(2.0)
        self._spin_fig_h.setSingleStep(0.5)
        self._spin_fig_h.setSuffix(" in")
        fig_row.addWidget(self._spin_fig_h)
        fig_row.addWidget(QLabel("DPI:"))
        self._spin_dpi = QSpinBox()
        self._spin_dpi.setRange(72, 600)
        self._spin_dpi.setValue(150)
        self._spin_dpi.setSingleStep(50)
        fig_row.addWidget(self._spin_dpi)
        fig_lay.addLayout(fig_row)
        self._export_px_label = QLabel()
        self._export_px_label.setStyleSheet("color: gray; font-style: italic;")
        fig_lay.addWidget(self._export_px_label)
        self._spin_fig_w.valueChanged.connect(self._on_figsize_changed)
        self._spin_fig_h.valueChanged.connect(self._on_figsize_changed)
        self._spin_dpi.valueChanged.connect(self._update_export_label)
        self._update_export_label()
        ll.addWidget(grp_fig)

        self._chk_superpose = QCheckBox("Superposition")
        ll.addWidget(self._chk_superpose)

        btn_row = QHBoxLayout()
        btn_hold = QPushButton("Hold")
        btn_hold.clicked.connect(self._on_hold)
        btn_row.addWidget(btn_hold)
        btn_all = QPushButton("Hold All")
        btn_all.clicked.connect(self._on_plot_all_frames)
        btn_row.addWidget(btn_all)
        btn_save = QPushButton("Save All")
        btn_save.clicked.connect(self._on_save_all)
        btn_row.addWidget(btn_save)
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._on_clear_all)
        btn_row.addWidget(btn_clear)
        ll.addLayout(btn_row)

        ll.addStretch()
        left.setMinimumWidth(360)
        splitter.addWidget(left)

        # ======== Right ========
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.addWidget(QLabel("Held Figures"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_w = QWidget()
        self._held_layout = QVBoxLayout(scroll_w)
        self._held_layout.setContentsMargins(4, 4, 4, 4)
        self._held_layout.addStretch()
        scroll.setWidget(scroll_w)
        rl.addWidget(scroll)
        right.setMinimumWidth(360)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)
        return self._panel

    def activate(self) -> None:
        pass
