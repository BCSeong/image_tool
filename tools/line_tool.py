"""Line Profile Tool with Hold Figure & Save All."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QPen, QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsEllipseItem,
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


# ------------------------------------------------------------------ Save All
class _SaveAllDialog(QDialog):
    def __init__(self, count: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save All Figures")
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)

        row_dir = QHBoxLayout()
        row_dir.addWidget(QLabel("Directory:"))
        self._dir_edit = QLineEdit()
        row_dir.addWidget(self._dir_edit, stretch=1)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._browse)
        row_dir.addWidget(btn_browse)
        layout.addLayout(row_dir)

        form = QVBoxLayout()

        def add_row(label, widget):
            r = QHBoxLayout()
            r.addWidget(QLabel(label))
            r.addWidget(widget, stretch=1)
            form.addLayout(r)

        self._prefix = QLineEdit("line_profile")
        add_row("Prefix:", self._prefix)

        self._postfix = QLineEdit("")
        add_row("Postfix:", self._postfix)

        self._start_idx = QSpinBox()
        self._start_idx.setRange(0, 99999)
        self._start_idx.setValue(1)
        add_row("Start index:", self._start_idx)

        self._zero_pad = QSpinBox()
        self._zero_pad.setRange(1, 6)
        self._zero_pad.setValue(3)
        add_row("Zero padding:", self._zero_pad)

        layout.addLayout(form)

        self._chk_csv = QCheckBox("Export CSV (all profiles in one file)")
        layout.addWidget(self._chk_csv)

        self._preview = QLabel()
        self._preview.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self._preview)

        self._count = count
        for w in (self._prefix, self._postfix):
            w.textChanged.connect(self._refresh_preview)
        for w in (self._start_idx, self._zero_pad):
            w.valueChanged.connect(self._refresh_preview)
        self._refresh_preview()

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

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            self._dir_edit.setText(d)

    def _make_name(self, idx: int) -> str:
        pad = self._zero_pad.value()
        pre = self._prefix.text()
        post = self._postfix.text()
        num = f"{idx:0{pad}d}"
        sep = "_" if pre else ""
        return f"{pre}{sep}{num}{post}.png"

    def _refresh_preview(self) -> None:
        start = self._start_idx.value()
        first = self._make_name(start)
        if self._count > 1:
            last = self._make_name(start + self._count - 1)
            self._preview.setText(f"Preview: {first} … {last}")
        else:
            self._preview.setText(f"Preview: {first}")

    @property
    def export_csv(self) -> bool:
        return self._chk_csv.isChecked()

    def get_config(self) -> dict | None:
        d = self._dir_edit.text().strip()
        if not d:
            return None
        return {
            "directory": Path(d),
            "prefix": self._prefix.text(),
            "postfix": self._postfix.text(),
            "start": self._start_idx.value(),
            "pad": self._zero_pad.value(),
        }

    def make_path(self, cfg: dict, idx: int) -> Path:
        return cfg["directory"] / self._make_name(idx)


# ------------------------------------------------------------------ LineTool
class LineTool(BaseTool):

    _HIT_TOLERANCE = 25
    _HANDLE_HALF = 5

    def __init__(self, viewer: ImageViewer, source=None) -> None:
        self._viewer = viewer
        self._source = source
        self._x0 = self._y0 = self._x1 = self._y1 = 0
        self._drawing = False
        self._moving = False
        self._move_ox = self._move_oy = 0
        self._has_line = False
        self._line_item = None
        self._handle_items: list[QGraphicsEllipseItem] = []
        self._dragging_handle: int = -1
        self._frame_idx = 0
        self._panel: QWidget | None = None
        self._canvas: FigureCanvasQTAgg | None = None
        self._length_label: QLabel | None = None
        self._held_figures: list[dict] = []
        self._held_layout: QVBoxLayout | None = None
        self._superpose_entry: dict | None = None

    @property
    def name(self) -> str:
        return "Line"

    def _pen(self) -> QPen:
        pen = QPen(QColor(255, 255, 0), 2)
        pen.setCosmetic(True)
        return pen

    # -- Shift: 0/45/90 스냅 --
    def _snap_endpoint(self, x: int, y: int) -> tuple[int, int]:
        dx = x - self._x0
        dy = y - self._y0
        if dx == 0 and dy == 0:
            return x, y
        angle = math.atan2(dy, dx)
        snapped = round(angle / (math.pi / 4)) * (math.pi / 4)
        dist = math.hypot(dx, dy)
        return (self._x0 + int(round(dist * math.cos(snapped))),
                self._y0 + int(round(dist * math.sin(snapped))))

    # -- hit test --
    def _hit_test(self, x: int, y: int) -> bool:
        dx = self._x1 - self._x0
        dy = self._y1 - self._y0
        len_sq = dx * dx + dy * dy
        if len_sq < 1:
            return math.hypot(x - self._x0, y - self._y0) <= self._HIT_TOLERANCE
        t = max(0.0, min(1.0, ((x - self._x0) * dx + (y - self._y0) * dy) / len_sq))
        px = self._x0 + t * dx
        py = self._y0 + t * dy
        return math.hypot(x - px, y - py) <= self._HIT_TOLERANCE

    def _shift_all(self, dx: int, dy: int) -> None:
        self._x0 += dx
        self._y0 += dy
        self._x1 += dx
        self._y1 += dy

    # -- 마우스 --
    def on_mouse_press(self, x: int, y: int, event) -> bool:
        h_idx = self._handle_hit_test(x, y)
        if h_idx >= 0:
            self._dragging_handle = h_idx
            return True
        if self._has_line and self._hit_test(x, y):
            self._moving = True
            self._move_ox, self._move_oy = x, y
            return True
        self._moving = False
        self._x0, self._y0 = x, y
        self._x1, self._y1 = x, y
        self._drawing = True
        if self._line_item is not None:
            self._viewer.scene_ref.removeItem(self._line_item)
            self._line_item = None
        self._remove_handles()
        return True

    def on_mouse_move(self, x: int, y: int, event) -> None:
        if self._dragging_handle >= 0:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                if self._dragging_handle == 1:
                    x, y = self._snap_endpoint(x, y)
            if self._dragging_handle == 0:
                self._x0, self._y0 = x, y
            else:
                self._x1, self._y1 = x, y
            self._update_line()
            self._compute_and_display()
            return
        if self._moving:
            dx, dy = x - self._move_ox, y - self._move_oy
            self._shift_all(dx, dy)
            self._move_ox, self._move_oy = x, y
            self._update_line()
            self._compute_and_display()
            return
        if not self._drawing:
            return
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            x, y = self._snap_endpoint(x, y)
        self._x1, self._y1 = x, y
        self._update_line()
        self._compute_and_display()

    def on_mouse_release(self, x: int, y: int, event) -> None:
        if self._dragging_handle >= 0:
            self._dragging_handle = -1
            self._compute_and_display()
            return
        if self._moving:
            self._moving = False
            self._compute_and_display()
            return
        if not self._drawing:
            return
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            x, y = self._snap_endpoint(x, y)
        self._x1, self._y1 = x, y
        self._drawing = False
        self._has_line = True
        self._update_line()
        self._compute_and_display()

    def on_key_press(self, key: int, event) -> bool:
        if not self._has_line:
            return False
        dx, dy = 0, 0
        if key == Qt.Key.Key_Left:
            dx = -1
        elif key == Qt.Key.Key_Right:
            dx = 1
        elif key == Qt.Key.Key_Up:
            dy = -1
        elif key == Qt.Key.Key_Down:
            dy = 1
        else:
            return False
        self._shift_all(dx, dy)
        self._update_line()
        self._compute_and_display()
        return True

    def on_frame_changed(self, idx: int, image: np.ndarray | None) -> None:
        self._frame_idx = idx
        if self._has_line:
            self._compute_and_display()

    def deactivate(self) -> None:
        self._remove_handles()
        self._line_item = None
        self._has_line = False

    # -- 도형 --
    def _update_line(self) -> None:
        if self._line_item is None:
            self._line_item = self._viewer.scene_ref.addLine(
                self._x0, self._y0, self._x1, self._y1, self._pen()
            )
        else:
            self._line_item.setLine(self._x0, self._y0, self._x1, self._y1)
        self._update_handles()

    # -- 핸들 --
    def _scene_pixel_size(self) -> float:
        t = self._viewer.transform()
        return 1.0 / t.m11() if t.m11() != 0 else 1.0

    def _update_handles(self) -> None:
        if not self._has_line and not self._drawing:
            self._remove_handles()
            return
        positions = [(self._x0, self._y0), (self._x1, self._y1)]
        ps = self._scene_pixel_size()
        half = self._HANDLE_HALF * ps
        scene = self._viewer.scene_ref
        pen = QPen(QColor(255, 255, 0), 1)
        pen.setCosmetic(True)
        brush = QBrush(QColor(255, 255, 255))
        while len(self._handle_items) < 2:
            item = QGraphicsEllipseItem()
            item.setPen(pen)
            item.setBrush(brush)
            item.setZValue(100)
            scene.addItem(item)
            self._handle_items.append(item)
        for i, (cx, cy) in enumerate(positions):
            self._handle_items[i].setRect(cx - half, cy - half, half * 2, half * 2)
            self._handle_items[i].setVisible(True)

    def _remove_handles(self) -> None:
        scene = self._viewer.scene_ref
        for item in self._handle_items:
            scene.removeItem(item)
        self._handle_items.clear()

    def _handle_hit_test(self, x: int, y: int) -> int:
        if not self._has_line:
            return -1
        ps = self._scene_pixel_size()
        radius = (self._HANDLE_HALF + 2) * ps
        for i, (cx, cy) in enumerate([(self._x0, self._y0), (self._x1, self._y1)]):
            if math.hypot(x - cx, y - cy) <= radius:
                return i
        return -1

    # -- profile 계산 --
    def _get_profile(self) -> tuple[np.ndarray, np.ndarray] | None:
        img = self._viewer.raw_image
        if img is None:
            return None
        return self._get_profile_from(img)

    def _get_profile_from(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        dx = self._x1 - self._x0
        dy = self._y1 - self._y0
        length = np.hypot(dx, dy)
        if length < 1:
            return None
        n = int(np.ceil(length))
        xs = np.linspace(self._x0, self._x1, n).astype(int)
        ys = np.linspace(self._y0, self._y1, n).astype(int)
        h, w = img.shape[:2]
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys = xs[valid], ys[valid]
        dist = np.linspace(0, length, n)[valid]
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
        """axes를 재사용하여 데이터만 갱신."""
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
        length = np.hypot(self._x1 - self._x0, self._y1 - self._y0)

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
            self._length_label.setText(f"Length: {length:.1f} px")
        self._update_coord_spins()

    def _update_coord_spins(self) -> None:
        if not self._coord_spins:
            return
        for sp in self._coord_spins.values():
            sp.blockSignals(True)
        self._coord_spins["x0"].setValue(self._x0)
        self._coord_spins["y0"].setValue(self._y0)
        self._coord_spins["x1"].setValue(self._x1)
        self._coord_spins["y1"].setValue(self._y1)
        for sp in self._coord_spins.values():
            sp.blockSignals(False)

    def _on_coord_spin_changed(self) -> None:
        self._x0 = self._coord_spins["x0"].value()
        self._y0 = self._coord_spins["y0"].value()
        self._x1 = self._coord_spins["x1"].value()
        self._y1 = self._coord_spins["y1"].value()
        self._has_line = True
        self._update_line()
        self._compute_and_display()

    def _apply_axis_settings(self, ax) -> None:
        if self._chk_auto_x.isChecked():
            ax.set_xlim(auto=True)
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=False)
            xlo, xhi = ax.get_xlim()
            self._spin_xmin.blockSignals(True)
            self._spin_xmax.blockSignals(True)
            self._spin_xmin.setValue(xlo)
            self._spin_xmax.setValue(xhi)
            self._spin_xmin.blockSignals(False)
            self._spin_xmax.blockSignals(False)
        else:
            ax.set_xlim(self._spin_xmin.value(), self._spin_xmax.value())

        if self._chk_auto_y.isChecked():
            ax.set_ylim(auto=True)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            ylo, yhi = ax.get_ylim()
            self._spin_ymin.blockSignals(True)
            self._spin_ymax.blockSignals(True)
            self._spin_ymin.setValue(ylo)
            self._spin_ymax.setValue(yhi)
            self._spin_ymin.blockSignals(False)
            self._spin_ymax.blockSignals(False)
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
        if self._has_line:
            self._compute_and_display()

    def _on_figsize_changed(self) -> None:
        w = self._spin_fig_w.value()
        h = self._spin_fig_h.value()
        self._canvas.figure.set_size_inches(w, h, forward=False)
        self._canvas.setFixedSize(int(w * 100), int(h * 100))
        self._update_export_label()
        if self._has_line:
            self._compute_and_display()

    def _update_export_label(self) -> None:
        w = self._spin_fig_w.value()
        h = self._spin_fig_h.value()
        dpi = self._spin_dpi.value()
        px_w, px_h = int(w * dpi), int(h * dpi)
        self._export_px_label.setText(f"Export: {px_w} × {px_h} px")

    def _frame_label(self, idx: int) -> str:
        if self._source is not None:
            return self._source.frame_name(idx)
        return f"frame{idx}"

    # ---------------------------------------------------------------- Superpose
    def _plot_profile_labeled(self, ax, dist, values, label: str) -> None:
        if values.ndim == 2 and values.shape[1] == 3:
            colors = ax.plot(dist, values[:, 0], linewidth=0.8, alpha=0.8, label=label)[0].get_color()
            ax.plot(dist, values[:, 1], linewidth=0.8, alpha=0.8, color=colors, linestyle="--")
            ax.plot(dist, values[:, 2], linewidth=0.8, alpha=0.8, color=colors, linestyle=":")
        else:
            ax.plot(dist, values, linewidth=0.8, label=label)
        ax.set_xlabel("Distance (px)", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.tick_params(labelsize=7)

    def _on_hold_superpose(self, dist, values, label, coords) -> None:
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
                "dist": dist, "values": values, "coords": coords,
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
            fig = entry["fig"]
            ax = fig.get_axes()[0]
            self._plot_profile_labeled(ax, dist, values, label)
            self._apply_axis_settings(ax)
            ax.legend(fontsize=6, loc="best")
            fig.tight_layout()
            entry["_profiles"].append((dist, values, label))
            entry["_all_profiles"] = entry["_profiles"]
            entry["canvas"].draw_idle()

    # ---------------------------------------------------------------- Hold
    def _on_hold(self) -> None:
        result = self._get_profile()
        if result is None:
            return
        dist, values = result
        coords = (self._x0, self._y0, self._x1, self._y1)
        label = self._frame_label(self._frame_idx)

        if self._chk_superpose.isChecked():
            self._on_hold_superpose(dist, values, label, coords)
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
                 "label": label, "coords": coords, "fig": fig}
        self._held_figures.append(entry)

        canvas.customContextMenuRequested.connect(
            lambda pos, e=entry: self._held_context_menu(e, canvas.mapToGlobal(pos))
        )

        if self._held_layout is not None:
            self._held_layout.addWidget(canvas)
        canvas.draw_idle()

    def _live_context_menu(self, pos) -> None:
        if not self._has_line or self._canvas is None:
            return
        menu = QMenu()
        act_save = menu.addAction("Save PNG")
        act_csv = menu.addAction("Export CSV")
        chosen = menu.exec(self._canvas.mapToGlobal(pos))
        if chosen == act_save:
            path, _ = QFileDialog.getSaveFileName(
                self._panel, "Save Figure", "line_profile.png",
                "PNG Files (*.png)"
            )
            if path:
                self._canvas.figure.savefig(path, dpi=self._spin_dpi.value())
        elif chosen == act_csv:
            result = self._get_profile()
            if result is not None:
                self._save_single_csv(result[0], result[1])

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
                "PNG Files (*.png)"
            )
            if path:
                entry["fig"].savefig(path, dpi=dpi)
        elif chosen == act_csv:
            if "_all_profiles" in entry:
                path, _ = QFileDialog.getSaveFileName(
                    self._panel, "Export Line Profiles", "",
                    "CSV Files (*.csv)"
                )
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

    # ---------------------------------------------------------------- All Frames
    def _on_plot_all_frames(self) -> None:
        if not self._has_line or self._source is None:
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
            coords = (self._x0, self._y0, self._x1, self._y1)
            label = self._frame_label(i)

            if superpose:
                self._on_hold_superpose(dist, values, label, coords)
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

                entry = {
                    "canvas": canvas, "fig": fig, "label": label,
                    "dist": dist, "values": values, "coords": coords,
                }
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

    # ---------------------------------------------------------------- Save All
    def _on_save_all(self) -> None:
        if not self._held_figures:
            return
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
            csv_path = cfg["directory"] / csv_name
            self._save_multi_csv(csv_path)

    # ---------------------------------------------------------------- Clear All
    def _on_clear_all(self) -> None:
        for entry in self._held_figures:
            canvas = entry["canvas"]
            canvas.customContextMenuRequested.disconnect()
            canvas.setParent(None)
            canvas.deleteLater()
        self._held_figures.clear()
        self._superpose_entry = None

    # ---------------------------------------------------------------- 패널
    def build_panel(self) -> QWidget | None:
        from PySide6.QtWidgets import QSplitter

        self._panel = QWidget()
        root = QHBoxLayout(self._panel)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ======== Left: live view + controls ========
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        fig = Figure(figsize=(3.5, 2.0), dpi=100)
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setFixedSize(350, 200)
        self._canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._canvas.customContextMenuRequested.connect(self._live_context_menu)
        left_layout.addWidget(self._canvas)

        self._length_label = QLabel("Length: —")
        left_layout.addWidget(self._length_label)

        # -- Line Coordinates --
        from PySide6.QtWidgets import QFormLayout
        self._coord_spins: dict[str, QSpinBox] = {}
        coord_form = QFormLayout()
        for key in ("x0", "y0", "x1", "y1"):
            sp = QSpinBox()
            sp.setRange(-99999, 99999)
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(self._on_coord_spin_changed)
            self._coord_spins[key] = sp
            coord_form.addRow(f"{key.upper()}:", sp)
        left_layout.addLayout(coord_form)

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
        left_layout.addWidget(grp_axis)

        # -- Export Figure --
        grp_fig = QGroupBox("Export Figure")
        fig_grp_layout = QVBoxLayout(grp_fig)
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
        fig_grp_layout.addLayout(fig_row)
        self._export_px_label = QLabel()
        self._export_px_label.setStyleSheet("color: gray; font-style: italic;")
        fig_grp_layout.addWidget(self._export_px_label)
        self._spin_fig_w.valueChanged.connect(self._on_figsize_changed)
        self._spin_fig_h.valueChanged.connect(self._on_figsize_changed)
        self._spin_dpi.valueChanged.connect(self._update_export_label)
        self._update_export_label()
        left_layout.addWidget(grp_fig)

        self._chk_superpose = QCheckBox("Superposition")
        left_layout.addWidget(self._chk_superpose)

        # -- Buttons --
        btn_row = QHBoxLayout()
        btn_hold = QPushButton("Hold")
        btn_hold.clicked.connect(self._on_hold)
        btn_row.addWidget(btn_hold)
        btn_all_frames = QPushButton("Hold All")
        btn_all_frames.clicked.connect(self._on_plot_all_frames)
        btn_row.addWidget(btn_all_frames)
        btn_save_all = QPushButton("Save All")
        btn_save_all.clicked.connect(self._on_save_all)
        btn_row.addWidget(btn_save_all)
        btn_clear_all = QPushButton("Clear All")
        btn_clear_all.clicked.connect(self._on_clear_all)
        btn_row.addWidget(btn_clear_all)
        left_layout.addLayout(btn_row)

        left_layout.addStretch()
        left.setMinimumWidth(360)
        splitter.addWidget(left)

        # ======== Right: held figures (scroll) ========
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.addWidget(QLabel("Held Figures"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self._held_layout = QVBoxLayout(scroll_widget)
        self._held_layout.setContentsMargins(4, 4, 4, 4)
        self._held_layout.addStretch()
        scroll.setWidget(scroll_widget)
        right_layout.addWidget(scroll)
        right.setMinimumWidth(360)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)
        return self._panel

    def _save_single_csv(self, dist: np.ndarray, values: np.ndarray) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self._panel, "Export Line Profile", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        self._write_csv(path, [(dist, values, None)])

    def _save_multi_csv(self, path: Path) -> None:
        entries = [(e["dist"], e["values"], e["label"]) for e in self._held_figures]
        self._write_csv(str(path), entries)

    @staticmethod
    def _write_csv(path: str, profiles: list[tuple]) -> None:
        """profiles: list of (dist, values, label|None)."""
        is_rgb = any(v.ndim == 2 and v.shape[1] == 3 for _, v, _ in profiles)
        with open(path, "w") as f:
            if len(profiles) == 1:
                dist, values, _ = profiles[0]
                if is_rgb:
                    f.write("distance,B,G,R\n")
                    for i in range(len(dist)):
                        f.write(f"{dist[i]:.2f},{values[i,0]},{values[i,1]},{values[i,2]}\n")
                else:
                    f.write("distance,value\n")
                    for i in range(len(dist)):
                        f.write(f"{dist[i]:.2f},{values[i]}\n")
            else:
                max_len = max(len(d) for d, _, _ in profiles)
                headers = ["distance"]
                for idx, (_, _, label) in enumerate(profiles):
                    tag = label or f"profile_{idx}"
                    if is_rgb:
                        headers.extend([f"{tag}_B", f"{tag}_G", f"{tag}_R"])
                    else:
                        headers.append(tag)
                f.write(",".join(headers) + "\n")
                for row in range(max_len):
                    cells = []
                    dist0 = profiles[0][0]
                    cells.append(f"{dist0[row]:.2f}" if row < len(dist0) else "")
                    for dist, values, _ in profiles:
                        if row < len(dist):
                            if is_rgb:
                                cells.extend([str(values[row, 0]), str(values[row, 1]), str(values[row, 2])])
                            else:
                                cells.append(str(values[row]))
                        else:
                            cells.extend([""] * (3 if is_rgb else 1))
                    f.write(",".join(cells) + "\n")
