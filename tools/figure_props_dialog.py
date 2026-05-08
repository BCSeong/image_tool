"""Figure Properties Editor Dialog for held figures."""

from __future__ import annotations

import copy

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


_LINE_STYLES = [
    ("Solid", "-"),
    ("Dashed", "--"),
    ("Dotted", ":"),
    ("DashDot", "-."),
]

_LEGEND_LOCS = [
    ("Best", "best"),
    ("Upper Right", "upper right"),
    ("Upper Left", "upper left"),
    ("Lower Left", "lower left"),
    ("Lower Right", "lower right"),
    ("Right", "right"),
    ("Center Left", "center left"),
    ("Center Right", "center right"),
    ("Lower Center", "lower center"),
    ("Upper Center", "upper center"),
    ("Center", "center"),
]


class FigurePropsDialog(QDialog):

    def __init__(self, entry: dict, default_dpi: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Figure Properties")
        self.setMinimumWidth(500)
        self._entry = entry
        self._fig: Figure = entry["fig"]
        self._canvas: FigureCanvasQTAgg = entry["canvas"]
        self._ax = self._fig.get_axes()[0]
        self._dpi = entry.get("_dpi", default_dpi)
        self._line_rows: list[dict] = []
        self._preview_on = False

        self._snapshot = self._take_snapshot()

        self._build_ui()
        self._load_from_figure()
        self._connect_preview_signals()

    def _take_snapshot(self) -> dict:
        ax = self._ax
        fig = self._fig
        w_in, h_in = fig.get_size_inches()
        lines = []
        for line in ax.get_lines():
            c = line.get_color()
            lines.append({
                "label": line.get_label(),
                "color": c if isinstance(c, str) else tuple(c),
                "linewidth": line.get_linewidth(),
                "linestyle": line.get_linestyle(),
                "visible": line.get_visible(),
            })
        legend = ax.get_legend()
        leg_snap = {
            "visible": legend is not None and legend.get_visible(),
            "fontsize": 6,
            "loc": "best",
            "title": "",
            "title_fs": 6,
            "frameon": True,
            "framealpha": 0.8,
            "ncol": 1,
        }
        if legend is not None:
            texts = legend.get_texts()
            if texts:
                leg_snap["fontsize"] = texts[0].get_fontsize()
            leg_snap["loc"] = legend._loc  # stored int code
            t = legend.get_title()
            leg_snap["title"] = t.get_text() if t else ""
            leg_snap["title_fs"] = t.get_fontsize() if t else 6
            leg_snap["frameon"] = legend.get_frame_on()
            frame = legend.get_frame()
            leg_snap["framealpha"] = frame.get_alpha() if frame.get_alpha() is not None else 0.8
            ncol = getattr(legend, '_ncols', getattr(legend, '_ncol', 1))
            leg_snap["ncol"] = ncol
        return {
            "title": ax.get_title(),
            "title_fs": ax.title.get_fontsize(),
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "label_fs": ax.xaxis.label.get_fontsize(),
            "tick_fs": ax.xaxis.get_ticklabels()[0].get_fontsize() if ax.xaxis.get_ticklabels() else 7,
            "xlim": ax.get_xlim(),
            "ylim": ax.get_ylim(),
            "w_in": w_in,
            "h_in": h_in,
            "canvas_size": (self._canvas.width(), self._canvas.height()),
            "dpi": self._dpi,
            "lines": lines,
            "legend": leg_snap,
        }

    def _restore_snapshot(self) -> None:
        s = self._snapshot
        ax = self._ax
        fig = self._fig

        ax.set_title(s["title"], fontsize=s["title_fs"])
        ax.set_xlabel(s["xlabel"], fontsize=s["label_fs"])
        ax.set_ylabel(s["ylabel"], fontsize=s["label_fs"])
        ax.tick_params(labelsize=s["tick_fs"])
        ax.set_xlim(s["xlim"])
        ax.set_ylim(s["ylim"])

        for line, saved in zip(ax.get_lines(), s["lines"]):
            line.set_label(saved["label"])
            line.set_color(saved["color"])
            line.set_linewidth(saved["linewidth"])
            line.set_linestyle(saved["linestyle"])
            line.set_visible(saved["visible"])

        legend = ax.get_legend()
        if legend:
            legend.remove()
        ls = s["legend"]
        if ls["visible"]:
            ax.legend(
                fontsize=ls["fontsize"], loc=ls["loc"], ncol=ls["ncol"],
                frameon=ls["frameon"], framealpha=ls["framealpha"],
                title=ls["title"], title_fontsize=ls["title_fs"],
            )

        fig.set_size_inches(s["w_in"], s["h_in"], forward=False)
        cw, ch = s["canvas_size"]
        self._canvas.setFixedSize(cw, ch)
        self._entry["_dpi"] = s["dpi"]

        fig.tight_layout()
        self._canvas.draw_idle()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        tabs = QTabWidget()

        tabs.addTab(self._build_figure_tab(), "Figure")
        tabs.addTab(self._build_axes_tab(), "Axes")
        tabs.addTab(self._build_lines_tab(), "Lines")
        tabs.addTab(self._build_legend_tab(), "Legend")

        root.addWidget(tabs)

        btn_row = QHBoxLayout()
        self._chk_preview = QCheckBox("Preview")
        btn_row.addWidget(self._chk_preview)
        btn_row.addStretch()
        btn_ok = QPushButton("OK")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self._on_ok)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        root.addLayout(btn_row)

    def _connect_preview_signals(self) -> None:
        self._chk_preview.toggled.connect(self._on_preview_toggled)

        for w in (self._edit_title, self._edit_xlabel, self._edit_ylabel):
            w.textChanged.connect(self._on_value_changed)
        for w in (self._spin_title_fs, self._spin_label_fs, self._spin_tick_fs,
                  self._spin_dpi):
            w.valueChanged.connect(self._on_value_changed)
        for w in (self._spin_w, self._spin_h,
                  self._spin_xmin, self._spin_xmax, self._spin_ymin, self._spin_ymax):
            w.valueChanged.connect(self._on_value_changed)
        for w in (self._chk_auto_x, self._chk_auto_y,
                  self._chk_legend_visible, self._chk_legend_frame):
            w.toggled.connect(self._on_value_changed)
        for w in (self._spin_legend_fs, self._spin_legend_ncol):
            w.valueChanged.connect(self._on_value_changed)
        self._spin_legend_alpha.valueChanged.connect(self._on_value_changed)
        self._combo_legend_loc.currentIndexChanged.connect(self._on_value_changed)

    def _connect_line_row_signals(self, row_data: dict) -> None:
        row_data["label"].textChanged.connect(self._on_value_changed)
        row_data["linewidth"].valueChanged.connect(self._on_value_changed)
        row_data["linestyle"].currentIndexChanged.connect(self._on_value_changed)
        row_data["visible"].toggled.connect(self._on_value_changed)

    def _on_preview_toggled(self, checked: bool) -> None:
        self._preview_on = checked
        if checked:
            self._apply()
        else:
            self._restore_snapshot()

    def _on_value_changed(self) -> None:
        if self._preview_on:
            self._apply()

    def _build_figure_tab(self) -> QWidget:
        w = QWidget()
        form = QGridLayout(w)
        row = 0

        form.addWidget(QLabel("Title:"), row, 0)
        self._edit_title = QLineEdit()
        form.addWidget(self._edit_title, row, 1, 1, 3)
        row += 1

        form.addWidget(QLabel("Title font size:"), row, 0)
        self._spin_title_fs = QSpinBox()
        self._spin_title_fs.setRange(4, 30)
        self._spin_title_fs.setValue(7)
        form.addWidget(self._spin_title_fs, row, 1)
        row += 1

        form.addWidget(QLabel("Width (in):"), row, 0)
        self._spin_w = QDoubleSpinBox()
        self._spin_w.setRange(2.0, 20.0)
        self._spin_w.setSingleStep(0.5)
        form.addWidget(self._spin_w, row, 1)

        form.addWidget(QLabel("Height (in):"), row, 2)
        self._spin_h = QDoubleSpinBox()
        self._spin_h.setRange(1.0, 15.0)
        self._spin_h.setSingleStep(0.5)
        form.addWidget(self._spin_h, row, 3)
        row += 1

        form.addWidget(QLabel("Export DPI:"), row, 0)
        self._spin_dpi = QSpinBox()
        self._spin_dpi.setRange(72, 600)
        self._spin_dpi.setSingleStep(50)
        form.addWidget(self._spin_dpi, row, 1)

        self._lbl_px = QLabel()
        self._lbl_px.setStyleSheet("color: gray; font-style: italic;")
        form.addWidget(self._lbl_px, row, 2, 1, 2)
        row += 1

        self._spin_w.valueChanged.connect(self._update_px_label)
        self._spin_h.valueChanged.connect(self._update_px_label)
        self._spin_dpi.valueChanged.connect(self._update_px_label)

        form.setRowStretch(row, 1)
        return w

    def _build_axes_tab(self) -> QWidget:
        w = QWidget()
        form = QGridLayout(w)
        row = 0

        form.addWidget(QLabel("X label:"), row, 0)
        self._edit_xlabel = QLineEdit()
        form.addWidget(self._edit_xlabel, row, 1, 1, 3)
        row += 1

        form.addWidget(QLabel("Y label:"), row, 0)
        self._edit_ylabel = QLineEdit()
        form.addWidget(self._edit_ylabel, row, 1, 1, 3)
        row += 1

        form.addWidget(QLabel("Label font size:"), row, 0)
        self._spin_label_fs = QSpinBox()
        self._spin_label_fs.setRange(4, 24)
        form.addWidget(self._spin_label_fs, row, 1)

        form.addWidget(QLabel("Tick font size:"), row, 2)
        self._spin_tick_fs = QSpinBox()
        self._spin_tick_fs.setRange(4, 24)
        form.addWidget(self._spin_tick_fs, row, 3)
        row += 1

        form.addWidget(QLabel("X range:"), row, 0)
        self._chk_auto_x = QCheckBox("Auto")
        form.addWidget(self._chk_auto_x, row, 1)
        self._spin_xmin = QDoubleSpinBox()
        self._spin_xmin.setRange(-1e9, 1e9)
        self._spin_xmin.setDecimals(2)
        form.addWidget(self._spin_xmin, row, 2)
        self._spin_xmax = QDoubleSpinBox()
        self._spin_xmax.setRange(-1e9, 1e9)
        self._spin_xmax.setDecimals(2)
        form.addWidget(self._spin_xmax, row, 3)
        row += 1

        form.addWidget(QLabel("Y range:"), row, 0)
        self._chk_auto_y = QCheckBox("Auto")
        form.addWidget(self._chk_auto_y, row, 1)
        self._spin_ymin = QDoubleSpinBox()
        self._spin_ymin.setRange(-1e9, 1e9)
        self._spin_ymin.setDecimals(2)
        form.addWidget(self._spin_ymin, row, 2)
        self._spin_ymax = QDoubleSpinBox()
        self._spin_ymax.setRange(-1e9, 1e9)
        self._spin_ymax.setDecimals(2)
        form.addWidget(self._spin_ymax, row, 3)
        row += 1

        self._chk_auto_x.toggled.connect(lambda c: (
            self._spin_xmin.setEnabled(not c), self._spin_xmax.setEnabled(not c)
        ))
        self._chk_auto_y.toggled.connect(lambda c: (
            self._spin_ymin.setEnabled(not c), self._spin_ymax.setEnabled(not c)
        ))

        form.setRowStretch(row, 1)
        return w

    def _build_lines_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self._lines_layout = QVBoxLayout(inner)
        self._lines_layout.setContentsMargins(4, 4, 4, 4)
        self._lines_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        return w

    def _build_legend_tab(self) -> QWidget:
        w = QWidget()
        form = QGridLayout(w)
        row = 0

        self._chk_legend_visible = QCheckBox("Show Legend")
        form.addWidget(self._chk_legend_visible, row, 0, 1, 2)
        row += 1

        form.addWidget(QLabel("Location:"), row, 0)
        self._combo_legend_loc = QComboBox()
        for name, code in _LEGEND_LOCS:
            self._combo_legend_loc.addItem(name, code)
        form.addWidget(self._combo_legend_loc, row, 1, 1, 3)
        row += 1

        form.addWidget(QLabel("Font size:"), row, 0)
        self._spin_legend_fs = QSpinBox()
        self._spin_legend_fs.setRange(4, 24)
        self._spin_legend_fs.setValue(6)
        form.addWidget(self._spin_legend_fs, row, 1)

        form.addWidget(QLabel("Columns:"), row, 2)
        self._spin_legend_ncol = QSpinBox()
        self._spin_legend_ncol.setRange(1, 10)
        self._spin_legend_ncol.setValue(1)
        form.addWidget(self._spin_legend_ncol, row, 3)
        row += 1

        # form.addWidget(QLabel("Title:"), row, 0)
        # self._edit_legend_title = QLineEdit()
        # form.addWidget(self._edit_legend_title, row, 1, 1, 3)
        # row += 1

        # form.addWidget(QLabel("Title font size:"), row, 0)
        # self._spin_legend_title_fs = QSpinBox()
        # self._spin_legend_title_fs.setRange(4, 24)
        # self._spin_legend_title_fs.setValue(6)
        # form.addWidget(self._spin_legend_title_fs, row, 1)
        # row += 1

        self._chk_legend_frame = QCheckBox("Show Frame")
        self._chk_legend_frame.setChecked(True)
        form.addWidget(self._chk_legend_frame, row, 0, 1, 2)

        form.addWidget(QLabel("Frame alpha:"), row, 2)
        self._spin_legend_alpha = QDoubleSpinBox()
        self._spin_legend_alpha.setRange(0.0, 1.0)
        self._spin_legend_alpha.setSingleStep(0.1)
        self._spin_legend_alpha.setDecimals(2)
        self._spin_legend_alpha.setValue(0.8)
        form.addWidget(self._spin_legend_alpha, row, 3)
        row += 1

        self._chk_legend_visible.toggled.connect(
            lambda c: self._set_legend_controls_enabled(c)
        )

        form.setRowStretch(row, 1)
        return w

    def _set_legend_controls_enabled(self, enabled: bool) -> None:
        for w in (self._combo_legend_loc, self._spin_legend_fs,
                  self._spin_legend_ncol, self._chk_legend_frame,
                  self._spin_legend_alpha):
            w.setEnabled(enabled)

    def _load_from_figure(self) -> None:
        ax = self._ax
        fig = self._fig

        self._edit_title.setText(ax.get_title())
        title_fs = ax.title.get_fontsize()
        self._spin_title_fs.setValue(int(title_fs) if title_fs else 7)

        w_in, h_in = fig.get_size_inches()
        self._spin_w.setValue(w_in)
        self._spin_h.setValue(h_in)
        self._spin_dpi.setValue(self._dpi)
        self._update_px_label()

        self._edit_xlabel.setText(ax.get_xlabel())
        self._edit_ylabel.setText(ax.get_ylabel())

        label_fs = ax.xaxis.label.get_fontsize()
        self._spin_label_fs.setValue(int(label_fs) if label_fs else 8)
        tick_fs = ax.xaxis.get_ticklabels()[0].get_fontsize() if ax.xaxis.get_ticklabels() else 7
        self._spin_tick_fs.setValue(int(tick_fs))

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        self._spin_xmin.setValue(xmin)
        self._spin_xmax.setValue(xmax)
        self._spin_ymin.setValue(ymin)
        self._spin_ymax.setValue(ymax)
        self._chk_auto_x.setChecked(False)
        self._chk_auto_y.setChecked(False)

        for line in ax.get_lines():
            self._add_line_row(line)

        ls = self._snapshot["legend"]
        self._chk_legend_visible.setChecked(ls["visible"])
        self._set_legend_controls_enabled(ls["visible"])
        loc_str = ls["loc"]
        if isinstance(loc_str, int):
            loc_codes = {0: "best", 1: "upper right", 2: "upper left",
                         3: "lower left", 4: "lower right", 5: "right",
                         6: "center left", 7: "center right", 8: "lower center",
                         9: "upper center", 10: "center"}
            loc_str = loc_codes.get(loc_str, "best")
        for i in range(self._combo_legend_loc.count()):
            if self._combo_legend_loc.itemData(i) == loc_str:
                self._combo_legend_loc.setCurrentIndex(i)
                break
        self._spin_legend_fs.setValue(int(ls["fontsize"]))
        self._spin_legend_ncol.setValue(ls["ncol"])
        # self._edit_legend_title.setText(ls["title"])
        # self._spin_legend_title_fs.setValue(int(ls["title_fs"]))
        self._chk_legend_frame.setChecked(ls["frameon"])
        self._spin_legend_alpha.setValue(ls["framealpha"])

    def _add_line_row(self, line) -> None:
        grp = QGroupBox()
        grid = QGridLayout(grp)

        lbl_edit = QLineEdit(line.get_label() or "")
        grid.addWidget(QLabel("Label:"), 0, 0)
        grid.addWidget(lbl_edit, 0, 1, 1, 3)

        c = line.get_color()
        color = QColor(c) if isinstance(c, str) else QColor(*[int(v * 255) for v in c[:3]])
        btn_color = QPushButton()
        btn_color.setFixedSize(24, 24)
        btn_color.setStyleSheet(f"background-color: {color.name()};")
        btn_color._color = color
        btn_color.clicked.connect(lambda _, b=btn_color: self._pick_color(b))
        grid.addWidget(QLabel("Color:"), 1, 0)
        grid.addWidget(btn_color, 1, 1)

        spin_lw = QDoubleSpinBox()
        spin_lw.setRange(0.1, 10.0)
        spin_lw.setSingleStep(0.5)
        spin_lw.setValue(line.get_linewidth())
        grid.addWidget(QLabel("Width:"), 1, 2)
        grid.addWidget(spin_lw, 1, 3)

        combo_ls = QComboBox()
        current_ls = line.get_linestyle()
        for i, (name, code) in enumerate(_LINE_STYLES):
            combo_ls.addItem(name, code)
            if code == current_ls:
                combo_ls.setCurrentIndex(i)
        grid.addWidget(QLabel("Style:"), 2, 0)
        grid.addWidget(combo_ls, 2, 1)

        chk_visible = QCheckBox("Visible")
        chk_visible.setChecked(line.get_visible())
        grid.addWidget(chk_visible, 2, 2, 1, 2)

        row_data = {
            "line": line,
            "label": lbl_edit,
            "color_btn": btn_color,
            "linewidth": spin_lw,
            "linestyle": combo_ls,
            "visible": chk_visible,
        }
        self._line_rows.append(row_data)
        self._lines_layout.insertWidget(self._lines_layout.count() - 1, grp)
        self._connect_line_row_signals(row_data)

    def _pick_color(self, btn: QPushButton) -> None:
        color = QColorDialog.getColor(btn._color, self, "Line Color")
        if color.isValid():
            btn._color = color
            btn.setStyleSheet(f"background-color: {color.name()};")
            self._on_value_changed()

    def _update_px_label(self) -> None:
        w = self._spin_w.value()
        h = self._spin_h.value()
        dpi = self._spin_dpi.value()
        self._lbl_px.setText(f"Export: {int(w * dpi)} × {int(h * dpi)} px")

    def _on_ok(self) -> None:
        self._apply()
        self.accept()

    def _on_cancel(self) -> None:
        self._restore_snapshot()
        self.reject()

    def _apply(self) -> None:
        ax = self._ax
        fig = self._fig

        title_fs = self._spin_title_fs.value()
        ax.set_title(self._edit_title.text(), fontsize=title_fs)

        label_fs = self._spin_label_fs.value()
        ax.set_xlabel(self._edit_xlabel.text(), fontsize=label_fs)
        ax.set_ylabel(self._edit_ylabel.text(), fontsize=label_fs)
        ax.tick_params(labelsize=self._spin_tick_fs.value())

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

        for row in self._line_rows:
            line = row["line"]
            line.set_label(row["label"].text())
            line.set_color(row["color_btn"]._color.name())
            line.set_linewidth(row["linewidth"].value())
            line.set_linestyle(row["linestyle"].currentData())
            line.set_visible(row["visible"].isChecked())

        legend = ax.get_legend()
        if legend:
            legend.remove()
        if self._chk_legend_visible.isChecked():
            ax.legend(
                fontsize=self._spin_legend_fs.value(),
                loc=self._combo_legend_loc.currentData(),
                ncol=self._spin_legend_ncol.value(),
                frameon=self._chk_legend_frame.isChecked(),
                framealpha=self._spin_legend_alpha.value(),
                # title=self._edit_legend_title.text() or None,
                # title_fontsize=self._spin_legend_title_fs.value(),
            )

        w = self._spin_w.value()
        h = self._spin_h.value()
        fig.set_size_inches(w, h, forward=False)
        self._canvas.setFixedSize(int(w * 100), int(h * 100))

        self._entry["_dpi"] = self._spin_dpi.value()

        fig.tight_layout()
        self._canvas.draw_idle()
