"""MainWindow: 메뉴바, 툴바, 중앙 뷰어, 프레임 슬라이더, 도킹 패널, 상태바."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QSlider,
    QSpinBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import cv2
import tifffile

from batch_crop_dialog import BatchCropDialog
from enhanced_dock import EnhancedDockWidget
from plugins import ALL_PLUGINS
from bc_dialog import BCDialog
from crop_dialog import CropDialog
from debayer_widget import DebayerWidget
from frame_offset_widget import FrameOffsetWidget
from image_matching_widget import ImageMatchingWidget
from folder_wizard import FolderWizard
from image_source import ImageSource
from save_sequence_dialog import SaveSequenceDialog
from measure_widget import MeasureWidget
from tool_base import BaseTool
from tools.roi_base import RoiToolBase
from tools.rect_tool import RectTool
from tools.ellipse_tool import EllipseTool
from tools.line_tool import LineTool
from viewer import ImageViewer, _normalize_to_u8


class MainWindow(QMainWindow):
    _open_windows: list[MainWindow] = []

    def __init__(self, source: ImageSource | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Image Tool")
        self.resize(1400, 900)

        self._source = source if source is not None else ImageSource()
        self._tools: list[BaseTool] = []
        self._active_tool: BaseTool | None = None
        self._tool_actions: list[QAction] = []
        self._bc_widget: BCDialog | None = None
        self._debayer_widget: DebayerWidget | None = None
        self._offset_widget: FrameOffsetWidget | None = None

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_dock()
        self._connect_signals()
        self._register_tools()

        if self._source.is_loaded:
            self._after_load(self._source.frame_count)

    @staticmethod
    def _new_window() -> MainWindow:
        win = MainWindow()
        MainWindow._open_windows.append(win)
        win.show()
        return win

    @staticmethod
    def open_stack_window(stack: np.ndarray, title: str,
                          names: list[str] | None = None) -> MainWindow:
        source = ImageSource()
        source.load_array(stack, title, names=names)
        win = MainWindow(source)
        win.setWindowTitle(f"Image Tool - {title}")
        MainWindow._open_windows.append(win)
        win.show()
        return win

    def closeEvent(self, event) -> None:
        if self in MainWindow._open_windows:
            MainWindow._open_windows.remove(self)
        super().closeEvent(event)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._viewer = ImageViewer()
        layout.addWidget(self._viewer, stretch=1)

        # 하단: 프레임 슬라이더 + SpinBox
        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(8, 2, 8, 2)
        self._frame_spin = QSpinBox()
        self._frame_spin.setMinimum(0)
        self._frame_spin.setMaximum(0)
        slider_row.addWidget(self._frame_spin)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._frame_label = QLabel("No image")
        slider_row.addWidget(self._slider, stretch=1)
        slider_row.addWidget(self._frame_label)
        layout.addLayout(slider_row)

        # 상태바
        self._status_coord = QLabel("")
        self._status_info = QLabel("")
        self.statusBar().addWidget(self._status_coord, stretch=1)
        self.statusBar().addPermanentWidget(self._status_info)

    def _build_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")

        act_new_window = QAction("&New Window", self)
        act_new_window.setShortcut(QKeySequence("Ctrl+N"))
        act_new_window.triggered.connect(self._new_window)
        file_menu.addAction(act_new_window)

        file_menu.addSeparator()

        act_open_folder = QAction("Open &Folder...", self)
        act_open_folder.setShortcut(QKeySequence("Ctrl+O"))
        act_open_folder.triggered.connect(self._open_folder)
        file_menu.addAction(act_open_folder)

        act_open_tiff = QAction("Open &TIFF Stack...", self)
        act_open_tiff.setShortcut(QKeySequence("Ctrl+T"))
        act_open_tiff.triggered.connect(self._open_tiff)
        file_menu.addAction(act_open_tiff)

        file_menu.addSeparator()

        act_save = QAction("&Save", self)
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self._on_save)
        file_menu.addAction(act_save)

        act_save_as = QAction("Save &As...", self)
        act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act_save_as.triggered.connect(self._on_save_as)
        file_menu.addAction(act_save_as)

        act_save_seq = QAction("Save as Se&quence...", self)
        act_save_seq.triggered.connect(self._on_save_sequence)
        file_menu.addAction(act_save_seq)

        file_menu.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        image_menu = menu.addMenu("&Image")

        act_crop = QAction("&Crop...", self)
        act_crop.triggered.connect(self._on_crop)
        image_menu.addAction(act_crop)

        image_menu.addSeparator()
        adjust_menu = image_menu.addMenu("&Adjust")

        act_bc = QAction("&Brightness/Contrast...", self)
        act_bc.setShortcut(QKeySequence("Ctrl+Shift+C"))
        act_bc.triggered.connect(self._open_bc_dialog)
        adjust_menu.addAction(act_bc)

        act_demosaic = QAction("&Demosaic...", self)
        act_demosaic.triggered.connect(self._open_demosaic)
        adjust_menu.addAction(act_demosaic)

        image_menu.addSeparator()
        act_offset = QAction("Frame &Offset...", self)
        act_offset.triggered.connect(self._open_offset)
        image_menu.addAction(act_offset)

        act_matching = QAction("Image &Matching...", self)
        act_matching.triggered.connect(self._open_matching)
        image_menu.addAction(act_matching)

        analyze_menu = menu.addMenu("&Analyze")

        act_measure = QAction("&Measure", self)
        act_measure.setShortcut(QKeySequence("Ctrl+M"))
        act_measure.triggered.connect(self._on_measure)
        analyze_menu.addAction(act_measure)

        act_measure_all = QAction("Measure Through &Frames", self)
        act_measure_all.setShortcut(QKeySequence("Ctrl+Shift+M"))
        act_measure_all.triggered.connect(self._on_measure_all_frames)
        analyze_menu.addAction(act_measure_all)

        analyze_menu.addSeparator()
        act_batch_crop = QAction("&Batch Crop...", self)
        act_batch_crop.triggered.connect(self._on_batch_crop)
        analyze_menu.addAction(act_batch_crop)

        analyze_menu.addSeparator()
        act_select_all = QAction("Select &All", self)
        act_select_all.setShortcut(QKeySequence("Ctrl+A"))
        act_select_all.triggered.connect(self._on_select_all)
        analyze_menu.addAction(act_select_all)

        plugin_menu = menu.addMenu("&Plugins")
        for plugin in ALL_PLUGINS:
            act = QAction(plugin.name, self)
            act.triggered.connect(
                lambda _, p=plugin: p.run(self._source, self._slider.value(), self)
            )
            plugin_menu.addAction(act)

        view_menu = menu.addMenu("&View")

        act_fit = QAction("Fit to &Window", self)
        act_fit.setShortcut(QKeySequence("Ctrl+0"))
        act_fit.triggered.connect(self._viewer.fit_view)
        view_menu.addAction(act_fit)

    def _build_toolbar(self) -> None:
        self._toolbar = QToolBar("Tools")
        self._toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._toolbar)
        self._tool_action_group = QActionGroup(self)
        self._tool_action_group.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.ExclusiveOptional
        )

    def _build_dock(self) -> None:
        self._dock = EnhancedDockWidget("Tool Settings", self)
        self._dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self._dock.setWidget(QWidget())
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock)
        self._dock.hide()

        self._bc_dock = EnhancedDockWidget("Brightness / Contrast", self)
        self._bc_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._bc_dock)
        self._bc_dock.hide()
        self._bc_dock.visibilityChanged.connect(self._on_bc_visibility)

        self._demosaic_dock = EnhancedDockWidget("Demosaic", self)
        self._demosaic_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._demosaic_dock)
        self._demosaic_dock.hide()
        self._demosaic_dock.visibilityChanged.connect(self._on_demosaic_visibility)

        self._offset_dock = EnhancedDockWidget("Frame Offset", self)
        self._offset_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._offset_dock)
        self._offset_dock.hide()
        self._offset_dock.visibilityChanged.connect(self._on_offset_visibility)

        self._matching_widget: ImageMatchingWidget | None = None
        self._matching_dock = EnhancedDockWidget("Image Matching", self)
        self._matching_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._matching_dock)
        self._matching_dock.hide()
        self._matching_dock.visibilityChanged.connect(self._on_matching_visibility)

        self._measure_widget = MeasureWidget()
        self._measure_dock = EnhancedDockWidget("Measure", self)
        self._measure_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self._measure_dock.setWidget(self._measure_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._measure_dock)
        self._measure_dock.hide()

    def _connect_signals(self) -> None:
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        self._viewer.mouse_moved.connect(self._on_mouse_moved)
        self._viewer.frame_scroll.connect(self._on_frame_scroll)
        self._viewer.right_clicked.connect(self._on_right_click)
        self._viewer.escape_pressed.connect(self._deselect_tool)

    def _register_tools(self) -> None:
        self.register_tool(RectTool(self._viewer))
        self.register_tool(EllipseTool(self._viewer))
        self.register_tool(LineTool(self._viewer, self._source))

    # ------------------------------------------------------------------ Tool 등록
    def register_tool(self, tool: BaseTool) -> None:
        """tool을 툴바에 등록."""
        self._tools.append(tool)
        action = QAction(tool.name, self)
        action.setCheckable(True)
        self._tool_action_group.addAction(action)
        self._toolbar.addAction(action)

        idx = len(self._tools) - 1
        action.triggered.connect(lambda checked, i=idx: self._on_tool_selected(i, checked))
        self._tool_actions.append(action)

    def _deselect_tool(self) -> None:
        if self._active_tool:
            self._active_tool.deactivate()
            self._viewer.clear_overlays()
            self._active_tool = None
            self._viewer.set_active_tool(None)
            self._dock.hide()
            for action in self._tool_actions:
                action.setChecked(False)

    def _on_tool_selected(self, idx: int, checked: bool) -> None:
        if self._active_tool:
            self._active_tool.deactivate()
            self._viewer.clear_overlays()

        if checked and 0 <= idx < len(self._tools):
            tool = self._tools[idx]
            self._active_tool = tool
            self._viewer.set_active_tool(tool)
            tool.activate()
            panel = tool.build_panel()
            if panel:
                self._dock.setWidget(panel)
                self._dock.setWindowTitle(tool.name)
                self._dock.show()
            else:
                self._dock.hide()
        else:
            self._active_tool = None
            self._viewer.set_active_tool(None)
            self._dock.hide()

    # ------------------------------------------------------------------ File 열기
    def _open_folder(self) -> None:
        wizard = FolderWizard(self)
        if wizard.exec() != FolderWizard.DialogCode.Accepted:
            return
        config = wizard.get_config()
        if config is None or not config.matched_paths:
            QMessageBox.warning(self, "Warning", "No images matched the filter.")
            return
        n = self._source.load_paths(config.matched_paths, config.folder)
        self._after_load(n)

    def _open_tiff(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF Stack", "", "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if not path:
            return
        try:
            n = self._source.load_tiff_stack(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        self._after_load(n)

    def _after_load(self, n: int) -> None:
        if self._bc_widget is not None:
            self._bc_widget.cleanup()
            self._bc_widget = None
            self._bc_dock.hide()
        if self._debayer_widget is not None:
            self._debayer_widget.cleanup()
            self._debayer_widget = None
            self._demosaic_dock.hide()
        if self._offset_widget is not None:
            self._offset_widget.cleanup()
            self._offset_widget = None
            self._offset_dock.hide()
        if self._matching_widget is not None:
            self._matching_widget.cleanup()
            self._matching_widget = None
            self._matching_dock.hide()
        self._viewer._auto_min = None
        self._viewer._auto_max = None
        self._viewer._display_min = None
        self._viewer._display_max = None
        self._slider.setMaximum(max(n - 1, 0))
        self._frame_spin.setMaximum(max(n - 1, 0))
        self._slider.setValue(0)
        self._frame_spin.setValue(0)
        self._update_frame_label(0)
        self._show_frame(0)
        self._viewer.fit_view()
        self.setWindowTitle(f"Image Tool - {self._source.display_name}")
        img = self._source.get_frame(0, copy=False)
        if img is not None:
            h, w = img.shape[:2]
            dtype = img.dtype
            self._status_info.setText(f"{w} x {h}  |  {dtype}  |  {n} frames")

    # ------------------------------------------------------------------ Frame
    def _show_frame(self, idx: int) -> None:
        img = self._source.get_frame(idx, copy=False)
        if img is None:
            return
        self._viewer.set_image(img)
        if self._bc_widget is not None:
            self._bc_widget.set_frame_idx(idx)
        if self._active_tool:
            self._active_tool.on_frame_changed(idx, img)
        if self._debayer_widget is not None:
            self._debayer_widget.set_frame_idx(idx)
        if self._offset_widget is not None:
            self._offset_widget.set_frame_idx(idx)
        if self._matching_widget is not None:
            self._matching_widget.set_frame_idx(idx)

    def _on_frame_scroll(self, delta: int) -> None:
        new_val = self._slider.value() + delta
        new_val = max(0, min(new_val, self._slider.maximum()))
        self._slider.setValue(new_val)

    def _on_slider_changed(self, idx: int) -> None:
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(idx)
        self._frame_spin.blockSignals(False)
        self._on_frame_changed(idx)

    def _on_frame_spin_changed(self, idx: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)
        self._on_frame_changed(idx)

    def _on_frame_changed(self, idx: int) -> None:
        self._update_frame_label(idx)
        if 0 <= idx < self._source.frame_count:
            self._show_frame(idx)

    def _update_frame_label(self, idx: int) -> None:
        n = self._source.frame_count
        name = self._source.frame_name(idx)
        self._frame_label.setText(f"{name}  ({idx + 1}/{n})")

    # ------------------------------------------------------------------ B/C Dock
    def _open_bc_dialog(self) -> None:
        if not self._source.is_loaded:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        if self._bc_widget is not None:
            self._bc_widget.cleanup()
        idx = self._slider.value()
        self._bc_widget = BCDialog(self._viewer, self._source, idx)
        self._bc_dock.setWidget(self._bc_widget)
        self._bc_dock.show()

    def _on_bc_visibility(self, visible: bool) -> None:
        if not visible and self._bc_widget is not None:
            self._bc_widget.cleanup()
            self._bc_widget = None
            idx = self._slider.value()
            self._show_frame(idx)

    # ------------------------------------------------------------------ Demosaic
    def _open_demosaic(self) -> None:
        if not self._source.is_loaded:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        if self._debayer_widget is not None:
            self._debayer_widget.cleanup()
        idx = self._slider.value()
        self._debayer_widget = DebayerWidget(self._viewer, self._source, idx)
        self._demosaic_dock.setWidget(self._debayer_widget)
        self._demosaic_dock.show()

    def _on_demosaic_visibility(self, visible: bool) -> None:
        if not visible and self._debayer_widget is not None:
            self._debayer_widget.cleanup()
            self._debayer_widget = None
            idx = self._slider.value()
            self._show_frame(idx)

    # ------------------------------------------------------------------ Frame Offset
    def _open_offset(self) -> None:
        if not self._source.is_loaded:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        if self._offset_widget is not None:
            self._offset_widget.cleanup()
        idx = self._slider.value()
        self._offset_widget = FrameOffsetWidget(self._viewer, self._source, idx)
        self._offset_dock.setWidget(self._offset_widget)
        self._offset_dock.show()

    def _on_offset_visibility(self, visible: bool) -> None:
        if not visible and self._offset_widget is not None:
            self._offset_widget.cleanup()
            self._offset_widget = None
            idx = self._slider.value()
            self._show_frame(idx)

    # ------------------------------------------------------------------ Image Matching
    def _open_matching(self) -> None:
        if not self._source.is_loaded:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        if self._matching_widget is not None:
            self._matching_widget.cleanup()
        idx = self._slider.value()
        self._matching_widget = ImageMatchingWidget(
            self._viewer, self._source, idx,
            MainWindow.open_stack_window,
        )
        self._matching_dock.setWidget(self._matching_widget)
        self._matching_dock.show()

    def _on_matching_visibility(self, visible: bool) -> None:
        if not visible and self._matching_widget is not None:
            self._matching_widget.cleanup()
            self._matching_widget = None

    # ------------------------------------------------------------------ Select All
    def _on_select_all(self) -> None:
        if not self._source.is_loaded:
            return
        tool = self._active_tool
        if not isinstance(tool, RoiToolBase):
            return
        img = self._viewer.raw_image
        if img is None:
            return
        h, w = img.shape[:2]
        tool.select_all(w, h)

    # ------------------------------------------------------------------ Measure
    def _on_measure(self) -> None:
        if not self._source.is_loaded:
            return
        tool = self._active_tool
        if not isinstance(tool, RoiToolBase) or not tool.has_roi:
            return
        img = self._viewer.raw_image
        if img is None:
            return
        h, w = img.shape[:2]
        mask = tool.get_mask(h, w)
        roi_rect = tool.get_roi_rect()
        if mask is None or roi_rect is None:
            return
        idx = self._slider.value()
        frame_name = self._source.frame_name(idx)
        self._measure_widget.add_measurement(tool.name, roi_rect, mask, img, frame_name)
        if not self._measure_dock.isVisible():
            self._measure_dock.show()

    def _on_measure_all_frames(self) -> None:
        if not self._source.is_loaded:
            return
        tool = self._active_tool
        if not isinstance(tool, RoiToolBase) or not tool.has_roi:
            return
        roi_rect = tool.get_roi_rect()
        if roi_rect is None:
            return

        n = self._source.frame_count
        first = self._source.get_frame(0, copy=False)
        if first is None:
            return
        h, w = first.shape[:2]
        mask = tool.get_mask(h, w)
        if mask is None:
            return

        progress = QProgressDialog("Measuring all frames...", "Cancel", 0, n, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        for i in range(n):
            if progress.wasCanceled():
                break
            img = self._source.get_frame(i, copy=False)
            if img is None:
                continue
            frame_name = self._source.frame_name(i)
            self._measure_widget.add_measurement(
                tool.name, roi_rect, mask, img, frame_name)
            progress.setValue(i + 1)
            QApplication.processEvents()

        progress.close()
        if not self._measure_dock.isVisible():
            self._measure_dock.show()

    # ------------------------------------------------------------------ Crop
    def _on_crop(self) -> None:
        if not self._source.is_loaded:
            return
        tool = self._active_tool
        if not isinstance(tool, RoiToolBase) or not tool.has_roi:
            QMessageBox.information(self, "Crop", "Draw a Rectangle ROI first.")
            return
        roi = tool.get_roi_rect()
        if roi is None:
            return

        idx = self._slider.value()
        n = self._source.frame_count
        dlg = CropDialog(roi, idx, n, self)
        if dlg.exec() != CropDialog.DialogCode.Accepted:
            return
        cfg = dlg.get_config()

        x0, y0, x1, y1 = roi
        if cfg["mode"] == "current":
            indices = [cfg["frame"]]
        elif cfg["mode"] == "custom":
            indices = cfg["indices"]
        else:
            indices = list(range(cfg["start"], cfg["end"] + 1))

        crops = []
        names = []
        for i in indices:
            img = self._source.get_frame(i, copy=False)
            if img is None:
                continue
            crops.append(img[y0:y1, x0:x1].copy())
            names.append(self._source.frame_name(i))
        if not crops:
            return
        stack = np.stack(crops)
        title = f"Crop ({len(crops)} frames)"

        MainWindow.open_stack_window(stack, title, names=names)

    # ------------------------------------------------------------------ Batch Crop
    def _on_batch_crop(self) -> None:
        if not self._source.is_loaded:
            return
        tool = self._active_tool
        if not isinstance(tool, RoiToolBase) or not tool.has_roi:
            QMessageBox.information(self, "Batch Crop", "Draw a Rectangle ROI first.")
            return
        roi = tool.get_roi_rect()
        if roi is None:
            return

        n = self._source.frame_count
        first = self._source.get_frame(0, copy=False)
        if first is None:
            return
        h, w = first.shape[:2]

        dlg = BatchCropDialog(roi, n, w, h, self._source, self)
        if dlg.exec() != BatchCropDialog.DialogCode.Accepted:
            return
        cfg = dlg.get_config()

        x0, y0, x1, y1 = roi
        rw, rh = x1 - x0, y1 - y0
        ox, oy = cfg["offset_x"], cfg["offset_y"]
        extra_dims = first.shape[2:] if first.ndim > 2 else ()

        progress = QProgressDialog("Batch cropping...", "Cancel", 0, n, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        crops = []
        names = []
        for i in range(n):
            if progress.wasCanceled():
                break
            img = self._source.get_frame(i, copy=False)
            if img is None:
                progress.setValue(i + 1)
                continue
            cx0 = x0 + i * ox
            cy0 = y0 + i * oy
            sx0 = max(0, cx0)
            sy0 = max(0, cy0)
            sx1 = min(w, cx0 + rw)
            sy1 = min(h, cy0 + rh)
            if sx1 <= sx0 or sy1 <= sy0:
                crops.append(np.zeros((rh, rw) + extra_dims, dtype=first.dtype))
                names.append(self._source.frame_name(i))
                progress.setValue(i + 1)
                continue
            crop = img[sy0:sy1, sx0:sx1]
            if crop.shape[0] != rh or crop.shape[1] != rw:
                padded = np.zeros((rh, rw) + extra_dims, dtype=first.dtype)
                px0 = sx0 - cx0
                py0 = sy0 - cy0
                padded[py0:py0 + crop.shape[0], px0:px0 + crop.shape[1]] = crop
                crop = padded
            crops.append(crop)
            names.append(self._source.frame_name(i))
            progress.setValue(i + 1)
            QApplication.processEvents()

        progress.close()

        if crops:
            stack = np.stack(crops)
            MainWindow.open_stack_window(stack, "Batch Crop", names=names)

    # ------------------------------------------------------------------ Save
    def _on_save(self) -> None:
        if not self._source.is_loaded:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save", "", "TIFF Files (*.tif *.tiff)")
        if not path:
            return
        self._save_tiff_stack(Path(path))

    def _on_save_as(self) -> None:
        if not self._source.is_loaded:
            return
        filters = "TIFF Files (*.tif *.tiff);;PNG Files (*.png);;BMP Files (*.bmp)"
        path, selected = QFileDialog.getSaveFileName(
            self, "Save As", "", filters)
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() in (".tif", ".tiff"):
            self._save_tiff_stack(p)
        else:
            idx = self._slider.value()
            img = self._source.get_frame(idx, copy=False)
            if img is not None:
                img = self._to_saveable(img, p.suffix)
                if not self._imwrite_safe(p, img, p.suffix):
                    QMessageBox.warning(self, "Error", f"Failed to save: {p.name}")

    def _on_save_sequence(self) -> None:
        if not self._source.is_loaded:
            return
        n = self._source.frame_count
        has_names = self._source._mode == "folder" or bool(self._source._names)
        sample_names = (
            [self._source.frame_name(i) for i in range(n)]
            if has_names else None
        )
        dlg = SaveSequenceDialog(n, has_names, sample_names, self)
        if dlg.exec() != SaveSequenceDialog.DialogCode.Accepted:
            return
        cfg = dlg.get_config()
        if cfg is None:
            return
        cfg["directory"].mkdir(parents=True, exist_ok=True)

        ext = {
            "TIFF": ".tif", "PNG": ".png", "BMP": ".bmp",
        }[cfg["format"]]

        progress = QProgressDialog("Saving...", "Cancel", 0, n, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        saved = 0
        failed: list[str] = []
        for i in range(n):
            if progress.wasCanceled():
                break
            img = self._source.get_frame(i, copy=False)
            if img is None:
                failed.append(f"Frame {i}: no data")
                progress.setValue(i + 1)
                continue
            frame_name = self._source.frame_name(i) if has_names else ""
            p = dlg.make_path(cfg, i, frame_name)
            ok = self._imwrite_safe(p, self._to_saveable(img, ext), ext)
            if ok:
                saved += 1
            else:
                failed.append(f"Frame {i}: {p.name}")
            progress.setValue(i + 1)
            QApplication.processEvents()

        progress.close()

        if failed:
            detail = "\n".join(failed[:20])
            if len(failed) > 20:
                detail += f"\n... and {len(failed) - 20} more"
            QMessageBox.warning(
                self, "Save Result",
                f"Saved {saved}/{n} frames.\n"
                f"{len(failed)} failed:\n\n{detail}",
            )
        else:
            self.statusBar().showMessage(f"Saved {saved} frames.", 5000)

    def _to_saveable(self, img: np.ndarray, ext: str) -> np.ndarray:
        """PNG/BMP 저장 시 display range를 적용하여 uint8로 변환."""
        if ext.lower() in (".tif", ".tiff") or img.dtype == np.uint8:
            return img
        dr = self._viewer.get_display_range()
        if dr is not None:
            return _normalize_to_u8(img, dr[0], dr[1])
        mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
        return _normalize_to_u8(img, mn, mx)

    @staticmethod
    def _imwrite_safe(path: Path, img: np.ndarray, ext: str) -> bool:
        """cv2.imencode + write_bytes로 non-ASCII 경로 안전하게 저장."""
        try:
            ok, buf = cv2.imencode(ext, img)
            if not ok:
                return False
            path.write_bytes(buf.tobytes())
            return True
        except Exception:
            return False

    def _save_tiff_stack(self, path: Path) -> None:
        n = self._source.frame_count
        progress = QProgressDialog("Saving TIFF...", "Cancel", 0, n, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        frames = []
        for i in range(n):
            if progress.wasCanceled():
                break
            img = self._source.get_frame(i, copy=False)
            if img is not None:
                frames.append(img)
            progress.setValue(i + 1)
            QApplication.processEvents()

        progress.close()

        if frames:
            stack = np.stack(frames)
            names = [self._source.frame_name(i) for i in range(len(frames))]
            metadata = {"frame_names": names}
            tifffile.imwrite(str(path), stack, description=str(metadata))

    # ------------------------------------------------------------------ 우클릭
    def _on_right_click(self, x: int, y: int, event: object) -> None:
        tool = self._active_tool
        if not isinstance(tool, RoiToolBase) or not tool.has_roi:
            return
        if not tool._hit_test(x, y):
            return
        menu = QMenu(self)
        act_crop = menu.addAction("Crop...")
        act_batch = menu.addAction("Batch Crop...")
        act_crop.triggered.connect(self._on_crop)
        act_batch.triggered.connect(self._on_batch_crop)
        menu.exec(self._viewer.mapToGlobal(event.pos()))

    # ------------------------------------------------------------------ 상태바
    def _on_mouse_moved(self, x: int, y: int, value: object) -> None:
        if value is None:
            self._status_coord.setText("")
            return
        if isinstance(value, np.ndarray):
            val_str = ", ".join(str(v) for v in value)
        else:
            val_str = str(value)
        self._status_coord.setText(f"x={x}  y={y}  |  value={val_str}")
