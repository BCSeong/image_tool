# Image Tool — CLAUDE.md

PySide6 기반 ImageJ-like 이미지 분석 도구. `python main.py`로 실행.

## Architecture

```
main.py              — QApplication 엔트리포인트
main_window.py       — MainWindow: 메뉴, 툴바, 뷰어, 슬라이더, dock 패널
viewer.py            — ImageViewer: 줌/팬 가능한 이미지 뷰어 (ImageViewer 클래스)
image_source.py      — ImageSource: 폴더/TIFF/array 모드 프레임 로더
                       get_frame(idx, copy), set_frame(idx, img), frame_name(idx)
tool_base.py         — BaseTool ABC: 모든 tool의 인터페이스
enhanced_dock.py     — EnhancedDockWidget: float/maximize/close 버튼이 있는 커스텀 dock
```

## Tools (tools/)

| File | Class | Description |
|------|-------|-------------|
| roi_base.py | RoiToolBase | ROI tool 공통 베이스 |
| rect_tool.py | RectTool | 사각형 ROI |
| ellipse_tool.py | EllipseTool | 타원 ROI |
| line_tool.py | LineTool | 1D 라인 프로파일 (2-column layout: live + held figures) |
| figure_props_dialog.py | FigurePropsDialog | Held figure 속성 편집 (Figure/Axes/Lines/Legend 탭) |

## Dock Panels (Image Processing)

| File | Class | Pattern |
|------|-------|---------|
| bc_dialog.py | BCDialog(QWidget) | dock panel, cleanup() 패턴 |
| debayer_widget.py | DebayerWidget(QWidget) | dock panel, set_frame_idx(), cleanup() 패턴 |
| frame_offset_widget.py | FrameOffsetWidget(QWidget) | dock panel, set_frame_idx(), cleanup() 패턴 |

### Dock Panel Pattern (BCDialog 패턴)

새 dock panel을 추가할 때 따라야 할 패턴:
1. `XxxWidget(QWidget)` 생성자: `(viewer, source, frame_idx, parent=None)`
2. `set_frame_idx(idx)` — 프레임 변경 시 동기화
3. `cleanup()` — dock 닫힐 때 원상복구
4. `main_window.py`에서:
   - `self._xxx_widget = None` 초기화
   - `EnhancedDockWidget` 생성 + `hide()`
   - `visibilityChanged` → `_on_xxx_visibility` (cleanup + `_show_frame`)
   - `_show_frame()`에서 `set_frame_idx()` 호출
   - 메뉴 액션 → `_open_xxx()` (cleanup 기존 → 생성 → setWidget → show)

## Plugins (plugins/)

- `plugin_base.py`: PluginBase ABC (`name`, `run(source, frame_idx, parent)`)
- `plugins/__init__.py`: ALL_PLUGINS 리스트
- `focus_analysis/`: FocusAnalysisDialog (modal QDialog), FocusAnalyzer, ResultPlotter, BlobDetectorDialog

## Other Dialogs

| File | Class | Type |
|------|-------|------|
| crop_dialog.py | CropDialog | modal QDialog |
| batch_crop_dialog.py | BatchCropDialog | modal QDialog |
| save_sequence_dialog.py | SaveSequenceDialog | modal QDialog |
| folder_wizard.py | FolderWizard | modal QDialog |

## Key Conventions

- **Import 경로**: `sys.path`에 `image_tool/` 루트가 들어있음. 모듈은 패키지 없이 직접 import (`from viewer import ImageViewer`). 단, `tools/` 하위에서 상위 모듈 접근 시 `sys.path.insert` 사용.
- **Canvas 크기**: matplotlib figure의 canvas는 `setFixedSize(int(w*dpi), int(h*dpi))`로 고정 (QT layout stretching 방지)
- **프레임 추적**: `self._slider.value()`가 현재 프레임 인덱스의 single source of truth
- **EnhancedDockWidget**: 모든 dock에 사용. Unicode 버튼 (`↗↙□⧉×`). `closable=True` 기본값.
- **LineTool 2-column layout**: 좌측 = live preview, 우측 = held figures. Superposition 모드 지원.
- **FigurePropsDialog**: snapshot/restore 패턴으로 preview + cancel 지원
- **한국어 OK**: .py, .md 파일에 한국어 주석/docstring 사용 가능. .bat/.sh 파일에는 영어만.

## Run & Test

```bash
# 가상환경 활성화 후
python main.py
```

테스트 프레임워크 없음. 수동 테스트로 검증:
1. 폴더/TIFF 열기 → 프레임 슬라이더 동작
2. 각 tool 선택 → ROI/라인 그리기 → 측정 결과
3. dock panel 열기/닫기 → cleanup 정상 동작
4. held figure 우클릭 → Edit Properties → preview/apply/cancel
