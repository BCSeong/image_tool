# Plugin Development Guide

## Architecture

```
image_tool/
  plugins/
    __init__.py              # ALL_PLUGINS registry
    plugin_base.py           # PluginBase ABC
    my_plugin/               # each plugin = sub-package
      __init__.py            # MyPlugin(PluginBase) class
      dialog.py              # QDialog for image_tool integration
      my_core_logic.py       # pure logic (no Qt dependency)
```

## Plugin Interface

All plugins inherit from `PluginBase`:

```python
from plugins.plugin_base import PluginBase

class MyPlugin(PluginBase):
    @property
    def name(self) -> str:
        return "My Plugin"          # shown in Plugins menu

    def run(self, source, frame_idx, parent) -> None:
        # source: ImageSource (loaded images)
        # frame_idx: current frame index
        # parent: MainWindow
        dlg = MyDialog(source, frame_idx, parent)
        dlg.exec()
```

Register in `plugins/__init__.py`:

```python
from plugins.my_plugin import MyPlugin

ALL_PLUGINS = [
    FocusAnalysisPlugin(),
    MyPlugin(),               # add here
]
```

## Development Workflow

### Phase 1: Standalone Development

Develop and verify core logic in a separate project/repo.

```
my_standalone_tool/
  core/
    my_analyzer.py            # pure logic (numpy, opencv, scipy, etc.)
    __init__.py
  gui/                        # standalone test GUI (optional)
    main_window.py
  main.py                     # entry point
  tests/
    test_my_analyzer.py
```

**Critical rule: core logic files must NEVER import PySide6/PyQt.**

This ensures the logic can be:
- Unit tested without a GUI
- Migrated to image_tool without code changes
- Reused in CLI scripts or other tools

### Phase 2: Migrate to image_tool Plugin

1. Copy core logic files into `plugins/my_plugin/`
2. Create `dialog.py` — a QDialog that uses `ImageSource` for input
3. Create `__init__.py` with the `PluginBase` subclass
4. Register in `plugins/__init__.py`

```
image_tool/
  plugins/
    my_plugin/
      __init__.py             # MyPlugin(PluginBase)
      dialog.py               # QDialog (uses ImageSource, not file loading)
      my_analyzer.py          # copied from standalone core/ (unchanged)
```

### What changes during migration

| Component | Standalone | Plugin |
|-----------|-----------|--------|
| Image loading | Own loader (file dialog) | `ImageSource` (already loaded) |
| GUI | Own QMainWindow | QDialog inside image_tool |
| Core logic | `core/my_analyzer.py` | Same file, no changes |
| Progress | Own progress bar | QProgressBar in dialog |
| Background work | QThread worker | Same pattern |

### What stays the same

- Core logic files: **zero modifications** needed
- Algorithm parameters, data structures, return types: identical
- Only the "glue" (dialog UI + ImageSource adapter) is new

## Git Strategy

- Develop standalone tools in their own repo
- When migrating, **copy files** into `plugins/` and commit to image_tool repo
- Record the original repo URL in the plugin's `__init__.py` docstring:

```python
"""My Plugin: description.

Migrated from: https://github.com/user/my-standalone-tool
Original core logic in core/ directory.
"""
```

- The standalone repo can be archived or kept for reference
- image_tool repo is the single source of truth after migration

## Example: Focus Analysis Plugin

```
plugins/
  focus_analysis/
    __init__.py               # FocusAnalysisPlugin
    dialog.py                 # FocusAnalysisDialog + _AnalysisWorker
    focus_analyzer.py         # from focus_analysis_gui/core/ (pure logic)
    result_plotter.py         # from focus_analysis_gui/core/ (pure logic)
```

- `focus_analyzer.py` depends only on: numpy, opencv, pandas, scipy
- `result_plotter.py` depends only on: numpy, pandas, matplotlib, scipy
- `dialog.py` adapts these for image_tool: ImageSource → grayscale uint8 list → FocusAnalyzer
