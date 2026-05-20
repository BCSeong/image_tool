"""ImageSource: 폴더 glob / multi-frame TIFF 통합 이미지 로더."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile


class ImageSource:
    """프레임 기반 이미지 소스. 폴더(개별 파일)와 TIFF 스택 모두 지원."""

    def __init__(self) -> None:
        self._mode: str = "empty"  # "empty", "folder", "tiff", "array"
        self._paths: list[Path] = []
        self._stack: np.ndarray | None = None
        self._folder: Path | None = None
        self._tiff_path: Path | None = None
        self._array_name: str = "Array"
        self._names: list[str] = []
        self._cache: dict[int, np.ndarray] = {}

    @property
    def frame_count(self) -> int:
        if self._mode == "folder":
            return len(self._paths)
        if self._mode in ("tiff", "array"):
            return self._stack.shape[0]
        return 0

    @property
    def is_loaded(self) -> bool:
        return self._mode != "empty"

    @property
    def source_path(self) -> Path | None:
        if self._mode == "folder":
            return self._folder
        if self._mode == "tiff":
            return self._tiff_path
        return None

    @property
    def display_name(self) -> str:
        if self._mode == "folder" and self._folder:
            return self._folder.name
        if self._mode == "tiff" and self._tiff_path:
            return self._tiff_path.name
        if self._mode == "array":
            return self._array_name
        return ""

    def load_folder(self, folder: str | Path, pattern: str = "*") -> int:
        """폴더에서 이미지 파일들을 glob으로 로드."""
        folder = Path(folder)
        exts = {".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}
        paths = sorted(
            p for p in folder.glob(pattern)
            if p.suffix.lower() in exts and p.is_file()
        )
        self._mode = "folder"
        self._paths = paths
        self._stack = None
        self._folder = folder
        self._tiff_path = None
        self._names = []
        self._cache.clear()
        return len(paths)

    def load_paths(self, paths: list[Path], folder: Path | None = None) -> int:
        """미리 필터링된 경로 리스트를 직접 로드."""
        self._mode = "folder"
        self._paths = list(paths)
        self._stack = None
        self._folder = folder or (paths[0].parent if paths else None)
        self._tiff_path = None
        self._names = []
        self._cache.clear()
        return len(self._paths)

    def load_array(self, stack: np.ndarray, name: str = "Array",
                   names: list[str] | None = None) -> int:
        """메모리 상의 numpy 배열 스택을 로드."""
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
        self._mode = "array"
        self._paths = []
        self._stack = stack
        self._folder = None
        self._tiff_path = None
        self._array_name = name
        self._names = list(names) if names else []
        self._cache.clear()
        return stack.shape[0]

    def load_tiff_stack(self, path: str | Path) -> int:
        """Multi-frame TIFF 파일 로드."""
        path = Path(path)
        stack = tifffile.imread(str(path))
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
        self._mode = "tiff"
        self._paths = []
        self._stack = stack
        self._folder = None
        self._tiff_path = path
        n = stack.shape[0]
        w = len(str(max(n - 1, 0)))
        stem = path.stem
        self._names = [f"{stem}_{i:0{w}d}" for i in range(n)]
        self._cache.clear()
        return n

    def get_frame(self, idx: int, copy: bool = True) -> np.ndarray | None:
        """idx번째 프레임을 numpy 배열로 반환.
        copy=False이면 원본 뷰 반환 (읽기 전용 용도)."""
        if idx < 0 or idx >= self.frame_count:
            return None
        if idx in self._cache:
            return self._cache[idx].copy() if copy else self._cache[idx]
        if self._mode == "folder":
            img = cv2.imread(str(self._paths[idx]), cv2.IMREAD_UNCHANGED)
            return img
        if self._mode in ("tiff", "array"):
            return self._stack[idx].copy() if copy else self._stack[idx]
        return None

    def frame_name(self, idx: int) -> str:
        """프레임의 표시 이름."""
        if 0 <= idx < len(self._names):
            return self._names[idx]
        if self._mode == "folder" and 0 <= idx < len(self._paths):
            return self._paths[idx].name
        return f"Frame {idx:03d}"

    def set_frame(self, idx: int, img: np.ndarray) -> None:
        """특정 프레임의 데이터를 교체 (Apply 시 사용)."""
        if idx < 0 or idx >= self.frame_count:
            return
        if self._mode == "folder":
            self._cache[idx] = img
        elif self._mode in ("tiff", "array"):
            self._stack[idx] = img

    def close(self) -> None:
        self._mode = "empty"
        self._paths = []
        self._stack = None
        self._folder = None
        self._tiff_path = None
        self._array_name = "Array"
        self._names = []
        self._cache.clear()
