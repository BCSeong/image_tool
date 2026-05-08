import cv2
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.signal import find_peaks
from typing import List, Tuple, Dict, Optional
from pathlib import Path

class FocusAnalyzer:
    """Focus 분석을 담당하는 클래스"""
    
    def __init__(self, grid_count: int = 7):
        self.grid_count = grid_count
        self.blob_params = None  # SimpleBlobDetector_Params 저장용
        
    def divide_into_grids(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """이미지를 grid_count x grid_count 개의 구역으로 나눕니다."""
        height, width = image.shape
        
        # Grid 크기 계산
        grid_height = height // self.grid_count
        grid_width = width // self.grid_count
        
        grids = []
        positions = []
        
        for y in range(self.grid_count):
            for x in range(self.grid_count):
                # Grid 좌표 계산
                start_y = y * grid_height
                start_x = x * grid_width
                end_y = start_y + grid_height
                end_x = start_x + grid_width
                
                # 이미지 경계 확인
                if end_y <= height and end_x <= width:
                    grid = image[start_y:end_y, start_x:end_x]
                    grids.append(grid)
                    positions.append((start_x, start_y))
        
        return grids, positions
    
    def set_blob_detector_params(self, params: cv2.SimpleBlobDetector_Params):
        """Blob detector 파라미터를 설정합니다."""
        self.blob_params = params
    
    def calculate_laplacian_variance(self, image: np.ndarray) -> float:
        """Laplacian 분산을 사용한 focus 측정"""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return np.var(laplacian)
    
    def calculate_energy_laplacian(self, image: np.ndarray) -> float:
        """Energy of Laplacian focus 측정"""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return np.sum(laplacian**2)
    
    def calculate_curvature_measure(self, image: np.ndarray) -> float:
        """Curvature-based focus 측정"""
        # 2차 미분을 사용한 곡률 측정
        dxx = cv2.Sobel(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3), cv2.CV_64F, 1, 0, ksize=3)
        dyy = cv2.Sobel(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3), cv2.CV_64F, 0, 1, ksize=3)
        curvature = np.abs(dxx) + np.abs(dyy)
        return np.mean(curvature)
    
    def calculate_edge_spread_fwhm(self, prof: np.ndarray) -> List[float]:
        """Profile을 미분하여 edge spread function을 구한 뒤 FWHM을 계산합니다."""
        prof = scipy.ndimage.gaussian_filter1d(prof, 1)
        d_prof = np.gradient(prof)
        fwhm_list = []
        
        # positive peak
        peaks_pos, _ = find_peaks(d_prof, height=np.max(d_prof)*0.5)
        if len(peaks_pos) > 0:
            main_peak = peaks_pos[np.argmax(d_prof[peaks_pos])]
            fwhm_list.append(self._calc_fwhm(d_prof, main_peak))
        
        # negative peak
        peaks_neg, _ = find_peaks(-d_prof, height=np.max(-d_prof)*0.5)
        if len(peaks_neg) > 0:
            main_peak = peaks_neg[np.argmax(-d_prof[peaks_neg])]
            fwhm_list.append(self._calc_fwhm(-d_prof, main_peak))
        
        return fwhm_list

    def _calc_fwhm(self, d_prof: np.ndarray, peak_idx: int) -> float:
        """FWHM 계산의 보조 함수"""
        peak_val = d_prof[peak_idx]
        half_max = peak_val / 2
        
        # 왼쪽
        left = peak_idx
        while left > 0 and d_prof[left] > half_max:
            left -= 1
        
        # 왼쪽 보간
        if left < peak_idx:
            x1, y1 = left, d_prof[left]
            x2, y2 = left+1, d_prof[left+1]
            if y2 != y1:
                left_x = x1 + (half_max - y1) / (y2 - y1)
            else:
                left_x = left
        else:
            left_x = left
        
        # 오른쪽
        right = peak_idx
        while right < len(d_prof)-1 and d_prof[right] > half_max:
            right += 1
        
        # 오른쪽 보간
        if right > peak_idx:
            x1, y1 = right-1, d_prof[right-1]
            x2, y2 = right, d_prof[right]
            if y2 != y1:
                right_x = x1 + (half_max - y1) / (y2 - y1)
            else:
                right_x = right
        else:
            right_x = right
        
        return right_x - left_x

    def calculate_edge_spread_function(self, roi_img: np.ndarray, 
                                     dot_diameter_px: int = 50, 
                                     debug_save_path: Optional[str] = None, 
                                     method: str = 'opencv_blob',
                                     output_path: Optional[str] = None,
                                     z_pos: Optional[float] = None,
                                     grid_idx: Optional[int] = None,
                                     grid_x: Optional[int] = None,
                                     grid_y: Optional[int] = None) -> float:
        """Edge spread function을 계산합니다."""
        img = roi_img.copy()
        h, w = img.shape
        
        # dot 중심 검출
        centers = []
        if method == 'opencv_blob':
            # 저장된 파라미터가 있으면 사용, 없으면 기본값 사용
            if self.blob_params is not None:
                params = self.blob_params
            else:
                params = cv2.SimpleBlobDetector_Params()
                params.minThreshold = 0
                params.maxThreshold = 255
                params.filterByArea = True
                params.minArea = 100
                params.maxArea = 100*100
                params.filterByCircularity = True
                params.minCircularity = 0.7
                params.filterByInertia = True
                params.minInertiaRatio = 0.7
                params.filterByConvexity = True
                params.minConvexity = 0.7
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(img)
            for kp in keypoints:
                centers.append((int(kp.pt[0]), int(kp.pt[1]), int(kp.size)))
        
        if not centers:
            # 실패한 이미지를 디버그 폴더에 저장
            if output_path:
                try:
                    debug_dir = Path(output_path) / 'debug_fail'
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 파일명 생성
                    if z_pos is not None and grid_idx is not None:
                        filename = f"fail_z{z_pos:.3f}_grid{grid_idx}_x{grid_x}_y{grid_y}.png"
                    else:
                        filename = f"fail_grid{grid_idx or 'unknown'}.png"
                    
                    filepath = debug_dir / filename
                    cv2.imwrite(str(filepath), img)
                except Exception as e:
                    print(f"디버그 이미지 저장 실패: {e}")
            
            return float('inf')
        
        # 각 dot별 FWHM 계산
        fwhm_all = []
        for cx, cy, diameter in centers:
            cross_len = int(dot_diameter_px * 1.2)
            # profile 추출
            x_profile = img[cy, max(0, cx-cross_len):min(w, cx+cross_len)]
            y_profile = img[max(0, cy-cross_len):min(h, cy+cross_len), cx]
            # FWHM 계산
            fwhm_list = self.calculate_edge_spread_fwhm(x_profile) + self.calculate_edge_spread_fwhm(y_profile)
            if fwhm_list:
                fwhm_mean = np.mean(fwhm_list)
                fwhm_all.append(fwhm_mean)
        
        focus_value = np.mean(fwhm_all) if fwhm_all else float('inf')
        return focus_value
    
    def get_grid_positions(self, img_shape: Tuple[int, int], max_grid_num: int = 7) -> Tuple[List[Tuple[int, int, int, int]], int, int, int, int]:
        """이미지 shape와 max_grid_num으로 grid 좌상단 좌표 리스트 반환"""
        height, width = img_shape
        if width >= height:
            grid_num_x = max_grid_num
            grid_w = width // grid_num_x
            grid_num_y = height // grid_w
            grid_h = grid_w
        else:
            grid_num_y = max_grid_num
            grid_h = height // grid_num_y
            grid_num_x = width // grid_h
            grid_w = grid_h
        
        # 좌상단 좌표 생성 (외곽 포함)
        xs = [0]
        for i in range(1, grid_num_x-1):
            xs.append(i*grid_w)
        xs.append(width-grid_w)
        ys = [0]
        for i in range(1, grid_num_y-1):
            ys.append(i*grid_h)
        ys.append(height-grid_h)
        
        # grid 좌상단 좌표 리스트
        grid_positions = []
        for y in ys:
            for x in xs:
                grid_positions.append((x, y, grid_w, grid_h))
        
        return grid_positions, grid_w, grid_h, grid_num_x, grid_num_y

    def get_diagonal_rois(self) -> List[Dict]:
        """5개의 diagonal ROI를 선택합니다: 대각선 방향으로 균등하게 분포"""
        if self.grid_count < 3:
            raise ValueError("Grid count must be at least 3 for diagonal ROI selection")
        
        # Grid 인덱스 계산 (0부터 시작)
        max_idx = self.grid_count - 1
        
        # 대각선 방향으로 5개 구획 균등 선택
        # 예: 10x10 -> (0,0), (3,3), (5,5), (7,7), (9,9)
        step = max_idx / 4  # 4개 간격으로 나누기
        
        diagonal_indices = []
        for i in range(5):
            idx = int(round(i * step))
            diagonal_indices.append((idx, idx))
        
        # ROI 정보 생성
        rois = []
        for i, (grid_y, grid_x) in enumerate(diagonal_indices):
            position_type = 'start' if i == 0 else 'end' if i == 4 else 'middle'
            rois.append({
                'roi_id': i,
                'grid_x': grid_x,
                'grid_y': grid_y,
                'grid_idx': grid_y * self.grid_count + grid_x,
                'position': position_type
            })
        
        return rois
    
    def analyze_focus_for_images(self, images: List[np.ndarray], z_positions: List[float], 
                                selected_metric: str = 'edge_spread_function', 
                                progress_callback=None,
                                output_path: Optional[str] = None) -> pd.DataFrame:
        """
        이미지들에 대해 focus 분석을 수행합니다.
        
        Args:
            images: 분석할 이미지 리스트
            z_positions: 각 이미지의 Z 위치
            selected_metric: 선택할 metric
            
        Returns:
            pd.DataFrame: 분석 결과
        """
        # 선택된 metric 함수 가져오기
        metric_functions = {
            'laplacian_variance': self.calculate_laplacian_variance,
            'energy_laplacian': self.calculate_energy_laplacian,
            'curvature_measure': self.calculate_curvature_measure,
            'edge_spread_function': self.calculate_edge_spread_function
        }
        
        if selected_metric not in metric_functions:
            raise ValueError(f"지원하지 않는 metric: {selected_metric}")
        
        metric_func = metric_functions[selected_metric]
        
        # 결과 저장용 리스트
        results = []
        
        print(f"Focus 분석 시작: {selected_metric}")
        
        for i, (image, z_pos) in enumerate(zip(images, z_positions)):
            print(f"이미지 {i+1}/{len(images)} 분석 중 (z={z_pos:.3f}mm)...")
            
            # 진행 상황 업데이트
            if progress_callback:
                progress = int((i / len(images)) * 100)
                progress_callback(progress)
            
            # 이미지를 구역으로 나누기
            grids, positions = self.divide_into_grids(image)
            
            # 각 구역에 대해 선택된 metric 계산
            for grid_idx, (grid, pos) in enumerate(zip(grids, positions)):
                if selected_metric == 'edge_spread_function':
                    # edge_spread_function의 경우 추가 파라미터 전달
                    value = metric_func(
                        grid,
                        output_path=output_path,
                        z_pos=z_pos,
                        grid_idx=grid_idx,
                        grid_x=pos[0],
                        grid_y=pos[1]
                    )
                else:
                    value = metric_func(grid)
                
                results.append({
                    'z_position': z_pos,
                    'grid_idx': grid_idx,
                    'grid_x': pos[0],
                    'grid_y': pos[1],
                    'focus_value': value
                })
        
        # 완료 시 100%로 설정
        if progress_callback:
            progress_callback(100)
        
        df_metric = pd.DataFrame(results)
        print(f"분석 완료: {len(df_metric)}개 데이터 포인트")
        
        return df_metric
