import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from typing import List, Tuple, Optional
from pathlib import Path

class ResultPlotter:
    """결과 시각화 및 저장을 담당하는 클래스"""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        # 폴더는 실제 파일 저장 시에만 생성
        
    def quadratic_function(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Quadratic function for curve fitting."""
        return a * x**2 + b * x + c
    
    def find_optimal_z_position(self, z_positions: np.ndarray, focus_values: np.ndarray, 
                               minimize_focus: bool = True) -> Tuple[float, float, float, float]:
        """Quadratic fitting을 통해 최적 Z position을 찾습니다."""
        try:
            # Quadratic fitting
            popt, _ = curve_fit(self.quadratic_function, z_positions, focus_values)
            a, b, c = popt
            
            # 최적점 계산 (미분 = 0)
            if abs(a) > 1e-10:  # a가 0이 아닌 경우
                optimal_z = -b / (2 * a)
                
                # 데이터 범위 내에 있는지 확인
                if z_positions.min() <= optimal_z <= z_positions.max():
                    return optimal_z, a, b, c
                else:
                    # 범위를 벗어나면 최대/최소값 반환
                    if minimize_focus:
                        return z_positions[np.argmin(focus_values)], a, b, c
                    else:
                        return z_positions[np.argmax(focus_values)], a, b, c
            else:
                # 선형 함수인 경우
                if minimize_focus:
                    return z_positions[np.argmin(focus_values)], a, b, c
                else:
                    return z_positions[np.argmax(focus_values)], a, b, c
                    
        except (RuntimeError, ValueError):
            # Fitting 실패 시 최대/최소값 반환
            if minimize_focus:
                return z_positions[np.argmin(focus_values)], 0, 0, 0
            else:
                return z_positions[np.argmax(focus_values)], 0, 0, 0
    
    def analyze_all_rois_with_quadratic_fitting(self, df_metric: pd.DataFrame, metric_name: str, 
                                               use_normalized_z: bool = True, 
                                               minimize_focus: bool = True) -> Tuple[List[dict], List[dict]]:
        """모든 ROI에 대해 quadratic fitting을 수행하고 결과를 반환합니다."""
        if df_metric is None or len(df_metric) == 0:
            print("No data available for analysis")
            return [], []
        
        # grid_idx 범위 계산
        grid_idx_min, grid_idx_max = df_metric['grid_idx'].min(), df_metric['grid_idx'].max()
        
        # 모든 고유한 grid_idx 찾기
        unique_grid_indices = sorted(df_metric['grid_idx'].unique())
        
        # 결과 저장용 리스트
        roi_analysis_results = []
        best_positions_data = []
        
        print(f"Analyzing {len(unique_grid_indices)} ROIs with quadratic fitting...")
        
        for grid_idx in unique_grid_indices:
            # 해당 grid_idx의 모든 Z position 데이터 추출
            roi_data = df_metric[df_metric['grid_idx'] == grid_idx].sort_values('z_position')
            
            if len(roi_data) >= 3:  # 최소 3개 점이 필요
                z_positions = roi_data['z_position'].values
                focus_values = roi_data['focus_value'].values
                grid_x = roi_data['grid_x'].iloc[0]  # 첫 번째 값 사용
                grid_y = roi_data['grid_y'].iloc[0]  # 첫 번째 값 사용
                
                # Z position 정규화 (사용자가 이미 mm 단위로 입력했으므로 단순히 첫 번째 값을 0으로)
                if use_normalized_z:
                    z_positions_normalized = z_positions - z_positions[0]  # 첫 번째 값을 0으로 정규화
                else:
                    z_positions_normalized = z_positions
                
                # Quadratic fitting으로 최적 Z position 찾기
                optimal_z, a, b, c = self.find_optimal_z_position(
                    z_positions_normalized, focus_values, minimize_focus
                )
                
                # ROI 분석 결과 저장
                roi_analysis_results.append({
                    'roi_id': grid_idx,
                    'grid_idx': grid_idx,
                    'grid_x': grid_x,
                    'grid_y': grid_y,
                    'z_positions': z_positions_normalized,
                    'focus_values': focus_values,
                    'optimal_z': optimal_z,
                    'quadratic_a': a,
                    'quadratic_b': b,
                    'quadratic_c': c,
                    'fitting_success': True
                })
                
                # Best position 데이터 저장
                best_positions_data.append({
                    'grid_idx': grid_idx,
                    'grid_x': grid_x,
                    'grid_y': grid_y,
                    'optimal_z_position': optimal_z,
                    'focus_value_at_optimal': self.quadratic_function(optimal_z, a, b, c)
                })
                
                if grid_idx % 10 == 0:
                    print(f"Processed ROI {grid_idx}/{grid_idx_max}")
            else:
                print(f"Warning: ROI {grid_idx} has insufficient data points ({len(roi_data)})")
        
        return roi_analysis_results, best_positions_data
    
    def create_best_position_map(self, best_positions_data: List[dict], metric_name: str, 
                                use_normalized_z: bool = True) -> str:
        """Quadratic fitting 결과로부터 best position map을 생성합니다."""
        if not best_positions_data:
            print("No best positions data available")
            return ""
        
        # 폴더 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # grid_idx를 2D grid로 변환하기 위한 정보 추출
        grid_indices = [pos['grid_idx'] for pos in best_positions_data]
        grid_x_coords = [pos['grid_x'] for pos in best_positions_data]
        grid_y_coords = [pos['grid_y'] for pos in best_positions_data]
        
        # Grid 크기 계산 (실제 ROI 개수 기반)
        grid_idx_max = max(grid_indices)
        
        # grid_idx를 2D 좌표로 변환하는 방법 찾기
        # CSV 데이터를 보면 grid_idx가 순차적으로 증가하므로, 
        # grid_x와 grid_y의 고유값 개수로 grid 크기를 추정
        unique_x = sorted(set(grid_x_coords))
        unique_y = sorted(set(grid_y_coords))
        
        grid_width = len(unique_x)
        grid_height = len(unique_y)
        
        print(f"Grid dimensions: {grid_height} x {grid_width} (Y x X)")
        print(f"Total ROIs: {len(grid_indices)}")
        
        # 2D grid 생성
        z_grid = np.zeros((grid_height, grid_width))
        
        for pos in best_positions_data:
            # grid_x, grid_y를 grid 인덱스로 변환
            x_idx = unique_x.index(pos['grid_x'])
            y_idx = unique_y.index(pos['grid_y'])
            
            z_grid[y_idx, x_idx] = pos['optimal_z_position']
        
        # Figure 생성
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(z_grid, cmap='gray', aspect='equal')
        z_unit = "mm"  # 사용자가 mm 단위로 입력했으므로 항상 mm로 표시
        ax.set_title(f'{metric_name}\nOptimal Z Position from Quadratic Fitting ({z_unit})', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel(f'Grid X (range: {min(unique_x)}-{max(unique_x)})', fontsize=12)
        ax.set_ylabel(f'Grid Y (range: {min(unique_y)}-{max(unique_y)})', fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        
        # 파일 저장
        suffix = "_normalized" if use_normalized_z else "_absolute"
        filename = f'best_position_map_{metric_name}{suffix}_quadratic.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filepath}")
        return str(filepath)
    
    def create_roi_focus_profiles(self, roi_analysis_results: List[dict], metric_name: str, 
                                  use_normalized_z: bool = True, 
                                  show_diagonal_only: bool = True,
                                  focus_analyzer=None, auto_depth_calculation: bool = True,
                                  depth_threshold: float = None, minimize_focus: bool = True) -> str:
        """Quadratic fitting 결과를 포함한 ROI focus profiles를 생성합니다."""
        if not roi_analysis_results:
            print("No ROI analysis results available")
            return ""
        
        # 폴더 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 대각선 ROI만 선택할지 전체 ROI를 선택할지 결정
        if show_diagonal_only:
            if focus_analyzer:
                # 새로운 diagonal ROI 선택 로직 사용
                diagonal_roi_info = focus_analyzer.get_diagonal_rois()
                selected_grid_indices = [roi['grid_idx'] for roi in diagonal_roi_info]
                
                # 해당하는 ROI 결과 찾기
                diagonal_rois = []
                for roi_info in diagonal_roi_info:
                    for roi_result in roi_analysis_results:
                        if roi_result['grid_idx'] == roi_info['grid_idx']:
                            # ROI 정보 추가
                            roi_result['roi_id'] = roi_info['roi_id']
                            roi_result['position'] = roi_info['position']
                            diagonal_rois.append(roi_result)
                            break
                
                selected_rois = diagonal_rois
                print(f"Selected diagonal ROIs: {[(roi['grid_idx'], roi['grid_x'], roi['grid_y'], roi['position']) for roi in diagonal_rois]}")
                
                # Diagonal ROI 데이터를 CSV로 저장
                diagonal_csv_path = self.save_diagonal_roi_data(diagonal_rois, metric_name, use_normalized_z)
                
                # 심도 분석 수행
                depth_analysis_path = self.create_depth_of_field_analysis(
                    diagonal_rois, metric_name, depth_threshold, use_normalized_z, 
                    minimize_focus, auto_depth_calculation
                )
            else:
                # 기존 로직 (fallback)
                grid_indices = [roi['grid_idx'] for roi in roi_analysis_results]
                grid_idx_min, grid_idx_max = min(grid_indices), max(grid_indices)
                
                diagonal_rois = []
                for i in range(5):
                    target_idx = int(grid_idx_min + (grid_idx_max - grid_idx_min) * i / 4)
                    
                    closest_roi = None
                    min_distance = float('inf')
                    for roi in roi_analysis_results:
                        distance = abs(roi['grid_idx'] - target_idx)
                        if distance < min_distance:
                            min_distance = distance
                            closest_roi = roi
                    
                    if closest_roi:
                        diagonal_rois.append(closest_roi)
                
                selected_rois = diagonal_rois
                print(f"Selected diagonal ROIs (fallback): {[(roi['grid_idx'], roi['grid_x'], roi['grid_y']) for roi in diagonal_rois]}")
        else:
            selected_rois = roi_analysis_results
        
                # Figure 생성
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # ROI별 색상 정의
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        
        # annotation 위치 추적용
        annotation_positions = []
        
        for i, roi in enumerate(selected_rois):
            z_positions = roi['z_positions']
            focus_values = roi['focus_values']
            optimal_z = roi['optimal_z']
            a, b, c = roi['quadratic_a'], roi['quadratic_b'], roi['quadratic_c']
            
            # ROI별 색상 선택
            color = colors[i % len(colors)]
            
            # 원본 데이터 플롯
            ax.plot(z_positions, focus_values, 'o-', color=color,
                   label=f'ROI {roi["grid_idx"]} ({roi["grid_x"]},{roi["grid_y"]})')
            
            # Quadratic fitting 곡선
            z_fine = np.linspace(z_positions.min(), z_positions.max(), 100)
            y_fine = self.quadratic_function(z_fine, a, b, c)
            ax.plot(z_fine, y_fine, '--', color=color, alpha=0.7)
            
            # 최적점 표시 (큰 별 모양, ROI별 색상)
            ax.axvline(x=optimal_z, color=color, linestyle=':', alpha=0.5)
            optimal_y = self.quadratic_function(optimal_z, a, b, c)
            ax.plot(optimal_z, optimal_y, '*', color=color, markersize=20, markeredgecolor='black', markeredgewidth=1)
            
            # annotation 위치 계산 (겹침 방지)
            annotation_x = optimal_z
            annotation_y = optimal_y
            
            # 기존 annotation들과 겹치지 않는 위치 찾기
            offset_x, offset_y = 10, 10
            while any(abs(annotation_x + offset_x - pos[0]) < 0.3 and abs(annotation_y + offset_y - pos[1]) < 20 
                      for pos in annotation_positions):
                offset_x += 15
                offset_y += 15
            
            annotation_positions.append((annotation_x + offset_x, annotation_y + offset_y))
            
            # Z position 값 표시
            ax.annotate(f'{optimal_z:.3f}', 
                       xy=(annotation_x, annotation_y), 
                       xytext=(offset_x, offset_y), 
                       textcoords='offset points',
                       fontsize=10, 
                       color=color,
                       weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        z_unit = "mm"  # 사용자가 mm 단위로 입력했으므로 항상 mm로 표시
        title_suffix = " (Diagonal ROIs)" if show_diagonal_only else " (All ROIs)"
        ax.set_title(f'{metric_name}\nFocus Profile vs Z Position with Quadratic Fitting{title_suffix}', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel(f'Z Position ({z_unit})', fontsize=12)
        ax.set_ylabel('Focus Value', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 파일 저장
        suffix = "_normalized" if use_normalized_z else "_absolute"
        diagonal_suffix = "_diagonal" if show_diagonal_only else "_all"
        filename = f'roi_focus_profiles_{metric_name}{suffix}_quadratic{diagonal_suffix}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filepath}")
        
        # Diagonal CSV 파일 경로와 심도 분석 파일 경로도 함께 반환
        if show_diagonal_only and 'diagonal_csv_path' in locals():
            depth_path = depth_analysis_path if 'depth_analysis_path' in locals() else ""
            return str(filepath), diagonal_csv_path, depth_path
        else:
            return str(filepath), "", ""
    
    def save_diagonal_roi_data(self, diagonal_rois: List[dict], metric_name: str, use_normalized_z: bool = True) -> str:
        """Diagonal ROI 데이터를 CSV 파일로 저장합니다."""
        if not diagonal_rois:
            print("No diagonal ROI data to save")
            return ""
        
        # 폴더 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV 데이터 구성
        csv_data = []
        for roi in diagonal_rois:
            roi_id = roi.get('roi_id', roi['grid_idx'])
            position = roi.get('position', 'unknown')
            
            for i, (z_pos, focus_val) in enumerate(zip(roi['z_positions'], roi['focus_values'])):
                csv_data.append({
                    'roi_id': roi_id,
                    'grid_idx': roi['grid_idx'],
                    'grid_x': roi['grid_x'],
                    'grid_y': roi['grid_y'],
                    'position': position,
                    'z_position': z_pos,
                    'focus_value': focus_val,
                    'optimal_z': roi['optimal_z'],
                    'quadratic_a': roi['quadratic_a'],
                    'quadratic_b': roi['quadratic_b'],
                    'quadratic_c': roi['quadratic_c']
                })
        
        # DataFrame 생성 및 저장
        df_diagonal = pd.DataFrame(csv_data)
        suffix = "_normalized" if use_normalized_z else "_absolute"
        filename = f'diagonal_roi_profiles_{metric_name}{suffix}.csv'
        filepath = self.output_dir / filename
        df_diagonal.to_csv(filepath, index=False)
        
        print(f"Saved diagonal ROI data: {filepath}")
        return str(filepath)
    
    def create_depth_of_field_analysis(self, diagonal_rois: List[dict], metric_name: str, 
                                      depth_threshold: float = None, use_normalized_z: bool = True,
                                      minimize_focus: bool = True, auto_calculation: bool = True) -> str:
        """심도 분석을 위한 새로운 figure를 생성합니다."""
        if not diagonal_rois:
            print("No diagonal ROI data available for depth analysis")
            return ""
        
        # 폴더 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 자동 심도 계산 (중앙 ROI의 best focus value의 sqrt(2) 배)
        if auto_calculation or depth_threshold is None:
            # 중앙 ROI 찾기 (position이 'middle'인 ROI)
            center_roi = None
            for roi in diagonal_rois:
                if roi.get('position') == 'middle':
                    center_roi = roi
                    break
            
            if center_roi is None:
                # 중앙 ROI가 없으면 3번째 ROI 사용 (5개 중 중앙)
                center_roi = diagonal_rois[2] if len(diagonal_rois) >= 3 else diagonal_rois[0]
            
            # 중앙 ROI의 최적 focus value
            center_optimal_focus = self.quadratic_function(
                center_roi['optimal_z'], 
                center_roi['quadratic_a'], 
                center_roi['quadratic_b'], 
                center_roi['quadratic_c']
            )
            
            # sqrt(2) 배 계산 (minimize_focus에 따라 조정)
            if minimize_focus:
                depth_threshold = center_optimal_focus * np.sqrt(2)  # minimum이 좋은 경우
            else:
                depth_threshold = center_optimal_focus / np.sqrt(2)  # maximum이 좋은 경우
            
            print(f"Auto depth calculation: Center ROI optimal value = {center_optimal_focus:.3f}, Depth threshold = {depth_threshold:.3f}")
        
        # Figure 생성
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # ROI별 색상 정의
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        
        # 모든 ROI의 Z 범위 계산
        all_z_positions = []
        for roi in diagonal_rois:
            all_z_positions.extend(roi['z_positions'])
        z_min, z_max = min(all_z_positions), max(all_z_positions)
        
        # 심도 구간 찾기
        depth_z_ranges = []
        annotation_positions = []  # annotation 위치 추적용
        
        for i, roi in enumerate(diagonal_rois):
            z_positions = roi['z_positions']
            focus_values = roi['focus_values']
            a, b, c = roi['quadratic_a'], roi['quadratic_b'], roi['quadratic_c']
            color = colors[i % len(colors)]
            
            # 원본 데이터 플롯
            ax.plot(z_positions, focus_values, 'o-', color=color,
                   label=f'ROI {roi["grid_idx"]} ({roi["grid_x"]},{roi["grid_y"]})')
            
            # Quadratic fitting 곡선
            z_fine = np.linspace(z_positions.min(), z_positions.max(), 100)
            y_fine = self.quadratic_function(z_fine, a, b, c)
            ax.plot(z_fine, y_fine, '--', color=color, alpha=0.7)
            
            # 최적점 표시
            optimal_z = roi['optimal_z']
            optimal_y = self.quadratic_function(optimal_z, a, b, c)
            ax.plot(optimal_z, optimal_y, '*', color=color, markersize=20, 
                   markeredgecolor='black', markeredgewidth=1)
            
            # 심도 기준선 그리기
            ax.axhline(y=depth_threshold, color=color, linestyle=':', alpha=0.5, 
                      label=f'Depth Threshold ({depth_threshold:.3f})' if i == 0 else "")
            
            # Interpolation을 사용한 정확한 심도 구간 계산
            z_fine = np.linspace(z_positions.min(), z_positions.max(), 1000)  # 고해상도 interpolation
            y_fine = self.quadratic_function(z_fine, a, b, c)
            
            # Interpolated 데이터에서 심도 구간 찾기
            if minimize_focus:
                # minimum이 좋은 경우: focus value <= threshold
                valid_mask = y_fine <= depth_threshold
            else:
                # maximum이 좋은 경우: focus value >= threshold
                valid_mask = y_fine >= depth_threshold
            
            # 연속된 구간 찾기
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                # 연속된 구간들을 찾기
                depth_ranges = []
                start_idx = valid_indices[0]
                prev_idx = start_idx
                
                for idx in valid_indices[1:]:
                    if idx != prev_idx + 1:  # 연속되지 않으면 새로운 구간 시작
                        depth_ranges.append((z_fine[start_idx], z_fine[prev_idx]))
                        start_idx = idx
                    prev_idx = idx
                
                # 마지막 구간 추가
                depth_ranges.append((z_fine[start_idx], z_fine[prev_idx]))
                
                # 가장 긴 구간을 심도로 선택
                longest_range = max(depth_ranges, key=lambda x: x[1] - x[0])
                depth_start, depth_end = longest_range
                depth_z_ranges.append((depth_start, depth_end))
                
                # 심도 구간 표시
                ax.axvspan(depth_start, depth_end, alpha=0.2, color=color)
                ax.axvline(x=depth_start, color=color, linestyle='-', alpha=0.7, linewidth=2)
                ax.axvline(x=depth_end, color=color, linestyle='-', alpha=0.7, linewidth=2)
                
                # annotation 위치 계산 (겹침 방지)
                annotation_x = (depth_start + depth_end) / 2
                annotation_y = depth_threshold
                
                # 기존 annotation들과 겹치지 않는 위치 찾기
                offset_y = 20
                while any(abs(annotation_x - pos[0]) < 0.5 and abs(annotation_y + offset_y - pos[1]) < 30 
                          for pos in annotation_positions):
                    offset_y += 25
                
                annotation_positions.append((annotation_x, annotation_y + offset_y))
                
                # 심도 구간 annotation
                ax.annotate(f'Depth: {depth_start:.3f}~{depth_end:.3f}', 
                           xy=(annotation_x, annotation_y), 
                           xytext=(0, offset_y), 
                           textcoords='offset points',
                           fontsize=9, 
                           color=color,
                           weight='bold',
                           ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 공통 심도 구간 계산
        if depth_z_ranges:
            common_depth_start = max([r[0] for r in depth_z_ranges])
            common_depth_end = min([r[1] for r in depth_z_ranges])
            
            if common_depth_start <= common_depth_end:
                # 공통 심도 구간 표시
                ax.axvspan(common_depth_start, common_depth_end, alpha=0.3, 
                          color='purple', hatch='///', label='Common Depth Range')
                ax.axvline(x=common_depth_start, color='purple', linestyle='-', 
                          alpha=0.8, linewidth=3)
                ax.axvline(x=common_depth_end, color='purple', linestyle='-', 
                          alpha=0.8, linewidth=3)
                
                # 공통 심도 annotation 위치 계산 (다른 annotation들과 겹치지 않도록)
                common_annotation_x = (common_depth_start + common_depth_end) / 2
                common_annotation_y = ax.get_ylim()[1] * 0.9
                
                # 기존 annotation들과 겹치지 않는 위치 찾기
                offset_y = 0
                while any(abs(common_annotation_x - pos[0]) < 1.0 and abs(common_annotation_y + offset_y - pos[1]) < 50 
                          for pos in annotation_positions):
                    offset_y += 40
                
                annotation_positions.append((common_annotation_x, common_annotation_y + offset_y))
                
                # 공통 심도 annotation
                ax.annotate(f'Common Depth: {common_depth_start:.3f}~{common_depth_end:.3f}\n'
                           f'Depth Range: {common_depth_end - common_depth_start:.3f}', 
                           xy=(common_annotation_x, common_annotation_y), 
                           xytext=(0, offset_y), 
                           textcoords='offset points',
                           fontsize=12, 
                           color='purple',
                           weight='bold',
                           ha='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9))
                
                print(f"Common depth range: {common_depth_start:.3f} ~ {common_depth_end:.3f} mm")
                print(f"Depth range: {common_depth_end - common_depth_start:.3f} mm")
            else:
                print("No common depth range found.")
        
        z_unit = "mm"  # 사용자가 mm 단위로 입력했으므로 항상 mm로 표시
        ax.set_title(f'{metric_name}\nDepth of Field Analysis\n'
                    f'Depth Threshold: {depth_threshold:.3f}', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel(f'Z Position ({z_unit})', fontsize=12)
        ax.set_ylabel('Focus Value', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 파일 저장
        suffix = "_normalized" if use_normalized_z else "_absolute"
        filename = f'depth_of_field_analysis_{metric_name}{suffix}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved depth analysis: {filepath}")
        return str(filepath)
    
    def save_results(self, df_metric: pd.DataFrame, metric_name: str) -> str:
        """분석 결과를 CSV 파일로 저장합니다."""
        # 폴더 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f'{metric_name}_all_results.csv'
        filepath = self.output_dir / filename
        df_metric.to_csv(filepath, index=False)
        print(f"Saved: {filepath}")
        return str(filepath)
    
    def process_and_save_results(self, df_metric: pd.DataFrame, metric_name: str, 
                                use_normalized_z: bool = True, 
                                minimize_focus: bool = True, 
                                show_diagonal_only: bool = True,
                                progress_callback=None,
                                focus_analyzer=None,
                                auto_depth_calculation: bool = True,
                                depth_threshold: float = None) -> dict:
        """
        결과를 처리하고 모든 파일을 저장합니다.
        
        Returns:
            dict: 생성된 파일들의 경로
        """
        # 1. 원본 데이터 저장
        if progress_callback:
            progress_callback(25)
        csv_path = self.save_results(df_metric, metric_name)
        
        # 2. Quadratic fitting 분석
        if progress_callback:
            progress_callback(50)
        roi_analysis_results, best_positions_data = self.analyze_all_rois_with_quadratic_fitting(
            df_metric, metric_name, use_normalized_z, minimize_focus
        )
        
        generated_files = {
            'csv': csv_path,
            'best_position_map': "",
            'roi_profiles': "",
            'diagonal_roi_csv': "",
            'depth_analysis': ""
        }
        
        if roi_analysis_results and best_positions_data:
            # 3. Best position map 생성
            if progress_callback:
                progress_callback(75)
            best_map_path = self.create_best_position_map(best_positions_data, metric_name, use_normalized_z)
            generated_files['best_position_map'] = best_map_path
            
            # 4. ROI focus profiles 생성
            if progress_callback:
                progress_callback(90)
            result = self.create_roi_focus_profiles(
                roi_analysis_results, metric_name, use_normalized_z, show_diagonal_only, 
                focus_analyzer, auto_depth_calculation, depth_threshold, minimize_focus
            )
            
            # 반환값이 튜플인지 확인
            if isinstance(result, tuple):
                if len(result) == 3:
                    roi_profiles_path, diagonal_csv_path, depth_analysis_path = result
                    generated_files['roi_profiles'] = roi_profiles_path
                    generated_files['diagonal_roi_csv'] = diagonal_csv_path
                    generated_files['depth_analysis'] = depth_analysis_path
                elif len(result) == 2:
                    roi_profiles_path, diagonal_csv_path = result
                    generated_files['roi_profiles'] = roi_profiles_path
                    generated_files['diagonal_roi_csv'] = diagonal_csv_path
            else:
                generated_files['roi_profiles'] = result
        
        # 완료 시 100%로 설정
        if progress_callback:
            progress_callback(100)
        
        return generated_files
