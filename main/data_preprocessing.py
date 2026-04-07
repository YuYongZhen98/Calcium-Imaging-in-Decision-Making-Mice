#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三任务学习数据处理器 - 支持不同帧率插值
支持将28Hz数据插值到55Hz
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关库
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split

# 插值相关
from scipy import interpolate

class CellProcessedThreeTaskProcessor:
    """
    Cell维度处理版三任务学习数据处理器
    支持不同帧率数据的插值处理
    """
    
    def __init__(self, processed_folders: List[str], frame_rates: List[float],
                 start_second: float = 1.0, duration_seconds: float = 1.0,
                 cell_process_method: str = 'random_cut', target_frame_rate: float = 55.0,
                 random_state: int = 42):
        """
        初始化处理器
        
        Args:
            processed_folders: 三个数据集的文件夹路径列表
            frame_rates: 对应的帧率列表 [8_32kHz_Data帧率, 4_16kHz_Data帧率, 7_28kHz_Data帧率]
            start_second: 从第几秒开始截取
            duration_seconds: 截取持续时间（秒）
            cell_process_method: Cell处理方法 ('random_cut' 随机截取)
            target_frame_rate: 目标帧率，将所有数据统一到此帧率
            random_state: 随机种子
        """
        self.processed_folders = [Path(folder) for folder in processed_folders]
        self.frame_rates = frame_rates
        self.random_state = random_state
        np.random.seed(random_state)
        
        # 验证输入
        if len(self.processed_folders) != 3:
            raise ValueError("需要提供3个数据集的文件夹路径")
        if len(frame_rates) != 3:
            raise ValueError("需要提供3个帧率值")
            
        self.frame_rate_7_28 = frame_rates[2]  # 7_28kHz_Data帧率
        self.frame_rate_4_16 = frame_rates[1]  # 4_16kHz_Data帧率
        self.frame_rate_8_32 = frame_rates[0]    # 8_32kHz_Data帧率
        
        self.start_second = start_second
        self.duration_seconds = duration_seconds
        self.cell_process_method = cell_process_method
        self.target_frame_rate = target_frame_rate  # 目标帧率
        
        # 数据存储
        self.X_features = []  # 三维特征列表 (trials, cells, frames)
        self.y_labels = []    # 三维标签列表 (trials, 3_labels)
        self.file_info = []   # 文件信息
        
        # Cell维度统计
        self.cell_stats = {
            'min_cells': None,
            'max_cells': None,
            'mean_cells': None,
            'all_cells': []
        }
        
        # 数据处理器
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        
        print("\n初始化Cell维度处理三任务学习数据处理器")
        print(f"目标帧率: {self.target_frame_rate}Hz")
        print(f"数据集1: {self.processed_folders[0].name}, 帧率={self.frame_rate_8_32}Hz")
        print(f"数据集2: {self.processed_folders[1].name}, 帧率={self.frame_rate_4_16}Hz") 
        print(f"数据集3: {self.processed_folders[2].name}, 帧率={self.frame_rate_7_28}Hz")
    
    def truncate_time_series_by_frame_rate(self, features: np.ndarray, frame_rate: float, folder, number) -> np.ndarray:
        """
        根据帧率截取时间序列数据
        
        Args:
            features: 时间序列数据，形状为 (trials, cells, frames)
            frame_rate: 帧率 (Hz)
            
        Returns:
            截取后的时间序列数据
        """
        if features is None or features.size == 0:
            raise ValueError("特征数据为空")
        
        original_shape = features.shape
        n_frames = original_shape[-1]
        
        # 根据帧率计算截取参数
        start_frame = int(self.start_second * frame_rate)
        duration_frames = int(self.duration_seconds * frame_rate)
        end_frame = start_frame + duration_frames
        
        # 检查数据是否足够长
        if n_frames < end_frame:
            print(f"警告: 数据长度不足，期望{end_frame}帧，实际{n_frames}帧")
            end_frame = n_frames
            duration_frames = end_frame - start_frame
            
            if duration_frames <= 0:
                raise ValueError("截取后的数据长度为0")
        
        # 截取时间序列
        truncated_data = features[..., start_frame:end_frame]
        
        print(f"时间序列截取: {n_frames} -> {duration_frames} 帧")
        
        return truncated_data
    
    def interpolate_to_target_frame_rate(self, features: np.ndarray, original_frame_rate: float) -> np.ndarray:
        """
        将数据插值到目标帧率
        
        Args:
            features: 特征数据，形状为 (trials, cells, frames)
            original_frame_rate: 原始帧率
            
        Returns:
            插值后的特征数据，形状为 (trials, cells, target_frames)
        """
        if features is None or features.size == 0:
            raise ValueError("特征数据为空")
            
        if original_frame_rate == self.target_frame_rate:
            print(f"帧率相同({original_frame_rate}Hz)，无需插值")
            return features
        
        n_trials, n_cells, n_frames = features.shape
        
        # 计算目标时间点数
        target_frames = int(self.duration_seconds * self.target_frame_rate)
        
        # 创建原始时间轴和目标时间轴
        original_time = np.linspace(0, self.duration_seconds, n_frames)
        target_time = np.linspace(0, self.duration_seconds, target_frames)
        
        # 初始化插值后的数据
        interpolated_features = np.zeros((n_trials, n_cells, target_frames))
        
        print(f"插值处理: {original_frame_rate}Hz -> {self.target_frame_rate}Hz, {n_frames} -> {target_frames} 时间点")
        
        # 对每个trial和每个Cell进行插值
        for trial in range(n_trials):
            for cell in range(n_cells):
                # 提取当前trial和Cell的时间序列
                time_series = features[trial, cell, :]
                
                # 创建插值函数
                interp_func = interpolate.interp1d(
                    original_time, 
                    time_series, 
                    kind='linear',
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                
                # 应用插值
                interpolated_series = interp_func(target_time)
                interpolated_features[trial, cell, :] = interpolated_series
                
        return interpolated_features
    
    def process_cell_dimension(self, features: np.ndarray, method: str = None) -> np.ndarray:
        """
        处理Cell维度
        
        Args:
            features: 特征数据，形状为 (trials, cells, frames)
            method: 处理方法 ('random_cut' 随机截取)
            
        Returns:
            处理后的特征数据
        """
        if features is None or features.size == 0:
            raise ValueError("特征数据为空")
            
        if method is None:
            method = self.cell_process_method
            
        n_trials, n_cells, n_frames = features.shape
        
        if method == 'random_cut':
            # 随机截取：支持多种选择策略
            target_cells = self.cell_stats['min_cells']
            
            if n_cells >= target_cells:
                # 计算每个Cell的方差
                cell_variances = np.var(features, axis=(0, 2))  # 形状: (n_cells,)
                
                # 1. 动态计算高方差Cell的参考阈值
                # 使用方差的中位数作为基准阈值
                variance_median = np.median(cell_variances)
                variance_std = np.std(cell_variances)
                
                # 计算动态阈值：高于平均值+0.5倍标准差的Cell视为高方差
                high_variance_threshold = np.mean(cell_variances) + 0.5 * variance_std
                
                # 找出高方差Cell
                high_variance_mask = cell_variances > high_variance_threshold
                high_variance_indices = np.where(high_variance_mask)[0]
                n_high_var = len(high_variance_indices)
                
                # 2. 动态计算高分比例（完全基于数据方差特征）
                if n_high_var > 0:
                    # 基础比例：高方差Cell占总Cell的比例
                    high_var_ratio_base = n_high_var / n_cells
                    
                    # 考虑高方差Cell的质量（内部差异）：差异越大，说明可挑选的"优质"高方差Cell越多，可适当提高其比例
                    if n_high_var > 1:
                        high_var_variances = cell_variances[high_variance_indices]
                        high_var_variance_range = np.max(high_var_variances) - np.min(high_var_variances)
                        total_variance_range = np.max(cell_variances) - np.min(cell_variances)
                        # 避免除零
                        variance_range_normalized = high_var_variance_range / (total_variance_range + 1e-8)
                        
                        # 动态比例计算：基础比例 + 质量调整，系数0.3可根据经验微调
                        dynamic_high_ratio = high_var_ratio_base * (1.0 + 0.3 * variance_range_normalized)
                    else:
                        # 只有一个高方差Cell，则比例完全由基础比例决定
                        dynamic_high_ratio = high_var_ratio_base
                    
                    # 设置合理的边界，防止比例极端化
                    dynamic_high_ratio = min(0.7, max(0.2, dynamic_high_ratio))
                else:
                    # 没有明显高方差Cell，设定一个较低的默认比例，主要依赖随机选择保证多样性
                    dynamic_high_ratio = 0.1
                
                # 4. 计算实际要选择的高方差Cell数量
                n_high_var_to_select = max(1, int(target_cells * dynamic_high_ratio))
                
                # 确保不超过实际高方差Cell数量
                n_high_var_to_select = min(n_high_var_to_select, n_high_var, target_cells - 1)
                
                # 5. 选择高方差Cell
                if n_high_var_to_select > 0 and n_high_var > 0:
                    # 在高方差Cell中按方差排序，选择前n个
                    high_var_sorted_indices = high_variance_indices[
                        np.argsort(cell_variances[high_variance_indices])[::-1]
                    ]
                    selected_high_var = high_var_sorted_indices[:n_high_var_to_select]
                else:
                    selected_high_var = np.array([], dtype=int)
                
                # 6. 计算需要随机选择的数量
                n_random_needed = target_cells - len(selected_high_var)
                
                # 7. 从剩余Cell中随机选择
                all_indices = np.arange(n_cells)
                remaining_indices = np.setdiff1d(all_indices, selected_high_var)
                
                if len(remaining_indices) >= n_random_needed:
                    random_indices = np.random.choice(
                        remaining_indices, 
                        size=n_random_needed, 
                        replace=False
                    )
                else:
                    # 如果剩余Cell不足，从所有Cell中重新随机选择补足
                    random_indices = np.random.choice(
                        all_indices, 
                        size=n_random_needed, 
                        replace=False
                    )
                
                # 8. 合并索引
                selected_indices = np.concatenate([selected_high_var, random_indices])
                selected_indices = np.sort(selected_indices)
                
                # 选择对应的Cell
                processed_features = features[:, selected_indices, :]
                
                # 打印动态计算的比例信息
                print(f"Cell动态智能截取: {n_cells} -> {target_cells} ")
                print(f"  高方差Cell总数: {n_high_var}, 阈值: {high_variance_threshold:.4f}")
                print(f"  动态高分比例: {dynamic_high_ratio:.2%}, 高方差选择: {len(selected_high_var)}")
                print(f"  随机选择: {len(random_indices)}")            
            
            else:
                # 用零填充
                pad_width = ((0, 0), (0, target_cells - n_cells), (0, 0))
                processed_features = np.pad(features, pad_width, mode='constant', constant_values=0)
                print(f"Cell填充: {n_cells} -> {target_cells} (零填充)")
        
        else:
            processed_features = features
            print(f"使用原始Cell维度: {n_cells}")
        
        return processed_features
    
    def load_and_process_data(self, exclude_passive: bool = True, remove_label_2: bool = True):
        """
        加载并处理三个数据集的三任务学习数据
        
        Args:
            exclude_passive: 是否排除Passive文件（对照组）
            remove_label_2: 是否移除标签值为2的样本
        """
        print("=" * 60)
        print("开始处理三个数据集")
        print("=" * 60)
        
        # 清空之前的数据
        self.X_features = []
        self.y_labels = []
        self.file_info = []
        
        # 处理每个数据集
        total_files = 0
        successful_files = 0
        
        for dataset_idx, (folder, frame_rate) in enumerate(zip(self.processed_folders, self.frame_rates)):
            print(f"\n处理数据集 {dataset_idx+1}: {folder.name}")
            print(f"帧率: {frame_rate} Hz")
            
            # 检查文件夹是否存在
            if not folder.exists():
                print(f"警告: 文件夹不存在 - {folder}")
                continue
                
            # 查找所有processed_*.mat文件
            processed_files = []
            for file_path in folder.glob("processed_*.mat"):
                file_name = file_path.name
                
                if exclude_passive and 'Passive' in file_name:
                    continue
                
                processed_files.append(file_path)
            
            if not processed_files:
                print(f"在 {folder} 中未找到processed_*.mat文件")
                continue
            
            # 按照文件名中的数字排序
            def extract_session_number(file_path):
                """从文件名中提取数字部分用于排序"""
                import re
                file_name = file_path.name
                # 查找文件名中的数字，可能有多个数字，取第一个或多个组合
                numbers = re.findall(r'\d+', file_name)
                if numbers:
                    # 如果有多个数字，只取第一个作为排序依据
                    return int(numbers[0])
                else:
                    # 如果没有数字，返回一个很大的数使其排在最后
                    return float('inf')
            
            # 对文件列表进行排序
            processed_files.sort(key=extract_session_number)
            print("排序后的文件列表:")
            for i, file_path in enumerate(processed_files, 1):
                print(f"  {i}. {file_path.name}")
            
            total_files += len(processed_files)
            
            # 加载当前数据集的数据
            dataset_files = []  # 存储当前数据集的文件
            for file_path in processed_files:
                try:
                    mat_data = sio.loadmat(file_path)
                    file_name = file_path.name
                    
                    # 1. 提取特征（三维数据）
                    features = None
                    if 'Frequency_Action_Reward' in mat_data:
                        features = mat_data['Frequency_Action_Reward']  # (trials, cells, frames)
                    else:
                        print(f"跳过文件 {file_name}: 未找到Frequency_Action_Reward字段")
                        continue
                    
                    # 检查特征维度
                    if features.ndim != 3:
                        print(f"跳过文件 {file_name}: 特征维度不是3D")
                        continue
                        
                    if features.size == 0:
                        print(f"跳过文件 {file_name}: 特征数据为空")
                        continue
                    
                    print(f"处理文件: {file_name}, 形状: {features.shape}")
                    
                    parts = file_name.split('_')
                    for part in parts:
                        if part.startswith('Sess') and part[4:].isdigit():
                            number =  int(part[4:])
                    
                    # 2. 根据帧率截取时间序列
                    features_cropped = self.truncate_time_series_by_frame_rate(features, frame_rate, folder, number)
                    
                    # 3. 将数据插值到目标帧率
                    features_interpolated = self.interpolate_to_target_frame_rate(features_cropped, frame_rate)  
                    
                    # 4. 提取标签
                    if 'Labels' in mat_data:
                        label_data = mat_data['Labels']  # (trials, labels)
                    else:
                        print(f"跳过文件 {file_name}: 未找到Labels字段")
                        continue    
                                      
                    # 5. 创建标签矩阵 (trials, 2),获取Frequency数据（第一列）和Action数据（第二列）
                    labels_matrix = label_data[:, :2]  # 获取前两列        
                    
                    # 6. 过滤标签值为2的样本
                    if remove_label_2:
                        features_interpolated, labels_matrix = self.filter_label_2_samples(
                            features_interpolated, labels_matrix, file_name
                        )
                        
                        if features_interpolated.shape[0] == 0:
                            print(f"跳过文件 {file_name}: 过滤后无有效样本")
                            continue
                    
                    # 存储数据
                    self.X_features.append(features_interpolated)
                    self.y_labels.append(labels_matrix)
                    
                    # 文件信息
                    file_info_entry = {
                        'file_name': file_name,
                        'dataset_idx': dataset_idx + 1,
                        'dataset_name': folder.name,
                        'original_frame_rate': frame_rate,
                        'target_frame_rate': self.target_frame_rate,
                        'n_trials': features_interpolated.shape[0],
                        'n_cells': features_interpolated.shape[1],
                        'n_frames': features_interpolated.shape[2],
                        'original_frames': features.shape[2],
                        'feature_shape': features_interpolated.shape,
                        'label_shape': labels_matrix.shape,
                        'interpolated': frame_rate != self.target_frame_rate
                    }
                    self.file_info.append(file_info_entry)
                    successful_files += 1
                    
                    print(f"成功加载: {file_name}, 最终形状: {features_interpolated.shape}")
                    
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")
                    continue
                        
        
        if not self.X_features:
            print("错误: 没有成功加载任何数据")
            return None, None
        
        # 收集全局Cell统计信息
        self.collect_cell_stats()
        
        # 处理Cell维度
        X_processed = []
        for i, features in enumerate(self.X_features):
            processed_features = self.process_cell_dimension(features)
            X_processed.append(processed_features)
            
            # 更新文件信息
            self.file_info[i]['processed_feature_shape'] = processed_features.shape
            self.file_info[i]['processed_cells'] = processed_features.shape[1]
        
        self.X_features = X_processed
        
        
        print("\n数据加载和处理完成!")
        print(f"总文件数: {total_files}, 成功处理: {successful_files}")
        print(f"总样本数: {sum([x.shape[0] for x in self.X_features])}")
        
        # 显示数据统计信息
        self.display_data_statistics()
        
        return self.X_features, self.y_labels
    
    def display_data_statistics(self):
        """显示数据统计信息"""
        if not self.file_info:
            return
            
        print("\n数据统计信息:")
        print("-" * 40)
        
        # 按数据集统计
        for dataset_idx in range(1, 4):
            dataset_files = [info for info in self.file_info if info['dataset_idx'] == dataset_idx]
            if not dataset_files:
                continue
                
            total_trials = sum([info['n_trials'] for info in dataset_files])
            avg_cells = np.mean([info['n_cells'] for info in dataset_files])
            
            print(f"数据集 {dataset_idx}:")
            print(f"  文件数: {len(dataset_files)}")
            print(f"  总试次数: {total_trials}")
            print(f"  平均Cell数: {avg_cells:.1f}")
        
        # 全局统计
        total_trials = sum([info['n_trials'] for info in self.file_info])
        print("全局统计:")
        print(f"  总文件数: {len(self.file_info)}")
        print(f"  总试次数: {total_trials}")
        print(f"  Cell数量范围: {self.cell_stats['min_cells']} - {self.cell_stats['max_cells']}")
        print(f"  平均Cell数: {self.cell_stats['mean_cells']:.1f}")
    
    def collect_cell_stats(self):
        """收集所有文件的Cell数量统计"""
        if not self.X_features:
            return None
            
        all_cells = [x.shape[1] for x in self.X_features]
        self.cell_stats = {
            'min_cells': min(all_cells),
            'max_cells': max(all_cells),
            'mean_cells': np.mean(all_cells),
            'median_cells': np.median(all_cells),
            'all_cells': all_cells
        }
        
        print("n全局Cell统计:")
        print(f"  最小值: {self.cell_stats['min_cells']}")
        print(f"  最大值: {self.cell_stats['max_cells']}")
        print(f"  平均值: {self.cell_stats['mean_cells']:.1f}")
        print(f"  中位数: {self.cell_stats['median_cells']}")
        
        return self.cell_stats
     
    def filter_label_2_samples(self, features, labels_matrix, file_name=""):
        """过滤标签值为2的样本"""
        if features is None or labels_matrix is None:
            return features, labels_matrix
        
        n_before = features.shape[0]
        
        # 创建掩码：保留Action != 2 且 Frequency != 2 的样本
        mask_action = labels_matrix[:, 1] != 2  # Action != 2
        mask_trial = labels_matrix[:, 0] != 2   # Frequency != 2
        
        # 综合掩码：保留所有标签都不为2的样本
        mask = mask_action  & mask_trial
        
        # 应用过滤
        features_filtered = features[mask]
        labels_filtered = labels_matrix[mask]
        n_after = features_filtered.shape[0]
        
        if n_before != n_after:
            print(f"  过滤标签2: {file_name} 从{n_before}个样本减少到{n_after}个样本")
        
        return features_filtered, labels_filtered
    
    
    def unify_features(self, remove_label_2: bool = True):
        """
        统一特征维度 - 移除归一化，只进行数据合并
        """
        if not self.X_features or not self.y_labels:
            print("错误: 请先加载数据")
            return None, None
        
        # 合并所有数据
        X_unified = np.concatenate(self.X_features, axis=0)
        y_unified = np.concatenate(self.y_labels, axis=0)
         
        # 存储统一后的数据（不进行归一化）
        self.X_unified = X_unified
        self.y_unified = y_unified
        
        print(f"统一后特征形状: {X_unified.shape}")
        print(f"统一后标签形状: {y_unified.shape}")
        
        return X_unified, y_unified
    
    def create_complete_dataset_split(self, test_size: float = 0.2, val_size: float = 0.2, 
                                     random_state: int = 42, remove_label_2: bool = True):
        """
        创建完整的数据集划分：训练集、验证集、测试集
        """
        print("\n创建完整数据集划分...")
        
        # 统一特征（不进行归一化）
        X, y = self.unify_features(remove_label_2=remove_label_2)
        if X is None or y is None:
            print("错误: 数据统一失败")
            return None
        
        # 检查数据是否为空
        if X.shape[0] == 0:
            print("错误: 过滤后没有有效样本")
            return None
        
        # 选择分层抽样的列
        stratify_col = self._select_stratify_column(y)
        if stratify_col is not None:
            stratify = y[:, stratify_col]
        else:
            stratify = None
        
        # 首先划分训练+验证集和测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify
        )
        
        # 然后从训练+验证集中划分训练集和验证集
        val_ratio = val_size / (1 - test_size)
        
        if stratify is not None and len(y_train_val) > 0:
            stratify_train_val = y_train_val[:, stratify_col]
        else:
            stratify_train_val = None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=stratify_train_val
        )
        
        print("数据集划分完成:")
        print(f"训练集: {X_train.shape}, {y_train.shape}")
        print(f"验证集: {X_val.shape}, {y_val.shape}")
        print(f"测试集: {X_test.shape}, {y_test.shape}")
        
        # 创建数据集字典
        dataset_dict = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_shape': X_train.shape[1:],
            'label_names': ['Frequency', 'Action'],
            'split_info': {
                'test_size': test_size,
                'val_size': val_size,
                'random_state': random_state,
                'remove_label_2': remove_label_2,
                'total_samples': X.shape[0],
                'train_samples': X_train.shape[0],
                'val_samples': X_val.shape[0],
                'test_samples': X_test.shape[0]
            }
        }
        
        return dataset_dict
    
    def _select_stratify_column(self, y):
        """选择用于分层抽样的列
        返回列索引，而不是列名
        """
        max_classes = 0
        selected_col = None
        
        # 遍历所有列
        for i in range(y.shape[1]):
            unique_classes = np.unique(y[:, i])
            n_classes = len(unique_classes)
            
            if n_classes > 1 and n_classes > max_classes:
                max_classes = n_classes
                selected_col = i
        
        if selected_col is not None:
            # 可选：打印列的含义（如果需要知道是哪一列）
            col_meanings = {0: "Frequency", 1: "Action"}
            col_name = col_meanings.get(selected_col, f"列{selected_col}")
            print(f"使用第{selected_col}列({col_name})进行分层抽样，有{max_classes}个类别")
            return selected_col
        else:
            print("警告: 无法找到合适的分层列，使用随机抽样")
            return None
    
    def save_dataset_split(self, dataset_dict: dict, save_dir: str):
        """
        保存划分后的数据集
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存划分数据
        np.save(save_path / "X_train.npy", dataset_dict['X_train'])
        np.save(save_path / "X_val.npy", dataset_dict['X_val'])
        np.save(save_path / "X_test.npy", dataset_dict['X_test'])
        np.save(save_path / "y_train.npy", dataset_dict['y_train'])
        np.save(save_path / "y_val.npy", dataset_dict['y_val'])
        np.save(save_path / "y_test.npy", dataset_dict['y_test'])
        
        # 保存分割信息
        split_info = dataset_dict['split_info']
        with open(save_path / "split_info.txt", 'w') as f:
            for key, value in split_info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"数据集已保存到: {save_path}")
        return save_path
    

def create_dataset():
    """创建数据集的便捷函数"""
    try:
        # 设置三个数据集的路径和帧率
        processed_folders = [
            "./8_32kHz_Data/processed_mat_files",
            "./4_16kHz_Data/processed_mat_files", 
            "./7_28kHz_Data/processed_mat_files" 
        ]
        
        frame_rates = [55.0, 28.0, 28.0]
        output_dir = "."
        
        # 检查文件夹是否存在
        for folder in processed_folders:
            if not Path(folder).exists():
                print(f"警告: 文件夹不存在 - {folder}")
        
        # 创建处理器
        processor = CellProcessedThreeTaskProcessor(
            processed_folders=processed_folders,
            frame_rates=frame_rates,
            start_second=1.0,
            duration_seconds=1.0,
            cell_process_method='random_cut',
            target_frame_rate=55.0
        )
        
        # 加载并处理数据
        X_list, y_list = processor.load_and_process_data(
            exclude_passive=True,
            remove_label_2=True
        )
        
        if X_list is None:
            print("数据加载失败")
            return None, None
        
        # 创建数据集划分
        dataset_dict = processor.create_complete_dataset_split(
            test_size=0.2,
            val_size=0.2,
            random_state=42,
            remove_label_2=True
        )
        
        if dataset_dict is None:
            print("数据集划分失败")
            return None, None
        
        # 保存数据集
        save_path = Path(output_dir) / "split_data"
        processor.save_dataset_split(dataset_dict, save_path)
        
        return processor, dataset_dict
        
    except Exception as e:
        print(f"创建数据集时出错: {str(e)}")
        return None, None

if __name__ == "__main__":
    processor, dataset_dict = create_dataset()
    if processor and dataset_dict:
        print("数据集创建成功!")
    else:
        print("数据集创建失败!")