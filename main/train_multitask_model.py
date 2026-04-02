#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多任务学习模型训练脚本 
支持多任务和单任务训练，使用十折交叉验证并保存最佳模型
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import traceback

import warnings
# 设置环境变量，禁用matplotlib GUI
os.environ['MPLBACKEND'] = 'Agg'
# 抑制所有警告
warnings.filterwarnings("ignore")

# 设置路径
sys.path.append(str(Path(__file__).parent))

# 基础库
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 深度学习库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 导入模型
from models import create_model, MultiTaskMLP, MultiTaskCNN, MultiTaskLSTM

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================

class Config:
    """训练配置参数"""
    
    # 数据路径
    DATA_DIR = r"."
    
    # 数据集配置
    DATASET_CONFIG = {
        'split_dir': 'split_data',
        'normalization_method': 'standard',
        'handle_missing': 'mean',
        'feature_selection': None,
    }
    
    # 训练配置
    TRAIN_CONFIG = {
        'use_cross_validation': True,
        'n_splits': 10,  # 十折交叉验证
        'epochs': 200,
        'batch_size': 32,
        'random_seed': 42,
        'test_size': 0.1,
        'early_stopping_patience': 40,  # epochs的 10%-20%
    }
    
    # 要训练的模型类型
    # MODEL_TYPES = ["LSTM"]
    # MODEL_TYPES = ["CNN" ]
    MODEL_TYPES = ["MLP", "CNN", "LSTM", "SVM"]
    #MODEL_TYPES = ["MLP", "CNN", "LSTM", "RF", "DT", "SVM"]
    
    # 任务配置 - 支持多任务和单任务
    TASK_MODES = {
        'multi_task': ['Trial_Type', 'Action_choice'],  # 多任务模式
        'Trial_Type': ['Trial_Type'],       # 单任务模式 - Trial_Type
        'Action_choice': ['Action_choice']  # 单任务模式 - Action_choice
    }
    
    # 选择训练模式（可以同时选择多个）
    # TRAIN_MODES = ['Trial_Type']
    TRAIN_MODES = ['Trial_Type', 'Action_choice']
    # TRAIN_MODES = ['multi_task', 'Trial_Type', 'Action_choice']
    
    # 输出配置
    OUTPUT_CONFIG = {
        'save_dir': 'trained_models',
        'save_models': True,
        'generate_plots': True,  # 总控制开关
        'plot_training_curves': True,  # 单独控制训练曲线
        'plot_confusion_matrix': True,  # 单独控制混淆矩阵
        'plot_model_comparison': True,  # 单独控制模型比较
        'plot_mode_comparison': True,   # 单独控制模式比较
        'plot_format': 'png',
        'verbose': True,
    }


# ==================== 数据处理器 ====================

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, normalization_method='standard', handle_missing='mean',clip_method='tanh'):  # 新增clip_method参数
        self.normalization_method = normalization_method
        self.handle_missing = handle_missing
        
        if handle_missing == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        elif handle_missing == 'median':
            self.imputer = SimpleImputer(strategy='median')
        else:
            self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        
        if normalization_method == 'standard':
            self.normalizer = StandardScaler()
        elif normalization_method == 'minmax':
            self.normalizer = MinMaxScaler()
        elif normalization_method == 'robust':
            self.normalizer = RobustScaler()
        else:
            self.normalizer = None
            
        self.clip_method = clip_method  # 'none', 'hard', 'tanh', 'adaptive'
    
    def fit_transform(self, X_train, X_val=None, X_test=None):
        """拟合并转换数据 - 修改为保留时序结构"""
        
        def preserve_temporal_structure(X, name=""):
            """保留时序结构进行处理"""
            if X is None:
                return None, None
            
            original_shape = X.shape
            if X.ndim == 3:
                # 三维数据 (samples, rois, timepoints)
                # 重塑为 (samples * timepoints, rois) 进行归一化
                n_samples, n_rois, n_timepoints = X.shape
                X_reshaped = X.transpose(0, 2, 1).reshape(-1, n_rois)  # (samples*timepoints, rois)
                return X_reshaped, original_shape
            elif X.ndim == 2:
                # 二维数据直接处理
                return X, original_shape
            else:
                raise ValueError(f"不支持的维度: {X.ndim}")
        
        def restore_temporal_structure(X_processed, original_shape):
            """恢复时序结构"""
            if X_processed is None or original_shape is None:
                return X_processed
            
            if len(original_shape) == 3:
                n_samples, n_rois, n_timepoints = original_shape
                # 恢复为原始形状
                X_restored = X_processed.reshape(n_samples, n_timepoints, n_rois).transpose(0, 2, 1)
                return X_restored
            return X_processed
        
        # 保留时序结构
        X_train_reshaped, train_shape = preserve_temporal_structure(X_train, "训练集")
        X_val_reshaped, val_shape = preserve_temporal_structure(X_val, "验证集")
        X_test_reshaped, test_shape = preserve_temporal_structure(X_test, "测试集")
        
        # 处理缺失值
        X_train_filled = self.imputer.fit_transform(X_train_reshaped)
        X_val_filled = self.imputer.transform(X_val_reshaped) if X_val_reshaped is not None else None
        X_test_filled = self.imputer.transform(X_test_reshaped) if X_test_reshaped is not None else None
        
        # 应用z-score归一化
        if self.normalizer is not None:
            X_train_norm = self.normalizer.fit_transform(X_train_filled)
            X_val_norm = self.normalizer.transform(X_val_filled) if X_val_filled is not None else None
            X_test_norm = self.normalizer.transform(X_test_filled) if X_test_filled is not None else None
        else:
            X_train_norm = X_train_filled
            X_val_norm = X_val_filled
            X_test_norm = X_test_filled
        
        # 根据clip_method处理数据
        def apply_clip_method(X, method, device='cpu'):
            if X is None:
                return X
        
            if method == 'none':
                return X
            elif method == 'hard':
                # 使用PyTorch的clamp函数，支持GPU
                X_tensor = torch.as_tensor(X, device=device)
                return torch.clamp(X_tensor, -1.0, 1.0).cpu().numpy()  # 计算完移回CPU
            elif method == 'tanh':
                # 使用PyTorch的tanh函数，支持GPU
                X_tensor = torch.as_tensor(X, device=device)
                return torch.tanh(X_tensor).cpu().numpy()  # 计算完移回CPU
            elif method == 'adaptive':
                # 自适应缩放也可以用PyTorch实现
                if X.size > 0:
                    X_tensor = torch.as_tensor(X, device=device)
                    max_abs = torch.max(torch.abs(X_tensor))
                    if max_abs > 0:
                        safety_factor = 1.1
                        scale_factor = 1.0 / (max_abs * safety_factor)
                        return (X_tensor * scale_factor).cpu().numpy()
                return X
            else:
                raise ValueError(f"不支持的clip_method: {method}")
        # 在fit_transform函数内部，调用apply_clip_method之前
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 应用裁剪方法
        X_train_clipped = apply_clip_method(X_train_norm, self.clip_method, device)
        X_val_clipped = apply_clip_method(X_val_norm, self.clip_method) if X_val_norm is not None else None
        X_test_clipped = apply_clip_method(X_test_norm, self.clip_method) if X_test_norm is not None else None
        
        # 恢复时序结构
        X_train_processed = restore_temporal_structure(X_train_clipped, train_shape)
        X_val_processed = restore_temporal_structure(X_val_clipped, val_shape) if X_val_clipped is not None else None
        X_test_processed = restore_temporal_structure(X_test_clipped, test_shape) if X_test_clipped is not None else None
        
        return X_train_processed, X_val_processed, X_test_processed


class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.task_sizes = []
        self.class_names = []
        self.class_values = []
        self.current_task_mode = 'multi_task'  # 默认多任务模式
    
    def set_task_mode(self, task_mode):
        """设置任务模式"""
        self.current_task_mode = task_mode
        print(f"设置任务模式为: {task_mode}")
    
    def load_pre_split_dataset(self, split_dir="split_data"):
        """加载预先划分的数据集"""
        split_path = self.data_dir / split_dir
        
        if not split_path.exists():
            raise FileNotFoundError(f"划分数据目录不存在: {split_path}")
        
        try:
            self.X_train = np.load(split_path / "X_train.npy")
            self.X_val = np.load(split_path / "X_val.npy")
            self.X_test = np.load(split_path / "X_test.npy")
            self.y_train = np.load(split_path / "y_train.npy")
            self.y_val = np.load(split_path / "y_val.npy")
            self.y_test = np.load(split_path / "y_test.npy")
            
            # 根据任务模式选择标签列
            if self.y_train is not None and self.y_train.shape[1] >= 2:
                if self.current_task_mode == 'Trial_Type':
                    # 只保留Trial_Type任务（第一列）
                    self.y_train = self.y_train[:, 0:1]
                    self.y_val = self.y_val[:, 0:1]
                    self.y_test = self.y_test[:, 0:1]
                    print("单任务模式: 只保留Trial_Type任务")
                elif self.current_task_mode == 'Action_choice':
                    # 只保留Action_choice任务（第二列）
                    self.y_train = self.y_train[:, 1:2]
                    self.y_val = self.y_val[:, 1:2]
                    self.y_test = self.y_test[:, 1:2]
                    print("单任务模式: 只保留Action_choice任务")
                else:
                    # 多任务模式：保留前两个任务
                    self.y_train = self.y_train[:, :2]
                    self.y_val = self.y_val[:, :2]
                    self.y_test = self.y_test[:, :2]
                    print("多任务模式: 保留Trial_Type和Action_choice任务")
            
            print("成功加载预先划分的数据集")
            print(f"  训练集: X={self.X_train.shape}, y={self.y_train.shape}")
            print(f"  验证集: X={self.X_val.shape}, y={self.y_val.shape}")
            print(f"  测试集: X={self.X_test.shape}, y={self.y_test.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌❌❌❌ 加载数据集失败: {e}")
            return False
    
    def preprocess_data(self, normalization_method='standard', handle_missing='mean'):
        
        # 保存预处理前的第一个样本用于对比
        sample_before = self.X_train[0].copy() if len(self.X_train) > 0 else None
        print("预处理前第一个样本的值:")
        print(sample_before)
        
        """预处理数据"""
        processor = DataProcessor(normalization_method, handle_missing)
        self.X_train, self.X_val, self.X_test = processor.fit_transform(
            self.X_train, self.X_val, self.X_test
        )
        
        # 处理标签中的无效值
        self.y_train = np.nan_to_num(self.y_train, nan=-1)
        self.y_val = np.nan_to_num(self.y_val, nan=-1)
        self.y_test = np.nan_to_num(self.y_test, nan=-1)
        
        # 打印预处理后的样本对比
        if sample_before is not None and len(self.X_train) > 0:
           sample_after = self.X_train[0]
           print("\n预处理后第一个样本的值:")
           print(sample_after)      
        
        print("数据预处理完成")
        return True
    
    def get_data_for_cv(self):
        """获取用于交叉验证的数据（合并训练集和验证集）"""
        X_train_val = np.concatenate([self.X_train, self.X_val], axis=0)
        y_train_val = np.concatenate([self.y_train, self.y_val], axis=0)
        return X_train_val, y_train_val, self.X_test, self.y_test
    
    def analyze_labels(self, task_names):
        """分析标签分布"""
        if self.y_train is None:
            return [], [], []
        
        print(f"\n标签分析 (模式: {self.current_task_mode}):")
        task_sizes = []
        class_names = []
        class_values = []
        
        for i, task_name in enumerate(task_names):
            if i < self.y_train.shape[1]:
                # 获取唯一类别（排除无效值）
                unique_vals = np.unique(self.y_train[:, i])
                unique_vals = unique_vals[unique_vals != -1]
                unique_vals = np.sort(unique_vals)
                
                if len(unique_vals) == 0:
                    task_sizes.append(0)
                    class_names.append([])
                    class_values.append([])
                    continue
                
                task_sizes.append(len(unique_vals))
                class_values.append(unique_vals)
                
                # 生成类别名称
                if task_name == 'Trial_Type':
                    names = [f'Type{int(val)}' for val in unique_vals]
                elif task_name == 'Action_choice':
                    names = [f'Choice{int(val)}' for val in unique_vals]
                else:
                    names = [f'Class{int(val)}' for val in unique_vals]
                
                class_names.append(names)
                
                # 打印统计信息
                train_counts = [np.sum(self.y_train[:, i] == val) for val in unique_vals]
                val_counts = [np.sum(self.y_val[:, i] == val) for val in unique_vals] if self.y_val is not None else []
                
                print(f"  {task_name}: {len(unique_vals)}类, 值{unique_vals}")
                print(f"    训练集分布: {dict(zip(names, train_counts))}")
                if val_counts:
                    print(f"    验证集分布: {dict(zip(names, val_counts))}")
            else:
                task_sizes.append(0)
                class_names.append([])
                class_values.append([])
        
        self.task_sizes = task_sizes
        self.class_names = class_names
        self.class_values = class_values
        
        return task_sizes, class_names, class_values


# ==================== 模型训练器 ====================

class KFoldTrainer:
    """十折交叉验证训练器"""
    
    def __init__(self, model_type, input_size, task_sizes, device='cpu', 
                 random_seed=42, n_splits=10, task_mode='multi_task'):
        self.model_type = model_type
        self.input_size = input_size
        self.task_sizes = task_sizes
        self.device = device
        self.random_seed = random_seed
        self.n_splits = n_splits
        self.task_mode = task_mode  # 新增：任务模式

        
        # 设置随机种子
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if device == 'cuda':
            torch.cuda.manual_seed(random_seed)
        
        self.kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        self.best_models = []  # 存储每一折的最佳模型
        self.cv_results = {
            'train_scores': [], 'val_scores': [], 'test_scores': [],
            'best_val_scores': [], 'fold_info': [], 'fold_predictions': []
        }
    
    def get_model_args(self, X_shape):
        """获取模型特定参数"""
        model_args = {}
        
        if self.model_type == "CNN":
            model_args.update({
                'input_channels': 1,
                'input_height': X_shape[1],
                'input_width': X_shape[2]
            })
        elif self.model_type == "MLP":
            model_args.update({
                'hidden_sizes': [256, 128, 64],
                'dropout_rate': 0.3
            })
        elif self.model_type == "LSTM":
            # 注意：预处理后X_shape应该是(batch, 55, 69)
            # 所以sequence_length=55, input_size=69
            model_args.update({
                'hidden_size': 128,      # 从128增加到256
                'num_layers': 4,         # 从2层增加到3层
                'bidirectional': False,
                'dropout_rate': 0.1,
                'sequence_length': X_shape[1],  # 应该是55
                'use_attention': True    # 新增注意力机制
            })
        
        return model_args

    def get_weighted_criterion(self, y, task_names):
        """为每个任务创建加权的损失函数"""
        task_criterions = []
        
        for i, task_name in enumerate(task_names):
            if i < y.shape[1]:
                y_task = y[:, i]
                valid_mask = y_task != -1
                y_task_valid = y_task[valid_mask]
                
                if len(y_task_valid) > 0:
                    # 计算类别分布
                    unique, counts = np.unique(y_task_valid, return_counts=True)
                    
                    if len(unique) > 0:
                        # 对所有任务使用标准加权交叉熵
                        if len(unique) > 1 and min(counts) > 0:
                            weights = torch.FloatTensor([
                                len(y_task_valid) / (len(unique) * count) 
                                for count in counts
                            ]).to(self.device)
                            task_criterions.append(nn.CrossEntropyLoss(weight=weights))
                        else:
                            task_criterions.append(nn.CrossEntropyLoss())
                    else:
                        task_criterions.append(nn.CrossEntropyLoss())
                else:
                    task_criterions.append(nn.CrossEntropyLoss())
            else:
                task_criterions.append(nn.CrossEntropyLoss())
        
        return task_criterions
    
    def preprocess_data_for_model(self, X, model_args):
        """根据模型类型预处理数据"""
        if self.model_type == "CNN":
            # 确保是4D数据 [batch, channels, height, width]
            if X.ndim == 3:
                X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        elif self.model_type == "LSTM":
            # 重新组织维度: (batch, roi, time) -> (batch, time, roi)
            if X.ndim == 3:
                # 转置维度，将时间维度放在中间
                X = np.transpose(X, (0, 2, 1))  # 从(batch, 69, 55)变为(batch, 55, 69)
            elif X.ndim == 2:
                # 如果是展平的，尝试reshape
                # 假设我们知道原始形状，这里需要根据实际情况调整
                X = X.reshape(X.shape[0], self.sequence_length, -1)
        else:  # MLP和其他模型
            if X.ndim == 3:
                X = X.reshape(X.shape[0], -1)
        
        return X    
    
    def train_deep_learning_model(self, X_train, y_train, X_val, y_val, 
                                    epochs=100, batch_size=32, **model_args):
            """
            训练深度学习模型
            修改说明：
            1. 所有模型使用统一的初始学习率 (0.001)
            2. 添加 ReduceLROnPlateau 调度器，当验证准确率停滞时自动降低学习率
            3. 早停机制改为监控验证准确率 (而非损失)
            4. 改进训练信息输出，包含当前学习率
            """
            model = create_model(self.model_type, self.input_size, self.task_sizes, 
                               self.device, **model_args)
            
            # 转换为Tensor
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_train_tensor = torch.LongTensor(y_train).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            
            # 使用标准的损失函数
            if hasattr(self, 'config') and hasattr(self.config, 'TASK_NAMES'):
                criterion_list = self.get_weighted_criterion(y_train, self.config.TASK_NAMES)
            else:
                # 如果没有配置，使用标准损失函数
                criterion_list = [nn.CrossEntropyLoss() for _ in range(len(self.task_sizes))]
            
            # --- 早停与监控变量初始化 ---
            # 关键修改：早停基于验证准确率
            best_val_accuracy = 0.0
            best_model_state = None
            patience_counter = 0
            # 从配置中获取早停耐心值，默认为20
            patience = self.config.TRAIN_CONFIG.get('early_stopping_patience', 20) if hasattr(self, 'config') else 20
            
            train_losses = []
            val_losses = []
            val_accuracies_history = []  # 记录整体验证准确率的历史，用于分析
            
            # --- 优化器与调度器设置 ---
            # 关键修改1: 所有模型使用统一的初始学习率
            base_lr = 0.001
            optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-4)
            
            # 关键修改2: 添加学习率调度器
            # 监控验证准确率，当其在连续 `patience` 轮内不再提升时，将学习率乘以 `factor`
            try:
                # 尝试使用 verbose 参数（新版本）
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='max',
                    factor=0.5,
                    patience=8,
                    verbose=True,  # 新版本支持
                    min_lr=1e-6
                )
            except TypeError:
                # 如果报错，使用不包含 verbose 参数的版本（旧版本）
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='max',
                    factor=0.5,
                    patience=8,
                    min_lr=1e-6
                )
                # print("注意：当前 PyTorch 版本较旧，ReduceLROnPlateau 不支持 verbose 参数")
            
            # --- 训练循环 ---
            for epoch in range(epochs):
                model.train()
                epoch_losses = []
                
                # 随机打乱训练数据
                indices = torch.randperm(len(X_train_tensor))
                X_train_shuffled = X_train_tensor[indices]
                y_train_shuffled = y_train_tensor[indices]
                
                n_batches = len(X_train_shuffled) // batch_size
                if n_batches == 0:  # 确保至少有一个批次
                    n_batches = 1
                
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, len(X_train_shuffled))
                    
                    batch_X = X_train_shuffled[start_idx:end_idx]
                    batch_y = y_train_shuffled[start_idx:end_idx]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    # --- 训练损失计算 (加权平均) ---
                    task_losses = []
                    task_sample_counts = []  # 记录每个任务的样本数，用于加权
                    
                    for task_idx in range(len(self.task_sizes)):
                        if task_idx < batch_y.shape[1]:
                            valid_mask = batch_y[:, task_idx] != -1
                            valid_sample_count = valid_mask.sum().item()
                            
                            if valid_sample_count > 0 and task_idx < len(outputs):
                                # 使用任务特定的损失函数
                                task_loss = criterion_list[task_idx](
                                    outputs[task_idx][valid_mask], 
                                    batch_y[valid_mask, task_idx]
                                )
                                task_losses.append(task_loss)
                                task_sample_counts.append(valid_sample_count)
                    
                    if task_losses:
                        # 加权平均损失（按各任务有效样本数）
                        total_samples = sum(task_sample_counts)
                        weighted_loss = 0.0
                        
                        for loss, samples in zip(task_losses, task_sample_counts):
                            weighted_loss += loss * (samples / total_samples)
                        
                        total_loss = weighted_loss
                        
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        epoch_losses.append(total_loss.item())
                
                # 计算平均训练损失
                if epoch_losses:
                    avg_train_loss = np.mean(epoch_losses)
                    train_losses.append(avg_train_loss)
                else:
                    avg_train_loss = 0.0
                    train_losses.append(0.0)
                
                # --- 验证阶段 ---
                model.eval()
                with torch.no_grad():
                    outputs_val = model(X_val_tensor)
                    val_task_losses = []
                    val_task_sample_counts = []
                    task_accuracies = []  # 存储每个任务的准确率
                    
                    for task_idx in range(len(self.task_sizes)):
                        if task_idx < y_val_tensor.shape[1]:
                            valid_mask = y_val_tensor[:, task_idx] != -1
                            valid_sample_count = valid_mask.sum().item()
                            
                            if valid_sample_count > 0 and task_idx < len(outputs_val):
                                # 计算任务损失
                                task_loss = criterion_list[task_idx](
                                    outputs_val[task_idx][valid_mask], 
                                    y_val_tensor[valid_mask, task_idx]
                                )
                                val_task_losses.append(task_loss.item())
                                val_task_sample_counts.append(valid_sample_count)
                                
                                # 计算任务准确率
                                pred = torch.argmax(outputs_val[task_idx][valid_mask], dim=1)
                                true = y_val_tensor[valid_mask, task_idx]
                                accuracy = (pred == true).float().mean().item()
                                task_accuracies.append(accuracy)
                            else:
                                # 该任务在验证集中无有效样本，准确率记为0（不参与后续平均）
                                task_accuracies.append(0.0)
                        else:
                            task_accuracies.append(0.0)
                    
                    # 计算加权平均验证损失
                    if val_task_losses and val_task_sample_counts:
                        weighted_val_loss = 0.0
                        total_weight = 0
                        
                        for loss, sample_count in zip(val_task_losses, val_task_sample_counts):
                            weighted_val_loss += loss * sample_count
                            total_weight += sample_count
                        
                        if total_weight > 0:
                            avg_val_loss = weighted_val_loss / total_weight
                        else:
                            avg_val_loss = float('inf')
                    else:
                        avg_val_loss = float('inf')
                    
                    val_losses.append(avg_val_loss)
                    
                    # 计算整体验证准确率（只考虑有有效样本的任务）
                    valid_task_accuracies = [acc for acc in task_accuracies if acc > 0]
                    if valid_task_accuracies:
                        avg_val_accuracy = np.mean(valid_task_accuracies)
                    else:
                        avg_val_accuracy = 0.0
                    
                    val_accuracies_history.append(avg_val_accuracy)
                    
                    # 关键修改3: 更新学习率调度器（基于当前验证准确率）
                    scheduler.step(avg_val_accuracy)
                    current_lr = optimizer.param_groups[0]['lr']

                    # 可以记录上一次的学习率，比较是否有变化
                    if not hasattr(scheduler, 'last_lr'):
                        scheduler.last_lr = current_lr
                        
                    if current_lr != scheduler.last_lr:
                        print(f"学习率调整: {scheduler.last_lr:.6f} -> {current_lr:.6f}")
                        scheduler.last_lr = current_lr
                    
                    # 关键修改4: 早停逻辑（基于验证准确率是否提升）
                    if avg_val_accuracy > best_val_accuracy:
                        best_val_accuracy = avg_val_accuracy
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                        # 可选：在达到新的最佳准确率时打印提示
                        print(f"  新的最佳验证准确率: {best_val_accuracy:.4f}")
                    else:
                        patience_counter += 1
                
                # --- 训练信息输出 ---
                # 关键修改5: 增强输出信息，包含学习率
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"[Epoch {epoch+1:3d}/{epochs}]训练损失: {avg_train_loss:.4f}|验证损失: {avg_val_loss:.4f}|验证准确率: {avg_val_accuracy:.4f}|学习率: {current_lr:.6f}")
                    
                    # 详细任务准确率（仅当有配置时）
                    if hasattr(self, 'config') and hasattr(self.config, 'TASK_NAMES'):
                        for i, task_name in enumerate(self.config.TASK_NAMES[:len(task_accuracies)]):
                            if task_accuracies[i] > 0:  # 只打印有样本的任务
                                print(f"    {task_name}: {task_accuracies[i]:.4f}", end='  ')
                        if hasattr(self, 'config') and hasattr(self.config, 'TASK_NAMES'):
                            print()  # 换行
                
                # --- 早停判断 ---
                if patience_counter >= patience:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"早停触发于第 {epoch+1} 轮,连续 {patience} 轮验证准确率未提升,最终学习率: {current_lr:.6f}")
                    break
            
            # --- 训练结束，加载最佳模型 ---
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(f"加载第 {epoch+1-patience_counter} 轮的最佳模型状态。最佳验证准确率: {best_val_accuracy:.4f}")
            else:
                print("警告: 未找到最佳模型状态，将使用最终训练轮次的模型。")
            
            # 返回模型、损失历史和最佳验证准确率
            return model, train_losses, val_losses, best_val_accuracy
    
    def train_machine_learning_model(self, X_train, y_train, **model_args):
        """训练机器学习模型"""
        models = []
        
        for task_idx in range(len(self.task_sizes)):
            if task_idx >= y_train.shape[1]:
                models.append(None)
                continue
            
            y_task = y_train[:, task_idx]
            valid_mask = y_task != -1
            X_task = X_train[valid_mask]
            y_task_clean = y_task[valid_mask]
            
            if len(X_task) == 0:
                models.append(None)
                continue
            
            # 使用标准类别权重
            unique, counts = np.unique(y_task_clean, return_counts=True)
            if len(unique) > 1:
                # 计算标准类别权重
                class_weight_dict = {}
                total = len(y_task_clean)
                
                for cls, count in zip(unique, counts):
                    if count > 0:
                        # 使用标准逆频率
                        class_weight_dict[cls] = total / (len(unique) * count)
                    else:
                        class_weight_dict[cls] = 1.0
                
                class_weight = class_weight_dict
            else:
                class_weight = 'balanced'
            
            # 数据预处理
            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()
            
            X_processed = scaler.fit_transform(imputer.fit_transform(X_task))
            
            # 创建模型，使用当前训练器的随机种子
            if self.model_type == "RF":
                model = RandomForestClassifier(
                    n_estimators=150, 
                    max_depth=15,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight=class_weight,
                    random_state=self.random_seed  # 使用独立种子
                )
            elif self.model_type == "DT":
                model = DecisionTreeClassifier(
                    max_depth=15,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight=class_weight,
                    random_state=self.random_seed  # 使用独立种子
                )
            elif self.model_type == "SVM":
                model = SVC(
                    kernel='rbf', 
                    probability=True,
                    C=1.0,
                    gamma='scale',
                    class_weight=class_weight,
                    random_state=self.random_seed  # 使用独立种子
                )
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            try:
                model.fit(X_processed, y_task_clean)
                models.append({
                    'model': model, 
                    'imputer': imputer, 
                    'scaler': scaler
                })
            except Exception as e:
                print(f"训练任务{task_idx}失败: {e}")
                models.append(None)
        
        return models
    
    def evaluate_model(self, model, X, y, model_args):
        """评估模型性能"""
        try:
            if self.model_type in ["MLP", "CNN", "LSTM"]:
                # 深度学习模型评估
                X_tensor = torch.FloatTensor(X).to(self.device)
                model.eval()
                
                with torch.no_grad():
                    outputs = model(X_tensor)
                    predictions = []
                    
                    for i in range(len(self.task_sizes)):
                        if i < len(outputs):
                            pred = torch.argmax(outputs[i], dim=1)
                            predictions.append(pred.cpu().numpy())
                        else:
                            predictions.append(np.zeros(X.shape[0]))
                    
                    y_pred = np.column_stack(predictions)
            else:
                # 机器学习模型评估
                predictions = []
                for i, model_info in enumerate(model):
                    if model_info is None or i >= y.shape[1]:
                        predictions.append(np.zeros(X.shape[0]))
                        continue
                    
                    X_processed = model_info['scaler'].transform(
                        model_info['imputer'].transform(X)
                    )
                    pred = model_info['model'].predict(X_processed)
                    predictions.append(pred)
                
                y_pred = np.column_stack(predictions)
            
            # 计算准确率
            accuracies = []
            for i in range(len(self.task_sizes)):
                if i < y.shape[1] and i < y_pred.shape[1]:
                    valid_mask = y[:, i] != -1
                    if valid_mask.sum() > 0:
                        accuracy = accuracy_score(y[valid_mask, i], y_pred[valid_mask, i])
                        accuracies.append(accuracy)
                    else:
                        accuracies.append(0.0)
                else:
                    accuracies.append(0.0)
            
            overall_accuracy = np.mean(accuracies) if accuracies else 0.0
            return overall_accuracy, accuracies, y_pred
            
        except Exception as e:
            print(f"评估模型失败: {e}")
            return 0.0, [0.0] * len(self.task_sizes), np.zeros_like(y)
    
    def cross_validate(self, X, y, epochs=100, batch_size=32):
        """执行十折交叉验证"""
        print(f"\n开始{self.n_splits}折交叉验证 - {self.model_type} (模式: {self.task_mode})")
        
        # 为分层抽样选择第一个任务
        if y.shape[1] > 0:
            stratify_target = y[:, 0]
        else:
            stratify_target = np.arange(len(y))
        
        fold = 0
        for train_idx, val_idx in self.kf.split(X, stratify_target):
            fold += 1
            print(f"\n第 {fold}/{self.n_splits} 折")
            
            # 划分训练集和验证集
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
            
            # 获取模型参数
            model_args = self.get_model_args(X_train.shape)
            
            # 预处理数据
            X_train_processed = self.preprocess_data_for_model(X_train, model_args)
            X_val_processed = self.preprocess_data_for_model(X_val, model_args)
            
            # 训练模型
            if self.model_type in ["MLP", "CNN", "LSTM"]:
                model, train_losses, val_losses, best_val_loss = self.train_deep_learning_model(
                    X_train_processed, y_train, X_val_processed, y_val,
                    epochs=epochs, batch_size=batch_size, **model_args
                )
                
                # 评估训练集和验证集
                train_score, train_task_scores, y_train_pred = self.evaluate_model(
                    model, X_train_processed, y_train, model_args
                )
                val_score, val_task_scores, y_val_pred = self.evaluate_model(
                    model, X_val_processed, y_val, model_args
                )
                
                # 存储模型和结果
                self.best_models.append({
                    'model': model,
                    'model_args': model_args,
                    'val_score': val_score,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                })
                
            else:
                model = self.train_machine_learning_model(
                    X_train_processed, y_train, **model_args
                )
                
                # 评估训练集和验证集
                train_score, train_task_scores, y_train_pred = self.evaluate_model(
                    model, X_train_processed, y_train, model_args
                )
                val_score, val_task_scores, y_val_pred = self.evaluate_model(
                    model, X_val_processed, y_val, model_args
                )
                
                # 存储模型和结果
                self.best_models.append({
                    'model': model,
                    'model_args': model_args,
                    'val_score': val_score
                })
            
            # 存储折叠结果
            self.cv_results['train_scores'].append(train_score)
            self.cv_results['val_scores'].append(val_score)
            self.cv_results['best_val_scores'].append(val_score)
            self.cv_results['fold_info'].append({
                'fold': fold,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'train_score': train_score,
                'val_score': val_score
            })
            
            # 存储预测结果
            self.cv_results['fold_predictions'].append({
                'y_val_true': y_val,
                'y_val_pred': y_val_pred
            })
            
            print(f"训练准确率: {train_score:.4f}, 验证准确率: {val_score:.4f}")
        
        return self.cv_results
    
    def evaluate_on_test_set(self, X_test, y_test, fold_idx=None):
        """在测试集上评估模型（增强健壮性版本）"""
        try:
            # 1. 安全地选择最佳折叠
            if fold_idx is None:
                if not hasattr(self, 'cv_results') or not self.cv_results.get('best_val_scores'):
                    print("警告: 无交叉验证结果，使用第0折模型")
                    fold_idx = 0
                else:
                    # 找到验证集性能最好的折叠
                    best_fold = np.argmax(self.cv_results['best_val_scores'])
                    if best_fold < len(self.best_models):
                        fold_idx = best_fold
                        print(f"使用第 {fold_idx + 1} 折模型 (验证准确率: {self.cv_results['best_val_scores'][best_fold]:.4f})")
                    else:
                        print(f"警告: 最佳折叠索引 {best_fold} 超出范围，使用第0折")
                        fold_idx = 0
            
            # 2. 边界检查
            if fold_idx >= len(self.best_models):
                print(f"错误: 折叠索引 {fold_idx} 超出范围 (0-{len(self.best_models)-1})")
                return 0.0, [0.0] * len(self.task_sizes), np.zeros_like(y_test), 0
            
            # 3. 获取模型和参数
            best_model_info = self.best_models[fold_idx]
            if not best_model_info:
                print(f"错误: 第 {fold_idx} 折模型信息为空")
                return 0.0, [0.0] * len(self.task_sizes), np.zeros_like(y_test), fold_idx
            
            model = best_model_info['model']
            model_args = best_model_info.get('model_args', {})
            
            # 4. 预处理测试数据（与训练时保持一致）
            X_test_processed = self.preprocess_data_for_model(X_test, model_args)
            
            # 5. 评估模型
            test_score, test_task_scores, y_test_pred = self.evaluate_model(
                model, X_test_processed, y_test, model_args
            )
            
            print(f"测试集评估完成 - 准确率: {test_score:.4f}")
            return test_score, test_task_scores, y_test_pred, fold_idx
            
        except Exception as e:
            print(f"测试集评估失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, [0.0] * len(self.task_sizes), np.zeros_like(y_test), 0
    
    def save_best_model(self, save_dir, fold_idx=None):
        """保存最佳模型"""
        if fold_idx is None:
            best_fold = np.argmax(self.cv_results['best_val_scores'])
            fold_idx = best_fold
        
        if fold_idx >= len(self.best_models):
            print(f"警告: 折叠索引{fold_idx}超出范围，使用第0折")
            fold_idx = 0
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_info = self.best_models[fold_idx]
        model = model_info['model']
        model_args = model_info['model_args']
        
        # 保存模型
        if self.model_type in ["MLP", "CNN", "LSTM"]:
            # 保存深度学习模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': self.model_type,
                'model_args': model_args,
                'input_size': self.input_size,
                'task_sizes': self.task_sizes,
                'device': self.device,
                'val_score': model_info['val_score'],
                'fold_idx': fold_idx,
                'task_mode': self.task_mode  # 保存任务模式
            }, save_path / "best_model.pth")
        else:
            # 保存机器学习模型
            with open(save_path / "best_model.pkl", 'wb') as f:
                pickle.dump({
                    'model': model,
                    'model_type': self.model_type,
                    'model_args': model_args,
                    'val_score': model_info['val_score'],
                    'fold_idx': fold_idx,
                    'task_mode': self.task_mode  # 保存任务模式
                }, f)
        
        # 保存交叉验证结果
        cv_results_path = save_path / "results.json"
        with open(cv_results_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python原生类型
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            serializable_results = convert_to_serializable(self.cv_results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"最佳模型已保存到: {save_path}")
        print(f"使用第{fold_idx + 1}折模型，验证集准确率: {model_info['val_score']:.4f}")
    
    def plot_training_curves(self, save_path=None):
        """绘制训练曲线（仅深度学习模型）"""
        if self.model_type not in ["MLP", "CNN", "LSTM"]:
            print("训练曲线仅适用于深度学习模型")
            return
        
        if not any('train_losses' in model_info for model_info in self.best_models):
            print("无训练损失数据可绘制")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.best_models)))
        
        for fold_idx, (model_info, color) in enumerate(zip(self.best_models, colors)):
            train_losses = model_info.get('train_losses', [])
            val_losses = model_info.get('val_losses', [])
            
            if train_losses:
                axes[0].plot(train_losses, color=color, alpha=0.7, 
                            label=f'Fold {fold_idx+1}', linewidth=2)
            if val_losses:
                axes[1].plot(val_losses, color=color, alpha=0.7, 
                            label=f'Fold {fold_idx+1}', linewidth=2)
        
        axes[0].set_title('训练损失曲线')
        axes[0].set_xlabel('训练轮次')
        axes[0].set_ylabel('损失')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('验证损失曲线')
        axes[1].set_xlabel('训练轮次')
        axes[1].set_ylabel('损失')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, task_names, save_path=None):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix
        
        n_tasks = min(len(task_names), y_true.shape[1])
        if n_tasks == 0:
            return
        
        fig, axes = plt.subplots(1, n_tasks, figsize=(5*n_tasks, 4))
        if n_tasks == 1:
            axes = [axes]
        
        for i, task_name in enumerate(task_names[:n_tasks]):
            if i >= y_true.shape[1] or i >= y_pred.shape[1]:
                continue
            
            # 处理无效值
            valid_mask = y_true[:, i] != -1
            if valid_mask.sum() == 0:
                continue
            
            y_true_task = y_true[valid_mask, i]
            y_pred_task = y_pred[valid_mask, i]
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_true_task, y_pred_task)
            
            # 获取唯一类别
            unique_classes = np.unique(np.concatenate([y_true_task, y_pred_task]))
            
            # 根据任务名称定义标签映射
            if task_name == 'Trial_Type':
                # 映射: 0 -> low, 1 -> high
                label_map = {0: 'low', 1: 'high'}
            elif task_name == 'Action_choice':
                # 映射: 0 -> left, 1 -> right
                label_map = {0: 'left', 1: 'right'}
            else:
                # 默认映射，使用数字标签
                label_map = {cls: f'Class{cls}' for cls in unique_classes}
            
            # 生成标签列表，确保顺序与混淆矩阵的行列一致
            labels = [label_map.get(cls, f'Class{cls}') for cls in unique_classes]
            
            # 归一化
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
            # 绘制热图
            im = axes[i].imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
            # axes[i].set_title(f'{task_name}')
            
            # 设置刻度
            tick_marks = np.arange(len(unique_classes))
            axes[i].set_xticks(tick_marks)
            axes[i].set_yticks(tick_marks)
            
            # 设置刻度标签
            axes[i].set_xticklabels(labels,  ha='center')
            axes[i].set_yticklabels(labels)
            
            # 添加数值标签
            thresh = cm_normalized.max() / 2.
            for j in range(len(cm)):
                for k in range(len(cm)):
                    axes[i].text(k, j, f'{cm[j, k]}\n({cm_normalized[j, k]:.2f})',
                               ha="center", va="center",
                               color="white" if cm_normalized[j, k] > thresh else "black",
                               fontsize=8)
            
            axes[i].set_ylabel('True')
            axes[i].set_xlabel('Predicted')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            # 确保目录存在
            dir_name = os.path.dirname(save_path) or '.'
            os.makedirs(dir_name, exist_ok=True)
            
            # 生成文件名
            base_name = f'{task_name}_{os.path.basename(save_path)}'
            
            png_path = os.path.join(dir_name, f'{base_name}.png')
            pdf_path = os.path.join(dir_name, f'{base_name}.pdf')
            
            plt.savefig(png_path, dpi=300, bbox_inches='tight', transparent=False)
            plt.savefig(pdf_path, bbox_inches='tight', transparent=False)
            print(f"模型比较图已保存到:\n  PNG: {png_path}\n  PDF: {pdf_path}")
        
        plt.show()
        

# ==================== 主训练器 ====================

class MultiTaskTrainer:
    """多任务学习主训练器 - 支持单任务和多任务"""
    
    def __init__(self, config):
        self.config = config
        self.dataset_manager = DatasetManager(self.config.DATA_DIR)
        self.all_results = {}
        self.best_overall_model = None
        self.output_dir = None
        self.current_task_mode = 'multi_task'  # 当前任务模式
        # 添加绘图控制
        self.plots_generated = False
        self.comparison_plot_generated = False
    
    def set_task_mode(self, task_mode):
        """设置任务模式"""
        self.current_task_mode = task_mode
        self.dataset_manager.set_task_mode(task_mode)
    
    def setup(self):
        """设置训练环境"""
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据目录: {self.config.DATA_DIR}")
        print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"任务模式: {self.current_task_mode}")
        
        # 创建输出目录
        self.output_dir = Path(self.config.DATA_DIR) / self.config.OUTPUT_CONFIG['save_dir'] / self.current_task_mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {self.output_dir}")
        # ============ 新增：一次性绘图环境检测与补丁 ============
        self._patch_plotting_for_headless_env()
        # ============ 新增结束 ============ 
        return True
    
    
    def _patch_plotting_for_headless_env(self):
        """
        检测当前环境是否支持图形化显示。
        如果不支持（无GUI环境），则“补丁”matplotlib.pyplot.show使其静默。
        这样后续所有绘图代码在调用plt.show()时不会报错，且仍能通过plt.savefig保存图片。
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            
            # 检测当前后端是否支持交互显示
            # 常见的无头（无GUI）后端: 'Agg', 'pdf', 'svg', 'PS', 'Cairo'
            current_backend = matplotlib.get_backend().lower()
            headless_backends = ['agg', 'pdf', 'svg', 'ps', 'cairo']
            
            # 检查是否在可能无DISPLAY的环境（如Linux服务器）中运行
            is_headless_env = (current_backend in headless_backends) or (os.name == 'posix' and 'DISPLAY' not in os.environ)
            
            if is_headless_env:
                print("  检测到无GUI环境，已禁用图形弹出显示。图像仍将保存至文件。")
                # 关键补丁：将plt.show替换为一个什么都不做的函数
                plt.show = lambda *args, **kwargs: None
                
        except ImportError:
            # 如果matplotlib都未安装，则完全禁用绘图
            print("  警告: matplotlib 不可用，将跳过所有绘图操作。")
            # 通过将总开关设为False，让后续所有绘图函数提前返回
            self.config.OUTPUT_CONFIG['generate_plots'] = False
        except Exception as e:
            # 其他意外错误，保守起见也禁用显示
            print(f"  绘图环境检测时遇到意外问题({e})，将禁用图形显示。")
            try:
                import matplotlib.pyplot as plt
                plt.show = lambda *args, **kwargs: None
            except:
                pass
    
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("\n1. 加载和预处理数据...")
        
        # 加载数据
        if not self.dataset_manager.load_pre_split_dataset(
            self.config.DATASET_CONFIG['split_dir']
        ):
            return False
        
        # 预处理数据（使用z-score归一化并保留时序信息）
        if not self.dataset_manager.preprocess_data(
            normalization_method=self.config.DATASET_CONFIG['normalization_method'],
            handle_missing=self.config.DATASET_CONFIG['handle_missing']
        ):
            return False
        
        # 分析标签
        task_names = self.config.TASK_MODES[self.current_task_mode]
        task_sizes, class_names, class_values = self.dataset_manager.analyze_labels(task_names)
        
        if not any(task_sizes):
            print("错误: 没有有效的任务数据")
            return False
        
        print(f"数据加载和预处理完成")
        return True
    
    def determine_input_size(self, X, model_type):
        """确定输入特征维度 - 修复版"""
        if X.ndim == 3:
            batch_size, seq_len, feature_dim = X.shape
            
            if model_type == "LSTM":
                # 注意：预处理后LSTM输入应该是(batch, 55, 69)
                # 所以input_size应该是69（ROI数量），sequence_length是55
                input_size = feature_dim  # 应该是69，而不是55
            elif model_type == "CNN":
                # CNN需要的是序列长度和特征维度
                input_size = seq_len * feature_dim
            else:  # MLP和其他模型
                # MLP需要的是展平后的总特征数
                input_size = seq_len * feature_dim  # 69 * 55 = 3795
        elif X.ndim == 2:
            input_size = X.shape[1]
        else:
            input_size = np.prod(X.shape[1:])
        
        print(f"模型类型: {model_type}, 输入数据形状: {X.shape}, 输入维度: {input_size}")
        return input_size
        
    def train_single_model(self, model_type, random_seed=None):
        """训练单个模型 - 优化版"""
        print(f"\n{'='*60}")
        print(f"训练模型: {model_type} (模式: {self.current_task_mode})")
        print(f"{'='*60}")
        
        # 如果未提供随机种子，则使用配置中的默认种子
        if random_seed is None:
            random_seed = self.config.TRAIN_CONFIG['random_seed']
        
        # 获取数据
        X_train_val, y_train_val, X_test, y_test = self.dataset_manager.get_data_for_cv()
        
        # 确定输入大小（根据模型类型）
        input_size = self.determine_input_size(X_train_val, model_type)
        
        # 获取任务大小
        task_sizes = self.dataset_manager.task_sizes
        
        # 创建训练器，传入独立随机种子
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = KFoldTrainer(
            model_type=model_type,
            input_size=input_size,
            task_sizes=task_sizes,
            device=device,
            random_seed=random_seed,  # 使用传入的独立种子
            n_splits=self.config.TRAIN_CONFIG['n_splits'],
            task_mode=self.current_task_mode
        )
        
        # 传递配置
        trainer.config = self.config
        
        # 执行交叉验证
        cv_results = trainer.cross_validate(
            X=X_train_val,
            y=y_train_val,
            epochs=self.config.TRAIN_CONFIG['epochs'],
            batch_size=self.config.TRAIN_CONFIG['batch_size']
        )
        
        # 在测试集上评估
        test_score, test_task_scores, y_pred, best_fold = trainer.evaluate_on_test_set(X_test, y_test)
        
        print(f" {model_type} 训练完成")
        print(f"测试集准确率: {test_score:.4f}")
        
        # 保存模型
        if self.config.OUTPUT_CONFIG['save_models']:
            model_save_dir = self.output_dir / model_type
            trainer.save_best_model(model_save_dir, best_fold)
            
            # 只在需要时绘制图表
            if self.config.OUTPUT_CONFIG['generate_plots']:
                # 绘制训练曲线（仅深度学习模型）
                if model_type in ["MLP", "CNN", "LSTM"] and self.config.OUTPUT_CONFIG['plot_training_curves']:
                    trainer.plot_training_curves(model_save_dir / "training_curves.png")
                
                # 绘制混淆矩阵
                if self.config.OUTPUT_CONFIG['plot_confusion_matrix']:
                    task_names = self.config.TASK_MODES[self.current_task_mode]
                    trainer.plot_confusion_matrix(
                        y_test, y_pred, task_names,
                        model_save_dir / f'{model_type}_confusion_matrix'
                    )
        
        return trainer, {
            'cv_results': cv_results,
            'test_score': test_score,
            'test_task_scores': test_task_scores,
            'y_pred': y_pred,
            'best_fold': best_fold
        }
    
    def train_all_models(self):
        """训练所有模型"""
        print("\n2. 训练所有模型...")
        
        successful_models = 0
        
        for model_idx, model_type in enumerate(self.config.MODEL_TYPES):
            try:
                # ============ 在训练每个新模型前，进行环境清理 ============
                import gc
                import random
                # 1. 清空PyTorch的CUDA缓存，释放上一个模型占用的显存
                torch.cuda.empty_cache()
                # 2. 调用Python的垃圾回收，释放不再使用的Python对象
                gc.collect()
                # 3. 为每个模型生成独立的随机种子
                base_seed = self.config.TRAIN_CONFIG['random_seed']
                # 使用模型索引来生成不同的种子，避免冲突
                model_specific_seed = base_seed + model_idx * 1000
                # 重置所有随机源
                random.seed(model_specific_seed)
                np.random.seed(model_specific_seed)
                torch.manual_seed(model_specific_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(model_specific_seed)
                # 4. 重置CuDNN的底层算法选择器，避免其缓存影响
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                # ============ 环境清理结束 ============
                trainer, results = self.train_single_model(model_type, model_specific_seed)
                self.all_results[model_type] = {
                    'trainer': trainer,
                    'results': results
                }
                
                # 更新最佳模型
                if (self.best_overall_model is None or 
                    results['test_score'] > self.best_overall_model['test_score']):
                    self.best_overall_model = {
                        'model_type': model_type,
                        'test_score': results['test_score'],
                        'trainer': trainer,
                        'results': results
                    }
                
                successful_models += 1
                print(f"{model_type} 模型训练成功")
            except Exception as e:
                print(f" {model_type} 模型训练失败: {e}")
                if self.config.OUTPUT_CONFIG['verbose']:
                    traceback.print_exc()
        
        print(f"\n训练完成: {successful_models}/{len(self.config.MODEL_TYPES)} 个模型成功")
        return successful_models > 0
    
    def analyze_results(self):
        """分析训练结果 - 优化版"""
        if not self.all_results:
            print("无训练结果可分析")
            return
        
        print("\n3. 分析训练结果...")
        
        # 模型比较
        comparison_data = []
        for model_type, data in self.all_results.items():
            results = data['results']
            cv_results = results['cv_results']
            
            comparison_data.append({
                'model_type': model_type,
                'mean_cv_val': np.mean(cv_results['val_scores']) if cv_results['val_scores'] else 0,
                'std_cv_val': np.std(cv_results['val_scores']) if cv_results['val_scores'] else 0,
                'test_score': results['test_score'],
                'best_fold': results['best_fold']
            })
        
        # 按测试集准确率排序
        comparison_data.sort(key=lambda x: x['test_score'], reverse=True)
        
        # 打印比较结果
        print(f"\n模型性能比较（模式：{self.current_task_mode}）:")
        print("-" * 60)
        print(f"{'模型':<8} {'验证准确率':<20} {'测试准确率':<12} {'最佳折数':<10}")
        print("-" * 60)
        
        for data in comparison_data:
            val_str = f"{data['mean_cv_val']:.4f}±{data['std_cv_val']:.4f}"
            
            print(f"{data['model_type']:<8} "
                  f"{val_str:<20} "
                  f"{data['test_score']:12.4f} "
                  f"{data['best_fold'] + 1:<10}")
        
        # 绘制比较图
        if self.config.OUTPUT_CONFIG['generate_plots'] and self.config.OUTPUT_CONFIG['plot_model_comparison']:
            self.plot_model_comparison(comparison_data, self.output_dir / "model_comparison.png")
        
        return comparison_data

    
    def plot_model_comparison(self, comparison_data, save_path=None):
        """绘制模型比较图"""
        if not comparison_data:
            return
        
        # 调整图形大小和布局
        fig, ax = plt.subplots(figsize=(10, 7))
        
        model_names = [data['model_type'] for data in comparison_data]
        test_scores = [data['test_score'] for data in comparison_data]
        cv_scores = [data['mean_cv_val'] for data in comparison_data]
        cv_stds = [data['std_cv_val'] for data in comparison_data]
        
        # 减小x轴上的点间距
        x = np.arange(len(model_names))
        
        # 减小柱子宽度，增加紧凑感
        width = 0.22  # 从0.35减小到0.22
        
        # 默认配色
        test_color = '#0066CC'  # 深蓝色
        verify_color = '#CC6600'  # 深橙色
        
        # 转换为百分比
        test_scores_pct = [score * 100 for score in test_scores]
        cv_scores_pct = [score * 100 for score in cv_scores]
        cv_stds_pct = [std * 100 for std in cv_stds]
        
        # 绘制条形图
        bars1 = ax.bar(x - width/1.5, test_scores_pct, width, 
                       label='Test', alpha=0.8, color=test_color,
                       edgecolor='black', linewidth=1)
        
        # 减小误差条帽子宽度
        bars2 = ax.bar(x + width/1.5, cv_scores_pct, width, 
                       yerr=cv_stds_pct, label='Validation', 
                       alpha=0.8, capsize=3, color=verify_color,  # capsize从5减小到3
                       edgecolor='black', linewidth=1)
        
        # 设置误差条颜色为黑色
        try:
            # 获取坐标轴中所有线条
            lines = ax.get_lines()
            # 设置误差条颜色
            for line in lines[-len(model_names)*2:]:
                line.set_color('black')
        except Exception as e:
            print(f"设置误差条颜色时出错: {e}")
        
        # 设置标签和标题
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        
        # 设置刻度
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, ha='center', fontsize=11)
        ax.tick_params(axis='both', labelsize=11)
        
        # 设置网格和坐标轴
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 添加图例
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        # 添加数值标签 - 更小的字体，避免覆盖
        for bar, score in zip(bars1, test_scores_pct):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{score:.2f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')  # 字体从10减小到9
        
        for bar, score in zip(bars2, cv_scores_pct):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{score:.2f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')  # 字体从10减小到9
        
        # 设置y轴范围
        max_score = max(max(test_scores_pct), max(cv_scores_pct))
        ax.set_ylim(0, max(100, max_score + 15))  # 增加一些顶部空间
        
        # 添加百分比符号到y轴刻度
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.0f}%'))
        
        # 调整x轴范围，让柱子看起来更紧凑
        ax.set_xlim(-0.5, len(model_names)-0.5)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            # 确保目录存在
            dir_name = os.path.dirname(save_path) or '.'
            os.makedirs(dir_name, exist_ok=True)
            
            # 生成文件名
            base_name = f'{self.current_task_mode}_Model_performance_comparison'
            
            png_path = os.path.join(dir_name, f'{base_name}.png')
            pdf_path = os.path.join(dir_name, f'{base_name}.pdf')
            
            plt.savefig(png_path, dpi=300, bbox_inches='tight', transparent=False)
            plt.savefig(pdf_path, bbox_inches='tight', transparent=False)
            print(f"模型比较图已保存到:\n  PNG: {png_path}\n  PDF: {pdf_path}")
        
        # 显示图表
        plt.show()
        
        # 可选：关闭图形以释放内存
        plt.close(fig)
        
    def generate_final_report(self):
        """生成最终报告"""
        print("\n4. 生成最终报告...")
        
        report_path = self.output_dir / "training_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("多任务学习模型训练报告\n")
            f.write("="*60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据目录: {self.config.DATA_DIR}\n")
            f.write(f"输出目录: {self.output_dir}\n")
            f.write(f"任务模式: {self.current_task_mode}\n\n")
            
            # 训练配置
            f.write("训练配置:\n")
            f.write("-"*40 + "\n")
            f.write(f"模型类型: {', '.join(self.config.MODEL_TYPES)}\n")
            f.write(f"交叉验证折数: {self.config.TRAIN_CONFIG['n_splits']}\n")
            f.write(f"训练轮数: {self.config.TRAIN_CONFIG['epochs']}\n")
            f.write(f"批量大小: {self.config.TRAIN_CONFIG['batch_size']}\n")
            f.write(f"任务列表: {', '.join(self.config.TASK_MODES[self.current_task_mode])}\n\n")
            
            # 训练结果
            f.write("训练结果:\n")
            f.write("-"*40 + "\n")
            f.write(f"成功训练的模型: {len(self.all_results)}/{len(self.config.MODEL_TYPES)}\n")
            
            if self.best_overall_model:
                f.write(f"最佳模型: {self.best_overall_model['model_type']}\n")
                f.write(f"最佳模型测试准确率: {self.best_overall_model['test_score']:.4f}\n\n")
            
            # 详细结果
            f.write("详细结果:\n")
            f.write("-"*40 + "\n")
            for model_type, data in self.all_results.items():
                results = data['results']
                f.write(f"{model_type}:\n")
                f.write(f"  测试集准确率: {results['test_score']:.4f}\n")
                f.write(f"  最佳折数: {results['best_fold'] + 1}\n")
                if 'test_task_scores' in results:
                    task_names = self.config.TASK_MODES[self.current_task_mode]
                    for i, task_name in enumerate(task_names):
                        if i < len(results['test_task_scores']):
                            f.write(f"  {task_name}: {results['test_task_scores'][i]:.4f}\n")
            print(f"训练报告已生成: {report_path}")
    
    def run_complete_pipeline(self):
        """运行完整的训练流程"""
        try:
            print("="*60)
            print(f"开始完整的训练流程 (模式: {self.current_task_mode})")
            print("="*60)
            
            # 1. 设置环境
            if not self.setup():
                return False
            
            # 2. 加载和预处理数据
            if not self.load_and_preprocess_data():
                return False
            
            # 3. 训练所有模型
            if not self.train_all_models():
                return False
            
            # 4. 分析结果
            comparison_data = self.analyze_results()
            
            # 5. 生成报告
            self.generate_final_report()
            
            # 6. 打印总结
            self.print_summary()
            
            return True
            
        except Exception as e:
            print(f"训练流程出错: {e}")
            traceback.print_exc()
            return False
    
    def print_summary(self):
        """打印训练总结"""
        print("\n" + "="*60)
        print("训练流程完成!")
        print("="*60)
        
        if self.best_overall_model:
            print(f"最佳模型: {self.best_overall_model['model_type']}")
            print(f"测试准确率: {self.best_overall_model['test_score']:.4f}")
        
        print(f"成功训练模型: {len(self.all_results)}/{len(self.config.MODEL_TYPES)}")
        print(f"结果保存到: {self.output_dir}")
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        

# ==================== 多模式训练器 ====================

class MultiModeTrainer:
    """多模式训练器 - 支持多任务和单任务训练"""
    
    def __init__(self, config):
        self.config = config
        self.all_results = {}  # 存储所有模式的结果
        self.best_models_by_mode = {}  # 每个模式的最佳模型
        self.output_dir = None
        self.mode_comparison_plot_generated = False
    
    def setup(self):
        """设置训练环境"""
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据目录: {self.config.DATA_DIR}")
        print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # 创建输出目录
        self.output_dir = Path(self.config.DATA_DIR) / self.config.OUTPUT_CONFIG['save_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {self.output_dir}")
        
        return True
    
    def train_all_modes(self):
        """训练所有模式（多任务和单任务）"""
        print("\n开始多模式训练...")
        print(f"训练模式: {', '.join(self.config.TRAIN_MODES)}")
        
        successful_modes = 0
        
        for mode in self.config.TRAIN_MODES:
            try:
                print(f"\n{'='*60}")
                print(f"训练模式: {mode}")
                print(f"{'='*60}")
                
                # 创建单模式训练器
                trainer = MultiTaskTrainer(self.config)
                trainer.set_task_mode(mode)
                
                # 运行该模式的完整流程
                success = trainer.run_complete_pipeline()
                
                if success:
                    self.all_results[mode] = trainer
                    self.best_models_by_mode[mode] = trainer.best_overall_model
                    successful_modes += 1
                    print(f" {mode} 模式训练成功")
                else:
                    print(f" {mode} 模式训练失败")
                    
            except Exception as e:
                print(f" {mode} 模式训练出错: {e}")
                if self.config.OUTPUT_CONFIG['verbose']:
                    traceback.print_exc()
        
        print(f"\n模式训练完成: {successful_modes}/{len(self.config.TRAIN_MODES)} 个模式成功")
        return successful_modes > 0
    
    def compare_all_modes(self):
        """比较所有模式的性能 - 优化版"""
        if not self.all_results:
            print("无训练结果可比较")
            return
        
        print("\n比较所有训练模式...")
        
        comparison_data = []
        for mode, trainer in self.all_results.items():
            if trainer.best_overall_model:
                best_model = trainer.best_overall_model
                comparison_data.append({
                    'mode': mode,
                    'best_model': best_model['model_type'],
                    'test_score': best_model['test_score'],
                    'task_count': len(trainer.config.TASK_MODES[mode])
                })
        
        # 按测试集准确率排序
        comparison_data.sort(key=lambda x: x['test_score'], reverse=True)
        
        # 打印比较结果
        print(f"\n模式性能比较:")
        print("-" * 80)
        print(f"{'模式':<25} {'最佳模型':<10} {'测试准确率':<12} {'任务数量':<10}")
        print("-" * 80)
        
        for data in comparison_data:
            print(f"{data['mode']:<25} "
                  f"{data['best_model']:<10} "
                  f"{data['test_score']:.4f} "
                  f"{data['task_count']}")
        
        # 绘制比较图
        if self.config.OUTPUT_CONFIG['generate_plots'] and self.config.OUTPUT_CONFIG['plot_mode_comparison']:
            self.plot_mode_comparison(comparison_data, self.output_dir / "mode_comparison.png")
        
        return comparison_data
    
    def plot_mode_comparison(self, comparison_data, save_path=None):
        """绘制模式比较图 """
            
        if not comparison_data:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        modes = [data['mode'] for data in comparison_data]
        scores = [data['test_score'] for data in comparison_data]
        task_counts = [data['task_count'] for data in comparison_data]
        
        x = np.arange(len(modes))
        colors = plt.cm.Set1(np.linspace(0, 1, len(modes)))
        
        # 绘制条形图
        bars = ax.bar(x, scores, color=colors, alpha=0.8)
        
        ax.set_xlabel('训练模式')
        ax.set_ylabel('测试准确率')
        ax.set_title('不同训练模式性能比较')
        ax.set_xticks(x)
        ax.set_xticklabels(modes, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, score, count in zip(bars, scores, task_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                   f'{score:.3f}\n({count}任务)', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"模式比较图已保存到: {save_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        print("\n生成综合报告...")
        
        report_path = self.output_dir / "comprehensive_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("多任务学习综合训练报告\n")
            f.write("="*60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据目录: {self.config.DATA_DIR}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            
            # 训练配置
            f.write("训练配置:\n")
            f.write("-"*40 + "\n")
            f.write(f"训练模式: {', '.join(self.config.TRAIN_MODES)}\n")
            f.write(f"模型类型: {', '.join(self.config.MODEL_TYPES)}\n")
            f.write(f"交叉验证折数: {self.config.TRAIN_CONFIG['n_splits']}\n\n")
            
            # 模式结果
            f.write("各模式结果:\n")
            f.write("-"*40 + "\n")
            f.write(f"成功训练的模式: {len(self.all_results)}/{len(self.config.TRAIN_MODES)}\n\n")
            
            for mode, trainer in self.all_results.items():
                f.write(f"{mode}:\n")
                if trainer.best_overall_model:
                    best_model = trainer.best_overall_model
                    f.write(f"  最佳模型: {best_model['model_type']}\n")
                    f.write(f"  测试准确率: {best_model['test_score']:.4f}\n")
                    f.write(f"  任务数量: {len(trainer.config.TASK_MODES[mode])}\n")
                f.write("\n")
            
            # 推荐配置
            f.write("推荐配置:\n")
            f.write("-"*40 + "\n")
            if self.best_models_by_mode:
                # 找出最佳模式
                best_mode = max(self.best_models_by_mode.items(), 
                               key=lambda x: x[1]['test_score'] if x[1] else 0)
                if best_mode[1]:
                    f.write(f"推荐使用模式: {best_mode[0]}\n")
                    f.write(f"推荐模型: {best_mode[1]['model_type']}\n")
                    f.write(f"预期准确率: {best_mode[1]['test_score']:.4f}\n")
        
        print(f"综合报告已生成: {report_path}")
    
    def run_complete_multi_mode_pipeline(self):
        """运行完整的多模式训练流程"""
        try:
            print("="*60)
            print("开始完整的多模式训练流程")
            print("="*60)
            
            # 1. 设置环境
            if not self.setup():
                return False
            
            # 2. 训练所有模式
            if not self.train_all_modes():
                return False
            
            # 3. 比较所有模式
            comparison_data = self.compare_all_modes()
            
            # 4. 生成综合报告
            self.generate_comprehensive_report()
            
            # 5. 打印总结
            self.print_comprehensive_summary()
            
            return True
            
        except Exception as e:
            print(f"多模式训练流程出错: {e}")
            traceback.print_exc()
            return False
    
    def print_comprehensive_summary(self):
        """打印综合总结"""
        print("\n" + "="*60)
        print("多模式训练流程完成!")
        print("="*60)
        
        print(f"成功训练模式: {len(self.all_results)}/{len(self.config.TRAIN_MODES)}")
        
        if self.best_models_by_mode:
            # 找出最佳模式
            best_mode, best_model_info = max(self.best_models_by_mode.items(), 
                                           key=lambda x: x[1]['test_score'] if x[1] else 0)
            if best_model_info:
                print(f"最佳模式: {best_mode}")
                print(f"最佳模型: {best_model_info['model_type']}")
                print(f"最佳准确率: {best_model_info['test_score']:.4f}")
        
        print(f"结果保存到: {self.output_dir}")
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    
    # 创建配置
    config = Config()
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='多任务学习模型训练')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR,
                       help='数据目录路径')
    parser.add_argument('--model_types', nargs='+', default=config.MODEL_TYPES,
                       help='要训练的模型类型')
    parser.add_argument('--train_modes', nargs='+', default=config.TRAIN_MODES,
                       help='训练模式: multi_task, Trial_Type, Action_choice')
    parser.add_argument('--epochs', type=int, default=config.TRAIN_CONFIG['epochs'],
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=config.TRAIN_CONFIG['batch_size'],
                       help='批量大小')
    parser.add_argument('--multi_mode', action='store_true', default=True,
                       help='是否运行多模式训练')
    parser.add_argument('--no_plots', action='store_true', default=False,
                       help='是否不生成图表')
    parser.add_argument('--plot_training', action='store_true', default=None,
                       help='是否绘制训练曲线')
    parser.add_argument('--plot_confusion', action='store_true', default=None,
                       help='是否绘制混淆矩阵')
    
    args = parser.parse_args()
    
    # 更新配置
    config.DATA_DIR = args.data_dir
    config.MODEL_TYPES = args.model_types
    config.TRAIN_MODES = args.train_modes
    config.TRAIN_CONFIG['epochs'] = args.epochs
    config.TRAIN_CONFIG['batch_size'] = args.batch_size
    
    # 处理绘图选项
    if args.no_plots:
        config.OUTPUT_CONFIG['generate_plots'] = False
    
    if args.plot_training is not None:
        config.OUTPUT_CONFIG['plot_training_curves'] = args.plot_training
    
    if args.plot_confusion is not None:
        config.OUTPUT_CONFIG['plot_confusion_matrix'] = args.plot_confusion
    
    if args.multi_mode:
        # 多模式训练
        print("运行多模式训练...")
        multi_mode_trainer = MultiModeTrainer(config)
        success = multi_mode_trainer.run_complete_multi_mode_pipeline()
        
        if success:
            print("\n多模式训练成功完成!")
            return multi_mode_trainer
        else:
            print("\n多模式训练失败!")
            return None
    else:
        # 单模式训练（默认多任务）
        print("运行单模式训练...")
        trainer = MultiTaskTrainer(config)
        success = trainer.run_complete_pipeline()
        
        if success:
            print("\n单模式训练成功完成!")
            return trainer
        else:
            print("\n单模式训练失败!")
            return None


if __name__ == "__main__":
    # 添加缺失的导入
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    main()                            
                            
                            