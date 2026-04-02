#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多任务学习模型定义 - 修复版
修复LSTM输入维度问题
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import hiddenlayer as hl  # 需要安装: pip install hiddenlayer
# ==================== 深度学习模型定义 ====================

class MultiTaskMLP(nn.Module):
    """优化版多任务MLP模型 - 经典结构提高准确率"""
    
    def __init__(self, input_size, hidden_sizes, task_sizes, dropout_rate=0.3, 
                 use_batch_norm=True, activation='leaky_relu'):
        """
        初始化优化版多任务MLP
        
        Args:
            input_size: 输入特征维度
            hidden_sizes: 隐藏层大小列表，如 [512, 256, 128]
            task_sizes: 每个任务的输出类别数列表
            dropout_rate: dropout比率
            use_batch_norm: 是否使用批归一化
            activation: 激活函数类型 ('relu', 'leaky_relu', 'elu')
        """
        super(MultiTaskMLP, self).__init__()
        
        self.task_sizes = task_sizes
        self.num_tasks = len(task_sizes)
        self.use_batch_norm = use_batch_norm
        self.activation_type = activation
        
        # 验证隐藏层大小
        if not hidden_sizes:
            hidden_sizes = [512, 256, 128]  # 默认结构
        
        # 构建共享层 - 使用经典的前馈网络结构
        shared_layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # 线性层
            shared_layers.append(nn.Linear(prev_size, hidden_size))
            
            # 批归一化
            if use_batch_norm:
                shared_layers.append(nn.BatchNorm1d(hidden_size))
            
            # 激活函数
            if activation == 'leaky_relu':
                shared_layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif activation == 'elu':
                shared_layers.append(nn.ELU(alpha=1.0))
            else:  # 默认使用ReLU
                shared_layers.append(nn.ReLU())
            
            # Dropout (除了最后一层)
            if i < len(hidden_sizes) - 1:  # 不在最后一层后加dropout
                shared_layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # 任务特定头 - 使用经典的两层结构
        self.task_heads = nn.ModuleList()
        for task_size in task_sizes:
            # 经典的两层任务头结构
            task_head = nn.Sequential(
                # 第一层
                nn.Linear(prev_size, prev_size // 2),
                nn.BatchNorm1d(prev_size // 2) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                # 第二层
                nn.Linear(prev_size // 2, max(32, prev_size // 4)),  # 确保最小维度
                nn.BatchNorm1d(max(32, prev_size // 4)) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),  # 最后一层使用更小的dropout
                
                # 输出层
                nn.Linear(max(32, prev_size // 4), task_size)
            )
            self.task_heads.append(task_head)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用Xavier初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def plot_model_architecture(self, input_size: int, save_path: str = None):
        """
        绘制模型架构图 - 用于论文展示
        """
        try:
            # 创建示例输入
            x = torch.randn(1, input_size)
            
            # 使用hiddenlayer可视化模型
            with torch.no_grad():
                # 构建模型图
                graph = hl.build_graph(self, x)
            
            if save_path:
                graph.save(save_path)
                print(f"模型架构图已保存到: {save_path}")
            
            return graph
        except Exception as e:
            print(f"绘制模型架构图失败: {e}")
            # 备用方案：打印模型结构
            print("模型结构:")
            print(self)
            return None
                
    
    def forward(self, x):
        # 处理输入维度
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # 共享特征提取
        shared_features = self.shared_layers(x)
        
        # 多任务输出
        outputs = []
        for task_head in self.task_heads:
            output = task_head(shared_features)
            outputs.append(output)
        
        return outputs


class MultiTaskCNN(nn.Module):
    """修复版多任务CNN模型 - 简化结构，修复Bug，增强正则化"""
    
    def __init__(self, input_channels, input_height, input_width, task_sizes, 
                 num_filters=32, dropout_rate=0.5, use_batch_norm=True, activation='leaky_relu'):
        """
        初始化修复版多任务CNN
        
        Args:
            input_channels: 输入通道数 (应为1)
            input_height: 输入高度 (ROI数量，例如69)
            input_width: 输入宽度 (时间点，例如55)
            task_sizes: 每个任务的输出类别数列表
            num_filters: 初始滤波器数量
            dropout_rate: dropout比率 (提高至0.5以增强正则化)
            use_batch_norm: 是否使用批归一化
            activation: 激活函数类型
        """
        super(MultiTaskCNN, self).__init__()
        
        self.task_sizes = task_sizes
        self.num_tasks = len(task_sizes)
        self.use_batch_norm = use_batch_norm
        self.activation_type = activation
        
        # --- 卷积特征提取部分 (3个块) ---
        # 简化：移除所有注意力模块，使用经典卷积块
        self.conv_block1 = self._make_conv_block(
            in_channels=input_channels,
            out_channels=num_filters,
            kernel_size=3,
            pool_kernel=2
        )
        
        self.conv_block2 = self._make_conv_block(
            in_channels=num_filters,
            out_channels=num_filters * 2,
            kernel_size=3,
            pool_kernel=2
        )
        
        # 关键修改1: 使用对称池化 (2) 替代非对称的 (1, 2)
        self.conv_block3 = self._make_conv_block(
            in_channels=num_filters * 2,
            out_channels=num_filters * 4,
            kernel_size=3,
            pool_kernel=2
        )
        
        # --- 计算卷积后特征图尺寸 & 自适应池化 ---
        # 3次2倍池化后，高和宽各缩小8倍
        conv_output_height = input_height // 8
        conv_output_width = input_width // 8
        
        # 关键修改2: 如果尺寸过小，使用全局平均池化代替展平
        if conv_output_height < 2 or conv_output_width < 2:
            # 全局平均池化输出形状: (batch, num_filters*4, 1, 1)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            fc_input_size = num_filters * 4  # 展平后就是通道数
        else:
            self.global_pool = None
            fc_input_size = num_filters * 4 * conv_output_height * conv_output_width
        
        # --- 共享全连接层 (特征抽象) ---
        hidden_sizes = [256, 128, 64]
        
        shared_fc_layers = []
        prev_size = fc_input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # 线性层
            shared_fc_layers.append(nn.Linear(prev_size, hidden_size))
            
            # 批归一化
            if use_batch_norm:
                shared_fc_layers.append(nn.BatchNorm1d(hidden_size))
            
            # 激活函数
            if activation == 'leaky_relu':
                shared_fc_layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif activation == 'elu':
                shared_fc_layers.append(nn.ELU(alpha=1.0))
            else:
                shared_fc_layers.append(nn.ReLU())
            
            # 关键修改3: 修复Bug - 确保所有层后都添加Dropout，包括最后一层
            # 最后一层使用较小的Dropout率
            current_dropout = dropout_rate if i < len(hidden_sizes) - 1 else dropout_rate * 0.5
            shared_fc_layers.append(nn.Dropout(current_dropout))
            
            prev_size = hidden_size
        
        self.shared_fc = nn.Sequential(*shared_fc_layers)
        
        # --- 任务特定输出头 ---
        # 简化：使用更小的单层分类器
        self.task_heads = nn.ModuleList()
        for task_size in task_sizes:
            task_head = nn.Sequential(
                nn.Linear(prev_size, 64),  # 固定大小的瓶颈层
                nn.BatchNorm1d(64) if use_batch_norm else nn.Identity(),
                self._get_activation_function(),
                nn.Dropout(dropout_rate * 0.3),  # 输出层使用更小的dropout
                nn.Linear(64, task_size)
            )
            self.task_heads.append(task_head)
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels, out_channels, kernel_size=3, pool_kernel=2):
        """创建标准的卷积块: Conv -> BN -> Activation -> Pool"""
        layers = []
        
        padding = kernel_size // 2
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, 
                               padding=padding, bias=not self.use_batch_norm))
        
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(self._get_activation_function())
        
        if pool_kernel > 1:
            layers.append(nn.MaxPool2d(kernel_size=pool_kernel))
        
        return nn.Sequential(*layers)
    
    def _get_activation_function(self):
        """获取激活函数实例"""
        if self.activation_type == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif self.activation_type == 'elu':
            return nn.ELU(alpha=1.0)
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        """权重初始化 (保持与之前一致)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        期望输入形状: (batch_size, 1, input_height, input_width) 或 (batch_size, input_height, input_width)
        """
        # 输入维度处理
        if x.dim() == 3:
            # 假设输入为 (batch, height, width)，添加通道维度
            x = x.reshape(x.size(0), 1, x.size(1), x.size(2))
        elif x.dim() != 4:
            raise ValueError(f"MultiTaskCNN 输入应为3D或4D张量，但得到了 {x.dim()}D")
        
        # 卷积特征提取
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # 应用全局平均池化（如果定义了）
        if self.global_pool is not None:
            x = self.global_pool(x)
        
        # 展平特征
        x = x.view(x.size(0), -1)
        
        # 共享全连接层处理
        shared_features = self.shared_fc(x)
        
        # 多任务输出
        outputs = []
        for task_head in self.task_heads:
            output = task_head(shared_features)
            outputs.append(output)
        
        return outputs



class MultiTaskLSTM(nn.Module):
    """多任务LSTM模型 - 使用层归一化(LN)优化版"""
    
    def __init__(self, input_size, hidden_size, num_layers, task_sizes, 
                 bidirectional=True, dropout_rate=0.3, sequence_length=69,
                 use_layer_norm=True, use_residual=False):
        """
        初始化多任务LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            task_sizes: 每个任务的输出类别数列表
            bidirectional: 是否使用双向LSTM
            dropout_rate: dropout比率
            sequence_length: 序列长度
            use_layer_norm: 是否使用层归一化
            use_residual: 是否使用残差连接
        """
        super(MultiTaskLSTM, self).__init__()
        
        self.task_sizes = task_sizes
        self.num_tasks = len(task_sizes)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.bidirectional = bidirectional

        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 计算LSTM输出维度
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # 层归一化 (替代批归一化)
        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(lstm_output_size)
            self.layer_norm2 = nn.LayerNorm(lstm_output_size // 8)
        
        # 共享特征处理层
        self.shared_fc = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_size // 4, lstm_output_size // 8),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
        )
        
        # 如果使用残差连接，添加投影层
        if use_residual and lstm_output_size != lstm_output_size // 2:
            self.residual_projection = nn.Linear(lstm_output_size, lstm_output_size // 2)
        else:
            self.residual_projection = None
        
        # 任务特定头
        self.task_heads = nn.ModuleList()
        for i, task_size in enumerate(task_sizes):
            # 为每个任务创建独立的头
            task_head = nn.Sequential(
                nn.Linear(lstm_output_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, task_size),
            )
            self.task_heads.append(task_head)
            
            # # 为每个任务头添加层归一化
            # if use_layer_norm:
            #     setattr(self, f'task_head_ln_{i}', nn.LayerNorm(64))
                
    def plot_model_architecture(self, sequence_length: int, input_size: int, 
                              save_path: str = None):
        """
        绘制LSTM模型架构图
        """
        try:
            # 创建示例输入
            x = torch.randn(1, sequence_length, input_size)
            
            # 使用hiddenlayer可视化模型
            with torch.no_grad():
                graph = hl.build_graph(self, x)
            
            if save_path:
                graph.save(save_path)
                print(f"LSTM模型架构图已保存到: {save_path}")
            
            return graph
        except Exception as e:
            print(f"绘制LSTM模型架构图失败: {e}")
            print("LSTM模型结构:")
            print(self)
            return None                
    
    def forward(self, x):
        """
        前向传播
        期望输入形状: (batch_size, sequence_length, input_size)
        """
        batch_size = x.size(0)
        original_shape = x.shape
        
        # 维度检查和调整
        if x.dim() == 2:
            # 2D输入: 尝试重塑为3D
            if self.sequence_length is not None:
                seq_len = self.sequence_length
                feature_dim = x.size(1) // seq_len
                if x.size(1) == seq_len * feature_dim:
                    x = x.view(batch_size, seq_len, feature_dim)
                else:
                    x = x.view(batch_size, -1, self.input_size)
            else:
                x = x.view(batch_size, 1, x.size(1))
        
        # 确保是3D张量
        if x.dim() != 3:
            raise ValueError(f"LSTM输入应该是3D张量，但是得到了 {x.dim()}D")
        
        # 检查特征维度
        if x.size(2) != self.input_size:
            # 尝试转置维度
            if x.size(1) == self.input_size:
                x = x.transpose(1, 2)
            else:
                # 如果维度不匹配，尝试投影
                if x.size(2) > self.input_size:
                    # 截断多余特征
                    x = x[:, :, :self.input_size]
                else:
                    # 填充不足特征
                    padding = torch.zeros(batch_size, x.size(1), 
                                         self.input_size - x.size(2)).to(x.device)
                    x = torch.cat([x, padding], dim=2)
        
        # LSTM前向传播
        lstm_out, (hn, cn) = self.lstm(x)
        # print(lstm_out.shape)
        # 取最后一个时间步
        if lstm_out.size(1) == 0:
            raise ValueError("LSTM输出序列长度为0")
        
        # last_output = lstm_out[:, -1, :]
        last_output = lstm_out.sum(dim=1)  # 对所有时间步取平均
        
        # print(last_output.shape)
        
        # # 应用层归一化
        # if self.use_layer_norm:
        #     last_output = self.layer_norm1(last_output)
        
        # # 共享特征提取
        # shared_features = self.shared_fc(last_output)
        
        # # 残差连接
        # if self.use_residual:
        #     if self.residual_projection is not None:
        #         residual = self.residual_projection(last_output)
        #     else:
        #         residual = last_output[:, :shared_features.size(1)]  # 切片匹配维度
            
        #     shared_features = shared_features + residual
        
        # # 应用第二层层归一化
        # if self.use_layer_norm:
        #     shared_features = self.layer_norm2(shared_features)
        
        shared_features = last_output
        # 多任务输出
        outputs = []
        for i, task_head in enumerate(self.task_heads):
            # 任务特定特征提取
            output = task_head(shared_features)  # 线性层 + ReLU + Dropout
            
            # # 应用任务特定的层归一化
            # if self.use_layer_norm:
            #     ln_layer = getattr(self, f'task_head_ln_{i}')
            #     task_features = ln_layer(task_features)
            
            # 最终输出层
            # output = task_head[2](task_features)  # 最后的线性层
            outputs.append(output)
        
        return outputs
    
    def init_hidden(self, batch_size, device='cpu'):
        """初始化LSTM隐藏状态"""
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.lstm.num_layers * num_directions, 
                         batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers * num_directions, 
                         batch_size, self.hidden_size).to(device)
        return (h0, c0)


# ==================== 模型工厂函数 ====================

def create_model(model_type, input_size, task_sizes, device='cpu', plot_architecture=False, **model_args):
    """
    创建模型实例的工厂函数 - 修复版
    
    Args:
        model_type: 模型类型 ('MLP', 'CNN', 'LSTM', 'Transformer')
        input_size: 输入特征维度（对于LSTM，应该是特征维度，不是序列长度）
        task_sizes: 每个任务的输出类别数列表
        device: 设备 ('cpu' 或 'cuda')
        **model_args: 模型特定参数
        
    Returns:
        创建的模型实例
    """
    
    model = None
    
    if model_type == "MLP":
        hidden_sizes = model_args.get('hidden_sizes', [256, 128, 64])
        dropout_rate = model_args.get('dropout_rate', 0.3)
        model = MultiTaskMLP(input_size, hidden_sizes, task_sizes, dropout_rate).to(device)
        
        if plot_architecture:
            model.plot_model_architecture(input_size, f"{model_type}_architecture.png")
    
    elif model_type == "CNN":
        input_channels = model_args.get('input_channels', 1)
        input_height = model_args.get('input_height', 69)
        input_width = model_args.get('input_width', 55)
        num_filters = model_args.get('num_filters', 32)
        dropout_rate = model_args.get('dropout_rate', 0.3)
        model = MultiTaskCNN(input_channels, input_height, input_width, task_sizes, 
                            num_filters, dropout_rate).to(device)
        
        if plot_architecture:
            model.plot_model_architecture(input_channels, input_height, 
                                        input_width, f"{model_type}_architecture.png")
    
    elif model_type == "LSTM":
        hidden_size = model_args.get('hidden_size', 128)
        num_layers = model_args.get('num_layers', 2)
        bidirectional = model_args.get('bidirectional', True)
        dropout_rate = model_args.get('dropout_rate', 0.3)
        sequence_length = model_args.get('sequence_length', 69)
        model = MultiTaskLSTM(input_size, hidden_size, num_layers, task_sizes,
                            bidirectional, dropout_rate, sequence_length).to(device)
        
        if plot_architecture:
            model.plot_model_architecture(sequence_length, input_size, 
                                        f"{model_type}_architecture.png")
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model