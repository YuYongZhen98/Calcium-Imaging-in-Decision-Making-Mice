概述
===========================
calcium-imaging-main/是 Calcium-Imaging-in-Decision-Making-Mice项目的核心分析与代码仓库。它包含从小鼠决策行为钙成像数据的预处理、特征工程，到多种机器学习模型（MLP, CNN, LSTM, SVM）的定义、训练、评估与结果可视化的完整流水线。所有生成的中间数据、最终数据集、预训练模型及分析图表均存储于此目录。

## 核心代码模块详解
本目录包含四个核心Python脚本，构成了从原始数据到模型结果的端到端分析流水线。
<div align="center">

|     脚本文件      |      核心功能      |        输入      |       输出       |     关键类/函数    |
| :---------------: | :---------------: | :---------------: | :---------------: | :---------------: |
| mat_dataset_processor.py |   原始数据提取与打标签     | 各数据集原始.mat文件   | 各数据集processed_mat_files/下的带标签.mat文件   | MATDatasetProcessor   |
| data_preprocessing.py |   数据预处理与数据集初步划分     | 上述带标签.mat文件   | split_data/目录下的.npy文件   | ROIProcessedThreeTaskProcessor   |
| models.py |   多任务学习模型定义     | -   | -   | MultiTaskMLP, MultiTaskCNN, MultiTaskLSTM, create_model()   |
| train_multitask_model.py |   主训练与评估脚本     | split_data/中的.npy文件   | trained_models/中的模型与图表   | Config, DatasetManager, KFoldTrainer   |

</div>

### mat_dataset_processor.py- 数据提取
**功能：** 读取HDF5格式的原始数据，提取钙成像时间序列 (data_aligned) 和行为标签 (Trial_Type, Action_choice等)，并进行初步整理。

**输出：** 在每个数据集的 processed_mat_files/子目录下生成 processed_SessX_data_save...文件。
> [!NOTE]
> 此步骤通常只需运行一次。

### data_preprocessing.py- 数据预处理
**功能：** 加载上一步的标准化文件，执行：
  * **帧率统一：** 将所有数据插值至55 Hz；
  * **ROI对齐：** 使用“动态截取”方法，将所有样本的ROI数量统一至最小值，保留高方差特征；
  * **样本过滤：** 自动移除标签（Action_choice或 Trial_Type）为2的无效样本；
  * **数据集划分：** 随机划分为训练集、验证集和测试集（支持分层抽样）。

**输出：** 生成 split_data/目录下的6个 .npy文件，这些文件是所有后续模型训练的直接数据源。
> [!IMPORTANT]
> 这是准备机器学习就绪数据的关键步骤。

### models.py- 模型架构
**包含模型：** 
  * **MultiTaskMLP：** 适用于展平后特征的多层感知机；
  * **MultiTaskCNN：** 用于处理具有时空结构（ROI×时间）特征图的卷积神经网络；
  * **MultiTaskLSTM：** 用于建模时间序列依赖的长短期记忆网络。

**设计：** 所有模型均为**多任务**，共享底层特征提取器，并为不同任务（Action_choice, Trial_Type）提供独立的输出头。
> [!TIP]
> create_model()可根据字符串参数（如 "CNN"）快速实例化对应模型。

### train_multitask_model.py- 主训练入口
**配置：** 通过修改**Config类**中的参数来定义实验：
```python
class Config:
    MODEL_TYPES = ["MLP", "CNN", "LSTM", "SVM"]  # 要训练的模型
    TRAIN_MODES = ['multi_task', 'Trial_Type', 'Action_choice'] # 训练模式
    TRAIN_CONFIG = {
        'use_cross_validation': True,
        'n_splits': 10,  # 十折交叉验证
        'epochs': 200,
        ...
    }
```
**工作流程：** 自动加载 split_data/中的数据，根据配置使用 KFoldTrainer对每个模型执行十折交叉验证训练，保存每一折的最佳模型，并评估其在测试集上的性能。

**输出：** 所有结果（模型权重、训练曲线、混淆矩阵、性能对比图）均保存于 trained_models/{任务模式}/{模型名称}/目录下。
> [!IMPORTANT]
> 这是项目最主要的用户交互脚本，通过修改其配置，可以控制整个训练实验。





