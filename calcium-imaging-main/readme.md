概述
===========================
calcium-imaging-main/是 Calcium-Imaging-in-Decision-Making-Mice项目的核心分析与代码仓库。它包含从小鼠决策行为钙成像数据的预处理、特征工程，到多种机器学习模型（MLP, CNN, LSTM, SVM）的定义、训练、评估与结果可视化的完整流水线。所有生成的中间数据、最终数据集、预训练模型及分析图表均存储于此目录。

## 核心代码模块详解
本目录包含四个核心Python脚本，构成了从原始数据到模型结果的端到端分析流水线。
<div align="center">

|  脚本文件  |  核心功能  |    输入    |     输出   | 关键类/函数|
| ---------- | ---------- | ---------- | ---------- | ---------- |
| mat_dataset_processor.py | 原始数据提取与打标签   | 各数据集原始.mat文件   | 各数据集processed_mat_files/下的带标签.mat文件   | MATDatasetProcessor   |
| data_preprocessing.py | 数据预处理与数据集初步划分   | 上述带标签.mat文件   | split_data/目录下的.npy文件   | ROIProcessedThreeTaskProcessor   |
| models.py | 多任务学习模型定义   | -   | -   | MultiTaskMLP, MultiTaskCNN, MultiTaskLSTM, create_model()   |
| train_multitask_model.py | 主训练与评估脚本   | split_data/中的.npy文件   | trained_models/中的模型与图表   | Config, DatasetManager, KFoldTrainer   |

</div>
