概述
===========================
calcium-imaging-main/是 Calcium-Imaging-in-Decision-Making-Mice项目的核心分析与代码仓库。它包含从小鼠决策行为钙成像数据的预处理、特征工程，到多种机器学习模型（MLP, CNN, LSTM, SVM）的定义、训练、评估与结果可视化的完整流水线。所有生成的中间数据、最终数据集、预训练模型及分析图表均存储于此目录。

## 项目核心结构
以下树状图展示了目录的完整组织方式：
<details>
<summary>calcium-imaging-main/</summary>

* 📁 21Sessions_Data/
  * 📁 processed_mat_files/
* 📁 4_16Sess/
  * 📁 processed_mat_files/
* 📁 7_28Sess/
  * 📁 processed_mat_files/
* 📁 split_data/
  * 🔢 X_train.npy
  * 🏷️ y_train.npy
  * ...（其他文件）
</details>
