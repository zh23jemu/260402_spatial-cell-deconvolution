# 空间转录组细胞解卷积原型

本仓库实现一个课程原型：利用单细胞参考表达、空间表达和空间坐标，完成 spot 级细胞比例预测，并提供基线模型、空间图模型、评估与可视化脚本。

## 快速开始

1. 安装项目依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

2. 查看数据摘要：

```powershell
.\.venv\Scripts\python.exe scripts\inspect_data.py --dataset human_lymph_node
```

3. 构造 pseudo-spot：

```powershell
.\.venv\Scripts\python.exe scripts\build_pseudospots.py --dataset human_lymph_node --num-spots 4000
```

4. 训练空间模型：

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py --dataset human_lymph_node --model spatial_gcn
```

5. 推理并导出图表：

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py --dataset human_lymph_node --checkpoint outputs\models\human_lymph_node_spatial_gcn.pt
```

## 数据集映射

- `human_lymph_node` -> `Data/4.Human_Lymph_Node`
- `simulated_seqfish` -> `Data/11.Simulated_seqFISH+`
- `human_breast_cancer` -> `Data/3.Human_Breast_Cancer`

## 输出目录

- `outputs\data`：缓存的伪 spot 数据
- `outputs\models`：训练好的模型
- `outputs\metrics`：指标表
- `outputs\predictions`：预测比例表
- `outputs\figures`：空间图
- `report\report_template.md`：实验报告模板
