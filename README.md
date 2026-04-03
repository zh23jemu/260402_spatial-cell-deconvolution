# 空间转录组细胞解卷积原型

本仓库实现一个课程原型：利用参考数据、空间表达和空间坐标，完成 spot 级细胞比例预测，并提供基线模型、空间图模型、评估与可视化脚本。

当前默认参考数据为单细胞 RNA-seq (`scRNA.h5ad`)；从接口设计上也允许后续扩展到亚细胞分辨率空间转录组或其他高分辨率参考表达数据，只要输入整理为 `h5ad` 并包含可用于监督或伪监督的细胞类型标签。

## 快速开始

### Windows / PowerShell

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

### Linux / Bash

任务书要求的目标操作系统为 Linux，因此集群或服务器上建议按下面方式运行：

1. 创建虚拟环境：

```bash
python3.11 -m venv .venv
```

2. 安装依赖：

```bash
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e .
```

3. 查看数据摘要：

```bash
./.venv/bin/python scripts/inspect_data.py --dataset human_lymph_node
```

4. 构造 pseudo-spot：

```bash
./.venv/bin/python scripts/build_pseudospots.py --dataset human_lymph_node --num-spots 4000
```

5. 训练空间模型：

```bash
./.venv/bin/python scripts/train_model.py --dataset human_lymph_node --model spatial_gcn --epochs 20 --num-spots 2000 --max-sc-cells 8000 --device cpu
```

6. 推理并导出图表：

```bash
./.venv/bin/python scripts/run_inference.py --dataset human_lymph_node --checkpoint outputs/models/human_lymph_node_spatial_gcn.pt --device cpu
```

## 输入约定

当前项目面向以下输入形式：

- 参考表达数据：
  - 默认支持单细胞 RNA-seq `h5ad`
  - 也可扩展为亚细胞分辨率 ST 或其他高分辨率参考数据，只要包含细胞类型标签列
- 空间转录组数据：
  - `h5ad`
  - 或 10x 风格 `filtered_feature_bc_matrix.h5` 配合 `spatial/tissue_positions_list.csv`

输出为每个 spot 的细胞类型比例矩阵，以及训练指标、预测表和空间热图。

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
