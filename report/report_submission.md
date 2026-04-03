# 基于基因表达与空间位置信息的空间转录组细胞解卷积原型研究报告

## 摘要

本项目围绕空间转录组 spot 级细胞类型解卷积任务，设计并实现了一个课程原型系统。系统以高分辨率参考表达数据作为先验，当前默认采用单细胞 RNA-seq 作为参考输入，输入空间转录组表达矩阵和 spot 坐标，输出每个 spot 的细胞类型比例。方法上，本文采用“表达编码器 + 空间图卷积 + 比例预测头”的三段式结构，使模型同时利用基因表达与空间邻域关系；同时实现了 NNLS 和 MLP-only 作为对照基线，并预留了轻量级跨域对齐损失模块。实验使用 Human Lymph Node 真实数据集与 Simulated seqFISH+ 模拟数据集进行验证。结果表明，该原型已经能够完成从数据读取、特征对齐、pseudo-spot 构造、模型训练、推理到空间可视化的完整流程，能够输出可用于课程展示和实验报告撰写的定量结果与图像结果。

## 1. 研究背景与问题定义

空间转录组技术能够在保留组织空间结构的同时测量基因表达，但多数平台的检测分辨率仍以 spot 为单位。一个 spot 往往包含多个细胞，因此原始表达是混合信号，无法直接对应单一细胞类型。细胞解卷积的目标，就是根据 spot 表达推断其内部不同细胞类型的组成比例。

传统解卷积方法多依赖表达矩阵本身，而忽略空间相邻 spot 在组织结构上的连续性。事实上，相邻 spot 往往具有相似或渐变的细胞组成，因此仅看表达容易丢失组织微环境带来的约束信息。基于这一考虑，本项目尝试设计一个同时利用“基因表达 + 空间位置”的深度学习原型，以提高解卷积结果的稳定性与可解释性。

本项目的具体任务定义如下：

- 输入：
  - 高分辨率参考表达矩阵及细胞类型标签
  - 当前默认使用单细胞 RNA-seq，也支持后续扩展到亚细胞分辨率空间转录组参考
  - 空间转录组 spot 表达矩阵
  - 每个 spot 的空间坐标
- 输出：
  - 每个 spot 对各细胞类型的比例分布
- 目标：
  - 完成一个可运行的软件原型
  - 输出实验结果、对比表和空间可视化图
  - 为后续扩展跨平台对齐和去批次模块打好基础

## 2. 任务书要求对齐说明

结合追加任务书，本项目的实现路线与要求对齐关系如下：

- 任务书要求研究 spot 分辨率空间转录组解卷积：本项目主线已完成
- 任务书允许从参考数据选择或空间图网络角度切入：本项目选择了“参考表达 + 空间图神经网络”路线
- 任务书要求整合为软件包或 Webserver：本项目采用软件包形式完成，不再额外开发 Webserver
- 任务书要求开发环境为 Python + Linux：代码主体保持跨平台，README 已补充 Linux 运行说明

因此，本项目当前版本在研究内容、方法路线和软件交付形式上均与任务书保持一致。

## 3. 数据集与数据基础

### 3.1 Human Lymph Node

本项目将 Human Lymph Node 作为主实验真实数据集。该数据集目录位于 [Data/4.Human_Lymph_Node](C:/Coding/260402_spatial-cell-deconvolution/Data/4.Human_Lymph_Node)，包含：

- `scRNA.h5ad`：单细胞参考表达数据
- `ST.h5ad`：空间转录组数据
- `manual_GC_annot.csv`：spot 注释辅助文件

实际检查结果表明：

- 单细胞数据维度为 73260 × 10237
- 空间数据维度为 4035 × 36588
- 空间数据带有 `obsm['spatial']` 坐标
- 单细胞数据中可自动识别到 `cell_type` 标签列
- 主数据集共识别到 34 类细胞类型

该数据集的优点是结构完整，能够支撑“真实参考 scRNA + 真实 ST”这一主实验流程。由于 spot 坐标和辅助注释都较为完整，适合展示空间热图和组织结构相关结果。

### 3.2 Simulated seqFISH+

本项目将 Simulated seqFISH+ 作为辅助定量评测数据集，目录位于 [Data/11.Simulated_seqFISH+](C:/Coding/260402_spatial-cell-deconvolution/Data/11.Simulated_seqFISH+)。

该数据集包含：

- `scRNA.h5ad`
- `Spatial.h5ad`

实际检查结果表明：

- 单细胞数据维度为 6948 × 20007
- 空间数据维度为 1000 × 26160
- 单细胞标签列为 `celltype_final`

由于该数据集缺少显式空间坐标，本项目采用规则网格作为回退坐标，用于保证空间图模块依然可以运行。该处理方式并不等价于真实组织坐标，因此在报告中将它定位为“流程验证与辅助定量评测”，而非真实空间结构验证。

## 4. 方法设计

### 4.1 总体流程

整个系统流程如下：

1. 读取单细胞数据与空间数据
2. 自动识别单细胞标签列和空间坐标
3. 对齐共享基因并进行表达归一化
4. 选择共享高变基因作为建模特征
5. 基于单细胞数据构造 pseudo-spot 监督样本
6. 分别训练基线模型和空间模型
7. 在真实 ST 上推理并输出 spot 级比例矩阵
8. 生成指标表、训练曲线和空间热图

### 4.2 数据预处理

数据预处理在 [src/spatial_deconv/data/preprocess.py](C:/Coding/260402_spatial-cell-deconvolution/src/spatial_deconv/data/preprocess.py) 中实现，核心步骤如下：

- 对 scRNA 与 ST 取共享基因交集
- 采用 library size normalization 后接 `log1p`
- 默认保留 2000 个共享高变基因；共享基因不足时降为 1000
- 基于 spot 坐标构建 KNN 图，默认 `k=6`

该策略的目的，是保证不同来源数据在相同特征空间中训练，并降低原始高维表达矩阵带来的噪声。

### 4.3 pseudo-spot 构造

真实 ST 数据通常不直接提供每个 spot 的真实细胞比例，因此本项目使用 pseudo-spot 方式生成监督标签。做法是：

- 按细胞类型从 scRNA 中随机采样若干细胞
- 按 Dirichlet 分布随机生成各类型混合权重
- 将采样到的单细胞表达求均值，构造一个 pseudo-spot
- 同时保留其真实组成比例作为监督标签

该策略有两个优点：

- 不依赖真实 ST 的真值标签
- 能够方便构造大量监督样本，适合课程型原型快速训练

### 4.4 模型结构

本项目实现了 3 类模型：

#### 4.4.1 NNLS 基线

NNLS 基线在 [src/spatial_deconv/models/baselines.py](C:/Coding/260402_spatial-cell-deconvolution/src/spatial_deconv/models/baselines.py) 中实现。其思想是先按细胞类型构建平均参考表达，再对每个 spot 求解非负最小二乘系数，最后归一化为比例。

该方法优点是实现简单、可解释性强，是经典的解卷积基线。

#### 4.4.2 MLP-only 基线

MLP-only 模型由两层全连接网络组成，输入高变基因表达，输出各细胞类型比例。该模型不引入任何空间结构，仅利用表达特征，因此可以直接用于评估“是否引入空间信息”的收益。

#### 4.4.3 Spatial GCN 主模型

Spatial GCN 在 [src/spatial_deconv/models/gcn.py](C:/Coding/260402_spatial-cell-deconvolution/src/spatial_deconv/models/gcn.py) 中实现，结构包括：

- 表达编码器：两层 MLP，将高维基因表达映射到潜在空间
- 图卷积模块：利用空间邻接矩阵聚合邻域信息
- 比例输出头：线性层后接 `softmax`

该结构的直觉是：相邻 spot 的细胞组成通常具有一定连续性，因此在表达编码后使用图卷积融合邻域信息，可以让模型学习局部空间平滑模式。

### 4.5 训练策略与损失函数

训练逻辑在 [src/spatial_deconv/train.py](C:/Coding/260402_spatial-cell-deconvolution/src/spatial_deconv/train.py) 中实现。

本项目采用两阶段策略：

1. 在 pseudo-spot 上进行监督训练
2. 将训练好的模型直接应用于真实 ST 做推理

主损失函数采用 MSE，用于回归预测比例与真实比例之间的差异。另实现了一个轻量级 MMD 项作为可选扩展，用于弱化训练分布与目标分布之间的偏移。但在当前课程原型中，MMD 只作为预留扩展，不作为主实验结论的核心部分。

## 5. 系统实现

### 5.1 代码结构

主要代码组织如下：

- [src/spatial_deconv/data/io.py](C:/Coding/260402_spatial-cell-deconvolution/src/spatial_deconv/data/io.py)：数据读取、标签识别、坐标解析
- [src/spatial_deconv/data/preprocess.py](C:/Coding/260402_spatial-cell-deconvolution/src/spatial_deconv/data/preprocess.py)：归一化、HVG、KNN 图、pseudo-spot
- [src/spatial_deconv/models/baselines.py](C:/Coding/260402_spatial-cell-deconvolution/src/spatial_deconv/models/baselines.py)：NNLS 与 MLP 基线
- [src/spatial_deconv/models/gcn.py](C:/Coding/260402_spatial-cell-deconvolution/src/spatial_deconv/models/gcn.py)：空间图卷积模型
- [src/spatial_deconv/evaluate.py](C:/Coding/260402_spatial-cell-deconvolution/src/spatial_deconv/evaluate.py)：MAE、RMSE、PCC 指标
- [src/spatial_deconv/visualize.py](C:/Coding/260402_spatial-cell-deconvolution/src/spatial_deconv/visualize.py)：训练曲线和空间热图

### 5.2 脚本入口

为了便于复现实验，本项目提供了以下脚本：

- [scripts/inspect_data.py](C:/Coding/260402_spatial-cell-deconvolution/scripts/inspect_data.py)：查看数据摘要
- [scripts/build_pseudospots.py](C:/Coding/260402_spatial-cell-deconvolution/scripts/build_pseudospots.py)：生成训练用 pseudo-spot
- [scripts/train_model.py](C:/Coding/260402_spatial-cell-deconvolution/scripts/train_model.py)：单模型训练
- [scripts/run_inference.py](C:/Coding/260402_spatial-cell-deconvolution/scripts/run_inference.py)：真实 ST 推理与图像导出
- [scripts/run_experiments.py](C:/Coding/260402_spatial-cell-deconvolution/scripts/run_experiments.py)：批量实验与结果汇总

### 5.3 环境选择

本机存在多个 Python 版本。为了兼容 `torch`、`anndata` 和生信数据处理依赖，本项目最终选用了 `.venv` 中的 **Python 3.11** 作为正式环境，这一版本在科学计算生态中兼容性较好，便于后续继续扩展 `scanpy` 等依赖。

### 5.4 平台与部署说明

任务书中规定开发语言为 Python、目标操作系统为 Linux。为此，本项目在编码时尽量采用跨平台路径和纯 Python 依赖管理方式，训练入口通过脚本统一提供，并在 README 中额外补充了 Linux/bash 的运行说明。

本项目当前以软件包形式交付，主要原因如下：

- 软件包更符合课程项目“算法整合”的要求
- 比 Webserver 更容易在本地与集群环境中复现
- 更适合后续扩展更多数据集与模型

## 6. 实验设计

### 6.1 对比设置

实验共设置 3 个模型：

- NNLS
- MLP-only
- Spatial GCN

其中：

- NNLS 用于提供传统线性解卷积基线
- MLP-only 用于验证纯表达深度模型的表现
- Spatial GCN 用于验证加入空间信息后的增益

### 6.2 指标

本项目使用以下指标评价 pseudo-spot 验证集上的解卷积效果：

- MAE：平均绝对误差
- RMSE：均方根误差
- PCC：按细胞类型计算 Pearson 相关系数后求平均

其中 MAE 和 RMSE 越小越好，PCC 越大越好。

### 6.3 实验脚本

正式实验可通过以下命令运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py --datasets human_lymph_node simulated_seqfish --models nnls mlp spatial_gcn --epochs 8 --num-spots 1200 --max-sc-cells 6000
```

实验结果会自动输出到：

- `outputs/metrics/experiment_summary.csv`
- `outputs/metrics/experiment_summary.json`
- `outputs/metrics/experiment_summary.md`

## 7. 实验结果与分析

### 7.1 当前已跑通的结果

当前仓库中已经生成了第一轮可用结果，包括：

- [outputs/metrics/human_lymph_node_spatial_gcn_validation.csv](C:/Coding/260402_spatial-cell-deconvolution/outputs/metrics/human_lymph_node_spatial_gcn_validation.csv)
- [outputs/metrics/simulated_seqfish_mlp_validation.csv](C:/Coding/260402_spatial-cell-deconvolution/outputs/metrics/simulated_seqfish_mlp_validation.csv)
- [outputs/predictions/human_lymph_node_spatial_gcn_predictions.csv](C:/Coding/260402_spatial-cell-deconvolution/outputs/predictions/human_lymph_node_spatial_gcn_predictions.csv)
- [outputs/figures/human_lymph_node_spatial_gcn](C:/Coding/260402_spatial-cell-deconvolution/outputs/figures/human_lymph_node_spatial_gcn)

这些结果证明系统已经能够稳定完成训练、推理和可视化闭环。

### 7.2 结果解释思路

在正式撰写最终提交版时，建议按照以下逻辑解释实验结果：

1. 先比较 NNLS 与 MLP-only，说明深度模型对非线性混合关系有更强拟合能力
2. 再比较 MLP-only 与 Spatial GCN，说明空间邻域约束有助于提高局部组织结构一致性
3. 对 Human Lymph Node 的空间图进行分析，说明某些免疫细胞亚群在局部区域呈现富集
4. 对 Simulated seqFISH+ 的结果强调其主要用于流程验证和辅助定量，而不应过度解读其空间结构意义

### 7.3 预期现象

从方法设计上看，预期会出现以下现象：

- Spatial GCN 在模拟集上相较 MLP-only 获得更低的误差或更高的相关性
- Human Lymph Node 的空间图会呈现一定区域性富集模式
- NNLS 的表现通常稳定，但对复杂非线性混合的拟合能力弱于深度模型

## 8. 讨论

### 8.1 本原型的优点

- 具备完整可运行流程，符合课程项目交付要求
- 同时利用表达信息与空间位置信息
- 可扩展性较强，后续可接更复杂 GNN 或域适配模块
- 已经能够在真实数据上生成比例矩阵和空间热图

### 8.2 当前局限

- pseudo-spot 监督不能完全等价于真实 spot 组成
- Simulated seqFISH+ 缺少真实坐标，采用了规则网格回退
- 去批次和跨平台能力目前仅为轻量级扩展接口，未做完整论文级实现
- 当前实验仍以小规模快速验证为主，若要进一步提高结果可信度，需要增加训练轮数和重复实验次数

### 8.3 后续改进方向

- 引入更强的图神经网络，如 GAT 或更深层的残差 GCN
- 对接真实批次标签，加入显式 domain adaptation 模块
- 使用更多配对公开数据集验证跨平台能力
- 增加消融实验，例如不同 `k` 值、不同 HVG 数量、是否使用 MMD

### 8.4 可持续性、规范与工程实践思考

任务书中还强调了行业背景、规范意识以及可持续发展相关要求。结合本项目场景，可以从以下角度进行理解：

- 在数据使用层面，应优先采用公开许可明确的数据集，避免不合规传播受限生物医学数据
- 在计算资源使用层面，课程原型优先采用小规模试跑、再进行正式训练，减少无效重复计算带来的能源浪费
- 在工程实现层面，通过统一脚本入口和可复现实验流程，可以降低重复调试成本，提高科研开发效率
- 在应用层面，空间转录组分析结果可能影响后续生物学解释，因此需要对模型局限性保持清醒认识，避免过度解读自动化结果

本项目虽然是课程原型，但在数据来源、训练流程和结果解释上都尽量遵循可复现、可说明、可扩展的工程实践思路。

## 9. 结论

本项目完成了一个面向空间转录组细胞解卷积任务的深度学习课程原型，实现了从数据读取、预处理、监督样本构造、模型训练到真实数据推理与空间可视化的完整流程。结果表明，该原型能够输出 spot 级细胞比例和空间热图，满足“代码 + 实验报告”的基本交付要求。尽管当前版本在跨平台对齐和去批次方面仍然采取了轻量方案，但系统框架已经为后续扩展打下了基础，适合作为课程项目、毕设前期原型或后续论文复现的起点。

## 附录：运行命令

### A. 数据检查

```powershell
.\.venv\Scripts\python.exe scripts\inspect_data.py --dataset human_lymph_node
```

### B. 构造 pseudo-spot

```powershell
.\.venv\Scripts\python.exe scripts\build_pseudospots.py --dataset human_lymph_node --num-spots 4000
```

### C. 训练单模型

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py --dataset human_lymph_node --model spatial_gcn --epochs 30
```

### D. 运行正式实验

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py --datasets human_lymph_node simulated_seqfish --models nnls mlp spatial_gcn --epochs 8 --num-spots 1200 --max-sc-cells 6000
```

### E. 推理与导图

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py --dataset human_lymph_node --checkpoint outputs\models\human_lymph_node_spatial_gcn.pt
```
