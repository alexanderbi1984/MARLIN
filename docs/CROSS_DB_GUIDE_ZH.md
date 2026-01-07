# Cross-Database 疼痛识别框架指南 (Syracuse)

本文档总结了基于 Syracuse 等数据集的跨库（Cross-Database）疼痛识别训练与测试流程。该框架支持“单源多目标”（One Source, Multi-Target）的实验设置，并结合了多示例学习（MIL）与 Transformer 聚合器。

## 1. 核心脚本与配置

- **训练脚本**: `/data/Nbi/Marlin/MARLIN/train_cross_db.py`
- **配置文件**: `/data/Nbi/Marlin/MARLIN/config/cross_db.yaml`
- **核心模型**: `MILCoralTransformer` (结合了 MIL、Coral Loss 和 Transformer)

## 2. 实验流程 (Pipeline)

该框架的主要逻辑如下：

1.  **特征提取**: 预先从视频中提取帧级或窗口级特征（如 Marlin, OpenFace AUs）。
2.  **源域训练 (Source Training)**: 在指定的一个或多个“源数据集”（如 Syracuse）上训练模型。
    -   可选：利用 BioVid 作为辅助数据集（Auxiliary Dataset）进行联合训练，并通过 Coral Loss 对齐特征分布。
3.  **目标域测试 (Target Testing)**: 模型训练完成后，自动在指定的“目标数据集”（如 Shoulder Pain, Hospital Pain）上进行推理和评估。

## 3. 配置文件详解 (`config/cross_db.yaml`)

在使用前，请根据特征类型（Marlin vs OpenFace）修改配置文件。

### 3.1 模型参数
```yaml
input_dim: 768       # 关键！如果使用 Marlin 特征，请设为 768；如果使用 OpenFace AUs，通常为 288 或 35
embed_dim: 256       # 投影后的维度
attn_type: xformer   # 聚合器类型 (xformer / simple / gated)
```

### 3.2 数据集定义 (Datasets Pool)
在 `datasets` 字段下定义所有可用数据集的路径和元数据：

- **Syracuse (Source)**
    - `scale`: "0-10" (VAS)
    - `label_col`: pain_level
- **Shoulder Pain (Target)**
    - `scale`: "0-10" (VAS)
    - `feature_root`: 指向该数据集的特征目录
- **Hospital Pain (Target)**
    - `scale`: "0-10" (VAS)
- **BioVid (Auxiliary)**
    - `scale`: "0-4" (用于辅助训练)

### 3.3 实验设置
指定谁是源，谁是目标：

```yaml
# 训练集 (Source)
train_sources:
  - syracuse

# 测试集 (Targets)
test_targets:
  - shoulder_pain
  - hospital_pain
```

### 3.4 标签映射 (Cutoffs)
框架使用 `pain_class_cutoffs` 将连续的 VAS 评分 (0-10) 映射为 5 个离散类别：
- 默认切分点: `[1.1, 3.1, 5.1, 7.1, 9.1]`
- 0-5 分制的数据集会自动将切分点除以 2。

## 4. 如何运行

### 步骤 1: 准备特征
确保你已经使用 `preprocess/marlin_feature_extract.py` 提取了所有涉及数据集的特征（`.npy` 格式）。

### 步骤 2: 修改配置
编辑 `config/cross_db.yaml`，确保 `feature_root` 指向正确的特征目录，且 `input_dim` 与特征维度一致。

### 步骤 3: 启动训练
```bash
python /data/Nbi/Marlin/MARLIN/train_cross_db.py --config /data/Nbi/Marlin/MARLIN/config/cross_db.yaml
```

## 5. 输出结果

训练日志和结果将保存在 `save_dir` 指定的目录下（默认 `CrossDB/runs` 或 `CrossDB/shoulder_pain/...`）：

- **Checkpoints**: 保存验证集效果最好的模型权重 (`.ckpt`)。
- **Metrics CSV**: `metrics_shoulder_pain.csv` (包含准确率、F1、QWK 等指标)。
- **Predictions CSV**: `preds_shoulder_pain.csv` (包含具体的预测值与真实标签，便于后续分析)。

## 6. 注意事项

1.  **维度匹配**: 经常检查 `input_dim` 是否与 `.npy` 文件中的特征维度一致。
    - Marlin ViT-Base = 768
    - OpenFace AUs = 35 或 288 (取决于是否包含 pose/gaze)
2.  **文件后缀**: 配置文件中的 `feature_suffix` (默认 `_windows.npy`) 必须与提取脚本生成的后缀一致。如果之前提取脚本生成的是直接的 `.npy`（无 `_windows`），请修改配置。
3.  **辅助任务**: 如果不想使用 BioVid 辅助训练，可以在配置中设置 `aux_dataset: enabled: false`。



