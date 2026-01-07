# Syracuse 痛感 MIL + CORAL + Temporal Transformer 训练与评估指南

本指南说明当前仓库中 Syracuse 数据集的最新训练思路、数据组织、脚本用法、配置项语义，以及我们为数据清洗与稳定训练所做的修复与改进。适用于单任务与受试者级 LOOCV 两条路径。

## 数据与目录
- 固定组合（fixed combos）离线特征根目录：
  - `/data/Nbi/Syracuse/syracuse_fixed_combo_features`
  - 10 个组合（示例）：`RGB, RGD, RGT, RBD, RBT, RDT, GBD, GBT, GDT, BDT`
  - 文件布局：`<combo>/<video_base>_windows.npy`，每个文件形状 `(N,768)`，`float32`，`N` 因视频不同而不同。
- 增广视图（swap-face）根目录：
  - `/data/Nbi/Syracuse/syracuse_aug_fixed_combo_features`
  - 带数字后缀的增广视图：`IMG_0003_1_windows.npy`、`IMG_0003_2_windows.npy` 等。同目录下无后缀文件视为“原始视图”，训练时默认不从增广目录读取无后缀以避免重复。
- 元数据 Excel：
  - `/data/Nbi/Syracuse/meta_with_outcomes.xlsx`
  - 最小列需求：`file_name, subject_id, pain_level, visit_type, outcome`；可选 `class_K`（如 `class_4`）。

## 数据加载与清洗（dataset/syracuse_bag.py）
- 加载特征：按 combo 目录读取 `video_base_windows.npy`，得到视频级 bag `(T,768)`。
- Excel 清洗与标签归一：
  - `pain_level` 使用 `pd.to_numeric(errors='coerce')` 强制为数值；非数值（空、注释如 “as note”）变为 NaN 并视为缺失。
  - 二分类 `outcome` 使用 `_normalize_outcome`：
    - 正类：`positive/pos/+/1/true/yes/y/t` → 1；负类：`negative/neg/-/0/false/no/n/f` → 0；
    - 空白/NaN/未知文本 → 缺失；若 outcome 缺失则回退用 `pain_level >= 4.0` 得到 1/0；两者都无效则样本剔除。
  - 序/多分类 `class_K` 使用 `_normalize_class_value`：
    - 支持 0..K-1 或 1..K（自动减 1）；空/NaN/非法文本 → 缺失。
    - 若 `class_K` 缺失则用 `pain_level + cutoffs` 分箱；两者都无效则样本剔除。
- 有效性判定：
  - 回归：`pain_level` 为数值。
  - 序/多分类：`class_K` 归一成功，或 `pain_level` 可分箱。
  - 二分类：`outcome` 归一成功，或 `pain_level` 可阈值化。
- 训练增广两种模式：
  - 枚举模式（`train_enumerate_all_views: true`）：
    - 对训练集物化为“每视频 × 每 combo ×（原始 + 至多 K 个增广）”。
    - K = `min(train_max_aug_per_combo, 实际可用增广数)`。
    - 此模式下每个 epoch 一次遍历 + shuffle，`WeightedRandomSampler` 不生效。
  - 概率模式（`train_enumerate_all_views: false`）：
    - 每个样本先随机选 combo，再以 `train_aug_ratio` 概率替换成增广视图；否则用原视图。
    - 可启用 `WeightedRandomSampler`（`use_weighted_sampler: true`），并用 `train_epoch_multiplier` 放大每个 epoch 步数。
- 受试者级拆分与 LOOCV：
  - 普通拆分：按 `subject_id` 分组后打乱，按 `val_split_ratio/test_split_ratio` 切分受试者集合；验证/测试仅使用 baseline combo 的原视图。
  - LOOCV：`loocv_test_subject` 指定本折测试受试者；其余受试者中抽取验证受试者（至少 1 个），剩余为训练。

## 模型（model/mil_coral_xformer.py）
- MIL 聚合器：
  - `xformer`：Perceiver 风格 latent cross-attn（latents 作为 Q，实例序列作为 K/V）+ latent self-attn，正弦位置编码，读出 latent 平均。
  - `simple/gated`：注意力池化（可选 gated）。
- 头部与损失：
  - CORAL（序回归）：共享权重与阈值；`coral_alpha` 控制权重。
  - CE（多分类）：`ce_weight` 控制权重；支持 `class_weights`（按训练集反频率计算）。
  - 损失与指标中的“无效标签”过滤：
    - CE/CORAL 都仅用 `0..K-1` 的有效标签参与计算，避免 GPU 上 CE 断言失败。
- 指标：
  - 训练：`train_loss`、分头损失。
  - 验证：`val_loss`、`val_qwk`（主）、`val_acc`（已显示在进度条）、`val_mae`。
  - 测试：除上面外，还打印混淆矩阵，记录 `test_f1_macro/weighted/micro`。
  - 预测：`predict_step` 输出 `y_true/y_pred/y_probs/video_ids/combos`；`y_probs` 为从 CORAL logits 转换的 K 类概率。

## 训练/评估脚本
- 单任务：`evaluate_syracuse_mil.py`
  - 读取 YAML，构建 DataModule 与模型；指定 `accelerator: gpu`、`devices: 1`。
  - 训练后测试：`ckpt_path='best'`（若无则 `'last'`）。
  - 导出测试逐样本预测：`save_dir/test_preds.csv`（含各类概率 `p_0..p_{K-1}`）；计算并打印 AUC（binary：列 1 概率；多类：OvR 宏平均）。
  - 写入 `save_dir/config_used.yaml` 保存本次实验配置与关键解析结果（便于溯源）。
- 受试者级 LOOCV：`evaluate_syracuse_mil_loocv.py`
  - 单一 combo（不混合）下循环各受试者为测试；验证/测试仅 baseline combo 原视图。
  - 若 `class_weights: true` 且任务为序/多分类，则对 CE 头按训练集反频率计算权重。
  - 每折保存：
    - 预测明细 `preds_<combo>_subject_<sid>.csv`
    - 混淆矩阵 `confusion_<combo>_subject_<sid>.csv`
    - 并把 `config_used.yaml` 写入 `--save_dir` 根目录。
  - 测试用 `ckpt_path='best'`，否则回退 `'last'`。
- 全部 10 个 combo 的 LOOCV：`evaluate_syracuse_mil_loocv_all.py`
  - 逐个 combo 执行 `evaluate_syracuse_mil_loocv.py` 的流程，保存：
    - 每个 combo 的折级结果 `Syracuse/xformer_mil_loocv/syracuse_loocv_<combo>.csv`（含 MEAN 行）
    - 组合级汇总混淆矩阵 `confusion_summary_<combo>.csv`（各折相加）
    - 跨 combo 的总汇总 `syracuse_loocv_summary.csv`
  - 写入 `Syracuse/xformer_mil_loocv/config_used.yaml`。

## 配置项要点（config/syracuse_mil_coral_xformer.yaml）
- 数据路径：`syracuse_feature_root`，`aug_feature_root`，`meta_excel_path`。
- 组合：`combos`，`baseline_combo`。
- 任务与标签：`task`（`ordinal|multiclass|binary|regression`），`num_classes`，`pain_class_cutoffs`。
- 训练增广：
  - 枚举：`train_enumerate_all_views: true`，`train_max_aug_per_combo`，`train_include_original`。
  - 概率：`train_enumerate_all_views: false` 时，`train_aug_ratio` 生效。
- 类均衡：
  - 采样层：`use_weighted_sampler: true`（仅在“非枚举模式”生效），`train_epoch_multiplier` 控制每 epoch 过采样倍数。
  - 损失层：`class_weights: true` 为 CE 头按训练集反频率加权（单任务与 LOOCV 均支持）。
- 优化与模型：`learning_rate`，`weight_decay`，`ce_weight`，`coral_alpha`，`attn_type`，`embed_dim` 等。
- 设备：`accelerator: gpu`，`devices: 1`。
- 输出：`save_dir`，`save_preds`。

## 常见问题与建议
- “预测塌缩到端点类（如 0/3）”：
  - 建议关闭枚举模式，启用加权采样与概率增广：
    - `train_enumerate_all_views: false`
    - `use_weighted_sampler: true`
    - `train_epoch_multiplier: 4`（或更高）
    - `train_aug_ratio: 0.5–0.75`
  - 调整损失配比：降低 CE 主导（如 `ce_weight: 0.5`）或临时关 CORAL（`coral_alpha: 0.0`）观察稳定性；保留 `class_weights: true`。
- “val_qwk 很高/波动大”：
  - LOOCV 验证集按受试者随机抽取且数量少，方差较大；可提高 `val_split_ratio` 或固定验证受试者列表（可扩展）。
- “CE/NLL 的 CUDA 断言失败”：
  - 通常是无效标签混入。我们已在数据清洗和损失中加入过滤；若仍出现，检查 Excel 值是否为空/注释文本，或 `num_classes` 与 `class_K/cutoffs` 是否一致。

## 运行示例
- 单任务（保存逐样本预测与 AUC）：
  - `python evaluate_syracuse_mil.py --config config/syracuse_mil_coral_xformer.yaml`
- 单 combo 的受试者级 LOOCV：
  - `python evaluate_syracuse_mil_loocv.py --config config/syracuse_mil_coral_xformer.yaml --combo RGB --save_dir Syracuse/xformer_mil_loocv`
- 全部 10 个 combo 的 LOOCV：
  - `python evaluate_syracuse_mil_loocv_all.py --config config/syracuse_mil_coral_xformer.yaml --save_dir Syracuse/xformer_mil_loocv`

## 版本与溯源
- 所有脚本在运行时会把“实际生效的配置”保存到对应的 `save_dir/config_used.yaml` 中（单任务）或 `--save_dir/config_used.yaml`（LOOCV 路径），以便结果溯源与复现实验。

如需进一步增强（自动等频分箱、CE label smoothing/focal、在枚举模式下的类均衡采样、二分类阈值可配置等），可以按需开启开发。

