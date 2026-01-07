## Syracuse MIL + Biovid 辅助任务多任务训练脚本说明（train_syracuse_biovid_aux.py）

本说明文档详细介绍 `train_syracuse_biovid_aux.py` 脚本的**用途、数据依赖、配置项含义、训练/评估流程、LOOCV 设置、输出文件与指标**，方便你后续复现实验或改代码。

---

## 1. 脚本功能概览

- **脚本位置**：`train_syracuse_biovid_aux.py`
- **核心功能**：  
  - 以 **Syracuse 数据集** 为主任务，使用 MIL + CORAL + Transformer 聚合器做 **ordinal/multiclass/binary/regression**（通常是 ordinal 的 5 类痛感）。  
  - 以 **Biovid 数据集** 的特征/标签为 **辅助任务（auxiliary CE head）**，在同一个 Transformer backbone 上做多任务训练。  
  - 支持：
    - 普通的 **subject-aware train/val/test 划分**（非 LOOCV）；  
    - 类似 `evaluate_syracuse_mil_loocv.py` 的 **受试者级 LOOCV**（只针对 Syracuse 主任务），Biovid 仍使用固定的 train/val/test split。
  - 在 LOOCV 模式下，自动：
    - 保存每个受试者的 **clip-level preds/confusion**；  
    - 汇总 **clip-level 混淆矩阵**；  
    - 额外计算 **video-level majority vote 的 top‑1/top‑2/top‑3 accuracy**。

---

## 2. 依赖数据与路径

### 2.1 Syracuse 主任务数据

- **配置文件**：`config/syracuse_mil_coral_xformer.yaml`
- 关键字段（与本脚本相关）：
  - **`syracuse_feature_root`**：Syracuse 固定组合特征根目录（例如 `/data/Nbi/Syracuse/syracuse_fixed_combo_features`）。  
  - **`meta_excel_path`**：Syracuse 元数据 Excel（例如 `/data/Nbi/Syracuse/meta_with_outcomes.xlsx`）。  
  - **`combos` / `baseline_combo`**：
    - `combos`：训练时可用的 combo 列表（如 `[RGB, RDT]`）。  
    - `baseline_combo`：验证/测试（以及 LOOCV）固定使用的 combo，同时要求 **`baseline_combo ∈ combos`**。  
  - **剪裁与裁剪**：
    - `clip_level: true` 时，本脚本会在 Syracuse DataModule 中启用 clip-level 模式（`clip_bag_size`、`clip_bag_stride` 控制）。  
    - 标签始终是 **video-level 标签**，但输入是由原始窗口分割出来的 clip bag。
  - 其余关于 Syracuse 数据的说明，可参考 `Syracuse/syracuse_mil_training_guide_zh.md`。

### 2.2 Biovid 辅助任务数据

- 在命令行参数中提供：
  - **`--aux_meta_excel`**：Biovid 元数据文件（可以是 CSV 或 Excel）。  
  - **`--aux_feature_root`**：Biovid 特征根目录，内部文件布局为：  
    - `<feature_root>/<video_id><suffix>`，例如：`/data/Nbi/biovid/MMA_RGB_features_new/12345_windows.npy`。  
  - **`--aux_feature_suffix`**：特征文件后缀，默认 `_windows.npy`。
- **Biovid metadata 要求列**（不区分大小写，会自动匹配）：
  - 分割列：`split`（或你指定的 `--aux_split_col`），值为 `train/val/test`。  
  - 视频 ID 列：`video_id`（或 `--aux_video_col`）。  
  - 标签列：`pain_level`（或 `--aux_label_col`）。  
- 脚本中的 `ExcelFeatureDataset` 会：
  - 根据 `split` 列筛选出 `train_split` / `val_split` / `test_split` 对应行。  
  - 丢弃缺失标签的样本。  
  - 从 `feature_root` 读取 `video_id + feature_suffix` 对应的特征文件。  
  - 标签：
    - 若提供 `--aux_cutoffs`，则用 `pain_to_class` 分箱为 ordinal 类别。  
    - 否则直接对 `pain_level` 四舍五入为整数标签。

---

## 3. 命令行参数详解

### 3.1 基本参数（主 + 辅）

- **`--config`** *(必选)*  
  - YAML 配置文件路径，例如：`config/syracuse_mil_coral_xformer.yaml`。  
  - 控制 Syracuse DataModule 与 Transformer 模型的所有超参数。
  - **新增**：现在也可以在该配置文件中直接指定下述 `aux_*` 参数和 `loocv` 开关，从而简化命令行。

- **Biovid 辅助任务相关**（若配置文件已包含，则命令行可选；命令行优先级高于配置）：
  - **`--aux_meta_excel`**：Biovid 元数据（CSV/Excel）。  
  - **`--aux_feature_root`**：Biovid 特征目录。  
  - **`--aux_feature_suffix`**：特征文件后缀，默认 `_windows.npy`。  
  - **`--aux_split_col`**：split 列名，默认 `split`。  
  - **`--aux_video_col`**：video id 列名，默认 `video_id`。  
  - **`--aux_label_col`**：label 列名，默认 `pain_level`。  
  - **`--aux_train_split`** / **`--aux_val_split`** / **`--aux_test_split`**：split 中对应的字符串，默认分别为 `train`、`val`、`test`。  
  - **`--aux_batch_size`**：Biovid dataloader 的 batch size，默认 64。  
  - **`--aux_num_workers`**：Biovid dataloader 的 worker 数，默认 4。  
  - **`--aux_cutoffs`**：可选的 pain level 分箱阈值列表。  
  - **`--aux_loss_weight`**：aux 分支权重。

- **通用训练参数**：
  - **`--default_root_dir`**：Lightning 的默认输出目录，默认 `Syracuse/aux_multitask_runs`（检查点、日志等会在这里）。  
  - **`--accelerator`**：覆盖配置里的 `accelerator`，可传 `gpu` / `cpu` / `auto`。  
  - **`--devices`**：覆盖配置里的 `devices`，指定 GPU 数量（如 `1`）。  
  - **`--max_epochs`**：覆盖配置中的 `max_epochs`。  
  - **`--seed`**：全局随机种子，默认 42。

### 3.2 LOOCV 相关参数（仅作用于 Syracuse 主任务）

- **`--loocv`** *(flag)*  
  - 如果在配置文件中设置了 `loocv: true`，则默认开启。命令行传入 `--loocv` 强制开启。
  - 不加且配置未设为 true：使用一次性 **subject-aware train/val/test 划分**。  
  - 开启：对 Syracuse 主任务执行 **受试者级 LOOCV**，Biovid 辅助任务保持原始 split 不变。

- **`--loocv_limit_subjects`** *(可选)*  
  - 仅在 `--loocv` 模式下生效，用于只跑前 N 个 subject，用于调试或快速试验。  

---

## 4. 代码结构与数据流

### 4.1 Biovid：ExcelFeatureDataset & ExcelBagDataModule

- **`ExcelFeatureDataset`**
  - 初始化时：
    - 读入 `meta_path`（Excel 或 CSV），根据 `split_col` = `train/val/test` 选行。  
    - 用 `video_col` 和 `label_col` 取出 `video_id` 与 `label`。  
    - 丢弃空的 `video_id` 或标签为 NaN 的行。
  - `__getitem__`：
    - 读取 `feature_root / (video_id + feature_suffix)` 为 `np.ndarray`，保证形状至少为 `(T, D)`。  
    - 若设置了 `cutoffs`：调用 Syracuse 通用的 `pain_to_class` 将连续标签分箱成 ordinal 类；否则四舍五入为整数标签。  
    - 返回 `(x, y_tensor, video_id, combo_name)`，其中 `combo_name="aux"`，从而可以复用 `collate_mil_batch`。

- **`ExcelBagDataModule`**
  - `setup` 内部构造 `train_dataset` / `val_dataset` / `test_dataset`。  
  - 对应的 `train_dataloader/val_dataloader/test_dataloader` 使用 Syracuse 的 `collate_mil_batch`，输出字典：
    - `"x"`: `(B, T, D)` padded 序列  
    - `"mask"`: `(B, T)` 有效位置掩码  
    - `"y"`: `(B,)` 标签  
    - `"video_ids"`、`"combos"`：元信息（combo 为 `"aux"`）。

### 4.2 Syracuse：SyracuseBagDataModule

- 由配置文件中的字段传入构造，逻辑与单任务脚本一致：
  - 加载元数据、过滤无效标签。  
  - 按 `subject_id` 做 **subject-aware split** 或 LOOCV（`loocv_test_subject` 不为 None 时）。  
  - 构造 `train/val/test` 三个列表（可能是 video-level 或 clip-level 索引，取决于 `clip_level`）。  
  - 数据增强逻辑（枚举/概率两种模式）、aug feature root、max bag size 等行为与单任务完全相同。

### 4.3 多任务 CombinedLoader

- 训练时使用 `CombinedLoader({"syracuse": ..., "aux": ...}, mode="max_size_cycle")`：
  - 每个 batch 中包含两个子 batch：
    - `batch["syracuse"]`：来自 Syracuse dataloader。  
    - `batch["aux"]`：来自 Biovid dataloader。
  - 模型 `MILCoralTransformer` 需在 `training_step` 中识别这两个 loader 的键，分别计算主任务 loss 和 aux loss，然后按 `aux_loss_weight` 加和。

### 4.4 MILCoralTransformer 主干结构详解

`train_syracuse_biovid_aux.py` 和单任务脚本共用 `model/mil_coral_xformer.py` 里的 `MILCoralTransformer`。理解其结构有助于决定是调数据、调聚合器还是调多头权重：

1. **输入与 clip 组织**  
   - Syracuse 与 Biovid 的 bag 都是 768 维窗口特征（MME backbone 输出）。  
   - 当 YAML 里启用 `clip_level: true`（配合 `clip_bag_size=5`、`clip_bag_stride=1`）时，Syracuse DataModule 会把视频滑窗成 clip bag，再把 video-level 标签广播给每个 clip。  
   - DataModule 会生成布尔 `mask`，用于屏蔽 padding clip，避免聚合器把无效帧计入 bag。

2. **MIL 聚合器（`attn_type`）**  
   - `attn_type` 支持 `xformer`、`simple`、`gated`、`mean`、`mme_xformer`。所有模式都把 `(B,T,768)` 压缩成 `(B, embed_dim)`。  
   - 当前配置 `config/syracuse_mil_coral_xformer.yaml` 中设为 `attn_type: simple`、`embed_dim: 256`、`xformer_heads: 4`、`xformer_latents: 16`、`xformer_dropout: 0.3`。虽然名字叫 `simple`，但内部仍通过 `Linear(768→embed_dim)` + 注意力权重完成实例加权；若改为 `xformer`，则会使用固定数量的 latent queries 做 cross-attn/self-attn（Perceiver 风格），latents/self_layers/head 数量由同名字段控制。

3. **Syracuse 主任务输出**  
   - **CORAL 头**：`SharedCoralLayer(embed_dim, num_classes)`，按 ordinal K-1 阶 logits 训练，loss 权重 `coral_alpha`（当前 0.5）。  
   - **CE 头**：`Linear(embed_dim, num_classes)`，loss 权重 `ce_weight`（当前 1.0），可配合 `class_weights: true`。  
   - **评估链路**：`eval_head` 默认 `"ce"`，`monitor_metric` 也常设为 `val_qwk`（CE logits 计算），因此即便保留 CORAL 输出，它既不驱动早停，也不影响 checkpoint 选择。若 `coral_alpha` 较小，再叠加 CE 主导监控，就会出现“CORAL 不起作用”的现象；想彻底关闭可把 `coral_alpha=0`，反之若想发挥其作用需要同步把 `eval_head/monitor_metric` 切到 `"coral"`。

4. **Biovid 辅助头**  
   - `_aux_step` 里复用同一个聚合器，把 Biovid bag 送入 `aux_ce_head: Linear(embed_dim, aux_num_classes)`；loss 权重为 `aux_loss_weight`（配置项 `biovid_aux_loss_weight`，示例中为 1e-3）。  
   - 训练日志会额外记录 `train_aux_loss/val_aux_loss` 与 `*_aux_acc`，可以快速判断辅助头是否 overfit 或完全不收敛。

5. **总损失**  
   - `total_loss = coral_alpha * L_coral + ce_weight * L_ce + aux_loss_weight * L_aux`。  
   - 只有当对应系数 > 0 且 batch 含有效标签时，相关分支才会回传梯度；这也是排查 CORAL 是否“真正起作用”的第一步——检查它的 loss 是否非零。

理解主干结构后，就能把“文档扩写”集中在真正影响性能的组件上，而不是一味调整数据侧参数。

---

## 5. 训练与评估模式

### 5.1 非 LOOCV 模式（默认）

1. 构建 Syracuse DataModule（无 `loocv_test_subject`）：
   - 调用 `SyracuseBagDataModule.setup()`，按 `val_split_ratio` / `test_split_ratio` 对 **受试者** 做一次性随机划分。  
   - 基于 `baseline_combo` 确定可用视频和切分。
2. 构建 Biovid DataModule：数据划分完全由 `aux_meta_excel` 中的 `split` 列决定。  
3. 构建 `MILCoralTransformer` 模型（含 Syracuse CORAL/CE 头 + Biovid CE 头）。  
4. 训练：

   - `trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=syracuse_dm.val_dataloader())`  
   - **注意**：验证集 **只使用 Syracuse 的 val**，Biovid 辅助任务不参与验证指标。

5. 测试：
   - 训练完后，从 `ModelCheckpoint` 取出 `best_model_path`（监控 `monitor_metric`，通常为 `val_qwk`），若无则 fallback 到 `'last'`。  
   - 使用对应 checkpoint 在 Syracuse 的 test split 上跑 `trainer.test`。

### 5.2 LOOCV 模式（`--loocv`）

1. 通过 `discover_subjects(...)` 获取所有有有效标签的 Syracuse 受试者 ID 列表。  
2. 按顺序遍历每个 subject `sid`：
   - 设置 `loocv_test_subject=sid` 构建 Syracuse DataModule。  
   - 该 subject 的所有视频进入 **test**；其余受试者按 `val_split_ratio` 再划为 train/val。  
   - Biovid DataModule 只初始化一次，整个 LOOCV 全程共用（Biovid train/val/test 不随 subject 变化）。
3. 每一折：
   - 重新构建模型（参数相同但初始化随机），进行训练/验证。  
   - 设置 `ModelCheckpoint(filename=f"syracuse_aux_loocv-subject={sid}-...")`，每折单独保存 best/last。  
   - 使用当前折 best（若无则 last）对 Syracuse test split 进行测试，并收集指标。

---

## 6. 输出文件与日志

假设配置中 `save_dir: Syracuse/xformer_mil_clip_level_auxiliary/5class`，则：

### 6.1 普通模式（非 LOOCV）

- `save_dir` 下会包含：
  - `syracuse_aux-epoch=...-val_qwk=....ckpt` 若干最佳 checkpoint。  
  - `last.ckpt` 与若干 `last-v*.ckpt`（Lightning 的 versioned last）。  
  - 默认 root dir 下（`--default_root_dir`）有 TensorBoard 日志：`syracuse_aux/version_x/...`。

> 目前普通模式下 **没有** 自动导出 Syracuse 逐样本预测 CSV，如需可仿照 LOOCV 分支的写法扩展。

### 6.2 LOOCV 模式输出

在 `save_dir` 下你会看到：

- **每折 checkpoint**：
  - `syracuse_aux_loocv-subject=<sid>-epoch=XX-val_qwk=YYY.ckpt`

- **clip-level 汇总指标**：
  - `syracuse_loocv_multitask.csv`  
    - 列：`subject, test_qwk, test_acc, test_mae, test_f1_macro, test_f1_weighted, test_f1_micro`  
    - 每行一个受试者；最后一行为 `subject=MEAN` 的平均值。

- **clip-level 汇总混淆矩阵**：
  - `syracuse_loocv_multitask_confusion.csv`  
    - 行列均为 `0..K-1` 类别，内容为所有受试者 test clip 上的混淆矩阵累加。

- **per-subject clip-level 预测与混淆**（位于 `save_dir/multitask/` 子目录）：
  - `preds_multitask_subject_<sid>.csv`：
    - 列：`video_id, y_true, y_pred`  
    - `video_id` 在 clip-level 模式下是 `"<video_base>_clip_<idx>"`，可据此恢复 video-level。  
  - `confusion_multitask_subject_<sid>.csv`：
    - 第一列为 `true\pred`，后续为各预测类计数。

- **video-level majority vote top‑k 指标**：
  - 控制台会打印：

    ```text
    Syracuse LOOCV (multitask) video-level majority-vote accuracy:
      top-1=..., top-2=..., top-3=...
    ```

  - 同时写 CSV：`syracuse_loocv_multitask_video_topk.csv`，格式：
    - `metric, value, correct, total`
    - 三行：
      - `top1_acc, <float>, correct_top1, total_videos`  
      - `top2_acc, <float>, correct_top2, total_videos`  
      - `top3_acc, <float>, correct_top3, total_videos`

---

## 7. video-level majority vote 计算细节

- 从 `multitask/preds_multitask_subject_*.csv` 读取所有折的 Syracuse test clip-level 预测。  
- 对每行：
  - `video_id`（可能为 `IMG_0003_clip_0005`）和 `y_true`、`y_pred`。  
  - 用 `'_clip_'` 分割，取前半部分作为 `video_base`，这样同一视频的所有 clips 会被聚合。  
  - `video_y_true[video_base]` 记录该视频的真标签（只要第一次读到即可）。  
  - `video_pred_votes[video_base]` 追加该 clip 的预测类别。
- 对每个 `video_base`：
  - 对 `video_pred_votes[video_base]` 做频数统计（`Counter`）。  
  - 按 “频数降序 + 类别 ID 升序” 排序，得到投票排序 `top_labels`。  
  - 计算：
    - **top‑1**：`top_labels[0] == y_true`。  
    - **top‑2**：`y_true in top_labels[:2]`。  
    - **top‑3**：`y_true in top_labels[:3]`。
- 所有视频统一累积计数，得到整体的 top‑1/top‑2/top‑3 accuracy。

> 这样得到的是 **真正 video-level 的性能**（每个视频一票），与 clip-level 指标互补，可以更公平地比较不同方法。

---

## 8. 典型运行示例

### 8.1 极简运行（所有参数均在 config 中指定）

由于现在 `aux_*` 参数和 `loocv` 开关均可写入 YAML，运行命令可以非常简洁：

```bash
cd /data/Nbi/Marlin/MARLIN

# 若 YAML 中已配置 aux_meta_excel, aux_feature_root, loocv=true 等
python train_syracuse_biovid_aux.py \
  --config config/syracuse_mil_coral_xformer.yaml
```

### 8.2 命令行覆盖示例

如果临时想换一个 Biovid 的特征目录，或者临时关闭 LOOCV（假设 YAML 里 `loocv: true`，目前脚本逻辑是 `OR` 关系，所以如果 YAML 是 true，无法通过 CLI 关掉；如果 YAML 是 false，可以通过 `--loocv` 开启。若需临时关闭建议修改 YAML 或复制一份）：

```bash
python train_syracuse_biovid_aux.py \
  --config config/syracuse_mil_coral_xformer.yaml \
  --aux_feature_root /data/Nbi/biovid/SOME_OTHER_FEATURES
```

### 8.3 传统完整命令（仅作参考）

```bash
python train_syracuse_biovid_aux.py \
  --config config/syracuse_mil_coral_xformer.yaml \
  --aux_meta_excel /data/Nbi/biovid/biovid_pain_labels.csv \
  --aux_feature_root /data/Nbi/biovid/MMA_RGB_features_new \
  --aux_feature_suffix _windows.npy \
  --aux_train_split train \
  --aux_val_split val \
  --aux_test_split test \
  --loocv
```

可选：只跑前 5 个受试者测试流程：

```bash
  --loocv \
  --loocv_limit_subjects 5
```

---

## 9. 实践建议与注意事项

- **Syracuse 与 Biovid 的损失权重平衡**：
  - 若发现 Syracuse 主任务指标明显下降，而 Biovid 训练过强，可以适当减小 `biovid_aux_loss_weight` 或 CLI 的 `--aux_loss_weight`。  
  - 也可以降低 Biovid 的 batch size，使得单位 step 内 aux loss 对总体更新的影响变小。

- **clip-level 与 video-level 指标的解读**：
  - 当前 Lightning 的 `test_acc/test_qwk` 等是 clip-level 的（在 Syracuse DataModule 的输出维度上统计）。  
  - 若研究问题是 “按视频评估痛感等级”，**应重点关注 video-level majority vote top‑k 结果**，而不是只看 clip-level。

- **CORAL 分支的实际状态**：
  - 由于多任务配置里 `coral_alpha` 相对较低（示例为 0.5），且 `eval_head` 默认使用 CE，训练/验证指标完全由 CE logits 驱动，CORAL loss 往往只是一个附加正则项。  
  - 如果日志中 `train_coral_loss` 长期接近 0 或梯度贡献微弱，可以把 `coral_alpha` 直接设为 0 以减少分支开销；若要重新启用，则需同步提高 `coral_alpha`、降低 `ce_weight`，并把 `monitor_metric`/`eval_head` 切换到 `"coral"`，否则仍会出现“看起来没用”的现象。

- **LOOCV 方差较大是常态**：
  - 受试者数量有限、每个 subject 的样本数差异大时，单个 subject 的指标抖动很正常。  
  - 建议重点看：
    - `syracuse_loocv_multitask.csv` 的 `MEAN` 行；  
    - `syracuse_loocv_multitask_video_topk.csv` 的总体 top‑1/2/3；  
    - 汇总混淆矩阵 `syracuse_loocv_multitask_confusion.csv`。

如果后续你希望在 Biovid 辅助任务上也做类似 LOOCV，或者想导出 Biovid 的逐样本预测与 video-level 指标，也可以在当前框架下按相同模式扩展。***
 End Patch

덞
Invalid Patch: The patch content is not valid according to the apply_patch grammar. Error: Expected one of: '*** Add File: ', '*** Update File: ' at line 154 col 1. Got: '덞' instead. Remember, this tool only accepts properly formatted patches with the expected structure. Use the exact markers (*** Begin Patch, *** Add File, *** Update File, *** End Patch). Don't include any other text before, after, or between these markers. Make sure every hunk inside the patch is either an add or update hunk. Also note: there's no support for deleting files via patch; apply an empty update instead. If you're trying to add more than one file or update, ensure each one is preceded by its own '*** Add File:' or '*** Update File:' line. Also, ensure there is a newline at the end of the patch.」




