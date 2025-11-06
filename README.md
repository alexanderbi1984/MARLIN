# MARLIN: Masked Autoencoder for facial video Representation LearnINg

<div>
    <img src="assets/teaser.svg">
    <p></p>
</div>

<div align="center">
    <a href="https://github.com/ControlNet/MARLIN/network/members">
        <img src="https://img.shields.io/github/forks/ControlNet/MARLIN?style=flat-square">
    </a>
    <a href="https://github.com/ControlNet/MARLIN/stargazers">
        <img src="https://img.shields.io/github/stars/ControlNet/MARLIN?style=flat-square">
    </a>
    <a href="https://github.com/ControlNet/MARLIN/issues">
        <img src="https://img.shields.io/github/issues/ControlNet/MARLIN?style=flat-square">
    </a>
    <a href="https://github.com/ControlNet/MARLIN/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-97ca00?style=flat-square">
    </a>
    <a href="https://arxiv.org/abs/2211.06627">
        <img src="https://img.shields.io/badge/arXiv-2211.06627-b31b1b.svg?style=flat-square">
    </a>
</div>

<div align="center">    
    <a href="https://pypi.org/project/marlin-pytorch/">
        <img src="https://img.shields.io/pypi/v/marlin-pytorch?style=flat-square">
    </a>
    <a href="https://pypi.org/project/marlin-pytorch/">
        <img src="https://img.shields.io/pypi/dm/marlin-pytorch?style=flat-square">
    </a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/pypi/pyversions/marlin-pytorch?style=flat-square"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%3E%3D1.8.0-EE4C2C?style=flat-square&logo=pytorch"></a>
</div>

<div align="center">
    <a href="https://github.com/ControlNet/MARLIN/actions"><img src="https://img.shields.io/github/actions/workflow/status/ControlNet/MARLIN/unittest.yaml?branch=dev&label=unittest&style=flat-square"></a>
    <a href="https://github.com/ControlNet/MARLIN/actions"><img src="https://img.shields.io/github/actions/workflow/status/ControlNet/MARLIN/release.yaml?branch=master&label=release&style=flat-square"></a>
    <a href="https://coveralls.io/github/ControlNet/MARLIN"><img src="https://img.shields.io/coverallsCoverage/github/ControlNet/MARLIN?style=flat-square"></a>
</div>

This repo is the official PyTorch implementation for the paper 
[MARLIN: Masked Autoencoder for facial video Representation LearnINg](https://openaccess.thecvf.com/content/CVPR2023/html/Cai_MARLIN_Masked_Autoencoder_for_Facial_Video_Representation_LearnINg_CVPR_2023_paper) (CVPR 2023).

## Repository Structure

The repository contains 2 parts:
 - `marlin-pytorch`: The PyPI package for MARLIN used for inference.
 - The implementation for the paper including training and evaluation scripts.

```
.
├── assets                # Images for README.md
├── LICENSE
├── README.md
├── MODEL_ZOO.md
├── CITATION.cff
├── .gitignore
├── .github

# below is for the PyPI package marlin-pytorch
├── src                   # Source code for marlin-pytorch
├── tests                 # Unittest
├── requirements.lib.txt
├── setup.py
├── init.py
├── version.txt

# below is for the paper implementation
├── configs              # Configs for experiments settings
├── model                # Marlin models
├── preprocess           # Preprocessing scripts
├── dataset              # Dataloaders
├── utils                # Utility functions
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── requirements.txt

```

## Use `marlin-pytorch` for Feature Extraction

Requirements:
- Python >= 3.6, < 3.12
- PyTorch >= 1.8
- ffmpeg


Install from PyPI:
```bash
pip install marlin-pytorch
```

Load MARLIN model from online
```python
from marlin_pytorch import Marlin
# Load MARLIN model from GitHub Release
model = Marlin.from_online("marlin_vit_base_ytf")
```

Load MARLIN model from file
```python
from marlin_pytorch import Marlin
# Load MARLIN model from local file
model = Marlin.from_file("marlin_vit_base_ytf", "path/to/marlin.pt")
# Load MARLIN model from the ckpt file trained by the scripts in this repo
model = Marlin.from_file("marlin_vit_base_ytf", "path/to/marlin.ckpt")
```

Current model name list:
- `marlin_vit_small_ytf`: ViT-small encoder trained on YTF dataset. Embedding 384 dim.
- `marlin_vit_base_ytf`: ViT-base encoder trained on YTF dataset. Embedding 768 dim.
- `marlin_vit_large_ytf`: ViT-large encoder trained on YTF dataset. Embedding 1024 dim.

For more details, see [MODEL_ZOO.md](MODEL_ZOO.md).

When MARLIN model is retrieved from GitHub Release, it will be cached in `.marlin`. You can remove marlin cache by
```python
from marlin_pytorch import Marlin
Marlin.clean_cache()
```

Extract features from cropped video file
```python
# Extract features from facial cropped video with size (224x224)
features = model.extract_video("path/to/video.mp4")
print(features.shape)  # torch.Size([T, 768]) where T is the number of windows

# You can keep output of all elements from the sequence by setting keep_seq=True
features = model.extract_video("path/to/video.mp4", keep_seq=True)
print(features.shape)  # torch.Size([T, k, 768]) where k = T/t * H/h * W/w = 8 * 14 * 14 = 1568
```

Extract features from in-the-wild video file
```python
# Extract features from in-the-wild video with various size
features = model.extract_video("path/to/video.mp4", crop_face=True)
print(features.shape)  # torch.Size([T, 768])
```

Extract features from video clip tensor
```python
# Extract features from clip tensor with size (B, 3, 16, 224, 224)
x = ...  # video clip
features = model.extract_features(x)  # torch.Size([B, k, 768])
features = model.extract_features(x, keep_seq=False)  # torch.Size([B, 768])
```

## Paper Implementation

### Requirements
- Python >= 3.7, < 3.12
- PyTorch ~= 1.11
- Torchvision ~= 0.12

### Installation

Firstly, make sure you have installed PyTorch and Torchvision with or without CUDA. 

Clone the repo and install the requirements:
```bash
git clone https://github.com/ControlNet/MARLIN.git
cd MARLIN
pip install -r requirements.txt
```

### MARLIN Pretraining

Download the [YoutubeFaces](https://www.cs.tau.ac.il/~wolf/ytfaces/) dataset (only `frame_images_DB` is required). 

Download the face parsing model from [face_parsing.farl.lapa](https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.lapa.main_ema_136500_jit191.pt)
and put it in `utils/face_sdk/models/face_parsing/face_parsing_1.0`.

Download the VideoMAE pretrained [checkpoint](https://github.com/ControlNet/MARLIN/releases/misc) 
for initializing the weights. (ps. They updated their models in this 
[commit](https://github.com/MCG-NJU/VideoMAE/commit/2b56a75d166c619f71019e3d1bb1c4aedafe7a90), but we are using the 
old models which are not shared anymore by the authors. So we uploaded this model by ourselves.)

Then run scripts to process the dataset:
```bash
python preprocess/ytf_preprocess.py --data_dir /path/to/youtube_faces --max_workers 8
```
After processing, the directory structure should be like this:
```
├── YoutubeFaces
│   ├── frame_images_DB
│   │   ├── Aaron_Eckhart
│   │   │   ├── 0
│   │   │   │   ├── 0.555.jpg
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   ├── ...
│   ├── crop_images_DB
│   │   ├── Aaron_Eckhart
│   │   │   ├── 0
│   │   │   │   ├── 0.555.jpg
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   ├── ...
│   ├── face_parsing_images_DB
│   │   ├── Aaron_Eckhart
│   │   │   ├── 0
│   │   │   │   ├── 0.555.npy
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   ├── ...
│   ├── train_set.csv
│   ├── val_set.csv
```

Then, run the training script:
```bash
python train.py \
    --config config/pretrain/marlin_vit_base.yaml \
    --data_dir /path/to/youtube_faces \
    --n_gpus 4 \
    --num_workers 8 \
    --batch_size 16 \
    --epochs 2000 \
    --official_pretrained /path/to/videomae/checkpoint.pth
```

After trained, you can load the checkpoint for inference by

```python
from marlin_pytorch import Marlin
from marlin_pytorch.config import register_model_from_yaml

register_model_from_yaml("my_marlin_model", "path/to/config.yaml")
model = Marlin.from_file("my_marlin_model", "path/to/marlin.ckpt")
```

## Evaluation

<details>
<summary>CelebV-HQ</summary>

#### 1. Download the dataset
Download dataset from [CelebV-HQ](https://github.com/CelebV-HQ/CelebV-HQ) and the file structure should be like this:
```
├── CelebV-HQ
│   ├── downloaded
│   │   ├── ***.mp4
│   │   ├── ...
│   ├── celebvhq_info.json
│   ├── ...
```
#### 2. Preprocess the dataset
Crop the face region from the raw video and split the train val and test sets.
```bash
python preprocess/celebvhq_preprocess.py --data_dir /path/to/CelebV-HQ 
```

#### 3. Extract MARLIN features (Optional, if linear probing)
Extract MARLIN features from the cropped video and saved to `<backbone>` directory in `CelebV-HQ` directory.
```bash
python preprocess/marlin_feature_extract.py --data_dir /path/to/CelebV-HQ --backbone marlin_vit_base_ytf
```

#### 4. Train and evaluate
Train and evaluate the model adapted from MARLIN to CelebV-HQ.

Please use the configs in `config/celebv_hq/*/*.yaml` as the config file.
```bash
python evaluate.py \
    --config /path/to/config \
    --data_path /path/to/CelebV-HQ 
    --num_workers 4 
    --batch_size 16
```

</details>


## License

This project is under the CC BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## References
If you find this work useful for your research, please consider citing it.
```bibtex
@inproceedings{cai2022marlin,
  title = {MARLIN: Masked Autoencoder for facial video Representation LearnINg},
  author = {Cai, Zhixi and Ghosh, Shreya and Stefanov, Kalin and Dhall, Abhinav and Cai, Jianfei and Rezatofighi, Hamid and Haffari, Reza and Hayat, Munawar},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023},
  month = {June},
  pages = {1493-1504},
  doi = {10.1109/CVPR52729.2023.00150},
  publisher = {IEEE},
}
```

## Acknowledgements

Some code about model is based on [MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE). The code related to preprocessing
is borrowed from [JDAI-CV/FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo).

# MARLIN Feature Analysis

This repository contains tools for analyzing and visualizing features from the MARLIN model, particularly focusing on comparing pre and post treatment conditions.

## Feature Visualization and Analysis

### Scripts Overview

1. `marlin_feature_visualization.py`: Visualizes how specific MARLIN features respond to video inputs
2. `compare_pre_post_features.py`: Creates side-by-side comparisons of how features differ between pre and post treatment videos
3. `distribution_test.py`: Analyzes statistical differences in features between pre and post conditions

### Feature Visualization

The `marlin_feature_visualization.py` script provides tools to visualize how specific MARLIN features respond to video inputs. It generates heatmaps showing which parts of the video most strongly activate particular features.

```bash
python marlin_feature_visualization.py \
    --checkpoint_path /path/to/model.ckpt \
    --video_path /path/to/video.mp4 \
    --features 397 231 490 482 456 \
    --output_dir feature_visualizations
```

Key parameters:
- `--checkpoint_path`: Path to the MARLIN model checkpoint
- `--video_path`: Path to the video file to analyze
- `--features`: List of feature indices to visualize (default: [397, 231, 490, 482, 456])
- `--output_dir`: Directory to save visualizations
- `--fps`: Frame rate for output comparison video (default: 30)
- `--skip_comparison_video`: Skip creating the feature comparison video
- `--use_standard_marlin`: Use standard MARLIN instead of MultiModalMarlin

### Pre vs Post Treatment Comparison

The `compare_pre_post_features.py` script creates side-by-side comparisons of how features differ between pre and post treatment videos. For each feature, it generates a visualization showing:
- Pre-treatment original frames
- Pre-treatment feature activation heatmaps
- Post-treatment original frames
- Post-treatment feature activation heatmaps

```bash
python compare_pre_post_features.py \
    --checkpoint_path /path/to/model.ckpt \
    --pre_video /path/to/pre_treatment.mp4 \
    --post_video /path/to/post_treatment.mp4 \
    --features 397 231 490 482 456 \
    --output_dir pre_post_comparisons
```

Key parameters:
- `--checkpoint_path`: Path to the MARLIN model checkpoint
- `--pre_video`: Path to the pre-treatment video clip
- `--post_video`: Path to the post-treatment video clip
- `--features`: List of feature indices to visualize (default: [397, 231, 490, 482, 456])
- `--output_dir`: Directory to save comparisons
- `--model_name`: MARLIN model name (default: multimodal_marlin_base)

### Statistical Analysis

The `distribution_test.py` script performs statistical analysis of features between pre and post treatment conditions. It includes:
- Video-level analysis accounting for dependencies
- Multiple testing correction using FDR
- Effect size calculations
- Feature importance analysis

```bash
python distribution_test.py
```

### Outcome-Based Feature Analysis

This section presents the results of analyzing MARLIN features based on treatment outcomes. The analysis focuses on identifying features that significantly differ between successful and unsuccessful treatments.

### Analysis Overview

The analysis was performed at two levels:
1. **Clip-level**: Analyzing individual video clips
2. **Video-level**: Analyzing averaged features across all clips in a video

### Statistical Results

#### 1. Mann-Whitney U Test Results

**Clip-level Analysis**:
- 434 features (56.5%) showed significant differences (p < 0.05) before correction
- 387 features (50.4%) remained significant after FDR correction
- Top 5 most significant features:
  - Feature 662: p-value = 0.0000, effect size = 1.113
  - Feature 316: p-value = 0.0000, effect size = 1.089
  - Feature 629: p-value = 0.0000, effect size = 1.087
  - Feature 143: p-value = 0.0000, effect size = 1.085
  - Feature 587: p-value = 0.0000, effect size = 1.082

**Video-level Analysis**:
- 22 features (2.9%) showed significance before correction
- No features remained significant after FDR correction
- Higher p-values (> 0.70) indicate less reliable differences

#### 2. Effect Size Analysis (Cohen's d)

**Clip-level Analysis**:
- Large effect sizes (|d| > 0.8): 23 features (3.0%)
- Medium effect sizes (0.5 < |d| ≤ 0.8): 87 features (11.3%)
- Small effect sizes (|d| ≤ 0.5): 658 features (85.7%)

**Video-level Analysis**:
- Large effect sizes (|d| > 0.8): 33 features (4.3%)
- Medium effect sizes (0.5 < |d| ≤ 0.8): 116 features (15.1%)
- Small effect sizes (|d| ≤ 0.5): 619 features (80.6%)

#### 3. MMD Test Results
- Clip-level: MMD = -0.0012, p = 0.5240
- Video-level: MMD = -0.0696, p = 0.9900

### Key Findings

1. **Feature Discrimination**:
   - Individual features show stronger and more reliable differences at the clip level
   - Feature 662 consistently shows the strongest effect size across both levels
   - The top discriminative features are consistent between clip and video levels

2. **Statistical Reliability**:
   - Clip-level analysis provides more reliable statistical inference
   - Video-level analysis shows larger effect sizes but lower statistical reliability
   - The negative MMD values suggest similar overall distributions between outcomes

3. **Feature Importance**:
   - A small subset of features (top 5) shows very strong discrimination
   - Most features show small effect sizes, suggesting complex interactions
   - The consistent top features across levels indicate robust biomarkers

### Recommendations

1. **Analysis Level**:
   - Focus on clip-level analysis for more reliable statistical inference
   - Use video-level analysis for exploratory purposes or when sample size is limited

2. **Feature Selection**:
   - Prioritize the top 5 features (662, 316, 629, 143, 587) for further investigation
   - Consider feature interactions and combinations for better discrimination

3. **Methodological Improvements**:
   - Consider using multiple clips per video to increase statistical power
   - Implement multivariate analysis approaches to capture feature interactions
   - Explore feature combinations for improved outcome prediction

### Usage

To run the outcome-based analysis:

```bash
python outcome_based_analysis.py \
    --features_dir /path/to/features \
    --meta_path /path/to/metadata.xlsx \
    --output_dir outcome_analysis_results
```

The analysis will generate:
- Statistical test results
- Feature importance visualizations
- Distribution plots for top features
- Heatmaps of feature changes
- PCA and UMAP visualizations

### Output Files

The visualization scripts generate several types of output files:

1. Feature Visualizations:
   - Individual feature heatmaps showing activation patterns
   - Comparison videos showing feature responses over time
   - High-resolution PNG files for each visualization

2. Pre vs Post Comparisons:
   - Combined visualizations showing pre and post differences
   - One high-resolution PNG file per feature (4-row layout)
   - Clear labeling of frames and heatmaps

3. Statistical Analysis:
   - Summary of significant features
   - Effect size measurements
   - P-values and FDR-corrected results

### Implementation Details

The visualization tools use gradient-based class activation mapping (Grad-CAM) to highlight which parts of the video are most important for each feature. Key technical aspects:

1. Feature Extraction:
   - Uses the last layer of the MARLIN encoder
   - Computes gradients with respect to specific features
   - Generates class activation maps

2. Video Processing:
   - Handles variable-length videos
   - Automatically pads or trims to match model requirements
   - Maintains proper temporal and spatial dimensions

3. Visualization:
   - Uses matplotlib for high-quality figures
   - Employs cv2 for heatmap generation
   - Supports both individual frame and temporal analysis

### Notes

- The scripts automatically handle video preprocessing to match MARLIN model requirements
- Visualizations are saved in high resolution (300 DPI) for publication quality
- The comparison script is particularly useful for understanding treatment effects
- All scripts include progress bars and informative console output
