import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.getcwd()) # Ensure local modules can be imported

try:
    import torch
    from model.mil_coral_xformer import MILCoralTransformer
except ImportError:
    print("Warning: Could not import torch or MILCoralTransformer. 'model_features' mode will not work.")

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def load_loocv_csv(path):
    """Read LOOCV summary CSV, return MEAN row"""
    df = pd.read_csv(path)
    # Find subject 'MEAN' (case insensitive) or calculate mean
    mean_row = df[df['subject'].astype(str).str.upper() == 'MEAN']
    if len(mean_row) > 0:
        return mean_row.iloc[0]
    else:
        # If no MEAN row, calculate manually for numeric columns
        return df.mean(numeric_only=True)

def plot_comparison(experiment_paths, labels, output_dir):
    """
    Compare metrics across different experiments.
    experiment_paths: List of paths to result CSVs
    labels: List of labels for each experiment (e.g. 'Baseline', 'My Method')
    """
    metrics = ['test_qwk', 'test_acc', 'test_f1_macro']
    data = []

    for path, label in zip(experiment_paths, labels):
        if not os.path.exists(path):
            print(f"Warning: File not found {path}")
            continue
        row = load_loocv_csv(path)
        res = {'Experiment': label}
        for m in metrics:
            res[m] = row.get(m, 0)
        data.append(res)

    if not data:
        print("No valid data found to plot.")
        return

    df_plot = pd.DataFrame(data)
    df_melted = df_plot.melt(id_vars='Experiment', value_vars=metrics, var_name='Metric', value_name='Score')

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melted, x='Metric', y='Score', hue='Experiment', palette='viridis')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)

    plt.title('Performance Comparison across Feature Sets/Methods')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'comparison_plot.png')
    plt.savefig(out_path, dpi=300)
    print(f"Comparison plot saved to: {out_path}")

def plot_confusion(csv_path, output_dir):
    """Plot confusion matrix heatmap from CSV"""
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    # Read confusion matrix CSV (assuming first column is true\pred labels)
    df = pd.read_csv(csv_path, index_col=0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix: {os.path.basename(csv_path)}')
    
    out_path = os.path.join(output_dir, f"conf_mat_{os.path.basename(csv_path)}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Confusion matrix plot saved to: {out_path}")

def analyze_features_tsne(meta_path, feature_root, output_dir, limit=500, feature_suffix='_windows.npy'):
    """
    Load features directly and perform t-SNE visualization.
    Shows if features are separable by Pain Level.
    """
    print(f"Loading features from {feature_root}...")
    
    # Read metadata
    ext = os.path.splitext(meta_path)[-1].lower()
    if ext in ['.xls', '.xlsx']:
        df = pd.read_excel(meta_path)
    else:
        df = pd.read_csv(meta_path)
    
    # Normalize columns
    df.columns = [c.lower() for c in df.columns]
    
    # Heuristic for video column
    video_col = None
    possible_video_cols = ['video_id', 'file_name', 'filename', 'video_name']
    for c in possible_video_cols:
        if c in df.columns:
            video_col = c
            break
            
    if video_col is None:
        video_col = df.columns[0] # Fallback
        
    label_col = 'pain_level' if 'pain_level' in df.columns else df.columns[1]

    features = []
    labels = []
    
    count = 0
    for _, row in df.iterrows():
        if count >= limit: break
        
        vid = str(row[video_col]).strip()
        
        # Remove extension if present (e.g. .MP4)
        if "." in vid:
            vid = os.path.splitext(vid)[0]
            
        if vid.endswith('_aligned'): vid = vid[:-8]
        
        # Construct feature path
        fpath = os.path.join(feature_root, f"{vid}{feature_suffix}")
        if not os.path.exists(fpath):
            # Try simple fallback if file not found
            fpath = os.path.join(feature_root, vid + ".npy")
            if not os.path.exists(fpath):
                continue
            
        try:
            arr = np.load(fpath)
            # Aggregate: if (N, T, D) or (T, D), average over time to get (D,) vector
            if arr.ndim == 2:
                feat = np.mean(arr, axis=0)
            elif arr.ndim == 3: # (N, T, D) -> mean over N, T
                feat = np.mean(arr, axis=(0, 1))
            else:
                feat = arr.flatten()
                
            features.append(feat)
            labels.append(row[label_col])
            count += 1
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    if not features:
        print("No features loaded. Check paths.")
        return

    X = np.array(features)
    y = np.array(labels)
    
    print(f"Running t-SNE on {X.shape}...")
    # Standardize
    X_scaled = StandardScaler().fit_transform(X)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(X)-1)))
    X_embedded = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='coolwarm', alpha=0.7, s=40)
    plt.colorbar(scatter, label='Pain Level')
    plt.title(f't-SNE of Raw Features (avg pooling)\n{os.path.basename(feature_root)}')
    
    out_path = os.path.join(output_dir, 'tsne_features.png')
    plt.savefig(out_path, dpi=300)
    print(f"t-SNE plot saved to: {out_path}")

def analyze_model_features(meta_path, feature_root, ckpt_path, output_dir, limit=500, feature_suffix='_windows.npy', strategy='mean', color_by='pain', feature_prefix=''):
    """
    Load trained model and extract features (after aggregation) for t-SNE.
    """
    print(f"Loading model from {ckpt_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = MILCoralTransformer.load_from_checkpoint(ckpt_path)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Loading features from {feature_root}...")
    
    # Read metadata
    ext = os.path.splitext(meta_path)[-1].lower()
    if ext in ['.xls', '.xlsx']:
        df = pd.read_excel(meta_path)
    else:
        df = pd.read_csv(meta_path)
    
    df.columns = [c.lower() for c in df.columns]
    
    # Heuristic for video column
    video_col = None
    possible_video_cols = ['video_id', 'file_name', 'filename', 'video_name', 'video'] # Added 'video'
    for c in possible_video_cols:
        if c in df.columns:
            video_col = c
            break
    if video_col is None: video_col = df.columns[0]
        
    # Heuristic for label column
    label_col = 'pain_level'
    possible_label_cols = ['pain_level', 'pain', 'vas', 'opr']
    for c in possible_label_cols:
        if c in df.columns:
            label_col = c
            break

    subject_col = 'subject_id' if 'subject_id' in df.columns else 'subject'

    features = []
    labels = []
    subjects = []
    
    count = 0
    with torch.no_grad():
        for _, row in df.iterrows():
            if count >= limit: break
            
            vid = str(row[video_col]).strip()
            if "." in vid: vid = os.path.splitext(vid)[0]
            if vid.endswith('_aligned'): vid = vid[:-8]
            
            # Construct filename with prefix and suffix
            fpath = os.path.join(feature_root, f"{feature_prefix}{vid}{feature_suffix}")
            if not os.path.exists(fpath):
                # Try simple fallback if file not found
                if not os.path.exists(fpath):
                    continue
                
            try:
                arr = np.load(fpath)
                if arr.ndim == 1: arr = arr[np.newaxis, :]
                elif arr.ndim == 3: 
                    # If (N, T, D), flatten to (N, T*D) if model expects it, 
                    # BUT wait, the model aggregator usually expects (B, T, D) or (B, N, D).
                    # MILCoralTransformer expects (B, T, D).
                    # Let's assume input is (N_clips, T_frames, D_dim).
                    # We treat the whole video as one bag (1, Total_Frames, D).
                    N, T, D = arr.shape
                    arr = arr.reshape(1, N * T, D)
                elif arr.ndim == 2:
                    # (T, D) -> (1, T, D)
                    arr = arr[np.newaxis, :]
                    
                x = torch.from_numpy(arr).float().to(device)
                
                # Forward pass to get bag embedding
                # We need to access the aggregator output directly.
                # model.forward() returns dict with 'bag' key.
                out = model(x)
                bag_feat = out['bag'].cpu().numpy() # (1, embed_dim)
                
                features.append(bag_feat.flatten())
                labels.append(row[label_col])
                
                # Subject handling
                if subject_col in df.columns:
                    subjects.append(str(row[subject_col]))
                else:
                    subjects.append("unknown")
                    
                count += 1
            except Exception as e:
                print(f"Error processing {fpath}: {e}")

    if not features:
        print("No features extracted.")
        return

    X = np.array(features)
    y_pain = np.array(labels)
    y_subj = np.array(subjects)
    
    print(f"Running t-SNE on model embeddings {X.shape}...")
    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(X)-1)))
    X_embedded = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    
    if color_by == 'pain':
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_pain, cmap='coolwarm', alpha=0.7, s=40)
        plt.colorbar(scatter, label='Pain Level')
        title_suffix = "(Colored by Pain)"
    else:
        # Map subjects to integers for coloring
        unique_subs = np.unique(y_subj)
        sub_map = {s: i for i, s in enumerate(unique_subs)}
        c_vals = [sub_map[s] for s in y_subj]
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=c_vals, cmap='tab20', alpha=0.7, s=40)
        # plt.colorbar(scatter, label='Subject ID') # Discrete categorical usually better with legend, but colorbar ok for quick check
        title_suffix = "(Colored by Subject)"
        
    plt.title(f't-SNE of Trained Model Embeddings\n{os.path.basename(ckpt_path)}\n{title_suffix}')
    
    out_path = os.path.join(output_dir, f'tsne_model_{color_by}.png')
    plt.savefig(out_path, dpi=300)
    print(f"Model t-SNE plot saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Experimental Results & Features")
    subparsers = parser.add_subparsers(dest='command', help='Mode')

    # Mode 1: Compare Metrics
    parser_cmp = subparsers.add_parser('compare', help='Compare metrics from multiple CSVs')
    parser_cmp.add_argument('--files', nargs='+', required=True, help='List of summary CSV files')
    parser_cmp.add_argument('--labels', nargs='+', required=True, help='List of labels for the files')
    parser_cmp.add_argument('--out', default='.', help='Output directory')

    # Mode 2: Confusion Matrix
    parser_conf = subparsers.add_parser('confusion', help='Plot confusion matrix heatmap')
    parser_conf.add_argument('--file', required=True, help='Confusion matrix CSV')
    parser_conf.add_argument('--out', default='.', help='Output directory')

    # Mode 4: Model Features Analysis
    parser_model = subparsers.add_parser('model_features', help='Analyze features using a trained model')
    parser_model.add_argument('--meta', required=True, help='Path to metadata Excel/CSV')
    parser_model.add_argument('--root', required=True, help='Directory containing .npy features')
    parser_model.add_argument('--ckpt', required=True, help='Path to model checkpoint (.ckpt)')
    parser_model.add_argument('--suffix', default='_windows.npy', help='Feature file suffix')
    parser_model.add_argument('--prefix', default='', help='Feature file prefix (e.g. IMG_)')
    parser_model.add_argument('--limit', type=int, default=1000, help='Max samples to plot')
    parser_model.add_argument('--out', default='.', help='Output directory')
    parser_model.add_argument('--strategy', default='mean', choices=['mean', 'patch'], help='Aggregation strategy: mean (per video) or patch (sample patches)')
    parser_model.add_argument('--color_by', default='pain', choices=['pain', 'subject'], help='Color by pain level or subject ID')

    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)

    if args.command == 'compare':
        if len(args.files) != len(args.labels):
            print("Error: Number of files must match number of labels.")
            return
        plot_comparison(args.files, args.labels, args.out)
    
    elif args.command == 'confusion':
        plot_confusion(args.file, args.out)
        
    elif args.command == 'features':
        analyze_features_tsne(args.meta, args.root, args.out, args.limit, args.suffix)
    
    elif args.command == 'model_features':
        analyze_model_features(args.meta, args.root, args.ckpt, args.out, args.limit, args.suffix, args.strategy, args.color_by, args.prefix)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
