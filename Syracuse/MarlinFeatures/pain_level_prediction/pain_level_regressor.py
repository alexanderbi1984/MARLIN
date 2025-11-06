import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import os
from pathlib import Path
import argparse

def extract_features_and_pain_levels(meta_path, feature_dir):
    """
    Extract features and corresponding pain levels from all available videos.
    
    Args:
        meta_path: Path to the meta data file
        feature_dir: Directory containing feature files
        
    Returns:
        Dictionary containing dataset information
    """
    print("Loading metadata...")
    # Load metadata directly
    meta_df = pd.read_excel(meta_path)
    
    # Convert pain_level to numeric, handling non-numeric values
    meta_df['pain_level'] = pd.to_numeric(meta_df['pain_level'], errors='coerce')
    
    # Filter out rows with missing pain levels
    valid_df = meta_df.dropna(subset=['pain_level'])
    print(f"Found {len(valid_df)} videos with valid pain levels out of {len(meta_df)} total videos")
    
    # Collect data from all videos with valid pain levels
    all_features = []
    all_pain_levels = []
    all_file_names = []
    
    for idx, row in valid_df.iterrows():
        file_name = row['file_name']
        pain_level = row['pain_level']
        
        # Get clips for this video
        try:
            clips = sorted([f for f in os.listdir(feature_dir) 
                            if f.startswith(file_name.replace('.MP4', '_clip_')) and f.endswith('_aligned.npy')])[:14]
        except:
            print(f"Warning: Could not find clips for {file_name}, skipping")
            continue
        
        if len(clips) < 14:
            print(f"Warning: Not enough clips for {file_name}, found {len(clips)}, skipping")
            continue
        
        # Load and process video features
        video_features = []
        for clip in clips:
            clip_path = os.path.join(feature_dir, clip)
            try:
                features = np.load(clip_path)
            except:
                print(f"Warning: Could not load clip {clip}, skipping")
                continue
                
            # Check feature dimensions
            if features.shape[1] != 768:
                print(f"Warning: Clip {clip} has unexpected feature dimension {features.shape[1]}, skipping")
                continue
                
            # Normalize to 4 frames if needed
            if features.shape[0] != 4:
                n_frames = features.shape[0]
                if features.shape[0] > 4:
                    indices = np.linspace(0, n_frames-1, 4, dtype=int)
                    features = features[indices]
                else:
                    indices = np.linspace(0, n_frames-1, 4)
                    interpolated_features = np.zeros((4, features.shape[1]))
                    for j in range(features.shape[1]):
                        interpolated_features[:, j] = np.interp(indices, np.arange(n_frames), features[:, j])
                    features = interpolated_features
            
            video_features.append(features)
        
        if len(video_features) < 14:
            print(f"Warning: Not enough valid clips for {file_name}, found {len(video_features)}, skipping")
            continue
            
        # Stack and average across time and clips
        video_features = np.stack(video_features)  # (14, 4, 768)
        features_avg = np.mean(video_features, axis=(0, 1))  # (768,)
        
        all_features.append(features_avg)
        all_pain_levels.append(pain_level)
        all_file_names.append(file_name)
    
    # Convert to arrays
    features = np.array(all_features)
    pain_levels = np.array(all_pain_levels)
    
    return {
        'features': features,
        'pain_levels': pain_levels,
        'file_names': all_file_names
    }

def select_features_by_effect_size(features, pain_levels, n_features=5):
    """
    Select features based on effect size (Cohen's d) for regression.
    We calculate effect size by comparing high and low pain groups.
    
    Args:
        features: numpy array of shape (n_samples, n_features)
        pain_levels: numpy array of pain levels
        n_features: number of features to select
    
    Returns:
        selected_features: numpy array of selected features
        selected_indices: indices of selected features
        effect_sizes: effect sizes of all features
    """
    n_samples, n_features_total = features.shape
    effect_sizes = np.zeros(n_features_total)
    
    # Split pain levels into high and low groups using median
    pain_median = np.median(pain_levels)
    high_pain_mask = pain_levels >= pain_median
    low_pain_mask = pain_levels < pain_median
    
    # Calculate Cohen's d for each feature
    for i in range(n_features_total):
        # Split features by pain group
        feature_values = features[:, i]
        low_pain_values = feature_values[low_pain_mask]
        high_pain_values = feature_values[high_pain_mask]
        
        # Calculate means and standard deviations
        mean_low = np.mean(low_pain_values)
        mean_high = np.mean(high_pain_values)
        std_low = np.std(low_pain_values, ddof=1)  # ddof=1 for sample standard deviation
        std_high = np.std(high_pain_values, ddof=1)
        
        # Calculate pooled standard deviation
        n_low = len(low_pain_values)
        n_high = len(high_pain_values)
        pooled_std = np.sqrt(((n_low - 1) * std_low**2 + (n_high - 1) * std_high**2) / (n_low + n_high - 2))
        
        # Calculate Cohen's d
        if pooled_std == 0:
            effect_sizes[i] = 0
        else:
            effect_sizes[i] = abs(mean_high - mean_low) / pooled_std
    
    # Select top features
    selected_indices = np.argsort(effect_sizes)[-n_features:]
    selected_features = features[:, selected_indices]
    
    # Print effect sizes of selected features
    print(f"\nSelected features by effect size (Cohen's d):")
    for i, idx in enumerate(selected_indices):
        print(f"Feature {idx}: {effect_sizes[idx]:.3f}")
    
    return selected_features, selected_indices, effect_sizes

def train_and_evaluate_models(features, pain_levels, test_size=0.2, random_state=42):
    """
    Train and evaluate multiple regression models.
    
    Args:
        features: numpy array of selected features
        pain_levels: numpy array of pain levels
        test_size: proportion of data to use for testing
        random_state: random seed for reproducibility
    
    Returns:
        Dictionary containing trained models and their performance metrics
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, pain_levels, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'SVR': SVR(),
        'RandomForest': RandomForestRegressor(random_state=random_state),
        'GradientBoosting': GradientBoostingRegressor(random_state=random_state)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='r2')
        
        results[name] = {
            'model': model,
            'scaler': scaler,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"{name} Performance:")
        print(f"Training R²: {train_r2:.3f}")
        print(f"Testing R²: {test_r2:.3f}")
        print(f"Training MSE: {train_mse:.3f}")
        print(f"Testing MSE: {test_mse:.3f}")
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return results

def plot_results(results, output_dir, X_test, y_test):
    """
    Generate plots for model comparison and performance visualization.
    
    Args:
        results: Dictionary containing model results
        output_dir: Directory to save plots
        X_test: Test features
        y_test: Test pain levels
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot R² scores comparison
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    train_r2 = [results[m]['train_r2'] for m in models]
    test_r2 = [results[m]['test_r2'] for m in models]
    cv_r2 = [results[m]['cv_mean'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.bar(x - width, train_r2, width, label='Training R²')
    plt.bar(x, test_r2, width, label='Testing R²')
    plt.bar(x + width, cv_r2, width, label='CV R²')
    
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()
    
    # Plot predictions vs actual for best model
    best_model = max(results.items(), key=lambda x: x[1]['test_r2'])[0]
    model = results[best_model]['model']
    scaler = results[best_model]['scaler']
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Pain Level')
    plt.ylabel('Predicted Pain Level')
    plt.title(f'Predictions vs Actual ({best_model})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_actual.png'))
    plt.close()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train and evaluate pain level regression models')
    parser.add_argument('--n_features', type=int, default=5,
                        help='Number of features to select by effect size')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path('Syracuse/pain_level_prediction/regression_results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract features and pain levels
    print("Extracting features and pain levels...")
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    data = extract_features_and_pain_levels(meta_path, feature_dir)
    
    features = data['features']
    pain_levels = data['pain_levels']
    
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(pain_levels)}")
    print(f"Pain level range: {pain_levels.min():.1f} - {pain_levels.max():.1f}")
    print(f"Mean pain level: {pain_levels.mean():.2f}")
    print(f"Median pain level: {np.median(pain_levels):.2f}")
    print(f"Standard deviation: {pain_levels.std():.2f}")
    
    # Select features based on effect size
    print(f"\nSelecting {args.n_features} features based on effect size...")
    selected_features, selected_indices, effect_sizes = select_features_by_effect_size(
        features, pain_levels, n_features=args.n_features
    )
    
    # Split the data for plotting
    X_train, X_test, y_train, y_test = train_test_split(
        selected_features, pain_levels, test_size=0.2, random_state=42
    )
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(selected_features, pain_levels)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(results, output_dir, X_test, y_test)
    
    # Save best model and preprocessing objects
    best_model = max(results.items(), key=lambda x: x[1]['test_r2'])[0]
    print(f"\nBest performing model: {best_model}")
    
    print("\nSaving model and preprocessing objects...")
    np.save(os.path.join(output_dir, 'model.npy'), results[best_model]['model'])
    np.save(os.path.join(output_dir, 'scaler.npy'), results[best_model]['scaler'])
    np.save(os.path.join(output_dir, 'selected_indices.npy'), selected_indices)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == '__main__':
    main() 