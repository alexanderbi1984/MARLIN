import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
from pygam import LinearGAM, s
import os
from typing import Dict, Tuple
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MarlinFeatures.syracuse_dataset import SyracuseDataset

class PSPIPredictor(SyracuseDataset):
    def __init__(self, meta_path: str, feature_dir: str, au_features_dir: str):
        """Initialize the predictor with AU features and pain data."""
        super().__init__(meta_path, feature_dir)
        self.au_features_dir = au_features_dir
        self.au_features = self._load_au_features()
        self.scaler = StandardScaler()
        
    def _load_au_features(self) -> dict:
        """Load all processed AU feature files."""
        features = {}
        for file in os.listdir(self.au_features_dir):
            if file.startswith('processed_') and file.endswith('.csv'):
                # Extract IMG_xxxx from filename (e.g., processed_IMG_0003.csv -> IMG_0003)
                video_id = file.split('_')[1] + '_' + file.split('_')[2].split('.')[0]
                df = pd.read_csv(os.path.join(self.au_features_dir, file))
                features[video_id] = df
        return features
    
    def calculate_pspi(self, df: pd.DataFrame) -> float:
        """Calculate PSPI score from AU features (modified version without AU43)."""
        # Get required AU features
        au4 = df['AU04_r']
        au6 = df['AU06_r']
        au7 = df['AU07_r']
        au9 = df['AU09_r']
        au10 = df['AU10_r']
        
        # Calculate PSPI components (without AU43)
        pspi = au4 + np.maximum(au6, au7) + np.maximum(au9, au10)
        
        # Return mean PSPI score across time
        return float(np.mean(pspi))
    
    def prepare_features(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and targets for all three prediction tasks."""
        feature_diffs = []
        binary_outcomes = []
        abs_reductions = []
        percent_reductions = []
        
        for pair in self.pairs:
            pre_file = pair['pre_file'].split('.')[0]  # e.g., IMG_0003
            post_file = pair['post_file'].split('.')[0]  # e.g., IMG_0004
            
            if pre_file not in self.au_features or post_file not in self.au_features:
                print(f"Skipping pair: {pre_file} or {post_file} not found in AU features")
                continue
            
            # Calculate PSPI scores
            pre_df = self.au_features[pre_file]
            post_df = self.au_features[post_file]
            
            pre_pspi = self.calculate_pspi(pre_df)
            post_pspi = self.calculate_pspi(post_df)
            
            # Calculate PSPI difference
            feature_diff = post_pspi - pre_pspi
            
            # Store targets
            binary_outcomes.append(1 if pair['change'] >= 4 else 0)
            abs_reductions.append(pair['change'])
            percent_reductions.append(pair['change'] / pair['pre_pain'] * 100)
            
            feature_diffs.append(feature_diff)
        
        # Create DataFrame before scaling
        X = pd.DataFrame(feature_diffs, columns=['PSPI_diff'])
        y_binary = np.array(binary_outcomes)
        y_abs = np.array(abs_reductions)
        y_percent = np.array(percent_reductions)
        
        # Scale features after taking differences
        X_scaled = self.scaler.fit_transform(X)
        
        print("\nPSPI Feature Statistics:")
        print(f"Mean PSPI difference: {np.mean(feature_diffs):.3f}")
        print(f"Std PSPI difference: {np.std(feature_diffs):.3f}")
        print(f"Range PSPI difference: [{np.min(feature_diffs):.3f}, {np.max(feature_diffs):.3f}]")
        
        return X_scaled, y_binary, y_abs, y_percent
    
    def train_models(self, X: np.ndarray, y_binary: np.ndarray, y_abs: np.ndarray, y_percent: np.ndarray):
        """Train and evaluate models for all three prediction tasks using 3-fold CV."""
        # 1. Logistic Regression for binary outcome
        print("\n1. Logistic Regression for Binary Outcome:")
        log_reg = LogisticRegression(max_iter=1000)
        
        # Cross-validation for accuracy
        cv_scores = cross_val_score(log_reg, X, y_binary, cv=3)
        print(f"Cross-validation accuracy scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Cross-validation for AUC
        cv_auc_scores = cross_val_score(log_reg, X, y_binary, cv=3, scoring='roc_auc')
        print(f"Cross-validation AUC scores: {cv_auc_scores}")
        print(f"Mean CV AUC: {cv_auc_scores.mean():.3f} (+/- {cv_auc_scores.std() * 2:.3f})")
        
        # Fit on all data
        log_reg.fit(X, y_binary)
        
        # 2. Linear Regression for absolute pain reduction
        print("\n2. Linear Regression for Absolute Pain Reduction:")
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'GAM_4splines': LinearGAM(s(0), n_splines=4),
            'GAM_8splines': LinearGAM(s(0), n_splines=8),
            'GAM_4splines_lam0.1': LinearGAM(s(0), n_splines=4, lam=0.1),
            'GAM_4splines_lam10': LinearGAM(s(0), n_splines=4, lam=10)
        }
        
        for name, model in models.items():
            print(f"\n{name}:")
            if 'GAM' in name:
                # GAM models need special handling for cross-validation
                cv_scores = []
                cv_mse_scores = []
                for train_idx, test_idx in KFold(n_splits=3, shuffle=True, random_state=42).split(X):
                    model.fit(X[train_idx], y_abs[train_idx])
                    score = model.score(X[test_idx], y_abs[test_idx])
                    y_pred = model.predict(X[test_idx])
                    mse = mean_squared_error(y_abs[test_idx], y_pred)
                    cv_scores.append(score)
                    cv_mse_scores.append(mse)
                cv_scores = np.array(cv_scores)
                cv_mse_scores = np.array(cv_mse_scores)
            else:
                cv_scores = cross_val_score(model, X, y_abs, cv=3, scoring='r2')
                cv_mse_scores = -cross_val_score(model, X, y_abs, cv=3, scoring='neg_mean_squared_error')
            
            print(f"Cross-validation R² scores: {cv_scores}")
            print(f"Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"Cross-validation MSE scores: {cv_mse_scores}")
            print(f"Mean CV MSE: {cv_mse_scores.mean():.3f} (+/- {cv_mse_scores.std() * 2:.3f})")
            print(f"RMSE: {np.sqrt(cv_mse_scores.mean()):.3f}")
            
            # Fit on all data
            model.fit(X, y_abs)
        
        # 3. Linear Regression for percentage pain reduction
        print("\n3. Linear Regression for Percentage Pain Reduction:")
        for name, model in models.items():
            print(f"\n{name}:")
            if 'GAM' in name:
                # GAM models need special handling for cross-validation
                cv_scores = []
                cv_mse_scores = []
                for train_idx, test_idx in KFold(n_splits=3, shuffle=True, random_state=42).split(X):
                    model.fit(X[train_idx], y_percent[train_idx])
                    score = model.score(X[test_idx], y_percent[test_idx])
                    y_pred = model.predict(X[test_idx])
                    mse = mean_squared_error(y_percent[test_idx], y_pred)
                    cv_scores.append(score)
                    cv_mse_scores.append(mse)
                cv_scores = np.array(cv_scores)
                cv_mse_scores = np.array(cv_mse_scores)
            else:
                cv_scores = cross_val_score(model, X, y_percent, cv=3, scoring='r2')
                cv_mse_scores = -cross_val_score(model, X, y_percent, cv=3, scoring='neg_mean_squared_error')
            
            print(f"Cross-validation R² scores: {cv_scores}")
            print(f"Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"Cross-validation MSE scores: {cv_mse_scores}")
            print(f"Mean CV MSE: {cv_mse_scores.mean():.3f} (+/- {cv_mse_scores.std() * 2:.3f})")
            print(f"RMSE: {np.sqrt(cv_mse_scores.mean()):.3f}")
            
            # Fit on all data
            model.fit(X, y_percent)
        
        return {
            'logistic_regression': log_reg,
            'regression_models': models
        }

def main():
    # Set paths
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    au_features_dir = '/Users/hd927/Documents/syracuse_pain_research/AUFeatures/processed'
    
    # Initialize predictor
    predictor = PSPIPredictor(meta_path, feature_dir, au_features_dir)
    
    # Prepare features and targets
    X, y_binary, y_abs, y_percent = predictor.prepare_features()
    
    # Train and evaluate models
    models = predictor.train_models(X, y_binary, y_abs, y_percent)

if __name__ == "__main__":
    main() 