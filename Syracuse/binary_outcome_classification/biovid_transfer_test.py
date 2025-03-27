import json
import numpy as np
import logging
import os
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
DATA_DIR = Path('Syracuse/binary_outcome_classification/processed_data')
MODEL_DIR = Path('Syracuse/binary_outcome_classification/classification_results')
OUTPUT_DIR = Path('Syracuse/binary_outcome_classification/transfer_results')

# Define the 5 selected features
SELECTED_FEATURES = [662, 316, 587, 629, 143]

def load_preprocessed_data():
    """Load preprocessed feature differences and metadata."""
    # Load metadata
    meta_path = DATA_DIR / 'pairs_metadata.json'
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    # Load feature differences
    diff_path = DATA_DIR / 'feature_diffs.npy'
    feature_diffs = np.load(diff_path, allow_pickle=True).item()
    
    # Initialize lists for positive and negative samples
    X_pos = []
    X_neg = []
    
    # Process all pairs
    for pair_id, diff in feature_diffs.items():
        pair_meta = metadata[pair_id]
        pain_diff = pair_meta['difference']
        pre_pain = pair_meta['pre_ground_truth']
        
        if pain_diff >= 3:  # Positive outcome: difference >= 3
            # Select only the 5 features used during training
            selected_diff = diff[SELECTED_FEATURES]
            X_pos.append(selected_diff)
        elif pre_pain >= 3 and pain_diff <= 2:  # Negative outcome: pre_pain >= 3 and difference <= 2
            # Select only the 5 features used during training
            selected_diff = diff[SELECTED_FEATURES]
            X_neg.append(selected_diff)
    
    # Convert to numpy arrays
    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)
    
    # Log data loading summary
    logging.info(f"\nData Loading Summary:")
    logging.info(f"Positive samples (diff >= 3): {len(X_pos)}")
    logging.info(f"Negative samples (pre >= 3, diff <= 2): {len(X_neg)}")
    logging.info(f"Selected features: {SELECTED_FEATURES}")
    
    return X_pos, X_neg

def main():
    """Main function to run transfer learning test."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load preprocessed data
    logging.info("Loading preprocessed data...")
    X_pos, X_neg = load_preprocessed_data()
    
    # Combine positive and negative samples
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))
    
    # Standardize features
    logging.info("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Load pre-trained model
    logging.info("Loading pre-trained model...")
    model_path = MODEL_DIR / 'model.joblib'
    model = joblib.load(model_path)
    
    # Evaluate model
    logging.info("\nEvaluating model on Biovid dataset:")
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y)
    pos_pred_rate = np.mean(y_pred[y == 1])
    neg_pred_rate = np.mean(y_pred[y == 0])
    
    logging.info(f"Overall Accuracy: {accuracy:.3f}")
    logging.info(f"Positive prediction rate: {pos_pred_rate:.3f}")
    logging.info(f"Negative prediction rate: {neg_pred_rate:.3f}")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'positive_prediction_rate': float(pos_pred_rate),
        'negative_prediction_rate': float(neg_pred_rate),
        'n_samples': len(X),
        'n_positive': len(X_pos),
        'n_negative': len(X_neg),
        'selected_features': SELECTED_FEATURES
    }
    
    results_path = OUTPUT_DIR / 'transfer_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    main() 