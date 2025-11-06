import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
from Syracuse.MarlinFeatures.syracuse_dataset import SyracuseDataset
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict

class PainLevelClassification:
    def __init__(self):
        self.dataset = SyracuseDataset(
            meta_path='/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx',
            feature_dir='/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
        )
        self.meta_df = self.dataset.meta_df
        self.output_dir = os.path.dirname(os.path.abspath(__file__))
        
    def assign_classes(self, pain_level: float) -> Dict[str, int]:
        """
        Assign class labels for a given pain level according to all three classification schemes.
        
        Args:
            pain_level: The pain level value
            
        Returns:
            Dictionary containing class assignments for 3, 4, and 5 class problems
        """
        # 3-Class Problem
        if 0 <= pain_level <= 2:
            class_3 = 0  # Low
        elif 2 < pain_level <= 5:
            class_3 = 1  # Medium
        else:
            class_3 = 2  # High
            
        # 4-Class Problem
        if 0 <= pain_level <= 1:
            class_4 = 0
        elif 1 < pain_level <= 3:
            class_4 = 1
        elif 3 < pain_level <= 6:
            class_4 = 2
        else:
            class_4 = 3
            
        # 5-Class Problem
        if 0 <= pain_level <= 1:
            class_5 = 0
        elif 1 < pain_level <= 3:
            class_5 = 1
        elif 3 < pain_level <= 5:
            class_5 = 2
        elif 5 < pain_level <= 7:
            class_5 = 3
        else:
            class_5 = 4
            
        return {
            '3_class': class_3,
            '4_class': class_4,
            '5_class': class_5
        }
    
    def prepare_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare features and labels for all classification problems.
        
        Returns:
            Tuple containing:
            - features: numpy array of shape (n_samples, n_features)
            - labels: dictionary containing labels for 3, 4, and 5 class problems
        """
        # Get all valid videos with pain levels
        valid_data = []
        features_list = []
        
        # Add class columns to meta_df
        self.meta_df['class_3'] = np.nan
        self.meta_df['class_4'] = np.nan
        self.meta_df['class_5'] = np.nan
        
        # Process each video in the meta data
        for idx, row in self.meta_df.iterrows():
            if pd.notna(row['pain_level']):
                # Get the video file name
                video_name = row['file_name'].replace('.MP4', '')
                
                # Get all clips for this video
                clips = sorted([f for f in os.listdir(self.dataset.feature_dir) 
                              if f.startswith(f"{video_name}_clip_") and f.endswith('_aligned.npy')])[:14]
                
                if len(clips) == 14:  # Only use videos with all 14 clips
                    # Load and process features
                    video_features = []
                    for clip in clips:
                        features = np.load(os.path.join(self.dataset.feature_dir, clip))
                        # Ensure 4 temporal frames
                        if features.shape[0] != 4:
                            if features.shape[0] > 4:
                                indices = np.linspace(0, features.shape[0]-1, 4, dtype=int)
                                features = features[indices]
                            else:
                                # Interpolate if less than 4 frames
                                temp_features = np.zeros((4, features.shape[1]))
                                for j in range(features.shape[1]):
                                    temp_features[:, j] = np.interp(
                                        np.linspace(0, features.shape[0]-1, 4),
                                        np.arange(features.shape[0]),
                                        features[:, j]
                                    )
                                features = temp_features
                        
                        video_features.append(features)
                    
                    # Stack all clips' features
                    video_features = np.stack(video_features)  # Shape: (14, 4, 768)
                    
                    # Flatten the features
                    flat_features = video_features.reshape(-1)
                    
                    features_list.append(flat_features)
                    
                    # Assign classes and save to meta_df
                    classes = self.assign_classes(row['pain_level'])
                    self.meta_df.at[idx, 'class_3'] = classes['3_class']
                    self.meta_df.at[idx, 'class_4'] = classes['4_class']
                    self.meta_df.at[idx, 'class_5'] = classes['5_class']
                    
                    valid_data.append({
                        'pain_level': row['pain_level'],
                        'classes': classes
                    })
        
        # Save updated meta_df to Excel
        output_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes_and_classes.xlsx'
        self.meta_df.to_excel(output_path, index=False)
        print(f"\nSaved updated meta data to: {output_path}")
        
        # Convert to numpy arrays
        X = np.stack(features_list)
        y_3class = np.array([d['classes']['3_class'] for d in valid_data])
        y_4class = np.array([d['classes']['4_class'] for d in valid_data])
        y_5class = np.array([d['classes']['5_class'] for d in valid_data])
        
        # Print detailed statistics
        print("\n=== Classification Data Statistics ===")
        print(f"Total number of samples: {len(X)}")
        print(f"Feature dimension: {X.shape[1]}")
        
        print("\n3-Class Distribution:")
        for i in range(3):
            count = np.sum(y_3class == i)
            percentage = (count / len(y_3class)) * 100
            print(f"Class {i}: {count} samples ({percentage:.1f}%)")
            
        print("\n4-Class Distribution:")
        for i in range(4):
            count = np.sum(y_4class == i)
            percentage = (count / len(y_4class)) * 100
            print(f"Class {i}: {count} samples ({percentage:.1f}%)")
            
        print("\n5-Class Distribution:")
        for i in range(5):
            count = np.sum(y_5class == i)
            percentage = (count / len(y_5class)) * 100
            print(f"Class {i}: {count} samples ({percentage:.1f}%)")
        
        # Print pain level ranges for each class
        print("\n=== Pain Level Ranges for Each Class ===")
        
        print("\n3-Class Pain Level Ranges:")
        for i in range(3):
            mask = y_3class == i
            pain_levels = [d['pain_level'] for d in valid_data if d['classes']['3_class'] == i]
            print(f"Class {i}: {min(pain_levels):.1f} - {max(pain_levels):.1f}")
            
        print("\n4-Class Pain Level Ranges:")
        for i in range(4):
            mask = y_4class == i
            pain_levels = [d['pain_level'] for d in valid_data if d['classes']['4_class'] == i]
            print(f"Class {i}: {min(pain_levels):.1f} - {max(pain_levels):.1f}")
            
        print("\n5-Class Pain Level Ranges:")
        for i in range(5):
            mask = y_5class == i
            pain_levels = [d['pain_level'] for d in valid_data if d['classes']['5_class'] == i]
            print(f"Class {i}: {min(pain_levels):.1f} - {max(pain_levels):.1f}")
        
        return X, {
            '3_class': y_3class,
            '4_class': y_4class,
            '5_class': y_5class
        }

if __name__ == "__main__":
    # Initialize and prepare data
    pain_clf = PainLevelClassification()
    X, y_dict = pain_clf.prepare_data()
    
    # Plot class distributions
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 3-Class distribution
    sns.countplot(x=y_dict['3_class'], ax=ax1)
    ax1.set_title('3-Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    
    # 4-Class distribution
    sns.countplot(x=y_dict['4_class'], ax=ax2)
    ax2.set_title('4-Class Distribution')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    
    # 5-Class distribution
    sns.countplot(x=y_dict['5_class'], ax=ax3)
    ax3.set_title('5-Class Distribution')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Count')
    
    plt.tight_layout()
    plot_path = os.path.join(pain_clf.output_dir, 'class_distributions.png')
    plt.savefig(plot_path)
    print(f"\nSaved class distribution plots to: {plot_path}")
    plt.close() 