import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Syracuse.MarlinFeatures.syracuse_dataset import SyracuseDataset

class MicroExprDataset(SyracuseDataset):
    def __init__(self, meta_path: str, feature_dir: str, micro_expr_dir: str):
        """
        Initialize the micro-expression dataset.
        
        Args:
            meta_path: Path to the meta_with_outcomes.xlsx file
            feature_dir: Directory containing the MARLIN feature files
            micro_expr_dir: Directory containing the micro-expression feature files
        """
        super().__init__(meta_path, feature_dir)
        self.micro_expr_dir = micro_expr_dir
        
        # Load micro-expression features for all pairs
        self.micro_expr_features = self._load_micro_expr_features()
        
    def _load_micro_expr_features(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load micro-expression features for all pairs.
        
        Returns:
            Dictionary mapping file names to their micro-expression features
        """
        features = {}
        for pair in self.pairs:
            # Load pre features
            pre_file = pair['pre_file'].replace('.MP4', '.csv')
            pre_path = os.path.join(self.micro_expr_dir, pre_file)
            if os.path.exists(pre_path):
                pre_df = pd.read_csv(pre_path)
                # Get logit columns
                logit_cols = [col for col in pre_df.columns if col.startswith('logit_')]
                # Average across frames
                pre_features = pre_df[logit_cols].mean().values
                features[pre_file] = pre_features
                
            # Load post features
            post_file = pair['post_file'].replace('.MP4', '.csv')
            post_path = os.path.join(self.micro_expr_dir, post_file)
            if os.path.exists(post_path):
                post_df = pd.read_csv(post_path)
                # Get logit columns
                logit_cols = [col for col in post_df.columns if col.startswith('logit_')]
                # Average across frames
                post_features = post_df[logit_cols].mean().values
                features[post_file] = post_features
                
        return features
    
    def get_micro_expr_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get micro-expression features for all pairs.
        
        Returns:
            Tuple of (pre_features, post_features, changes) arrays
        """
        pre_features = []
        post_features = []
        changes = []
        
        for pair in self.pairs:
            pre_file = pair['pre_file'].replace('.MP4', '.csv')
            post_file = pair['post_file'].replace('.MP4', '.csv')
            
            if pre_file in self.micro_expr_features and post_file in self.micro_expr_features:
                pre_features.append(self.micro_expr_features[pre_file])
                post_features.append(self.micro_expr_features[post_file])
                changes.append(pair['change'])
        
        return (np.stack(pre_features), 
                np.stack(post_features), 
                np.array(changes))
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of the micro-expression features.
        
        Returns:
            List of feature names
        """
        # Get the first file to extract feature names
        first_file = next(iter(self.micro_expr_features.keys()))
        first_df = pd.read_csv(os.path.join(self.micro_expr_dir, first_file))
        return [col for col in first_df.columns if col.startswith('logit_')]
    
    def get_pain_levels(self) -> np.ndarray:
        """Get pre-treatment pain levels for all samples."""
        return np.array([pair['pre_pain'] for pair in self.pairs])

    def _load_all_micro_expr_features(self) -> Dict[str, np.ndarray]:
        """
        Load micro-expression features for all videos with valid pain levels.
        
        Returns:
            Dictionary mapping file names to their micro-expression features
        """
        features = {}
        for _, video in self.meta_df.iterrows():  # Use meta_df instead of videos
            if pd.notna(video['pain_level']):  # Only include videos with valid pain levels
                file = video['file_name'].replace('.MP4', '.csv')
                file_path = os.path.join(self.micro_expr_dir, file)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    # Get logit columns
                    logit_cols = [col for col in df.columns if col.startswith('logit_')]
                    # Average across frames
                    video_features = df[logit_cols].mean().values
                    features[file] = video_features
        return features

    def get_all_micro_expr_features(self) -> np.ndarray:
        """
        Get micro-expression features for all videos with valid pain levels.
        
        Returns:
            Array of features for all videos with valid pain levels
        """
        if not hasattr(self, 'all_micro_expr_features'):
            self.all_micro_expr_features = self._load_all_micro_expr_features()
        
        return np.stack(list(self.all_micro_expr_features.values()))

    def get_all_pain_levels(self) -> np.ndarray:
        """
        Get pain levels for all videos with valid pain levels.
        
        Returns:
            Array of pain levels for all videos with valid pain levels
        """
        pain_levels = []
        for _, video in self.meta_df.iterrows():  # Use meta_df instead of videos
            if pd.notna(video['pain_level']):  # Only include videos with valid pain levels
                file = video['file_name'].replace('.MP4', '.csv')
                if file in self.all_micro_expr_features:
                    pain_levels.append(video['pain_level'])
        return np.array(pain_levels) 