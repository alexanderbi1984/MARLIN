import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional

class SyracuseDataset:
    def __init__(self, meta_path: str, feature_dir: str):
        """
        Initialize the Syracuse dataset.
        
        Args:
            meta_path: Path to the meta_with_outcomes.xlsx file
            feature_dir: Directory containing the feature files
        """
        self.meta_path = meta_path
        self.feature_dir = feature_dir
        
        # Load meta data
        self.meta_df = pd.read_excel(meta_path)
        
        # Convert pain_level to numeric
        self.meta_df['pain_level'] = pd.to_numeric(self.meta_df['pain_level'], errors='coerce')
        
        # Extract visit number and type
        self.meta_df['visit_number'] = self.meta_df['visit_type'].str.extract('(\d+)')
        self.meta_df['visit_type'] = self.meta_df['visit_type'].str.extract('(pre|post)')
        
        # Create pre-post pairs
        self.pairs = self._create_prepost_pairs()
        
    def _create_prepost_pairs(self) -> List[Dict]:
        """
        Create pairs of pre-post visits for each subject.
        
        Returns:
            List of dictionaries containing pre-post pairs
        """
        pairs = []
        for subj in self.meta_df['subject_id'].unique():
            subject_data = self.meta_df[self.meta_df['subject_id'] == subj]
            
            # Analyze 1st visit
            first_pre = subject_data[subject_data['visit_type'] == 'pre']
            first_post = subject_data[subject_data['visit_type'] == 'post']
            
            if len(first_pre) > 0 and len(first_post) > 0:
                pre_pain = first_pre['pain_level'].iloc[0] if not first_pre['pain_level'].isna().all() else None
                post_pain = first_post['pain_level'].iloc[0] if not first_post['pain_level'].isna().all() else None
                
                if pre_pain is not None and post_pain is not None:
                    if pd.notna(pre_pain) and pd.notna(post_pain):
                        change = pre_pain - post_pain
                        pairs.append({
                            'subject': subj,
                            'visit_number': '1',
                            'pre_pain': pre_pain,
                            'post_pain': post_pain,
                            'change': change,
                            'pre_file': first_pre['file_name'].iloc[0],
                            'post_file': first_post['file_name'].iloc[0]
                        })
            
            # Analyze 2nd visit
            second_pre = subject_data[subject_data['visit_type'] == 'pre']
            second_post = subject_data[subject_data['visit_type'] == 'post']
            
            if len(second_pre) > 0 and len(second_post) > 0:
                pre_pain = second_pre['pain_level'].iloc[0] if not second_pre['pain_level'].isna().all() else None
                post_pain = second_post['pain_level'].iloc[0] if not second_post['pain_level'].isna().all() else None
                
                if pre_pain is not None and post_pain is not None:
                    if pd.notna(pre_pain) and pd.notna(post_pain):
                        change = pre_pain - post_pain
                        pairs.append({
                            'subject': subj,
                            'visit_number': '2',
                            'pre_pain': pre_pain,
                            'post_pain': post_pain,
                            'change': change,
                            'pre_file': second_pre['file_name'].iloc[0],
                            'post_file': second_post['file_name'].iloc[0]
                        })
        
        return pairs
    
    def load_features_for_pair(self, pair: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features for a pre-post pair, using exactly 14 clips.
        
        Args:
            pair: Dictionary containing pair information
            
        Returns:
            Tuple of (pre_features, post_features) arrays, each with shape (14, 4, 768)
        """
        pre_file = pair['pre_file']
        post_file = pair['post_file']
        
        # Get all clips for pre and post videos
        pre_clips = sorted([f for f in os.listdir(self.feature_dir) 
                          if f.startswith(pre_file.replace('.MP4', '_clip_')) and f.endswith('_aligned.npy')])[:14]
        post_clips = sorted([f for f in os.listdir(self.feature_dir) 
                           if f.startswith(post_file.replace('.MP4', '_clip_')) and f.endswith('_aligned.npy')])[:14]
        
        if len(pre_clips) < 14 or len(post_clips) < 14:
            raise ValueError(f"Not enough clips for pair: Subject {pair['subject']}, Visit {pair['visit_number']}")
        
        # Load and stack features for pre video
        pre_features = []
        for clip in pre_clips:
            clip_path = os.path.join(self.feature_dir, clip)
            features = np.load(clip_path)  # Shape: (4, 768)
            pre_features.append(features)
        pre_features = np.stack(pre_features)  # Shape: (14, 4, 768)
        
        # Load and stack features for post video
        post_features = []
        for clip in post_clips:
            clip_path = os.path.join(self.feature_dir, clip)
            features = np.load(clip_path)  # Shape: (4, 768)
            post_features.append(features)
        post_features = np.stack(post_features)  # Shape: (14, 4, 768)
        
        return pre_features, post_features
    
    def get_all_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load features for all pairs.
        
        Returns:
            Tuple of (pre_features, post_features, changes) arrays
        """
        all_pre_features = []
        all_post_features = []
        changes = []
        
        for pair in self.pairs:
            pre_features, post_features = self.load_features_for_pair(pair)
            all_pre_features.append(pre_features)
            all_post_features.append(post_features)
            changes.append(pair['change'])
        
        return (np.stack(all_pre_features), 
                np.stack(all_post_features), 
                np.array(changes))
    
    def get_feature_statistics(self) -> Dict:
        """
        Calculate statistics about the features.
        
        Returns:
            Dictionary containing feature statistics
        """
        pre_features, post_features, changes = self.get_all_features()
        
        stats = {
            'num_pairs': len(self.pairs),
            'num_clips_per_video': pre_features.shape[1],
            'feature_dimension': pre_features.shape[2],
            'pre_features_shape': pre_features.shape,
            'post_features_shape': post_features.shape,
            'changes_shape': changes.shape,
            'mean_change': np.mean(changes),
            'std_change': np.std(changes),
            'min_change': np.min(changes),
            'max_change': np.max(changes)
        }
        
        return stats
    
    def get_pair_info(self) -> pd.DataFrame:
        """
        Get information about all pairs.
        
        Returns:
            DataFrame containing pair information
        """
        return pd.DataFrame(self.pairs)

    def compute_temporal_statistics(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute temporal statistics for a video's features.
        
        Args:
            features: Array of shape (num_clips, num_frames, feature_dim)
            
        Returns:
            Dictionary containing various temporal statistics
        """
        # Mean across frames for each clip
        clip_means = np.mean(features, axis=1)  # (num_clips, feature_dim)
        
        # Standard deviation across frames for each clip
        clip_stds = np.std(features, axis=1)  # (num_clips, feature_dim)
        
        # Temporal differences between consecutive clips
        clip_diffs = np.diff(clip_means, axis=0)  # (num_clips-1, feature_dim)
        
        # Rate of change (first derivative approximation)
        rate_of_change = np.gradient(clip_means, axis=0)  # (num_clips, feature_dim)
        
        # Acceleration (second derivative approximation)
        acceleration = np.gradient(rate_of_change, axis=0)  # (num_clips, feature_dim)
        
        return {
            'means': clip_means,
            'stds': clip_stds,
            'diffs': clip_diffs,
            'rate_of_change': rate_of_change,
            'acceleration': acceleration
        }

    def analyze_temporal_patterns(self, pre_features: np.ndarray, post_features: np.ndarray) -> Dict:
        """Analyze temporal patterns in pre and post features."""
        def compute_stats(features: np.ndarray) -> Dict:
            means = np.mean(features, axis=0)  # Mean across time for each feature
            rate_of_change = np.diff(features, axis=0)  # First derivative
            acceleration = np.diff(rate_of_change, axis=0)  # Second derivative
            stability = np.std(features, axis=0)  # Standard deviation across time
            
            return {
                'means': means,
                'rate_of_change': rate_of_change,
                'acceleration': acceleration,
                'stability': stability
            }
        
        pre_stats = compute_stats(pre_features)
        post_stats = compute_stats(post_features)
        
        # Compute differences and correlations
        mean_diff = post_stats['means'] - pre_stats['means']
        temporal_correlation = np.array([np.corrcoef(pre_features[:, i], post_features[:, i])[0, 1] 
                                       for i in range(pre_features.shape[1])])
        stability_change = post_stats['stability'] - pre_stats['stability']
        
        return {
            'pre_stats': pre_stats,
            'post_stats': post_stats,
            'mean_diff': mean_diff,
            'temporal_correlation': temporal_correlation,
            'stability_change': stability_change
        }

    def analyze_all_temporal_patterns(self) -> Tuple[List[Dict], Dict]:
        """Analyze temporal patterns for all pre-post pairs."""
        analyses = []
        all_mean_diffs = []
        all_temporal_corrs = []
        all_stability_changes = []
        
        for pair in self.pairs:
            pre_features, post_features = self.load_features_for_pair(pair)
            
            analysis = self.analyze_temporal_patterns(pre_features, post_features)
            analyses.append(analysis)
            
            all_mean_diffs.append(analysis['mean_diff'])
            all_temporal_corrs.append(analysis['temporal_correlation'])
            all_stability_changes.append(analysis['stability_change'])
        
        # Compute correlations between temporal patterns and pain reduction
        changes = np.array([pair['change'] for pair in self.pairs])
        correlations = {
            'mean_diff': np.mean([np.corrcoef(changes, np.array(all_mean_diffs)[:, i])[0, 1] 
                                for i in range(len(all_mean_diffs[0]))]),
            'temporal_correlation': np.mean([np.corrcoef(changes, np.array(all_temporal_corrs)[:, i])[0, 1] 
                                          for i in range(len(all_temporal_corrs[0]))]),
            'stability_change': np.mean([np.corrcoef(changes, np.array(all_stability_changes)[:, i])[0, 1] 
                                       for i in range(len(all_stability_changes[0]))])
        }
        
        return analyses, correlations 