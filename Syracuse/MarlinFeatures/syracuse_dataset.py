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
            first_pre = subject_data[(subject_data['visit_type'] == 'pre') & (subject_data['visit_number'] == '1')]
            first_post = subject_data[(subject_data['visit_type'] == 'post') & (subject_data['visit_number'] == '1')]
            
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
            second_pre = subject_data[(subject_data['visit_type'] == 'pre') & (subject_data['visit_number'] == '2')]
            second_post = subject_data[(subject_data['visit_type'] == 'post') & (subject_data['visit_number'] == '2')]
            
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
        For clips with different temporal dimensions (e.g., 5,768 or 3,768),
        we average them to match the expected shape (4,768).
        
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
            features = np.load(clip_path)  # Shape: (N, 768) where N might be 3, 4, or 5
            if features.shape[1] != 768:
                print(f"Warning: Pre clip {clip} has unexpected feature dimension {features.shape[1]}, expected 768")
                continue
                
            # If temporal dimension is not 4, average to get 4 frames
            if features.shape[0] != 4:
                print(f"Info: Pre clip {clip} has {features.shape[0]} frames, averaging to 4")
                # Reshape to get 4 frames by averaging
                if features.shape[0] > 4:
                    # If more than 4 frames, average consecutive frames
                    n_frames = features.shape[0]
                    indices = np.linspace(0, n_frames-1, 4, dtype=int)
                    features = features[indices]
                else:
                    # If less than 4 frames, interpolate each feature dimension separately
                    n_frames = features.shape[0]
                    indices = np.linspace(0, n_frames-1, 4)
                    interpolated_features = np.zeros((4, features.shape[1]))
                    for j in range(features.shape[1]):
                        interpolated_features[:, j] = np.interp(indices, np.arange(n_frames), features[:, j])
                    features = interpolated_features
            
            pre_features.append(features)
        
        if not pre_features:
            raise ValueError(f"No valid pre clips found for pair: Subject {pair['subject']}, Visit {pair['visit_number']}")
        
        pre_features = np.stack(pre_features)  # Shape: (14, 4, 768)
        
        # Load and stack features for post video
        post_features = []
        for clip in post_clips:
            clip_path = os.path.join(self.feature_dir, clip)
            features = np.load(clip_path)  # Shape: (N, 768) where N might be 3, 4, or 5
            if features.shape[1] != 768:
                print(f"Warning: Post clip {clip} has unexpected feature dimension {features.shape[1]}, expected 768")
                continue
                
            # If temporal dimension is not 4, average to get 4 frames
            if features.shape[0] != 4:
                print(f"Info: Post clip {clip} has {features.shape[0]} frames, averaging to 4")
                # Reshape to get 4 frames by averaging
                if features.shape[0] > 4:
                    # If more than 4 frames, average consecutive frames
                    n_frames = features.shape[0]
                    indices = np.linspace(0, n_frames-1, 4, dtype=int)
                    features = features[indices]
                else:
                    # If less than 4 frames, interpolate each feature dimension separately
                    n_frames = features.shape[0]
                    indices = np.linspace(0, n_frames-1, 4)
                    interpolated_features = np.zeros((4, features.shape[1]))
                    for j in range(features.shape[1]):
                        interpolated_features[:, j] = np.interp(indices, np.arange(n_frames), features[:, j])
                    features = interpolated_features
            
            post_features.append(features)
        
        if not post_features:
            raise ValueError(f"No valid post clips found for pair: Subject {pair['subject']}, Visit {pair['visit_number']}")
        
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

    def load_features_for_clip(self, file_name: str, clip_name: str) -> Dict:
        """
        Load features for a single clip and normalize to 4 frames.
        
        Args:
            file_name: Name of the video file (e.g., 'IMG_0001.MP4')
            clip_name: Name of the clip file (e.g., 'IMG_0001_clip_001_aligned.npy')
            
        Returns:
            Dictionary containing:
            - features: numpy array of shape (4, 768)
            - metadata: dictionary with clip information
                - subject_id: subject identifier
                - video_name: name of the video
                - clip_number: clip number (1-14)
                - pain_level: pain level for the video
                - visit_type: pre or post
                - visit_number: visit number (1 or 2)
        """
        # Get video metadata
        video_meta = self.meta_df[self.meta_df['file_name'] == file_name].iloc[0]
        
        # Load and normalize features
        clip_path = os.path.join(self.feature_dir, clip_name)
        features = np.load(clip_path)
        
        # Check feature dimensions
        if features.shape[1] != 768:
            raise ValueError(f"Clip {clip_name} has unexpected feature dimension {features.shape[1]}, expected 768")
        
        # Normalize to 4 frames
        if features.shape[0] != 4:
            if features.shape[0] > 4:
                # If more than 4 frames, average consecutive frames
                n_frames = features.shape[0]
                indices = np.linspace(0, n_frames-1, 4, dtype=int)
                features = features[indices]
            else:
                # If less than 4 frames, interpolate
                n_frames = features.shape[0]
                indices = np.linspace(0, n_frames-1, 4)
                interpolated_features = np.zeros((4, features.shape[1]))
                for j in range(features.shape[1]):
                    interpolated_features[:, j] = np.interp(indices, np.arange(n_frames), features[:, j])
                features = interpolated_features
        
        # Extract clip number from clip name (handling zero-padded numbers)
        clip_str = clip_name.split('_clip_')[1].split('_')[0]
        clip_number = int(clip_str.lstrip('0') or '0')  # Handle cases like '000' correctly
        
        return {
            'features': features,
            'metadata': {
                'subject_id': video_meta['subject_id'],
                'video_name': file_name,
                'clip_number': clip_number,
                'pain_level': video_meta['pain_level'],
                'visit_type': video_meta['visit_type'],
                'visit_number': video_meta['visit_number']
            }
        }
    
    def load_all_clips(self) -> List[Dict]:
        """
        Load features for all clips in the dataset.
        
        Returns:
            List of dictionaries, each containing:
            - features: numpy array of shape (4, 768)
            - metadata: dictionary with clip information
                - subject_id: subject identifier
                - video_name: name of the video
                - clip_number: clip number
                - pain_level: pain level for the video
                - visit_type: pre or post
                - visit_number: visit number (1 or 2)
        """
        all_clips = []
        total_videos = 0
        total_clips_found = 0
        total_clips_loaded = 0
        failed_clips = 0
        
        # Process each video in the meta data
        for _, video_meta in self.meta_df.iterrows():
            if pd.notna(video_meta['pain_level']):  # Only include videos with valid pain levels
                total_videos += 1
                file_name = video_meta['file_name']
                video_name = file_name.replace('.MP4', '')
                
                # Get all clips for this video, ensuring proper sorting of zero-padded numbers
                clips = sorted([f for f in os.listdir(self.feature_dir) 
                              if f.startswith(f"{video_name}_clip_") and f.endswith('_aligned.npy')],
                             key=lambda x: int(x.split('_clip_')[1].split('_')[0]))
                
                total_clips_found += len(clips)
                
                # Load all clips for this video
                for clip in clips:
                    try:
                        clip_data = self.load_features_for_clip(file_name, clip)
                        all_clips.append(clip_data)
                        total_clips_loaded += 1
                    except Exception as e:
                        print(f"Warning: Could not load clip {clip}: {str(e)}")
                        failed_clips += 1
                        continue
        
        # Print statistics
        print("\n=== Clip Loading Statistics ===")
        print(f"Total videos with valid pain levels: {total_videos}")
        print(f"Total clips found: {total_clips_found}")
        print(f"Total clips successfully loaded: {total_clips_loaded}")
        print(f"Failed to load clips: {failed_clips}")
        print(f"Average clips per video: {total_clips_found/total_videos:.2f}")
        print(f"Success rate: {(total_clips_loaded/total_clips_found)*100:.2f}%")
        print("=" * 30)
        
        return all_clips 