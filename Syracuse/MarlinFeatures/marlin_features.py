import os
import json
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path

"""
MarlinFeatures Class Documentation

This class provides functionality to load and process MARLIN features from clip files.

Metadata Structure:
The JSON file contains metadata for each clip with the following structure:
{
    "filename": "IMG_0005_4_aligned_clip_007.npy",
    "video_id": "0005",
    "clip_id": "007",
    "video_type": "aug",  # or "original"
    "meta_info": {
        "file_name": "IMG_0005.MP4",
        "creation_time": "2024-09-04T14:21:00",
        "duration": "00:01:01.700000",
        "subject_id": 28,
        "pain_level": 3.0,
        "visit_type": "post",
        "comment": null,
        "outcome": "negative",
        "visit_number": 1.0,
        "class_3": 1.0,
        "class_4": 1.0,
        "class_5": 1.0
    }
}

Methods:

1. __init__(base_dir: str)
   - Initializes the MarlinFeatures class
   - Args: base_dir - Base directory containing clips and metadata (default: '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2')

2. _load_metadata()
   - Internal method to load clips metadata from JSON file
   - No arguments
   - Raises FileNotFoundError if metadata file not found

3. _normalize_features(features: np.ndarray) -> np.ndarray
   - Internal method to normalize features to shape (4, 768)
   - Args: features - Raw features array
   - Returns: Normalized features with shape (4, 768)
   - Raises ValueError if feature dimension is not 768

4. get_clips_for_video(video_id: str, include_augmented: bool = False) -> Dict[str, Dict]
   - Loads all clips for a specific video ID
   - Args:
     * video_id - The video ID to load clips for (e.g., '0005')
     * include_augmented - If True, include augmented clips; if False, only include original clips
   - Returns: Dictionary mapping clip filenames to dictionaries containing:
     * features: numpy array of shape (4, 768)
     * metadata: dictionary with clip metadata from JSON (see Metadata Structure above)

5. get_clip(filename: str) -> Optional[Dict]
   - Loads a single clip by filename
   - Args: filename - The clip filename
   - Returns: Dictionary containing:
     * features: numpy array of shape (4, 768)
     * metadata: dictionary with clip metadata from JSON (see Metadata Structure above)
     * None if clip not found or error loading

6. get_clip_metadata(filename: str) -> Optional[Dict]
   - Gets metadata for a specific clip
   - Args: filename - The clip filename
   - Returns: The clip metadata if found, None otherwise (see Metadata Structure above)

7. get_all_video_ids() -> List[str]
   - Gets a list of all unique video IDs in the dataset
   - No arguments
   - Returns: List of video IDs

8. get_clip_count_by_video(video_id: str, include_augmented: bool = False) -> Dict[str, int]
   - Gets the count of original and augmented clips for a video
   - Args:
     * video_id - The video ID
     * include_augmented - Whether to include augmented clips in the count
   - Returns: Dictionary with counts of original and augmented clips
"""

class MarlinFeatures:
    def __init__(self, base_dir: str = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'):
        """
        Initialize MarlinFeatures class.
        
        Args:
            base_dir (str): Base directory containing the clips and metadata
        """
        self.base_dir = Path(base_dir)
        self.clips_json_path = self.base_dir / 'clips_json.json'
        self._load_metadata()
        
    def _load_metadata(self):
        """Load the clips metadata from JSON file."""
        if not self.clips_json_path.exists():
            raise FileNotFoundError(f"Clips metadata file not found at {self.clips_json_path}")
            
        with open(self.clips_json_path, 'r') as f:
            self.clips_metadata = json.load(f)
            
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to ensure shape (4, 768).
        
        Args:
            features (np.ndarray): Raw features array
            
        Returns:
            np.ndarray: Normalized features with shape (4, 768)
        """
        # Verify feature dimension is 768
        if features.shape[1] != 768:
            raise ValueError(f"Feature dimension is {features.shape[1]}, expected 768")
            
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
                
        return features
            
    def get_clips_for_video(self, video_id: str, include_augmented: bool = False) -> Dict[str, Dict]:
        """
        Load all clips for a specific video ID.
        
        Args:
            video_id (str): The video ID to load clips for (e.g., '0005')
            include_augmented (bool): If True, include augmented clips. If False, only include original clips.
            
        Returns:
            Dict[str, Dict]: Dictionary mapping clip filenames to dictionaries containing:
                - features: numpy array of shape (4, 768)
                - metadata: dictionary with clip metadata from JSON
        """
        # Find all clips for this video
        video_clips = {}
        for filename, metadata in self.clips_metadata.items():
            if metadata['video_id'] == video_id:
                # Skip augmented clips if not requested
                if not include_augmented and metadata['video_type'] == 'aug':
                    continue
                    
                # Load the clip features
                clip_path = self.base_dir / filename
                if clip_path.exists():
                    try:
                        # Load raw features
                        raw_features = np.load(clip_path)
                        
                        # Normalize features to (4, 768)
                        normalized_features = self._normalize_features(raw_features)
                        
                        # Store features and metadata
                        video_clips[filename] = {
                            'features': normalized_features,
                            'metadata': metadata
                        }
                    except Exception as e:
                        print(f"Error loading clip {filename}: {e}")
                        
        return video_clips
    
    def get_clip(self, filename: str) -> Optional[Dict]:
        """
        Load a single clip by filename.
        
        Args:
            filename (str): The clip filename
            
        Returns:
            Optional[Dict]: Dictionary containing:
                - features: numpy array of shape (4, 768)
                - metadata: dictionary with clip metadata from JSON
                None if clip not found or error loading
        """
        # Check if clip exists in metadata
        if filename not in self.clips_metadata:
            print(f"Clip {filename} not found in metadata")
            return None
            
        # Load the clip features
        clip_path = self.base_dir / filename
        if not clip_path.exists():
            print(f"Clip file {filename} not found at {clip_path}")
            return None
            
        try:
            # Load raw features
            raw_features = np.load(clip_path)
            
            # Normalize features to (4, 768)
            normalized_features = self._normalize_features(raw_features)
            
            # Return features and metadata
            return {
                'features': normalized_features,
                'metadata': self.clips_metadata[filename]
            }
        except Exception as e:
            print(f"Error loading clip {filename}: {e}")
            return None
    
    def get_clip_metadata(self, filename: str) -> Optional[Dict]:
        """
        Get metadata for a specific clip.
        
        Args:
            filename (str): The clip filename
            
        Returns:
            Optional[Dict]: The clip metadata if found, None otherwise
        """
        return self.clips_metadata.get(filename)
    
    def get_all_video_ids(self) -> List[str]:
        """
        Get a list of all unique video IDs in the dataset.
        
        Returns:
            List[str]: List of video IDs
        """
        return list(set(metadata['video_id'] for metadata in self.clips_metadata.values()))
    
    def get_clip_count_by_video(self, video_id: str, include_augmented: bool = False) -> Dict[str, int]:
        """
        Get the count of original and augmented clips for a video.
        
        Args:
            video_id (str): The video ID
            include_augmented (bool): Whether to include augmented clips in the count
            
        Returns:
            Dict[str, int]: Dictionary with counts of original and augmented clips
        """
        original_count = 0
        aug_count = 0
        
        for metadata in self.clips_metadata.values():
            if metadata['video_id'] == video_id:
                if metadata['video_type'] == 'original':
                    original_count += 1
                elif metadata['video_type'] == 'aug' and include_augmented:
                    aug_count += 1
                    
        return {
            'original': original_count,
            'augmented': aug_count
        } 