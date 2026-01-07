import os
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _to_numeric(x):
    try:
        v = float(x)
        # Treat NaN as missing
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def _video_base_from_file_name(file_name: str) -> str:
    # E.g., IMG_0003.MP4 -> IMG_0003
    base = os.path.basename(file_name)
    if "." in base:
        return base.split(".")[0]
    return base


def _default_cutoffs() -> List[float]:
    # Reasonable default; override via DataModule config
    return [2.0, 4.0, 6.0, 8.0]


def pain_to_class(pain: float, cutoffs: List[float]) -> Optional[int]:
    if pain is None:
        return None
    # classes = len(cutoffs) + 1
    c = 0
    for th in cutoffs:
        if pain > th:
            c += 1
        else:
            break
    return c


def _normalize_outcome(val: Any) -> Optional[int]:
    """Normalize outcome field to {0,1} or None.
    Accepts common truthy/positive/negative strings and numeric aliases.
    Empty strings, NaN, unknown tokens => None (treated as missing).
    """
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("", "nan", "none", "na", "n/a", "null"):
        return None
    # positive aliases
    if s in ("positive", "pos", "+", "1", "true", "yes", "y", "t"):
        return 1
    # negative aliases
    if s in ("negative", "neg", "-", "0", "false", "no", "n", "f"):
        return 0
    # unknown token -> treat as missing
    return None


def _normalize_class_value(val: Any, num_classes: int) -> Optional[int]:
    """Normalize class label to 0..K-1 or None.
    Accepts numeric-like values possibly NaN/empty or 1-based indices.
    - If NaN/empty/None -> None
    - If 0 <= v < K -> v
    - If 1 <= v <= K -> v-1 (assume 1-based)
    - Else -> None
    """
    if val is None:
        return None
    # handle strings like '', 'nan'
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "none", "na", "n/a", "null"):
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if np.isnan(v):
        return None
    # integer-ish
    iv = int(v)
    if abs(v - iv) < 1e-6:
        if 0 <= iv < num_classes:
            return iv
        if 1 <= iv <= num_classes:
            return iv - 1
        return None
    # non-integer values are not accepted as classes
    return None


def collate_mil_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, str, str]]):
    # batch items: (seq[T, D], label, video_id, combo)
    if len(batch) == 0:
        return None
    lengths = [b[0].shape[0] for b in batch]
    max_len = int(max(lengths))
    # Dynamically detect feature dim from the first sample
    d = int(batch[0][0].shape[1])
    bsz = len(batch)
    # Use new_zeros to ensure device/dtype compatibility
    x = batch[0][0].new_zeros((bsz, max_len, d))
    mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    labels = []
    video_ids: List[str] = []
    combos: List[str] = []
    for i, (seq, y, vid, cmb) in enumerate(batch):
        t = seq.shape[0]
        x[i, :t] = seq
        mask[i, :t] = True
        labels.append(y)
        video_ids.append(vid)
        combos.append(cmb)
    y = torch.stack(labels, dim=0)
    return {
        "x": x,
        "mask": mask,
        "y": y,
        "video_ids": video_ids,
        "combos": combos,
    }


@dataclass
class SyracuseBagDatasetConfig:
    feature_root: str
    combos: List[str]
    baseline_combo: str
    meta_excel_path: str
    task: str  # 'ordinal' | 'regression' | 'binary' | 'multiclass'
    num_classes: int = 5
    pain_class_cutoffs: Optional[List[float]] = None
    split: str = "train"
    instance_pool: str = "mean"  # reserved for future (when instances are (16,768))
    normalize_features: bool = True
    max_bag_size: Optional[int] = None  # cap bag length if needed
    # training-time view sampling
    train_sample_all_combos: bool = False  # if True, return multiple views per item (not implemented here)
    # augmented features support
    aug_feature_root: Optional[str] = None
    use_aug_in_train: bool = True
    train_aug_ratio: float = 0.5  # probability to draw from aug variants if available
    clip_level: bool = False  # if True, treat each window/clip (or clip bag) as an individual sample
    clip_bag_size: int = 1
    clip_bag_stride: int = 1
    exclude_video_ids: Optional[List[str]] = None
    feature_suffix: str = "_windows.npy"


class SyracuseBagDataset(Dataset):
    """
    Video-level MIL dataset for Syracuse fixed-combo features.

    Layout assumed: <feature_root>/<combo>/<video_base>_windows.npy with array shape (N, 768), float32.
    Labels loaded from an Excel file with at least 'file_name' and 'pain_level'.
    Optional columns: subject_id, visit_type, outcome, class_k.
    """

    def __init__(self, cfg: SyracuseBagDatasetConfig, index_items: List[Union[str, Tuple[str, str, Optional[str]]]], meta: Dict[str, Dict[str, Any]], aug_variants: Optional[Dict[str, Dict[str, List[str]]]] = None, materialized_items: bool = False):
        super().__init__()
        self.cfg = cfg
        self.index_items = index_items  # list of video_bases OR list of (video_base, combo, aug_path)
        self.meta = meta  # video_base -> metadata dict
        self.aug_variants = aug_variants or {}
        self.materialized_items = bool(materialized_items)
        assert cfg.baseline_combo in cfg.combos, "baseline_combo must be in combos"
        if self.cfg.pain_class_cutoffs is None:
            self.cfg.pain_class_cutoffs = _default_cutoffs()

    def __len__(self):
        return len(self.index_items)

    def _load_bag(self, video_base: str, combo: str, aug_variant_path: Optional[str] = None) -> np.ndarray:
        if aug_variant_path is not None:
            path = aug_variant_path
        else:
            path = os.path.join(self.cfg.feature_root, combo, f"{video_base}{self.cfg.feature_suffix}")
        arr = np.load(path)
        if arr.ndim != 2:
            # Relaxed check: allow 3D if it can be flattened, otherwise error
            if arr.ndim == 3:
                # Auto-flatten (N, T, D) -> (N, T*D) for compatibility
                N, T, D = arr.shape
                arr = arr.reshape(N, T * D)
            else:
                raise ValueError(f"Unexpected feature shape at {path}: {arr.shape}. Expected 2D array (N, Dim).")
        if self.cfg.max_bag_size is not None and arr.shape[0] > self.cfg.max_bag_size:
            # Uniform subsample to cap bag size
            idx = np.linspace(0, arr.shape[0] - 1, self.cfg.max_bag_size, dtype=int)
            arr = arr[idx]
        return arr.astype(np.float32, copy=False)

    def _label_for(self, video_base: str) -> torch.Tensor:
        m = self.meta.get(video_base, {})
        task = self.cfg.task
        if task == "regression":
            pain = _to_numeric(m.get("pain_level"))
            if pain is None:
                pain = float("nan")
            return torch.tensor(pain, dtype=torch.float32)
        elif task in ("ordinal", "multiclass"):
            # Prefer existing class_k that matches num_classes; else derive from pain_level
            key = f"class_{self.cfg.num_classes}"
            raw = m.get(key)
            cls = _normalize_class_value(raw, self.cfg.num_classes)
            if cls is None:
                pain = _to_numeric(m.get("pain_level"))
                cls = pain_to_class(pain, self.cfg.pain_class_cutoffs)
            if cls is None:
                cls = -1
            return torch.tensor(cls, dtype=torch.long)
        elif task == "binary":
            # Normalize outcome; if missing/unknown, fall back to pain_level threshold if available
            norm = _normalize_outcome(m.get("outcome"))
            if norm is not None:
                return torch.tensor(int(norm), dtype=torch.long)
            pain = _to_numeric(m.get("pain_level"))
            if pain is not None:
                y = 1 if pain >= 4.0 else 0
                return torch.tensor(y, dtype=torch.long)
            # Unknown label; caller should have filtered, return -1 to be safe
            return torch.tensor(-1, dtype=torch.long)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def __getitem__(self, idx: int):
        clip_idx: Optional[int] = None
        if self.materialized_items:
            if self.cfg.clip_level:
                video_base, combo, aug_path, clip_idx = self.index_items[idx]  # type: ignore[index]
            else:
                video_base, combo, aug_path = self.index_items[idx]  # type: ignore[index]
        else:
            entry = self.index_items[idx]  # type: ignore[index]
            if self.cfg.clip_level:
                if isinstance(entry, (list, tuple)):
                    video_base = entry[0]
                    clip_idx = int(entry[1])
                else:
                    video_base = entry
                    clip_idx = None
            else:
                video_base = entry
            # Choose combo
            if self.cfg.split == "train":
                combo = random.choice(self.cfg.combos)
            else:
                combo = self.cfg.baseline_combo
            # Possibly choose an augmented variant for training
            aug_path = None
            if self.cfg.split == "train" and self.cfg.use_aug_in_train and (random.random() < float(self.cfg.train_aug_ratio)):
                cdict = self.aug_variants.get(combo)
                if cdict is not None:
                    vlist = cdict.get(video_base)
                    if vlist:
                        aug_path = random.choice(vlist)
        arr = self._load_bag(video_base, combo, aug_variant_path=aug_path)
        if self.cfg.clip_level:
            bag_size = max(1, int(self.cfg.clip_bag_size))
            if clip_idx is None:
                clip_idx = 0
            clip_idx = max(0, min(int(clip_idx), max(0, arr.shape[0] - 1)))
            end_idx = clip_idx + bag_size
            if bag_size > 1 and end_idx > arr.shape[0]:
                clip_idx = max(0, arr.shape[0] - bag_size)
                end_idx = arr.shape[0]
            arr = arr[clip_idx:end_idx]
            if arr.shape[0] == 0:
                arr = self._load_bag(video_base, combo, aug_variant_path=aug_path)[-1:]

        x = torch.from_numpy(arr)
        if self.cfg.normalize_features:
            # Simple per-feature standardization across time (avoid div by zero)
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True, unbiased=False) + 1e-6
            x = (x - mean) / std
        y = self._label_for(video_base)
        if self.cfg.clip_level and clip_idx is not None:
            clip_tag = f"{video_base}_clip_{clip_idx:04d}"
            return x, y, clip_tag, combo
        return x, y, video_base, combo


class SyracuseBagDataModule:
    """
    Minimal DataModule-like wrapper (avoids requiring Lightning at import time).
    Provides train/val/test DataLoaders for the Syracuse MIL dataset.
    """

    def __init__(
        self,
        feature_root: str,
        meta_excel_path: str,
        task: str = "ordinal",
        num_classes: int = 5,
        pain_class_cutoffs: Optional[List[float]] = None,
        combos: Optional[List[str]] = None,
        baseline_combo: str = "RGB",
        batch_size: int = 8,
        num_workers: int = 0,
        val_split_ratio: float = 0.15,
        test_split_ratio: float = 0.15,
        random_state: int = 42,
        max_bag_size: Optional[int] = None,
        normalize_features: bool = True,
        # Augmented features support
        aug_feature_root: Optional[str] = None,
        use_aug_in_train: bool = True,
        train_aug_ratio: float = 0.5,
        # Class balancing
        use_weighted_sampler: bool = True,
        train_epoch_multiplier: int = 1,
        # Enumerate all combos and fixed number of aug views per epoch
        train_enumerate_all_views: bool = False,
        train_max_aug_per_combo: int = 2,
        train_include_original: bool = True,
        clip_level: bool = False,
        clip_bag_size: int = 1,
        clip_bag_stride: int = 1,
        exclude_video_ids: Optional[List[str]] = None,
        loocv_test_subject: Optional[str] = None,
        feature_suffix: str = "_windows.npy",
    ):
        self.feature_root = feature_root
        self.meta_excel_path = meta_excel_path
        self.task = task
        self.num_classes = num_classes
        self.pain_class_cutoffs = pain_class_cutoffs or _default_cutoffs()
        self.combos = combos or self._discover_combos()
        self.baseline_combo = baseline_combo
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
        self.random_state = random_state
        self.max_bag_size = max_bag_size
        self.normalize_features = normalize_features
        self.aug_feature_root = aug_feature_root
        self.use_aug_in_train = use_aug_in_train
        self.train_aug_ratio = train_aug_ratio
        self.use_weighted_sampler = use_weighted_sampler
        self.train_epoch_multiplier = max(1, int(train_epoch_multiplier))
        self.train_enumerate_all_views = bool(train_enumerate_all_views)
        self.train_max_aug_per_combo = max(0, int(train_max_aug_per_combo))
        self.train_include_original = bool(train_include_original)
        self.clip_level = bool(clip_level)
        self.clip_bag_size = max(1, int(clip_bag_size))
        self.clip_bag_stride = max(1, int(clip_bag_stride))
        self.exclude_video_ids = {str(vb).strip() for vb in exclude_video_ids} if exclude_video_ids else set()
        self.loocv_test_subject = str(loocv_test_subject) if loocv_test_subject is not None else None
        self.feature_suffix = feature_suffix

        self._meta: Dict[str, Dict[str, Any]] = {}
        self._train_items: List[str] = []
        self._val_items: List[str] = []
        self._test_items: List[str] = []
        self._aug_variants: Dict[str, Dict[str, List[str]]] = {}
        self._train_is_materialized: bool = False
        self._clip_len_cache: Dict[str, int] = {}

    def _discover_combos(self) -> List[str]:
        names = [d for d in os.listdir(self.feature_root) if os.path.isdir(os.path.join(self.feature_root, d))]
        names.sort()
        return names

    def _list_videos_for_combo(self, combo: str) -> List[str]:
        d = os.path.join(self.feature_root, combo)
        vids = []
        if not os.path.isdir(d):
            return []
        for fn in os.listdir(d):
            if not fn.endswith(self.feature_suffix):
                continue
            base = fn[:-len(self.feature_suffix)]
            vids.append(base)
        vids.sort()
        if self.exclude_video_ids:
            vids = [vb for vb in vids if vb not in self.exclude_video_ids]
        return vids

    def _scan_aug_variants(self) -> Dict[str, Dict[str, List[str]]]:
        """Build mapping: combo -> video_base -> [aug_variant_paths].
        Only include suffixed variants like IMG_0003_1_windows.npy to avoid duplicating originals.
        """
        out: Dict[str, Dict[str, List[str]]] = {}
        if not self.aug_feature_root:
            return out
        if not os.path.isdir(self.aug_feature_root):
            return out
        for combo in self.combos:
            combo_dir = os.path.join(self.aug_feature_root, combo)
            if not os.path.isdir(combo_dir):
                continue
            mapping: Dict[str, List[str]] = {}
            for fn in os.listdir(combo_dir):
                if not fn.endswith(self.feature_suffix):
                    continue
                name = fn[:-len(self.feature_suffix)]  # e.g., IMG_0003 or IMG_0003_1
                # collect only names with numeric suffix
                if "_" in name and name.split("_")[-1].isdigit():
                    video_base = "_".join(name.split("_")[:-1])
                    mapping.setdefault(video_base, []).append(os.path.join(combo_dir, fn))
            if mapping:
                out[combo] = mapping
        return out

    def _get_clip_count(self, video_base: str) -> int:
        if video_base in self._clip_len_cache:
            return self._clip_len_cache[video_base]
        path = os.path.join(self.feature_root, self.baseline_combo, f"{video_base}{self.feature_suffix}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing feature file for clip expansion: {path}")
        arr = np.load(path)
        count = int(arr.shape[0])
        self._clip_len_cache[video_base] = count
        return count

    def _bag_start_indices(self, video_base: str) -> List[int]:
        total = self._get_clip_count(video_base)
        bag = self.clip_bag_size
        stride = self.clip_bag_stride
        if bag <= 1:
            return list(range(total))
        if total <= bag:
            return [0]
        starts: List[int] = []
        max_start = total - bag
        pos = 0
        while pos <= max_start:
            starts.append(pos)
            pos += stride
        if starts[-1] != max_start:
            starts.append(max_start)
        return starts

    def _expand_items_to_clips(self, items: List[Any]) -> List[Any]:
        expanded: List[Any] = []
        for item in items:
            if isinstance(item, (list, tuple)):
                if len(item) == 4:
                    # already expanded (video, combo, aug_path, clip_idx)
                    expanded.append(tuple(item))
                    continue
                if len(item) == 3:
                    video_base, combo, aug_path = item
                    for start_idx in self._bag_start_indices(video_base):
                        expanded.append((video_base, combo, aug_path, start_idx))
                    continue
                if len(item) == 2 and isinstance(item[1], int):
                    video_base, clip_idx = item
                    expanded.append((video_base, clip_idx))
                    continue
                video_base = item[0]
            else:
                video_base = item
            for clip_idx in self._bag_start_indices(video_base):
                expanded.append((video_base, clip_idx))
        return expanded

    def _load_meta_excel(self) -> Dict[str, Dict[str, Any]]:
        import pandas as pd
        df = pd.read_excel(self.meta_excel_path)
        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]
        # Ensure 'file_name' present
        fname_col = None
        for cand in ("file_name", "filename", "video_file", "video_name"):
            if cand in df.columns:
                fname_col = cand
                break
        if fname_col is None:
            raise KeyError("Expected a 'file_name' column in the Excel metadata")
        # Basic coercions
        if "pain_level" in df.columns:
            df["pain_level"] = pd.to_numeric(df["pain_level"], errors="coerce")
        # Coerce class_* columns to numeric where present
        for c in list(df.columns):
            if isinstance(c, str) and c.startswith("class_"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # Derive video_base
        df["video_base"] = df[fname_col].apply(_video_base_from_file_name)
        # Build mapping
        meta: Dict[str, Dict[str, Any]] = {}
        keep_cols = set(df.columns)
        for _, row in df.iterrows():
            vb = str(row["video_base"]).strip()
            meta[vb] = {k: row[k] for k in keep_cols if k != "video_base"}
        return meta

    def setup(self):
        # Load meta
        self._meta = self._load_meta_excel()
        # Use baseline combo to enumerate available videos (ensures val/test loadability)
        base_videos = self._list_videos_for_combo(self.baseline_combo)
        # Filter to those present in metadata
        videos = [vb for vb in base_videos if vb in self._meta]

        def _has_valid_label(vb: str) -> bool:
            m = self._meta.get(vb, {})
            if self.task == "regression":
                return _to_numeric(m.get("pain_level")) is not None
            elif self.task in ("ordinal", "multiclass"):
                key = f"class_{self.num_classes}"
                cls = _normalize_class_value(m.get(key), self.num_classes)
                if cls is not None:
                    return True
                # derive from pain_level if available
                return _to_numeric(m.get("pain_level")) is not None
            elif self.task == "binary":
                # outcome must be a recognized token; else require numeric pain_level
                if _normalize_outcome(m.get("outcome")) is not None:
                    return True
                return _to_numeric(m.get("pain_level")) is not None
            else:
                return False

        # Drop samples without usable labels
        videos = [vb for vb in videos if _has_valid_label(vb)]

        # Subject-aware split if subject_id available
        rng = random.Random(self.random_state)
        subjects: Dict[str, List[str]] = {}
        for vb in videos:
            subj = str(self._meta[vb].get("subject_id")) if self._meta[vb].get("subject_id") is not None else None
            key = subj if (subj and subj.lower() != "nan") else f"__no_subject__"
            subjects.setdefault(key, []).append(vb)

        subject_keys = list(subjects.keys())
        rng.shuffle(subject_keys)
        # LOOCV support: if a specific test subject is set, use it as the test split
        if getattr(self, 'loocv_test_subject', None) is not None:
            test_key = self.loocv_test_subject
            if test_key in subjects:
                remaining = [k for k in subject_keys if k != test_key]
                test_keys = {test_key}
            else:
                # Fallback: empty test; use ratio for all subjects
                remaining = subject_keys
                test_keys = set()
            n_rem = len(remaining)
            if n_rem <= 1:
                val_keys = set(remaining[:1])
                train_keys = set(remaining[1:])
            else:
                n_val = max(1, int(round(self.val_split_ratio * n_rem)))
                val_keys = set(remaining[:n_val])
                train_keys = set(remaining[n_val:])
        else:
            n_subj = len(subject_keys)
            n_test = max(1, int(round(self.test_split_ratio * n_subj)))
            n_val = max(1, int(round(self.val_split_ratio * n_subj)))
            test_keys = set(subject_keys[:n_test])
            val_keys = set(subject_keys[n_test:n_test + n_val])
            train_keys = set(subject_keys[n_test + n_val:])

        def gather(keys: set) -> List[str]:
            lst: List[str] = []
            for k in keys:
                lst.extend(subjects.get(k, []))
            lst.sort()
            return lst

        self._train_items = gather(train_keys)
        self._val_items = gather(val_keys)
        self._test_items = gather(test_keys)

        # Build augmented variants index for training-time sampling
        self._aug_variants = self._scan_aug_variants()

        # Optionally materialize all (video, combo, view) items for training
        if self.train_enumerate_all_views:
            materialized: List[Tuple[str, str, Optional[str]]] = []
            for vb in self._train_items:
                for combo in self.combos:
                    # include original view from feature_root
                    if self.train_include_original:
                        materialized.append((vb, combo, None))
                    # add up to K aug views if available
                    aug_map = self._aug_variants.get(combo, {})
                    vlist = sorted(aug_map.get(vb, []))
                    if self.train_max_aug_per_combo > 0 and vlist:
                        for pth in vlist[: self.train_max_aug_per_combo]:
                            materialized.append((vb, combo, pth))
            # replace train_items with materialized entries and mark flag
            self._train_items = materialized  # type: ignore[assignment]
            self._train_is_materialized = True
        else:
            self._train_is_materialized = False

        if self.clip_level:
            self._train_items = self._expand_items_to_clips(self._train_items)
            self._val_items = self._expand_items_to_clips(self._val_items)
            self._test_items = self._expand_items_to_clips(self._test_items)

    def _make_dataset(self, split: str) -> SyracuseBagDataset:
        cfg = SyracuseBagDatasetConfig(
            feature_root=self.feature_root,
            combos=self.combos,
            baseline_combo=self.baseline_combo,
            meta_excel_path=self.meta_excel_path,
            task=self.task,
            num_classes=self.num_classes,
            pain_class_cutoffs=self.pain_class_cutoffs,
            split=split,
            max_bag_size=self.max_bag_size,
            normalize_features=self.normalize_features,
            aug_feature_root=self.aug_feature_root,
            use_aug_in_train=self.use_aug_in_train,
            train_aug_ratio=self.train_aug_ratio,
            clip_level=self.clip_level,
            clip_bag_size=self.clip_bag_size,
            clip_bag_stride=self.clip_bag_stride,
            exclude_video_ids=list(self.exclude_video_ids),
            feature_suffix=self.feature_suffix,
        )
        items = {
            "train": self._train_items,
            "val": self._val_items,
            "test": self._test_items,
        }[split]
        materialized = (split == "train" and self._train_is_materialized)
        return SyracuseBagDataset(cfg, items, self._meta, aug_variants=self._aug_variants, materialized_items=materialized)

    def train_dataloader(self) -> DataLoader:
        ds = self._make_dataset("train")
        # If we've materialized all views, iterate once per epoch with shuffle for stability
        if self._train_is_materialized:
            return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_mil_batch, drop_last=True)
        if self.use_weighted_sampler and self.task in ("ordinal", "multiclass", "binary"):
            # Build weights inverse to class frequency
            from torch.utils.data import WeightedRandomSampler
            labels: List[int] = []
            for entry in ds.index_items:
                if isinstance(entry, (list, tuple)):
                    vb = entry[0]
                else:
                    vb = entry
                y = ds._label_for(vb).item()
                labels.append(int(y))
            # compute frequencies for valid classes
            from collections import Counter
            cnt = Counter([y for y in labels if y >= 0])
            # avoid zero division; if a class missing, skip weighting for that class
            weights = []
            for y in labels:
                w = 1.0 / cnt[y] if y in cnt and cnt[y] > 0 else 0.0
                weights.append(w)
            total_samples = len(labels) * self.train_epoch_multiplier
            sampler = WeightedRandomSampler(weights=weights, num_samples=total_samples, replacement=True)
            return DataLoader(ds, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers, collate_fn=collate_mil_batch, drop_last=True)
        else:
            return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_mil_batch, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        ds = self._make_dataset("val")
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_mil_batch)

    def test_dataloader(self) -> DataLoader:
        ds = self._make_dataset("test")
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_mil_batch)
