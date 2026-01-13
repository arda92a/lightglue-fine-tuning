"""
Product Pairs Dataset for LightGlue training.

This dataset loads pairs of shelf images with product bounding boxes and embeddings.
Ground truth matches are loaded from a separate matches.json file.

Expected data structure:
    data_dir/
    ├── images/
    │   ├── shelf_001.jpg
    │   ├── shelf_002.jpg
    │   └── ...
    ├── annotations/
    │   ├── shelf_001.json  # Contains products with numeric product_id, bbox, embedding
    │   ├── shelf_002.json
    │   └── ...
    └── matches.json        # Explicit product correspondences between image pairs

Annotation JSON format (per image):
    {
        "products": [
            {
                "product_id": 0,              # Numeric ID within this image
                "bbox": [x1, y1, x2, y2],
                "embedding": [0.1, 0.2, ...]  # 128-dim metric embedding
            },
            ...
        ]
    }

Matches JSON format:
    {
        "pairs": [
            {
                "image0": "shelf_001",
                "image1": "shelf_002",
                "matches": [[0, 0], [2, 1], [3, 3]]  # [idx0, idx1] pairs
            },
            ...
        ]
    }
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

# Support both package import and direct execution
try:
    from ..settings import DATA_PATH
    from .base_dataset import BaseDataset
except ImportError:
    # Direct execution - add path and import
    import sys
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    sys.path.insert(0, str(root_dir))
    from gluefactory.settings import DATA_PATH
    from gluefactory.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class ProductPairsDataset(BaseDataset):
    """Dataset for loading product pairs from shelf images with explicit matches."""
    
    default_conf = {
        # Data paths
        "data_dir": "lightglue_dataset",       # Top-level directory
        "image_dir": "images/",                 # Subdirectory with images
        "annotation_dir": "annotations/",       # Subdirectory with JSON annotations
        "matches_file": "matches.json",         # File with explicit product matches
        
        # Splits (ratios - must sum to 1.0)
        "train_ratio": 0.7,
        "val_ratio": 0.2,
        "test_ratio": 0.1,
        "shuffle_seed": 42,
        
        # Image settings
        "grayscale": False,
        "resize": [640, 480],                   # [width, height]
        
        # Product settings
        "embedding_dim": 128,                   # Your metric embedding dimension
        "max_products": 400,                     # Max products per image
        "pad_products": True,                   # Pad to max_products
    }

    def _init(self, conf):
        data_dir = DATA_PATH / conf.data_dir
        
        if not data_dir.exists():
            logger.warning(f"Data directory {data_dir} not found.")
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Load matches
        matches_file = data_dir / conf.matches_file
        if not matches_file.exists():
            raise FileNotFoundError(f"Matches file not found: {matches_file}")
        
        with open(matches_file) as f:
            matches_data = json.load(f)
        
        # Build pairs list with match info
        pairs = []
        for pair_info in matches_data["pairs"]:
            pairs.append({
                "image0": pair_info["image0"],
                "image1": pair_info["image1"],
                "matches": pair_info["matches"],  # [[idx0, idx1], ...]
            })
        
        # Shuffle and split by ratios
        if conf.shuffle_seed is not None:
            np.random.RandomState(conf.shuffle_seed).shuffle(pairs)
        
        total = len(pairs)
        train_size = int(total * conf.train_ratio)
        val_size = int(total * conf.val_ratio)
        # test gets the remainder to ensure all pairs are used
        
        train_pairs = pairs[:train_size]
        val_pairs = pairs[train_size:train_size + val_size]
        test_pairs = pairs[train_size + val_size:]
        
        self.pairs = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
        
        logger.info(f"Loaded {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test pairs")

    def get_dataset(self, split):
        return _ProductDataset(self.conf, self.pairs[split], split)


class _ProductDataset(torch.utils.data.Dataset):
    """Internal dataset class for product pairs."""
    
    def __init__(self, conf, pairs, split):
        self.conf = conf
        self.pairs = pairs
        self.split = split
        
        self.data_dir = DATA_PATH / conf.data_dir
        self.image_dir = self.data_dir / conf.image_dir
        self.annotation_dir = self.data_dir / conf.annotation_dir

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_info = self.pairs[idx]
        name0, name1 = pair_info["image0"], pair_info["image1"]
        explicit_matches = pair_info["matches"]  # [[idx0, idx1], ...]
        
        # Load view 0
        view0 = self._load_view(name0)
        
        # Load view 1
        view1 = self._load_view(name1)
        
        # Build ground truth matches from explicit annotations
        gt_matches0, gt_matches1 = self._build_gt_matches(
            explicit_matches,
            view0["n_products"],
            view1["n_products"],
            self.conf.max_products if self.conf.pad_products else None
        )
        
        # Build gt_assignment matrix (loss için gerekli)
        gt_assignment = self._build_assignment_matrix(
            explicit_matches,
            view0["n_products"],
            view1["n_products"],
            self.conf.max_products if self.conf.pad_products else None
        )
        
        data = {
            "name": f"{name0}_{name1}",
            "idx": idx,
            "view0": view0,
            "view1": view1,
            "gt_matches0": gt_matches0,  # [N0] - index into view1, -1 if unmatched
            "gt_matches1": gt_matches1,  # [N1] - index into view0, -1 if unmatched
            "gt_assignment": gt_assignment,  # [N0, N1] - binary assignment matrix
        }
        
        return data

    def _load_view(self, name: str) -> Dict:
        """Load annotation for a single view."""
        # Load annotation
        annotation_path = self.annotation_dir / f"{name}.json"
        if annotation_path.exists():
            with open(annotation_path) as f:
                annotation = json.load(f)
            products = annotation.get("products", [])
        else:
            products = []
            logger.warning(f"Annotation not found: {annotation_path}")
        
        # Extract keypoints (bbox centers) and embeddings
        keypoints = []
        embeddings = []
        
        for product in products[:self.conf.max_products]:
            bbox = product["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            keypoints.append([cx, cy])
            embeddings.append(product["embedding"])
        
        n_products = len(keypoints)
        
        if self.conf.pad_products and n_products < self.conf.max_products:
            pad_n = self.conf.max_products - n_products
            keypoints.extend([[0, 0]] * pad_n)
            embeddings.extend([[0.0] * self.conf.embedding_dim] * pad_n)
        
        # Convert to tensors
        keypoints = torch.tensor(keypoints, dtype=torch.float32) if keypoints else torch.zeros((0, 2))
        embeddings = torch.tensor(embeddings, dtype=torch.float32) if embeddings else torch.zeros((0, self.conf.embedding_dim))
        
        # Validity mask (keypoint_scores)
        total_len = len(keypoints) if len(keypoints) > 0 else 0
        keypoint_scores = torch.zeros(total_len)
        keypoint_scores[:n_products] = 1.0
        
        view = {
            "image_size": torch.tensor(self.conf.resize if self.conf.resize else [640, 480], dtype=torch.float32),
            "keypoints": keypoints,
            "descriptors": embeddings,
            "keypoint_scores": keypoint_scores,
            "n_products": n_products,
        }
        
        return view

    def _build_gt_matches(
        self,
        explicit_matches: List[List[int]],
        n0: int,
        n1: int,
        padded_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build ground truth match tensors from explicit match list.
        
        Args:
            explicit_matches: List of [idx0, idx1] pairs
            n0: Number of actual products in image 0
            n1: Number of actual products in image 1
            padded_size: If set, pad output tensors to this size
        
        Returns:
            gt_matches0: [N0] tensor, gt_matches0[i] = j means product i in img0 matches product j in img1
                        -1 means unmatched
            gt_matches1: [N1] tensor, inverse mapping
        """
        size0 = padded_size if padded_size else n0
        size1 = padded_size if padded_size else n1
        
        # Initialize with -1 (unmatched)
        gt_matches0 = torch.full((size0,), -1, dtype=torch.long)
        gt_matches1 = torch.full((size1,), -1, dtype=torch.long)
        
        # Fill in explicit matches
        for idx0, idx1 in explicit_matches:
            if idx0 < n0 and idx1 < n1:  # Valid indices only
                gt_matches0[idx0] = idx1
                gt_matches1[idx1] = idx0
        
        return gt_matches0, gt_matches1

    def _build_assignment_matrix(
        self,
        explicit_matches: List[List[int]],
        n0: int,
        n1: int,
        padded_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Build ground truth assignment matrix for loss computation.
        
        Returns:
            gt_assignment: [N0, N1] binary matrix, 1 where match exists
        """
        size0 = padded_size if padded_size else n0
        size1 = padded_size if padded_size else n1
        
        assignment = torch.zeros((size0, size1), dtype=torch.float32)
        
        for idx0, idx1 in explicit_matches:
            if idx0 < n0 and idx1 < n1:
                assignment[idx0, idx1] = 1.0
        
        return assignment


# For testing the dataset
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent paths for direct execution
    current_dir = Path(__file__).resolve().parent
    gluefactory_dir = current_dir.parent
    root_dir = gluefactory_dir.parent
    sys.path.insert(0, str(root_dir))
    
    from gluefactory.settings import DATA_PATH
    from omegaconf import OmegaConf
    
    print(f"DATA_PATH: {DATA_PATH}")
    
    conf = OmegaConf.create({
        "batch_size": 2,
        "num_workers": 0,
        "prefetch_factor": None,  
    })
    
    # Re-import class after path fix
    from gluefactory.datasets.product_pairs import ProductPairsDataset
    
    dataset = ProductPairsDataset(conf)
    loader = dataset.get_data_loader("train")
    
    print(f"Dataset has {len(loader)} batches")
    
    for batch in loader:
        print("Batch keys:", batch.keys())
        print("View0 keypoints shape:", batch["view0"]["keypoints"].shape)
        print("View0 descriptors shape:", batch["view0"]["descriptors"].shape)
        print("GT matches0:", batch["gt_matches0"])
        print("GT matches1:", batch["gt_matches1"])
        print("GT assignment shape:", batch["gt_assignment"].shape)
        break
