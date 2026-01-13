"""
Evaluate BFMatcher (Brute Force Matcher) on Test Set and Log to W&B.

This script evaluates OpenCV's BFMatcher as a baseline comparison
against LightGlue models.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import numpy as np
import wandb
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm


def bf_match(desc0, desc1, ratio_thresh=0.75):
    """
    Brute Force Matching with Lowe's ratio test.
    
    Args:
        desc0: [M, D] descriptors for image 0
        desc1: [N, D] descriptors for image 1
        ratio_thresh: Lowe's ratio threshold
    
    Returns:
        matches: List of (idx0, idx1) tuples
    """
    import cv2
    
    # Ensure float32 for OpenCV
    desc0 = np.float32(desc0)
    desc1 = np.float32(desc1)
    
    # Create BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    # Find k=2 nearest neighbors
    if len(desc0) < 2 or len(desc1) < 2:
        return []
    
    matches = bf.knnMatch(desc0, desc1, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match in matches:
        if len(match) >= 2:
            m, n = match
            if m.distance < ratio_thresh * n.distance:
                good_matches.append((m.queryIdx, m.trainIdx))
    
    return good_matches


def calculate_metrics(predicted_matches, gt_matches):
    """
    Calculate precision, recall, F1 for matching.
    
    Args:
        predicted_matches: List of (idx0, idx1) predicted matches
        gt_matches: List of [idx0, idx1] ground truth matches
    
    Returns:
        dict with precision, recall, f1, accuracy
    """
    # Convert to sets for comparison
    pred_set = set(tuple(m) for m in predicted_matches)
    gt_set = set(tuple(m) for m in gt_matches)
    
    if len(pred_set) == 0 and len(gt_set) == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
        }
    
    if len(pred_set) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }
    
    if len(gt_set) == 0:
        return {
            "precision": 0.0,
            "recall": 1.0,  # No GT to recall
            "f1_score": 0.0,
        }
    
    # True positives = intersection
    tp = len(pred_set & gt_set)
    
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def main():
    # Paths
    config_path = Path("gluefactory/configs/product+lightglue_finetune.yaml")
    
    # Load config
    conf = OmegaConf.load(config_path)
    data_dir = Path("../data") / conf.data.data_dir
    
    print("Loading dataset...")
    
    # Load matches file
    matches_file = data_dir / conf.data.matches_file
    with open(matches_file) as f:
        matches_data = json.load(f)
    
    # Build pairs list
    pairs = []
    for pair_info in matches_data["pairs"]:
        pairs.append({
            "image0": pair_info["image0"],
            "image1": pair_info["image1"],
            "matches": pair_info["matches"],
        })
    
    # Shuffle with seed and get test split
    np.random.seed(42)
    np.random.shuffle(pairs)
    
    total = len(pairs)
    train_size = int(total * conf.data.train_ratio)
    val_size = int(total * conf.data.val_ratio)
    test_pairs = pairs[train_size + val_size:]
    
    print(f"Found {len(test_pairs)} test pairs")
    
    # Initialize W&B
    wandb.init(
        project="lightglue-fine-tuning",
        entity="rempeople",
        name="bfmatcher_baseline",
        config={
            "method": "BFMatcher",
            "ratio_thresh": 0.75,
            "test_pairs": len(test_pairs),
        },
        tags=["bfmatcher", "baseline"],
    )
    
    # Evaluate on test set
    print("\nEvaluating BFMatcher on test set...")
    
    all_metrics = {
        "precision": [],
        "recall": [],
        "f1_score": [],
        "num_predicted": [],
        "num_gt": [],
    }
    
    for idx, pair_info in enumerate(tqdm(test_pairs, desc="BFMatcher")):
        # Load annotations
        ann0_path = data_dir / "annotations" / f"{pair_info['image0']}.json"
        ann1_path = data_dir / "annotations" / f"{pair_info['image1']}.json"
        
        with open(ann0_path) as f:
            ann0 = json.load(f)
        with open(ann1_path) as f:
            ann1 = json.load(f)
        
        # Get embeddings
        emb0 = np.array([p["embedding"] for p in ann0["products"]], dtype=np.float32)
        emb1 = np.array([p["embedding"] for p in ann1["products"]], dtype=np.float32)
        
        # Run BFMatcher
        predicted_matches = bf_match(emb0, emb1, ratio_thresh=0.75)
        
        # Ground truth
        gt_matches = pair_info["matches"]
        
        # Calculate metrics
        metrics = calculate_metrics(predicted_matches, gt_matches)
        
        for k, v in metrics.items():
            all_metrics[k].append(v)
        all_metrics["num_predicted"].append(len(predicted_matches))
        all_metrics["num_gt"].append(len(gt_matches))
    
    # Calculate averages
    avg_precision = np.mean(all_metrics["precision"])
    avg_recall = np.mean(all_metrics["recall"])
    # F1 calculated from average P/R (same as LightGlue training)
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-8)
    
    # Print results
    print("\n" + "="*50)
    print("BFMATCHER - TEST RESULTS")
    print("="*50)
    print(f"  Precision:     {avg_precision:.4f}")
    print(f"  Recall:        {avg_recall:.4f}")
    print(f"  F1 Score:      {f1_score:.4f}")
    print(f"  Avg Predicted: {np.mean(all_metrics['num_predicted']):.1f}")
    print(f"  Avg GT:        {np.mean(all_metrics['num_gt']):.1f}")
    print("="*50)
    
    # Log to W&B - using same names as LightGlue training
    wandb.log({
        "test/match_precision": avg_precision,
        "test/match_recall": avg_recall,
        "test/f1_score": f1_score,
    })
    
    # Log as summary
    wandb.run.summary["test/match_precision"] = avg_precision
    wandb.run.summary["test/match_recall"] = avg_recall
    wandb.run.summary["test/f1_score"] = f1_score
    wandb.run.summary["model_type"] = "bfmatcher"
    
    wandb.finish()
    print("\nResults logged to W&B!")


if __name__ == "__main__":
    main()
