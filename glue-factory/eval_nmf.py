"""
Evaluate Neighborhood Metric Features Matching on Test Set and Log to W&B.

This script evaluates the original neighborhood metric features matching method
against the current ground truth and logs results to W&B for comparison.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import numpy as np
import wandb
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm


def calculate_metrics(predicted_matches, gt_matches):
    """
    Calculate precision, recall, F1 for matching.
    """
    # Convert to sets for comparison
    pred_set = set(tuple(m) for m in predicted_matches)
    gt_set = set(tuple(m) for m in gt_matches)
    
    if len(pred_set) == 0 and len(gt_set) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}
    
    if len(pred_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    
    if len(gt_set) == 0:
        return {"precision": 0.0, "recall": 1.0, "f1_score": 0.0}
    
    # True positives = intersection
    tp = len(pred_set & gt_set)
    
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {"precision": precision, "recall": recall, "f1_score": f1}


def main():
    # Paths
    config_path = Path("gluefactory/configs/product+lightglue_finetune.yaml")
    nmf_matches_path = Path("../data/lightglue_dataset_final/matches_neighborhood_metric_features.json")
    gt_matches_path = Path("../data/lightglue_dataset_final/matches.json")
    
    # Load config
    conf = OmegaConf.load(config_path)
    
    print("Loading datasets...")
    
    # Load NMF matches (predictions)
    with open(nmf_matches_path) as f:
        nmf_data = json.load(f)
    
    # Load GT matches
    with open(gt_matches_path) as f:
        gt_data = json.load(f)
    
    # Build lookup for NMF predictions
    nmf_lookup = {}
    for pair in nmf_data["pairs"]:
        key = (pair["image0"], pair["image1"])
        nmf_lookup[key] = pair["matches"]
    
    # Build pairs list from GT
    pairs = []
    for pair_info in gt_data["pairs"]:
        pairs.append({
            "image0": pair_info["image0"],
            "image1": pair_info["image1"],
            "matches": pair_info["matches"],
        })
    
    # Shuffle with seed and get test split (same as training)
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
        name="neighborhood_metric_features",
        config={
            "method": "Neighborhood Metric Features",
            "test_pairs": len(test_pairs),
        },
        tags=["nmf", "baseline"],
    )
    
    # Evaluate on test set
    print("\nEvaluating Neighborhood Metric Features on test set...")
    
    all_metrics = {
        "precision": [],
        "recall": [],
        "f1_score": [],
        "num_predicted": [],
        "num_gt": [],
    }
    
    for idx, pair_info in enumerate(tqdm(test_pairs, desc="NMF")):
        # Get NMF predictions for this pair
        key = (pair_info["image0"], pair_info["image1"])
        
        if key in nmf_lookup:
            predicted_matches = nmf_lookup[key]
        else:
            # Try reverse order
            key_rev = (pair_info["image1"], pair_info["image0"])
            if key_rev in nmf_lookup:
                # Reverse the match indices
                predicted_matches = [[m[1], m[0]] for m in nmf_lookup[key_rev]]
            else:
                predicted_matches = []
        
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
    print("NEIGHBORHOOD METRIC FEATURES - TEST RESULTS")
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
    wandb.run.summary["model_type"] = "neighborhood_metric_features"
    
    wandb.finish()
    print("\nResults logged to W&B!")


if __name__ == "__main__":
    main()
