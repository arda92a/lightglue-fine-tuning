"""
Evaluate Pretrained LightGlue on Test Set and Log to W&B.

This script evaluates the pretrained DISK LightGlue model (before fine-tuning)
on the test set and logs metrics to W&B for comparison with fine-tuned results.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import wandb
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device
from gluefactory.utils.tools import set_seed, fork_rng


def evaluate_model(model, loader, device, loss_fn):
    """Run evaluation on a data loader."""
    model.eval()
    results = {}
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluation"):
            data = batch_to_device(data, device, non_blocking=True)
            pred = model(data)
            losses, metrics = loss_fn(pred, data)
            
            # Accumulate results
            numbers = {**metrics, **{"loss/" + k: v for k, v in losses.items()}}
            for k, v in numbers.items():
                if k not in results:
                    results[k] = []
                if isinstance(v, torch.Tensor):
                    v = v.mean().item()
                results[k].append(v)
    
    # Average results
    avg_results = {k: sum(v) / len(v) for k, v in results.items()}
    return avg_results


def main():
    # Load config
    config_path = Path(__file__).parent / "gluefactory/configs/product+lightglue_finetune.yaml"
    conf = OmegaConf.load(config_path)
    
    # Set seed
    set_seed(42)
    
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Initialize W&B
    wandb.init(
        project="lightglue-fine-tuning",
        entity="rempeople",
        name="pretrained_baseline",
        config=OmegaConf.to_container(conf, resolve=True),
        tags=["pretrained", "baseline"],
    )
    
    # Load dataset
    print("Loading dataset...")
    data_conf = conf.data
    dataset = get_dataset(data_conf.name)(data_conf)
    test_loader = dataset.get_data_loader("test")
    print(f"Test loader has {len(test_loader)} batches")
    
    # Load pretrained model (no fine-tuning weights)
    print("Loading pretrained LightGlue model...")
    model = get_model(conf.model.name)(conf.model).to(device)
    loss_fn = model.loss
    
    # Evaluate on test set
    print("Running evaluation on test set...")
    with fork_rng(seed=42):
        test_results = evaluate_model(model, test_loader, device, loss_fn)
    
    # Calculate F1 score
    precision = test_results.get("match_precision", 0)
    recall = test_results.get("match_recall", 0)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    test_results["f1_score"] = f1_score
    
    # Print results
    print("\n" + "="*50)
    print("PRETRAINED MODEL - TEST RESULTS")
    print("="*50)
    for k, v in sorted(test_results.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print("="*50)
    
    # Log to W&B
    wandb.log({
        **{f"test/{k}": v for k, v in test_results.items() if isinstance(v, (int, float))},
        "test/f1_score": f1_score,
    })
    
    # Log as summary metrics
    wandb.run.summary["test/precision"] = precision
    wandb.run.summary["test/recall"] = recall
    wandb.run.summary["test/f1_score"] = f1_score
    wandb.run.summary["test/accuracy"] = test_results.get("accuracy", 0)
    wandb.run.summary["test/loss"] = test_results.get("loss/total", 0)
    wandb.run.summary["model_type"] = "pretrained"
    
    wandb.finish()
    print("\nResults logged to W&B!")


if __name__ == "__main__":
    main()
