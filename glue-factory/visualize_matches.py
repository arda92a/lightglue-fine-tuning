"""
Visualize Product Matching: Pretrained vs Fine-tuned LightGlue

This script visualizes matched product boxes on test image pairs,
comparing pretrained and fine-tuned model predictions.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device
from gluefactory.utils.tools import set_seed


def generate_colors(n):
    """Generate n distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        # Convert HSV to RGB
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def draw_matches_on_images(img0, img1, boxes0, boxes1, matches, title=""):
    """
    Draw matched boxes on both images with connecting colors.
    
    Args:
        img0, img1: PIL Images
        boxes0, boxes1: List of [x1, y1, x2, y2] boxes
        matches: List of (idx0, idx1) match pairs
        title: Title for the visualization
    """
    # Create side-by-side image
    w0, h0 = img0.size
    w1, h1 = img1.size
    
    # Resize to same height if needed
    target_h = max(h0, h1)
    if h0 != target_h:
        ratio = target_h / h0
        img0 = img0.resize((int(w0 * ratio), target_h))
        w0, h0 = img0.size
        boxes0 = [[int(x * ratio) for x in box] for box in boxes0]
    if h1 != target_h:
        ratio = target_h / h1
        img1 = img1.resize((int(w1 * ratio), target_h))
        w1, h1 = img1.size
        boxes1 = [[int(x * ratio) for x in box] for box in boxes1]
    
    # Create combined image
    combined = Image.new('RGB', (w0 + w1 + 20, target_h + 50), (255, 255, 255))
    combined.paste(img0, (0, 50))
    combined.paste(img1, (w0 + 20, 50))
    
    draw = ImageDraw.Draw(combined)
    
    # Draw title
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), title, fill=(0, 0, 0), font=font)
    
    # Generate colors for matches
    colors = generate_colors(len(matches))
    
    # Draw matches
    for i, (idx0, idx1) in enumerate(matches):
        color = colors[i]
        
        # Draw box on image 0
        box0 = boxes0[idx0]
        draw.rectangle([box0[0], box0[1] + 50, box0[2], box0[3] + 50], outline=color, width=5)
        
        # Draw box on image 1 (offset by w0 + 20)
        box1 = boxes1[idx1]
        draw.rectangle([box1[0] + w0 + 20, box1[1] + 50, box1[2] + w0 + 20, box1[3] + 50], 
                       outline=color, width=5)
        
        # Draw connecting line
        center0 = ((box0[0] + box0[2]) // 2, (box0[1] + box0[3]) // 2 + 50)
        center1 = ((box1[0] + box1[2]) // 2 + w0 + 20, (box1[1] + box1[3]) // 2 + 50)
        draw.line([center0, center1], fill=color, width=4)
    
    # Add match count
    draw.text((w0 + w1 - 150, 10), f"Matches: {len(matches)}", fill=(0, 0, 0), font=font)
    
    return combined


def load_annotations(data_dir, image_name):
    """Load product annotations for an image."""
    ann_path = data_dir / "annotations" / f"{image_name}.json"
    with open(ann_path) as f:
        ann = json.load(f)
    
    boxes = []
    embeddings = []
    for product in ann["products"]:
        boxes.append(product["bbox"])
        embeddings.append(product["embedding"])
    
    return boxes, np.array(embeddings, dtype=np.float32)


def run_model(model, data, device):
    """Run model on a single pair and get matches."""
    model.eval()
    with torch.no_grad():
        data = batch_to_device(data, device, non_blocking=True)
        pred = model(data)
    
    matches0 = pred["matches0"][0].cpu().numpy()
    scores0 = pred["matching_scores0"][0].cpu().numpy()
    
    # Get valid matches (not -1)
    valid = matches0 != -1
    matches = []
    for i in range(len(matches0)):
        if valid[i]:
            matches.append((i, matches0[i]))
    
    return matches, scores0


def main():
    set_seed(42)
    
    # Paths - outputs and data are one level up from glue-factory/glue-factory
    config_path = Path("gluefactory/configs/product+lightglue_finetune.yaml")
    checkpoint_path = Path("../outputs/training/product_lightglue_v4/checkpoint_best.tar")
    output_dir = Path("../visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Load config
    conf = OmegaConf.load(config_path)
    data_dir = Path("../data") / conf.data.data_dir
    
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load dataset to get test pairs
    print("Loading dataset...")
    dataset = get_dataset(conf.data.name)(conf.data)
    test_pairs = dataset.pairs["test"]
    print(f"Found {len(test_pairs)} test pairs")
    
    # Load pretrained model
    print("Loading pretrained model...")
    pretrained_model = get_model(conf.model.name)(conf.model).to(device)
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    finetuned_model = get_model(conf.model.name)(conf.model).to(device)
    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        finetuned_model.load_state_dict(checkpoint["model"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        return
    
    # Process each test pair
    print(f"\nProcessing {len(test_pairs)} test pairs...")
    
    for idx, pair_info in enumerate(test_pairs):
        print(f"\nPair {idx + 1}/{len(test_pairs)}: {pair_info['image0']} <-> {pair_info['image1']}")
        
        # Load images
        img0_path = data_dir / "images" / f"{pair_info['image0']}.jpg"
        img1_path = data_dir / "images" / f"{pair_info['image1']}.jpg"
        
        if not img0_path.exists():
            img0_path = img0_path.with_suffix('.png')
        if not img1_path.exists():
            img1_path = img1_path.with_suffix('.png')
        
        img0 = Image.open(img0_path).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')
        
        # Load annotations
        boxes0, emb0 = load_annotations(data_dir, pair_info['image0'])
        boxes1, emb1 = load_annotations(data_dir, pair_info['image1'])
        
        # Prepare data for model
        kpts0 = torch.tensor([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in boxes0], dtype=torch.float32)
        kpts1 = torch.tensor([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in boxes1], dtype=torch.float32)
        desc0 = torch.tensor(emb0, dtype=torch.float32)
        desc1 = torch.tensor(emb1, dtype=torch.float32)
        
        data = {
            "keypoints0": kpts0.unsqueeze(0),
            "keypoints1": kpts1.unsqueeze(0),
            "descriptors0": desc0.unsqueeze(0),
            "descriptors1": desc1.unsqueeze(0),
            "view0": {"image_size": torch.tensor([[img0.width, img0.height]])},
            "view1": {"image_size": torch.tensor([[img1.width, img1.height]])},
        }
        
        # Run pretrained model
        pretrained_matches, _ = run_model(pretrained_model, data, device)
        print(f"  Pretrained: {len(pretrained_matches)} matches")
        
        # Run fine-tuned model
        finetuned_matches, _ = run_model(finetuned_model, data, device)
        print(f"  Fine-tuned: {len(finetuned_matches)} matches")
        
        # Ground truth matches
        gt_matches = pair_info['matches']
        print(f"  Ground truth: {len(gt_matches)} matches")
        
        # Create visualizations
        vis_pretrained = draw_matches_on_images(
            img0.copy(), img1.copy(), boxes0, boxes1, pretrained_matches,
            title=f"PRETRAINED - Pair {idx + 1}"
        )
        
        vis_finetuned = draw_matches_on_images(
            img0.copy(), img1.copy(), boxes0, boxes1, finetuned_matches,
            title=f"FINE-TUNED - Pair {idx + 1}"
        )
        
        vis_gt = draw_matches_on_images(
            img0.copy(), img1.copy(), boxes0, boxes1, gt_matches,
            title=f"GROUND TRUTH - Pair {idx + 1}"
        )
        
        # Save visualizations separately
        vis_pretrained.save(output_dir / f"pair_{idx:03d}_pretrained.jpg", quality=95)
        vis_finetuned.save(output_dir / f"pair_{idx:03d}_finetuned.jpg", quality=95)
        vis_gt.save(output_dir / f"pair_{idx:03d}_groundtruth.jpg", quality=95)
    
    print(f"\n✅ Visualizations saved to: {output_dir}")
    print(f"   - {len(test_pairs)} × 3 images (pretrained, finetuned, groundtruth)")


if __name__ == "__main__":
    main()
