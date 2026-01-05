#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Thermodynamics-inspired (heat-conduction) anomaly detection with vHeat features.

Core idea
---------
We treat the vHeat backbone's intermediate feature map as a "diffusion field" that
encodes structural / material irregularities. We then derive a heatmap by channel-mean
pooling, upsample it to the original resolution, and detect abnormal regions via:

1) Global z-score thresholding (statistical outliers)
2) Intensity thresholding on [0, 1] normalized heatmap (saliency / confidence)
3) Morphological closing to reduce speckle noise
4) Connected components + minimum area filtering to keep meaningful regions

Outputs
-------
- 2D overlay heatmaps with contour outlines
- 3D surface plots of z-score heatmap
- A summary text report ranking images by:
  (a) anomaly area percentage (damage/alterations)
  (b) mean anomaly index (overall aging/yellowing)
  (c) pattern complexity (std of normalized heatmap)
- A sorted folder of 2D overlays (by mean anomaly)

Notes
-----
- This script assumes you have a local clone of the vHeat repository and a compatible weight file.
- For PyTorch 2.6+ safe loading: if the checkpoint includes YACS CfgNode, we register it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
import glob
import matplotlib.pyplot as plt
from scipy.stats import zscore
from torchvision import transforms
import shutil
import warnings

# =================================================================
# 1. GLOBAL CONFIGURATION (User: Modify these paths)
# =================================================================
CONFIG = {
    "WEIGHTS_PATH": r"D:\models\vHeat\vHeat_base.pth",
    "INPUT_DIR": r"D:\models\vHeat\Unnamed Manuscripts 001",
    "OUTPUT_ROOT": r"D:\models\vHeat\Research_Results",
    "VHEAT_SRC_PATH": r"D:\models\vHeat",  # Path to vHeat source code
    "HYPERPARAMS": {
        "IMG_SIZE": 224,
        "Z_THRESHOLD": 1.3,      # Sensitivity for anomaly detection (lower = more sensitive)
        "INTENSITY_MIN": 0.45,   # Minimum brightness to consider as anomaly
        "MIN_AREA_PCT": 0.005,   # Ignore clusters smaller than 0.5% of image area
    },
    "TOP_N": 30                  # Number of most anomalous images to rank
}

# --- Environment Setup ---
sys.path.append(CONFIG["VHEAT_SRC_PATH"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# =================================================================
# 2. MODEL INITIALIZATION
# =================================================================
def load_vheat_model():
    """
    Initializes the vHeat architecture and loads pre-trained weights.
    vHeat is a State Space Model (SSM) optimized for visual feature extraction.
    """
    try:
        from vHeat.classification.models.vHeat import vHeat
        model = vHeat(
            patch_size=4, in_chans=3, num_classes=1000,
            dims=[128, 256, 512, 1024], depths=[2, 2, 18, 2], drop_path_rate=0.3
        )
        
        # Load weights with compatibility for PyTorch 2.6+
        checkpoint = torch.load(CONFIG["WEIGHTS_PATH"], map_location=device, weights_only=False)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        print(f"[INFO] Model loaded successfully on {device}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
        return None

# =================================================================
# 3. CORE RESEARCH ALGORITHM
# =================================================================
def process_analysis(model, image_path, dirs):
    """
    Research Pipeline:
    1. Feature Extraction via Vision State Space Model.
    2. Heatmap generation from feature channel averaging.
    3. Anomaly isolation using Z-Score statistical analysis.
    """
    try:
        # Read image supporting Unicode paths
        img_raw = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_raw is None: return None
        
        h, w = img_raw.shape[:2]
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # Preprocessing (Normalization based on ImageNet stats)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((CONFIG["HYPERPARAMS"]["IMG_SIZE"], CONFIG["HYPERPARAMS"]["IMG_SIZE"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(img_rgb).unsqueeze(0).to(device)

        # Feature inference
        with torch.no_grad():
            features = model.forward_features(input_tensor)
        
        # RESEARCH NOTE: Compress channels to get spatial activation map
        heatmap = features.mean(dim=1, keepdim=True)
        heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze().cpu().numpy()
        
        # Statistical Analysis
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_z = zscore(heatmap, axis=None)

        # RESEARCH NOTE: Define anomalies as outliers (Z > threshold) with high activation
        mask = (heatmap_z > CONFIG["HYPERPARAMS"]["Z_THRESHOLD"]) & \
               (heatmap_norm > CONFIG["HYPERPARAMS"]["INTENSITY_MIN"])
        
        # Morphology to remove noise and fill gaps
        kernel = np.ones((5,5), np.uint8)
        mask_clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # Connected component analysis for area calculation
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean)
        min_area = int(CONFIG["HYPERPARAMS"]["MIN_AREA_PCT"] * h * w)
        anomaly_area = sum([stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area])
        
        # Visualization: 2D Overlay
        heatmap_jet = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
        overlay_res = cv2.addWeighted(img_raw, 0.6, heatmap_jet, 0.4, 0)
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imencode('.png', overlay_res)[1].tofile(os.path.join(dirs['2d'], f"{filename}_overlay.png"))

        # Optional: Save 3D Surface for depth analysis
        generate_3d_plot(heatmap_z, filename, dirs['3d'])

        return {
            'filename': filename,
            'anomaly_pct': (anomaly_area / (h * w)) * 100,
            'mean_score': float(heatmap_norm.mean()),
            'complexity': float(np.std(heatmap_norm))
        }
    except Exception as e:
        print(f"[SKIP] Error processing {image_path}: {e}")
        return None

def generate_3d_plot(data, filename, out_dir):
    # Downsample for performance
    stride = max(1, int(max(data.shape) / 200))
    data_small = data[::stride, ::stride]
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(np.arange(data_small.shape[1]), np.arange(data_small.shape[0]))
    ax.plot_surface(x, y, data_small, cmap='magma', edgecolor='none')
    ax.set_title(f"3D Outlier Mapping: {filename}")
    plt.savefig(os.path.join(out_dir, f"{filename}_3d.png"))
    plt.close(fig)

# =================================================================
# 4. EXECUTION ENGINE
# =================================================================
def main():
    # Folder Management
    base_out = CONFIG["OUTPUT_ROOT"]
    dirs = {
        '2d': os.path.join(base_out, "Visual_2D_Overlays"),
        '3d': os.path.join(base_out, "Visual_3D_Surfaces"),
        'rank': os.path.join(base_out, "Top_Anomalies_Ranked"),
        'log': os.path.join(base_out, "Analysis_Reports")
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    # Initialize Model
    vheat_model = load_vheat_model()
    if vheat_model is None: return

    # File Discovery
    exts = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(CONFIG["INPUT_DIR"], f"*{e}")))
    
    print(f"[START] Processing {len(files)} documents...")

    all_stats = []
    for i, f_path in enumerate(files):
        res = process_analysis(vheat_model, f_path, dirs)
        if res:
            all_stats.append(res)
            print(f"[{i+1}/{len(files)}] {res['filename']} | Anomaly Area: {res['anomaly_pct']:.2f}%")

    # Research Synthesis & Ranking
    if all_stats:
        # Sort by Mean Anomaly Score (Global degradation)
        ranked_list = sorted(all_stats, key=lambda x: x['mean_score'], reverse=True)
        
        # Export Ranked Files
        for rank, item in enumerate(ranked_list):
            src = os.path.join(dirs['2d'], f"{item['filename']}_overlay.png")
            dst = os.path.join(dirs['rank'], f"Rank_{rank+1:03d}_{item['filename']}.png")
            if os.path.exists(src): shutil.copy(src, dst)
        
        # Final Summary Report
        report_path = os.path.join(dirs['log'], "research_summary.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("vHeat Document Research: Anomaly Detection Report\n")
            f.write("="*50 + "\n\n")
            f.write(f"Top {CONFIG['TOP_N']} Anomalous Documents:\n")
            for r in ranked_list[:CONFIG["TOP_N"]]:
                f.write(f"- {r['filename']}: Score {r['mean_score']:.4f}, Area {r['anomaly_pct']:.2f}%\n")
        
    print(f"\n[COMPLETE] Results saved to: {base_out}")

if __name__ == "__main__":
    main()

