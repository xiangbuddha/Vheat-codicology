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

from __future__ import annotations

import argparse
import glob
import os
import sys
import shutil
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import zscore
from torchvision import transforms

# Matplotlib is only used for 3D surface plots.
import matplotlib
import matplotlib.pyplot as plt


# -----------------------------
# PyTorch 2.6+ safe load support
# -----------------------------
def register_torch_safe_globals() -> None:
    """
    Register extra globals for torch.load(..., weights_only=False) under PyTorch 2.6 safety rules.
    """
    try:
        from yacs.config import CfgNode  # type: ignore

        torch.serialization.add_safe_globals([CfgNode])
        print("[Info] Registered yacs.config.CfgNode into PyTorch safe-load allowlist.")
    except Exception:
        # If yacs is not installed, ignore.
        print("[Warn] yacs not found. If your checkpoint contains CfgNode, torch.load may fail.")


# -----------------------------
# Config
# -----------------------------
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class ScanConfig:
    # Input / output
    input_dir: str
    output_dir: str

    # vHeat repo + weights
    vheat_repo: str
    weights_path: str

    # Preprocess
    img_size: int = 224

    # Thresholds
    z_thr: float = 1.3
    int_thr: float = 0.45

    # Reporting
    top_n: int = 30

    # Random seeds
    seed: int = 42


# -----------------------------
# Utilities
# -----------------------------
def imread_unicode(path: str) -> Optional[np.ndarray]:
    """
    Read image with support for non-ASCII paths (e.g., Chinese filenames) on Windows.
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_images(input_dir: str) -> List[str]:
    files: List[str] = []
    for ext in VALID_EXTENSIONS:
        files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    # Deterministic order
    return sorted(files)


def setup_matplotlib_fonts() -> None:
    """
    Optional: better Chinese font rendering on some Windows setups.
    Safe to ignore if unavailable.
    """
    try:
        matplotlib.rcParams["font.sans-serif"] = ["SimSun"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# -----------------------------
# vHeat loading
# -----------------------------
def import_vheat(vheat_repo: str):
    """
    Import vHeat model class from a local clone.
    """
    if not os.path.isdir(vheat_repo):
        raise FileNotFoundError(f"vHeat repo folder not found: {vheat_repo}")

    sys.path.append(vheat_repo)
    try:
        from vHeat.classification.models.vHeat import vHeat  # type: ignore

        return vHeat
    except Exception as e:
        raise ImportError(
            "Failed to import vHeat. Check that vheat_repo points to the vHeat repository root."
        ) from e


def build_model(vHeatClass, device: torch.device) -> torch.nn.Module:
    """
    Build vHeat-Base (the same architecture config you used).
    """
    model = vHeatClass(
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        dims=[128, 256, 512, 1024],
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3,
    )
    model.to(device)
    model.eval()
    return model


def load_weights(model: torch.nn.Module, weights_path: str, device: torch.device) -> None:
    """
    Load checkpoint weights. Supports:
    - raw state_dict
    - dict with key 'model'
    Strips 'module.' prefixes if present.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    if not isinstance(ckpt, dict):
        raise ValueError("Unexpected checkpoint format: expected a state_dict-like dict.")

    state_dict = {k.replace("module.", ""): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("[Info] Weights loaded.")
    if missing:
        print(f"[Warn] Missing keys (strict=False): {len(missing)}")
    if unexpected:
        print(f"[Warn] Unexpected keys (strict=False): {len(unexpected)}")


# -----------------------------
# Core anomaly analysis
# -----------------------------
def make_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def analyze_one_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    cfg: ScanConfig,
    out_2d: str,
    out_3d: str,
) -> Optional[Dict[str, float]]:
    """
    Analyze a single image and save artifacts (2D overlay, 3D surface).
    Returns a dict of metrics for reporting.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]

    # 1) Load image
    img_bgr = imread_unicode(image_path)
    if img_bgr is None:
        print(f"[Error] Failed to read: {image_path}")
        return None

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2) Preprocess
    tfm = make_transform(cfg.img_size)
    x = tfm(img_rgb).unsqueeze(0).to(device)

    # 3) Feature extraction -> heatmap
    with torch.no_grad():
        feats = model.forward_features(x)  # expected 4D: [B, C, Hf, Wf]

    if feats is None:
        print(f"[Error] forward_features returned None: {base}")
        return None

    # Channel-mean heatmap (thermodynamics-inspired "diffusion field" projection)
    heat = feats.mean(dim=1, keepdim=True)  # [B, 1, Hf, Wf]
    heat = F.interpolate(heat, size=(h, w), mode="bilinear", align_corners=False)
    heat = heat.squeeze().detach().cpu().numpy()  # [H, W]

    # Normalize to [0,1] for intensity-based thresholding
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    # 4) Outlier mask (z-score) + intensity threshold
    heat_z = zscore(heat, axis=None)
    mask = (heat_z > cfg.z_thr) & (heat_norm > cfg.int_thr)
    mask_u8 = mask.astype(np.uint8)

    # 5) Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6) Connected components + min area filter
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

    min_area = max(1, int(0.005 * h * w))  # keep regions >= 0.5% of image area
    kept_regions = 0
    clean_mask_stats = np.zeros_like(mask_clean)

    for i in range(1, num_labels):  # skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask_stats[labels == i] = 1
            kept_regions += 1

    # 7) Metrics
    mean_anomaly_index = float(heat_norm.mean())
    max_anomaly_intensity = float(heat_norm.max())
    anomaly_area = int(clean_mask_stats.sum())
    anomaly_pct = float(anomaly_area / (h * w) * 100.0)
    pattern_complexity = float(np.std(heat_norm))

    print(
        f"[OK] {base} | mean={mean_anomaly_index:.4f} | regions={kept_regions} | area={anomaly_pct:.2f}%"
    )

    # 8) Save 2D overlay
    heat_color = cv2.applyColorMap((255 * heat_norm).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.55, heat_color, 0.45, 0.0)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    overlay_path = os.path.join(out_2d, f"{base}_overlay.png")
    cv2.imencode(".png", overlay)[1].tofile(overlay_path)

    # 9) Save 3D surface plot (z-score downsampled)
    scale = max(1, int(max(h, w) / 256))
    heat_z_small = heat_z[::scale, ::scale]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(np.arange(heat_z_small.shape[1]), np.arange(heat_z_small.shape[0]))
    ax.plot_surface(X, Y, heat_z_small, cmap="viridis", rstride=1, cstride=1, alpha=0.9, edgecolor="none")
    ax.set_title(f"vHeat Anomaly Surface: {base}")
    ax.set_zlabel("Z-Score")
    ax.view_init(elev=50, azim=-60)

    surface_path = os.path.join(out_3d, f"{base}_3d_surface.png")
    plt.savefig(surface_path)
    plt.close(fig)

    return {
        "filename": base,
        "anomaly_percentage": anomaly_pct,
        "pattern_complexity": pattern_complexity,
        "mean_anomaly_index": mean_anomaly_index,
        "max_anomaly_intensity": max_anomaly_intensity,
        "highlighted_regions": float(kept_regions),
    }


# -----------------------------
# Reporting + sorting
# -----------------------------
def generate_report(results: List[Dict[str, float]], report_dir: str, top_n: int) -> List[Dict[str, float]]:
    """
    Print + save a summary report. Returns list sorted by mean anomaly (high->low).
    """
    if not results:
        print("[Warn] No images processed successfully; report skipped.")
        return []

    ensure_dir(report_dir)
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append(f"Document Set Summary Report  (N={len(results)})")
    lines.append("=" * 60)
    lines.append("")

    # 1) By anomaly area
    by_area = sorted(results, key=lambda x: x["anomaly_percentage"], reverse=True)
    lines.append(f"--- Top {top_n} by anomaly area percentage (damage/alteration) ---")
    for i, r in enumerate(by_area[:top_n], start=1):
        lines.append(f"{i:>3d}. {r['filename']}  | area={r['anomaly_percentage']:.2f}%")
    lines.append("")

    # 2) By mean anomaly (overall aging/yellowing)
    by_mean = sorted(results, key=lambda x: x["mean_anomaly_index"], reverse=True)
    lines.append(f"--- Top {top_n} by mean anomaly index (global aging/yellowing) ---")
    for i, r in enumerate(by_mean[:top_n], start=1):
        lines.append(f"{i:>3d}. {r['filename']}  | mean={r['mean_anomaly_index']:.4f}")
    lines.append("")

    # 3) By pattern complexity
    by_std = sorted(results, key=lambda x: x["pattern_complexity"], reverse=True)
    lines.append(f"--- Top {top_n} by heatmap pattern complexity (std of normalized heat) ---")
    for i, r in enumerate(by_std[:top_n], start=1):
        lines.append(f"{i:>3d}. {r['filename']}  | std={r['pattern_complexity']:.4f}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("End of report.")

    # print
    print("\n".join(lines))

    # save
    report_path = os.path.join(report_dir, "summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[Info] Report saved: {report_path}")

    return by_mean


def copy_sorted_overlays(sorted_list: List[Dict[str, float]], src_2d: str, dst_sorted: str) -> None:
    """
    Copy overlays into a new folder with rank prefix.
    """
    ensure_dir(dst_sorted)
    print(f"\n[Info] Copying {len(sorted_list)} overlays to: {dst_sorted}")

    for idx, r in enumerate(sorted_list, start=1):
        fn = r["filename"]
        src = os.path.join(src_2d, f"{fn}_overlay.png")
        dst = os.path.join(dst_sorted, f"{idx:03d}_{fn}_overlay.png")

        if not os.path.exists(src):
            print(f"[Warn] Missing overlay: {src}")
            continue

        shutil.copy(src, dst)

    print("[Info] Sorted overlay copy done.")


# -----------------------------
# Main
# -----------------------------
def parse_args() -> ScanConfig:
    p = argparse.ArgumentParser(
        description="Batch anomaly scan with vHeat features (thermodynamics-inspired heatmap outlier detection)."
    )

    p.add_argument("--input_dir", required=True, help="Folder containing input images.")
    p.add_argument("--output_dir", required=True, help="Folder to store outputs.")
    p.add_argument("--vheat_repo", required=True, help="Local path to the cloned vHeat repository.")
    p.add_argument("--weights", required=True, help="Path to vHeat weight file (.pth).")

    p.add_argument("--img_size", type=int, default=224, help="Input resize for the backbone.")
    p.add_argument("--z_thr", type=float, default=1.3, help="Z-score threshold for anomaly outliers.")
    p.add_argument("--int_thr", type=float, default=0.45, help="Intensity threshold on normalized heatmap [0,1].")
    p.add_argument("--top_n", type=int, default=30, help="Top-N in summary report.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    a = p.parse_args()

    return ScanConfig(
        input_dir=a.input_dir,
        output_dir=a.output_dir,
        vheat_repo=a.vheat_repo,
        weights_path=a.weights,
        img_size=a.img_size,
        z_thr=a.z_thr,
        int_thr=a.int_thr,
        top_n=a.top_n,
        seed=a.seed,
    )


def main() -> None:
    cfg = parse_args()

    # Seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    setup_matplotlib_fonts()
    register_torch_safe_globals()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    if not os.path.isdir(cfg.input_dir):
        raise FileNotFoundError(f"Input folder not found: {cfg.input_dir}")

    # Output folders
    out_2d = os.path.join(cfg.output_dir, "2D_Overlays")
    out_3d = os.path.join(cfg.output_dir, "3D_Surface_Plots")
    out_sorted = os.path.join(cfg.output_dir, "Sorted_Mean_Anomaly_High_to_Low")
    out_reports = os.path.join(cfg.output_dir, "Analysis_Reports")

    for d in [out_2d, out_3d, out_sorted, out_reports]:
        ensure_dir(d)

    # Import + model
    vHeatClass = import_vheat(cfg.vheat_repo)
    model = build_model(vHeatClass, device)
    load_weights(model, cfg.weights_path, device)

    # Scan
    images = list_images(cfg.input_dir)
    if not images:
        print("[Warn] No images found in input_dir.")
        return

    print(f"\n[Info] Found {len(images)} images. Starting scan...\n")
    results: List[Dict[str, float]] = []

    for i, path in enumerate(images, start=1):
        print(f"[{i:>4d}/{len(images)}] {os.path.basename(path)}")
        r = analyze_one_image(model, path, device, cfg, out_2d, out_3d)
        if r is not None:
            results.append(r)

    print("\n[Info] Scan complete.")

    # Report + sorted copy
    sorted_by_mean = generate_report(results, out_reports, top_n=cfg.top_n)
    if sorted_by_mean:
        copy_sorted_overlays(sorted_by_mean, out_2d, out_sorted)

    print("\n[Info] Output structure:")
    print(f"  1) 2D overlays:    {out_2d}")
    print(f"  2) 3D surfaces:    {out_3d}")
    print(f"  3) Sorted overlays:{out_sorted}")
    print(f"  4) Reports:        {out_reports}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
