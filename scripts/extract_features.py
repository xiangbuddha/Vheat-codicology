"""Step 1: extract cell features for every folio (v3 procedure, parameters fixed before results were examined).

Each folio image is resized to 896 x 1344 px and cut into 24 tiles of 224 x 224 px. Features from the
second and third stages of vHeat-base are combined, smoothed (3 x 3 average) and pooled (2 x 2) into a grid
of 84 x 56 cells per folio, then projected to 192 dimensions with a fixed random projection (seed 1234).

Usage:
  python extract_features.py --images DIR --vheat_repo PATH --weights vHeat_base.pth --out feats.npz [--device auto|cpu|cuda]

--device auto (default) uses a CUDA GPU when one is available. Use --device cpu to reproduce the stored
features bit for bit; GPU arithmetic differs from CPU arithmetic in the last decimal places.
Image files are expected to be named ZD_<n>.png (n = folio number); other files are ignored.
"""
import argparse, os, re, time
import cv2, numpy as np, torch
import torch.nn.functional as F
from vheat_loader import build_vheat_base, pick_device


def folio(name):
    m = re.fullmatch(r"ZD_(\d+)\.png", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--vheat_repo", required=True)
    ap.add_argument("--weights", required=True); ap.add_argument("--out", default="feats.npz")
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--device", default="auto")
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    dev = pick_device(a.device); print("device:", dev)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    m = build_vheat_base(a.vheat_repo, a.weights, dev)
    st = {}
    m.layers[1][-2].register_forward_hook(lambda mod, i, o: st.__setitem__(1, o))
    m.layers[2][-2].register_forward_hook(lambda mod, i, o: st.__setitem__(2, o))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
    g = torch.Generator().manual_seed(1234); proj = (torch.randn(768, 192, generator=g) / np.sqrt(192)).to(dev)  # drawn on the CPU for every device
    files = sorted([f for f in os.listdir(a.images) if folio(f)], key=folio)
    out, t = {}, time.time()
    for f in files:
        bgr = cv2.imdecode(np.fromfile(os.path.join(a.images, f), dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(cv2.resize(bgr, (896, 1344), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.
        tiles = x.unfold(1, 224, 224).unfold(2, 224, 224).permute(1, 2, 0, 3, 4).reshape(24, 3, 224, 224).to(dev)
        with torch.no_grad():
            m.forward_features((tiles - mean) / std)
        b = F.interpolate(st[2], size=(28, 28), mode="bilinear", align_corners=False)
        feat = torch.cat([st[1], b], 1)
        feat = F.avg_pool2d(F.avg_pool2d(feat, 3, 1, 1), 2, 2)                       # 24, 768, 14, 14
        feat = feat.reshape(6, 4, 768, 14, 14).permute(2, 0, 3, 1, 4).reshape(768, 84, 56)
        v = F.normalize(feat.permute(1, 2, 0).reshape(-1, 768), dim=1) @ proj
        out[str(folio(f))] = F.normalize(v, dim=1).cpu().numpy().astype(np.float16)
    np.savez(a.out, **out)
    print(len(out), "folios", round(time.time() - t, 1), "s ->", a.out)


if __name__ == "__main__":
    main()
