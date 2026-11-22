"""Build vHeat-base with the configuration its checkpoint was trained with and load the weights strictly.

The released vHeat-base checkpoint uses layer scale (1e-5) and post-norm. The model must be built
with layer_scale=1e-5 and post_norm=True and loaded with strict=True; a non-strict load would drop
the 48 layer-scale parameters without raising an error.
"""
import sys
import torch


def pick_device(name: str = "auto") -> torch.device:
    """'auto' uses the first CUDA GPU if one is available, otherwise the CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but no GPU is available to PyTorch; use --device cpu or install a CUDA build of PyTorch.")
    return torch.device(name)


def build_vheat_base(vheat_repo: str, weights: str, device="cpu"):
    sys.path.insert(0, vheat_repo)
    from vHeat.classification.models.vHeat import vHeat  # from https://github.com/MzeroMiko/vHeat

    model = vHeat(patch_size=4, in_chans=3, num_classes=1000,
                  dims=[128, 256, 512, 1024], depths=[2, 2, 18, 2],
                  drop_path_rate=0.3, post_norm=True, layer_scale=1e-5)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    print(model.load_state_dict(state, strict=True))  # must report: <All keys matched successfully>
    return model.to(device).eval()
