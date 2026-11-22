"""Build vHeat-base with the configuration its checkpoint was trained with and load the weights strictly.

The released vHeat-base checkpoint uses layer scale (1e-5) and post-norm. The model must be built
with layer_scale=1e-5 and post_norm=True and loaded with strict=True; a non-strict load would drop
the 48 layer-scale parameters without raising an error.
"""
import sys
import torch


def build_vheat_base(vheat_repo: str, weights: str, device: str = "cpu"):
    sys.path.insert(0, vheat_repo)
    from vHeat.classification.models.vHeat import vHeat  # from https://github.com/MzeroMiko/vHeat

    model = vHeat(patch_size=4, in_chans=3, num_classes=1000,
                  dims=[128, 256, 512, 1024], depths=[2, 2, 18, 2],
                  drop_path_rate=0.3, post_norm=True, layer_scale=1e-5)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    print(model.load_state_dict(state, strict=True))  # must report: <All keys matched successfully>
    return model.to(device).eval()
