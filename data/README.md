# Cell scores

- `cell_scores.npz`: cell scores for ZD1–ZD94 from `scripts/score.py` (one array of 84 × 56 = 4,704 cells per folio, keyed by folio number).
- `cell_scores_dered.npz`: the same after red-removal.

With these files the maps can be redrawn without rerunning the model:

```bash
cd scripts
python make_maps.py --images /path/to/folio_images --scores ../data/cell_scores.npz --scores_dered ../data/cell_scores_dered.npz --out maps
```

A complete rerun of `extract_features.py` and `score.py` reproduces `cell_scores.npz` exactly.
