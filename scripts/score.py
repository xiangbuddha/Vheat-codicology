"""Step 2: score every cell against the rest of the codex and write folio scores.

Each cell is scored by its distance to the most similar of 40,000 cells drawn (seed 2026) from all
other folios (PatchCore-style nearest neighbour). The folio itself and the two folios on either side are
excluded from the reference. A folio's score is the 99th percentile of its interior cell scores after a
6% margin is excluded on every edge.

Usage:
  python score.py --feats feats.npz --out scores.npz --csv folio_scores.csv
"""
import argparse, csv
import numpy as np, torch

R, C = 84, 56
MR, MC = int(round(0.06 * R)), int(round(0.06 * C))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="feats.npz"); ap.add_argument("--out", default="scores.npz")
    ap.add_argument("--csv", default="folio_scores.csv"); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    d = np.load(a.feats); pages = sorted(int(k) for k in d.files)
    F = {p: torch.from_numpy(d[str(p)].astype(np.float32)) for p in pages}
    g = torch.Generator().manual_seed(2026); scores = {}
    for p in pages:
        bank = torch.cat([F[q] for q in pages if abs(q - p) > 2])
        bank = bank[torch.randperm(len(bank), generator=g)[:40000]]
        scores[p] = torch.cat([torch.cdist(F[p][i:i + 1200], bank).min(1).values
                               for i in range(0, len(F[p]), 1200)]).numpy()
    np.savez(a.out, **{str(p): scores[p] for p in pages})
    main_score = {p: float(np.percentile(scores[p].reshape(R, C)[MR:R - MR, MC:C - MC], 99)) for p in pages}
    order = sorted(pages, key=lambda p: -main_score[p]); rank = {p: i + 1 for i, p in enumerate(order)}
    with open(a.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["folio", "score_p99", "rank"])
        for p in pages: w.writerow([f"ZD{p}", round(main_score[p], 5), rank[p]])
    print("top ten:", [f"ZD{p}" for p in order[:10]])


if __name__ == "__main__":
    main()
