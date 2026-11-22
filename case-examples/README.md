# Case examples

All images in this folder show *Zhai duan miyu* 齋短秘語 (author's collection). The manuscript images are not covered by the MIT License of the code.

## `figures/`

- `figure1_folio_scores.png`: folio scores for ZD1–ZD94 in folio order; the ten highest in blue, outer nine leaves at either end shaded.
- `figure2_ZD2_ZD94_ZD19.png`: cells above the manuscript-wide 99th percentile on ZD2, ZD94 and ZD19 (median score).
- `figure3_ZD63_ZD51.png`: details of ZD63 (passages concealed with red pigment) and ZD51 (a correction).
- `overview_all_folios.jpg`: thumbnails of all ninety-four folios in folio order, with the top ten labelled in blue.

## `maps/` (ZD_01.jpg … ZD_94.jpg)

One three-panel map per folio:

1. **original**: the unprocessed scan;
2. **heat**: cell scores on a colour scale shared by all folios (manuscript-wide 95th to 99.9th percentile; lighter cells are not shown), so folios can be compared directly;
3. **cells above the manuscript-wide 99th percentile**: outlined in blue.

The dashed line marks the 6% margin excluded from scoring. The title line gives the folio score (99th percentile of interior cell scores), its rank, its rank after red-removal, the number of cells above the manuscript-wide 99th percentile, and the share of inked cells.

A high score marks visual heterogeneity, not wear. Colours do not represent temperature or any physical quantity. Water stains, creases, losses, show-through and red seals can all score high and must be checked against the folio itself.

## `folio_index.csv`

One row per folio: `score`, `rank`, `dered_score`, `dered_rank` (after red-removal), `cells_above_p99`, `share_near_margin` (share of those cells within five cells of the excluded margin), `ink_share_pct`, `outer_nine` (ZD1–9 or ZD86–94).
