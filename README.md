## Overview

This repository provides a reproducible analysis pipeline, example visualizations, and a short case study for the project:  
**Computational Codicology via Thermal Diffusion: Visualizing Material Anomalies in Lanten Religious Manuscripts**

The project explores how heat-conduction–inspired vision models can be adapted for the study of manuscript materiality. By applying the **vHeat** visual backbone to historical document images, the workflow visualizes and quantifies page-level material anomalies such as ink diffusion, wear patterns, stains, and layout-induced texture variation.

All computational results are intended to **support, not replace**, traditional philological and codicological analysis.

---

## Case Studies & Reports

We have performed a detailed analysis on a 19th-century Vietnamese Yao (Lanten / Kim Mun) religious manuscript: *Zhai duan (Wang) miyu* (齋短（亡）秘語).

*  **[Full Case Study Report](./Report)**: Click here to view the interpretive summary connecting computational results with codicological observations.
*  **[Core Analysis Script](./scripts)**: Click here to view the Python engine used for this research.

### Representative Figures

> **Note:** The figures below demonstrate the model's ability to map physical degradation and ink penetration.

**Case example: 2D overlay (Heatmap + Contours)** 

**Case example: 3D surface plot (Z-Score Landscape)** 

---

## Repository Structure

```text
vheat-codicology/
├── README.md           # Project documentation
├── scripts/            
│   └── anomaly_scan_vheat.py  # Main batch processing engine
├── examples/           
│   ├── figures/        # Visualization outputs (2D/3D plots)
│   └── mini_case_study.md # Detailed experimental report
├── requirements.txt    # Dependency list
├── LICENSE             # License info
└── .gitignore

```

---

## Quick Start

### 1. Environment Setup

```bash
git clone [https://github.com/your-username/vheat-codicology.git](https://github.com/your-username/vheat-codicology.git)
cd vheat-codicology
pip install -r requirements.txt

```

### 2. Dependencies

1. Clone the original [vHeat Repository](https://github.com/MzeroMiko/vHeat).
2. Download the pretrained weights (e.g., `vHeat_base.pth`).

### 3. Run the Research Script

You can use the provided script to replicate our results. It performs feature extraction, **Z-score outlier detection**, and automated anomaly ranking.

```bash
python scripts/anomaly_scan_vheat.py \
  --input_dir "./your_manuscript_images" \
  --output_dir "./research_outputs" \
  --vheat_repo "/path/to/vHeat_source" \
  --weights "/path/to/vHeat_base.pth" \
  --z_thr 1.3 \
  --int_thr 0.45

```

---

## Methodological Orientation

This project treats computational methods as **analytical aids**. The thermal diffusion metaphor of the vHeat model is particularly suitable for codicology, as it parallels material processes such as:

* **Ink Penetration**: How ink interacts with porous paper fibers over centuries.
* **Material Stress**: Identifying patterns of mold, stains, or heavy ritual handling.
* **Statistical Outliers**: Using Z-Scores to separate intentional writing from accidental material damage.

---

## vHeat Model and Attribution

This project builds upon the **vHeat** visual backbone model proposed in:

> Wang, Zhaozhi, Yue Liu, Yunjie Tian, Yunfan Liu, Yaowei Wang, and Qixiang Ye.
> *Building Vision Models upon Heat Conduction.* > Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

```bibtex
@InProceedings{Wang_2025_CVPR,
    author    = {Wang, Zhaozhi and Liu, Yue and Tian, Yunjie and Liu, Yunfan and Wang, Yaowei and Ye, Qixiang},
    title     = {Building Vision Models upon Heat Conduction},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    year      = {2025}
}

```

---

## Citation

If you use or reference this project, please cite:
**Wei, Xiang. Computational Codicology via Thermal Diffusion: Visualizing Material Anomalies in Lanten Religious Manuscripts. GitHub repository, vHeat-Codicology.**

```

### Final Checklist for GitHub:
1.  **`scripts/`**: Put the Python script I wrote for you earlier into this folder and name it `anomaly_scan_vheat.py`.
2.  **`examples/figures/`**: Make sure your `.png` files are placed here and renamed to match the links (e.g., `003_overlay.png`).
3.  **`requirements.txt`**: Create this file and add:
    ```text
    torch
    torchvision
    numpy
    opencv-python
    matplotlib
    scipy
    ```

**Is there any specific data or license information you want to add before you push this to GitHub?**

```
