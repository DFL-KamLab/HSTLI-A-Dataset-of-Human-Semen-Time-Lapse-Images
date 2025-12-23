# HSTLI: A Dataset of Human Semen Time-Lapse Images

This repository provides code associated with the paper:

**HSTLI: A Dataset of Human Semen Time-Lapse Images for Detection, Tracking, and Motility Parameter Analysis**

The code supports **reproducible detection, tracking, and CASA-style motility analysis** on the HSTLI dataset, using modern computer vision and multi-object tracking pipelines.

---

## Overview

The provided scripts implement an end-to-end analysis pipeline for time-lapse semen microscopy videos, including:

- Sperm head detection using **YOLOv5**
- Multi-object tracking using **SORT**
- Sliding-window computation of classical CASA motility parameters
- Visualization via:
  - Motility spider (radar) plots
  - Motility density heatmaps

The pipeline supports **both YOLO-based detections and ground-truth annotations**.

To run YOLO-based detection, clone the official YOLOv5 repository and install its dependencies:
```bash
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt
```

---

## Dataset

The HSTLI dataset is hosted on Hugging Face:

**Hugging Face Dataset:**  
https://huggingface.co/datasets/DFL-KamLab/HSTLI_A-Dataset-of-Human-Semen-Time-Lapse-Images

HSTLI contains **3,266 time-lapse microscopy videos of human sperm** acquired using:

- A commercial **CASA system (Sperm Class Analyzer)**
- An **optical microscope** (Swift M10DB-MP + Fujifilm X-T30)

A subset of videos is manually annotated with bounding boxes for sperm heads, enabling detection, tracking, and motility analysis.

---

## Code Usage

The main entry point is:

run_motility_pipeline.py


All user-editable paths and parameters are located at the **top of the script** and clearly marked.

The script supports two modes:

- `MODE = "yolo"`  
  Uses YOLOv5 detections followed by SORT tracking.
- `MODE = "ground_truth"`  
  Uses ground-truth annotations directly for tracking.

The script generates:

- `motility_spider_plot.png`
- `motility_heatmaps.png`

No GUI is required; figures are saved automatically.

---

## Citation

If you use this dataset or code in your research, please cite the accompanying **bioRxiv preprint**:

**HSTLI: A Dataset of Human Semen Time-Lapse Images for Detection, Tracking, and Motility Parameters Analysis**  
Atilla Sivri, JiWon Choi, Justin Bopp, Albert Anouna, Matthew VerMilyea, Gustave Alkhoury,  
Omer Onder Hocaoglu, Moshe Kam, Ludvik Alkhoury  
*bioRxiv*, 2025  
DOI: https://doi.org/10.64898/2025.12.15.694470

### BibTeX

```bibtex
@article {Sivri2025.12.15.694470,
  author = {Sivri, Atilla and Choi, JiWon and Bopp, Justin and Anouna, Albert and VerMilyea, Matthew and Alkhoury, Gustave and Hocaoglu, Omer Onder and Kam, Moshe and Alkhoury, Ludvik},
  title = {HSTLI, a Dataset of Human Semen Time-lapse Images for Detection, Tracking, and Motility Parameters Analysis},
  elocation-id = {2025.12.15.694470},
  year = {2025},
  doi = {10.64898/2025.12.15.694470},
  publisher = {Cold Spring Harbor Laboratory},
  journal = {bioRxiv}
}
```

## Links

- **Dataset:** https://huggingface.co/datasets/DFL-KamLab/HSTLI_A-Dataset-of-Human-Semen-Time-Lapse-Images
- **Paper:** https://doi.org/10.64898/2025.12.15.694470
- **Code repository:** https://github.com/DFL-KamLab
