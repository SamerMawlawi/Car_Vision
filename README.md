#  Car_Vision — Car Exterior Damage Detection

> **Instance Segmentation** model for identifying and localizing exterior damage on vehicles from photographs.

---

##  Overview

Car Vision is a deep learning project that detects **exterior body damage** on cars using instance segmentation. The pipeline is straightforward:

```
User uploads a photo  →  AI model processes it  →  User gets damage results
```

The model identifies **what** the damage is (scratch, dent, crack, …) and **where** it is on the vehicle — producing pixel-level masks rather than just bounding boxes.

---

## Architecture & Tech Stack

| Layer | Technology |
|---|---|
| **Model** | [YOLOv26 / Ultralytics](https://docs.ultralytics.com/) — Instance Segmentation |
| **Framework** | PyTorch + TorchVision |
| **UI** | [Streamlit](https://streamlit.io/) — interactive web interface |
| **Image Processing** | OpenCV, Pillow, NumPy, Matplotlib |
| **Language** | Python 3.11.9 |

---

## Project Structure

```
Car_Vision/
│
├── data/                          # Datasets (git-ignored)
│   ├── Car damages dataset/       # Raw damage annotations (JSON + images)
│   ├── Car healthy dataset/       # Undamaged vehicle images
│   ├── Car parts dataset/         # Car part annotations (JSON + images)
│   └── unified_dataset/           # Merged & cleaned dataset
│       ├── images/                #   └── Vehicle_0001.png … Vehicle_1371.png
│       └── labels/                #   └── Vehicle_0001.txt … Vehicle_1371.txt (YOLO format)
│
├── notebooks/                     # Jupyter notebooks for exploration & preprocessing
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│
├── src/                           # Source code
│   └── utils/
│       ├── master_mapper.py       # Unified class name → ID map (0–28)
│       ├── master_mapper_IDs.py   # Source annotation ID → unified ID map
│       └── unified_dataset.py     # Dataset merging & preprocessing logic
│
├── venv/                          # Python virtual environment (git-ignored)
│
├── check_setup.py                 # Environment & GPU health check
├── requirements.txt               # Python dependencies
├── .python-version                # Python version pin (3.11.9)
└── .gitignore
```

---

## 🏷️ Class Taxonomy

The model recognizes **29 classes** total — **8 damage types** and **21 car parts**:

### Damage Types (Classes 0–7)

| ID | Label |
|---:|-------|
| 0 | Dent |
| 1 | scratch|
| 2 | crack |
| 3 | glass shatter |
| 4 | lamp broken |
| 5 | tire flat |


### Car Parts (Classes 8–28)

| ID | Label | | ID | Label |
|---:|-------|---|---:|-------|
| 8 | Quarter-panel | | 19 | Back-wheel |
| 9 | Front-wheel | | 20 | Back-windshield |
| 10 | Back-window | | 21 | Hood |
| 11 | Trunk | | 22 | Fender |
| 12 | Front-door | | 23 | Tail-light |
| 13 | Rocker-panel | | 24 | License-plate |
| 14 | Grille | | 25 | Front-bumper |
| 15 | Windshield | | 26 | Back-bumper |
| 16 | Front-window | | 27 | Mirror |
| 17 | Back-door | | 28 | Roof |
| 18 | Headlight | | | |

> Part classes allow the model to contextualize *which* component is damaged.

---

## Dataset

| Property | Value |
|---|---|
| **Total images** | 4000+ |
| **Annotation format** | YOLO segmentation (polygon `.txt`) |
| **Sources** | Car damages dataset, Car parts dataset, Car healthy dataset |
| **Unified split** | `unified_dataset/images/` + `unified_dataset/labels/` |

The raw datasets ship in JSON format and are converted to YOLO-compatible `.txt` labels via the `json_to_text.ipynb` and `unified_dataset.ipynb` notebooks.

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd Car_Vision
```

### 2. Create & activate a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify your environment

```bash
python check_setup.py
```

Expected output (example with NVIDIA GPU):
```
--- Environment Check on Linux ---
Python Version: 3.11.9
PyTorch Version: 2.x.x
✅ NVIDIA GPU Detected (CUDA)
Device: NVIDIA GeForce RTX 3060ti
```

> The script also detects **AMD (ROCm)** and **Apple Silicon (MPS)** GPUs.

### 5. Place the data

Download or extract the datasets into the `data/` folder following the structure shown above. The `data/` directory is git-ignored — you'll need to obtain it separately.

---

## 🔄 Pipeline Overview

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw Data    │     │  Preprocessing   │     │  Training       │
│  (JSON + PNG)│────▶│  (Notebooks)    │────▶│  (Ultralytics) │
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                     ┌──────────────────┐     ┌─────────────────┐
                     │  Streamlit UI    │◀────│  Inference      │
                     │  (Upload + View) │     │  (YOLOv8-seg)   │
                     └──────────────────┘     └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  XAI / Grad-CAM │
                                              │  (Captum)       │
                                              └─────────────────┘
```

1. **Data Preprocessing** — Merge and convert raw annotations into a unified YOLO-format dataset.
2. **Training** — Fine-tune a YOLOv8 segmentation model on the unified dataset.
3. **Inference** — Run the trained model on user-uploaded images.
4. **Explainability** — Use Grad-CAM to highlight which regions influence the model's decisions.
5. **UI** — Streamlit app provides an interactive upload-and-predict experience.

---

## 👥 Team

*Intelligent Systems Design — University Course Project*

---

## 📜 License

This project is developed for academic purposes.
