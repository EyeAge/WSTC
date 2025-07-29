# WSTC: Task‑Adaptive Visual–Linguistic Modeling via Semantic Embedding and Diffusion Projection

> **AAAI 2026 Supplementary Code**  
> Corresponding paper: _Task‑adaptive Visual‑Linguistic Modeling via Semantic Embedding and Diffusion Projection for Few/Zero‑Shot Medical Tasks_

---

## 1. Repository Structure

```text
.
├── README.md                # <- THIS FILE
├── requirements.txt         # Python dependencies (see §2)
├── baselines/
│   └── UniMed-CLIP/
│       └── checkpoints/
│           └── b16_400m.pt  # pre‑trained UniMedCLIP weights (download separately)
├── datasets/                # external data (see §3)
├── modules/                 # plug‑and‑play adapters
│   ├── wsam.py
│   ├── tcdam.py
│   ├── text_projector.py
│   ├── meta_conditioning.py
│   └── wstc.py
├── runner/                  # zero‑shot & retrieval pipeline
│   ├── run.py
│   ├── dataset_zoo.py
│   ├── logger.py
│   └── model_zoo.py
└── trainer/                 # few‑shot training pipelines
    ├── train_cls.py
    └── train_seg.py
```
Please get the initial code for UniMed-CLIP before running the code. Thanks to UniMed-CLIP model for supporting this project.
The datasets used in this article are all open source datasets.
---

## 2. Quick Environment Setup

The project is tested on **Python 3.9 / CUDA 11.8 / PyTorch 2.2+**.  
We recommend creating a **conda** environment:

```bash
conda create -n wstc python=3.9
conda activate wstc

# ---- essential DL stack ----
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# ---- remaining Python libs ----
pip install -r requirements.txt
```

`requirements.txt` (feel free to auto‑generate):

```
open_clip_torch>=2.23
transformers>=4.41
timm>=0.9
scikit-learn>=1.5
pandas
tqdm
tifffile
Pillow
matplotlib
```

---

## 3. Dataset Preparation

| Dataset            | Script ID     | Default Path (after extraction) | Notes |
|--------------------|---------------|---------------------------------|-------|
| **ChestXray14**    | `chestxray14` | `datasets/chestxray14/`         | Place images in `images_flat/`, CSV in root |
| **RSNA Pneumonia** | `rsna_pneu`   | `datasets/rsna_pneu/`           | Expect `images/*.png` + `labels.csv` |
| **PCam**           | `pcam`        | `datasets/pcam/`                | Tiles handled internally |
| **ISIC 2019**      | `isic2019`    | `datasets/isic2019/`            | follow official split |
| **ROCO‑v2**        | `rocov2`      | `datasets/rocov2/`              | image–caption retrieval |
| **ISIC 2018**      | `isic2018`    | `datasets/isic2018/`          | optional |

> **Tip** – each dataset loader is defined in `runner/dataset_zoo.py`; modify paths if needed.

---

## 4. Pre‑trained UniMedCLIP Checkpoint

Download `b16_400m.pt` from the official UniMed‑CLIP release and place it under  
`baselines/UniMed-CLIP/checkpoints/`.

---

## 5. Zero‑Shot & Retrieval Evaluation

```bash
# zero‑shot classification
python runner/run.py   --baseline unimedclip   --variant +wstc   --dataset chestxray14   --batch_size 64   --split test --seed 3 --metric acc --sample_limit 2000 --tau 1.3

# retrieval
python runner/run.py   --baseline unimedclip   --variant +wstc   --dataset rocov2 
```
Key flags:

| Flag             | Description                                               | Example      |
|------------------|-----------------------------------------------------------|--------------|
| `--variant`      | `baseline` / `+wsam` / `+tcdam` / `+wstc` / `+wstc_train` | `+wstc`      |
| `--metric`       | `auc` / `acc` (dataset‑specific default)                  | `auc`        |
| `--sample_limit` | cap validation subset for speed                           | `2000`       |
| `--baseline`     | `unimedclip` / `clip` / `medclip` / `biovil`              | `unimedclip` |
| `--split`        | `test` / `val` / `train`                                  | `test`       |


---

## 6. Few‑Shot Classification (k‑shot)

```bash
python trainer/train_cls.py   --baseline unimedclip   --variant +wstc_train   --dataset chestxray14   --kshot 5   --epochs 20   --batch_size 32
```

Key flags:

| Flag             | Description                                               | Example      |
|------------------|-----------------------------------------------------------|--------------|
| `--variant`      | `baseline` / `+wsam` / `+tcdam` / `+wstc` / `+wstc_train` | `+wstc`      |
| `--metric`       | `auc` / `acc` (dataset‑specific default)                  | `auc`        |
| `--baseline`     | `tip-adapter_f` / `meta-adapter_f` / `unimedclip`         | `unimedclip` |

---

## 7. Few‑Shot Segmentation (ISIC2018)

```bash
python trainer/train_seg.py   --baseline unimedclip   --variant +wstc_train   --dataset isic2018   --epochs 50   --batch_size 32
```

Loss = Dice + BCE (see `train_seg.py`).  Tip/Meta‑Adapter baselines use `forward_seg()` branch and do **not** affect classification pipelines.

---

## 8. Reproducing Table 2 & Table 3

| Paper Table  | Script                           | Seeds | GPU (RTX4090‑24 GB) | Time / run |
|--------------|----------------------------------|-------|---------------------|------------|
| **Table 2**  | `trainer/train_cls.py` (k‑shot)  | 0,1,2 | ≤ 14 GB             | 2 h        |
| **Table 3**  | `runner/run.py` (zero‑shot)      | 0     | ≤ 12 GB             | 20 min     |

---

