# ESBO: Enhanced Semantic Boundary Optimization for Robust Point Cloud Classification

> **Beyond Empirical Risk: Achieving Point Cloud Robustness through Margin Refinement and Data Purification**

Shurong Dong, Jie Lin, Ruida Mao, Xuepei Yang, Shuping Zhang, Guanghui Zhao

Expert Systems With Applications (ESWA), Manuscript **ESWA-D-26-03463_R1** (under review)

---

## Overview

LiDAR point clouds are highly vulnerable to noise, adverse weather, and complex environmental
disturbances, which cause geometric distortion and degrade the robustness of deep-learning
perception models. This work reveals that model failures under distribution shift are
predominantly concentrated in the **inter-class boundary regions** of the feature space, and
proposes the **Enhanced Semantic Boundary Optimization (ESBO)** framework with two
complementary knowledge-driven mechanisms:

1. **Entropy-informed data purification** — a dynamic evaluation mechanism based on
   information entropy that assesses the semantic reliability of (augmented) samples and
   suppresses over-augmented or semantically corrupted data via sample-wise loss weighting.
2. **Similarity-guided adversarial boundary optimization** — an inter-class guided
   adversarial training strategy that attacks each sample toward its *semantically
   neighboring* (top-2 predicted) class and significantly enlarges the feature margin
   between similar categories.

ESBO is **model-agnostic** and integrates seamlessly with existing architectures
(PointNet++, DGCNN, PointKAN, Point-MAE) and augmentation strategies (PointWOLF, AugTune).

## Key Results (from the manuscript)

| Benchmark | Architecture | Gain |
|---|---|---|
| ModelNet40 | PointNet++ | +2.9% |
| ModelNet40 | DGCNN | +1.1% |
| ModelNet40-C | PointNet++ | +6.0% |
| ModelNet40-C | DGCNN | +9.1% |
| ScanObjectNN | PointNet++ | +6.8% |
| ScanObjectNN | DGCNN | +5.3% |

Representative ModelNet40-C evaluation logs (MOA / MCE) are provided in
[`docs/results/`](docs/results/):

| Model / Setting | Clean Acc. | MOA |
|---|---|---|
| DGCNN + ESBO (eps=0.01) | 93.11% | 79.50% |
| PointNet++ + ESBO (eps=0.01) | — | 81.60% |
| DGCNN + ESBO (ka variant, 10ep, k=0.5) | 92.95% | 77.41% |

## Repository Structure

```
ESBO/
├── pointwolf/                 # Main entry: DGCNN / PointNet / PointNet++ + PointWOLF/AugTune + ESBO (--advf)
│   ├── main.py                #   unified entry (vanilla / AugTune / ESBO / ModelNet40-C eval)
│   ├── train.py               #   train_vanilla / train_AugTune / train_advf (entropy weighting + similar-label PGD)
│   ├── testmodelnetc.py       #   ModelNet40-C evaluation
│   ├── model.py / PointWOLF.py / data.py / util.py / ...
│   └── analysis/              #   analysis & plotting scripts (sample distribution, similarity, stats)
├── pointnet2/                 # PointNet++ experiments
│   ├── advf.py                #   ESBO adversarial fine-tuning on ModelNet40
│   ├── advf_scan.py           #   ESBO adversarial fine-tuning on ScanObjectNN
│   ├── advf_scans.py          #   (multi-seed variant)
│   ├── plotadvf.py            #   visualization
│   ├── test_classification_modelc.py  # ModelNet40-C evaluation
│   ├── train_classification.py / test_classification.py / train_cls_scan.py / test_cls_scan.py
│   ├── models/                #   PointNet / PointNet++ model definitions
│   └── data_utils/            #   ModelNet40 / ScanObjectNN loaders
├── pointkan/                  # PointKAN experiments
│   ├── classification_ModelNet40/   # advf.py, advf_plot.py, evaluate.py, models/, utils/
│   └── classification_ScanObjectNN/ # advf.py, adversial_finetune.py, evaluate.py, models/, utils/
├── pointmae/                  # Self-supervised Point-MAE + ESBO fine-tuning
│   ├── tools/runner_finetune.py     #   fine-tuning runner (ESBO)
│   ├── cfgs/finetune_modelnet_advf.yaml
│   ├── models/ datasets/ utils/ util.py main.py
├── baselines/
│   ├── sinpoint/              # SinPoint augmentation baseline (ModelNet40 / ScanObjectNN)
│   └── cap_pointnet/          # Adversarial attack/defense baseline (IFGM, GenPert, CAP-PointNet)
├── data_utils/
│   └── modelnet_c_loader.py   # ModelNet40-C .npy loader (from readmodelnet)
├── docs/
│   ├── figures/               # Paper / experiment figures
│   └── results/               # ModelNet40-C evaluation logs (MOA/MCE)
├── requirements.txt
└── LICENSE
```

## Installation

- Python 3.8+ (recommended 3.9/3.10), PyTorch >= 1.8 with CUDA.
- See [`requirements.txt`](requirements.txt).
- `pointkan/` additionally requires `pointnet2_ops` (compiled extension) and
  `rational_kat_cu` (KAN kernel); follow the instructions of the upstream
  [PointKAN-pytorch](https://github.com/ma-xu/PointKAN-pytorch) repo.
- `pointmae/` uses `tensorboardX` for logging.

## Data Preparation

1. **ModelNet40**: download the aligned ModelNet40 and place it as
   `pointnet2/data/modelnet40_normal_resampled/` (see `pointnet2/data_utils/ModelNetDataLoader.py`)
   or `pointwolf/data/modelnet40_normal_resampled/`.
2. **ModelNet40-C**: obtain the `.npy` corrupted files (16 corruptions × 5 severities) and
   point `--modelnet_c_path` / the loader path to them. `data_utils/modelnet_c_loader.py`
   loads `data_<corruption>_<severity>.npy` with a unified `label.npy`.
3. **ScanObjectNN**: download the official dataset; the loaders are provided in
   `pointnet2/data_utils/ScanObjectNN.py` and `pointkan/classification_ScanObjectNN/`.

## Reproduction

### DGCNN + PointWOLF/AugTune + ESBO (main experiment)

```bash
cd pointwolf

# Vanilla DGCNN baseline
python main.py --exp_name dgcnn_vanilla --model dgcnn --dataset modelnet40 --epochs 250

# DGCNN + PointWOLF augmentation
python main.py --exp_name dgcnn_pointwolf --model dgcnn --dataset modelnet40 --epochs 250 --PointWOLF

# DGCNN + ESBO (entropy-weighted similar-label adversarial fine-tuning)
python main.py --exp_name dgcnn_advf --model dgcnn --dataset modelnet40 --epochs 250 --advf

# ModelNet40-C evaluation
python main.py --model dgcnn --dataset modelnet40 --eval --test_modelnet_c \
    --modelnet_c_path /path/to/modelnet40_c --model_path checkpoints/<exp>/models/model.t7
```

### PointNet++ + ESBO

```bash
cd pointnet2

# Standard classification training (baseline)
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_ssg_wo_normals

# ESBO adversarial fine-tuning on ModelNet40
python advf.py --model pointnet2_cls_ssg --log_dir pointnet2_ssg_wo_normals_advf \
    --pretrained_path log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth \
    --epoch 10 --batch_size 24

# ModelNet40-C evaluation
python test_classification_modelc.py --model pointnet2_cls_ssg \
    --log_dir pointnet2_ssg_wo_normals_advf

# ScanObjectNN: training and ESBO fine-tuning
python train_cls_scan.py ...
python advf_scan.py --pretrained_path <ckpt> ...
```

### PointKAN + ESBO

```bash
cd pointkan/classification_ModelNet40
python advf.py --model pointKAN --pretrained_path <pretrained-pointKAN.pth> --epoch 30
cd ../classification_ScanObjectNN
python advf.py --model pointKAN --pretrained_path <pretrained-pointKAN.pth> ...
```

### Point-MAE (self-supervised pretrain + ESBO fine-tune)

```bash
cd pointmae
# Self-supervised pretraining first (see upstream Point-MAE), then ESBO fine-tuning:
python main.py --config cfgs/finetune_modelnet_advf.yaml \
    --finetune_model --ckpts <pretrained_point_mae.pth> \
    --exp_name finetune_modelnet_advf
```

### Baselines

```bash
# SinPoint augmentation baseline
cd baselines/sinpoint/classification_ModelNet40 && bash train.sh

# CAP-PointNet attack/defense baseline
cd baselines/cap_pointnet && python cap_train.py && python train.py
```

## Upstream Repositories & Acknowledgements

This repository reorganizes the authors' experiment code, which is built on and modified from:

- [PointNet/PointNet++ PyTorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- [PointWOLF](https://github.com/VisualAI-KHU/PointWOLF) — augmentation baseline
- [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) — self-supervised backbone
- [PointKAN-pytorch](https://github.com/ma-xu/PointKAN-pytorch) — KAN backbone
- [SinPoint](https://github.com/xiaoyuan1996/SinPoint) — augmentation baseline
- [PointCloud-Adversarial-Attacks-and-CAP-PointNet](https://github.com/abhishekbose22/PointCloud-Adversarial-Attacks-and-CAP-PointNet) — attack/defense baseline
- [ModelNet40-C](https://github.com/ldkong1205/ModelNet40-C) — robustness benchmark

Third-party code retains its respective upstream licenses (see the `LICENSE` files inside
each submodule). The ESBO implementation and analysis scripts are released under MIT.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{dong2025esbo,
  title={Beyond Empirical Risk: Achieving Point Cloud Robustness through Margin Refinement and Data Purification},
  author={Dong, Shurong and Lin, Jie and Mao, Ruida and Yang, Xuepei and Zhang, Shuping and Zhao, Guanghui},
  journal={Expert Systems With Applications},
  year={2026},
  note={Manuscript ESWA-D-26-03463_R1}
}
```

## Contact

For questions or issues, please open a GitHub issue.
