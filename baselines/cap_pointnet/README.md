# PointCloud Adversarial Attacks and CAP PointNet

This repository focuses on adversarial attacks and defense strategies for point cloud data, using the ModelNet10 dataset. It includes implementations of **IFGM** and **GenPert** attacks and applies the [CAP PointNet framework](#references), comparing classification performance with and without CAP to assess its robustness on point cloud classification.

## Table of Contents

- [Requirements](#requirements)
- [Dataset](#dataset)
- [Training Models](#training-models)
- [Running Attacks](#running-attacks)
- [References](#references)

## Requirements

- Python 3.x
- PyTorch
- Other dependencies specified in `requirements.txt`

## Dataset

1. Download the ModelNet40 dataset and place it in the `data/` directory.
2. To use ModelNet10, filter the relevant categories or adjust `data_utils/modelnet_dataloader.py` to target ModelNet10 classes.

## Training Models

1. **Train CAPPointNet**:

   ```bash
   python cap_train.py
   ```

   This command trains the CAPPointNet model on ModelNet10, preparing it for adversarial defense experiments.

2. **Train PointNet++**:

   ```bash
   python train.py
   ```

   This command trains a baseline PointNet++ model on ModelNet10 for comparison.

## Running Attacks

- **GenPert Attack**:
  Run targeted or untargeted attacks on PointNet++ or CAPPointNet:

  ```bash
  python attack/GenPert/cap_pointnet_targeted.py
  ```

- **IFGM Attack**:
  Run targeted or untargeted IFGM attacks:

  ```bash
  python attack/IFGM/cap_pointnet_targeted.py
  ```

## References

This repository implements methods and structures based on the following works. Please consider citing them if you find this repository useful in your research:

- **CAPPointNet Framework**:  

  ```bibtex
  @article{Ding_Jiang_Huang_Zhang_Li_Yang,
    title = {CAP: Robust Point Cloud Classification via Semantic and Structural Modeling},
    author = {Ding, Daizong and Jiang, Erling and Huang, Yuanmin and Zhang, Mi and Li, Wenxuan and Yang, Min},
    journal = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2024},
    language = {en-US}
  }
  ```

- **GenPert Attack**:  

  ```bibtex
  @inproceedings{Xiang_Qi_Li_2019,
    title = {Generating 3D Adversarial Point Clouds},
    author = {Xiang, Chong and Qi, Charles R. and Li, Bo},
    booktitle = {2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019},
    month = {Jun},
    url = {http://dx.doi.org/10.1109/cvpr.2019.00935},
    DOI = {10.1109/cvpr.2019.00935},
    language = {en-US}
  }
  ```

- **IFGM Attack**:  

  ```bibtex
  @inproceedings{Liu_Yu_Su_2019,
    title = {Extending Adversarial Attacks and Defenses to Deep 3D Point Cloud Classifiers},
    author = {Liu, Daniel and Yu, Ronald and Su, Hao},
    booktitle = {2019 IEEE International Conference on Image Processing (ICIP)},
    year = {2019},
    month = {Sep},
    url = {http://dx.doi.org/10.1109/icip.2019.8803770},
    DOI = {10.1109/icip.2019.8803770},
    language = {en-US}
  }
  ```

- **PointNet++ Reference Implementation**:  
  [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

---

For more information, please refer to the cited papers. This README provides structured guidance for setting up and using this repository for experiments in adversarial defenses on point cloud data.