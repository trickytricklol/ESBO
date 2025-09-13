import open3d as o3d
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils import mkdir_p, cal_loss, save_model, Logger, progress_bar
from ScanObjectNN import ScanObjectNN
import models as models
import numpy as np
import datetime
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设 args 是通过 argparse 解析得到的参数
args = argparse.Namespace(batch_size=1, num_points=2048, workers=4)

test_loader = DataLoader(ScanObjectNN(partition='training', num_points=args.num_points),
                         batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

# 从 test_loader 中读取数据并可视化
for batch_id, batch_data in enumerate(tqdm(test_loader)):
    points, _ = batch_data  # 假设 batch_data 包含点云数据和标签
    points = points.squeeze().cpu().numpy()  # 转换为 numpy 数组并移除多余的维度

    # 使用 matplotlib 绘制点云
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(f'Batch {batch_id}')
    plt.show()

    # 可以设置一个条件来决定何时停止可视化，例如可视化前几个批次
    # if batch_id >= 2:  # 只可视化前三个批次
    #     break
