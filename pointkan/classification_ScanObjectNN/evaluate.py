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


def parse_args():
    parser = argparse.ArgumentParser('Fine-tuning')
    parser.add_argument('--top_k', type=int, default=5, help='top-k similar labels to consider')
    parser.add_argument('--model', default='pointKAN', help='model name')
    parser.add_argument('--pretrained_path', default='/mnt/c/Users/Administrator/PointKAN-pytorch/checkpoints/pointKAN-20250606204314/best_checkpoint.pth', help='path to pre-trained model (last_checkpoint.pth)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


def evaluate_model(model, test_loader, device):
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x = x.permute(0, 2, 1)  # [B, N, 3] -> [B, 3, N]
            x, y = x.to(device), y.to(device)

            # 前向传播
            outputs = model(x)
            preds = outputs.argmax(dim=1)

            # 更新统计信息
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True

    # Model
    net = models.__dict__[args.model](num_classes=args.num_classes)
    net = net.to(device)

    for module in net.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0  # 将 Dropout 的概率设置为 0

    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    # state = torch.load(args.pretrained_path, weights_only=True)['net']
    # net.load_state_dict(state)

    state = torch.load(args.pretrained_path, weights_only=True)['net']
    state = {key[7:]: val for key, val in state.items()}
    net.load_state_dict(state)

    # Datasets (use subset for fine-tuning)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    test_acc = evaluate_model(net, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()
