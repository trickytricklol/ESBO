import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
sys.path.append('.')  # 确保可以导入models
from torch.utils.data import DataLoader
import models as models
from data import ModelNet40
import sklearn.metrics as metrics
from helper import cal_loss


def extract_classifier_weights(net):
    """提取分类器最后一层权重"""
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    classifier_weights = net.classifier[-1].weight.detach().cpu()
    return classifier_weights

def compute_cosine_similarity(class_weights):
    """计算类别间余弦相似度矩阵"""
    normed = F.normalize(class_weights, dim=1)
    sim_matrix = torch.mm(normed, normed.t())
    return sim_matrix

def build_similar_label_mapping(sim_matrix, topk=10):
    """构建相似标签映射"""
    mapping = {}
    for i in range(sim_matrix.size(0)):
        sim_scores = sim_matrix[i].clone()
        sim_scores[i] = -1e6
        topk_indices = torch.topk(sim_scores, k=topk).indices.tolist()
        mapping[i] = topk_indices
    return mapping

import torch
import torch.nn.functional as F

def pgd_attack_similar_labels(model, inputs, true_labels, target_labels, loss_fn,
                             epsilon=0.05, alpha=0.05, steps=1):
    """PGD对抗攻击"""
    original_mode = model.training
    model.eval()
    
    ori_inputs = inputs.detach()
    perturbed = ori_inputs.clone()
    
    with torch.no_grad():
        for i in range(len(inputs)):
            sample = ori_inputs[i].unsqueeze(0).requires_grad_(True)
            target = target_labels[i].unsqueeze(0)
            
            for _ in range(steps):
                with torch.enable_grad():
                    outputs = model(sample)
                    loss = loss_fn(outputs, target)
                    grad = torch.autograd.grad(loss, sample,
                                            retain_graph=False,
                                            create_graph=False)[0]
                
                sample = sample.detach() - alpha * grad.sign()
                sample = torch.clamp(sample,
                                   min=ori_inputs[i].unsqueeze(0)-epsilon,
                                   max=ori_inputs[i].unsqueeze(0)+epsilon)
                sample = torch.clamp(sample, min=-1, max=1)
                sample.requires_grad_()
            
            perturbed[i] = sample.squeeze(0)
    
    model.train(original_mode)
    return perturbed

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    device = 'cuda'
    
    # 创建输出目录
    args_checkpoint = 'test_results'
    if not os.path.isdir(args_checkpoint):
        os.makedirs(args_checkpoint)
    
    # 日志设置
    logging.basicConfig(
        filename=os.path.join(args_checkpoint, 'test_log.txt'),
        level=logging.INFO,
        format='%(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    def printf(str):
        logging.info(str)
    
    # 加载模型
    printf('==> Building model..')
    net = models.__dict__['pointKAN'](num_classes=40)
    net = net.to(device)
    
    # 加载预训练权重
    pretrained_path = '/mnt/c/Users/Administrator/PointKAN-pytorch/finetune_ckpt/pointKAN-finetune-20250626113705-Modelnet40/best_checkpoint.pth'
    printf(f'==> Loading pretrained model from {pretrained_path}')
    state = torch.load(pretrained_path, weights_only=True)['net']
    net.load_state_dict(state)
    
    # 提取分类器权重并计算相似度
    class_weights = extract_classifier_weights(net)
    sim_matrix = compute_cosine_similarity(class_weights)
    class_mapping = build_similar_label_mapping(sim_matrix)
    
    # ======== 计算 D_inter（平均类间可分性距离）========
    C = sim_matrix.size(0)
    mask = ~torch.eye(C, dtype=bool)
    d_inter = (1 - sim_matrix[mask]).mean()
    printf(f"D_inter (avg inter-class distance): {d_inter:.6f}")
    with open(os.path.join(args_checkpoint, "d_inter.txt"), "w") as f:
        f.write(f"D_inter = {d_inter:.6f}\n")
    
    # ModelNet40类别名称
    class_names = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
        'bottle', 'bowl', 'car', 'chair', 'cone',
        'cup', 'curtain', 'desk', 'door', 'dresser',
        'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
        'laptop', 'mantel', 'monitor', 'night_stand', 'person',
        'piano', 'plant', 'radio', 'range_hood', 'sink',
        'sofa', 'stairs', 'stool', 'table', 'tent',
        'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]
    
    def plot_similarity_heatmap(matrix, class_names, title, save_path):
        plt.figure(figsize=(20, 18))
        sns.heatmap(matrix, annot=False, cmap="YlOrRd",
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={"shrink": 0.5})
        plt.title(title, fontsize=16)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    plot_similarity_heatmap(sim_matrix.numpy(), class_names,
                            "Class Cosine Similarity Matrix",
                            os.path.join(args_checkpoint, "class_similarity.png"))
    
    # 测试数据
    printf('==> Preparing data..')
    test_loader = DataLoader(ModelNet40(partition='test', num_points=1024),
                             batch_size=32, shuffle=False, num_workers=4)
    
    # 执行测试
    printf('==> Testing..')
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    error_dict = {}
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = cal_loss(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            mask = preds != label
            for true, pred in zip(label[mask].cpu().numpy(), preds[mask].cpu().numpy()):
                key = (true, pred)
                error_dict[key] = error_dict.get(key, 0) + 1
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
    
    # 错误分布按相似度排名统计
    error_counts = {}
    for (true, pred), count in error_dict.items():
        sim_scores = sim_matrix[true]
        sorted_classes = torch.argsort(sim_scores, descending=True)
        rank = (sorted_classes == pred).nonzero().item()
        error_counts[(true, pred, rank)] = count
    
    rank_error_counts = {}
    for (true, pred, rank), count in error_counts.items():
        rank_error_counts[rank] = rank_error_counts.get(rank, 0) + count
    
    plt.figure(figsize=(10, 6))
    ranks = sorted(rank_error_counts.keys())
    counts = [rank_error_counts[r] for r in ranks]
    plt.bar(ranks, counts)
    plt.title("Classification Errors by Class Similarity Rank")
    plt.xlabel("Similarity Rank (0 = most similar)")
    plt.ylabel("Error Count")
    plt.savefig(os.path.join(args_checkpoint, "error_distribution.png"))
    plt.close()
    
    with open(os.path.join(args_checkpoint, "error_details.txt"), "w") as f:
        f.write("TrueClass\tPredClass\tSimilarityRank\tCount\n")
        for (true, pred, rank), count in sorted(error_counts.items(), key=lambda x: -x[1]):
            f.write(f"{class_names[true]}\t{class_names[pred]}\t{rank}\t{count}\n")
    
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    acc = 100. * metrics.accuracy_score(test_true, test_pred)
    acc_avg = 100. * metrics.balanced_accuracy_score(test_true, test_pred)
    printf(f"Test Accuracy: {acc:.3f}% | Balanced Accuracy: {acc_avg:.3f}%")
    
    # 混淆矩阵
    cm = metrics.confusion_matrix(test_true, test_pred)
    np.savetxt(os.path.join(args_checkpoint, "confusion_matrix.txt"), cm, fmt="%d")
    plot_similarity_heatmap(cm, class_names, "Confusion Matrix",
                            os.path.join(args_checkpoint, "confusion_matrix.png"))
    printf(f"Results saved to {args_checkpoint}")

if __name__ == '__main__':
    main()
