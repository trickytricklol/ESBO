import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import logging
import datetime
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from data import ModelNet40
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置
import matplotlib.pyplot as plt

def extract_classifier_weights(net):
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    classifier_weights = net.classifier[-1].weight.detach().cpu()
    return classifier_weights

def compute_cosine_similarity(class_weights):
    normed = F.normalize(class_weights, dim=1)
    sim_matrix = torch.mm(normed, normed.t())
    # 取绝对值，确保所有相似度值都是非负的
    # sim_matrix = torch.abs(sim_matrix)
    return sim_matrix

def build_similar_label_mapping(sim_matrix):
    mapping = {}
    for i in range(sim_matrix.size(0)):
        sim_scores = sim_matrix[i].clone()
        sim_scores[i] = -1e6  # 排除自己
        sorted_indices = torch.argsort(sim_scores, descending=True)
        mapping[i] = sorted_indices.tolist()
    return mapping

def plot_error_distribution(error_counts, class_names, save_path):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[f"{class_names[true]}\nvs\n{class_names[pred]}" 
                  for true, pred in error_counts.keys()], 
                y=list(error_counts.values()))
    plt.title("Classification Errors by Class Similarity")
    plt.xlabel("True Class vs Predicted Class")
    plt.ylabel("Error Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def test(net, testloader, criterion, device, class_sim_matrix, class_names,args):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    
    # 记录分类错误（真实类别，预测类别）-> 错误次数
    error_dict = {}
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            
            # 记录错误分类
            mask = preds != label
            for true, pred in zip(label[mask].cpu().numpy(), preds[mask].cpu().numpy()):
                key = (true, pred)
                error_dict[key] = error_dict.get(key, 0) + 1
            
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # 生成错误分布图
    error_counts = {}
    for (true, pred), count in error_dict.items():
        # 计算真实类别与预测类别的相似度排名
        sim_scores = class_sim_matrix[true]
        sorted_classes = torch.argsort(sim_scores, descending=True)
        rank = (sorted_classes == pred).nonzero().item()
        error_counts[(true, pred, rank)] = count
    
    # 按照相似度排名分组统计
    rank_error_counts = {}
    for (true, pred, rank), count in error_counts.items():
        rank_error_counts[rank] = rank_error_counts.get(rank, 0) + count
    
    # 绘制图表
    plt.figure(figsize=(10, 6))
    ranks = sorted(rank_error_counts.keys())
    counts = [rank_error_counts[r] for r in ranks]
    plt.bar(ranks, counts)
    plt.title("Classification Errors by Class Similarity Rank")
    plt.xlabel("Similarity Rank (0 = most similar)")
    plt.ylabel("Error Count")
    plt.savefig(os.path.join(args.checkpoint, "error_distribution.png"))
    plt.close()
    
    # 保存错误详细信息
    with open(os.path.join(args.checkpoint, "error_details.txt"), "w") as f:
        f.write("TrueClass\tPredClass\tSimilarityRank\tCount\n")
        for (true, pred, rank), count in sorted(error_counts.items(), key=lambda x: -x[1]):
            f.write(f"{class_names[true]}\t{class_names[pred]}\t{rank}\t{count}\n")
    
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss/(batch_idx+1))),
        "acc": float("%.3f" % (100.*metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100.*metrics.balanced_accuracy_score(test_true, test_pred))),
        "confusion_matrix": metrics.confusion_matrix(test_true, test_pred)
    }

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    device = 'cuda'
    
    # 创建输出目录
    args.checkpoint = 'test_results'
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    
    # 日志设置
    logging.basicConfig(
        filename=os.path.join(args.checkpoint, 'test_log.txt'),
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
    net = models.__dict__[args.model](num_classes=args.num_classes)
    net = net.to(device)
    
    # 加载预训练权重
    printf(f'==> Loading pretrained model from {args.pretrained_path}')

    parent_dir = os.path.basename(os.path.dirname(os.path.dirname(args.pretrained_path)))
    if parent_dir == 'checkpoints':
        state = torch.load(args.pretrained_path, weights_only=True)['net']
        state = {key[7:]: val for key, val in state.items()}
        net.load_state_dict(state)
    else:
        state = torch.load(args.pretrained_path, weights_only=True)['net']
        net.load_state_dict(state)
    
        # 提取分类器权重并计算相似度
    class_weights = extract_classifier_weights(net)
    sim_matrix = compute_cosine_similarity(class_weights)
    class_mapping = build_similar_label_mapping(sim_matrix)

    # ======== 计算 D_inter（平均类间可分性距离）========
    # D_inter = (1 - M_ij) 的平均值（i != j）
    C = sim_matrix.size(0)
    mask = ~torch.eye(C, dtype=bool)
    d_inter = (1 - sim_matrix[mask]).mean()

    printf(f"D_inter (avg inter-class distance): {d_inter:.6f}")

    # 输出到文件
    with open(os.path.join(args.checkpoint, "d_inter.txt"), "w") as f:
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

    # ========== 新增可视化函数 ==========
    def plot_similarity_heatmap(matrix, class_names, title, save_path):
        """绘制相似度/混淆矩阵热力图"""
        plt.figure(figsize=(20, 18))
        sns.heatmap(
            matrix,
            annot=False,  # 40类太多不显示数值
            cmap="YlOrRd",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"shrink": 0.5}
        )
        plt.title(title, fontsize=16)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    # 绘制类别相似度矩阵
    plot_similarity_heatmap(
        sim_matrix.numpy(), 
        class_names,
        "Class Cosine Similarity Matrix",
        os.path.join(args.checkpoint, "class_similarity.png")
    )

    # 测试数据
    printf('==> Preparing data..')
    test_loader = DataLoader(
        ModelNet40(partition='test', num_points=args.num_points),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    # 执行测试
    printf('==> Testing..')
    test_results = test(
        net, test_loader, cal_loss, device, 
        sim_matrix, class_names, args
    )

    # 打印结果
    printf('\n=== Final Test Results ===')
    printf(f"Test Loss: {test_results['loss']}")
    printf(f"Test Accuracy: {test_results['acc']}%")
    printf(f"Balanced Accuracy: {test_results['acc_avg']}%")

    # 保存混淆矩阵
    np.savetxt(
        os.path.join(args.checkpoint, "confusion_matrix.txt"),
        test_results["confusion_matrix"],
        fmt="%d"
    )

    # 绘制混淆矩阵热力图
    plot_similarity_heatmap(
        test_results["confusion_matrix"],
        class_names,
        "Confusion Matrix",
        os.path.join(args.checkpoint, "confusion_matrix.png")
    )

    printf(f"Results saved to {args.checkpoint}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pointKAN', help='model name')
    parser.add_argument('--pretrained_path', type=str, required=True, help='path to pretrained model')
    parser.add_argument('--num_classes', type=int, default=40, help='number of classes')
    parser.add_argument('--num_points', type=int, default=1024, help='point number')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    return parser.parse_args()

if __name__ == '__main__':
    main()
