"""
Author: Benny
Date: Nov 2019
Modified: Added visualization features
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# ========== 新增可视化函数 ==========
def extract_classifier_weights(model):
    """提取分类器最后一层权重"""
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model.classifier[-1].weight.detach().cpu()

def compute_cosine_similarity(weights):
    """计算余弦相似度矩阵"""
    normed = F.normalize(weights, dim=1)
    return torch.mm(normed, normed.t())

def plot_heatmap(matrix, class_names, title, save_path, annot=False):
    """通用热力图绘制函数"""
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        matrix,
        annot=annot,
        fmt=".2f" if annot else "d",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"shrink": 0.75}
    )
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()

def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    all_preds = []
    all_targets = []

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        noise = torch.randn_like(points) * 0.01
        points = points + noise

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        # 收集预测结果用于混淆矩阵
        all_preds.extend(pred_choice.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    
    # 返回新增的混淆矩阵
    return instance_acc, class_acc, np.array(all_targets), np.array(all_preds)

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir
    vis_dir = os.path.join(experiment_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)  # 创建可视化目录

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # ========== 新增：类别相似度分析 ==========
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
    
    # 提取权重并计算相似度
    weights = extract_classifier_weights(classifier)
    sim_matrix = compute_cosine_similarity(weights)
    
    # 绘制并保存相似度矩阵
    plot_heatmap(
        sim_matrix.numpy(),
        class_names,
        "Class Cosine Similarity Matrix",
        os.path.join(vis_dir, 'class_similarity.png')
    )
    log_string(f"Saved similarity matrix to {vis_dir}/class_similarity.png")

    '''TESTING'''
    with torch.no_grad():
        instance_acc, class_acc, all_targets, all_preds = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        
        # ========== 新增：混淆矩阵 ==========
        cm = confusion_matrix(all_targets, all_preds)
        np.savetxt(os.path.join(vis_dir, 'confusion_matrix.txt'), cm, fmt='%d')
        
        plot_heatmap(
            cm,
            class_names,
            "Confusion Matrix",
            os.path.join(vis_dir, 'confusion_matrix.png'),
            annot=False  # 40类太多不显示数值
        )
        log_string(f"Saved confusion matrix to {vis_dir}/confusion_matrix.png")

if __name__ == '__main__':
    args = parse_args()
    main(args)