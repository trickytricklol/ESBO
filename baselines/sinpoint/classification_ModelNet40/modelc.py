"""
ModelNet-C 评估脚本
功能：
1. 支持单个损坏类型+严重程度的评估
2. 支持自动评估所有损坏类型和严重程度
3. 计算MOA（平均准确率）和MCE（平均损坏误差）
"""
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from sklearn.metrics import accuracy_score
import re
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

class ModelNetCDataLoader(torch.utils.data.Dataset):
    """ModelNet-C数据加载器"""
    def __init__(self, data_dir, corruption_type='uniform', severity=1, num_points=1024):
        self.num_points = num_points
        self.data_file = os.path.join(data_dir, f"data_{corruption_type}_{severity}.npy")
        self.label_file = os.path.join(data_dir, "label.npy")
        
        self.points = np.load(self.data_file)
        self.labels = np.load(self.label_file).astype(np.int64)
        
        if len(self.points) != len(self.labels):
            raise ValueError(f"数据与标签数量不匹配: 数据{len(self.points)}个, 标签{len(self.labels)}个")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        pt_idxs = np.arange(0, self.points.shape[1])
        np.random.shuffle(pt_idxs)
        
        current_points = self.points[index][pt_idxs[:self.num_points]]
        current_label = self.labels[index]
        
        current_points = torch.from_numpy(current_points).float()
        current_label = torch.from_numpy(np.array([current_label])).long().squeeze()
        
        return current_points, current_label

def parse_args():
    parser = argparse.ArgumentParser('ModelNet-C Evaluation')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--num_point', type=int, default=1024, help='point number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate scores with voting')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    
    # ModelNet-C specific
    parser.add_argument('--modelnet_c_path', type=str, default='data/modelnet40_c', help='path to ModelNet-C')
    parser.add_argument('--corruption', type=str, default='uniform', help='corruption type')
    parser.add_argument('--severity', type=int, default=1, help='corruption severity (1-5)')
    parser.add_argument('--eval_all', action='store_true', help='evaluate all corruptions and severities')
    return parser.parse_args()

def evaluate(model, loader, num_class=40, vote_num=1):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for points, target in tqdm(loader, desc='Evaluating'):
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], num_class).cuda()

            for _ in range(vote_num):
                pred, _ = model(points)
                vote_pool += pred
            
            pred = vote_pool / vote_num
            pred_choice = pred.data.max(1)[1]

            all_preds.extend(pred_choice.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    return acc

def load_model(args):
    """加载预训练模型"""
    model = importlib.import_module(os.listdir(f'log/classification/{args.log_dir}/logs')[0].split('.')[0])
    classifier = model.get_model(40, normal_channel=args.use_normals)
    
    if not args.use_cpu:
        classifier = classifier.cuda()
    
    checkpoint = torch.load(f'log/classification/{args.log_dir}/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    return classifier

def get_all_corruption_types(data_path):
    """自动检测所有可用的损坏类型"""
    corruption_types = set()
    
    # 检查数据目录下的所有文件
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据目录不存在: {data_path}")
    
    for filename in os.listdir(data_path):
        if filename.startswith('data_') and filename.endswith('.npy'):
            # 使用正则表达式匹配 data_xxx_?.npy 格式
            match = re.match(r'data_([a-zA-Z0-9_]+)_(\d+)\.npy', filename)
            if match:
                corruption_type = match.group(1)
                corruption_types.add(corruption_type)
    
    return sorted(list(corruption_types))

def calculate_moa_mce(results, clean_acc):
    """
    计算MOA（平均准确率）和MCE（平均损坏误差）
    
    Args:
        results: 字典，包含所有损坏类型和严重程度的准确率
        clean_acc: 干净数据的准确率
    
    Returns:
        moa: 平均准确率
        mce: 平均损坏误差
    """
    all_accuracies = []
    all_errors = []
    
    for corruption in results:
        for severity in results[corruption]:
            if results[corruption][severity] is not None:
                acc = results[corruption][severity]
                all_accuracies.append(acc)
                
                # 计算相对误差 (1 - acc/clean_acc)
                if clean_acc > 0:
                    error = 1 - (acc / clean_acc)
                    all_errors.append(error)
    
    moa = np.mean(all_accuracies) if all_accuracies else 0
    mce = np.mean(all_errors) if all_errors else 0
    
    return moa, mce

def evaluate_single(args):
    """评估单个损坏类型和严重程度"""
    # 首先检查该损坏类型是否存在
    available_corruptions = get_all_corruption_types(args.modelnet_c_path)
    if args.corruption not in available_corruptions:
        print(f"错误: 损坏类型 '{args.corruption}' 不存在")
        print(f"可用的损坏类型: {available_corruptions}")
        return
    
    dataset = ModelNetCDataLoader(
        data_dir=args.modelnet_c_path,
        corruption_type=args.corruption,
        severity=args.severity,
        num_points=args.num_point
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    model = load_model(args)
    acc = evaluate(model, loader, vote_num=args.num_votes)
    
    print(f"\nEvaluation Result:")
    print(f"Corruption: {args.corruption}")
    print(f"Severity: {args.severity}")
    print(f"Accuracy: {acc:.4f}")

def evaluate_all(args):
    """评估所有损坏类型和严重程度"""
    # 自动检测所有可用的损坏类型
    available_corruptions = get_all_corruption_types(args.modelnet_c_path)
    
    if not available_corruptions:
        print("错误: 没有找到任何损坏类型数据文件")
        print(f"请检查目录: {args.modelnet_c_path}")
        print("目录内容:", os.listdir(args.modelnet_c_path))
        return
    
    print(f"找到 {len(available_corruptions)} 个可用的损坏类型:")
    for i, corruption in enumerate(available_corruptions, 1):
        print(f"{i}. {corruption}")
    
    # 准备结果记录
    results = {}
    model = load_model(args)
    
    # 首先评估干净数据（severity=0）
    print("\n评估干净数据...")
    try:
        # 检查干净数据文件
        clean_data_file = os.path.join(args.modelnet_c_path, "data_clean.npy")
        if os.path.exists(clean_data_file):
            clean_dataset = ModelNetCDataLoader(
                data_dir=args.modelnet_c_path,
                corruption_type='clean',
                severity=0,
                num_points=args.num_point
            )
            clean_loader = torch.utils.data.DataLoader(
                clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
            clean_acc = evaluate(model, clean_loader, vote_num=args.num_votes)
            print(f"干净数据准确率: {clean_acc:.4f}")
        else:
            print("警告: 未找到干净数据文件 'data_clean.npy'")
            clean_acc = None
    except Exception as e:
        print(f"评估干净数据时出错: {str(e)}")
        clean_acc = None
    
    # 遍历所有损坏类型和严重程度
    for corruption in available_corruptions:
        results[corruption] = {}
        print(f"\n评估损坏类型: {corruption}")
        
        for severity in range(1, 6):
            try:
                # 检查文件是否存在
                data_file = os.path.join(args.modelnet_c_path, f"data_{corruption}_{severity}.npy")
                if not os.path.exists(data_file):
                    print(f"  严重程度 {severity}: 文件不存在")
                    results[corruption][severity] = None
                    continue
                
                dataset = ModelNetCDataLoader(
                    data_dir=args.modelnet_c_path,
                    corruption_type=corruption,
                    severity=severity,
                    num_points=args.num_point
                )
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
                )
                
                acc = evaluate(model, loader, vote_num=args.num_votes)
                results[corruption][severity] = acc
                
                print(f"  严重程度 {severity}: {acc:.4f}")
                
            except Exception as e:
                print(f"  评估 {corruption} 严重程度 {severity} 时出错: {str(e)}")
                results[corruption][severity] = None
    
    # 计算MOA和MCE
    if clean_acc is not None:
        moa, mce = calculate_moa_mce(results, clean_acc)
        print(f"\nMOA (平均准确率): {moa:.4f}")
        print(f"MCE (平均损坏误差): {mce:.4f}")
    else:
        moa = np.mean([acc for corruption in results for severity in results[corruption] 
                      if results[corruption][severity] is not None])
        print(f"\nMOA (平均准确率): {moa:.4f}")
        print("MCE (平均损坏误差): 无法计算（缺少干净数据准确率）")
        mce = None
    
    # 保存结果到文件
    result_file = os.path.join(f'log/classification/{args.log_dir}', 'modelnet_c_results.txt')
    with open(result_file, 'w') as f:
        f.write("ModelNet-C Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"数据目录: {args.modelnet_c_path}\n")
        f.write(f"评估时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if clean_acc is not None:
            f.write(f"干净数据准确率: {clean_acc:.4f}\n")
        f.write(f"MOA (平均准确率): {moa:.4f}\n")
        if mce is not None:
            f.write(f"MCE (平均损坏误差): {mce:.4f}\n")
        f.write("\n")
        
        # 计算平均准确率
        all_accuracies = []
        for corruption in results:
            for severity in results[corruption]:
                if results[corruption][severity] is not None:
                    all_accuracies.append(results[corruption][severity])
        
        if all_accuracies:
            f.write(f"平均准确率: {np.mean(all_accuracies):.4f}\n\n")
        
        for corruption in results:
            f.write(f"\nCorruption: {corruption}\n")
            f.write("-" * 30 + "\n")
            for severity in sorted(results[corruption].keys()):
                acc = results[corruption][severity]
                if acc is not None:
                    f.write(f'Severity {severity}: Accuracy={acc:.4f}\n')
                else:
                    f.write(f'Severity {severity}: N/A\n')
    
    print(f"\n所有评估完成! 结果已保存到 {result_file}")

if __name__ == '__main__':
    args = parse_args()
    
    if args.eval_all:
        evaluate_all(args)
    else:
        evaluate_single(args)
