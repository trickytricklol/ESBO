"""
ModelNet-C 评估脚本 (精简版)
功能：评估模型在ModelNet-C所有损坏类型和严重程度上的表现
"""
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# 添加模型路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# 动态导入模型
try:
    import models
except ImportError:
    print("警告: 无法导入models模块，请确保模型定义正确")

class ModelNetCDataLoader(Dataset):
    """ModelNet-C数据加载器"""
    def __init__(self, data_dir, corruption_type='uniform', severity=1, num_points=1024):
        self.num_points = num_points
        self.data_file = os.path.join(data_dir, f"data_{corruption_type}_{severity}.npy")
        self.label_file = os.path.join(data_dir, "label.npy")
        
        # 检查文件是否存在
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"数据文件不存在: {self.data_file}")
        if not os.path.exists(self.label_file):
            raise FileNotFoundError(f"标签文件不存在: {self.label_file}")
        
        # 加载数据
        self.points = np.load(self.data_file)  # [N, 1024, 3]
        self.labels = np.load(self.label_file).astype(np.int64)
        
        # 验证数据
        if len(self.points) != len(self.labels):
            raise ValueError(f"数据与标签数量不匹配: 数据{len(self.points)}个, 标签{len(self.labels)}个")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        # 随机采样固定数量的点
        pt_idxs = np.arange(0, self.points.shape[1])
        np.random.shuffle(pt_idxs)
        
        current_points = self.points[index][pt_idxs[:self.num_points]]
        current_label = self.labels[index]
        
        # 转换为torch tensor
        current_points = torch.from_numpy(current_points).float()
        current_label = torch.from_numpy(np.array([current_label])).long().squeeze()
        
        return current_points, current_label

def parse_args():
    parser = argparse.ArgumentParser('ModelNet-C Evaluation')
    parser.add_argument('--model', type=str, default='pointKAN', help='model name')
    parser.add_argument('--gpu', type=str, default='0', help='GPU设备ID')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--num_point', type=int, default=1024, help='点云数量')
    parser.add_argument('--pretrained_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_root', type=str, default='/mnt/c/Users/Administrator/readmodelnet/modelnet40_c', help='ModelNet-C根目录')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--num_classes', type=int, default=40, help='number of classes')
    return parser.parse_args()

def evaluate(model, loader, device):
    """评估模型在单个数据集上的表现"""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for points, labels in tqdm(loader, desc='Evaluating'):
            points = points.to(device).transpose(2, 1)  # [B, 3, N]
            labels = labels.to(device)
            
            preds = model(points)
            preds = preds.max(dim=1)[1]
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    return accuracy_score(all_targets, all_preds)

def get_available_corruptions(data_root):
    """获取所有可用的损坏类型"""
    corruptions = set()
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"数据目录不存在: {data_root}")
    
    print(f"扫描目录: {data_root}")
    print("目录内容:", os.listdir(data_root))
    
    for fname in os.listdir(data_root):
        if fname.startswith('data_') and fname.endswith('.npy'):
            # 更灵活的文件名解析
            try:
                # 移除前缀和后缀
                base_name = fname[5:-4]  # 移除"data_"和".npy"
                
                # 按最后一个下划线分割
                if '_' in base_name:
                    parts = base_name.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        corruption_type = parts[0]
                        corruptions.add(corruption_type)
                else:
                    print(f"警告: 无法解析文件名格式: {fname}")
                    
            except Exception as e:
                print(f"解析文件名 {fname} 时出错: {e}")
    
    # 如果没有找到，尝试手动列出已知的损坏类型
    if not corruptions:
        print("未通过文件名解析找到损坏类型，尝试手动检查...")
        known_corruptions = [
            'uniform', 'gaussian', 'impulse', 'background', 'upsampling',
            'distortion', 'rotation', 'dropout_global', 'dropout_local',
            'density', 'cutout', 'scale', 'shear', 'taper', 'twist', 'lidar'
        ]
        
        for corruption in known_corruptions:
            test_file = os.path.join(data_root, f"data_{corruption}_1.npy")
            if os.path.exists(test_file):
                corruptions.add(corruption)
                print(f"找到损坏类型: {corruption}")
    
    return sorted(list(corruptions))

def load_model(args, device):
    """加载预训练模型"""
    print('==> Building model..')
    
    # 动态导入模型
    try:
        net = models.__dict__[args.model](num_classes=args.num_classes)
    except KeyError:
        raise ValueError(f"模型 '{args.model}' 未在models模块中定义")
    
    net = net.to(device)
    
    # 加载预训练权重
    print(f'==> Loading pretrained model from {args.pretrained_path}')
    
    if not os.path.exists(args.pretrained_path):
        raise FileNotFoundError(f"模型文件不存在: {args.pretrained_path}")
    
    try:
        checkpoint = torch.load(args.pretrained_path, weights_only=True)
        
        if 'net' in checkpoint:
            state_dict = checkpoint['net']
            # 处理可能的DataParallel前缀
            if all(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key[7:]: value for key, value in state_dict.items()}
            net.load_state_dict(state_dict)
        elif 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            net.load_state_dict(checkpoint)
            
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {str(e)}")
    
    return net

def main():
    args = parse_args()
    
    # 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 加载模型
    model = load_model(args, device)
    
    # 获取所有损坏类型
    try:
        corruptions = get_available_corruptions(args.data_root)
        print(f'找到 {len(corruptions)} 个损坏类型: {corruptions}')
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print(f"请检查数据目录: {args.data_root}")
        return
    
    # 评估结果存储
    results = {}
    
    # 遍历所有损坏类型和严重程度
    for corruption in corruptions:
        results[corruption] = {}
        print(f'\n评估损坏类型: {corruption}')
        
        for severity in range(1, 6):
            try:
                # 检查文件是否存在
                data_file = os.path.join(args.data_root, f"data_{corruption}_{severity}.npy")
                if not os.path.exists(data_file):
                    print(f'  严重程度 {severity}: 文件不存在')
                    results[corruption][severity] = None
                    continue
                
                # 创建数据集和加载器
                dataset = ModelNetCDataLoader(
                    data_dir=args.data_root,
                    corruption_type=corruption,
                    severity=severity,
                    num_points=args.num_point
                )
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.workers
                )
                
                # 评估
                acc = evaluate(model, loader, device)
                results[corruption][severity] = acc
                print(f'  严重程度 {severity}: {acc:.4f}')
                
            except Exception as e:
                print(f'  评估 {corruption} 严重程度 {severity} 时出错: {str(e)}')
                results[corruption][severity] = None
    
    # 保存结果
    result_dir = os.path.dirname(args.pretrained_path)
    result_file = os.path.join(result_dir, 'modelnet_c_results.txt')
    
    with open(result_file, 'w') as f:
        f.write('ModelNet-C Evaluation Results\n')
        f.write('='*50 + '\n')
        f.write(f'Model: {args.pretrained_path}\n')
        f.write(f'Data Root: {args.data_root}\n')
        f.write(f'Batch Size: {args.batch_size}\n')
        f.write(f'Num Points: {args.num_point}\n\n')
        
        # 计算总体统计
        all_accuracies = []
        for corruption in results:
            for severity in results[corruption]:
                if results[corruption][severity] is not None:
                    all_accuracies.append(results[corruption][severity])
        
        if all_accuracies:
            f.write(f'Overall Average Accuracy: {np.mean(all_accuracies):.4f}\n')
            f.write(f'Overall Median Accuracy: {np.median(all_accuracies):.4f}\n\n')
        
        # 写入每个损坏类型的结果
        for corruption in results:
            f.write(f'\nCorruption: {corruption}\n')
            f.write('-' * 30 + '\n')
            for severity in sorted(results[corruption].keys()):
                acc = results[corruption][severity]
                if acc is not None:
                    f.write(f'Severity {severity}: {acc:.4f}\n')
                else:
                    f.write(f'Severity {severity}: N/A\n')
    
    print(f'\n评估完成! 结果已保存到: {result_file}')

if __name__ == '__main__':
    main()
