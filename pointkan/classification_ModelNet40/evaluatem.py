import argparse
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys
import logging
sys.path.append('.')
import models as models

class ModelNetCDataLoader(Dataset):
    """ModelNet-C数据加载器"""
    def __init__(self, data_dir, corruption_type='uniform', severity=1, num_points=1024):
        self.num_points = num_points
        self.data_file = os.path.join(data_dir, f"data_{corruption_type}_{severity}.npy")
        self.label_file = os.path.join(data_dir, "label.npy")
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"数据文件不存在: {self.data_file}")
        if not os.path.exists(self.label_file):
            raise FileNotFoundError(f"标签文件不存在: {self.label_file}")
        self.points = np.load(self.data_file)
        self.labels = np.load(self.label_file).astype(np.int64)
        if len(self.points) != len(self.labels):
            raise ValueError(f"数据与标签数量不匹配")

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
    parser.add_argument('--model', type=str, default='pointKAN', help='model name')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_point', type=int, default=1024)
    parser.add_argument('--pretrained_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_root', type=str, default='/mnt/c/Users/Administrator/readmodelnet/modelnet40_c')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=40)
    return parser.parse_args()


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for points, labels in tqdm(loader, desc='Evaluating'):
            points = points.to(device).transpose(2, 1)
            labels = labels.to(device)
            preds = model(points)
            preds = preds.max(dim=1)[1]
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    return accuracy_score(all_targets, all_preds)


def get_available_corruptions(data_root):
    corruptions = set()
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"数据目录不存在: {data_root}")
    for fname in os.listdir(data_root):
        if fname.startswith('data_') and fname.endswith('.npy'):
            try:
                base_name = fname[5:-4]
                if '_' in base_name:
                    parts = base_name.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        corruptions.add(parts[0])
            except Exception:
                pass
    if not corruptions:
        known_corruptions = [
            'uniform', 'gaussian', 'impulse', 'background', 'upsampling',
            'distortion', 'rotation', 'dropout_global', 'dropout_local',
            'density', 'cutout', 'scale', 'shear', 'taper', 'twist', 'lidar'
        ]
        for corruption in known_corruptions:
            if os.path.exists(os.path.join(data_root, f"data_{corruption}_1.npy")):
                corruptions.add(corruption)
    return sorted(list(corruptions))


def load_model(args, device):
    net = models.__dict__[args.model](num_classes=args.num_classes)
    net = net.to(device)
    checkpoint = torch.load(args.pretrained_path, weights_only=True)
    if 'net' in checkpoint:
        state_dict = checkpoint['net']
        if all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key[7:]: value for key, value in state_dict.items()}
        net.load_state_dict(state_dict)
    elif 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint)
    return net


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)
    corruptions = get_available_corruptions(args.data_root)
    print(f'找到 {len(corruptions)} 个损坏类型: {corruptions}')
    results = {}
    for corruption in corruptions:
        results[corruption] = {}
        print(f'\n评估损坏类型: {corruption}')
        for severity in range(1, 6):
            try:
                data_file = os.path.join(args.data_root, f"data_{corruption}_{severity}.npy")
                if not os.path.exists(data_file):
                    results[corruption][severity] = None
                    continue
                dataset = ModelNetCDataLoader(args.data_root, corruption, severity, args.num_point)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
                acc = evaluate(model, loader, device)
                results[corruption][severity] = acc
                print(f'  严重程度 {severity}: {acc:.4f}')
            except Exception as e:
                print(f'  评估 {corruption} 严重程度 {severity} 时出错: {str(e)}')
                results[corruption][severity] = None
    result_dir = os.path.dirname(args.pretrained_path)
    result_file = os.path.join(result_dir, 'modelnet_c_results.txt')
    with open(result_file, 'w') as f:
        f.write('ModelNet-C Evaluation Results\n')
        f.write('='*50 + '\n')
        f.write(f'Model: {args.pretrained_path}\n')
        f.write(f'Data Root: {args.data_root}\n\n')
        all_accuracies = []
        for corruption in results:
            for severity in results[corruption]:
                if results[corruption][severity] is not None:
                    all_accuracies.append(results[corruption][severity])
        if all_accuracies:
            f.write(f'Overall Average Accuracy: {np.mean(all_accuracies):.4f}\n')
            f.write(f'Overall Median Accuracy: {np.median(all_accuracies):.4f}\n\n')
        for corruption in results:
            f.write(f'\nCorruption: {corruption}\n')
            f.write('-' * 30 + '\n')
            for severity in sorted(results[corruption].keys()):
                acc = results[corruption][severity]
                f.write(f'Severity {severity}: {acc:.4f}\n' if acc is not None else f'Severity {severity}: N/A\n')
    print(f'\n评估完成! 结果已保存到: {result_file}')

if __name__ == '__main__':
    main()
