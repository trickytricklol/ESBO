import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import torch.nn.functional as F
import re
import datetime

from data import ModelNet40, ModelNetCDataLoader
from model import PointNet, DGCNN, pointnet2
from util import cal_loss

def test_modelnet_c(args, io):
    """
    ModelNet-C测试函数
    测试模型在不同损坏类型和严重程度下的鲁棒性
    """
    device = torch.device("cuda" if args.cuda else "cpu")
    
    # 加载模型
    if args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    elif args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'pointnet2':
        model = pointnet2(args).to(device)
    else:
        raise ValueError(f"不支持的模型: {args.model}")
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    
    # 获取所有损坏类型
    corruption_types = get_all_corruption_types(args.modelnet_c_path)
    if not corruption_types:
        io.cprint("错误: 没有找到任何损坏类型数据")
        return
    
    results = {}
    
    # 首先测试原始ModelNet40的干净数据
    io.cprint("\n=== 测试原始ModelNet40干净数据 ===")
    try:
        clean_dataset = ModelNet40(args, partition='test')  # 使用原始ModelNet40测试集
        clean_loader = DataLoader(clean_dataset, batch_size=args.test_batch_size, shuffle=False)
        
        clean_acc = evaluate_modelnet_c(model, clean_loader, device)
        io.cprint(f"原始ModelNet40测试集准确率: {clean_acc:.4f}")
        results['clean'] = clean_acc
    except Exception as e:
        io.cprint(f"测试干净数据时出错: {str(e)}")
        clean_acc = None
    
    # 测试所有损坏类型和严重程度
    io.cprint("\n=== 测试ModelNet-C损坏数据 ===")
    
    for corruption in corruption_types:
        results[corruption] = {}
        io.cprint(f"\n测试损坏类型: {corruption}")
        
        for severity in range(1, 6):  # 严重程度1-5
            try:
                # 检查数据文件是否存在
                data_file = os.path.join(args.modelnet_c_path, f"data_{corruption}_{severity}.npy")
                if not os.path.exists(data_file):
                    io.cprint(f"  严重程度 {severity}: 文件不存在")
                    results[corruption][severity] = None
                    continue
                
                # 创建数据加载器
                dataset = ModelNetCDataLoader(
                    data_dir=args.modelnet_c_path,
                    corruption_type=corruption,
                    severity=severity,
                    num_points=args.num_points
                )
                loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False)
                
                # 评估模型
                acc = evaluate_modelnet_c(model, loader, device)
                results[corruption][severity] = acc
                io.cprint(f"  严重程度 {severity}: 准确率 = {acc:.4f}")
                
            except Exception as e:
                io.cprint(f"  测试 {corruption} 严重程度 {severity} 时出错: {str(e)}")
                results[corruption][severity] = None
    
    # 计算统计指标
    io.cprint("\n=== 测试结果汇总 ===")
    
    if clean_acc is not None:
        # 计算MOA (Mean Accuracy) - 所有损坏数据的平均准确率
        all_corrupted_acc = []
        for corruption in results:
            if corruption != 'clean':
                for severity in results[corruption]:
                    if results[corruption][severity] is not None:
                        all_corrupted_acc.append(results[corruption][severity])
        
        if all_corrupted_acc:
            moa = np.mean(all_corrupted_acc)
            io.cprint(f"MOA (平均准确率): {moa:.4f}")
            
            # 计算MCE (Mean Corruption Error) - 平均损坏误差
            mce = np.mean([1 - (acc / clean_acc) for acc in all_corrupted_acc])
            io.cprint(f"MCE (平均损坏误差): {mce:.4f}")
            
            # 计算mCE (mean Corruption Error) - 标准化平均损坏误差
            # 这里需要每个损坏类型的基准误差，通常需要参考论文中的基准模型
            io.cprint("注: mCE(标准化平均损坏误差)需要基准模型数据来计算")
    
    # 保存详细结果
    save_modelnet_c_results(results, args, io)
    
    return results

def evaluate_modelnet_c(model, data_loader, device):
    """
    评估模型在给定数据加载器上的性能
    """
    model.eval()
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            
            logits = model(data)
            preds = logits.max(dim=1)[1]
            
            all_true.append(label.cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())
    
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    
    return metrics.accuracy_score(all_true, all_pred)

def get_all_corruption_types(data_path):
    """
    获取所有可用的损坏类型
    """
    corruption_types = set()
    
    if not os.path.exists(data_path):
        return []
    
    for filename in os.listdir(data_path):
        if filename.startswith('data_') and filename.endswith('.npy'):
            # 匹配 data_xxx_?.npy 格式
            match = re.match(r'data_([a-zA-Z0-9_]+)_(\d+)\.npy', filename)
            if match:
                corruption_type = match.group(1)
                if corruption_type != 'clean':  # 排除干净数据
                    corruption_types.add(corruption_type)
    
    return sorted(list(corruption_types))

def save_modelnet_c_results(results, args, io):
    """
    保存ModelNet-C测试结果到文件
    """
    result_file = os.path.join('checkpoints',args.exp_name, 'modelnet_c_results.txt')
    
    with open(result_file, 'w') as f:
        f.write("ModelNet-C Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 干净数据结果
        if 'clean' in results:
            f.write(f"Clean Data Accuracy: {results['clean']:.4f}\n\n")
        
        # 损坏数据结果
        for corruption in results:
            if corruption == 'clean':
                continue
                
            f.write(f"Corruption: {corruption}\n")
            f.write("-" * 30 + "\n")
            
            for severity in sorted(results[corruption].keys()):
                acc = results[corruption][severity]
                if acc is not None:
                    f.write(f"Severity {severity}: {acc:.4f}\n")
                else:
                    f.write(f"Severity {severity}: N/A\n")
            f.write("\n")
        
        # 计算统计指标
        if 'clean' in results and results['clean'] is not None:
            all_acc = []
            for corruption in results:
                if corruption != 'clean':
                    for severity in results[corruption]:
                        if results[corruption][severity] is not None:
                            all_acc.append(results[corruption][severity])
            
            if all_acc:
                moa = np.mean(all_acc)
                mce = np.mean([1 - (acc / results['clean']) for acc in all_acc])
                
                f.write(f"MOA (Mean Accuracy): {moa:.4f}\n")
                f.write(f"MCE (Mean Corruption Error): {mce:.4f}\n")
    
    io.cprint(f"详细结果已保存到: {result_file}")