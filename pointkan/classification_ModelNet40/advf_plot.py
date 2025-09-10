import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data import ModelNet40
from torch.utils.data import DataLoader
import models as models

def visualize_adversarial_comparison(original_points, adversarial_points, original_label, target_label, title="Adversarial Attack Comparison"):
    """
    可视化对抗攻击前后的点云对比
    """
    # 转换为numpy数组
    if isinstance(original_points, torch.Tensor):
        original_points = original_points.cpu().numpy()
    if isinstance(adversarial_points, torch.Tensor):
        adversarial_points = adversarial_points.cpu().numpy()
    
    # 创建画布
    fig = plt.figure(figsize=(15, 7))
    
    # 绘制原始点云
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], 
                          c='blue', s=10, alpha=0.6, label=f'Original (Class: {original_label})')
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # 设置统一的坐标轴范围
    all_points = np.concatenate([original_points, adversarial_points], axis=0)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    
    # 绘制对抗样本点云
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(adversarial_points[:, 0], adversarial_points[:, 1], adversarial_points[:, 2], 
                          c='red', s=10, alpha=0.6, label=f'Adversarial (Target: {target_label})')
    ax2.set_title('Adversarial Point Cloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # 设置相同的坐标轴范围以便对比
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig('adversarial_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def pgd_attack_similar_labels(model, inputs, true_labels, target_labels, loss_fn,
                             epsilon=0.05, alpha=0.05, steps=1):
    """
    PGD对抗攻击
    """
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

def cal_loss(pred, gold, smoothing=True, mean=True):
    """计算损失函数"""
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        if mean:
            loss = loss.mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean' if mean else 'none')
    return loss

def main():
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 加载模型
    print('==> Building model..')
    net = models.__dict__['pointKAN'](num_classes=40)
    net = net.to(device)
    
    # 加载预训练权重（这里需要你提供预训练模型路径）
    pretrained_path = '/mnt/c/Users/Administrator/PointKAN-pytorch/finetune_ckpt/pointKAN-finetune-20250626113705-Modelnet40/best_checkpoint.pth'  # 修改为你的模型路径
    state = torch.load(pretrained_path, map_location=device, weights_only=True)['net']
    net.load_state_dict(state)
    net.eval()
    
    # 加载测试数据
    print('==> Loading data..')
    test_loader = DataLoader(ModelNet40(partition='test', num_points=1024), 
                            batch_size=1, shuffle=True, drop_last=False)
    
    # 获取一个样本
    data, label = next(iter(test_loader))
    data, label = data.to(device), label.to(device).squeeze()
    data = data.permute(0, 2, 1)  # [batch, 3, num_points]
    
    print(f"Original sample - Class: {label.item()}")
    
    # 获取模型预测
    with torch.no_grad():
        outputs = net(data)
        probs = F.softmax(outputs, dim=1)
        pred_class = outputs.argmax(dim=1).item()
        print(f"Model prediction - Class: {pred_class}, Confidence: {probs[0][pred_class]:.4f}")
    
    # 选择目标标签（选择预测概率第二高的类别）
    top2_probs, top2_indices = torch.topk(probs[0], k=2)
    if top2_indices[0] == label:
        target_label = top2_indices[1].item()
    else:
        target_label = top2_indices[0].item()
    
    print(f"Target label for attack: {target_label}")
    
    # 生成对抗样本
    target_labels_tensor = torch.tensor([target_label], dtype=torch.long).to(device)
    adversarial_data = pgd_attack_similar_labels(
        net, data.clone(), label, target_labels_tensor,
        loss_fn=cal_loss, epsilon=0.02, alpha=0.01, steps=2
    )
    
    # 验证对抗样本的效果
    with torch.no_grad():
        adv_outputs = net(adversarial_data)
        adv_probs = F.softmax(adv_outputs, dim=1)
        adv_pred_class = adv_outputs.argmax(dim=1).item()
        print(f"Adversarial prediction - Class: {adv_pred_class}, Confidence: {adv_probs[0][adv_pred_class]:.4f}")
    
    # 可视化
    original_points = data[0].permute(1, 0)  # [1024, 3]
    adversarial_points = adversarial_data[0].permute(1, 0)  # [1024, 3]
    
    title = f"Adversarial Attack on Point Cloud\nOriginal: {label.item()} → Target: {target_label} → Actual: {adv_pred_class}"
    visualize_adversarial_comparison(original_points, adversarial_points, label.item(), target_label, title)
    
    # 计算扰动幅度
    perturbation = torch.norm(adversarial_points - original_points, dim=1).mean()
    print(f"Average perturbation magnitude: {perturbation:.6f}")

if __name__ == '__main__':
    main()
