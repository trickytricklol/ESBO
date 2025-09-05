"""
Visualize adversarial examples for point clouds (before & after attack)
Author: Adapted from original code
Date: 2023
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_utils.ScanObjectNN import ScanObjectNN
from torch.utils.data import DataLoader
import argparse
import importlib
import sys
import provider

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    parser = argparse.ArgumentParser('Visualize adversarial point clouds')
    parser.add_argument('--model', default='pointnet_cls', help='Model name')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_normals', action='store_true', default=False, help='Use normals')
    parser.add_argument('--num_category', default=15, type=int, help='Number of categories')
    parser.add_argument('--pretrained_path', required=True, help='Path to pretrained model')
    parser.add_argument('--output_dir', default='./adv_visualization', help='Output directory')
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of sample to visualize')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='Use CPU')
    return parser.parse_args()

def load_model(args):
    """Load pretrained model"""
    model = importlib.import_module(args.model)
    classifier = model.get_model(args.num_category, normal_channel=args.use_normals)
    
    if not args.use_cpu:
        classifier = classifier.cuda()
    
    state = torch.load(args.pretrained_path)
    classifier.load_state_dict(state['model_state_dict'])
    classifier.eval()
    return classifier

def pgd_attack(model, inputs, target_labels, epsilon=0.01, alpha=0.005, steps=2):
    """Generate adversarial examples using PGD"""
    perturbed = inputs.clone()
    
    for _ in range(steps):
        perturbed = perturbed.requires_grad_()
        with torch.enable_grad():
            outputs, _ = model(perturbed)
            loss = torch.nn.functional.cross_entropy(outputs, target_labels)
            grad = torch.autograd.grad(loss, perturbed)[0]
        
        with torch.no_grad():
            perturbed = perturbed.detach() - alpha * grad.sign()
            perturbed = torch.clamp(perturbed, 
                                  min=inputs-epsilon, 
                                  max=inputs+epsilon)
            perturbed = torch.clamp(perturbed, min=-1, max=1)
    
    return perturbed

def plot_point_cloud(points, title, filename):
    """Visualize and save point cloud"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    
    ax.scatter(xs, ys, zs, s=5, c=zs, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and data
    classifier = load_model(args)
    dataset = ScanObjectNN(partition='test', num_points=args.num_point)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get specific sample
    if args.sample_idx >= len(dataset):
        print(f"Warning: sample_idx {args.sample_idx} exceeds dataset size {len(dataset)}")
        args.sample_idx = 0
    
    points, label = dataset[args.sample_idx]
    points = points.unsqueeze(0)  # Add batch dimension
    
    if not args.use_cpu:
        points = points.cuda()
    
    # Get model prediction
    with torch.no_grad():
        outputs, _ = classifier(points.transpose(2, 1))
        pred = outputs.argmax(1).item()
    
    # Find target label (most confusing wrong class)
    probs = torch.softmax(outputs, dim=1)
    top_preds = torch.topk(probs, k=2)
    if top_preds.indices[0][0] == label:
        target_label = top_preds.indices[0][1].item()
    else:
        target_label = top_preds.indices[0][0].item()
    
    # Generate adversarial example
    target_tensor = torch.tensor([target_label], dtype=torch.long)
    if not args.use_cpu:
        target_tensor = target_tensor.cuda()
    
    adv_points = pgd_attack(
        classifier, 
        points.transpose(2, 1).clone(), 
        target_tensor,
        epsilon=0.01,
        alpha=0.005,
        steps=2
    )
    
    # Convert to numpy for visualization
    original_pts = points.squeeze(0).cpu().numpy()
    adv_pts = adv_points.squeeze(0).transpose(1, 0).cpu().numpy()
    
    # Visualize and save
    original_title = f"Original (Label: {label}, Pred: {pred})"
    adv_title = f"Adversarial (Target: {target_label})"
    
    original_path = os.path.join(args.output_dir, f"sample_{args.sample_idx}_original.png")
    adv_path = os.path.join(args.output_dir, f"sample_{args.sample_idx}_adversarial.png")
    
    plot_point_cloud(original_pts, original_title, original_path)
    plot_point_cloud(adv_pts, adv_title, adv_path)
    
    print(f"Visualizations saved to {args.output_dir}")
    print(f"Original label: {label}, Model prediction: {pred}")
    print(f"Adversarial target: {target_label}")

if __name__ == '__main__':
    main()