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
    parser.add_argument('--pretrained_path', required=True, help='path to pre-trained model (last_checkpoint.pth)')
    parser.add_argument('--save_path', default='finetune_ckpt', help='path to save finetuned model')
    parser.add_argument('--msg', default='finetune', help='message appended to save path')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
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
    # torch.autograd.set_detect_anomaly(True)  # 在训练前调用
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True

    save_dir = os.path.join(args.save_path, f'{args.model}_{args.msg}')
    mkdir_p(save_dir)

    logger = Logger(os.path.join(save_dir, 'log.txt'), title='FineTune-' + args.model)
    logger.set_names(["Epoch", "Train Loss", "Train Acc", "Train Acc Avg","test_acc"])

    # Model
    net = models.__dict__[args.model](num_classes=args.num_classes)
    net = net.to(device)

    for module in net.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0  # 将 Dropout 的概率设置为 0

    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    parent_dir = os.path.basename(os.path.dirname(os.path.dirname(args.pretrained_path)))
    if parent_dir == 'checkpoints':
        state = torch.load(args.pretrained_path, weights_only=True)['net']
        state = {key[7:]: val for key, val in state.items()}
        net.load_state_dict(state)
    else:
        state = torch.load(args.pretrained_path, weights_only=True)['net']
        net.load_state_dict(state)


    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate / 10)

    # Datasets (use subset for fine-tuning)


    train_loader = DataLoader(ScanObjectNN(partition='training', num_points=args.num_points),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    best_acc = 0
    # test_acc = evaluate_model(net, test_loader, device)
    # print(f"Test Accuracy: {test_acc:.4f}")

    for epoch in range(args.epoch):
        print(f'\nEpoch {epoch + 1}/{args.epoch}')
        train_out = run_epoch(net, train_loader, args.top_k, optimizer, device)
        # test_out = run_epoch(net, test_loader, args.top_k, device=device, train=False)
        scheduler.step()

        test_acc = evaluate_model(net, test_loader, device)
        print(f"Test Accuracy: {test_acc:.4f}")
        logger.append([epoch,train_out["loss"], train_out["acc"], train_out["acc_avg"],test_acc])
        
        if test_acc > best_acc:
            save_model(net, epoch, path=save_dir, acc=train_out["acc"], is_best=True)
            best_acc = test_acc
        else:
            save_model(net, epoch, path=save_dir, acc=train_out["acc"], is_best=False)
        

        print(f"Train Loss: {train_out['loss']} Acc: {train_out['acc']} Acc_B: {train_out['acc_avg']}")

    logger.close()
    print("Fine-tuning complete.")
    test_acc = evaluate_model(net, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")


def extract_classifier_weights(net):
    # 检查是否使用了 DataParallel
    if isinstance(net, torch.nn.DataParallel):
        net = net.module  # 获取原始模型
    # 提取最后一层的权重
    classifier_weights = net.classifier[-1].weight.detach().cpu()
    return classifier_weights

def compute_cosine_similarity(class_weights):
    normed = F.normalize(class_weights, dim=1)
    sim_matrix = torch.mm(normed, normed.t())
    return sim_matrix

def build_similar_label_mapping(sim_matrix, topk=10):
    mapping = {}
    for i in range(sim_matrix.size(0)):
        sim_scores = sim_matrix[i].clone()
        sim_scores[i] = -1e6  # 排除自己
        topk_indices = torch.topk(sim_scores, k=topk).indices.tolist()
        mapping[i] = topk_indices
    return mapping

def pgd_attack_similar_labels(model, inputs, true_labels, target_labels, loss_fn,
                               epsilon=0.03, alpha=0.01, steps=5):
    ori_inputs = inputs.clone().detach()
    perturbed = ori_inputs.clone().detach().requires_grad_(True)

    for _ in range(steps):
        outputs = model(perturbed)
        loss = loss_fn(outputs, target_labels)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            perturbed = perturbed - alpha * perturbed.grad.sign() 
            perturbation = torch.clamp(perturbed - ori_inputs, min=-epsilon, max=epsilon)
            perturbed = torch.clamp(ori_inputs + perturbation, min=-1, max=1).detach()
        perturbed.requires_grad = True

    return perturbed.detach()


def run_epoch(net, dataloader, top_k, optimizer=None, device='cpu'):

    net.train()


    total_loss = 0
    total_correct = 0
    total_seen = 0
    true_labels = []
    pred_labels = []
    time_cost = datetime.datetime.now()

    for i, (points, labels) in enumerate(dataloader):
        class_weights = extract_classifier_weights(net)
        sim_matrix = compute_cosine_similarity(class_weights)
        similar_label_map = build_similar_label_mapping(sim_matrix, topk=top_k)

        x = points.permute(0, 2, 1).to(device)
        y = labels.to(device).squeeze()

        assert not torch.isnan(x).any(), "Input contains NaN!"

        target_labels_np = []
        target_label_weights = []  # 用于存储每个样本的权重
        for y_i in y:
            similar_labels = [l for l in similar_label_map[y_i.item()] if l != y_i.item()]
            target_label = np.random.choice(similar_labels)
            target_labels_np.append(target_label)
            target_label_index = similar_labels.index(target_label)
            target_label_weights.append(0.5 ** target_label_index)  # 计算权重

        target_labels = torch.tensor(target_labels_np, dtype=torch.long).to(device)
        target_label_weights = torch.tensor(target_label_weights, dtype=torch.float32).to(device)

        x_adv = pgd_attack_similar_labels(net, x.clone(), y, target_labels,
                                        loss_fn=cal_loss,
                                        epsilon=0.03, alpha=0.01, steps=7)
        
        outputs = net(x)
        loss = cal_loss(outputs, y)  # 计算每个样本的损失
        # weighted_loss = (loss * target_label_weights).mean()  # 加权损失s)
        print(f"Outputs - min: {outputs.min().item():.4f}, max: {outputs.max().item():.4f}")
        print(f"Loss: {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()


        preds = outputs.max(dim=1)[1]
        total_loss += loss.item()
        total_correct += preds.eq(y).sum().item()
        total_seen += y.size(0)

        true_labels.append(y.cpu().numpy())
        pred_labels.append(preds.detach().cpu().numpy())

        progress_bar(i, len(dataloader), 'Loss: %.3f | Acc: %.3f%%' %
                     (total_loss / (i + 1), 100. * total_correct / total_seen))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)

    return {
        "loss": float("%.3f" % (total_loss / (i + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(true_labels, pred_labels))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(true_labels, pred_labels))),
        "time": time_cost
    }


if __name__ == '__main__':
    main()
