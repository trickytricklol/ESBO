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
import math



def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--model', default='pointKAN', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=30, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='min lr')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='decay rate')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='workers')
    parser.add_argument('--pretrained_path', required=True, help='path to pre-trained model (last_checkpoint.pth)')
    parser.add_argument('--num_classes', default=40, type=int, help='default value for classes of Modelnet40')
    return parser.parse_args()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    if args.seed is None:
        args.seed = np.random.randint(1, 100)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."
    device = 'cuda'
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.set_printoptions(10)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        message = time_str
    else:
        message = "-" + args.msg
    args.checkpoint = 'finetune_ckpt/' + args.model + "-" + "finetune" + message + "-" + 'Modelnet40'
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    # Model
    printf(f"args: {args}")
    printf('==> Building model..')
    net = models.__dict__[args.model](num_classes=args.num_classes)
    criterion = cal_loss
    net = net.to(device)
    # criterion = criterion.to(device)

    parent_dir = os.path.basename(os.path.dirname(os.path.dirname(args.pretrained_path)))
    if parent_dir == 'checkpoints':
        state = torch.load(args.pretrained_path, weights_only=True)['net']
        state = {key[7:]: val for key, val in state.items()}
        net.load_state_dict(state)
    else:
        state = torch.load(args.pretrained_path, weights_only=True)['net']
        net.load_state_dict(state)

    best_test_acc = 0.  # best test accuracy
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    optimizer_dict = None

    if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model)
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'Train-Loss', 'Train-acc-B', 'Train-acc',
                          'Valid-Loss', 'Valid-acc-B', 'Valid-acc'])
    else:
        printf(f"Resuming last checkpoint from {args.checkpoint}")
        checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc_avg = checkpoint['best_test_acc_avg']
        best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model, resume=True)
        optimizer_dict = checkpoint['optimizer']

    printf('==> Preparing data..')
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=args.workers,
                             batch_size=args.batch_size // 2, shuffle=False, drop_last=False)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr, last_epoch=start_epoch - 1)

    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(net, train_loader, optimizer, criterion, device,1)  # {"loss", "acc", "acc_avg", "time"}
        test_out = validate(net, test_loader, criterion, device)
        scheduler.step()

        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss

        save_model(
            net, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc,  # best test accuracy
            best_train_acc=best_train_acc,
            best_test_acc_avg=best_test_acc_avg,
            best_train_acc_avg=best_train_acc_avg,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict()
        )
        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["loss"], train_out["acc_avg"], train_out["acc"],
                       test_out["loss"], test_out["acc_avg"], test_out["acc"]])
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")
        printf(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n")
    logger.close()

    printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
    printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")
    printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
    printf(f"++++++++" * 5)


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

import torch
import torch.nn as nn
import torch.nn.functional as F

def pgd_attack_similar_labels(model, inputs, true_labels, target_labels, loss_fn,
                             epsilon=0.005, alpha=0.005, steps=1):
    """
    Per-sample PGD adversarial attack
    """
    original_mode = model.training
    model.eval()  # 固定BN/Dropout等层
    
    ori_inputs = inputs.detach()
    perturbed = ori_inputs.clone()
    
    with torch.no_grad():  # 主计算环境不追踪梯度
        for i in range(len(inputs)):
            # 当前样本初始化
            sample = ori_inputs[i].unsqueeze(0).requires_grad_(True)
            target = target_labels[i].unsqueeze(0)
            
            # PGD迭代
            for _ in range(steps):
                # 局部启用梯度计算
                with torch.enable_grad():
                    outputs = model(sample)
                    loss = loss_fn(outputs, target)
                    grad = torch.autograd.grad(loss, sample, 
                                            retain_graph=False,
                                            create_graph=False)[0]
                
                # 更新扰动
                sample = sample.detach() - alpha * grad.sign()
                sample = torch.clamp(sample, 
                                   min=ori_inputs[i].unsqueeze(0)-epsilon,
                                   max=ori_inputs[i].unsqueeze(0)+epsilon)
                sample = torch.clamp(sample, min=-1, max=1)  # 数据范围约束
                sample.requires_grad_()
            
            perturbed[i] = sample.squeeze(0)
    
    model.train(original_mode)  # 恢复原始模式
    return perturbed

def train(net, trainloader, optimizer, criterion, device,flag):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label) in enumerate(trainloader):
        if flag == 1:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 1024]
            # 获取模型当前预测结果
            with torch.no_grad():
                outputs = net(data)  # 假设data需要permute
                probs = F.softmax(outputs, dim=1)
                
            target_labels_np = []
            target_label_weights = []

            for i, y_i in enumerate(label):
                # 获取当前样本的预测概率
                sample_probs = probs[i]
                
                # 计算信息熵作为权重（归一化到0-1范围）
                entropy = -torch.sum(sample_probs * torch.log(sample_probs + 1e-10))  # 加1e-10防止log(0)
                normalized_entropy = entropy / torch.log(torch.tensor(probs.size(1)))  # 除以最大可能熵值(log(num_classes))
                weight = normalized_entropy.item()  # 转换为Python标量
                
                # 找到预测概率最高的两个类别
                top2_preds = torch.topk(sample_probs, k=2)
                
                # 如果最高预测就是真实标签，则取第二高预测作为目标
                if top2_preds.indices[0] == y_i:
                    target_label = top2_preds.indices[1].item()
                else:
                    target_label = top2_preds.indices[0].item()
                    weight = 2 - weight
                
                target_labels_np.append(target_label)
                target_label_weights.append(weight)  # 使用信息熵作为权重

            target_labels = torch.tensor(target_labels_np, dtype=torch.long).to(device)
            target_label_weights = torch.tensor(target_label_weights, dtype=torch.float32).to(device)


            x_adv = pgd_attack_similar_labels(net, data.clone(), label, target_labels,
                                    loss_fn=cal_loss,
                                    epsilon=0.05, alpha=0.05, steps=1)

            
            optimizer.zero_grad()
            logits = net(x_adv)
            loss = criterion(logits, label, mean=False)
            weighted_loss = (loss * target_label_weights).mean()  # 加权损失s)
            weighted_loss.backward()
            optimizer.step()
            train_loss += weighted_loss.item()
        else:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 1024]
            optimizer.zero_grad()
            logits = net(data)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        preds = logits.max(dim=1)[1]

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }


def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


if __name__ == '__main__':
    main()
