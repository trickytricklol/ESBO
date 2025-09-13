"""
for training with resume functions.
Usage:
python main.py --model PointNet --msg demo
or
CUDA_VISIBLE_DEVICES=0 nohup python main.py --model PointNet --msg demo > nohup/PointNet_demo.out &
"""
import argparse
import os
import logging
import datetime
import torch
import torch.optim
from torch.utils.data import DataLoader
import models as models
import torch.nn.functional as F
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from ScanObjectNN import ScanObjectNN
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np


def parse_args():
    """Parameters"""

    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size in training')
    parser.add_argument('--model', default='pointKAN', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=15, type=int, help='default value for classes of ScanObjectNN')
    parser.add_argument('--epoch', default=30, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='decay rate')
    parser.add_argument('--smoothing', action='store_true', default=False, help='loss smoothing')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='workers')
    parser.add_argument('--pretrained_path', required=True, help='path to pre-trained model (last_checkpoint.pth)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = 'cuda'
        if args.seed is not None:
            torch.cuda.manual_seed(args.seed)
    else:
        device = 'cpu'
        
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg:
        message = "-" + args.msg
    else:
        message = "-" + time_str
    args.checkpoint = 'finetune_ckpt/' + args.model + "-" + "finetune" + message + "-" + 'Scanobjectnn'
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
    train_loader = DataLoader(ScanObjectNN(partition='training', num_points=args.num_points), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=args.workers,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate / 100, last_epoch=start_epoch - 1)

    # test_out = validate(net, test_loader, criterion, device)
    # print('acc:%.2f,avgacc:%.2f'%(test_out["acc"],test_out["acc_avg"]))

    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(net, train_loader, optimizer, criterion, device,1)  # {"loss", "acc", "acc_avg", "time"}
        # train_out = train(net, train_loader, optimizer, criterion, device,0)
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

# def pgd_attack_similar_labels(model, inputs, true_labels, target_labels, loss_fn,
#                                epsilon=0.01, alpha=0.005, steps=7):
#     ori_inputs = inputs.clone().detach()
#     perturbed = ori_inputs.clone().detach().requires_grad_(True)

#     for _ in range(steps):
#         outputs = model(perturbed)
#         loss = loss_fn(outputs, target_labels)
#         model.zero_grad()
#         loss.backward()
#         with torch.no_grad():
#             perturbed = perturbed - alpha * perturbed.grad.sign() 
#             perturbation = torch.clamp(perturbed - ori_inputs, min=-epsilon, max=epsilon)
#             perturbed = torch.clamp(ori_inputs + perturbation, min=-1, max=1).detach()
#         perturbed.requires_grad = True

#     return perturbed.detach()

def pgd_attack_similar_labels(model, inputs, true_labels, target_labels, loss_fn,
                             epsilon=0.005, alpha=0.005, steps=1):
    """
    优化的逐样本PGD对抗攻击
    改进点：
    1. 使用torch.autograd.grad替代backward()避免干扰模型梯度
    2. 减少不必要的张量克隆
    3. 更安全的梯度计算环境管理
    参数:
        model: 目标模型 [nn.Module]
        inputs: 输入张量 [B, C, N]
        true_labels: 真实标签 [B]
        target_labels: 目标攻击标签 [B] 
        loss_fn: 损失函数
        epsilon: 扰动范围 (建议: 0.01-0.1)
        alpha: 单步扰动强度 (建议: epsilon/steps)
        steps: 攻击步数 (建议: 3-10)
    返回:
        对抗样本 [B, C, N]
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
            # class_weights = extract_classifier_weights(net)
            # sim_matrix = compute_cosine_similarity(class_weights)
            # similar_label_map = build_similar_label_mapping(sim_matrix, topk=5)
            
            # target_labels_np = []
            # target_label_weights = []  # 用于存储每个样本的权重
            # for y_i in label:
            #     similar_labels = [l for l in similar_label_map[y_i.item()] if l != y_i.item()]
            #     target_label = np.random.choice(similar_labels)
            #     target_labels_np.append(target_label)
            #     target_label_index = similar_labels.index(target_label)
            #     target_label_weights.append(0.5 ** target_label_index)  # 计算权重
            # target_labels = torch.tensor(target_labels_np, dtype=torch.long).to(device)
            # target_label_weights = torch.tensor(target_label_weights, dtype=torch.float32).to(device)
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
