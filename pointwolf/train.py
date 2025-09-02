"""
@Origin : main.py by Yue Wang
@Contact: yuewangx@mit.edu
@Time: 2018/10/13 10:39 PM

modified by {Sanghyeok Lee, Sihyeon Kim}
@Contact: {cat0626, sh_bs15}@korea.ac.kr
@File: train.py
@Time: 2021.09.29
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import torch.nn.functional as F
from tqdm import tqdm

from data import ModelNet40
from data2 import ScanObjectNN
from model import PointNet, DGCNN, pointnet2
from util import cal_loss

def calculate_normalized_entropy(pred_data):
    """
    计算归一化熵 H_norm(p) = H(p) / log(C)
    
    参数:
        pred_data: 模型输出的预测概率，形状为 [batch_size, num_classes]
    
    返回:
        normalized_entropy: 归一化熵，形状为 [batch_size]
    """
    # 计算香农熵 H(p) = -sum(p * log(p))
    entropy = -torch.sum(pred_data * torch.log(pred_data + 1e-8), dim=1)
    
    # 获取类别数 C
    num_classes = pred_data.size(1)
    
    # 计算归一化熵 H_norm(p) = H(p) / log(C)
    normalized_entropy = entropy / torch.log(torch.tensor(num_classes, dtype=torch.float32))
    
    return normalized_entropy

def calculate_entropy_weight(normalized_entropy, k=10.0, a=0.5):
    """
    计算基于熵的权重 w1 = 1 / (1 + exp(-k * (H_norm(p) - a)))
    
    参数:
        normalized_entropy: 归一化熵，形状为 [batch_size]
        k: 斜率参数，控制sigmoid函数的陡峭程度
        a: 偏移参数，控制sigmoid函数的中心位置
    
    返回:
        weights: 熵权重，形状为 [batch_size]
    """
    # 计算 w1 = 1 / (1 + exp(-k * (H_norm(p) - a)))
    weights = 1.0 / (1.0 + torch.exp(-k * (normalized_entropy - a)))
    
    return weights


def train_vanilla(args, io):
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(args, partition='train'), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(args, partition='test'), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    else:
        train_loader = DataLoader(ScanObjectNN(args,partition='training'), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(args,partition='test'), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    device = torch.device("cuda" if args.cuda else "cpu")
    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args,args.classes).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args,args.classes).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss
    best_test_acc = 0
    
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in tqdm(train_loader, desc=f"Train Epoch {epoch}", unit="batch"):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            
        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            
            
def train_AugTune(args, io):
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(args, partition='train'), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(args, partition='test'), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    else:
        train_loader = DataLoader(ScanObjectNN(args,partition='training'), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(args,partition='test'), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    device = torch.device("cuda" if args.cuda else "cpu")
    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args,args.classes).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args,args.classes).to(device)
    elif args.model == 'pointnet2':
        model = pointnet2(args,args.classes).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss
    best_test_acc = 0
    
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for origin, data, label in tqdm(train_loader, desc=f"Train Epoch {epoch}", unit="batch"):
            origin, data, label = origin.to(device), data.to(device), label.to(device).squeeze()
            origin = origin.permute(0, 2, 1)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            
            #Forward original & augmented sample to get confidence score
            with torch.no_grad():
                pred_origin = model(origin)
                pred_data = model(data)
            
            
            if args.ka:
                # 计算归一化熵和熵权重
                with torch.no_grad():
                    # 转换为概率分布
                    prob_origin = F.softmax(pred_origin, dim=1)
                    prob_data = F.softmax(pred_data, dim=1)
                    
                    # 计算归一化熵 H_norm(p) = H(p) / log(C)
                    entropy_origin = -torch.sum(prob_origin * torch.log(prob_origin + 1e-8), dim=1)
                    entropy_data = -torch.sum(prob_data * torch.log(prob_data + 1e-8), dim=1)
                    num_classes = pred_origin.shape[-1]
                    normalized_entropy_origin = entropy_origin / torch.log(torch.tensor(num_classes, dtype=torch.float32))
                    normalized_entropy_data = entropy_data / torch.log(torch.tensor(num_classes, dtype=torch.float32))
                    
                    # 计算熵权重 w1 = 1 / (1 + exp(-k * (H_norm(p) - a)))
                    k = args.s  # 斜率参数
                    a = args.u   # 中心参数
                    w_data = 1.0 / (1.0 + torch.exp(-k * (normalized_entropy_data - a)))
                    w_data = w_data.reshape(-1, 1, 1)  # 调整形状为 [B, 1, 1]

                #Tune the Sample with entropy weight
                data = w_data * origin + (1 - w_data) * data


            #Re-normalize Tuned Sample
            data = normalize_point_cloud_batch(data)
            #CDA
            data = translate_pointcloud_batch(data)

            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)

            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            
        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
    outstr = 'best test accuracy is %.6f' % (best_test_acc)
    io.cprint(outstr)


def pgd_attack_similar_labels(model, inputs, target_labels, loss_fn,
                             epsilon=0.005, alpha=0.005, steps=5):
    """ 批量并行化PGD攻击 (推荐) """
    original_mode = model.training
    model.eval()
    
    perturbed = inputs.clone()
    for _ in range(steps):
        perturbed = perturbed.requires_grad_()
        with torch.enable_grad():
            outputs = model(perturbed)
            loss = loss_fn(outputs, target_labels)
            grad = torch.autograd.grad(loss, perturbed)[0]
        
        with torch.no_grad():
            perturbed = perturbed.detach() - alpha * grad.sign()
            perturbed = torch.clamp(perturbed, 
                                   min=inputs-epsilon, 
                                   max=inputs+epsilon)
            perturbed = torch.clamp(perturbed, min=-1, max=1)
    
    model.train(original_mode)
    return perturbed

def cal_loss(pred, gold, smoothing=True, mean = True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1) # gold is the groudtruth label in the dataloader

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)  # the number of feature_dim of the ouput, which is output channels

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        if mean:
            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = -(one_hot * log_prb).sum(dim=1)
    else:
        if mean:
            loss = F.cross_entropy(pred, gold, reduction='none')
        else:
            loss = F.cross_entropy(pred, gold, reduction='none')

    return loss


def train_advf(args, io):
    train_loader = DataLoader(ModelNet40(args, partition='train'), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(args, partition='test'), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    device = torch.device("cuda" if args.cuda else "cpu")
    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args,args.classes).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args,args.classes).to(device)
    elif args.model == 'pointnet2':
        model = pointnet2(args,args.classes).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    model.load_state_dict(torch.load(args.model_path))

    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss
    best_test_acc = 0
    
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        # for origin, data, label in tqdm(train_loader, desc=f"Train Epoch {epoch}", unit="batch"):
        #     origin, data, label = origin.to(device), data.to(device), label.to(device).squeeze()
        #     origin = origin.permute(0, 2, 1)
        #     data = data.permute(0, 2, 1)
        #     batch_size = data.size()[0]
            
        #     #Forward original & augmented sample to get confidence score
        #     with torch.no_grad():
        #         pred_origin = model(origin)
        #         pred_data = model(data)
        #         c_origin = (pred_origin.exp() * F.one_hot(label, pred_origin.shape[-1])).sum(1) #(B)
        #         c_data = (pred_data.exp() * F.one_hot(label, pred_data.shape[-1])).sum(1) #(B)

        #     #Calculate Target Confidence Score
        #     c_target = torch.max((1-args.l) * c_origin, c_data) #(B)
        #     alpha = ((c_target-c_data)/(c_origin-c_data + 1e-4)).unsqueeze(1) 
        #     alpha = torch.clamp(alpha, min=0, max=1).reshape(-1,1,1)

        #     #Tune the Sample with alpha
        #     data = alpha * origin + (1-alpha) * data
        #     #Re-normalize Tuned Sample
        #     data = normalize_point_cloud_batch(data)
        #     #CDA
        #     data = translate_pointcloud_batch(data)
                
        for data, label in tqdm(train_loader, desc=f"Train Epoch {epoch}", unit="batch"):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            
            opt.zero_grad()

            with torch.no_grad():
                outputs = model(data) 
                probs = F.softmax(outputs, dim=1)
                
            target_labels_np = []
            target_label_weights = []

            for i, y_i in enumerate(label):
                # 获取当前样本的预测概率
                sample_probs = probs[i]
                
                # 找到预测概率最高的三个类别
                top_preds = torch.topk(sample_probs, k=3)

                # 过滤掉真实标签，得到候选目标类别
                candidate_indices = []
                candidate_weights = []

                for k in range(len(top_preds)):  # 遍历前3个预测
                    if top_preds.indices[k] != y_i:  # 排除真实标签
                        candidate_indices.append(top_preds.indices[k].item())
                        # 权重计算
                        decayed_weight = torch.exp(torch.tensor(- (k+1) ** 2 / (2 * args.sigma ** 2), dtype=torch.float32))
                        candidate_weights.append(decayed_weight)

                if top_preds.indices[0] != y_i:
                    target_label = candidate_indices[0]
                    final_weight = candidate_weights[0]

                else:
                    selected_idx = torch.randint(0, len(candidate_indices), (1,)).item()
                    target_label = candidate_indices[selected_idx]
                    final_weight = candidate_weights[selected_idx]
                
                
                
                target_labels_np.append(target_label)
                target_label_weights.append(final_weight)  # 使用信息熵乘以衰减作为权重

            target_labels = torch.tensor(target_labels_np, dtype=torch.long).cuda()
            target_label_weights = torch.tensor(target_label_weights, dtype=torch.float32).cuda()

            x_adv = pgd_attack_similar_labels(model, data.clone(), target_labels,
                                    loss_fn=cal_loss,
                                    epsilon=0.02, alpha=0.01, steps=2)


            
            logits = model(x_adv)
            loss = cal_loss(logits, label, mean=False)
            weighted_loss = (loss * target_label_weights).mean()  # 加权损失s)

            weighted_loss.backward()

            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += weighted_loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            
        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)

            
def test(args, io):
    test_loader = DataLoader(ModelNet40(args, partition='test'),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)

def normalize_point_cloud_batch(pointcloud):
    """
    input : 
        pointcloud([B,3,N])
        
    output :
        pointcloud([B,3,N]) : Normalized Pointclouds
    """
    pointcloud = pointcloud - pointcloud.mean(dim=-1, keepdim=True) #(B,3,N)
    scale = 1/torch.sqrt((pointcloud**2).sum(1)).max(axis=1)[0]*0.999999 # (B)
    pointcloud = scale.view(-1, 1, 1) * pointcloud
    return pointcloud


def translate_pointcloud_batch(pointcloud):
    """
    input : 
        pointcloud([B,3,N])
        
    output :
        translated_pointcloud([B,3,N]) : Pointclouds after CDA
    """
    B, _, _ = pointcloud.shape
    
    xyz1 = torch.FloatTensor(B,3,1).uniform_(2./3., 3./2.).to(pointcloud.device)
    xyz2 = torch.FloatTensor(B,3,1).uniform_(-0.2, 0.2).to(pointcloud.device)
       
    translated_pointcloud = xyz1 * pointcloud + xyz2
    return translated_pointcloud