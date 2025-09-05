"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np
import torch.functional as F
import datetime
import logging
import provider
import importlib
import shutil
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from data_utils.ScanObjectNN import ScanObjectNN
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=15, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=30, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--min_lr', default=0.001, help='min learning rate')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=5e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--pretrained_path',required=True)
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


import torch
import torch.nn as nn
import torch.nn.functional as F

def pgd_attack_similar_labels(model, inputs, target_labels, loss_fn,
                             epsilon=0.005, alpha=0.005, steps=5):
    """ 批量并行化PGD攻击 (推荐) """
    original_mode = model.training
    model.eval()
    
    perturbed = inputs.clone()
    for _ in range(steps):
        perturbed = perturbed.requires_grad_()
        with torch.enable_grad():
            outputs, _ = model(perturbed)
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


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    trainDataLoader = DataLoader(ScanObjectNN(partition='training', num_points=args.num_point), num_workers=10,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    testDataLoader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_point), num_workers=10,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)
    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    state = torch.load(args.pretrained_path)
    classifier.load_state_dict(state['model_state_dict'])
    start_epoch = 0
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate)


    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr, last_epoch=start_epoch - 1)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1).cuda()

            with torch.no_grad():
                outputs, _ = classifier(points)  # 明确解包元组，忽略trans_feat
                probs = F.softmax(outputs, dim=1)
                
            target_labels_np = []
            target_label_weights = []

            for i, y_i in enumerate(target):
                # 获取当前样本的预测概率
                sample_probs = probs[i]
                
                # 计算信息熵作为权重（归一化到0-1范围）
                entropy = -torch.sum(sample_probs * torch.log(sample_probs + 1e-10))  # 加1e-10防止log(0)
                normalized_entropy = entropy / torch.log(torch.tensor(probs.size(1)))  # 除以最大可能熵值(log(num_classes))
                weight = normalized_entropy.item()  # 转换为Python标量

                
                # 找到预测概率最高的三个类别
                top_preds = torch.topk(sample_probs, k=3)

                # 过滤掉真实标签，得到候选目标类别
                candidate_indices = []
                candidate_weights = []

                for k in range(len(top_preds)):  # 遍历前3个预测
                    if top_preds.indices[k] != y_i:  # 排除真实标签
                        candidate_indices.append(top_preds.indices[k].item())
                        # 权重计算：原始weight * (0.5)^(k+1)
                        decayed_weight = 0.5 ** (k + 1)
                        candidate_weights.append(decayed_weight)

                if top_preds.indices[0] != y_i:
                    target_label = candidate_indices[0]
                    final_weight = candidate_weights[0]
                    weight = 2 - weight
                else:
                    selected_idx = torch.randint(0, len(candidate_indices), (1,)).item()
                    target_label = candidate_indices[selected_idx]
                    final_weight = candidate_weights[selected_idx]
                
                
                
                target_labels_np.append(target_label)
                target_label_weights.append(weight*final_weight)  # 使用信息熵乘以衰减作为权重

            target_labels = torch.tensor(target_labels_np, dtype=torch.long).cuda()
            target_label_weights = torch.tensor(target_label_weights, dtype=torch.float32).cuda()

            x_adv = pgd_attack_similar_labels(classifier, points.clone(), target_labels,
                                    loss_fn=cal_loss,
                                    epsilon=0.01, alpha=0.005, steps=2)


            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
            
            logits,trans_feat = classifier(x_adv)
            loss = cal_loss(logits, target.long(), mean=False)
            weighted_loss = (loss * target_label_weights).mean()  # 加权损失s)

            # pred, trans_feat = classifier(points)
            # loss = criterion(pred, target.long(), trans_feat)
            pred_choice = logits.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            weighted_loss.backward()
            optimizer.step()
            global_step += 1

        scheduler.step()
        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)