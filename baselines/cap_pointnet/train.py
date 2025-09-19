import torch
import numpy as np
import sys
import os
import datetime
import logging
import model.provider as provider
import importlib
import shutil

from pathlib import Path
from tqdm import tqdm
from data_utils.modelnet_dataloader import ModelNetDataLoader

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    use_cpu = False
else:
    device = torch.device("cpu")
    use_cpu = True

log_dir = None
model = "pointnet2"
process_data = False
optimizer = "Adam"
batch_size = 24
num_category = 40
total_epoch = 10
learning_rate = 0.001
num_point = 1024
decay_rate = 1e-4
use_normals = False
use_uniform_sample = False

logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)


def log_string(str):
    logger.info(str)
    print(str)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def test(model, loader, num_class=num_category):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not use_cpu:
            points, target = points.to(device), target.to(device)

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = (
                pred_choice[target == cat]
                .eq(target[target == cat].long().data)
                .cpu()
                .sum()
            )
            class_acc[cat, 0] += classacc.item() / float(
                points[target == cat].size()[0]
            )
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = 0

    """CREATE DIR"""
    timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    exp_dir = Path("./log/")
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath("classification")
    exp_dir.mkdir(exist_ok=True)
    if log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath("checkpoints/")
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath("logs/")
    log_dir.mkdir(exist_ok=True)

    """LOG"""
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/%s.txt" % (log_dir, model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    """DATA LOADING"""
    log_string("Load dataset ...")
    data_path = "data"

    train_dataset = ModelNetDataLoader(
        root=data_path, split="train", process_data=process_data
    )
    test_dataset = ModelNetDataLoader(
        root=data_path, split="test", process_data=process_data
    )
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=10
    )
    # modelpath = Path('./model')
    # modelpath = modelpath.joinpath(model)
    # 当前工作目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 添加 model 文件夹到模块搜索路径
    model_dir = os.path.join(current_dir, "model")
    sys.path.append(model_dir)
    """MODEL LOADING"""
    num_class = num_category
    model = importlib.import_module(model)
    # shutil.copy('./pointnet.py', str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    # shutil.copy('./train.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not use_cpu:
        classifier = classifier.to(device)
        criterion = criterion.to(device)

    try:
        checkpoint = torch.load(str(exp_dir) + "/checkpoints/best_model.pth")
        start_epoch = checkpoint["epoch"]
        classifier.load_state_dict(checkpoint["model_state_dict"])
        log_string("Use pretrain model")
    except:
        log_string("No existing model, starting training from scratch...")
        start_epoch = 0

    if optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    """TRANING"""
    logger.info("Start training...")
    for epoch in range(start_epoch, total_epoch):
        log_string("Epoch %d (%d/%s):" % (global_epoch + 1, epoch + 1, total_epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(
            enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9
        ):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not use_cpu:
                points, target = points.to(device), target.to(device)

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string("Train Instance Accuracy: %f" % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(
                classifier.eval(), testDataLoader, num_class=num_class
            )

            if instance_acc >= best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if class_acc >= best_class_acc:
                best_class_acc = class_acc
            log_string(
                "Test Instance Accuracy: %f, Class Accuracy: %f"
                % (instance_acc, class_acc)
            )
            log_string(
                "Best Instance Accuracy: %f, Class Accuracy: %f"
                % (best_instance_acc, best_class_acc)
            )

            if instance_acc >= best_instance_acc:
                logger.info("Save model...")
                savepath = str(checkpoints_dir) + "/best_model.pth"
                log_string("Saving at %s" % savepath)
                state = {
                    "epoch": best_epoch,
                    "instance_acc": instance_acc,
                    "class_acc": class_acc,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info("End of training...")