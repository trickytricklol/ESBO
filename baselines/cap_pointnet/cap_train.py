import torch
import numpy as np
import datetime
import logging
import importlib
from pathlib import Path
from tqdm import tqdm
from data_utils.cap_dataloader import CAPModelNetDataLoader
from data_utils.modelnet_dataloader import ModelNetDataLoader

# Configuration settings
model_name = "cap_pointnet2"
optimizer_type = "Adam"
batch_size = 1
num_classes = 40
total_epochs = 10
learning_rate = 0.001
num_points = 1024
decay_rate = 1e-4
use_normals = False
lambda_ = 0.5
eps = 1


# Logger setup
def setup_logger(log_dir):
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_dir / f"{model_name}.txt")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def inplace_relu(m):
    if m.__class__.__name__.find("ReLU") != -1:
        m.inplace = True


def test(model, loader, num_classes):
    model.eval()
    mean_correct = []
    class_acc = np.zeros((num_classes, 3))

    for points, target in tqdm(loader):
        points, target = points.to(device), target.to(device)
        points = points.transpose(2, 1)
        pred, _ = model(points)
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

    instance_acc = np.mean(mean_correct)
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    return instance_acc, np.mean(class_acc[:, 2])


if __name__ == "__main__":
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # Directory setup
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_dir = Path("./log/classification") / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Initialize logger
    logger = setup_logger(log_dir)

    # Data Loading
    logger.info("Loading dataset...")
    train_dataset = CAPModelNetDataLoader(root="data", split="train")
    test_dataset = ModelNetDataLoader(root="data", split="test")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=10
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=10
    )

    # Model and Optimizer
    model = importlib.import_module(model_name).get_model(
        num_classes, normal_channel=use_normals
    )
    model.apply(inplace_relu)
    model.to(device)

    criterion = importlib.import_module(model_name).get_loss().to(device)
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay_rate)
        if optimizer_type == "Adam"
        else torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # Training
    logger.info("Start training...")
    best_instance_acc, best_class_acc = 0.0, 0.0
    for epoch in range(total_epochs):
        logger.info(f"Epoch {epoch + 1}/{total_epochs}")
        model.train()
        mean_correct = []

        for points, target in tqdm(train_loader):
            optimizer.zero_grad()
            points, target = points.to(device), target.to(device)
            points = points.transpose(2, 1)

            pred, trans_feat = model(points)
            cls_loss = criterion(pred, target.long(), trans_feat)

            anchor_feat = trans_feat[0, :]
            same_feat = trans_feat[1:17, :]
            diff_feat = trans_feat[17:, :]
            pos_dist = torch.norm(anchor_feat - same_feat, dim=1)
            neg_dist = torch.norm(anchor_feat - diff_feat, dim=1)

            # Include margin loss
            final_loss = (
                torch.relu((pos_dist.unsqueeze(1) - neg_dist.unsqueeze(0)) + eps).mean()
                + cls_loss
            )
            final_loss.backward()
            optimizer.step()

            correct = pred.data.max(1)[1].eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

        train_instance_acc = np.mean(mean_correct)
        logger.info(f"Train Instance Accuracy: {train_instance_acc:.4f}")

        # Testing
        with torch.no_grad():
            instance_acc, class_acc = test(model, test_loader, num_classes)
            if instance_acc > best_instance_acc:
                best_instance_acc, best_class_acc = instance_acc, class_acc
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoints_dir / "best_model.pth",
                )
                logger.info(f"Saving best model at epoch {epoch + 1}")
            logger.info(
                f"Test Instance Accuracy: {instance_acc:.4f}, Class Accuracy: {class_acc:.4f}"
            )
    logger.info("End of training...")