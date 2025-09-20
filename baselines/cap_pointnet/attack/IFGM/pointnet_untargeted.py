import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_utils.modelnet_dataloader import ModelNetDataLoader
import importlib


# Define IFGM adversarial attack
def ifgm(
    x_pl,
    model,
    alpha,
    t_pl,
    iter=10,
    eps=0.1,
    clip_min=None,
    clip_max=None,
    targeted=False,
):
    ord_fn = lambda x: x / torch.sqrt(
        torch.sum(x**2, dim=list(range(1, x.ndim)), keepdim=True)
    )
    x_adv = x_pl.clone().detach()

    for _ in range(iter):
        x_adv.requires_grad_()
        pred, _ = model(x_adv)
        loss = F.nll_loss(pred, t_pl.long())

        perturb = alpha * ord_fn(torch.autograd.grad(loss, x_adv)[0])
        x_adv = x_adv - perturb if targeted else x_adv + perturb

        # Clip perturbation within epsilon bounds
        perturb_diff = x_adv - x_pl
        perturb_diff_norm = torch.norm(perturb_diff.view(x_adv.size(0), -1), p=2, dim=1)
        clip = perturb_diff_norm > eps
        if clip.any():
            clip = clip.view(-1, 1, 1)
            perturb_diff_norm = perturb_diff_norm.view(-1, 1, 1)
            x_adv = torch.where(
                clip, x_pl + (perturb_diff * eps / perturb_diff_norm), x_adv
            )

        # Clamp values if min and max bounds are set
        if clip_min is not None and clip_max is not None:
            x_adv = torch.clamp(x_adv, clip_min, clip_max)

        x_adv = x_adv.detach()

    return x_adv


# Visualize a point cloud
def visualize_point_cloud(point_cloud, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c="r", marker="o"
    )
    ax.set_title(title)
    plt.show()


# Test function to evaluate the model's accuracy and adversarial robustness
def test(model, loader, num_class=10):
    correct_cnt, total_cnt = 0, 0
    class_success = {i: {"total": 0, "success": 0} for i in range(num_class)}

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points, target = points.to(device).transpose(2, 1), target.to(device)
        adv_points = ifgm(points, model, alpha=0.02, t_pl=target, iter=50, eps=0.5)

        pred, _ = model(adv_points)
        pred_choice = pred.data.max(1)[1]

        for cat in torch.unique(target.cpu()):
            class_success[cat.item()]["total"] += (target == cat).sum().item()
            class_success[cat.item()]["success"] += (
                (pred_choice[target == cat] == target[target == cat]).sum().item()
            )

        correct = pred_choice.eq(target).sum().item()
        correct_cnt += correct
        total_cnt += len(points)
        print(f"Batch {j}, Accuracy: {100 * correct_cnt / total_cnt:.2f}%")

    acc = correct_cnt / total_cnt
    for cat in class_success:
        total = class_success[cat]["total"]
        success = class_success[cat]["success"]
        class_success[cat]["acc"] = (success / total) * 100 if total > 0 else 0.0

    return acc, class_success


if __name__ == "__main__":
    class_names = {
        0: "Bathtub",
        1: "Bed",
        2: "Chair",
        3: "Desk",
        4: "Dresser",
        5: "Monitor",
        6: "Nightstand",
        7: "Sofa",
        8: "Table",
        9: "Toilet",
    }
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    test_loader = torch.utils.data.DataLoader(
        ModelNetDataLoader(root="data", split="test"),
        batch_size=24,
        shuffle=False,
        num_workers=10,
    )
    model = importlib.import_module("pointnet2").get_model(10, normal_channel=False)
    checkpoint = torch.load(
        "log/classification/2024-08-27_14-30/checkpoints/best_model.pth"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    acc, class_success = test(model, test_loader, num_class=10)
    print(f"Overall Accuracy: {acc * 100:.2f}%")

    # Display results in DataFrame
    print(
        pd.DataFrame(
            [
                {
                    "Class": class_names[cat],
                    "Success": class_success[cat]["success"],
                    "Total": class_success[cat]["total"],
                    "Acc (%)": f"{class_success[cat]['acc']:.2f}",
                }
                for cat in class_success
            ]
        ).to_string(index=False)
    )