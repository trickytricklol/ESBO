import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_utils.modelnet_dataloader import ModelNetDataLoader


# Visualization function for point clouds
def visualize_point_cloud(point_cloud, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c="r", marker="o"
    )
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    ax.set_title(title)
    plt.show()


# Iterative Fast Gradient Method (IFGM) attack
def ifgm(x_pl, model, alpha, t_pl=None, iter=10, eps=0.5, atk_type="all"):
    targeted = False
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

        # Clip perturbation if exceeds eps
        perturb_diff = x_adv - x_pl
        perturb_diff_norm = torch.norm(
            perturb_diff.reshape(x_adv.size(0), -1), p=2, dim=1
        )
        clip = perturb_diff_norm > eps
        if clip.any():
            x_adv = torch.where(
                clip.view(-1, 1, 1),
                x_pl + (perturb_diff * eps / perturb_diff_norm.view(-1, 1, 1)),
                x_adv,
            )

        x_adv = x_adv.detach()

    return x_adv


# Test function to evaluate attack success rate
def test(cap_model, model, loader, num_class=10, atk_type="all", eps=0.5):
    cap_model.eval()
    model.eval()
    correct_cnt, total_cnt = 0, 0
    class_success = {i: {"total": 0, "success": 0} for i in range(num_class)}

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points, target = points.to(device), target.to(device)
        points = points.transpose(2, 1)

        visualize_point_cloud(
            points[0].permute(1, 0).cpu().detach().numpy(),
            title=f"{class_names[target[0].item()]}",
        )
        adv_points = ifgm(
            points, model, alpha=0.02, t_pl=target, iter=50, eps=eps, atk_type=atk_type
        )

        # Evaluate on adversarial examples
        pred, _ = cap_model(adv_points)
        pred_choice = pred.data.max(1)[1]
        visualize_point_cloud(
            adv_points[0].permute(1, 0).cpu().detach().numpy(),
            title=f"After Attack: {class_names[target[0].item()]} -> {class_names[pred_choice[0].item()]}",
        )

        for cat in np.unique(target.cpu()):
            class_success[cat.item()]["total"] += len(points[target == cat])
            class_success[cat.item()]["success"] += (
                (pred_choice[target == cat] == target[target == cat].long())
                .cpu()
                .sum()
                .item()
            )

        correct = pred_choice.eq(target.long()).cpu().sum()
        correct_cnt += correct.item()
        total_cnt += len(points)

    # Calculate success rate per class
    acc = (correct_cnt / total_cnt) * 100
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

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # Load dataset
    test_dataset = ModelNetDataLoader(root="data", split="test", process_data=False)
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=36, shuffle=False, num_workers=10
    )

    # Load models
    cap_model = importlib.import_module("cap_pointnet2")
    cap_classifier = cap_model.get_model(10, normal_channel=False)
    cap_checkpoint = torch.load(
        "log/classification/cap_pointnet/checkpoints/best_model.pth",
        map_location=device,
    )
    cap_classifier.load_state_dict(cap_checkpoint["model_state_dict"])
    cap_classifier.to(device)

    model = importlib.import_module("pointnet2")
    classifier = model.get_model(10, normal_channel=False)
    checkpoint = torch.load("log/classification/pointnet/checkpoints/best_model.pth")
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.to(device)

    # Run test
    acc, class_success = test(
        cap_classifier,
        classifier,
        testDataLoader,
        num_class=10,
        atk_type="all",
        eps=0.5,
    )
    print(f"Attack Success Rate: {acc:.2f}%")

    # Display class-wise success rate
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