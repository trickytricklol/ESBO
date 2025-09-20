import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import importlib
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
def ifgm(
    x_pl,
    model,
    alpha,
    t_pl=None,
    iter=10,
    eps=0.5,
    clip_min=None,
    clip_max=None,
    atk_type="all",
):
    targeted = True
    ord_fn = lambda x: x / torch.sqrt(
        torch.sum(x**2, dim=list(range(1, x.ndim)), keepdim=True)
    )
    x_adv = x_pl.clone().detach()

    for _ in range(iter):
        x_adv.requires_grad_()
        pred, _ = model(x_adv)
        loss = F.nll_loss(pred, t_pl.long())
        perturb = alpha * ord_fn(
            torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
        )

        x_adv = x_adv - perturb if targeted else x_adv + perturb

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

        if clip_min is not None and clip_max is not None:
            x_adv = torch.clamp(x_adv, clip_min, clip_max)

        x_adv = x_adv.detach()
        if loss.item() < 0.01:
            break

    return x_adv


# Testing function for evaluating attack success rate
def test(
    cap_model, model, loader, num_class=10, atk_type="all", eps=0.5, alpha=0.02, iter=70
):
    cap_model.eval()
    model.eval()
    correct_cnt, total_cnt = 0, 0
    class_attack_success = {i: {"total": 0, "success": 0} for i in range(num_class)}

    for _, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points, target = points.to(device), target.to(device)
        points = points.transpose(2, 1)  # Adjust dimensions for the model

        # Generate a random target label for the attack
        random_label = torch.randint(0, 10, (1,), device=device).item()
        mask = target != random_label
        selected_indices = torch.nonzero(mask).squeeze()
        points, target = points[selected_indices], torch.full(
            (len(points),), random_label, device=device
        )

        if len(points) == 0:
            continue

        adv_points = ifgm(
            points,
            model,
            alpha=alpha,
            t_pl=target,
            iter=iter,
            eps=eps,
            atk_type=atk_type,
        )

        pred, _ = cap_model(adv_points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            class_attack_success[cat.item()]["total"] += len(points[target == cat])
            class_attack_success[cat.item()]["success"] += (
                (pred_choice[target == cat] == target[target == cat].long())
                .cpu()
                .sum()
                .item()
            )

        correct = pred_choice.eq(target.long()).cpu().sum()
        correct_cnt += correct.item()
        total_cnt += len(points)

    # Calculate attack success rate per class
    asr = (correct_cnt / total_cnt) * 100
    for cat in class_attack_success:
        total = class_attack_success[cat]["total"]
        success = class_attack_success[cat]["success"]
        class_attack_success[cat]["asr"] = (success / total) * 100 if total > 0 else 0.0

    return asr, class_attack_success


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
        test_dataset, batch_size=24, shuffle=False, num_workers=10
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

    # Run the test function
    asr, class_attack_success = test(
        cap_classifier,
        classifier,
        testDataLoader,
        num_class=10,
        alpha=0.02,
        iter=50,
        atk_type="each",
        eps=0.03,
    )
    print(f"Attack Success Rate: {asr:.2f}%")

    # Display attack success rate per class
    attack_success_rate_df = pd.DataFrame(
        [
            {
                "Class": class_names[cat],
                "Success": class_attack_success[cat]["success"],
                "Total": class_attack_success[cat]["total"],
                "ASR (%)": f"{class_attack_success[cat]['asr']:.2f}",
            }
            for cat in class_attack_success
        ]
    )
    print(attack_success_rate_df.to_string(index=False))