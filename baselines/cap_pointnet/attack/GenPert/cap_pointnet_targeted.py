import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_utils.modelnet_dataloader import ModelNetDataLoader
import importlib
import matplotlib.pyplot as plt


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


def attack_one_batch(
    model,
    attacked_data,
    pert,
    target,
    target_class,
    initial_weight,
    upper_bound_weight,
    binary_search_step,
    num_iterations,
    lr_attack,
    eps=0.5,
):
    device = pert.device
    pert = torch.zeros_like(attacked_data, device=device, requires_grad=True)
    optimizer = optim.Adam([pert], lr=lr_attack)

    weight = torch.ones(attacked_data.size(0), device=device) * initial_weight
    upper_bound = torch.ones(attacked_data.size(0), device=device) * upper_bound_weight
    o_bestdist = torch.ones(attacked_data.size(0), device=device) * 1e10
    o_bestscore = torch.ones(attacked_data.size(0), device=device) * -1
    o_bestattack = torch.ones_like(attacked_data, device=device)

    for _ in range(binary_search_step):
        pert.data = torch.normal(0, 1e-7, size=pert.shape, device=device)
        bestdist = torch.ones(attacked_data.size(0), device=device) * 1e10

        for _ in range(num_iterations):
            optimizer.zero_grad()
            pointclouds_input = attacked_data + pert
            pred, _ = model(pointclouds_input.transpose(2, 1))
            adv_loss = nn.functional.nll_loss(pred, target.long())
            pert_norm = torch.norm(pert.view(attacked_data.size(0), -1), dim=1)
            loss = adv_loss + torch.mean(weight * pert_norm)
            loss.backward()
            optimizer.step()

            pred_val = torch.argmax(pred, dim=1)
            dist_val = pert_norm.detach()

            for e in range(attacked_data.size(0)):
                if dist_val[e] < bestdist[e] and pred_val[e] == target_class:
                    bestdist[e] = dist_val[e]
                if (
                    dist_val[e] < o_bestdist[e]
                    and pred_val[e] == target_class
                    and dist_val[e] <= eps
                ):
                    o_bestdist[e] = dist_val[e]
                    o_bestscore[e] = pred_val[e]
                    o_bestattack[e] = pointclouds_input[e].detach()

        weight = (weight + upper_bound) / 2

    return o_bestattack, o_bestscore


def attack(
    model,
    cap_model,
    test_loader,
    initial_weight,
    upper_bound_weight,
    binary_search_step,
    num_iterations,
    lr_attack,
    num_class,
):
    model.eval()
    cap_model.eval()
    class_attack_success = {i: {"total": 0, "success": 0} for i in range(num_class)}
    correct_cnt, total_cnt = 0, 0

    for points, target in tqdm(test_loader):
        points, target = points.to(device), target.to(device)
        random_label = random.randint(0, num_class - 1)
        mask = target != random_label
        selected_points = points[mask]
        selected_target = target[mask]
        target = torch.full(
            (len(selected_points),), random_label, device=device, dtype=torch.int32
        )

        best_attack, best_score = attack_one_batch(
            model,
            selected_points,
            torch.zeros_like(selected_points),
            target,
            random_label,
            initial_weight,
            upper_bound_weight,
            binary_search_step,
            num_iterations,
            lr_attack,
        )
        pred, _ = cap_model(best_attack)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            class_attack_success[cat.item()]["total"] += len(
                selected_points[target == cat]
            )
            class_attack_success[cat.item()]["success"] += (
                (pred_choice[target == cat] == target[target == cat].long())
                .cpu()
                .sum()
                .item()
            )

        correct_cnt += pred_choice.eq(target.long()).cpu().sum().item()
        total_cnt += len(selected_points)

    asr = (correct_cnt / total_cnt) * 100
    return asr, class_attack_success


def load_model(model_name, num_class, device, path):
    model_module = importlib.import_module(model_name)
    model = model_module.get_model(num_class, normal_channel=False)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


if __name__ == "__main__":
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
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

    test_dataset = ModelNetDataLoader(root="data", split="test", process_data=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=24, shuffle=True, num_workers=10
    )

    model = load_model(
        "pointnet2",
        10,
        device,
        "log/classification/pointnet/checkpoints/best_model.pth",
    )
    cap_model = load_model(
        "cap_pointnet2",
        10,
        device,
        "log/classification/cap_pointnet/checkpoints/best_model.pth",
    )

    asr, class_attack_success = attack(
        model,
        cap_model,
        test_loader,
        initial_weight=10,
        upper_bound_weight=80,
        binary_search_step=5,
        num_iterations=10,
        lr_attack=0.01,
        num_class=10,
    )

    print(f"Attack Success Rate: {asr:.2f}%")
    attack_success_rate_df = pd.DataFrame(
        [
            {
                "Class": class_names[cat],
                "Successful": class_attack_success[cat]["success"],
                "Total": class_attack_success[cat]["total"],
                "ASR (%)": f"{class_attack_success[cat]['asr']:.2f}",
            }
            for cat in class_attack_success
        ]
    )
    print(attack_success_rate_df.to_string(index=False))