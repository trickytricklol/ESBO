import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_utils.modelnet_dataloader import ModelNetDataLoader
import importlib
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Visualization Function
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


# Adversarial Loss Function
def adv_loss_function(logits, target_class):
    target_logit = logits[torch.arange(logits.size(0)), target_class]
    max_non_target_logit = torch.max(
        logits + torch.eye(logits.size(1)).to(logits.device)[target_class] * -1e10,
        dim=1,
    )[0]
    loss = torch.clamp(target_logit - max_non_target_logit, min=0)
    return torch.mean(loss)


# Perform Attack on a Single Batch
def attack_one_batch(
    model,
    attacked_data,
    original_target,
    initial_weight,
    upper_bound_weight,
    binary_search_step,
    num_iterations,
    lr_attack,
    eps=0.5,
):
    device = attacked_data.device
    pert = torch.zeros_like(attacked_data, requires_grad=True, device=device)
    optimizer = optim.Adam([pert], lr=lr_attack)

    # Bounds and best results initialization
    lower_bound, weight, upper_bound = (
        torch.zeros(attacked_data.size(0), device=device),
        initial_weight,
        upper_bound_weight,
    )
    o_bestdist, o_bestattack = torch.ones(
        attacked_data.size(0), device=device
    ) * 1e10, torch.ones_like(attacked_data, device=device)

    for _ in range(binary_search_step):
        pert.data = torch.normal(0, 1e-7, size=pert.shape, device=device)

        for _ in range(num_iterations):
            optimizer.zero_grad()
            pred, _ = model((attacked_data + pert).transpose(2, 1))
            adv_loss = adv_loss_function(pred, original_target)
            pert_norm = torch.norm(pert.view(attacked_data.size(0), -1), dim=1)
            loss = adv_loss + torch.mean(weight * pert_norm)
            loss.backward()
            optimizer.step()

            pred_val, dist_val = torch.argmax(pred, dim=1), pert_norm.detach()
            for e in range(attacked_data.size(0)):
                if (
                    dist_val[e] < o_bestdist[e]
                    and pred_val[e] != original_target[e]
                    and dist_val[e] <= eps
                ):
                    o_bestdist[e], o_bestattack[e] = (
                        dist_val[e],
                        (attacked_data + pert)[e].detach(),
                    )
        weight = (lower_bound + upper_bound) / 2

    return o_bestattack


# Main Attack Function
def attack(
    cap_model,
    model,
    test_loader,
    initial_weight,
    upper_bound_weight,
    binary_search_step,
    num_iterations,
    lr_attack,
    batch_size,
    num_class=10,
):
    cap_model.eval()
    model.eval()
    correct_cnt, total_cnt = 0, 0
    class_success = {i: {"total": 0, "success": 0} for i in range(num_class)}

    for points, target in tqdm(test_loader):
        points, target = points.to(device), target.to(device)

        # Visualize the original point cloud before attack
        visualize_point_cloud(
            points.transpose(2, 1)[0].permute(1, 0).cpu().detach().numpy(),
            title=f"{class_names[target[0].item()]}",
        )

        # Generate adversarial points
        adv_points = attack_one_batch(
            model,
            points,
            target,
            initial_weight,
            upper_bound_weight,
            binary_search_step,
            num_iterations,
            lr_attack,
        )
        adv_points = adv_points.transpose(2, 1)

        # Prediction on adversarial points
        pred, _ = cap_model(adv_points)
        pred_choice = pred.data.max(1)[1]

        # Visualize the adversarial point cloud after attack
        visualize_point_cloud(
            adv_points[0].permute(1, 0).cpu().detach().numpy(),
            title=f"After Attack: {class_names[pred_choice[0].item()]}",
        )

        # Calculate success metrics per class
        for cat in np.unique(target.cpu()):
            class_success[cat.item()]["total"] += len(points[target == cat])
            class_success[cat.item()]["success"] += (
                (pred_choice[target == cat] == target[target == cat].long().data)
                .cpu()
                .sum()
                .item()
            )

        correct_cnt += pred_choice.eq(target.long().data).cpu().sum().item()
        total_cnt += len(points)

    accuracy = correct_cnt / total_cnt * 100
    return accuracy, class_success


# Load Pretrained Model
def load_model(model_name, num_class, device, model_path):
    model_module = importlib.import_module(model_name)
    model = model_module.get_model(num_class, normal_channel=False)
    checkpoint = torch.load(model_path, map_location=device)
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
        i: name
        for i, name in enumerate(
            [
                "Bathtub",
                "Bed",
                "Chair",
                "Desk",
                "Dresser",
                "Monitor",
                "Nightstand",
                "Sofa",
                "Table",
                "Toilet",
            ]
        )
    }

    # Load the dataset
    test_dataset = ModelNetDataLoader(root="data", split="test", process_data=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=36, shuffle=True, num_workers=10
    )

    # Load the models
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

    # Run the attack
    accuracy, class_success = attack(
        cap_model,
        model,
        test_loader,
        initial_weight=1,
        upper_bound_weight=8,
        binary_search_step=5,
        num_iterations=10,
        lr_attack=0.01,
        batch_size=36,
    )

    # Print results
    print(
        pd.DataFrame(
            [
                {
                    "Class": class_names[cat],
                    "Success": class_success[cat]["success"],
                    "Total": class_success[cat]["total"],
                    "Acc (%)": f"{(class_success[cat]['success'] / class_success[cat]['total'] * 100) if class_success[cat]['total'] > 0 else 0.0:.2f}",
                }
                for cat in class_success
            ]
        ).to_string(index=False)
    )