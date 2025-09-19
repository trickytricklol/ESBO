import torch
import random
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from data_utils.modelnet_dataloader import ModelNetDataLoader
import importlib
import matplotlib.pyplot as plt


# Visualization Function
def visualize_point_cloud(point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c="r", marker="o"
    )
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.show()


# Attack Function for a Single Batch
def attack_one_batch(
    model,
    attacked_data,
    pert,
    target,
    initial_weight,
    upper_bound_weight,
    binary_search_step,
    num_iterations,
    lr_attack,
    eps=0.5,
):
    device = pert.device
    pert = torch.zeros_like(attacked_data, requires_grad=True, device=device)
    optimizer = optim.Adam([pert], lr=lr_attack)

    lower_bound = torch.zeros(attacked_data.size(0), device=device)
    weight = torch.ones(attacked_data.size(0), device=device) * initial_weight
    upper_bound = torch.ones(attacked_data.size(0), device=device) * upper_bound_weight

    o_bestdist = torch.ones(attacked_data.size(0), device=device) * 1e10
    o_bestscore = torch.ones(attacked_data.size(0), device=device) * -1
    o_bestattack = torch.ones_like(attacked_data, device=device)

    for _ in range(binary_search_step):
        pert.data = torch.normal(0, 1e-7, size=pert.shape, device=device)

        for iteration in range(num_iterations):
            optimizer.zero_grad()
            pointclouds_input = attacked_data + pert
            pred, _ = model(pointclouds_input.transpose(2, 1))

            adv_loss = nn.functional.nll_loss(pred, target.long())
            pert_norm = torch.norm(pert.view(attacked_data.size(0), -1), dim=1)
            loss = adv_loss + torch.mean(weight * pert_norm)
            loss.backward()
            optimizer.step()

            pred_val, dist_val = torch.argmax(pred, dim=1), pert_norm.detach()
            for e in range(attacked_data.size(0)):
                if (
                    dist_val[e] < o_bestdist[e]
                    and pred_val[e] != target[e]
                    and dist_val[e] <= eps
                ):
                    o_bestdist[e] = dist_val[e]
                    o_bestscore[e] = pred_val[e]
                    o_bestattack[e] = pointclouds_input[e].detach()

        # Adjust weight bounds for binary search
        for e in range(attacked_data.size(0)):
            if o_bestscore[e] != target[e] and o_bestdist[e] <= o_bestdist[e]:
                lower_bound[e] = max(lower_bound[e], weight[e])
            else:
                upper_bound[e] = min(upper_bound[e], weight[e])
            weight[e] = (lower_bound[e] + upper_bound[e]) / 2

    # Calculate success
    successful_pred = (o_bestscore != target).sum().item()
    return successful_pred, attacked_data.size(0), o_bestdist


# Main Attack Function
def attack(
    model,
    test_loader,
    initial_weight,
    upper_bound_weight,
    binary_search_step,
    num_iterations,
    lr_attack,
    batch_size,
    class_names,
):
    model.eval()
    pert = torch.zeros([batch_size, 1024, 3], device=device, requires_grad=True)
    optimizer = optim.Adam([pert], lr=lr_attack)

    total_successful_pred, total_count = 0, 0
    class_success, class_total = {}, {}

    for j, (points, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
        points, target = points.to(device), target.to(device)
        print(f"Original target: {target}")

        successful_pred, total_cnt, _ = attack_one_batch(
            model,
            points,
            pert,
            target,
            initial_weight,
            upper_bound_weight,
            binary_search_step,
            num_iterations,
            lr_attack,
        )
        total_successful_pred += successful_pred
        total_count += total_cnt

        print(f"Batch {j}, Accuracy: {total_successful_pred / total_count:.2%}")

        # Track per-class success
        for t in target:
            class_success[t.item()] = class_success.get(t.item(), 0) + successful_pred
            class_total[t.item()] = class_total.get(t.item(), 0) + total_cnt

        # Print per-class success rates
        data = [
            {
                "Class Name": class_names[class_id],
                "Successful Pred": class_success.get(class_id, 0),
                "Total Pred": class_total.get(class_id, 0),
                "Accuracy (%)": f"{(class_success.get(class_id, 0) / class_total.get(class_id, 1e-10)) * 100:.2f}%",
            }
            for class_id in class_names
        ]
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
    print(f"Overall Accuracy: {total_successful_pred / total_count:.2%}")


# Model Loader
def load_model(model_name, num_class, device):
    model_module = importlib.import_module(model_name)
    model = model_module.get_model(num_class, normal_channel=False)
    checkpoint = torch.load(
        "log/classification/2024-08-27_14-30/checkpoints/best_model.pth"
    )
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

    # Load the dataset
    test_dataset = ModelNetDataLoader(root="data", split="test", process_data=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=36, shuffle=True, num_workers=10
    )

    # Load the model
    model = load_model("pointnet2", 10, device)

    # Run the attack
    attack(
        model,
        test_loader,
        initial_weight=10,
        upper_bound_weight=80,
        binary_search_step=5,
        num_iterations=10,
        lr_attack=0.01,
        batch_size=36,
        class_names=class_names,
    )