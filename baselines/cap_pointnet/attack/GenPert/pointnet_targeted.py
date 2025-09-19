import torch
import random
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import sys
import os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

# 添加 data_utils 模块所在的目录到 sys.path
model_path = os.path.join(project_root,'model')
sys.path.append(project_root)
sys.path.append(model_path)

# 导入自定义模块
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
    target_class,
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
        bestdist = torch.ones(attacked_data.size(0), device=device) * 1e10
        bestscore = torch.ones(attacked_data.size(0), device=device) * -1

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
                    and pred_val[e] == target_class
                    and dist_val[e] <= eps
                ):
                    o_bestdist[e] = dist_val[e]
                    o_bestscore[e] = pred_val[e]
                    o_bestattack[e] = pointclouds_input[e].detach()

        # Adjust weight bounds for binary search
        for e in range(attacked_data.size(0)):
            if bestscore[e] == target_class and bestdist[e] <= o_bestdist[e]:
                lower_bound[e] = max(lower_bound[e], weight[e])
            else:
                upper_bound[e] = min(upper_bound[e], weight[e])
            weight[e] = (lower_bound[e] + upper_bound[e]) / 2

    return (o_bestscore == target_class).sum().item(), attacked_data.size(0), o_bestdist


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

    total_successful_attacks, total_count = 0, 0
    class_success, class_total = {}, {}

    for j, (points, target) in tqdm(enumerate(test_loader), total=3):
        points, target = points.to(device), target.to(device)
        random_label = random.randint(0, 39)

        mask = target != random_label
        points = points[mask]
        target = torch.full(
            (len(points),), random_label, device=device, dtype=torch.int32
        )

        successful_attk, total_cnt, best_dist = attack_one_batch(
            model,
            points,
            pert,
            target,
            random_label,
            initial_weight,
            upper_bound_weight,
            binary_search_step,
            num_iterations,
            lr_attack,
        )
        total_successful_attacks += successful_attk
        total_count += total_cnt

        # Update per-class success and total counts
        if random_label not in class_success:
            class_success[random_label] = successful_attk
        else:
            class_success[random_label] += successful_attk
        if random_label not in class_total:
            class_total[random_label] = total_cnt
        else:
            class_total[random_label] += total_cnt

        print(
            f"Batch {j}, Attack success rate: {total_successful_attacks / total_count:.2%}"
        )

    # Generate and print attack success rate summary
    data = [
        {
            "Class Name": class_names[class_id],
            "Successful Attacks": class_success.get(class_id, 0),
            "Total Attacks": class_total.get(class_id, 0),
            "ASR (%)": f"{(class_success.get(class_id, 0) / class_total.get(class_id, 1e-10)) * 100:.2f}%",
        }
        for class_id in class_names
    ]
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print(f"Overall Attack success rate: {total_successful_attacks / total_count:.2%}")


# Model Loader
def load_model(model_name, num_class, device):
    model_module = importlib.import_module(model_name)
    model = model_module.get_model(num_class, normal_channel=False)
    checkpoint = torch.load(
        "log/classification/2025-05-23_15-59/checkpoints/best_model.pth",
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


if __name__ == "__main__":
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    class_names = {
        0: "airplane",
        1: "bathtub",
        2: "bed",
        3: "bench",
        4: "bookshelf",
        5: "bottle",
        6: "bowl",
        7: "car",
        8: "chair",
        9: "cone",
        10: "cup",
        11: "curtain",
        12: "desk",
        13: "door",
        14: "dresser",
        15: "flower_pot",
        16: "glass_box",
        17: "guitar",
        18: "keyboard",
        19: "lamp",
        20: "laptop",
        21: "mantel",
        22: "monitor",
        23: "night_stand",
        24: "person",
        25: "piano",
        26: "plant",
        27: "radio",
        28: "range_hood",
        29: "sink",
        30: "sofa",
        31: "stairs",
        32: "stool",
        33: "table",
        34: "tent",
        35: "toilet",
        36: "tv_stand",
        37: "vase",
        38: "wardrobe",
        39: "xbox",
    }

    # Load the dataset
    test_dataset = ModelNetDataLoader(root="data", split="test", process_data=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=36, shuffle=True, num_workers=10
    )

    # Load the model
    model = load_model("pointnet2", 40, device)

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