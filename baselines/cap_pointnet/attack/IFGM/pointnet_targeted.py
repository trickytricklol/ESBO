import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os,sys
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

# 添加 data_utils 模块所在的目录到 sys.path
model_path = os.path.join(project_root,'model')
sys.path.append(project_root)
sys.path.append(model_path)
from data_utils.modelnet_dataloader import ModelNetDataLoader
import importlib


# Iterative Fast Gradient Method (IFGM) attack
def ifgm(
    x_pl,
    model,
    alpha,
    t_pl,
    iter=10,
    eps=0.5,
    clip_min=None,
    clip_max=None,
    targeted=True,
):
    x_adv = x_pl.clone().detach()
    ord_fn = lambda x: x / torch.sqrt(
        torch.sum(x**2, dim=list(range(1, x.ndim)), keepdim=True)
    )

    for _ in range(iter):
        x_adv.requires_grad_()
        pred, _ = model(x_adv)
        loss = (
            F.nll_loss(pred, t_pl.long()) if targeted else F.nll_loss(pred, t_pl.long())
        )

        # Calculate perturbation and apply it
        perturb = alpha * ord_fn(torch.autograd.grad(loss, x_adv)[0])
        x_adv = x_adv - perturb if targeted else x_adv + perturb

        # Apply clipping on perturbation
        perturb_diff = x_adv - x_pl
        clip = torch.norm(perturb_diff, p=2, dim=1, keepdim=True) > eps
        x_adv = torch.where(
            clip,
            x_pl
            + perturb_diff * eps / torch.norm(perturb_diff, p=2, dim=1, keepdim=True),
            x_adv,
        )

        if clip_min is not None and clip_max is not None:
            x_adv = torch.clamp(x_adv, clip_min, clip_max)

        x_adv = x_adv.detach()

    return x_adv


# Visualize point cloud data
def visualize_point_cloud(point_cloud, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c="r", marker="o"
    )
    ax.set_title(title)
    plt.show()


# Define test function for evaluating model robustness to adversarial attacks
def test(model, loader, num_class=10):
    correct_cnt, total_cnt = 0, 0
    class_attack_success = {i: {"total": 0, "success": 0} for i in range(num_class)}

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points, target = points.to(device).transpose(2, 1), target.to(device)
        random_label = torch.randint(0, num_class, (1,), device=device).item()
        selected_indices = torch.nonzero(target != random_label).squeeze()

        if len(selected_indices) == 0:
            continue
        points, target = points[selected_indices], torch.full(
            (len(selected_indices),), random_label, device=device
        )

        # Generate adversarial points and evaluate
        adv_points = ifgm(points, model, alpha=0.02, t_pl=target, iter=80, eps=0.03)
        pred, _ = model(adv_points)
        pred_choice = pred.data.max(1)[1]

        correct = pred_choice.eq(target).sum().item()
        correct_cnt += correct
        total_cnt += len(points)

        for cat in torch.unique(target):
            class_attack_success[cat.item()]["total"] += (target == cat).sum().item()
            class_attack_success[cat.item()]["success"] += (
                (pred_choice[target == cat] == target[target == cat]).sum().item()
            )

        print(f"Batch {j}: Attack Success Rate = {100 * correct_cnt / total_cnt:.2f}%")

    asr = correct_cnt / total_cnt
    for cat in class_attack_success:
        total = class_attack_success[cat]["total"]
        success = class_attack_success[cat]["success"]
        class_attack_success[cat]["asr"] = (success / total) * 100 if total > 0 else 0.0

    return asr, class_attack_success


# Main function to set up model and data
if __name__ == "__main__":
    class_names = {
        i: name
        for i, name in enumerate(
    [
        "airplane",
        "bathtub",
        "bed",
        "bench",
        "bookshelf",
        "bottle",
        "bowl",
        "car",
        "chair",
        "cone",
        "cup",
        "curtain",
        "desk",
        "door",
        "dresser",
        "flower_pot",
        "glass_box",
        "guitar",
        "keyboard",
        "lamp",
        "laptop",
        "mantel",
        "monitor",
        "night_stand",
        "person",
        "piano",
        "plant",
        "radio",
        "range_hood",
        "sink",
        "sofa",
        "stairs",
        "stool",
        "table",
        "tent",
        "toilet",
        "tv_stand",
        "vase",
        "wardrobe",
        "xbox",
    ]
)
    }
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load data and model
    test_loader = torch.utils.data.DataLoader(
        ModelNetDataLoader(root="data", split="test"),
        batch_size=24,
        shuffle=True,
        num_workers=10,
    )
    model = importlib.import_module("pointnet2").get_model(40, normal_channel=False)
    checkpoint = torch.load(
        "log/classification/2025-05-23_15-59/checkpoints/best_model.pth",
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    asr, class_attack_success = test(model, test_loader, num_class=40)
    print(f"Overall Attack Success Rate: {asr * 100:.2f}%")

    # Display results in DataFrame format
    attack_success_rate_df = pd.DataFrame(
        [
            {
                "Class": class_names[cat],
                "Total": data["total"],
                "Success": data["success"],
                "ASR (%)": f"{data['asr']:.2f}",
            }
            for cat, data in class_attack_success.items()
        ]
    )
    print(attack_success_rate_df.to_string(index=False))