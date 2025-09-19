import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pointnet2_utils import PointNetSetAbstraction


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, input_dim, num_classes, temperature):
        super(MultiHeadAttentionModule, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.query = nn.Parameter(torch.randn(num_classes, input_dim))
        self.conv = nn.Conv1d(num_classes, 1, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size, N_prime, D = x.shape  # [B, N, D]

        attention_scores = torch.zeros(batch_size, N_prime, self.num_classes).to(
            x.device
        )
        for c in range(self.num_classes):
            attention_scores[:, :, c] = (x @ self.query[c]) / self.temperature

        attention_scores = F.softmax(attention_scores, dim=-1)

        global_features = torch.zeros(batch_size, self.num_classes, D).to(x.device)
        for c in range(self.num_classes):
            global_features[:, c, :] = torch.sum(
                attention_scores[:, :, c].unsqueeze(-1) * x, dim=1
            )

        global_representation = self.conv(global_features)  # [B, 1, D]
        global_representation = global_representation.squeeze(-1)  # [B, D]

        return global_representation


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=in_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)
        self.attention = MultiHeadAttentionModule(1024, num_class, 1.25)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l1_points = torch.max(l1_points, 2)[0]

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = torch.max(l2_points, 2)[0]

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = l3_points.squeeze(-1).permute(0, 2, 1)  # [B, N, 1024]
        l3_points = self.attention(l3_points)

        l3_points = l3_points.view(B, 1024)
        x = l3_points

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss