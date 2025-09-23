import torch
import numpy as np


class SinPoint:

    def __init__(self, args):
        self.rand_center_num = args.rand_center_num
        self.w = args.w
        self.A = args.A
        self.sample = args.sample
        self.isCat = args.isCat
        self.shuffle = args.shuffle

    def Local(self, data):
        """
        Args:
            data (B,N,3)
        """
        device = data.device
        B, N, C = data.shape
        if self.sample == "RPS":
            # RPS  B * k
            idxs = self.generate_random_permutations_batch(B,N,self.rand_center_num)
        if self.sample == "FPS":
            # FPS  B * k
            idxs = self.farthest_point_sample(data, self.rand_center_num)
        dist = torch.zeros_like(data).to(device)
        for i in range(self.rand_center_num):
            center = self.index_points(data, idxs[:,i]).unsqueeze(1)
            dist = dist + data - center
        dist = dist / self.rand_center_num
        w = -self.w + (self.w + self.w) * torch.rand([1, 1, C])
        A = -self.A + (self.A + self.A) * torch.rand([1, 1, C])
        move = A.to(device) * torch.sin(w.to(device) * dist)
        newdata = data + move
        return newdata

    def Global(self, data):
        """
        Args:
            data (B,N,3)
        """
        device = data.device
        B, N, C = data.shape
        newdata = torch.zeros_like(data)
        w = -self.w + (self.w + self.w) * torch.rand([1, 1, C])
        A = -self.A + (self.A + self.A) * torch.rand([1, 1, C])
        move = A.to(device) * torch.sin(w.to(device) * data)
        newdata = data + move
        return newdata

    def Sin(self, data, label=[]):
        """
        Args:
            data (B,N,3)
            label (B)
        """
        B, _, _ = data.shape
        newdata, shift, scale = self.normalize_point_clouds(data)
        if self.rand_center_num == 0:
            newdata = self.Global(newdata)
        else:
            newdata = self.Local(newdata)
        newdata = newdata * scale + shift
        label = label.unsqueeze(1)
        if self.isCat:
            newdata = torch.cat([data, newdata],dim=0)
            label = torch.cat([label, label],dim=0)
            if self.shuffle:
                idxs = torch.randperm(B*2)
                newdata = newdata[idxs,:,:]
                label = label[idxs,:]
        return newdata, label.squeeze(1)

    def index_points(self, points, idx):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def farthest_point_sample(self, xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
            dist = self.square_distance(xyz, centroid).squeeze(2)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, -1)[1]
        return centroids

    def generate_random_permutations_batch(self, B, N, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        # 生成B个长度为N的随机排列的批次
        all_permutations = torch.stack([torch.randperm(N) for _ in range(B)])
        # 选择每个随机排列的前top_k个位置
        centroids = all_permutations[:, :npoint]
        return centroids

    def square_distance(self,src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def normalize_point_clouds(self, pcs):
        B, N, C = pcs.shape
        shift = torch.mean(pcs, dim=1).unsqueeze(1)
        scale = torch.std(pcs.view(B, N * C), dim=1).unsqueeze(1).unsqueeze(1)
        newpcs = (pcs - shift) / scale
        return newpcs, shift, scale