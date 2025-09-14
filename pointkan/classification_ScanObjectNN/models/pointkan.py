import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .kan import KAN


from pointnet2_ops import pointnet2_utils

# 获取激活函数
def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

# 计算欧氏距离的平方
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
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

# 根据索引在输入点数据中选择特定的点
def index_points(points, idx):
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

# FPS
def farthest_point_sample(xyz, npoint):
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
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

# 从给定的一组点（xyz）中，根据查询点（new_xyz）及其周围的半径，获取每个查询点周围的点的索引。
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

# KNN
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class Geometric_Affine_Module(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(Geometric_Affine_Module, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1,1,1,channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [batch, points, xyz]
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        # Group-Norm
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta # 仿射变换 
 
        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        # S-Pool
        e_new_points = torch.exp(new_points.permute(0,3,1,2))  # B 2C G K
        up = (new_points.permute(0,3,1,2) * e_new_points).mean(-1) # B 2C G
        down = e_new_points.mean(-1)
        center = torch.div(up, down) # B 2C G
        return new_xyz, new_points, center


class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0., no_kan=False):
        super(KANLayer,self).__init__()
        # out_features = in_features
        # hidden_features = in_features
        self.dim = in_features
        layers_hidden = [in_features, int(hidden_features/2), int(hidden_features/2), out_features]
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]
        if not no_kan:
            self.KAN = KAN(
                        layers_hidden = layers_hidden,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
        self.dwconv = nn.Conv1d(in_features, in_features, kernel_size=1, groups=in_features, bias=True)
    def forward(self, x):
        y = self.KAN(x)
        y = y.transpose(1, 2)  # (B, C, N)
        y = self.dwconv(y)
        y = y.transpose(1, 2)  # (B, N, C)
        return x + y

class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class Local_Feature_Processing(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: batch size, groups, k neighbors, channels
        output:[b,d,g]:batch size, out_channels, groups
        :param channels:
        :param blocks:
        """
        super(Local_Feature_Processing, self).__init__()
        self.use_xyz = use_xyz
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer1 = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        self.transfer2 = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation1 = []
        for _ in range(blocks):
            operation1.append(
                KANLayer(in_features = out_channels,hidden_features=out_channels, out_features=out_channels, act_layer=nn.GELU, drop=0., no_kan=False)
            )
        self.operation1 = nn.Sequential(*operation1)
        operation2 = []
        for _ in range(blocks):
            operation2.append(
                KANLayer(in_features = out_channels,hidden_features=out_channels, out_features=out_channels, act_layer=nn.GELU, drop=0., no_kan=False)
            )
        self.operation2 = nn.Sequential(*operation2)
        self.norm1 =  nn.LayerNorm(out_channels)
        self.norm2 =  nn.LayerNorm(out_channels)
        
    def forward(self, x, center):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) b:batch_size, n:groups, s:k_neighbors, d:channels
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s) 
        x = self.transfer1(x)
        batch_size, _, _ = x.size()
        x = x.permute(0, 2, 1)
        x = self.operation1(x)  # [b, k, d]
        x = self.norm1(x)
        x = x.permute(0, 2, 1)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) 
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        center = self.transfer2(center)
        center = center.permute(0,2,1)
        center = self.operation2(center)
        center = self.norm2(center)
        center = center.permute(0,2,1)
        x = x + center
        return x
    
    
class Global_Feature_Processing(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(Global_Feature_Processing, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)
    def forward(self, x):  # [b, d, g]
        x = self.operation(x)  # [b, d, g]
        return x


class Model(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], LFP_blocks=[2, 2, 2, 2], GFP_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        super(Model, self).__init__()
        self.stages = len(LFP_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)# 将3D坐标转换为embed_dim=64维特征
        assert len(LFP_blocks) == len(k_neighbors) == len(reducers) == len(GFP_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.GAM_list = nn.ModuleList()
        self.LFP_blocks_list = nn.ModuleList()
        self.GFP_blocks_list = nn.ModuleList()
        last_channel = embed_dim             
        anchor_points = self.points
        for i in range(len(LFP_blocks)):
            out_channel = last_channel * dim_expansion[i]
            LFP_block_num = LFP_blocks[i]
            GFP_block_num = GFP_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append GAM_list
            local_grouper = Geometric_Affine_Module(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.GAM_list.append(local_grouper)
            # append LFP_block_list
            LFP_block_module = Local_Feature_Processing(last_channel, out_channel, LFP_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.LFP_blocks_list.append(LFP_block_module)
            # append GFP_block_list
            GFP_block_module = Global_Feature_Processing(out_channel, GFP_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.GFP_blocks_list.append(GFP_block_module)

            last_channel = out_channel

        self.act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num)
        )
    
    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embedding(x)  # B,D,N
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x , center= self.GAM_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.LFP_blocks_list[i](x, center)  # [b,d,g]
            x = self.GFP_blocks_list[i](x)  # [b,d,g]
        
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x 
    
    


def pointKAN(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=32, groups=1, res_expansion=1.0,
                   activation="relu", bias=False, use_xyz = False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], LFP_blocks=[1, 1, 1, 1], GFP_blocks=[1, 1, 1, 1],
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


if __name__ == '__main__':
    
    print("===> testing pointKAN ...")
    model = pointKAN()
    device = torch.device('cuda:0')
    model.to(device)
    data = torch.rand(2, 3, 1024).to(device)
    out = model(data)
    print(out.shape)
