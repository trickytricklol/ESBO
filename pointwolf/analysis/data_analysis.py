import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tqdm
import os
from torch.utils.data import DataLoader
import argparse
from data import ModelNet40
from model import DGCNN
import matplotlib.colors as mcolors
import pyvista as pv
pv.start_xvfb()  # 启动虚拟帧缓冲
pv.OFF_SCREEN = True  # 禁用交互窗口
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import open3d as o3d
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
import math
import random
from PIL import Image, ImageDraw, ImageFilter,ImageFont

class PointWolfAnalyzer:
    def __init__(self, args):
        self.args = args
        self.output_dir = os.path.join(
            "pointwolf_analysis_results",
            f"{'with_pointwolf' if args.PointWOLF else 'without_pointwolf'}_{args.dataset}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.train_set = ModelNet40(args, partition="train")
        self.train_loader = DataLoader(
            self.train_set, num_workers=8,
            batch_size=args.batch_size, shuffle=True, drop_last=True
        )

        self.stats = {
            "train": {
                "entropy_origin": [],
                "entropy_corrupted": [],
                "entropy_corrupted2": [],
                "label_rank_origin": [],
                "label_rank_corrupted": [],
                "labels": []
            },
            "test": {
                "entropy_origin": [],
                "entropy_corrupted": [],
                "entropy_corrupted2": [],
                "label_rank_origin": [],
                "label_rank_corrupted": [],
                "labels": []
            }
        }
        self.num_classes = args.classes


    def compute_sample_entropy(self, logits=None, batch_size=None):
        if logits is not None:
            probs = F.softmax(logits, dim=-1)
        else:
            assert batch_size is not None, "无模型时必须指定batch_size"
            logits = torch.randn(batch_size, self.num_classes)
            probs = F.softmax(logits, dim=-1)

        # entropy = -torch.sum(probs * torch.log2(probs + 1e-8), dim=-1)
        C = probs.size(-1)  # 如果probs的最后一维是类别维度，可这样获取
        entropy = -torch.sum(probs * torch.log2(probs + 1e-8), dim=-1) / torch.log2(torch.tensor(C, dtype=torch.float32))
        return entropy.cpu().numpy()


    def compute_correct_label_rank(self, logits, true_labels):
        """修复维度不匹配问题：确保logits和true_labels的批次大小一致"""
        # 安全检查：确保批次大小一致
        assert logits.size(0) == true_labels.size(0), \
            f"logits批次大小 {logits.size(0)} 与标签批次大小 {true_labels.size(0)} 不匹配"
        
        probs = F.softmax(logits, dim=1)  # [batch_size, num_classes]
        batch_size = probs.size(0)
        
        # 修正索引方式：使用torch.gather获取正确标签的概率
        # 避免因维度不匹配导致的索引错误
        correct_probs = torch.gather(probs, 1, true_labels.unsqueeze(1)).squeeze(1)  # [batch_size]
        
        # 计算排名（概率 >= 正确标签概率的类别数量）
        ranks = (probs >= correct_probs.unsqueeze(1)).sum(dim=1)  # [batch_size]
        return ranks.cpu().numpy()

    def visualize_pointcloud_pair(self, origin_pc, corrupted_pc, label, split, batch_idx, sample_idx, output_dir="output",w=-1):
        """点云对比可视化（大球体+类别名称）"""
        from PIL import Image, ImageDraw, ImageFilter
        import matplotlib.pyplot as plt
        from matplotlib import patches
        from mpl_toolkits.mplot3d import art3d

        # 获取当前样本的标签
        label_val = int(label[sample_idx].cpu().numpy())
        
        # ModelNet40 类别名称映射
        modelnet40_classes = [
            'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 
            'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 
            'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 
            'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 
            'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 
            'wardrobe', 'xbox'
        ]
        
        # 提取数据并转为numpy
        origin = origin_pc[sample_idx].cpu().numpy().T  # [N, 3]
        corrupted = corrupted_pc[sample_idx].cpu().numpy().T
        label_val = int(label[sample_idx].cpu().numpy())

        # -------------------------- 新增：点云向右旋转90度，修正倒置 --------------------------
        def rotate_90_right(points):
            # 坐标映射：新X=原Z，新Y=原Y，新Z=-原X（绕Y轴右旋90度，摆正朝上方向）
            x_old, y_old, z_old = points[:,0], points[:,1], points[:,2]
            x_new = x_old
            y_new = -z_old
            z_new = y_old
            return np.stack([x_new, y_new, z_new], axis=1)

        # 对原始点云和损坏点云分别旋转
        origin = rotate_90_right(origin)
        corrupted = rotate_90_right(corrupted)
        
        # 获取类别名称
        if 0 <= label_val < len(modelnet40_classes):
            label_name = modelnet40_classes[label_val]
        else:
            label_name = f"Unknown_{label_val}"

        # 随机下采样（球体较大，减少点数）
        if origin.shape[0] > 1024:
            idx = np.random.choice(origin.shape[0], 1024, replace=False)
            origin, corrupted = origin[idx], corrupted[idx]

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        def render_3d_spheres(points, color, title=""):
            """使用大球体渲染点云"""
            fig = plt.figure(figsize=(10, 10), facecolor='white', dpi=200)  # 增大尺寸
            ax = fig.add_subplot(111, projection='3d')
            
            # 设置颜色
            rgb_color = np.array(color)
            
            # 计算合适的球体半径
            max_range = np.array([points[:,0].ptp(), points[:,1].ptp(), points[:,2].ptp()]).max()
            sphere_radius = max_range * 0.02  # 增大球体
            
            # 渲染每个点作为3D球体
            for i in range(len(points)):
                x, y, z = points[i]
                
                # 创建球体
                u = np.linspace(0, 2 * np.pi, 16)  # 增加面数
                v = np.linspace(0, np.pi, 16)
                
                x_sphere = x + sphere_radius * np.outer(np.cos(u), np.sin(v))
                y_sphere = y + sphere_radius * np.outer(np.sin(u), np.sin(v))
                z_sphere = z + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
                
                # 绘制球体
                ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                            color=rgb_color, alpha=0.9, shade=True, 
                            edgecolor='none', antialiased=True, linewidth=0)
            
            # 设置视角
            ax.view_init(elev=25, azim=45)
            
            # 关键修改：彻底移除边距和坐标轴
            ax.set_axis_off()
            
            # 设置标题
            ax.set_title(title, fontsize=20, pad=10, fontweight='bold', color='black')
            
            # 关键修改：精确设置坐标范围，让点云填满画面
            margin = max_range * 0.1  # 减少边距
            
            mid_x = (points[:,0].max() + points[:,0].min()) * 0.5
            mid_y = (points[:,1].max() + points[:,1].min()) * 0.5
            mid_z = (points[:,2].max() + points[:,2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range/2 - margin, mid_x + max_range/2 + margin)
            ax.set_ylim(mid_y - max_range/2 - margin, mid_y + max_range/2 + margin)
            ax.set_zlim(mid_z - max_range/2 - margin, mid_z + max_range/2 + margin)
            
            # 设置背景
            ax.set_facecolor('white')
            ax.grid(False)
            
            # 关键修改：移除所有边距
            plt.tight_layout(pad=0)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            
            # 保存到缓冲区
            from io import BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0,
                    facecolor='white', edgecolor='none', transparent=False)
            buf.seek(0)
            img = Image.open(buf)
            plt.close()
            
            return img

        def enhance_3d_effect(img):
            """增强3D效果"""
            # 转换为RGBA
            img_rgba = img.convert('RGBA')
            width, height = img_rgba.size
            
            # 创建边框效果
            border_size = 2
            bordered = Image.new('RGBA', (width + border_size*2, height + border_size*2), (0, 0, 0, 0))
            bordered.paste(img_rgba, (border_size, border_size))
            
            # 添加轻微阴影
            shadow = bordered.filter(ImageFilter.GaussianBlur(radius=3))
            shadow_array = np.array(shadow)
            shadow_array[..., 3] = shadow_array[..., 3] * 0.3  # 降低阴影透明度
            shadow = Image.fromarray(shadow_array)
            
            # 合并图像
            result = Image.new('RGBA', (width + border_size*2 + 5, height + border_size*2 + 5), (255, 255, 255, 255))
            result.paste(shadow, (3, 3), shadow)
            result.paste(bordered, (0, 0), bordered)
            
            return result.convert('RGB')

        # === 渲染两幅点云 ===
        try:
            print(f"渲染3D球体点云 - 类别: {label_name}, 点数: {len(origin)}")
            
            img_origin = render_3d_spheres(
                origin, 
                color=(0.6, 0.9, 0.6),  # 鲜艳的绿色
                title=f"Original: {label_name}"
            )
            
            img_corrupted = render_3d_spheres(
                corrupted,
                color=(1.0, 0.6, 0.6),  # 鲜艳的红色
                title=f"Corrupted: {label_name}"
            )
            
            # 增强3D效果
            img_origin = enhance_3d_effect(img_origin)
            img_corrupted = enhance_3d_effect(img_corrupted)
            
        except Exception as e:
            print(f"3D渲染失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return

        # === 合并图像并保存 ===
        # 调整图像大小确保一致
        img_origin = img_origin.resize((1000, 1000), Image.LANCZOS)
        img_corrupted = img_corrupted.resize((1000, 1000), Image.LANCZOS)
        
        merged = Image.new('RGB', (2000, 1000), (255, 255, 255))
        merged.paste(img_origin, (0, 0))
        merged.paste(img_corrupted, (1000, 0))
        
        # 添加总体标题
        draw = ImageDraw.Draw(merged)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        overall_title = f"Point Cloud Comparison - Batch {batch_idx}, Sample {sample_idx}"
        text_width = draw.textlength(overall_title, font=font)
        draw.text(((1300 - text_width) // 2, 10), overall_title, fill=(0, 0, 0), font=font)
        
        if w == -1:
            save_path = os.path.join(
                output_dir,
                f'{split}_batch{batch_idx}_sample{sample_idx}_{label_name}.png'
            )
        else:
            save_path = os.path.join(
                output_dir,
                f'{split}_batch{batch_idx}_sample{sample_idx}_w{float(w)}_{label_name}.png'
            )
        merged.save(save_path, format='PNG', dpi=(600, 600), quality=100, 
           optimize=True, compress_level=0)
        print(f"[Visualize] 3D球体渲染完成: {save_path}")

    def interpolate_pointclouds(self, origin_pc, corrupted_pc, entropy_origin, entropy_corrupted, k=1.0, a=0.5):
        """
        根据信息熵对原始点云和调整后点云进行线性插值
        
        参数:
            origin_pc: 原始点云，形状为[B, 3, N]
            corrupted_pc: 调整后点云，形状为[B, 3, N]
            entropy_origin: 原始点云的信息熵，形状为[B]
            entropy_corrupted: 调整后点云的信息熵，形状为[B]
            k: 插值公式中的斜率参数
            a: 插值公式中的偏移参数
        
        返回:
            interpolated_pc: 插值后的点云，形状为[B, 3, N]
            weights: 计算得到的权重，形状为[B, 1, 1]（便于广播）
        """
        # 确保输入点云形状一致
        assert origin_pc.shape == corrupted_pc.shape, "原始点云和调整后点云形状必须一致"
        assert origin_pc.dim() == 3 and origin_pc.size(1) == 3, "点云必须是[B, 3, N]格式"
        
        # 计算每个样本的权重 s = entropy_corrupted - entropy_origin
        s = entropy_corrupted - entropy_origin  # 形状为[B]
        
        # 应用sigmoid-like权重公式 w = 1 / (1 + exp(-k*(s - a)))
        # 增加两个维度以支持广播 [B] -> [B, 1, 1]
        weights = 1.0 / (1.0 + torch.exp(-k * (s - a))).view(-1, 1, 1)
        
        # 线性插值 P'' = w*p + (1-w)*p'
        interpolated_pc = weights * origin_pc + (1 - weights) * corrupted_pc
        
        return interpolated_pc, weights


    def process_dataset(self, dataloader, split,idx):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DGCNN(self.args, self.args.classes).to(device)
        model.load_state_dict(torch.load(self.args.model_path, map_location=device))
        model.eval()
        print(f'\n=== Processing {split} Set (PointWOLF: {self.args.PointWOLF}) ===')

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm.tqdm(dataloader, desc=f'{split} Batch')):
                if self.args.PointWOLF:
                    origin_pc, corrupted_pc, label = batch_data
                    label = label.squeeze()
                    origin_pc = origin_pc.to(device)
                    corrupted_pc = corrupted_pc.to(device)
                else:
                    origin_pc = batch_data[0].to(device)
                    corrupted_pc = origin_pc.clone()
                    label = batch_data[1].squeeze()
                
                # 关键修复：确保点云维度正确 [B, 3, N]
                if origin_pc.dim() == 3 and origin_pc.size(1) != 3:
                    origin_pc = origin_pc.permute(0, 2, 1)
                if corrupted_pc.dim() == 3 and corrupted_pc.size(1) != 3:
                    corrupted_pc = corrupted_pc.permute(0, 2, 1)

                # # 可视化前几个批次
                # if batch_idx < 20:  # 减少可视化数量
                #     for sample_idx in [0]:  # 每个批次只可视化一个样本
                #         self.visualize_pointcloud_pair(
                #             origin_pc=origin_pc,
                #             corrupted_pc=corrupted_pc,
                #             label=label,
                #             split=split,
                #             batch_idx=batch_idx,
                #             sample_idx=sample_idx
                #         )

                # 模型推理需要 [B, 3, N] 格式
                logits_origin = model(origin_pc)
                logits_corrupted = model(corrupted_pc)

                # 确保标签与logits批次大小一致
                batch_size = origin_pc.shape[0]
                assert label.size(0) == batch_size, \
                    f"标签批次大小 {label.size(0)} 与点云批次大小 {batch_size} 不匹配"

                true_labels_tensor = label.to(device)

                # 计算熵值和排名
                entropy_origin = self.compute_sample_entropy(logits=logits_origin)
                entropy_corrupted = self.compute_sample_entropy(logits=logits_corrupted)
                rank_origin = self.compute_correct_label_rank(logits_origin, true_labels_tensor)
                rank_corrupted = self.compute_correct_label_rank(logits_corrupted, true_labels_tensor)

                interpolated_pc, weights = self.interpolate_pointclouds(
                    origin_pc, corrupted_pc,
                    torch.tensor(entropy_origin).to(device),
                    torch.tensor(entropy_corrupted).to(device),
                    k=10.0, a=0.5
                )
                
                if batch_idx < 500:  # 减少可视化数量
                    for sample_idx in [i for i in range(0, batch_size)]:  # 可视化所有样本
                        # 获取当前样本的标签
                        current_label = int(label[sample_idx].cpu().numpy())
                        
                        # 只可视化person类别（索引24）
                        if current_label == idx:
                            self.visualize_pointcloud_pair(
                                origin_pc=origin_pc,
                                corrupted_pc=corrupted_pc,
                                label=label,
                                split=split,
                                batch_idx=batch_idx,
                                sample_idx=sample_idx
                            )
                            self.visualize_pointcloud_pair(
                                origin_pc=origin_pc,
                                corrupted_pc=interpolated_pc,
                                label=label,
                                split=split,
                                batch_idx=batch_idx,
                                sample_idx=sample_idx,
                                w=weights[sample_idx]
                            )

                # 存储统计结果
                self.stats[split]["entropy_origin"].extend(entropy_origin)
                self.stats[split]["entropy_corrupted"].extend(entropy_corrupted)


                for i in range(len(entropy_corrupted)):
                    if entropy_corrupted[i] > 0.5:
                        # 值越高概率越大，0.5对应0%，1.0对应100%
                        prob = 2 * (entropy_corrupted[i] - 0.5)
                        if random.random() < prob:
                            self.stats[split]["entropy_corrupted2"].append(entropy_corrupted[i])

                    # if entropy_corrupted[i] > 0.5:
                    #     prob = 1 / (1 + math.exp(-10 * (entropy_corrupted[i] - 0.75)))  # 可调整参数
                    #     if random.random() < prob:
                    #         self.stats[split]["entropy_corrupted2"].append(entropy_corrupted[i])
                self.stats[split]["label_rank_origin"].extend(rank_origin)
                self.stats[split]["label_rank_corrupted"].extend(rank_corrupted)
                self.stats[split]["labels"].extend(label.cpu().numpy())

        # 转换为numpy数组
        for key in self.stats[split]:
            self.stats[split][key] = np.array(self.stats[split][key])

        self.print_stats_summary(split)


    def print_stats_summary(self, split):
        ent_origin_mean = self.stats[split]["entropy_origin"].mean()
        ent_origin_std = self.stats[split]["entropy_origin"].std()
        ent_corr_mean = self.stats[split]["entropy_corrupted"].mean()
        ent_corr_std = self.stats[split]["entropy_corrupted"].std()

        rank_origin_mean = self.stats[split]["label_rank_origin"].mean()
        rank_origin_std = self.stats[split]["label_rank_origin"].std()
        rank_corr_mean = self.stats[split]["label_rank_corrupted"].mean()
        rank_corr_std = self.stats[split]["label_rank_corrupted"].std()

        sample_count = len(self.stats[split]["labels"])
        unique_labels = np.unique(self.stats[split]["labels"]).shape[0]

        print(f'\n[{split} Set Summary]')
        print(f'  Total Samples: {sample_count} | Unique Labels: {unique_labels}')
        print(f'  -------------------------- Entropy (Uncertainty) --------------------------')
        print(f'  Origin Point Cloud: {ent_origin_mean:.3f} ± {ent_origin_std:.3f}')
        print(f'  Corrupted Point Cloud: {ent_corr_mean:.3f} ± {ent_corr_std:.3f}')
        print(f'  -------------------------- Correct Label Rank (1=Best) --------------------------')
        print(f'  Origin Point Cloud: {rank_origin_mean:.3f} ± {rank_origin_std:.3f}')
        print(f'  Corrupted Point Cloud: {rank_corr_mean:.3f} ± {rank_corr_std:.3f}')


    def plot_stats_distribution(self, split):
        plt.rcParams.update({"font.size": 12, "figure.facecolor": "white"})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        pw_mark = "With PointWOLF" if self.args.PointWOLF else "Without PointWOLF"
        origin_color = "green"
        corrupted_color = "red"
        corrupted_color2 = "yellow"

        # ========= 1. 熵值归一化分布 =========
        ax1.hist(
            self.stats[split]["entropy_origin"], bins=30, alpha=0.6, color=origin_color,
            label='Origin', density=True               # ← density=True
        )
        ax1.hist(
            self.stats[split]["entropy_corrupted"], bins=30, alpha=0.6, color=corrupted_color,
            label='Augmented', density=True            # ← density=True
        )
        ax1.hist(
            self.stats[split]["entropy_corrupted2"], bins=30, alpha=0.6, color=corrupted_color2,
            label='Corrupted', density=True            # ← density=True
        )

        ax1.set_xlabel('Sample Entropy')
        ax1.set_ylabel('Density')                      # ← 修改为 Density
        ax1.set_title(f'Entropy Distribution (Normalized)\n{pw_mark}', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # ========= 2. 标签排名归一化分布 =========
        max_rank = min(
            max(
                np.max(self.stats[split]["label_rank_origin"]),
                np.max(self.stats[split]["label_rank_corrupted"])
            ) + 2,
            self.num_classes
        )

        bins = np.arange(1, max_rank + 1)

        ax2.hist(
            self.stats[split]["label_rank_origin"], bins=bins,
            alpha=0.7, color=origin_color, label='Origin', density=True   # ← density=True
        )
        ax2.hist(
            self.stats[split]["label_rank_corrupted"], bins=bins,
            alpha=0.5, color=corrupted_color, label='Augmented', density=True  # ← density=True
        )

        ax2.set_xlabel('Correct Label Rank (1 = Best)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Correct Label Rank Distribution (Normalized)\n{pw_mark}', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)

        save_path = os.path.join(self.output_dir, f'train_stats_comparison.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'\n[Plot] Saved Normalized Stats Distribution: {save_path}')



    def run_full_analysis(self,idx):
        split = "test"
        self.process_dataset(self.train_loader, split,idx)
        self.plot_stats_distribution(split)
        print(f'\n=== All Analysis Done! Results Saved to: {self.output_dir} ===')


def main():
    parser = argparse.ArgumentParser(description='PointWOLF Corrupted Data Analysis')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], help='数据集类型')
    parser.add_argument('--batch_size', type=int, default=32, help='训练集批次大小')
    parser.add_argument('--num_points', type=int, default=1024, help='每样本点云数量')
    parser.add_argument('--emb_dims', type=int, default=1024, help='嵌入维度')
    parser.add_argument('--k', type=int, default=20, help='DGCNN KNN的k值')
    parser.add_argument('--dropout', type=float, default=0.5, help='模型dropout概率')
    parser.add_argument('--model_path', type=str, default='checkpoints/dgcnn/models/model.t7', help='预训练模型路径')
    parser.add_argument('--classes', type=int, default=40, help='分类类别数')
    parser.add_argument('--PointWOLF', action='store_true', help='是否使用PointWOLF')
    parser.add_argument('--AugTune', action='store_true', help='是否使用PointWOLF')
    parser.add_argument('--w_num_anchor', type=int, default=4, help='PointWOLF锚点数量')
    parser.add_argument('--w_sample_type', type=str, default='fps', choices=['fps', 'random'], help='PointWOLF锚点采样方式')
    parser.add_argument('--w_sigma', type=float, default=0.5, help='PointWOLF核带宽')
    parser.add_argument('--w_R_range', type=float, default=10, help='最大旋转范围')
    parser.add_argument('--w_S_range', type=float, default=3, help='最大缩放范围')
    parser.add_argument('--w_T_range', type=float, default=0.25, help='最大平移范围')

    args = parser.parse_args()
    label = 'night_stand'
    modelnet40_classes = [
            'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 
            'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 
            'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 
            'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 
            'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 
            'wardrobe', 'xbox'
        ]
    if label in modelnet40_classes:
        idx = modelnet40_classes.index(label)
    else:
        idx = -1
    
    print('=== Analysis Args ===')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')

    analyzer = PointWolfAnalyzer(args)
    analyzer.run_full_analysis(idx)


if __name__ == '__main__':
    main()