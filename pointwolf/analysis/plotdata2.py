import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适用于无GUI环境
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import argparse
import os
import tqdm

# 导入你的数据集和模型（需确保data.py和model.py路径正确）
from data import ModelNet40
from model import DGCNN


class EnhancedSampleVisualizer:
    def __init__(self, args):
        self.args = args
        # 结果保存目录（自动区分是否使用PointWOLF）
        self.save_dir = os.path.join(
            "enhanced_sample_distribution",
            f"{'with_pointwolf' if args.PointWOLF else 'without_pointwolf'}"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设备配置
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 初始化数据集和数据加载器
        self._init_dataset()
        # 初始化模型（加载预训练权重）
        self._init_model()

        # 存储隐空间特征和标签（区分原始/增强样本）
        self.features = {
            "origin": [],    # 原始样本隐空间特征
            "corrupted": [], # 增强（Corrupted）样本隐空间特征
            "labels": []     # 样本标签（用于着色和LDA降维）
        }

        # 定义要可视化的特定类别
        self.target_classes = ["night_stand", "plant", "dresser", "flower_pot"]
        # ModelNet40类别列表
        self.class_names = [
            'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 
            'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 
            'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 
            'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 
            'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 
            'wardrobe', 'xbox'
        ]


    def _init_dataset(self):
        """初始化数据集（支持ModelNet40，区分原始/增强样本）"""
        self.dataset = ModelNet40(self.args, partition="train")
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False
        )
        print(f"数据集加载完成：{len(self.dataset)}个样本，批次大小：{self.args.batch_size}")


    def _init_model(self):
        """初始化DGCNN模型并加载预训练权重，提取隐空间特征"""
        self.model = DGCNN(self.args).to(self.device)
        
        assert os.path.exists(self.args.model_path), f"模型路径不存在：{self.args.model_path}"
        checkpoint = torch.load(self.args.model_path, map_location=self.device)
        if "module." in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"模型加载完成：{self.args.model_path}")


    def extract_hidden_features(self):
        """提取原始样本和增强样本的隐空间特征"""
        print("\n开始提取隐空间特征...")
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm.tqdm(self.dataloader, desc="提取特征")):
                if self.args.PointWOLF:
                    origin_pc, corrupted_pc, labels = batch_data
                    origin_pc = self._adjust_pc_dim(origin_pc)
                    corrupted_pc = self._adjust_pc_dim(corrupted_pc)
                else:
                    origin_pc, labels = batch_data
                    corrupted_pc = origin_pc.clone()
                    origin_pc = self._adjust_pc_dim(origin_pc)
                    corrupted_pc = self._adjust_pc_dim(corrupted_pc)

                origin_pc = origin_pc.to(self.device)
                corrupted_pc = corrupted_pc.to(self.device)
                labels = labels.squeeze().cpu().numpy()

                logits_origin, embed_origin = self.model(origin_pc, return_embedding=True)
                logits_corr, embed_corr = self.model(corrupted_pc, return_embedding=True)

                self.features["origin"].append(embed_origin.cpu().numpy())
                self.features["corrupted"].append(embed_corr.cpu().numpy())
                self.features["labels"].append(labels)

        # 合并所有批次的特征和标签
        self.features["origin"] = np.concatenate(self.features["origin"], axis=0)
        self.features["corrupted"] = np.concatenate(self.features["corrupted"], axis=0)
        self.features["labels"] = np.concatenate(self.features["labels"], axis=0)

        print(f"特征提取完成：")
        print(f"  原始样本特征形状：{self.features['origin'].shape}")
        print(f"  增强样本特征形状：{self.features['corrupted'].shape}")
        print(f"  标签形状：{self.features['labels'].shape}")

        # 过滤出目标类别的数据
        self._filter_target_classes()


    def _filter_target_classes(self):
        """过滤出目标类别的数据"""
        target_indices = []
        for class_name in self.target_classes:
            if class_name in self.class_names:
                idx = self.class_names.index(class_name)
                target_indices.append(idx)
            else:
                print(f"警告：类别 '{class_name}' 不在类别列表中")
        
        if not target_indices:
            raise ValueError("没有找到目标类别，请检查类别名称")
        
        print(f"目标类别索引: {target_indices}")
        
        # 创建掩码，选择目标类别的样本
        mask = np.isin(self.features["labels"], target_indices)
        
        # 应用掩码过滤数据
        self.features["origin"] = self.features["origin"][mask]
        self.features["corrupted"] = self.features["corrupted"][mask]
        self.features["labels"] = self.features["labels"][mask]
        
        print(f"过滤后数据形状：")
        print(f"  原始样本特征形状：{self.features['origin'].shape}")
        print(f"  增强样本特征形状：{self.features['corrupted'].shape}")
        print(f"  标签形状：{self.features['labels'].shape}")
        
        # 打印各类别样本数量
        unique_labels, counts = np.unique(self.features["labels"], return_counts=True)
        for label, count in zip(unique_labels, counts):
            class_name = self.class_names[label]
            print(f"  类别 {class_name}: {count} 个样本")


    def _adjust_pc_dim(self, pc):
        """调整点云维度为 [B, 3, N]（DGCNN标准输入格式）"""
        if pc.dim() != 3:
            raise ValueError(f"点云维度错误：需为3维，当前为{pc.dim()}维")
        if pc.size(1) != 3 and pc.size(2) == 3:
            pc = pc.permute(0, 2, 1)
        return pc


    def reduce_dimension(self, features, method="pca", n_components=3):
        """
        降维：支持PCA（无监督）和LDA（有监督，需标签）
        """
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
            low_dim = reducer.fit_transform(features_scaled)
            print(f"PCA降维完成：解释方差比 = {reducer.explained_variance_ratio_}")
            return low_dim
        elif method == "lda":
            # 使用组合标签进行LDA（区分类别和样本类型）
            n_unique_labels = len(np.unique(self.combined_labels))
            max_components = min(n_components, n_unique_labels - 1, features_scaled.shape[1])
            print(f"LDA: {n_unique_labels}个组，最多可降到{max_components}维")
            
            reducer = LDA(n_components=max_components)
            low_dim = reducer.fit_transform(features_scaled, self.combined_labels)
            
            print(f"LDA降维完成：使用 {low_dim.shape[1]} 个组件")
            if hasattr(reducer, 'explained_variance_ratio_'):
                print(f"LDA解释方差比: {reducer.explained_variance_ratio_}")
            return low_dim
        else:
            raise ValueError(f"不支持的降维方法：{method}")


    def prepare_combined_features_and_labels(self):
        """准备合并的特征和对应的组合标签（区分原始/增强样本）"""
        # 合并原始和增强样本的特征
        combined_features = np.vstack([self.features["origin"], self.features["corrupted"]])
        
        # 创建组合标签
        n_samples = len(self.features["origin"])
        self.combined_labels = []
        
        # 获取目标类别的索引映射
        target_indices = [self.class_names.index(cls) for cls in self.target_classes]
        class_to_group = {}
        
        # 为每个类别创建原始和增强样本的组索引
        for i, class_idx in enumerate(target_indices):
            # 原始样本组索引
            class_to_group[class_idx] = i
            # 增强样本组索引
            class_to_group[class_idx + 100] = i + len(target_indices)  # 用+100来区分增强样本
        
        # 创建组名称
        self.group_names = []
        for class_idx in target_indices:
            class_name = self.class_names[class_idx].replace('_', ' ').title()
            self.group_names.append(f"{class_name} (Original)")
            self.group_names.append(f"{class_name} (Enhanced)")
        
        # 为每个样本分配标签
        # 原始样本
        for label in self.features["labels"]:
            self.combined_labels.append(class_to_group[label])
        
        # 增强样本
        for label in self.features["labels"]:
            self.combined_labels.append(class_to_group[label + 100])
        
        self.combined_labels = np.array(self.combined_labels)
        
        print(f"合并特征形状: {combined_features.shape}")
        print(f"组合标签形状: {self.combined_labels.shape}")
        print(f"组名称: {self.group_names}")
        print(f"标签分布: {np.unique(self.combined_labels, return_counts=True)}")
        
        return combined_features


    def plot_2d_distribution(self, low_dim_features, method="pca"):
        """2D分布可视化"""
        plt.rcParams.update({"font.size": 12, "figure.facecolor": "white"})
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # 定义颜色和标记 - 每个类别有原始和增强两种
        colors = ['blue', 'lightblue', 'red', 'pink', 'green', 'lightgreen', 'orange', 'wheat']
        markers = ['o', '^', 'o', '^', 'o', '^', 'o', '^']  # 圆形表示原始，三角表示增强
        sizes = [60, 50, 60, 50, 60, 50, 60, 50]
        alphas = [0.8, 0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7]
        
        unique_labels = np.unique(self.combined_labels)
        
        for label in unique_labels:
            mask = self.combined_labels == label
            label_data = low_dim_features[mask]
            
            if len(label_data) > 0:
                ax.scatter(
                    label_data[:, 0], label_data[:, 1],
                    c=colors[label], s=sizes[label], alpha=alphas[label],
                    edgecolors='white', linewidths=1,
                    marker=markers[label], label=self.group_names[label]
                )

        ax.set_title(
            f"Sample Distribution (2D {method.upper()})\nClasses: {', '.join([name.replace('_', ' ').title() for name in self.target_classes])}",
            fontsize=14, pad=20
        )
        ax.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
        ax.set_ylabel(f"{method.upper()} Component 2", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9)
        
        # 保存图像
        save_path = os.path.join(self.save_dir, f"sample_distribution_{method}_2d_four_classes_separated.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor="white")
        plt.close()
        print(f"2D分布图保存完成：{save_path}")


    def plot_3d_distribution(self, low_dim_features, method="pca"):
        """3D分布可视化"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 定义颜色和标记 - 每个类别有原始和增强两种
        colors = ['blue', 'lightblue', 'red', 'pink', 'green', 'lightgreen', 'orange', 'wheat']
        markers = ['o', '^', 'o', '^', 'o', '^', 'o', '^']  # 圆形表示原始，三角表示增强
        sizes = [60, 50, 60, 50, 60, 50, 60, 50]
        alphas = [0.7, 0.6, 0.7, 0.6, 0.7, 0.6, 0.7, 0.6]
        
        unique_labels = np.unique(self.combined_labels)
        
        for label in unique_labels:
            mask = self.combined_labels == label
            label_data = low_dim_features[mask]
            
            if len(label_data) > 0:
                ax.scatter(
                    label_data[:, 0], label_data[:, 1], label_data[:, 2],
                    c=colors[label], s=sizes[label], alpha=alphas[label],
                    edgecolors='white', linewidths=0.5,
                    marker=markers[label], label=self.group_names[label]
                )

        ax.set_title(
            f"Sample Distribution (3D {method.upper()})\nClasses: {', '.join([name.replace('_', ' ').title() for name in self.target_classes])}",
            fontsize=14, pad=20
        )
        ax.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
        ax.set_ylabel(f"{method.upper()} Component 2", fontsize=12)
        ax.set_zlabel(f"{method.upper()} Component 3", fontsize=12)
        ax.legend(loc="upper right", framealpha=0.9)
        
        # 保存图像
        save_path = os.path.join(self.save_dir, f"sample_distribution_{method}_3d_four_classes_separated.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor="white")
        plt.close()
        print(f"3D分布图保存完成：{save_path}")


    def run(self):
        """完整流程：提取特征 → 过滤类别 → 降维 → 可视化"""
        self.extract_hidden_features()
        
        # 准备合并的特征和标签
        combined_features = self.prepare_combined_features_and_labels()
        
        for method in ["pca", "lda"]:
            print(f"\n使用{method.upper()}降维...")
            
            # 2D可视化
            print("生成2D可视化...")
            low_dim_2d = self.reduce_dimension(combined_features, method=method, n_components=2)
            self.plot_2d_distribution(low_dim_2d, method=method)
            
            # 3D可视化
            print("生成3D可视化...")
            low_dim_3d = self.reduce_dimension(combined_features, method=method, n_components=3)
            self.plot_3d_distribution(low_dim_3d, method=method)
        
        print(f"\n所有任务完成！结果保存在：{self.save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Sample Distribution Visualization")
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], help='数据集类型')
    parser.add_argument('--num_points', type=int, default=1024, help='每个样本的点云数量')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    # 模型参数
    parser.add_argument('--model_path', type=str, default='checkpoints/dgcnn/models/model.t7', help='预训练DGCNN模型路径')
    parser.add_argument('--classes', type=int, default=40, help='分类类别数')
    parser.add_argument('--emb_dims', type=int, default=1024, help='DGCNN嵌入层维度')
    parser.add_argument('--k', type=int, default=20, help='DGCNN KNN的k值')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    # PointWOLF参数
    parser.add_argument('--PointWOLF', action='store_true', help='是否使用PointWOLF')
    parser.add_argument('--AugTune', action='store_true', help='是否使用PointWOLF')
    parser.add_argument('--w_num_anchor', type=int, default=4, help='PointWOLF锚点数量')
    parser.add_argument('--w_sample_type', type=str, default='fps', choices=['fps', 'random'], help='PointWOLF锚点采样方式')
    parser.add_argument('--w_sigma', type=float, default=0.5, help='PointWOLF核带宽')
    parser.add_argument('--w_R_range', type=float, default=10, help='最大旋转范围')
    parser.add_argument('--w_S_range', type=float, default=3, help='最大缩放范围')
    parser.add_argument('--w_T_range', type=float, default=0.25, help='最大平移范围')
    
    args = parser.parse_args()
    print("="*50)
    print("参数配置：")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("="*50)

    visualizer = EnhancedSampleVisualizer(args)
    visualizer.run()


if __name__ == '__main__':
    main()