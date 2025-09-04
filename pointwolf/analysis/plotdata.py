import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import argparse
import os
import tqdm

from data import ModelNet40
from model import DGCNN


class EnhancedSampleVisualizer:
    def __init__(self, args):
        self.args = args
        self.save_dir = os.path.join(
            "enhanced_sample_distribution",
            f"{'with_pointwolf' if args.PointWOLF else 'without_pointwolf'}"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self._init_dataset()
        self._init_model()

        self.features = {
            "origin": [],
            "corrupted": [], 
            "labels": []     
        }

        self.target_classes = ["night_stand", "plant", "dresser", "flower_pot"]
        self.class_names = [
            'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 
            'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 
            'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 
            'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 
            'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 
            'wardrobe', 'xbox'
        ]


    def _init_dataset(self):
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
        self.model = DGCNN(self.args).to(self.device)
        
        assert os.path.exists(self.args.model_path), f"模型路径不存在：{self.args.model_path}"
        checkpoint = torch.load(self.args.model_path, map_location=self.device)
        if "module." in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"模型加载完成：{self.args.model_path}")


    def extract_hidden_features(self):
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

        self.features["origin"] = np.concatenate(self.features["origin"], axis=0)
        self.features["corrupted"] = np.concatenate(self.features["corrupted"], axis=0)
        self.features["labels"] = np.concatenate(self.features["labels"], axis=0)

        print(f"特征提取完成：")
        print(f"  原始样本特征形状：{self.features['origin'].shape}")
        print(f"  增强样本特征形状：{self.features['corrupted'].shape}")
        print(f"  标签形状：{self.features['labels'].shape}")

        self._filter_target_classes()


    def _filter_target_classes(self):
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
        
        mask = np.isin(self.features["labels"], target_indices)
        
        self.features["origin"] = self.features["origin"][mask]
        self.features["corrupted"] = self.features["corrupted"][mask]
        self.features["labels"] = self.features["labels"][mask]
        
        print(f"过滤后数据形状：")
        print(f"  原始样本特征形状：{self.features['origin'].shape}")
        print(f"  增强样本特征形状：{self.features['corrupted'].shape}")
        print(f"  标签形状：{self.features['labels'].shape}")
        
        unique_labels, counts = np.unique(self.features["labels"], return_counts=True)
        for label, count in zip(unique_labels, counts):
            class_name = self.class_names[label]
            print(f"  类别 {class_name}: {count} 个样本")


    def _adjust_pc_dim(self, pc):
        if pc.dim() != 3:
            raise ValueError(f"点云维度错误：需为3维，当前为{pc.dim()}维")
        if pc.size(1) != 3 and pc.size(2) == 3:
            pc = pc.permute(0, 2, 1)
        return pc


    def prepare_combined_features_and_labels(self):
        """修复后的组合标签准备"""
        combined_features = np.vstack([self.features["origin"], self.features["corrupted"]])
        
        # 获取目标类别的索引映射
        target_indices = [self.class_names.index(cls) for cls in self.target_classes]
        
        # LDA标签 - 只关注类别信息（不区分原始/增强）
        self.lda_labels = np.concatenate([self.features["labels"], self.features["labels"]])
        
        # 可视化标签 - 区分原始和增强样本
        self.viz_labels = []
        
        # 创建映射：目标类别索引 -> 可视化组索引
        self.class_to_viz_index = {}
        for i, class_idx in enumerate(target_indices):
            self.class_to_viz_index[class_idx] = i * 2  # 原始样本
            self.class_to_viz_index[class_idx + 1000] = i * 2 + 1  # 增强样本（使用+1000避免冲突）
        
        # 为原始样本分配可视化标签
        for label in self.features["labels"]:
            self.viz_labels.append(self.class_to_viz_index[label])
        
        # 为增强样本分配可视化标签
        for label in self.features["labels"]:
            self.viz_labels.append(self.class_to_viz_index[label + 1000])
        
        self.viz_labels = np.array(self.viz_labels)
        
        # 创建组名称
        self.group_names = []
        for class_name in self.target_classes:
            class_name_formatted = class_name.replace('_', ' ').title()
            self.group_names.append(f"{class_name_formatted} (Original)")
            self.group_names.append(f"{class_name_formatted} (Enhanced)")
        
        print(f"合并特征形状: {combined_features.shape}")
        print(f"LDA标签形状: {self.lda_labels.shape}")
        print(f"可视化标签形状: {self.viz_labels.shape}")
        print(f"组名称: {self.group_names}")
        print(f"可视化标签分布: {np.unique(self.viz_labels, return_counts=True)}")
        
        return combined_features


    def reduce_dimension(self, features, method="pca", n_components=3):
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
            low_dim = reducer.fit_transform(features_scaled)
            print(f"PCA降维完成：解释方差比 = {reducer.explained_variance_ratio_}")
            return low_dim
            
        elif method == "lda":
            # 修改：LDA只使用原始样本，不包含增强样本
            print("LDA: 只使用原始样本（不包含增强样本）...")
            
            # 只取原始样本部分
            n_origin = len(self.features["origin"])
            features_origin = features_scaled[:n_origin]  # 只使用原始样本
            labels_origin = self.features["labels"]       # 原始样本的标签
            
            n_unique_labels = len(np.unique(labels_origin))
            max_components = min(n_components, n_unique_labels - 1, features_origin.shape[1])
            
            print(f"LDA: 使用{len(np.unique(labels_origin))}个类别，{n_origin}个原始样本，最多可降到{max_components}维")
            
            if max_components < 2:
                print("警告：LDA维度不足，使用PCA替代")
                reducer = PCA(n_components=n_components, random_state=42)
                low_dim = reducer.fit_transform(features_origin)
            else:
                reducer = LDA(n_components=max_components)
                low_dim = reducer.fit_transform(features_origin, labels_origin)
                if hasattr(reducer, 'explained_variance_ratio_'):
                    print(f"LDA解释方差比: {reducer.explained_variance_ratio_}")
            
            print(f"LDA降维完成：只包含原始样本 {low_dim.shape}")
            return low_dim
            
        elif method == "tsne":
            print(f"使用t-SNE降维到{n_components}维...")
            print("注意：t-SNE计算可能较慢，特别是对于大数据集")
            
            # 确保数据类型正确
            features_scaled = features_scaled.astype(np.float64)
            
            # 如果数据量太大，采样一部分进行可视化
            max_samples = 1000  # 减少采样数量
            if len(features_scaled) > max_samples:
                print(f"数据量较大({len(features_scaled)})，随机采样{max_samples}个样本进行t-SNE")
                indices = np.random.choice(len(features_scaled), max_samples, replace=False)
                features_sampled = features_scaled[indices]
                self.viz_labels_sampled = self.viz_labels[indices]
                self.lda_labels_sampled = self.lda_labels[indices]
            else:
                features_sampled = features_scaled
                self.viz_labels_sampled = self.viz_labels
                self.lda_labels_sampled = self.lda_labels
            
            # 更保守的t-SNE参数
            tsne_params = {
                'n_components': n_components,
                'random_state': 42,
                'perplexity': min(30, len(features_sampled) - 1),
                'n_iter': 500,  # 减少迭代次数
                'learning_rate': 200,  # 固定学习率
                'init': 'random',
                'verbose': 1  # 显示进度
            }
            
            try:
                reducer = TSNE(**tsne_params)
                low_dim = reducer.fit_transform(features_sampled)
                print("t-SNE降维完成")
                return low_dim
            except Exception as e:
                print(f"t-SNE失败: {e}")
                # 如果t-SNE失败，使用PCA替代
                print("使用PCA替代t-SNE")
                reducer = PCA(n_components=n_components, random_state=42)
                return reducer.fit_transform(features_sampled)
            
        else:
            raise ValueError(f"不支持的降维方法：{method}")


    def plot_2d_distribution(self, low_dim_features, method="pca"):
        """修复后的2D可视化"""
        plt.rcParams.update({"font.size": 12, "figure.facecolor": "white"})
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # 为4个类别定义颜色（每个类别有原始和增强）
        colors = [
            '#1f77b4', '#aec7e8',  # night_stand: 深蓝, 浅蓝
            '#ff7f0e', '#ffbb78',  # plant: 深橙, 浅橙  
            '#2ca02c', '#98df8a',  # dresser: 深绿, 浅绿
            '#d62728', '#ff9896'   # flower_pot: 深红, 浅红
        ]
        
        markers = ['o', '^']  # 圆形=原始, 三角=增强
        sizes = [80, 60]
        alphas = [0.8, 0.7]
        
        # 对于LDA，只显示原始样本
        if method == "lda":
            # 只使用原始样本的标签
            n_origin = len(self.features["origin"])
            viz_labels_to_use = self.viz_labels[:n_origin]  # 只取原始样本的标签
            
            # 只显示原始样本的数据
            low_dim_features = low_dim_features  # 已经是原始样本的数据
            
            print(f"LDA可视化：只显示{len(low_dim_features)}个原始样本")
        elif method == "tsne":
            # 对于t-SNE，使用采样后的标签
            viz_labels_to_use = self.viz_labels_sampled
        else:
            # 对于PCA，使用所有标签
            viz_labels_to_use = self.viz_labels
        
        unique_viz_labels = np.unique(viz_labels_to_use)
        
        for viz_label in unique_viz_labels:
            mask = viz_labels_to_use == viz_label
            
            # 对于LDA，只显示偶数标签（原始样本）
            if method == "lda" and viz_label % 2 == 1:
                continue  # 跳过增强样本
                
            label_data = low_dim_features[mask]
            
            if len(label_data) > 0:
                # 确定颜色索引和标记类型
                color_idx = viz_label  # 直接使用可视化标签作为颜色索引
                marker_idx = viz_label % 2  # 0=原始, 1=增强
                
                # 对于LDA，所有点都是原始样本，使用原始样本的样式
                if method == "lda":
                    marker_idx = 0  # 强制使用圆形标记
                
                ax.scatter(
                    label_data[:, 0], label_data[:, 1],
                    c=colors[color_idx], s=sizes[marker_idx], 
                    alpha=alphas[marker_idx],
                    edgecolors='white', linewidths=1,
                    marker=markers[marker_idx], 
                    label=self.group_names[viz_label]
                )

        method_display = "t-SNE" if method == "tsne" else method.upper()
        title_suffix = " (Original samples only)" if method == "lda" else ""
        ax.set_title(
            f"Sample Distribution (2D {method_display}){title_suffix}\nClasses: {', '.join([name.replace('_', ' ').title() for name in self.target_classes])}",
            fontsize=14, pad=20
        )
        ax.set_xlabel(f"{method_display} Component 1", fontsize=12)
        ax.set_ylabel(f"{method_display} Component 2", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9)
        
        save_path = os.path.join(self.save_dir, f"sample_distribution_{method}_2d_four_classes_separated.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor="white")
        plt.close()
        print(f"2D分布图保存完成：{save_path}")


    def plot_3d_distribution(self, low_dim_features, method="pca"):
        """修复后的3D可视化"""
        # t-SNE 通常只用于2D可视化，跳过3D t-SNE
        if method == "tsne":
            print("跳过3D t-SNE可视化（t-SNE通常用于2D）")
            return
            
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = [
            '#1f77b4', '#aec7e8',  # night_stand
            '#ff7f0e', '#ffbb78',  # plant  
            '#2ca02c', '#98df8a',  # dresser
            '#d62728', '#ff9896'   # flower_pot
        ]
        
        markers = ['o', '^']
        sizes = [60, 50]
        alphas = [0.7, 0.6]
        
        # 对于LDA，只显示原始样本
        if method == "lda":
            # 只使用原始样本的标签
            n_origin = len(self.features["origin"])
            viz_labels_to_use = self.viz_labels[:n_origin]  # 只取原始样本的标签
            print(f"LDA 3D可视化：只显示{len(low_dim_features)}个原始样本")
        else:
            viz_labels_to_use = self.viz_labels
        
        unique_viz_labels = np.unique(viz_labels_to_use)
        
        for viz_label in unique_viz_labels:
            # 对于LDA，只显示偶数标签（原始样本）
            if method == "lda" and viz_label % 2 == 1:
                continue  # 跳过增强样本
                
            if method == "lda":
                mask = viz_labels_to_use == viz_label
            else:
                mask = self.viz_labels == viz_label
                
            label_data = low_dim_features[mask]
            
            if len(label_data) > 0:
                color_idx = viz_label
                marker_idx = viz_label % 2
                
                # 对于LDA，所有点都是原始样本，使用原始样本的样式
                if method == "lda":
                    marker_idx = 0  # 强制使用圆形标记
                
                ax.scatter(
                    label_data[:, 0], label_data[:, 1], label_data[:, 2],
                    c=colors[color_idx], s=sizes[marker_idx], alpha=alphas[marker_idx],
                    edgecolors='white', linewidths=0.5,
                    marker=markers[marker_idx], label=self.group_names[viz_label]
                )

        title_suffix = " (Original samples only)" if method == "lda" else ""
        ax.set_title(
            f"Sample Distribution (3D {method.upper()}){title_suffix}\nClasses: {', '.join([name.replace('_', ' ').title() for name in self.target_classes])}",
            fontsize=14, pad=20
        )
        ax.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
        ax.set_ylabel(f"{method.upper()} Component 2", fontsize=12)
        ax.set_zlabel(f"{method.upper()} Component 3", fontsize=12)
        ax.legend(loc="upper right", framealpha=0.9)
        
        save_path = os.path.join(self.save_dir, f"sample_distribution_{method}_3d_four_classes_separated.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor="white")
        plt.close()
        print(f"3D分布图保存完成：{save_path}")


    def run(self):
        self.extract_hidden_features()
        combined_features = self.prepare_combined_features_and_labels()
        
        # 原有的PCA和LDA可视化
        for method in ["pca", "lda"]:
            print(f"\n使用{method.upper()}降维...")
            
            print("生成2D可视化...")
            low_dim_2d = self.reduce_dimension(combined_features, method=method, n_components=2)
            self.plot_2d_distribution(low_dim_2d, method=method)
            
            print("生成3D可视化...")
            low_dim_3d = self.reduce_dimension(combined_features, method=method, n_components=3)
            self.plot_3d_distribution(low_dim_3d, method=method)
        
        # 新增t-SNE可视化（只做2D）
        print(f"\n使用t-SNE降维...")
        print("生成2D t-SNE可视化...")
        try:
            low_dim_tsne = self.reduce_dimension(combined_features, method="tsne", n_components=2)
            self.plot_2d_distribution(low_dim_tsne, method="tsne")
            print("t-SNE可视化完成")
        except Exception as e:
            print(f"t-SNE可视化失败：{e}")
            print("这可能是因为数据量太大或内存不足，可以尝试减少样本数量")
        
        print(f"\n所有任务完成！结果保存在：{self.save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Sample Distribution Visualization")
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'])
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='checkpoints/dgcnn/models/model.t7')
    parser.add_argument('--classes', type=int, default=40)
    parser.add_argument('--emb_dims', type=int, default=1024)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--PointWOLF', action='store_true')
    parser.add_argument('--AugTune', action='store_true')
    parser.add_argument('--w_num_anchor', type=int, default=4)
    parser.add_argument('--w_sample_type', type=str, default='fps', choices=['fps', 'random'])
    parser.add_argument('--w_sigma', type=float, default=0.3)
    parser.add_argument('--w_R_range', type=float, default=10)
    parser.add_argument('--w_S_range', type=float, default=3)
    parser.add_argument('--w_T_range', type=float, default=0.25)
    
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
