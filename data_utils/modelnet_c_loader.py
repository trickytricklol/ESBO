import os
import numpy as np

def load_modelnet_c(data_dir, corruption_type='background', severity=1):
    """
    加载ModelNet-C数据（统一标签文件版本）
    
    参数:
        data_dir: 包含.npy文件的目录路径
        corruption_type: 损坏类型 (如'background')
        severity: 损坏严重程度 (1-5)
        
    返回:
        points: 点云数据 (n_samples, n_points, 3)
        labels: 对应标签 (n_samples,)
    """
    # 构建文件路径
    data_file = os.path.join(data_dir, f"data_{corruption_type}_{severity}.npy")
    label_file = os.path.join(data_dir, "label.npy")  # 统一标签文件
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        available_files = [f for f in os.listdir(data_dir) 
                          if f.startswith(f"data_{corruption_type}")]
        raise FileNotFoundError(
            f"数据文件 {data_file} 不存在\n"
            f"该损坏类型可用的严重级别: {sorted(list(set(f.split('_')[-1].split('.')[0] for f in available_files)))}"
        )
    
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"标签文件 {label_file} 不存在")
    
    # 加载数据
    points = np.load(data_file)
    labels = np.load(label_file)
    
    # 确保数据点数量与标签数量匹配
    if len(points) != len(labels):
        raise ValueError(
            f"数据与标签数量不匹配: "
            f"数据有{len(points)}个样本, "
            f"标签有{len(labels)}个"
        )
    
    return points, labels

def list_corruption_types(data_dir):
    """列出所有可用的损坏类型"""
    files = os.listdir(data_dir)
    corruptions = set()
    
    for f in files:
        if f.startswith("data_") and f.endswith(".npy"):
            parts = f.split('_')
            if len(parts) == 3:  # data_xxx_?.npy
                corruptions.add(parts[1])
    
    return sorted(list(corruptions))

def list_severity_levels(data_dir, corruption_type):
    """列出指定损坏类型可用的严重级别"""
    files = os.listdir(data_dir)
    severities = set()
    
    for f in files:
        if f.startswith(f"data_{corruption_type}_") and f.endswith(".npy"):
            severity = f.split('_')[-1].split('.')[0]
            if severity.isdigit():
                severities.add(int(severity))
    
    return sorted(list(severities))

if __name__ == "__main__":
    # 设置数据目录 - 使用您提供的绝对路径
    data_dir = "/mnt/c/Users/Administrator/readmodelnet/modelnet40_c"
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    print(f"正在扫描目录: {data_dir}")
    print("目录内容:", os.listdir(data_dir))
    
    # 列出所有可用的损坏类型
    corruptions = list_corruption_types(data_dir)
    print("\n可用的损坏类型:", corruptions)
    
    if not corruptions:
        raise ValueError("没有找到任何数据文件 (data_xxx_?.npy)")
    
    # 使用第一个可用的损坏类型
    corruption = corruptions[0]
    print(f"\n选择的损坏类型: {corruption}")
    
    # 列出该损坏类型可用的严重级别
    severities = list_severity_levels(data_dir, corruption)
    print("可用的严重级别:", severities)
    
    if not severities:
        raise ValueError(f"损坏类型 {corruption} 没有可用的数据文件")
    
    # 使用最低严重级别
    severity = severities[0]
    
    try:
        print(f"\n尝试加载: {corruption} 类型, 严重级别 {severity}")
        points, labels = load_modelnet_c(data_dir, corruption, severity)
        
        print("\n数据加载成功!")
        print(f"加载的数据文件: data_{corruption}_{severity}.npy")
        print(f"加载的标签文件: label.npy")
        print(f"点云数据形状: {points.shape} (样本数×点数×3)")
        print(f"标签数据形状: {labels.shape}")
        print("\n数据示例:")
        print(f"第一个样本的点云 (前5个点):\n{points[0][:5].round(4)}")
        print(f"对应标签: {labels[0]}")
        
        # 验证标签范围
        print(f"\n标签范围: {np.min(labels)} 到 {np.max(labels)}")
        print(f"唯一标签值: {np.unique(labels)}")
        
    except Exception as e:
        print(f"\n加载数据失败: {str(e)}")
        print("\n调试建议:")
        print("1. 确认目录内容是否符合预期")
        print("2. 检查文件命名是否准确")
        print("3. 验证文件是否完整无损坏")