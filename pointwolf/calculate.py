data = {
    "background": [0.7929, 0.7909, 0.7731, 0.7646, 0.7306],
    "cutout": [0.9068, 0.9100, 0.9040, 0.8963, 0.8963],
    "density": [0.9048, 0.9040, 0.9060, 0.9024, 0.8975],
    "density_inc": [0.9024, 0.9036, 0.9068, 0.9032, 0.9052],
    "distortion": [0.8914, 0.8748, 0.8635, 0.8217, 0.7658],
    "distortion_rbf": [0.8947, 0.8841, 0.8635, 0.8213, 0.7788],
    "distortion_rbf_inv": [0.8983, 0.8845, 0.8618, 0.8323, 0.7950],
    "gaussian": [0.8922, 0.8821, 0.8598, 0.8241, 0.7731],
    "impulse": [0.8691, 0.8626, 0.8444, 0.8185, 0.7763],
    "lidar": [0.5194, 0.6058, 0.6220, 0.5182, 0.4526],
    "occlusion": [0.5357, 0.5770, 0.5972, 0.5308, 0.4724],
    "rotation": [0.8938, 0.8780, 0.8136, 0.7289, 0.5332],
    "shear": [0.8955, 0.8874, 0.8562, 0.8193, 0.7792],
    "uniform": [0.8845, 0.8525, 0.8246, 0.7804, 0.7152],
    "upsampling": [0.8720, 0.8614, 0.8525, 0.8395, 0.7925]
}

# 计算每个corruption的平均值
corruption_means = {}
for corruption, accuracies in data.items():
    mean_acc = sum(accuracies) / len(accuracies)
    corruption_means[corruption] = mean_acc

# 计算所有准确率的总体平均值
all_accuracies = []
for accuracies in data.values():
    all_accuracies.extend(accuracies)

overall_mean = sum(all_accuracies) / len(all_accuracies)

# 打印结果
print("各Corruption的平均准确率:")
print("-" * 40)
for corruption, mean_acc in sorted(corruption_means.items()):
    print(f"{corruption:20}: {mean_acc:.4f}")

print("-" * 40)
print(f"{'总体平均值':20}: {overall_mean:.4f}")
print("-" * 40)

# 按平均值排序
print("\n按平均准确率排序:")
print("-" * 40)
sorted_means = sorted(corruption_means.items(), key=lambda x: x[1], reverse=True)
for i, (corruption, mean_acc) in enumerate(sorted_means, 1):
    print(f"{i:2d}. {corruption:20}: {mean_acc:.4f}")
