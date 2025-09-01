# bottleneck_test.py
import time
import torch
from torch.utils.data import DataLoader
from data import ModelNet40
from model import pointnet2

def test_bottleneck(args):
    device = torch.device("cuda")
    
    # 测试数据加载
    dataset = ModelNet40(args, partition='train')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, pin_memory=True)
    
    # 测试数据加载速度
    start = time.time()
    for i, batch in enumerate(loader):
        if args.AugTune:
            # AugTune模式下返回3个值
            origin, data, label = batch
        else:
            # 普通模式下返回2个值
            data, label = batch
    data_loading_time = time.time() - start
    print(f"数据加载速度: {100/data_loading_time:.2f} batches/s")
    
    # 测试模型计算速度
    model = pointnet2(args).to(device)
    if args.AugTune:
        # AugTune模式下返回3个值
        origin, data, label = next(iter(loader))
    else:
        # 普通模式下返回2个值
        data, label = next(iter(loader))
    data, label = data.to(device), label.to(device)
    data = data.permute(0, 2, 1)
    
    # Warm up
    for _ in range(10):
        _ = model(data)
    
    # 正式测试
    start = time.time()
    for i in range(100):
        _ = model(data)
    torch.cuda.synchronize()
    computation_time = time.time() - start
    print(f"模型计算速度: {100/computation_time:.2f} batches/s")
