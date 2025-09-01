"""
ScanObjectNN download: http://103.24.77.34/scanobjectnn/h5_files.zip
"""

import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# 添加PointWOLF导入
from PointWOLF import PointWOLF


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'h5_files')):
        os.mkdir(os.path.join(DATA_DIR, 'h5_files'))
        # note that this link only contains the hardest perturbed variant (PB_T50_RS).
        # for full versions, consider the following link.
        www = 'https://github.com/ma-xu/pointMLP-pytorch/releases/download/dataset/h5_files.zip'
        # www = 'http://103.24.77.34/scanobjectnn/h5_files.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_scanobjectnn_data(partition):
    #download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []

    h5_name = BASE_DIR + '/data/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def normalize_point_cloud_batch(pointcloud):
    """
    input : 
        pointcloud([B,3,N])
        
    output :
        pointcloud([B,3,N]) : Normalized Pointclouds
    """
    pointcloud = pointcloud - pointcloud.mean(dim=-1, keepdim=True) #(B,3,N)
    scale = 1/torch.sqrt((pointcloud**2).sum(1)).max(axis=1)[0]*0.999999 # (B)
    pointcloud = scale.view(-1, 1, 1) * pointcloud
    return pointcloud


def translate_pointcloud_batch(pointcloud):
    """
    input : 
        pointcloud([B,3,N])
        
    output :
        translated_pointcloud([B,3,N]) : Pointclouds after CDA
    """
    B, _, _ = pointcloud.shape
    
    xyz1 = torch.FloatTensor(B,3,1).uniform_(2./3., 3./2.).to(pointcloud.device)
    xyz2 = torch.FloatTensor(B,3,1).uniform_(-0.2, 0.2).to(pointcloud.device)
       
    translated_pointcloud = xyz1 * pointcloud + xyz2
    return translated_pointcloud


class ScanObjectNN(Dataset):
    def __init__(self, args, partition='training'):
        self.data, self.label = load_scanobjectnn_data(partition)
        self.num_points = args.num_points
        self.partition = partition
        self.PointWOLF = PointWOLF(args) if args.PointWOLF else None
        self.AugTune = args.AugTune

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points] #(1024,3)
        label = self.label[item]
        
        if self.partition == 'training':
            np.random.shuffle(pointcloud)
            
            if self.PointWOLF is not None:
                origin, pointcloud = self.PointWOLF(pointcloud)
                if self.AugTune:
                    # When AugTune used, we conduct CDA after AugTune.      
                    return origin, pointcloud, label
                
            pointcloud = translate_pointcloud(pointcloud)
            
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    # 测试代码
    class Args:
        def __init__(self):
            self.num_points = 1024
            self.PointWOLF = True
            self.AugTune = True
    
    args = Args()
    
    train = ScanObjectNN(args, 'training')
    test = ScanObjectNN(args, 'test')
    
    # 测试普通模式
    data, label = train[0]
    print("普通模式:", data.shape, label)
    
    # 测试AugTune模式
    origin, augmented, label = train[1]
    print("AugTune模式 - origin:", origin.shape)
    print("AugTune模式 - augmented:", augmented.shape)
    print("AugTune模式 - label:", label)
    
    # 测试测试集
    data, label = test[0]
    print("测试集:", data.shape, label)
