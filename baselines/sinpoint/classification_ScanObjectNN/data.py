#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) 
    return pointcloud


def normalize_pointcloud(pointcloud):
    pointcloud -= pointcloud.mean(0)
    d = ((pointcloud**2).sum(-1)**(1./2)).max()
    pointcloud /= d
    return pointcloud
   
   
class ScanObjectNN(Dataset):
    def __init__(self, num_points, partition='train', ver = "easy"):
        self.num_points = num_points
        self.partition = partition 
        self.ver = ver
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'h5_files')
        
        if ver=="easy":
            if self.partition == 'train':
                file = os.path.join(DATA_DIR, "main_split/training_objectdataset_augmentedrot_scale75.h5")
            else:
                file = os.path.join(DATA_DIR, "main_split/test_objectdataset_augmentedrot_scale75.h5")
                
        elif ver =="hard":
            if self.partition == 'train':
                file = os.path.join(DATA_DIR, "main_split/training_objectdataset_augmentedrot_scale75.h5")
            else:
                file = os.path.join(DATA_DIR, "main_split/test_objectdataset_augmentedrot_scale75.h5")
                
        else:
            raise NotImplementedError

        f = h5py.File(file, 'r')
        self.data = f['data'][:].astype('float32')
        self.label = f['label'][:].astype('int64')
        f.close()

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        if self.partition == 'train':
            #pointcloud = rotate_pointcloud(pointcloud)
            #pointcloud = jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05)
            #pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]