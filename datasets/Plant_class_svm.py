'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import sys
import glob
import h5py

import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *
import torch

warnings.filterwarnings('ignore')
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

@DATASETS.register_module()
class Plant_class_svm(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.pc_root = config.PC_PATH
        self.uniform = True
        split = config.subset
        self.subset = config.subset
        self.catfile = os.path.join(self.root, 'class_to_number.txt')
        self.cat = [line.rstrip().split(" ")[1] for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        assert (split == 'train' or split == 'test')
        if split == 'train':
            self.data_list = [line.rstrip() for line in open(os.path.join(self.root, 'SVM_train_set.txt'))]
        elif  split == 'test':
            self.data_list = [line.rstrip() for line in open(os.path.join(self.root, 'SVM_test_set.txt'))]

        print_log('The size of %s data is %d' % (split, len(self.data_list)), logger = 'Plant_class_svm')

    def __len__(self):
        return len(self.data_list)
    def _get_item(self, index):
        fn = self.data_list[index]
        cls = self.classes[self.data_list[index].split("-")[0]]
        label = np.array([cls]).astype(np.int32)
        point_set = np.load(os.path.join(self.pc_root,fn)).astype(np.float32)
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints, :]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        return point_set, label


    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return current_points, label

@DATASETS.register_module()
class Plant_class_linear(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.pc_root = config.PC_PATH
        self.uniform = True
        split = config.subset
        self.subset = config.subset
        self.catfile = os.path.join(self.root, 'class_to_number.txt')
        self.cat = [line.rstrip().split(" ")[1] for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        assert (split == 'train' or split == 'test')
        if split == 'train':
            self.data_list = [line.rstrip() for line in open(os.path.join(self.root, 'SVM_train_set.txt'))]
        elif split == 'test':
            self.data_list = [line.rstrip() for line in open(os.path.join(self.root, 'SVM_test_set.txt'))]

        print_log('The size of %s data is %d' % (split, len(self.data_list)), logger = 'Plant_class_svm')

    def __len__(self):
        return len(self.data_list)
    def _get_item(self, index):
        fn = self.data_list[index]
        cls = self.classes[self.data_list[index].split("-")[0]]
        label = np.array([cls]).astype(np.int32)
        point_set = np.load(os.path.join(self.pc_root,fn)).astype(np.float32)
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints, :]
        # point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        return point_set, label


    def __getitem__(self, index):
        points, label = self._get_item(index)
        current_points = torch.from_numpy(points).float()
        return current_points, label

