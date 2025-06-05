import random
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from pointnet2_utils import farthest_point_sample, pc_normalize
import json
from all_tools import read_ply2np
from util.data_util import data_prepare_v101 as data_prepare
from util.data_util import collate_fn
# from util import transform


class PartNormalDataset(Dataset):
    def __init__(self, root, npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)

class PartPlants(Dataset):
    def __init__(self, root='', npoints=2500, split='train', class_choice=None, normal_channel=False,sample_num=1,loop=1,transform=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'split/classs2number.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)
        self.meta = {}
        train_list = []
        test_list = []
        val_list = []
        self.cloud_names = []
        with open(os.path.join(self.root, 'split', 'train.txt'), 'r') as f:
            for line in f:
                train_list.append(line.strip().split("/")[1].split(".")[0])
        with open(os.path.join(self.root, 'split', 'val.txt'), 'r') as f:
            for line in f:
                val_list.append(line.strip().split("/")[1].split(".")[0])
        with open(os.path.join(self.root, 'split', 'test.txt'), 'r') as f:
            for line in f:
                test_list.append(line.strip().split("/")[1].split(".")[0])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_list) or (fn[0:-4] in val_list))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_list]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_list]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_list]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.ply'))
                self.cloud_names.append(token)
        self.loop = loop
        self.transform =transform
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        if split == 'test':
            self.datapath = self.datapath
        else:
            if sample_num<len(self.datapath):
                self.datapath = random.sample(self.datapath,sample_num)
            else:
                self.datapath = self.datapath
        self.data_idx = np.arange(len(self.datapath))

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index% len(self.data_idx)]
        else:
            fn = self.datapath[index%len(self.data_idx)]
            cat = self.datapath[index% len(self.data_idx)][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = read_ply2np(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)  #straw xiugai
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if self.transform is not None:
            point_set[:, 0:3] =  self.transform(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        return point_set, cls, seg,self.cloud_names[index% len(self.data_idx)]

    def __len__(self):
        return len(self.datapath)* self.loop



class Plant_second(Dataset):
    def __init__(self, split='train', data_root='trainval', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1,class_choice=None):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        self.data_root = data_root
        self.catfile = os.path.join(self.data_root, 'classs2number.txt')
        self.cat = {}
        self.class_choice = class_choice
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        self.meta = {}
        train_list = []
        test_list = []
        val_list = []
        self.cloud_names = []
        with open(os.path.join(self.data_root, 'split', 'train.txt'), 'r') as f:
            for line in f:
                train_list.append(line.strip().split("/")[1].split(".")[0])
        with open(os.path.join(self.data_root, 'split', 'val.txt'), 'r') as f:
            for line in f:
                val_list.append(line.strip().split("/")[1].split(".")[0])
        with open(os.path.join(self.data_root, 'split', 'test.txt'), 'r') as f:
            for line in f:
                test_list.append(line.strip().split("/")[1].split(".")[0])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.data_root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_list) or (fn[0:-4] in val_list))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_list]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_list]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_list]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.ply'))
                self.cloud_names.append(token)

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        #data_num = self.classes[]
        self.data_idx = np.arange(len(self.datapath))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index % len(self.data_idx)]
        else:
            fn = self.datapath[index% len(self.data_idx)]
            cat = self.datapath[index% len(self.data_idx)][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = read_ply2np(fn[1]).astype(np.float32)
            point_set = data[:, 0:3]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index% len(self.data_idx)] = (point_set, cls, seg)
        if data.shape[1]>4:
            color = data[:,3:6]
        else:
            color = np.zeros((len(data), 3))
        point_set = pc_normalize(point_set)
        coord, color, seg = data_prepare(point_set, color, seg, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)

        return coord,color,seg

    def __len__(self):
        return len(self.datapath) * self.loop
