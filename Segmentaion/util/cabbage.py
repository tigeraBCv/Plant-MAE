import os
from multiprocessing import Pool

import numpy as np
import SharedArray as SA

import torch
from torch.utils.data import Dataset

from util.edgeExtraction import edge_extraction
from util.s3dis import S3DIS
from util.voxelize import voxelize
from util.data_util import sa_create, collate_fn
from util.data_util import data_prepare_edge as data_prepare
from util.data_util import data_prepare_v101 as data_prepare_notedge
from zhu_util.all_tools import read_ply2np,save_ply_from_np


def get_edge(file_name):
    # data = read_ply2np("data/" + file_name)
    print(file_name)
    if not os.path.exists("/dev/shm/{}".format(file_name + "_edge")):
        data = SA.attach("shm://{}".format(file_name)).copy()
        if data.shape[0] > 300000:
            ran_sample = np.random.choice(len(data), 300000, replace=False)
            data = data[ran_sample]
        edge_idx = edge_extraction(data)
        sa_create("shm://{}".format(file_name + "_edge"), data[edge_idx])
        # save_ply_from_np(data[edge_idx], "result/" + file_name)
        print(file_name + "_edge get")
    return


class Cabbage(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None,
                 transform=None, shuffle_index=False, loop=1, use_edge=None, edge_num=None):
        super().__init__()
        self.use_edge = use_edge
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop, self.edge_num = split, voxel_size, transform, voxel_max, shuffle_index, loop, edge_num
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list]
        if split == 'train':
            self.data_list = [item for item in data_list if not 'test' in item]
        else:
            self.data_list = [item for item in data_list if 'test' in item]
        self.data_root = data_root
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item + '.ply')
                data = read_ply2np(data_path)  # xyzrgbl, N*7
                sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))

        # multithread
        # pool = Pool(20)
        # pool.map(get_edge, self.data_list)
        # pool.close()
        # pool.join()

        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()

        if self.use_edge:
            edge = SA.attach("shm://{}".format(self.data_list[data_idx] + "_edge")).copy()
            # edge_nums sample to 2000
            if edge.shape[0] >= self.edge_num:
                sample_idx = np.random.choice(len(edge), self.edge_num, replace=False)
                edge = edge[sample_idx, :]
            else:
                rand_sample_idx = np.random.choice(len(data), self.edge_num - len(edge), replace=False)
                edge = np.vstack((edge, data[rand_sample_idx]))

            data = np.vstack((data, edge))

            item = self.data_list[data_idx]
            # data_path = os.path.join(self.data_root, item + '.npy')
            # data = np.load(data_path)
            if data.shape[1] >6 :
                coord, feat, label = (data[:, 0:3] - np.mean(data[:, 0:3], axis=0)) * 10, np.zeros_like(data[:, :3]), data[:, 6]
            else:
                coord, feat, label = (data[:, 0:3] - np.mean(data[:, 0:3], axis=0)) / 100, np.zeros_like(data[:, :3]), data[:, 3]
            coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                                              self.transform, self.shuffle_index, self.edge_num)
            return coord, feat, label
        else:
            item = self.data_list[data_idx]
            # data_path = os.path.join(self.data_root, item + '.npy')
            # data = np.load(data_path)
            if data.shape[1] > 6:
                coord, feat, label = (data[:, 0:3] - np.mean(data[:, 0:3], axis=0)) *10, data[:, 3:6], data[:, 6]
            else:
                coord, feat, label = (data[:, 0:3] - np.mean(data[:, 0:3], axis=0)) / 100, np.zeros_like(data[:, :3]), data[:, 3]

            coord, feat, label = data_prepare_notedge(coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                                                      self.transform, self.shuffle_index)
            return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop


class Cabbage_v2(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None,
                 transform=None, shuffle_index=False, loop=1, use_edge=None, edge_num=None):
        super().__init__()
        self.use_edge = use_edge
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop, self.edge_num = split, voxel_size, transform, voxel_max, shuffle_index, loop, edge_num
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list]
        if split == 'train':
            self.data_list = [item for item in data_list if not 'test' in item]
        else:
            self.data_list = [item for item in data_list if 'test' in item]
        self.data_root = data_root
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item + '.ply')
                data = read_ply2np(data_path)  # xyzrgbl, N*7
                sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))

        # multithread
        # pool = Pool(20)
        # pool.map(get_edge, self.data_list)
        # pool.close()
        # pool.join()

        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()

        if self.use_edge:
            edge = SA.attach("shm://{}".format(self.data_list[data_idx] + "_edge")).copy()
            # edge_nums sample to 2000
            if edge.shape[0] >= self.edge_num:
                sample_idx = np.random.choice(len(edge), self.edge_num, replace=False)
                edge = edge[sample_idx, :]
            else:
                rand_sample_idx = np.random.choice(len(data), self.edge_num - len(edge), replace=False)
                edge = np.vstack((edge, data[rand_sample_idx]))

            data = np.vstack((data, edge))

            item = self.data_list[data_idx]
            # data_path = os.path.join(self.data_root, item + '.npy')
            # data = np.load(data_path)
            if data.shape[1] >6 :
                coord, feat, label = (data[:, 0:3] - np.mean(data[:, 0:3], axis=0)) *10, data[:, 3:6], data[:, 6]
            else:
                coord, feat, label = (data[:, 0:3] - np.mean(data[:, 0:3], axis=0)) / 100, np.zeros_like(data[:, :3]), data[:, 3]
            coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                                              self.transform, self.shuffle_index, self.edge_num)
            return coord, feat, label
        else:
            item = self.data_list[data_idx]
            # data_path = os.path.join(self.data_root, item + '.npy')
            # data = np.load(data_path)
            if data.shape[1] > 6:
                coord, feat, label = (data[:, 0:3] - np.mean(data[:, 0:3], axis=0)) *10, data[:, 3:6], data[:, 6]
            else:
                coord, feat, label = (data[:, 0:3] - np.mean(data[:, 0:3], axis=0)) / 100, np.zeros_like(data[:, :3]), data[:, 3]

            coord, feat, label = data_prepare_notedge(coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                                                      self.transform, self.shuffle_index)
            return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop


if __name__ == '__main__':
    data_root = '/home/share/Dataset/s3dis'
    test_area, voxel_size, voxel_max = 5, 0.04, 80000

    point_data = S3DIS(split='train', data_root=data_root, test_area=test_area, voxel_size=voxel_size,
                       voxel_max=voxel_max)
    print('point data size:', point_data.__len__())
    import torch, time, random

    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)


    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)


    train_loader = torch.utils.data.DataLoader(point_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
                                               collate_fn=collate_fn)
    for idx in range(1):
        end = time.time()
        voxel_num = []
        for i, (coord, feat, label, offset) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            print('tag', coord.shape, feat.shape, label.shape, offset.shape, torch.unique(label))
            voxel_num.append(label.shape[0])
            end = time.time()
    print(np.sort(np.array(voxel_num)))
