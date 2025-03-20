import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class Plant_part(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.catfile = os.path.join(self.data_root, 'class_to_number.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.plant_class = dict(zip(self.cat, range(len(self.cat))))
        self.loop = config.loop

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        if self.subset =="train":
            self.sample_points_num = config.npoints
        else:
            self.sample_points_num = config.npoints

        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='PlantData')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='PlantData')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='PlantData')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })

        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='PlantData')
        print_log(f'[DATASET] {len(self.file_list)*self.loop} instances were trained', logger='PlantData')

        self.permutation = np.arange(self.npoints)
        self.data_idx = np.arange(len(self.file_list))

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx%len(self.data_idx)]
        if self.subset !="train":
            label = self.plant_class[str(sample['taxonomy_id'])]
            label = np.array([label]).astype(np.int32)

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        # data = np.load(os.path.join(self.pc_path, sample['file_path']),allow_pickle=True).astype(np.float32)
        #data = self.random_sample(data, self.sample_points_num)
        sample_idx = np.random.choice(len(data), self.sample_points_num, replace=False)
        np.random.shuffle(sample_idx)
        data= data[sample_idx, :3]
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        if self.subset !="train":
            return sample['taxonomy_id'], sample['model_id'], (data, label[0])
        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)* self.loop