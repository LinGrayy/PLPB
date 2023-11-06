from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from glob import glob
import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as Ft
import imgaug.augmenters as iaa
import dataloaders.net as net
from dataloaders.utils import *
from dataloaders.mypath import MYPath
import cv2
from torch.utils.data import DataLoader
import torch.nn as nn

class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir=MYPath.db_root_dir('fundus'),
                 dataset='refuge',
                 split='train',
                 testid=None,
                 transform=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.split = split

        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []

        self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
        print(self._image_dir)
        imagelist = glob(self._image_dir + "/*.png") #png
        for image_path in imagelist:
            gt_path = image_path.replace('image/', 'mask/') #
            #gt_path = gt_path.replace('.tif', '-1.tif') #RIGA
            self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})

        self.transform = transform
        # self._read_img_into_memory()
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _target = Image.open(self.image_list[index]['label'])
        if _target.mode is 'RGB':
            _target = _target.convert('L')
        _img_name = self.image_list[index]['image'].split('/')[-1]

        # _img = self.image_pool[index]
        # _target = self.label_pool[index]
        # _img_name = self.img_name_pool[index]
        anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'image1': _img}

        if self.transform is not None:
            anco_sample = self.transform(anco_sample)

        return anco_sample

    def _read_img_into_memory(self):

        img_num = len(self.image_list)
        for index in range(img_num):
            self.image_pool.append(Image.open(self.image_list[index]['image']).convert('RGB'))
            _target = Image.open(self.image_list[index]['label'])
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)


    def __str__(self):
        return 'Fundus(split=' + str(self.split) + ')'

# load adversarial samples
class FundusSegmentation_pgdtest(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir=MYPath.db_root_dir('fundus'),
                 dataset='refuge',
                 split='train',
                 testid=None,
                 transform=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.split = split

        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []

        self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
        print(self._image_dir+'/pgd')
        imagelist = glob(self._image_dir + "/*.png")
        
        for image_path in imagelist:
            gt_path = image_path.replace('image', 'mask')
            p1_path = image_path.replace('Domain1/test/ROIs/image/', 'PGD/DPL/Domain1/')
            # p1_path = image_path.replace('Domain1/test/ROIs/', 'PGD/OURS/Domain1/test/')
            #print(p1_path)
            self.image_list.append({'image': p1_path, 'label': gt_path, 'id': testid}) 


        self.transform = transform
        
        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _target = Image.open(self.image_list[index]['label'])
        if _target.mode is 'RGB':
            _target = _target.convert('L')
        _img_name = self.image_list[index]['image'].split('/')[-1]

        
        anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name}

        if self.transform is not None:
            anco_sample = self.transform(anco_sample)

        return anco_sample


    def __str__(self):
        return 'Fundus(split=' + str(self.split) + ')'



if __name__ == '__main__':
    data_dir = '/mnt/data1/llr_data/Cell'
    dataset = 'Domain3'
    cell_dataset = CellSegmentation(base_dir=data_dir, dataset=dataset, split='test/')

    domain_loader = DataLoader(cell_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    for batch_idx, (sample) in enumerate(domain_loader):
            data, img_name = sample['image'], sample['img_name']
            print(img_name)

    
    