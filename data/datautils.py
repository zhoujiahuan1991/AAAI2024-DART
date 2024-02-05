import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

import albumentations as A

from torchvision.datasets import CIFAR100

ID_to_DIRNAME={
    'I': 'imagenet-val',
    'A': 'imagenet-a',
    'K': 'imagenet-sketch/sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'C': 'imagenet-c',
    'T': 'tiny-imagenet-200/val-I',
    'Cifar100': 'Cifar100',
    'flower102': '102flowers',
    'dtd': 'DTD',
    'pets': 'OxfordPets/images',
    'cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101/101_ObjectCategories',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft/data',
    'eurosat': 'EuroSAT'
}

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False, domain_id='brightness'):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]))
        # testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
        

    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    else:
        raise NotImplementedError
        
    return testset

# AugMix Transforms
def get_preaugment(args):
    if args.aug_type == 'default':
        if args.resize_flag is True:
            rnt =  transforms.Compose([
                # transforms.Resize(args.resize),
                transforms.RandomCrop((args.resolution, args.resolution)),
                transforms.RandomHorizontalFlip(),
            ])
        else:   
            rnt =  transforms.Compose([
                transforms.transforms.RandomResizedCrop((args.resolution, args.resolution)),
                transforms.RandomHorizontalFlip(),
            ])
    return rnt

def augmix(image, preprocess, aug_list, severity=1, args=None):
    """对原始数据进行数据增强，返回增强后的图片

    Args:
        image (_type_): 预处理后的图片
        preprocess (_type_): 预处理的transform
        aug_list (_type_): 如果没有自己指定，则为空列表
        severity (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """    
    preaugment = get_preaugment(args)
    # print(type(image))
    # print(image.shape)
    if args.aug_type == 'default':
        x_orig = preaugment(image)
    
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1, args=None):
        """生成数据增强变换的transform

        Args:
            base_transform (_type_): 基transform，统一图片的大小
            preprocess (_type_): 数据预处理的transform，将数据转换为tensor
            n_views (int, optional): _description_. Defaults to 2.
            augmix (bool, optional): 是否另外指定数据增强的方式. Defaults to False.
            severity (int, optional): _description_. Defaults to 1.
        """
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        self.args = args
        
    def __call__(self, x):
        """_summary_

        Args:
            x (_type_): 图片输入

        Returns:
            _type_: 原图和增强后的图片
        """
        # image预处理
        image = self.preprocess(self.base_transform(x))
        if self.args.resize_flag is True:
            tran = transforms.Resize(self.args.resize)
            x = tran(x)
        # 生成增强后的图片
        views = [augmix(x, self.preprocess, self.aug_list, self.severity, self.args) for _ in range(self.n_views)]
        # 返回原图和增强后的图片
        return [image] + views


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, clusters, val_dataset):
        super().__init__()
        self.val_dataset = val_dataset
        self.clusters = torch.IntTensor(clusters).reshape(-1)
        sorted_, self.indices = torch.sort(self.clusters)
        # print(self.val_dataset[0])
        print(len(self.val_dataset[0]))
        print(len(self.val_dataset[0][0]))
        print(self.val_dataset[0][0][0].shape)
        # input()
        # sorted_, self.indices = torch.sort(self.clusters[0:100])
        # print(clusters[0:100])
        # print(clusters.shape)
        # print(sorted_[0:100])
        # print(indices[0:100])
        # input()
    
    def __len__(self):
        return self.val_dataset.__len__()
    
    def __getitem__(self, index):
        idx = self.indices[index].tolist()
        # print(idx)
        # input()
        return self.val_dataset[idx]






