from torch.utils.data import Dataset
from torchvision import transforms
import os
from torch.utils.data import DataLoader
import torch
import torchvision.transforms.functional as F
import scipy.ndimage
import random
from PIL import Image
import numpy as np
import cv2
from skimage import transform as tf
import numbers
from torchvision.transforms import InterpolationMode

class ToTensor(object):

    def __call__(self, data):
        image, label, mask = data['image'], data['label'], data['mask']
        mask = np.array(mask)[:, :,:3].mean(-1)
        mask = self.discretize_mask(mask)
        mask = torch.LongTensor(mask.astype('int32'))
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label), 'mask': mask}
    
    def discretize_mask(self, mask):
        # 将接近 85 的像素值归类为 1，接近 255 的像素值归类为 2，其余为 0
        mask_discrete = np.zeros_like(mask)
        mask_discrete[np.abs(mask - 85) < 42] = 1  # 85 附近的像素值归为 1
        mask_discrete[np.abs(mask - 255) < 127] = 2  # 255 附近的像素值归为 2
        return mask_discrete


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label, mask = data['image'], data['label'], data['mask']

        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC), 'mask': F.resize(mask, self.size, interpolation=InterpolationMode.BICUBIC)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label, mask = data['image'], data['label'], data['mask']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label), 'mask': F.hflip(mask)}

        return {'image': image, 'label': label, 'mask': mask}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label, mask = data['image'], data['label'], data['mask']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label), 'mask': F.vflip(mask)}

        return {'image': image, 'label': label, 'mask': mask}




class Normalize(object):
    def __init__(self):
        self.norm_min = -1
        self.norm_max = 1
        self.norm_range = self.norm_max - self.norm_min
        self.clip = True
        self.min_quantile = 0.02
        self.max_quantile = 1 -  self.min_quantile

    def __call__(self, sample):
        image, label, mask = sample['image'], sample['label'], sample['mask']
        
        image = image * 2 - 1
        label = label * 2 - 1
        return {'image': image, 'label': label, 'mask': mask}


class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        self.image_root = image_root
        self.gt_root = gt_root
        self.size = size
        self.mode = mode
        # 获取所有图像文件名
        self.image_files = sorted([f for f in os.listdir(image_root) if f.endswith('.jpg')])

        #self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        #self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        #self.images = sorted(self.images)
        #self.gts = sorted(self.gts)
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_root, self.image_files[idx])
        gt_path = os.path.join(self.gt_root, self.image_files[idx].replace('.jpg', '_mask.png'))
        image = self.rgb_loader(image_path)
        label = self.binary_loader(gt_path)
        mask = self.mask_loader(gt_path)
        
        data = {'image': image, 'label': label, 'mask': mask}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.image_files)

    def rgb_loader(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def binary_loader(self, path):
        img = Image.open(path).convert("L")
        return img
    
    def mask_loader(self, path):
        img = Image.open(path)
        return img
        
# loader_generator = torch.Generator().manual_seed(2015)
# image_path = '/home/wenxue/Data/Transparent/Trans10Kv1_v2/train/images/'
# mask_path = '/home/wenxue/Data/Transparent/Trans10Kv1_v2/train/masks/'
# dataset = FullDataset(image_path, mask_path, 512, mode='train')
# train_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=2, shuffle=True, generator=loader_generator,)


# for batch in train_loader:
#     image, mask_true, label = batch['image'], batch['label'], batch['mask']
#     print(label.max())
