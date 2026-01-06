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
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label), 'mask': F.to_tensor(mask)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label, mask = data['image'], data['label'], data['mask']

        return {'image': F.resize(image, self.size), 
                'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC),
                'mask': F.resize(mask, self.size, interpolation=InterpolationMode.BICUBIC)}


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
        
        self.image_path_trans10k = '/home/wenxue/Data/Transparent/Trans10k_stuff/train/image/'
        self.mask_path_trans10k = '/home/wenxue/Data/Transparent/Trans10k_stuff/train/mask/'
        self.image_path_gdd = '/home/wenxue/Data/Transparent/GDD/train/image/'
        self.mask_path_gdd = '/home/wenxue/Data/Transparent/GDD/train/mask/'
        self.image_path_gsd = '/home/wenxue/Data/Transparent/GSD/train/image/'
        self.mask_path_gsd = '/home/wenxue/Data/Transparent/GSD/train/mask/'
        self.image_path_hso = '/home/wenxue/Data/Transparent/HSO/train/image/'
        self.mask_path_hso = '/home/wenxue/Data/Transparent/HSO/train/mask/'
        
        
        self.image_trans10k = sorted([f for f in os.listdir(self.image_path_trans10k) if f.endswith('.jpg')])
        self.mask_trans10k = sorted([f for f in os.listdir(self.mask_path_trans10k) if f.endswith('.png')])
        
        self.image_gdd = sorted([f for f in os.listdir(self.image_path_gdd) if f.endswith('.jpg')])
        self.mask_gdd = sorted([f for f in os.listdir(self.mask_path_gdd) if f.endswith('.png')])

        self.image_gsd = sorted([f for f in os.listdir(self.image_path_gsd) if f.endswith('.jpg')])
        self.mask_gsd = sorted([f for f in os.listdir(self.mask_path_gsd) if f.endswith('.png')])
        
        self.image_hso = sorted([f for f in os.listdir(self.image_path_hso) if f.endswith('.jpg')])
        self.mask_hso = sorted([f for f in os.listdir(self.mask_path_hso) if f.endswith('.png')])


        self.image_list = []
        self.gt_list = []

        #trans10k
        image_root = self.image_path_trans10k
        gt_root = self.mask_path_trans10k
        for i in self.image_trans10k:
            self.image_list.append(os.path.join(image_root, i))
            self.gt_list.append(os.path.join(gt_root, i.replace('.jpg', '.png')))
            
        #trans10k
        image_root = self.image_path_gdd
        gt_root = self.mask_path_gdd
        for i in self.image_gdd:
            self.image_list.append(os.path.join(image_root, i))
            self.gt_list.append(os.path.join(gt_root, i.replace('.jpg', '.png')))
            
        #trans10k
        image_root = self.image_path_gsd
        gt_root = self.mask_path_gsd
        for i in self.image_gsd:
            self.image_list.append(os.path.join(image_root, i))
            self.gt_list.append(os.path.join(gt_root, i.replace('.jpg', '.png')))
            
        #trans10k
        image_root = self.image_path_hso
        gt_root = self.mask_path_hso
        for i in self.image_hso:
            self.image_list.append(os.path.join(image_root, i))
            self.gt_list.append(os.path.join(gt_root, i.replace('.jpg', '.png')))


        
        self.size = size
        self.mode = mode
        # 获取所有图像文件名

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
        image_path = self.image_list[idx]
        gt_path = self.gt_list[idx]
        image = self.rgb_loader(image_path)
        label = self.binary_loader(gt_path)
        mask = label
        data = {'image': image, 'label': label, 'mask': mask}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.image_list)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        
        
class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tiff')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            #transforms.ToTensor()
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image)
        image = np.asarray(image)
        image = np.transpose(image, (2, 0, 1)).astype(int)
        image = torch.from_numpy(image).int()

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = self.images[self.index].split('/')[-1]

        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            #img = np.asarray(img)
            return img

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

if __name__ == "__main__":
    loader_generator = torch.Generator().manual_seed(2015)
    image_path = '/home/wenxue/Data/Transparent/GDD/train/image/'
    mask_path = '/home/wenxue/Data/Transparent/GDD/train/mask/'
    dataset = FullDataset(image_path, mask_path, 352, mode='train')
    train_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=2, shuffle=True, generator=loader_generator,)
    print(len(train_loader))

    # for batch in train_loader:
    #     image, label, mask = batch['image'], batch['label'], batch['mask']
    #     print(label.min())
