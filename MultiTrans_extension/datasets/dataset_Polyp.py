import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

from scipy.ndimage.interpolation import zoom

import cv2

# 进行 flip、rotate、crop
class Resize(object):
    def __init__(self, size, split):
        self.size = size
        self.split = split

    def __call__(self, input):
        
        if self.split == 'image':

            x, y, z = input.shape # 这是个 RGB  

            # 将 image resize 到 output_size 的大小
            if x != self.size[0] or y != self.size[1]:
                image = cv2.resize(input, self.size[::-1], interpolation=cv2.INTER_LINEAR)    # 对 img 进行双线性插值
            
            return image

        elif self.split == 'label':

            x, y = input.shape # 这是个灰度 

            # 将 image resize 到 output_size 的大小
            if x != self.size[0] or y != self.size[1]:         
                label = cv2.resize(input, self.size[::-1], interpolation=cv2.INTER_NEAREST)   # 对 label 进行最邻近插值
            
            return label


class Polyp_dataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, base_dir, list_dir, split, imag_size, transform=None):

        self.transform = transform  # using transform in torch!
        self.split = split

        # 从 list_dir 文件夹下，读取对应 train、test 数据集的 txt 文件
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()  

        self.data_dir = base_dir  # 这应该是 dataset path

        self.imag_size = imag_size

        # 进行 transform
        if self.split == 'train':        
            # imag 进行 resize、ToTensor、Normalization
            self.img_transform = transforms.Compose([
                Resize((self.imag_size, self.imag_size), 'image'),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])

            # label 仅进行 resize、ToTensor
            self.gt_transform = transforms.Compose([
                Resize((self.imag_size, self.imag_size), 'label'),
                transforms.ToTensor()])
        else:
            self.img_transform = transforms.Compose([
                Resize((self.imag_size, self.imag_size), 'image'),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.sample_list)  # 获取样本总数

    def __getitem__(self, idx):

        slice_name = self.sample_list[idx].strip('\n')  # 去除字符串首尾的空格
        slice_name = slice_name.strip()   # 去除字符串首尾的空格
        slice_name = slice_name.split(' ')   # 按空格进行分割

        image_path = os.path.join(self.data_dir, slice_name[0])   # 获取 img 的绝对地址
        label_path = os.path.join(self.data_dir, slice_name[1])   # 获取 label 的绝对地址    

        # 将 img 名字提出出来
        case_name = slice_name[0].split('.')
        case_name = case_name[0].split('/')
        name = case_name[1]  # sample 名字还是原 image 的名字       

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)   
        # BGR 转 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)

        # 读取灰度图
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        if self.split == 'train':  
        
            image = self.img_transform(image)

            label = label/255.0  # 因为 Polyp 数据集的 label 为 0：背景、255：目标。所以要除以 255.0。

            label = self.gt_transform(label)
            
            # size = (self.imag_size, self.imag_size)
            # label = cv2.resize(label, size[::-1], interpolation=cv2.INTER_NEAREST)   # 对 label 进行最邻近插值
            
        else:
            image = self.img_transform(image)
                
            label = label/255.0   # 因为 Polyp 数据集的 label 为 0：背景、255：目标。所以要除以 255.0。
            
        return image, label, name



