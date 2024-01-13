import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

from PIL import Image
import cv2

import torchvision.transforms as transforms

# =======================================================================================
# 对 pre-process 之后的数据集进行 argument

# 随机 flip
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)   # 这应该是随机选择 axis
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

# 随机 rotate
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

# 进行 flip、rotate、crop
# 这里按照 ParNet 中进行处理
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

        # 进行 transform
        # imag 进行 resize、ToTensor、Normalization
        self.img_transform = transforms.Compose([
            transforms.Resize((self.output_size[0], self.output_size[1])),         
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        # label 仅进行 resize、ToTensor
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.output_size[0], self.output_size[1])),
            transforms.ToTensor()])


    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # if random.random() > 0.5:
        #     image, label = random_rot_flip(image, label)   # 对 img, label 进行随机 flip
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)    # 对 image, label 进行随机 rotate
        
        # 这里按照 PraNet 多加一个 Norm
        # resize ---> ToTensor ---> Norm 
        image = self.img_transform(image)

        # resize ---> ToTensor
        label = self.gt_transform(label)

        sample = {'image': image, 'label': label.long()}  # 将处理后的 image 返回
        return sample


# =======================================================================================
# 创建 dataset
class Polyp_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split

        # 从 list_dir 文件夹下，读取对应 train、test 数据集的 txt 文件
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()  

        self.data_dir = base_dir  # 这应该是 dataset path

    def __len__(self):
        return len(self.sample_list)  # 获取样本总数

    def __getitem__(self, idx):

 
        slice_name = self.sample_list[idx].strip('\n')  # 去除字符串首尾的空格
        slice_name = slice_name.strip()   # 去除字符串首尾的空格
        slice_name = slice_name.split(' ')   # 按空格进行分割

        image_path = os.path.join(self.data_dir, slice_name[0])   # 获取 img 的绝对地址
        label_path = os.path.join(self.data_dir, slice_name[1])   # 获取 label 的绝对地址            

        # 利用 cv2 读出来是 np,而想要用 
        # cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道，读取出来为 BGR 类：H * W * 3
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)   
        # BGR 转 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        # 读取灰度图
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W


        sample = {'image': image, 'label': label}

        # 利用 pytorch 中的 transform function 进行 
        if self.transform:
            sample = self.transform(sample)

        # 将 img 名字提出出来
        case_name = slice_name[0].split('.')
        case_name = case_name[0].split('/')
        sample['case_name'] = case_name[1]  # sample 名字还是原 image 的名字

        return sample

