import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

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
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)   # 对 img, label 进行随机 flip
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)    # 对 image, label 进行随机 rotate
        
        x, y = image.shape # 这是个 2D slices。而且是单通道  

        # 将 image resize 到 output_size 的大小
        if x != self.output_size[0] or y != self.output_size[1]:
            # order: The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # image (应该是 numpy format) ---> np.float32 ---> torch Tensor ---> 插入一个 dimension：估计是为了加入 batch
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # x, y ---> 1, x, y
        label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label.long()}  # 将处理后的 image 返回
        return sample

# =======================================================================================
# 创建 dataset
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()  # 从 txt 文件中读取 samples 的文件名列表
        self.data_dir = base_dir  # 这应该是 dataset 所在目录

    def __len__(self):
        return len(self.sample_list)  # 获取样本总数

    def __getitem__(self, idx):

        # 读取 train dataset 的 img、label。是 .npz 格式
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')  # 获得切片名字
            data_path = os.path.join(self.data_dir, slice_name+'.npz')

            data = np.load(data_path)  # 读取 .npz 文件
            image, label = data['image'], data['label'] # 从 .npz 文件里提取单个 slice 的 img 和 label

        else:
        # 读取 val dataset 的 3D img、label。是 .npy.h5 格式。
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name) # 获取文件名

            data = h5py.File(filepath)  # 读取 .npy.h5 文件

            image, label = data['image'][:], data['label'][:]  # 获取 slice sequences

        sample = {'image': image, 'label': label}

        # 利用 pytorch 中的 transform function 进行 
        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = self.sample_list[idx].strip('\n')  # sample 名字还是原 image 的名字

        return sample
