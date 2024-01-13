import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np


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
                transforms.Resize((self.imag_size, self.imag_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])

            # label 仅进行 resize、ToTensor
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.imag_size, self.imag_size)),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.imag_size, self.imag_size)),
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

        if self.split == 'train':  
            # 读取为 PIL 图像，这样方便使用 torchvision.transforms 进行转换
            image = self.rgb_loader(image_path)
            image = self.img_transform(image)

            gt = self.binary_loader(label_path)  # 读出来为 PIL 还不能直接 /255.0

            gt = self.gt_transform(gt)
            gt = gt/255.0  # 因为 Polyp 数据集的 label 为 0：背景、255：目标。所以要除以 255.0。
            
        else:
            image = self.rgb_loader(image_path)
            image = self.transform(image)
            # # test 时，这里为啥要加 1 维。Train 时没有。因为：test 时候，作者没有使用 DataLoader
            # image = self.transform(image).unsqueeze(0)    
                 
            gt = self.binary_loader(label_path)  # 这还是 PIL image。 还不能直接 /255.0

            gt = np.array(gt)   # PIL 转 np
            gt = gt/255.0   # 因为 Polyp 数据集的 label 为 0：背景、255：目标。所以要除以 255.0。
            
        
        return image, gt, name

    # 利用 PIL 中的 Image 读取 rgb 图像
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)      # open 只是打开函数，不读入内存。
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')



