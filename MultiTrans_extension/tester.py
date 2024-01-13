import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import test_single_volume

from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_Polyp import Polyp_dataset

# =======================================================================
# 在 3D 数据集上的 inference
def inference_Synapse(args, model, test_save_path=None):

    # 加载数据集
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        
        # 读取 3D sequences 数据集
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        # 单个 volume 的精度
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size_width], test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        
        # test 数据集总精度
        metric_list += np.array(metric_i)

        # 打印单个 volume 的 the average DSC 和 average Hausdorff Distance (HD) 精度
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    # 总精度
    metric_list = metric_list / len(db_test)

    # 打印每个类别的精度，第 0  类为背景，所以不显示精度
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"

# =======================================================================
from utils import test_RGB_image
# 在 2D 数据集上的 inference
def inference_Polyp(args, model, split=None, test_save_path=None):

    # 加载数据集
    db_test = Polyp_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split=split, imag_size=args.img_size)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):

        # 读取 2D 数据集
        image, label, case_name = sampled_batch
        h, w = image.size()[2:]

        # 单个 volume 的精度
        metric_i = test_RGB_image(args, image, label, model, classes=args.num_classes, test_save_path=test_save_path, case=case_name)
        
        # test 数据集总精度
        metric_list += np.array(metric_i)

        # # 打印单个 image 的精度
        # logging.info('idx %d case %s mean_dice %f mean_hd95 %f mean_IoU %f mean_dice %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))

    # 总精度
    metric_list = metric_list / len(db_test)

    # # 打印每个类别的精度，第 0  类为背景，所以不显示精度
    # for i in range(1, args.num_classes):
    #     logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    mean_dice1 = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mIoU = np.mean(metric_list, axis=0)[2]
    # mean_dice2 = np.mean(metric_list, axis=0)[3]

    logging.info('Testing performance in best val model: mean_dice: %f mean_hd95: %f mean_IoU: %f' % (mean_dice1, mean_hd95, mIoU))
    return "Testing Finished!"
