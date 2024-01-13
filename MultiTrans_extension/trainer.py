import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import torch.nn.functional as F

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# ==============================================================================
# 读取 3D 分出来的 slides
# synapse 数据集
def trainer_synapse(args, model, Log_path, TensorboardX_path, Model_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=Log_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size_width])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        # print('使用多 GPU')
        # model = nn.DataParallel(model, device_ids=[0,1])
        model = nn.DataParallel(model)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # 这里换用 Trans 最常用的 Adam。暂时按照 MMSeg 中 TopFormer 的参数进行设置
    # optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01)

    writer = SummaryWriter(TensorboardX_path)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # print(image_batch.size())
            # print(label_batch.size())

            outputs = model(image_batch)

            if args.If_Deep_Supervision:

                loss_ce = ce_loss(outputs[0], label_batch[:].long())
                loss_dice = dice_loss(outputs[0], label_batch, softmax=True)
                main_loss = 0.5 * loss_ce + 0.5 * loss_dice

                # weights = np.array([1 / (2 ** i) for i in range(len(outputs[1]))])
                # weights = weights / weights.sum()

                loss = main_loss
                branches_out = outputs[1]
                for i in range(len(branches_out)):
                    aux_loss = 0.5 * ce_loss(branches_out[i], label_batch[:].long()) + 0.5 * dice_loss(branches_out[i], label_batch, softmax=True)
                    loss += args.bran_weights[i]*aux_loss

                outputs = outputs[0]  # 由于要进行 val 所以这里要加个这个

            else:

                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(Model_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(Model_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

# ==============================================================================
from tester import inference_Synapse, inference_Polyp

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

# 读取 2D RGB images
# 输入：hyperparameters，实例化模型，结果保存路径
def trainer_Polyp(args, model, Log_path, TensorboardX_path, Model_path):
    
    # 加载 Synapse 数据集加载器、augment和crop函数
    # 注意！！！ 换数据集的话，这里要改
    from datasets.dataset_Polyp import Polyp_dataset  

    #--------------------------------------------------
    # 生成日志文件
    logging.basicConfig(filename=Log_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    # 加载 train 数据集，并进行 augment、crop
    db_train = Polyp_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", imag_size=args.img_size)
    print("The length of train set is: {}".format(len(db_train)))
    
    # 设置 seed
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # 加载数据集
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # loss function：ce_loss + dice_loss
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    
    if args.optimizer == 'SGD_Poly':
        # 优化函数：SGD
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    elif args.optimizer == 'Adam_Clip_grad_norm':
        # 这里换用 Trans 最常用的 Adam。暂时按照 MMSeg 中 TopFormer 的参数进行设置
        # optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01)
        # optimizer = torch.optim.Adam(model.parameters(), base_lr, betas=(0.5, 0.999))
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    elif args.optimizer == 'Adam_Poly':
        # 这里换用 Trans 最常用的 Adam。暂时按照 MMSeg 中 TopFormer 的参数进行设置
        # optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01)
        # optimizer = torch.optim.Adam(model.parameters(), base_lr, betas=(0.5, 0.999))
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)


    writer = SummaryWriter(TensorboardX_path)
    iter_num = 0
    max_epoch = args.max_epochs
    # len(trainloader) = len(dataset) / batch size 
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    # ---- multi-scale training ----
    if args.If_Multiscale_Train:
        size_rates = [0.75, 1, 1.25]
    else:
        size_rates = [1]
        
    # 按 epoch 
    for epoch_num in iterator:
        # len(trainloader) = len(dataset) / batch size
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):

            # ---- multi-scale training ----
            for rate in size_rates:
                image_batch, label_batch, name = sampled_batch
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
         
                # ---- rescale ----
                trainsize = int(round(args.img_size*rate/32)*32)
                if rate != 1:
                    # F.upsample 需要 4D input
                    image_batch = F.upsample(image_batch, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    label_batch = F.upsample(label_batch, size=(trainsize, trainsize), mode='bilinear', align_corners=True)         

                outputs = model(image_batch)

                if args.loss_name == 'structure_loss':
                    # output = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    # print(output.shape[0])
                    # print(output.shape[1])
                    # outputs = torch.tensor(outputs, dtype=torch.float32)
                    # label_batch = torch.tensor(label_batch, dtype=torch.float32)
                    if args.If_Deep_Supervision:
                        main_loss = structure_loss(outputs[0], label_batch)

                        branches_out = outputs[1]
                        # for i in range(len(branches_out)):
                        #     loss = loss + structure_loss(branches_out[i], label_batch)

                        branch1 = structure_loss(branches_out[0], label_batch)
                        branch2 = structure_loss(branches_out[1], label_batch)
                        branch3 = structure_loss(branches_out[2], label_batch)
                        branch4 = structure_loss(branches_out[3], label_batch)

                        loss = main_loss + args.bran_weights[0]*branch1 + args.bran_weights[1]*branch2 + args.bran_weights[2]*branch3 + args.bran_weights[3]*branch4

                        loss_ce = loss_dice = loss

                    else:
                        loss = structure_loss(outputs, label_batch)
                        loss_ce = loss_dice = loss

                elif args.loss_name == 'ce_dice_loss': 

                    if args.If_Deep_Supervision:
                        label_batch = label_batch.squeeze(1)   # 用了 transforms.ToTensor() 后再用 DataLoader 会自动给 label 加一个维度，得去掉       
                        # -----------------------------------------------------
                        # 计算 loss function
                        main_loss = 0.5 *ce_loss(outputs[0], label_batch[:].long()) + 0.5 *dice_loss(outputs[0], label_batch, softmax=True)
                        
                        branches_out = outputs[1]
                        branch1 = 0.5 *ce_loss(branches_out[0], label_batch[:].long()) + 0.5 *dice_loss(branches_out[0], label_batch, softmax=True)
                        branch2 = 0.5 *ce_loss(branches_out[1], label_batch[:].long()) + 0.5 *dice_loss(branches_out[1], label_batch, softmax=True)
                        branch3 = 0.5 *ce_loss(branches_out[2], label_batch[:].long()) + 0.5 *dice_loss(branches_out[2], label_batch, softmax=True)
                        branch4 = 0.5 *ce_loss(branches_out[3], label_batch[:].long()) + 0.5 *dice_loss(branches_out[3], label_batch, softmax=True)

                        loss = main_loss + args.bran_weights[0]*branch1 + args.bran_weights[1]*branch2 + args.bran_weights[2]*branch3 + args.bran_weights[3]*branch4

                        loss_ce = loss_dice = loss

                    else:
                        label_batch = label_batch.squeeze(1)   # 用了 transforms.ToTensor() 后再用 DataLoader 会自动给 label 加一个维度，得去掉       
                        # -----------------------------------------------------
                        # 计算 loss function
                        loss_ce = ce_loss(outputs, label_batch[:].long())
                        loss_dice = dice_loss(outputs, label_batch, softmax=True)
                        loss = 0.5 * loss_ce + 0.5 * loss_dice

                if args.optimizer == 'SGD_Poly':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # -----------------------------------------------------
                    # 使用 poly loss 更新 loss
                    lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_

                elif args.optimizer == 'Adam_Clip_grad_norm':
                    decay_rate = 0.1
                    decay_epoch = 50
                    epoch = epoch_num + 1
                    decay = decay_rate ** (epoch // decay_epoch)  # 这是参照 ParNet
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= decay
                    lr_ = param_group['lr']

                    optimizer.zero_grad()
                    loss.backward()
                    # 使用 ParNet 中的 clip_grad_norm
                    # 对于梯度爆炸问题，解决方法之一便是进行梯度剪裁，即设置一个梯度大小的上限。
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            if param.grad is not None:
                                param.grad.data.clamp_(-args.grad_clip, args.grad_clip)
                    
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                elif args.optimizer == 'Adam_Poly':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # -----------------------------------------------------
                    # 使用 poly loss 更新 loss
                    lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_

                if rate == 1:
                    iter_num = iter_num + 1

                if rate == 1:
                    # Tensorboard
                    writer.add_scalar('info/lr', lr_, iter_num)
                    writer.add_scalar('info/total_loss', loss, iter_num)
                    writer.add_scalar('info/loss_ce', loss_ce, iter_num)

                    logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

                    # 这是保存可视化图像？
                    # if iter_num % 20 == 0:
                    #     image = image_batch[1, 0:1, :, :]
                    #     image = (image - image.min()) / (image.max() - image.min())  # 归一化到 0~1 之间
                    #     writer.add_image('train/Image', image, iter_num)
                    #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True) # 返回 dim 维度，最大值的 indices

                    #     # 这里 [1, ...] 是什么？然后 *50 又是干啥？
                    #     writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num) 
                    #     labs = label_batch[1, ...].unsqueeze(0) * 50
                    #     writer.add_image('train/GroundTruth', labs, iter_num)
        

        # 保存断点
        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(Model_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # 保存最后一个 epoch 模型
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(Model_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
        
        args.volume_path = '/home/zhangyanhua/Code_python/Dataset/Medical_Dataset/Polyp/TestDataset'
        inference_Polyp(args, model, split='CVC-ClinicDB_test', test_save_path=None)
        inference_Polyp(args, model, split='Kvasir_test', test_save_path=None)
        inference_Polyp(args, model, split='CVC-ColonDB_test', test_save_path=None)
        inference_Polyp(args, model, split='ETIS-LaribPolypDB_test', test_save_path=None)
        inference_Polyp(args, model, split='CVC-300_test', test_save_path=None)

        # 保存最后一个 epoch 模型
        if epoch_num >= max_epoch - 1:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

