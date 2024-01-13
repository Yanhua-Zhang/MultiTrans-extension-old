import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


#==========================================================================
parser = argparse.ArgumentParser()

# ---------------------------------------------------------------------------
# 模型名字、数据集名字
# My_Model
parser.add_argument('--Model_Name', type=str,
                    default='My_Model', help='experiment_name')

# Polyp
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')

# parser.add_argument('--use_dilation',type=bool, default=False, help='use_dilation')  # 注意 bool 型的参数传入没有意义
# parser.add_argument('--If_direct_sum_fusion',type=bool, default=True, help='If_direct_sum_fusion')
parser.add_argument('--backbone', type=str, default='resnet50_Deep', help='experiment_name')
parser.add_argument('--use_dilation',type=str, default='False', help='use_dilation')  
parser.add_argument('--If_pretrained',type=str, default='True', help='If_pretrained')
parser.add_argument('--If_Deep_Supervision',type=str, default='True', help='If_Deep_Supervision')
parser.add_argument('--If_in_deep_sup',type=str, default='True', help='If_in_deep_sup')

parser.add_argument('--bran_weights', nargs='+', type=float, help='bran_weights')
parser.add_argument('--If_use_UNet_fusion_stage_features',type=str, default='True', help='If_use_UNet_fusion_stage_features')  
parser.add_argument('--If_weight_init',type=str, default='False', help='If_weight_init')


parser.add_argument('--branch_key_channels', nargs='+', type=int, help='branch_key_channels')
parser.add_argument('--branch_in_channels', nargs='+', type=int, help='branch_in_channels')
parser.add_argument('--branch_out_channels', type=int, help='branch_out_channels')
parser.add_argument('--branch_choose', nargs='+', type=int, help='branch_choose')

parser.add_argument('--branch_depths', nargs='+', type=int, help='branch_depths')
parser.add_argument('--branch_num_heads', nargs='+', type=int, help='branch_num_heads')


# parser.add_argument('--Dropout_Rate_CNN', type=float, default=0.2, help='Dropout_Rate_CNN')
parser.add_argument('--Dropout_Rate_CNN', nargs='+', type=float, help='Dropout_Rate_CNN')
parser.add_argument('--Dropout_Rate_Trans', type=float, default=0, help='Dropout_Rate_Trans')
parser.add_argument('--Drop_path_rate_Trans', nargs='+', type=float, help='Drop_path_rate_Trans')
parser.add_argument('--Dropout_Rate_SegHead', type=float, default=0.1, help='Dropout_Rate_SegHead')
# parser.add_argument('--Dropout_Rate_Local_Global_Fusion', type=float, default=0, help='Dropout_Rate_Local_Global_Fusion')
parser.add_argument('--Dropout_Rate_Local_Global_Fusion', nargs='+', type=float, help='Dropout_Rate_Local_Global_Fusion')
parser.add_argument('--Dropout_Rate_Multi_branch_fusion', type=float, default=0.1, help='Dropout_Rate_Multi_branch_fusion')
parser.add_argument('--If_backbone_use_Stoch_Depth',type=str, default='False', help='If_backbone_use_Stoch_Depth')
parser.add_argument('--Dropout_Rate_UNet', nargs='+', type=float, help='Dropout_Rate_UNet')
parser.add_argument('--Dropout_Rate_Pos', nargs='+', type=float, help='Dropout_Rate_Pos')


parser.add_argument('--one_kv_head',type=str, default='True', help='one_kv_head')
parser.add_argument('--share_kv',type=str, default='True', help='share_kv')
parser.add_argument('--If_efficient_attention',type=str, default='True', help='If_efficient_attention')
parser.add_argument('--If_attention_scale',type=str, default='False', help='If_attention_scale')

parser.add_argument('--If_use_position_embedding',type=str, default='True', help='If_use_position_embedding')
parser.add_argument('--name_position_method', type=str, default='Sinusoid', help='name_position_method')
parser.add_argument('--If_out_side',type=str, default='True', help='If_out_side')


parser.add_argument('--If_Local_GLobal_Fuison',type=str, default='True', help='If_Local_GLobal_Fuison')
parser.add_argument('--Local_Global_fusion_method',type=str, default='Attention_Gate', help='Local_Global_fusion_method')


parser.add_argument('--If_use_UNet_decoder', type=str, default='False', help='If_use_UNet_decoder')
parser.add_argument('--is_deconv', type=str, default='False', help='is_deconv')
parser.add_argument('--if_sum_fusion', type=str, default='True', help='if_sum_fusion')

parser.add_argument('--Multi_branch_concat_fusion',type=str, default='False', help='Multi_branch_concat_fusion')


parser.add_argument('--If_remove_Norm', type=str, default='False', help='If_remove_Norm')
parser.add_argument('--If_remove_ReLU', type=str, default='False', help='If_remove_ReLU')

# ---------------------------------------------------------------------------
# 专门为 TransFuse 加入的控制参数：
parser.add_argument('--Scale_Choose', type=str, default='Scale_L', help='Scale_Choose')

# ---------------------------------------------------------------------------
# 换用不同的 optimizer：
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer')
parser.add_argument('--momentum', type=float,  default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,  default=0.0001, help='weight_decay')


parser.add_argument('--grad_clip', type=float, default=0.5, help='gradient clipping norm')

parser.add_argument('--loss_name', type=str, default='ce_dice_loss', help='loss function')
parser.add_argument('--If_binary_prediction',type=str, default='False', help='If_binary_prediction')

parser.add_argument('--If_Multiscale_Train',type=str, default='True', help='If_Multiscale_Train')

# ---------------------------------------------------------------------------
# 模型训练的超参数
# 352
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')

parser.add_argument('--img_size_width', type=int,
                    default=224, help='input patch size of network input')

parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')  # 这的话 Synapse 总 Iteration 为 13.7K 与原文一致 
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')

parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')

parser.add_argument('--base_lr', type=float,  default=0.1,
                    help='segmentation network learning rate')

# -----------------
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')    # 这个 deterministic training 是啥原理？ 

parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')

#==========================================================================
args = parser.parse_args()

# -----------------------------------------------------------
if args.use_dilation == 'False':
    args.use_dilation = False
else:
    args.use_dilation = True

if args.one_kv_head == 'False':
    args.one_kv_head = False
else:
    args.one_kv_head = True

if args.share_kv == 'False':
    args.share_kv = False
else:
    args.share_kv = True

if args.If_efficient_attention == 'False':
    args.If_efficient_attention = False
else:
    args.If_efficient_attention = True

if args.Multi_branch_concat_fusion == 'False':
    args.Multi_branch_concat_fusion = False
else:
    args.Multi_branch_concat_fusion = True

if args.If_Local_GLobal_Fuison == 'False':
    args.If_Local_GLobal_Fuison = False
else:
    args.If_Local_GLobal_Fuison = True

if args.If_binary_prediction == 'False':
    args.If_binary_prediction = False
else:
    args.If_binary_prediction = True

if args.If_Multiscale_Train == 'False':
    args.If_Multiscale_Train = False
else:
    args.If_Multiscale_Train = True

if args.If_Deep_Supervision == 'False':
    args.If_Deep_Supervision = False
else:
    args.If_Deep_Supervision = True

if args.If_pretrained == 'False':
    args.If_pretrained = False
else:
    args.If_pretrained = True

if args.If_weight_init == 'False':
    args.If_weight_init = False
else:
    args.If_weight_init = True

if args.If_use_UNet_decoder == 'False':
    args.If_use_UNet_decoder = False
else:
    args.If_use_UNet_decoder = True

if args.is_deconv == 'False':
    args.is_deconv = False
else:
    args.is_deconv = True

if args.if_sum_fusion == 'False':
    args.if_sum_fusion = False
else:
    args.if_sum_fusion = True

if args.If_use_UNet_fusion_stage_features == 'False':
    args.If_use_UNet_fusion_stage_features = False
else:
    args.If_use_UNet_fusion_stage_features = True

if args.If_use_position_embedding == 'False':
    args.If_use_position_embedding = False
else:
    args.If_use_position_embedding = True

if args.If_attention_scale == 'False':
    args.If_attention_scale = False
else:
    args.If_attention_scale = True

if args.If_out_side == 'False':
    args.If_out_side = False
else:
    args.If_out_side = True

if args.If_in_deep_sup == 'False':
    args.If_in_deep_sup = False
else:
    args.If_in_deep_sup = True

if args.If_backbone_use_Stoch_Depth == 'False':
    args.If_backbone_use_Stoch_Depth = False
else:
    args.If_backbone_use_Stoch_Depth = True

if args.If_remove_Norm == 'False':
    args.If_remove_Norm = False
else:
    args.If_remove_Norm = True

if args.If_remove_ReLU == 'False':
    args.If_remove_ReLU = False
else:
    args.If_remove_ReLU = True

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 选择数据集：
    dataset_config = {
        'Synapse': {
            'root_path': '../preprocessed_data/Synapse/train_npz',   # 换为绝对路径
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'Polyp': {
            'root_path': '/home/zhangyanhua/Code_python/Dataset/Medical_Dataset/Polyp/TrainDataset',
            'list_dir': './lists/lists_Polyp',
            'num_classes': 2,
            'z_spacing': 1,  # 这个是啥？？
        },
    }

    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    args.is_pretrain = True

    if args.If_binary_prediction:  # 如果采用 2 值计算的话
        args.num_classes = 1

    # ================================================================================
    # 实例化网络模型
    if args.Model_Name == 'TransUNet':

        from networks.vit_seg_modeling import VisionTransformer as ViT_seg
        from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
        # ---------------------------------------------------------------
        args.vit_name = 'R50-ViT-B_16'

        # 从 vit_seg_configs.py 中加载 网络搭建具体参数
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        
        if args.vit_name.find('R50') != -1:
            # 这是 position embedding 的 grid?
            config_vit.patches.grid = (int(args.img_size / config_vit.vit_patches_size), int(args.img_size / config_vit.vit_patches_size)) 

        # ---------------------------------------------------------------
        # 生成 保存数据集结果 的文件名
        args.exp = 'TU_' + dataset_name + str(args.img_size)

        Log_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Log')
        TensorboardX_path = "../Results/model_Trained/{}/{}".format(args.exp, 'TensorboardX')
        Model_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Model')
        Summary_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Summary')

        # --------------------------------
        # 创建文件名
        snapshot_path = '/TU_pretrain' if args.is_pretrain else '/TU'
        snapshot_path += '_' + args.vit_name
        snapshot_path = snapshot_path + '_skip' + str(config_vit.n_skip)  # 用了几个 skip connect

        # 如果不是默认值，就把修改值在文件名中体现出来
        snapshot_path = snapshot_path + '_vitpatch' + str(config_vit.vit_patches_size) if config_vit.vit_patches_size!=16 else snapshot_path
        snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
        snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
        snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
        snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
        snapshot_path = snapshot_path + '_'+str(args.img_size)

        # 可以通过改 seed 来进行重复实验
        snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path  

        # --------------------------------
        Log_path = Log_path + snapshot_path
        TensorboardX_path = TensorboardX_path + snapshot_path
        Model_path = Model_path + snapshot_path 
        Summary_path = Summary_path + snapshot_path

        if not os.path.exists(Log_path):
            os.makedirs(Log_path)

        if not os.path.exists(TensorboardX_path):
            os.makedirs(TensorboardX_path)

        if not os.path.exists(Model_path):
            os.makedirs(Model_path)

        if not os.path.exists(Summary_path):
            os.makedirs(Summary_path)

        # ---------------------------------------------------------------
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()  # 网络实例化
        net.load_from(weights=np.load(config_vit.pretrained_path))   # 加载预训练参数
        
    elif args.Model_Name == 'My_Model':
        from networks_my.configs_My_Model import get_My_Model_V10_config
        from networks_my.model_V10_Modified import My_Model

        config = get_My_Model_V10_config()

        config.backbone_name = args.backbone   # 替换为交互窗口的输入的 backbone
        if args.branch_key_channels is not None:
            config.branch_key_channels = args.branch_key_channels
        config.use_dilation = args.use_dilation

        config.Local_Global_fusion_method = args.Local_Global_fusion_method

        if args.branch_in_channels is not None:
            config.branch_in_channels = args.branch_in_channels
            
        if args.branch_out_channels is not None:
            config.branch_out_channels = args.branch_out_channels

        if args.branch_choose is not None:
            config.branch_choose = args.branch_choose

        if args.one_kv_head is not None:
            config.one_kv_head = args.one_kv_head

        if args.share_kv is not None:
            config.share_kv = args.share_kv

        if args.If_efficient_attention is not None:
            config.If_efficient_attention = args.If_efficient_attention

        if args.Multi_branch_concat_fusion is not None:
            config.Multi_branch_concat_fusion = args.Multi_branch_concat_fusion

        if args.If_Local_GLobal_Fuison is not None:
            config.If_Local_GLobal_Fuison = args.If_Local_GLobal_Fuison

        config.If_Deep_Supervision = args.If_Deep_Supervision

        config.If_pretrained = args.If_pretrained

        if args.Dropout_Rate_CNN is not None:
            config.Dropout_Rate_CNN = args.Dropout_Rate_CNN

        config.Dropout_Rate_Trans = args.Dropout_Rate_Trans
        config.Dropout_Rate_SegHead = args.Dropout_Rate_SegHead

        # config.drop_path_rate = args.Drop_Path_Rate

        # config.Dropout_Rate_Local_Global_Fusion = args.Dropout_Rate_Local_Global_Fusion

        config.Dropout_Rate_Multi_branch_fusion = args.Dropout_Rate_Multi_branch_fusion

        config.If_weight_init = args.If_weight_init

        config.If_use_UNet_decoder = args.If_use_UNet_decoder 
        config.is_deconv = args.is_deconv
        config.if_sum_fusion = args.if_sum_fusion

        if args.branch_depths is not None:
            config.branch_depths = args.branch_depths

        if args.branch_num_heads is not None:
            config.branch_num_heads = args.branch_num_heads

        config.If_use_UNet_fusion_stage_features = args.If_use_UNet_fusion_stage_features

        config.img_size  = args.img_size

        config.If_use_position_embedding = args.If_use_position_embedding

        config.name_position_method = args.name_position_method

        config.If_attention_scale = args.If_attention_scale

        config.If_out_side = args.If_out_side

        config.If_in_deep_sup = args.If_in_deep_sup

        config.If_backbone_use_Stoch_Depth = args.If_backbone_use_Stoch_Depth

        if args.Dropout_Rate_UNet is not None:
            config.Dropout_Rate_UNet = args.Dropout_Rate_UNet

        if args.Drop_path_rate_Trans is not None:
            config.Drop_path_rate_Trans = args.Drop_path_rate_Trans

        if args.Dropout_Rate_Local_Global_Fusion is not None:
            config.Dropout_Rate_Local_Global_Fusion = args.Dropout_Rate_Local_Global_Fusion

        if args.Dropout_Rate_Pos is not None:
            config.Dropout_Rate_Pos = args.Dropout_Rate_Pos

        config.If_remove_Norm = args.If_remove_Norm
        config.If_remove_ReLU = args.If_remove_ReLU
        
        # print(args.branch_key_channels)
        # print(config.branch_key_channels)
        # print(args.use_dilation)
        # print(config.use_dilation)
        # print(args.If_direct_sum_fusion)
        # print(config.If_direct_sum_fusion)

        # ---------------------------------------------------------------
        # 生成 保存数据集结果 的文件名
        args.exp = 'My_Model_' + dataset_name + str(args.img_size)

        Log_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Log')
        TensorboardX_path = "../Results/model_Trained/{}/{}".format(args.exp, 'TensorboardX')
        Model_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Model')
        Summary_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Summary')

        # --------------------------------
        # 创建文件名
        snapshot_path = '/My_Model_pretrain' if args.is_pretrain else '/My_Model'
        snapshot_path += '_' + config.backbone_name
        snapshot_path = snapshot_path + '_' + config.version
       
        # 如果不是默认值，就把修改值在文件名中体现出来
        snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
        snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
        snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
        snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
        snapshot_path = snapshot_path + '_'+str(args.img_size)

        # 可以通过改 seed 来进行重复实验
        snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path  

        # --------------------------------
        Log_path = Log_path + snapshot_path
        TensorboardX_path = TensorboardX_path + snapshot_path
        Model_path = Model_path + snapshot_path
        Summary_path = Summary_path + snapshot_path 

        if not os.path.exists(Log_path):
            os.makedirs(Log_path)

        if not os.path.exists(TensorboardX_path):
            os.makedirs(TensorboardX_path)

        if not os.path.exists(Model_path):
            os.makedirs(Model_path)

        if not os.path.exists(Summary_path):
            os.makedirs(Summary_path)

        # ---------------------------------------------------------------
        # 实例化网络
        net = My_Model(config, classes=args.num_classes).cuda()  # 网络实例化
        # net.load_from(weights=np.load(config_vit.pretrained_path))   # 加载预训练参数
    
    elif args.Model_Name == 'UTNet':
        from networks_UTNet.utnet import UTNet, UTNet_Encoderonly   # 加载自己的模型
        from networks_UTNet.configs_UTNet import get_My_Model_V10_config

        config = get_My_Model_V10_config()

        # ---------------------------------------------------------------
        # 生成 保存数据集结果 的文件名
        args.exp = 'UTNet_' + dataset_name + str(args.img_size)

        Log_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Log')
        TensorboardX_path = "../Results/model_Trained/{}/{}".format(args.exp, 'TensorboardX')
        Model_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Model')
        Summary_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Summary')

        # --------------------------------
        # 创建文件名
        snapshot_path = '/UTNet_pretrain' if args.is_pretrain else '/UTNet'
        snapshot_path += '_' + config.backbone_name
        snapshot_path = snapshot_path + '_' + config.version
       
        # 如果不是默认值，就把修改值在文件名中体现出来
        snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
        snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
        snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
        snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
        snapshot_path = snapshot_path + '_'+str(args.img_size)

        # 可以通过改 seed 来进行重复实验
        snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path  

        # --------------------------------
        Log_path = Log_path + snapshot_path
        TensorboardX_path = TensorboardX_path + snapshot_path
        Model_path = Model_path + snapshot_path
        Summary_path = Summary_path + snapshot_path 

        if not os.path.exists(Log_path):
            os.makedirs(Log_path)

        if not os.path.exists(TensorboardX_path):
            os.makedirs(TensorboardX_path)

        if not os.path.exists(Model_path):
            os.makedirs(Model_path)

        if not os.path.exists(Summary_path):
            os.makedirs(Summary_path)

        # ---------------------------------------------------------------
        net = UTNet(1, base_chan=32, num_classes=args.num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True).cuda()


    elif args.Model_Name == 'TransFuse':
        from networks_TransFuse.TransFuse import TransFuse_S, TransFuse_L, TransFuse_L_384

        # ---- build models ----
        if args.Scale_Choose == 'Scale_S':
            backbone_name = 'resnet34'
            version = 'Scale_S'
        elif args.Scale_Choose == 'Scale_L':
            backbone_name = 'resnet50'
            version = 'Scale_L'
        elif args.Scale_Choose == 'Scale_L_384':
            backbone_name = 'resnet50'
            version = 'Scale_L_384'

        # ---------------------------------------------------------------
        # 生成 保存数据集结果 的文件名
        args.exp = 'TransFuse_' + dataset_name + str(args.img_size)

        Log_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Log')
        TensorboardX_path = "../Results/model_Trained/{}/{}".format(args.exp, 'TensorboardX')
        Model_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Model')
        Summary_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Summary')

        # --------------------------------
        # 创建文件名
        snapshot_path = '/TransFuse_pretrain' if args.is_pretrain else '/TransFuse'
        snapshot_path += '_' + backbone_name
        snapshot_path = snapshot_path + '_' + version
       
        # 如果不是默认值，就把修改值在文件名中体现出来
        snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
        snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
        snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
        snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
        snapshot_path = snapshot_path + '_'+str(args.img_size)

        # 可以通过改 seed 来进行重复实验
        snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path  

        # --------------------------------
        Log_path = Log_path + snapshot_path
        TensorboardX_path = TensorboardX_path + snapshot_path
        Model_path = Model_path + snapshot_path
        Summary_path = Summary_path + snapshot_path 

        if not os.path.exists(Log_path):
            os.makedirs(Log_path)

        if not os.path.exists(TensorboardX_path):
            os.makedirs(TensorboardX_path)

        if not os.path.exists(Model_path):
            os.makedirs(Model_path)

        if not os.path.exists(Summary_path):
            os.makedirs(Summary_path)

        # ---- build models ----
        if args.Scale_Choose == 'Scale_S':
            net = TransFuse_S(num_classes=args.num_classes, pretrained=True).cuda()
        elif args.Scale_Choose == 'Scale_L':
            net = TransFuse_L(num_classes=args.num_classes, pretrained=True).cuda()
        elif args.Scale_Choose == 'Scale_L_384':
            net = TransFuse_L_384(num_classes=args.num_classes, pretrained=True).cuda()


    from trainer import trainer_synapse, trainer_Polyp
    # from model_summary_GFLOPs_params_FPS_Use_argparse import calculate_GFLOPs_params_FPS
    from model_summary_GFLOPs_params_FPS_Use_argparse_V1 import calculate_GFLOPs_params_FPS
    from model_summary_backbone import summary_backbone        

     # 先计算 GFLOPs_params_FPS
    # calculate_GFLOPs_params_FPS(Summary_path, args.Model_Name, net)
    # 加入 img_szie，不然由于 position embedding 会报错
    calculate_GFLOPs_params_FPS(Summary_path, args.Model_Name, net, image_size = config.img_size)  # 会把 net 从 cuda 放到 cpu 上
    summary_backbone(Summary_path, args.Model_Name, net) 

    trainer = {'Synapse': trainer_synapse, 'Polyp': trainer_Polyp}
    trainer[dataset_name](args, net.cuda(), Log_path, TensorboardX_path, Model_path) # 重新把 net 放到 cuda 上。

