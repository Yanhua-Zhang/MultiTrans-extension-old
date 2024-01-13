# ------------------------------------------------------------------------------
# 注意！HRNet 中提供的 get_model_summary 和 from torchsummary import summary 
# 都无法计算 AdaptiveAvgPool2d 中的 parameters 和 GFLOPs
# ------------------------------------------------------------------------------
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn as nn

from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port, get_args, build_logger, check_makedirs

# from torchsummary import summary
from torchsummary.torchsummary import summary
from torchstat import stat

from util.model_FLOPS_HRNet import get_model_summary     # HRNet 中计算方式
from util.model_FLOPS import get_model_complexity_info  # SFNet 中计算方式

from util.model_FPS import speed_test, FPS_counter

# ------------------------------------------------------------------------------------
# from model.backbone import resnet18_SFNet, resnet50_SFNet   # 加载预训练模型与参数

# from model.model_FCN_native_FPN import FCN_native_FPN

# from model.model_FPN_Bottom_up import FPN_Bottom_up
# from model.model_FPN_Scales_fuse import FPN_Scales_fuse

# from model.model_FPN_Bottom_up_Scales_fuse import FPN_Bottom_up_Scales_fuse
# ------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# 利用不同方式计算 GFLOPs、params，然后存入 logger 中
def GFLOPs_params_counter(model, model_name, height, weight, logger):
    logger.info('-----------------------------------------------------------------------')
    logger.info('model name：'+model_name+': ')
    # HRNet 计算方式----------------------------
    logger.info('开始利用 HRNet 中方式进行计算：')
    dump_input = torch.rand((1, 3, height, weight))
    logger.info(get_model_summary(model, dump_input))
    logger.info('End')
    logger.info('--------------------------------')

    # SFNet 计算方式----------------------------
    logger.info('开始利用 SFNet 中方式进行计算：')
    flops_count, params_count = get_model_complexity_info(model, (height, weight), print_per_layer_stat=False, as_strings=True, channel=3)
    logger.info('FLOPs 为： '+flops_count)
    logger.info('params 总数为： '+params_count)
    logger.info('End')
    logger.info('--------------------------------')

    # torch 计算方式----------------------------
    logger.info('开始利用 torch 中方式进行计算：')
    # stat(model, (3, height, weight))
    logger.info('End')
    logger.info('--------------------------------')

#-------------------------------------------------------------------------------------------------

def calculate_GFLOPs_params_FPS(summary_path, model_name, model):

    model = model.cpu()
         
    file_name_log = summary_path+'/'+"model_GFLOPs_params_FPS.log"  # logger 的文件名
    logger = build_logger('summary_FPS', file_name_log)

    height, weight = 512, 512  # 1024, 1024
    GFLOPs_params_counter(model, model_name, height, weight, logger)
    height, weight = 512, 512  # 1024, 1024  1024, 2048
    FPS_counter(model, model_name, height, weight, logger, iteration=100)
