# ------------------------------------------------------------------------------
# 注意！HRNet 中提供的 get_model_summary 和 from torchsummary import summary 
# 都无法计算 AdaptiveAvgPool2d 中的 parameters 和 GFLOPs
# ------------------------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import torch.nn as nn

from fvcore.nn import FlopCountAnalysis, parameter_count_table

from util.util import build_logger



def calculate_GFLOPs_params_FPS(summary_path, model_name, model, image_size):

    model = model.cpu()
    model.eval()
         
    file_name_log = summary_path+'/'+"model_GFLOPs_params_FPS.log"  # logger 的文件名
    logger = build_logger('summary_FPS', file_name_log)

    # tensor = (torch.rand(1, 256, 56, 56), torch.rand(1, 512, 28, 28), torch.rand(1, 1024, 14, 14), torch.rand(1, 2048, 7, 7))
    tensor = torch.rand(1, 3, image_size, image_size)

    flops = FlopCountAnalysis(model, tensor)
    # print("FLOPs: ", flops.total())
    logger.info("FLOPs: ")
    logger.info(flops.total())
    logger.info('--------------------------------')
    
    # print("flops.by_operator: ", flops.by_operator())
    logger.info("flops.by_operator: ")
    logger.info(flops.by_operator())
    logger.info('--------------------------------')

    # print("flops.by_module(): ", flops.by_module())
    # logger.info("flops.by_module(): ")
    # logger.info(flops.by_module())
    # logger.info('--------------------------------')

    # print(parameter_count_table(model))
    logger.info(parameter_count_table(model))

