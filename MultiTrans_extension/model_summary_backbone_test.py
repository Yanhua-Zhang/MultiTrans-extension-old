
from networks_my.backbone import resnet18_SFNet, resnet50_SFNet   # 加载预训练模型与参数

from torchvision.models import resnet34
from torchvision.models import resnet50

 
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port, get_args, build_logger, check_makedirs

# # --------------------------------------------
# # 检查保存 summary log 的文件夹是否存在，不存在进行生成
# summary_path = '/home/zhangyanhua/Code_python/Semantic-seg-multiprocessing-general-Test-final-final/save/summary'
# check_makedirs(summary_path)            

# file_name_log = summary_path+'/'+"backbone_summary.log"  # logger 的文件名
# logger = build_logger('summary_backbone', file_name_log)


# --------------------------------------------------------------------------------
def summary_backbone(summary_path, model_name, model):
    # --------------------------------------------          
    file_name_log = summary_path+'/'+"backbone_summary_" + model_name + ".log" # logger 的文件名
    logger = build_logger('summary_backbone', file_name_log)


    logger.info('backbone name：'+model_name+': ')
    logger.info(model)
    logger.info('-----------------------------------------------------------------------------')
    logger.info('   ')


# --------------------------------------------------------------------------------
summary_path = '/home/zhangyanhua/Code_python/Project_TransUNet_My_Modified_V12/Summary_Test'
model_name = 'ResNet50'
model = resnet50()

summary_backbone(summary_path, model_name, model)

# --------------------------------------------------------------------------------
summary_path = '/home/zhangyanhua/Code_python/Project_TransUNet_My_Modified_V12/Summary_Test'
model_name = 'ResNet50_SFNet'
model = resnet50_SFNet(True)

summary_backbone(summary_path, model_name, model)