import math
import torch
from torch import nn
import torch.nn.functional as F

# from mmcv.cnn import ConvModule
# from mmcv.cnn import build_norm_layer
# from mmcv.runner import BaseModule
# from mmcv.runner import _load_checkpoint
# from mmseg.utils import get_root_logger

# from ..builder import BACKBONES

# =========================================================================================
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape

# =========================================================================================
# 利用 mmcv 中的 build_norm_layer 来构建具有 BN 层的 conv 层，并且卷积核大小为 1x1。这是为了用于构建 MLP
class Conv2d_BN(nn.Sequential):
    def __init__(self, inp=1, oup=1, kernel_size=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = inp
        self.out_channel = oup
        self.ks = kernel_size
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            inp, oup, kernel_size, stride, pad, dilation, groups, bias=False))   # 这个 add_module 是干嘛的？？？

        # 这个 norm_cfg=dict(type='BN', requires_grad=True) 怎么设置？
        # bn = build_norm_layer(norm_cfg, b)[1]  
        bn = nn.BatchNorm2d(oup)    
        for param in bn.parameters():
            param.requires_grad = True

        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

# ========================================================================
# 构建 conv + norm + active

class ConvModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, stride=1, pad=0, dilation=1,
                 groups=1, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=None):
        super().__init__()
        self.inp_channel = inp
        self.out_channel = oup
        self.ks = kernel_size
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            inp, oup, kernel_size, stride, pad, dilation, groups, bias=False))   # 这个 add_module 是干嘛的？？？

        if norm_cfg is not None:
            # 这个 norm_cfg=dict(type='BN', requires_grad=True) 怎么设置？
            # bn = build_norm_layer(norm_cfg, b)[1]  
            bn = nn.BatchNorm2d(oup)    
            for param in bn.parameters():
                param.requires_grad = True
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
            self.add_module('bn', bn)  
        
        if act_cfg is not None:
            self.add_module('act', nn.ReLU())


# =========================================================================================
# 使用 TopFormer 中的 fusion 方法 
# 这个 sigmod 后面 +3 / 6 是干嘛？
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class local_global_Fusion_TopFormer(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        norm_cfg=dict(type='BN', requires_grad=True),
        activations = None,
    ) -> None:
        super(local_global_Fusion_TopFormer, self).__init__()
        self.norm_cfg = norm_cfg

        # act_cfg=None 意思是不加激活函数？
        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, local_feature, global_feature):
        '''
        x_g: global features
        x_l: local features
        '''
        local_feat = self.local_embedding(local_feature)
        
        global_act = self.global_act(global_feature)
        sig_act = self.act(global_act)
        
        global_feat = self.global_embedding(global_feature)
        
        out = local_feat * sig_act + global_feat
        return out

# =========================================================================================
# 利用 global average pooling 进行 fusion  
class local_global_Fusion_Average(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        norm_cfg=dict(type='BN', requires_grad=True),
        activations = None,
    ) -> None:
        super(local_global_Fusion_Average, self).__init__()
        self.norm_cfg = norm_cfg

        # act_cfg=None 意思是不加激活函数？
        # self.local_channel_change = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        # self.global_channel_change = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.local_channel_change = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1, bias=False),
            nn.BatchNorm2d(oup),
            )
        self.global_channel_change = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1, bias=False),
            nn.BatchNorm2d(oup),
            )
        

    def forward(self, local_feature, global_feature):
     
        local_feat = self.local_channel_change(local_feature)
        global_feat = self.global_channel_change(global_feature)

        global_weight = nn.functional.adaptive_avg_pool2d(global_feat, (1,1))
        
        out = local_feat * global_weight + global_feat

        return out