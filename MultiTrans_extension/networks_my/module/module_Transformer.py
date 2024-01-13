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

# =========================================================================================

# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output


# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)

from timm.models.layers import trunc_normal_, DropPath

# =========================================================================================
def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape

# =========================================================================================
# 利用 mmcv 中的 build_norm_layer 来构建具有 BN 层的 conv 层，并且卷积核大小为 1x1。这是为了用于构建 MLP
class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))   # 这个 add_module 是干嘛的？？？

        # 这个 norm_cfg=dict(type='BN', requires_grad=True) 怎么设置？
        # bn = build_norm_layer(norm_cfg, b)[1]  
        bn = nn.BatchNorm2d(b)    
        for param in bn.parameters():
            param.requires_grad = True

        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

# ========================================================================
# 构建 conv + norm + active
class ConvModule(nn.Module):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=None):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))   # 这个 add_module 是干嘛的？？？

        if norm_cfg is not None:
            # 这个 norm_cfg=dict(type='BN', requires_grad=True) 怎么设置？
            # bn = build_norm_layer(norm_cfg, b)[1]  
            bn = nn.BatchNorm2d(b)    
            for param in bn.parameters():
                param.requires_grad = True
            # nn.init.constant_(bn.weight, 1)
            # nn.init.constant_(bn.bias, 0)
            self.add_module('bn', bn)  
        
        if act_cfg is not None:
            self.add_module('act', nn.ReLU())

# =========================================================================================
# offset map 的计算
class SFNet_warp_grid(nn.Module):
    def __init__(self, in_channel, middle_channel):
        super(SFNet_warp_grid, self).__init__()

        self.channel_change1 = nn.Conv2d(in_channel, middle_channel, kernel_size=1, bias=False)    # 利用 1x1 卷积核进行 channel 调整    
        self.channel_change2 = nn.Conv2d(in_channel, middle_channel, kernel_size=1, bias=False)    # 利用 1x1 卷积核进行 channel 调整 
        self.offset_map = nn.Conv2d(middle_channel*2, 2, kernel_size=3, padding=1, bias=False)    # 用于计算 offset map
           
    def forward(self, low_feature, h_feature):
        n, c, h, w = low_feature.size()           # low_feature 的 size
        
        h_feature = self.channel_change1(h_feature)   # channel 调整
        h_feature_up = F.interpolate(h_feature, (h, w), mode='bilinear', align_corners=True)   # 插值到大特征层的大小
        low_feature = self.channel_change2(low_feature)   # channel 调整

        fuse_feature = torch.cat([low_feature, h_feature_up], 1)         # 融合
        flow_field = self.offset_map(fuse_feature)                       # 计算 Flow Field：[n,2,h,w]

        norm = torch.tensor([[[[w,h]]]]).type_as(low_feature).to(low_feature.device)   # 用于归一化，grid 值要在 -1~1 之间：[1,1,1,2]
        grid_h = torch.linspace(-1,1,h).view(-1,1).repeat(1,w)              # [h,w]
        grid_w = torch.linspace(-1,1,w).repeat(h,1)                         # [h,w]
        grid = torch.cat((grid_w.unsqueeze(2), grid_h.unsqueeze(2)), 2)                           # 生成用于 grid_upsample 的网格：[h,w,2]  
        grid = grid.repeat(n,1,1,1).type_as(low_feature).to(low_feature.device)    # [n,h,w,2]

        warp_grid = grid + flow_field.permute(0,2,3,1)/norm      # 论文 Eq(2) 要除以 2，但论文代码并没有除以 2

        # out_h_feature = F.grid_sample(h_feature_origin, warp_grid)  # 利用网格法进行 feature 的 upsample
        return warp_grid

# =========================================================================================
# 对 efficient self-attention 代码进行优化：
class efficient_Attention_modified(torch.nn.Module):
    def __init__(self, 
                 dim,          # 输入 Transformer 的 channel 个数。或者说是各 stage feature concat 后的 channel 数。
                 key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 key_value_ratio=1,
                 norm_cfg=dict(type='BN', requires_grad=True),):
        super().__init__() 
        self.num_heads = num_heads  # head 的总数
        self.scale = key_dim ** -0.5
        self.key_value_ratio = key_value_ratio

        # key_dim 是每一个 self-attention head 中 query 和 key 的 channel
        self.key_dim = key_dim         
        self.nh_kd = nh_kd = key_dim * num_heads   # 所有 key 或 query 的总 channel 数。这里把 multi-heads 这个操作加了进去。后面可以通过 reshape 分离出来

        # 计算 self.d 是 value 的 channel。
        self.d = int(attn_ratio * key_dim)     # 每一个 value head 的 channel 数
        self.dh = int(attn_ratio * key_dim) * num_heads   # multi head 的 channel 数
        self.attn_ratio = attn_ratio

        # 这是利用 1x1 conv 实现 query, key, value 的 project。利用 1x1 covn 代替 linear layer。
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        if self.key_value_ratio > 1:
            self.key_value_spatial_scale = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=self.key_value_ratio, stride=self.key_value_ratio, groups=dim, bias=True),   # FLOPs: K*K*D*(N/K^2) = D*N
                nn.BatchNorm2d(dim, eps=1e-5),
            )

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, input_):  # x (B,N,C)
        B, C, H, W = get_shape(input_)

        queries = self.to_q(input_).reshape(B, self.num_heads, self.key_dim, H * W)   # B, heads, dk, n   # FLOPs：D*Dq*N

        if self.key_value_ratio == 1:
            # 利用 reshape 将图像展开
            keys = self.to_k(input_).reshape((B, self.num_heads, self.key_dim, H * W))    # B, heads, dk, n    # FLOPs：D*Dk*N
            values = self.to_v(input_).reshape((B, self.num_heads, self.d, H * W))        # B, heads, dv, n    # FLOPs：D*Dv*N
        else:
            # 对 key, value 的 spatial 进行 reduction。
            input_ = self.key_value_spatial_scale(input_)  # B, C, H, W ---> B, C, H/key_value_ratio, W/key_value_ratio   # FLOPs: D*N 
            # 进行 project 和 reshape
            keys = self.to_k(input_).reshape((B, self.num_heads, self.key_dim, -1))    # B, heads, dk, n'       # FLOPs：D*Dk*N/ratio^2
            values = self.to_v(input_).reshape((B, self.num_heads, self.d, -1))        # B, heads, dv, n'       # FLOPs：D*Dv*N/ratio^2    

        key = F.softmax(keys, dim=3)       # B, heads, dk, n'
        query = F.softmax(queries, dim=2)  # B, heads, dk, n
        value = values                     # B, heads, dv, n' 

        context = torch.matmul(key, value.permute(0, 1, 3, 2))  # 输出：B, heads, dk, dv    # FLOPs：n'*Dk*Dv = Dk*Dv*N/ratio^2

        attended_value = torch.matmul(context.permute(0, 1, 3, 2), query)  # B, heads, dv, n  # FLOPs：N*Dq*Dv = N*Dk*Dv  
        attended_value = attended_value.reshape(B, self.dh, H, W)          # 重新展开为图像

        reprojected_value = self.proj(attended_value)   # 重新映射回输入图像的 channel 数   # D*Dv*N

        return reprojected_value

# -----------------------------------------------------------------------
# 使用 Linformer 中的 param sharing 对 efficient self-attention 进行修改：
class efficient_Attention_modified_Param_sharing(torch.nn.Module):
    def __init__(self, 
                 dim,          # 输入 Transformer 的 channel 个数。或者说是各 stage feature concat 后的 channel 数。
                 key_dim, 
                 num_heads,
                 attn_ratio=4,
                 one_kv_head = True,
                 share_kv = True,
                 activation=None,
                 key_value_ratio=1,
                 norm_cfg=dict(type='BN', requires_grad=True),):
        super().__init__() 
        self.num_heads = num_heads  # head 的总数

        qkv_dim = key_dim

        self.scale = qkv_dim ** -0.5
        self.key_value_ratio = key_value_ratio

        self.share_kv = share_kv
        self.one_kv_head = one_kv_head 

        self.qkv_dim = qkv_dim

        # query 的 dim 及 line projection
        self.q_dim = qkv_dim         
        self.multi_q_dim = multi_q_dim = qkv_dim * num_heads   
        self.to_q = Conv2d_BN(dim, multi_q_dim, 1, norm_cfg=norm_cfg)

        # 根据是否 one_kv_head，确定 kv_dim
        kv_dim = qkv_dim if one_kv_head else qkv_dim*num_heads   
        self.kv_heads = 1 if one_kv_head else num_heads

        # 根据是否 share_kv 确定 key、value 的 line projection
        if not share_kv:
            self.to_k = Conv2d_BN(dim, kv_dim, 1, norm_cfg=norm_cfg)
            self.to_v = Conv2d_BN(dim, kv_dim, 1, norm_cfg=norm_cfg)     
        else:
            self.to_kv = Conv2d_BN(dim, kv_dim, 1, norm_cfg=norm_cfg)

        if self.key_value_ratio > 1:
            self.input_spatial_scale = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=self.key_value_ratio, stride=self.key_value_ratio, groups=dim, bias=True),   # FLOPs: K*K*D*(N/K^2) = D*N
                nn.BatchNorm2d(dim, eps=1e-5),
            )

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            multi_q_dim, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, input_):  # x (B,N,C)
        B, C, H, W = get_shape(input_)

        queries = self.to_q(input_).reshape(B, self.num_heads, self.qkv_dim, H * W)   # B, heads, dq, n   # FLOPs：D*Dq*N

        if self.key_value_ratio == 1:
            if self.share_kv:
                    keys = values = self.to_kv(input_).reshape((B, self.kv_heads, self.qkv_dim, H * W))    # B, heads, dkv, n    # FLOPs：D*Dk*N
            else:
                    keys = self.to_k(input_).reshape((B, self.kv_heads, self.qkv_dim, H * W))    # B, heads, dkv, n    # FLOPs：D*Dk*N
                    values = self.to_v(input_).reshape((B, self.kv_heads, self.qkv_dim, H * W))        # B, heads, dkv, n    # FLOPs：D*Dv*N
        else:
            # 对 key, value 的 spatial 进行 reduction。
            input_ = self.input_spatial_scale(input_)  # B, C, H, W ---> B, C, H/key_value_ratio, W/key_value_ratio   # FLOPs: D*N 
            # 进行 project 和 reshape
            if self.share_kv:
                    keys = values = self.to_kv(input_).reshape((B, self.kv_heads, self.qkv_dim, -1))    # B, heads, dkv, n'    # FLOPs：D*Dk*N
            else:
                    keys = self.to_k(input_).reshape((B, self.kv_heads, self.qkv_dim, -1))    # B, heads, dkv, n'    # FLOPs：D*Dk*N
                    values = self.to_v(input_).reshape((B, self.kv_heads, self.qkv_dim, -1))        # B, heads, dkv, n'    # FLOPs：D*Dv*N 

        key = F.softmax(keys, dim=3)       # B, heads, dkv, n'
        query = F.softmax(queries, dim=2)  # B, heads, dq, n
        value = values                     # B, heads, dkv, n' 

        context = torch.matmul(key, value.permute(0, 1, 3, 2))  # 输出：B, heads, dkv, dkv    # FLOPs：n'*Dk*Dv = Dk*Dv*N/ratio^2
        context = context.expand(-1, self.num_heads, -1, -1).permute(0, 1, 3, 2) # B, heas, dkv, dkv

        attended_value = torch.matmul(context, query)  # B, heads, dkv, n  # FLOPs：N*Dq*Dv = N*Dk*Dv  
        attended_value = attended_value.reshape(B, self.multi_q_dim, H, W)          # 重新展开为图像

        reprojected_value = self.proj(attended_value)   # 重新映射回输入图像的 channel 数   # D*Dv*N

        return reprojected_value

# -----------------------------------------------------------------------
# 使用 Linformer 中的 param sharing 对 efficient self-attention 进行修改：
class efficient_Attention_modified_Param_sharing_Share_spatial_reduction(torch.nn.Module):
    def __init__(self, 
                 dim,          # 输入 Transformer 的 channel 个数。或者说是各 stage feature concat 后的 channel 数。
                 key_dim, 
                 num_heads,
                 attn_ratio=4,
                 one_kv_head = True,
                 share_kv = True,
                 activation=None,
                 key_value_ratio=1,
                 If_attention_scale=False,
                 norm_cfg=dict(type='BN', requires_grad=True),):
        super().__init__() 
        self.num_heads = num_heads  # head 的总数

        qkv_dim = key_dim

        self.If_attention_scale = If_attention_scale

        self.key_value_ratio = key_value_ratio

        self.share_kv = share_kv
        self.one_kv_head = one_kv_head 

        self.qkv_dim = qkv_dim

        # query 的 dim 及 line projection
        self.q_dim = qkv_dim         
        self.multi_q_dim = multi_q_dim = qkv_dim * num_heads   
        self.to_q = Conv2d_BN(dim, multi_q_dim, 1, norm_cfg=norm_cfg)

        # 根据是否 one_kv_head，确定 kv_dim
        kv_dim = qkv_dim if one_kv_head else qkv_dim*num_heads   
        self.kv_heads = 1 if one_kv_head else num_heads

        # 根据是否 share_kv 确定 key、value 的 line projection
        if not share_kv:
            self.to_k = Conv2d_BN(dim, kv_dim, 1, norm_cfg=norm_cfg)
            self.to_v = Conv2d_BN(dim, kv_dim, 1, norm_cfg=norm_cfg)     
        else:
            self.to_kv = Conv2d_BN(dim, kv_dim, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            multi_q_dim, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, input_, key_v_input_reduction):  # x (B,N,C)
        B, C, H, W = get_shape(input_)

        queries = self.to_q(input_).reshape(B, self.num_heads, self.qkv_dim, H * W)   # B, heads, dq, n   # FLOPs：D*Dq*N

        if self.key_value_ratio == 1:
            if self.share_kv:
                keys = values = self.to_kv(input_).reshape((B, self.kv_heads, self.qkv_dim, H * W))    # B, heads, dkv, n    # FLOPs：D*Dk*N
            else:
                keys = self.to_k(input_).reshape((B, self.kv_heads, self.qkv_dim, H * W))    # B, heads, dkv, n    # FLOPs：D*Dk*N
                values = self.to_v(input_).reshape((B, self.kv_heads, self.qkv_dim, H * W))        # B, heads, dkv, n    # FLOPs：D*Dv*N
        else:
            # 在 BasicLayer_Share_spatial_reduction 对 key, value 的 spatial 进行 reduction。
            input_ = key_v_input_reduction  # 同一个 branch 的所有 attention layer 共用一个 key, value 的 spatial reduction
            
            # 进行 project 和 reshape
            if self.share_kv:
                keys = values = self.to_kv(input_).reshape((B, self.kv_heads, self.qkv_dim, -1))    # B, heads, dkv, n'    # FLOPs：D*Dk*N
            else:
                keys = self.to_k(input_).reshape((B, self.kv_heads, self.qkv_dim, -1))    # B, heads, dkv, n'    # FLOPs：D*Dk*N
                values = self.to_v(input_).reshape((B, self.kv_heads, self.qkv_dim, -1))        # B, heads, dkv, n'    # FLOPs：D*Dv*N 

        key = F.softmax(keys, dim=3)       # B, 1/heads, dkv, n'
        query = F.softmax(queries, dim=2)  # B, heads, dq, n
        value = values                     # B, 1/heads, dkv, n' 

        if self.If_attention_scale:
            scale = (key.size()[3]) ** -0.5  # 没用到

        if self.If_attention_scale:
            context = torch.matmul(key*scale, value.permute(0, 1, 3, 2))  # 输出：B, 1/heads, dkv, dkv    # FLOPs：n'*Dk*Dv = Dk*Dv*N/ratio^2
        else:
            context = torch.matmul(key, value.permute(0, 1, 3, 2))  # 输出：B, 1/heads, dkv, dkv    # FLOPs：n'*Dk*Dv = Dk*Dv*N/ratio^2
            
        if self.one_kv_head:
            context = context.expand(-1, self.num_heads, -1, -1).permute(0, 1, 3, 2) # B, heads, dkv, dkv
        else:
            context = context.permute(0, 1, 3, 2)

        attended_value = torch.matmul(context, query)  # B, heads, dkv, n           # FLOPs：N*Dq*Dv = N*Dk*Dv  
        attended_value = attended_value.reshape(B, self.multi_q_dim, H, W)          # 重新展开为图像

        reprojected_value = self.proj(attended_value)   # 重新映射回输入图像的 channel 数   # D*Dv*N

        return reprojected_value

# -----------------------------------------------------------------------
# 这是标准 attention 
# 构建 self-attention block。MHSA layer
class Attention_Param_sharing_Kv_sharing(torch.nn.Module):
    def __init__(self, 
                 dim, 
                 key_dim, 
                 num_heads,
                 attn_ratio=4,
                 one_kv_head = True,
                 share_kv = True,   
                 activation=None,
                 If_attention_scale=False,
                 norm_cfg=dict(type='BN', requires_grad=True),):
        super().__init__() 
        self.num_heads = num_heads
        self.one_kv_head = one_kv_head
        self.share_kv = share_kv

        self.If_attention_scale = If_attention_scale

        if If_attention_scale:
            self.scale = key_dim ** -0.5

        self.key_dim = key_dim  # key_dim 是每一个 self-attention head 的 channel

        # q k 是否使用 one head
        qk_dim = key_dim if one_kv_head else key_dim * num_heads   
        self.qk_heads = 1 if one_kv_head else num_heads

        # self.d = int(attn_ratio * key_dim)
        # self.dh = int(attn_ratio * key_dim) * num_heads

        self.d = key_dim
        self.dh =  key_dim * num_heads
        
        self.attn_ratio = attn_ratio

        # 这是利用 1x1 conv 实现 query, key, value 的 project。利用 1x1 covn 代替 linear layer。
        if not share_kv:
            self.to_q = Conv2d_BN(dim, qk_dim, 1, norm_cfg=norm_cfg)
            self.to_k = Conv2d_BN(dim, qk_dim, 1, norm_cfg=norm_cfg)
        else:
            self.to_qk = Conv2d_BN(dim, qk_dim, 1, norm_cfg=norm_cfg)

        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, x, key_v_input_reduction):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        
        if not self.share_kv:
            qq = self.to_q(x).reshape(B, self.qk_heads, self.key_dim, H * W).permute(0, 1, 3, 2)   # B, heads, n, dk
            kk = self.to_k(x).reshape(B, self.qk_heads, self.key_dim, H * W)                        # B, heads, dk, n
        else:
            qq = kk = self.to_qk(x).reshape(B, self.qk_heads, self.key_dim, H * W)
            qq = qq.permute(0, 1, 3, 2)


        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)           # B, heads, n, dv

        if self.If_attention_scale:
            attn = torch.matmul(qq*self.scale, kk)    # 矩阵乘法。得到 B, heads, n, n
        else:
            attn = torch.matmul(qq, kk)    # 矩阵乘法。得到 B, heads, n, n


        attn = attn.softmax(dim=-1)    # 进行归一化。按行

        if self.one_kv_head:
            attn.expand(-1, self.num_heads, -1, -1)

        xx = torch.matmul(attn, vv)    # B, heads, n, dv

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W) # B x num_heads x self.d x HW ---> B x self.dh x H x W    
        xx = self.proj(xx) # 利用 1x1 conv 进行 linear project
        return xx

# -----------------------------------------------------------------------
# 这是标准 attention 
# 构建 self-attention block。MHSA layer
class Attention(torch.nn.Module):
    def __init__(self, 
                 dim, 
                 key_dim, 
                 num_heads,
                 attn_ratio=4,   
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True),):
        super().__init__() 
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5

        self.key_dim = key_dim  # key_dim 是每一个 self-attention head 的 channel
        self.nh_kd = nh_kd = key_dim * num_heads # num_head key_dim  # 这里把 multi-heads 这个操作加了进去。后面可以通过 reshape 分离出来

        # self.d = int(attn_ratio * key_dim)
        # self.dh = int(attn_ratio * key_dim) * num_heads

        self.d = key_dim
        self.dh =  key_dim * num_heads
        
        self.attn_ratio = attn_ratio

        # 这是利用 1x1 conv 实现 query, key, value 的 project。利用 1x1 covn 代替 linear layer。
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, x, key_v_input_reduction):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)   # B, heads, n, dk
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)                        # B, heads, dk, n
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)           # B, heads, n, dv

        attn = torch.matmul(qq, kk)    # 矩阵乘法。得到 B, heads, n, n
        attn = attn.softmax(dim=-1)    # 进行归一化。按行

        xx = torch.matmul(attn, vv)    # B, heads, n, dv

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W) # B x num_heads x self.d x HW ---> B x self.dh x H x W    
        xx = self.proj(xx) # 利用 1x1 conv 进行 linear project
        return xx

# -----------------------------------------------------------------------
# 利用 1x1 conv 来构建 MLP。这里是 TopFormer 里的 Feed-Forward Network
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0., norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg) # 这是 1x1 conv + BN               # FLOPs：Din*Dhid*N

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)   # FLOPs：3*3*Dhid*N
        self.act = act_layer()  # 激活函数

        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg) # 这是 1x1 conv + BN              # FLOPs：Dhid*Dout*N

        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)     
        x = self.fc2(x)
        x = self.drop(x)  # 这里这个 dropout 在外面 forward 中被 droppath 代替
        return x

# ------------------------------------------------------------
# 构建一个 Transformer block: Self-attention + FFN
class Block_Spatial_Battleneck_Share_spatial_reduction(nn.Module):

    def __init__(self, dim, key_dim, num_heads, Spatial_ratio, key_value_ratio, mlp_ratio=4., attn_ratio=2., drop=0., drop_path=0., one_kv_head = True, share_kv = True, If_efficient_attention = True, If_attention_scale = False, act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim      # 输入 Transformer 的 channel 个数。或者说是各 stage feature concat 后的 channel 数。
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.Spatial_ratio = Spatial_ratio

        one_kv_head = one_kv_head
        share_kv = share_kv
        self.If_efficient_attention = If_efficient_attention

        # 进行 spatial reduction
        if self.Spatial_ratio > 1:
            self.spatial_reduction = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=self.Spatial_ratio, stride=self.Spatial_ratio, groups=dim, bias=True),   # FLOPs: K*K*D*(N/K^2) = D*N
                # eps=1e-5 是默认值。a value added to the denominator for numerical stability. Default: 1e-5
                nn.BatchNorm2d(dim, eps=1e-5),  
            )

        # Total FLOPs：D*Dq*N + D*N + D*Dk*N/ratio^2 + D*Dv*N/ratio^2 + Dk*Dv*N/ratio^2 + N*Dk*Dv + D*Dv*N 
        if self.If_efficient_attention:
            self.attn = efficient_Attention_modified_Param_sharing_Share_spatial_reduction(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, one_kv_head = one_kv_head, share_kv = share_kv, activation=act_layer, key_value_ratio=key_value_ratio, If_attention_scale=If_attention_scale, norm_cfg=norm_cfg)
        else:
            self.attn = Attention_Param_sharing_Kv_sharing(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, one_kv_head = one_kv_head, share_kv = share_kv, activation=act_layer, If_attention_scale=If_attention_scale, norm_cfg=norm_cfg)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # nn.Identity() 是占位置用的
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)   # 计算隐含层的 channel 数

        # Total FLOPs：  Din*Dhid*N + 3*3*Dhid*N + Dhid*Dout*N 
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)

        # 进行 spatial recover
        if self.Spatial_ratio > 1:
            self.spatial_recover = nn.ModuleList()
            self.spatial_recover.append(SFNet_warp_grid(dim, dim//2))
            

    def forward(self, input, key_v_input_reduction):

        if self.Spatial_ratio > 1:
            input_reduction = self.spatial_reduction(input) # 利用 conv 进行 spatial 的 reduction
        else:
            input_reduction = input

        attention_output = input_reduction + self.drop_path(self.attn(input_reduction, key_v_input_reduction)) # self-attention + project
        
        MLP_output = attention_output + self.drop_path(self.mlp(attention_output))  # MLP

        if self.Spatial_ratio > 1:
            # stages_warp_grid = self.spatial_recover[0](input, MLP_output)   # 学习 offset map
            stages_warp_grid = self.spatial_recover[0](input, input_reduction)   # 学习 offset map
            output_recover = F.grid_sample(MLP_output, stages_warp_grid, align_corners=True)  # 利用网格法进行 spatial 的 recover
        else:
            output_recover = MLP_output

        return output_recover


# =========================================================================================
# 构建 多层 Transformers。也就是一个 branch
class BasicLayer_Share_spatial_reduction(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads, Spatial_ratio, key_value_ratio,
                mlp_ratio=4., attn_ratio=2., drop=0., drop_path=0., 
                one_kv_head = True, share_kv = True, If_efficient_attention = True,
                If_attention_scale = False,
                norm_cfg=dict(type='BN2d', requires_grad=True), 
                act_layer=None):
        super().__init__()
        self.block_num = block_num

        self.key_value_ratio = key_value_ratio

        # -----------------------------------------
        # 同一个 branch 的 attention layer 共用一个 key, value 的 spatial reduction     
        if key_value_ratio > 1:
            self.input_spatial_scale = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=key_value_ratio, stride=key_value_ratio, groups=embedding_dim, bias=True),   # FLOPs: K*K*D*(N/K^2) = D*N
                nn.BatchNorm2d(embedding_dim, eps=1e-5),
            )

            # self.input_spatial_scale = nn.Sequential(
            #     torch.nn.AvgPool2d((key_value_ratio)),   # 利用 kernel 和 stride 为 key_value_ratio 的 AvgPool2d 进行下采样
            #     # nn.BatchNorm2d(embedding_dim, eps=1e-5),
            # )

        # -----------------------------------------
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            # 可选 Block_Spatial_Battleneck 或 Block
            self.transformer_blocks.append(Block_Spatial_Battleneck_Share_spatial_reduction(
                embedding_dim,               # 输入 Transformer 的 channel 个数。或者说是各 stage feature concat 后的 channel 数。 
                key_dim=key_dim, 
                num_heads=num_heads,
                Spatial_ratio=Spatial_ratio,
                key_value_ratio=key_value_ratio,
                mlp_ratio=mlp_ratio, 
                attn_ratio=attn_ratio,
                drop=drop, 
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                one_kv_head = one_kv_head, 
                share_kv = share_kv,
                If_efficient_attention = If_efficient_attention,
                If_attention_scale = If_attention_scale,
                norm_cfg=norm_cfg,
                act_layer=act_layer))

    def forward(self, x):
        # token * N 
        if self.key_value_ratio > 1:
            key_v_input_reduction = self.input_spatial_scale(x)
        else:
            key_v_input_reduction = None

        for i in range(self.block_num):
            x = self.transformer_blocks[i](x, key_v_input_reduction)  

            # 同一个 branch 的所有 attention layer 共用一个 key, value 的 spatial reduction
            if self.key_value_ratio > 1:
                key_v_input_reduction = self.input_spatial_scale(x)
            else:
                key_v_input_reduction = None

        return x
