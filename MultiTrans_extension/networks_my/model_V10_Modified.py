import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models import resnet34
from torchvision.models import resnet50

from .module import BasicLayer_Share_spatial_reduction, local_global_Fusion_Average, SFNet_warp_grid

from .module.grid_attention_layer import GridAttentionBlock2D
from .module.UNet_utils import unetConv2, unetUp

from .module.position_embedding import Pos_Embed_Sinusoid

import math

from timm.models.layers import trunc_normal_, DropPath

from .backbone.backbone_resnet_Deep import resnet18_Deep, resnet50_Deep   # 加载预训练模型与参数


# =====================================================
# 使用 TransFuse 中的 init_weights 的方法
def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # print(str(m)+'is initialized by kaiming')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        # print(str(m)+'is initialized by constant')
    
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
        # print(str(m)+'is initialized by constant')


class My_Model(nn.Module):
    def __init__(self,  
                    config,
                    classes=2,                     
                    ):
        super(My_Model, self).__init__()

        self.use_dilation = config.use_dilation  
    
        self.norm_cfg = config.norm_cfg 
        
        self.branch_choose = config.branch_choose

        self.key_value_ratios = config.key_value_ratios   # 控制各个 branch 中 self-attention 中 key 和 value 的 spatial reduction

        self.If_direct_upsampling = config.If_direct_upsampling

        self.is_dw = config.is_dw
        
        self.branch_in_channels = config.branch_in_channels

        self.Multi_branch_concat_fusion = config.Multi_branch_concat_fusion

        self.If_Local_GLobal_Fuison = config.If_Local_GLobal_Fuison

        self.If_Deep_Supervision = config.If_Deep_Supervision

        self.Dropout_Rate_CNN = config.Dropout_Rate_CNN
        self.Dropout_Rate_Trans = config.Dropout_Rate_Trans
        self.Dropout_Rate_SegHead = config.Dropout_Rate_SegHead # seg head 中的 Dropout rate 可以设低一点。

        self.Dropout_Rate_Local_Global_Fusion = config.Dropout_Rate_Local_Global_Fusion

        self.Dropout_Rate_Multi_branch_fusion = config.Dropout_Rate_Multi_branch_fusion

        self.If_weight_init = config.If_weight_init
        
        # 选择 local globa fusion 的方式
        self.Local_Global_fusion_method = config.Local_Global_fusion_method

        # 添加 UNet decoder
        self.If_use_UNet_decoder = config.If_use_UNet_decoder
        self.is_deconv=config.is_deconv
        self.if_sum_fusion = config.if_sum_fusion

        # 添加 branch depths 和 num_heads 的调整
        self.branch_depths = config.branch_depths
        self.branch_num_heads = config.branch_num_heads

        self.If_use_UNet_fusion_stage_features = config.If_use_UNet_fusion_stage_features

        # 添加 position embedding:
        self.If_use_position_embedding = config.If_use_position_embedding
        self.name_position_method = config.name_position_method

        self.img_size = config.img_size

        self.If_attention_scale = config.If_attention_scale

        self.If_out_side = config.If_out_side 

        self.If_in_deep_sup  = config.If_in_deep_sup 

        self.If_backbone_use_Stoch_Depth  = config.If_backbone_use_Stoch_Depth

        self.Dropout_Rate_UNet  = config.Dropout_Rate_UNet 

        self.Dropout_Rate_Pos = config.Dropout_Rate_Pos

        self.If_remove_ReLU = config.If_remove_ReLU
        self.If_remove_Norm = config.If_remove_Norm

        # -------------------------------------------------------------------
        # backbone 加载
        if config.backbone_name == 'resnet18_Deep':
            resnet = resnet18_Deep(config.If_pretrained)
            stage_channels = [32, 64, 128, 256, 512]  # 'resnet18' 各 stage 的输出 channel

        elif config.backbone_name == 'resnet50_Deep':
            resnet = resnet50_Deep(config.If_pretrained)
            stage_channels = [128, 256, 512, 1024, 2048]  # 'resnet18' 各 stage 的输出 channel
        
        elif config.backbone_name == 'resnet50':
            resnet = resnet50()
            if config.If_pretrained:
                resnet.load_state_dict(torch.load('../pre_trained_Resnet/resnet50-19c8e357.pth'))
            stage_channels = [128, 256, 512, 1024, 2048]  # 各 stage 的输出 channel

        self.stage_channels = stage_channels

        if not self.If_backbone_use_Stoch_Depth:
            # 利用 imagenet 预训练层构建 backbone
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, nn.Dropout2d(self.Dropout_Rate_CNN[0]) if self.Dropout_Rate_CNN[0] > 0. else nn.Identity())
            self.maxpool = resnet.maxpool  

            # 这里仿照 TransFuse 尝试给每个 CNN stage 后面加入一个 Dropout2d
            self.layer1, self.layer2, self.layer3, self.layer4 = nn.Sequential(resnet.layer1, nn.Dropout2d(self.Dropout_Rate_CNN[1]) if self.Dropout_Rate_CNN[1] > 0. else nn.Identity()), nn.Sequential(resnet.layer2, nn.Dropout2d(self.Dropout_Rate_CNN[2]) if self.Dropout_Rate_CNN[2] > 0. else nn.Identity()), nn.Sequential(resnet.layer3, nn.Dropout2d(self.Dropout_Rate_CNN[3]) if self.Dropout_Rate_CNN[3] > 0. else nn.Identity()), nn.Sequential(resnet.layer4, nn.Dropout2d(self.Dropout_Rate_CNN[4]) if self.Dropout_Rate_CNN[4] > 0. else nn.Identity())

        else:
            dpr = [x.item() for x in torch.linspace(0, self.Dropout_Rate_CNN[-1], len(stage_channels))]  # stochastic depth decay rule

            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, DropPath(dpr[0]) if dpr[0] > 0. else nn.Identity())
            self.maxpool = resnet.maxpool  

            # 这里仿照 TransFuse 尝试给每个 CNN stage 后面加入一个 Dropout2d
            self.layer1, self.layer2, self.layer3, self.layer4 = nn.Sequential(resnet.layer1, DropPath(dpr[1]) if dpr[1] > 0. else nn.Identity()), nn.Sequential(resnet.layer2, DropPath(dpr[2]) if dpr[2] > 0. else nn.Identity()), nn.Sequential(resnet.layer3, DropPath(dpr[3]) if dpr[3] > 0. else nn.Identity()), nn.Sequential(resnet.layer4, DropPath(dpr[4]) if dpr[4] > 0. else nn.Identity())


        del resnet  # 这里删除变量名，释放内存
        
        # -------------------------------------------------------------------
        # Backbone 中 layer3,layer4 的 4 个 conv 层替换为空洞卷积
        if self.use_dilation:
        # if False:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (2, 2), (2, 2)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (4, 4), (4, 4)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)

        # --------------------------------------------------------------------
        # 在 multi Trans branch 前使用一个 UNet
        if self.If_use_UNet_fusion_stage_features:
            self.UNet_for_stage_features = nn.ModuleList()
            j = 0

            # for i in range(len(stage_channels)-1):
            for i in range(len(stage_channels)-1-self.branch_choose[0]):  # UNet 一直用到 branch_choose 的最高层

                # 要计算一下不同 branch 间的 scale factor
                scale_factor = 2
                
                if i == 0:
                    # 这里直接改为利用 UNet 的同时将 out channel 进行了调整
                    self.UNet_for_stage_features.append(unetUp(stage_channels[len(stage_channels)-2-i], config.branch_in_channels[len(stage_channels)-2-i], scale_factor, self.is_deconv, self.if_sum_fusion, if_need_channel_change = True, in_size_low_resolution = stage_channels[len(stage_channels)-1-i], drop_rate = self.Dropout_Rate_UNet[len(stage_channels)-2-i], If_remove_ReLU = self.If_remove_ReLU, If_remove_Norm=self.If_remove_Norm))
                else:
                    self.UNet_for_stage_features.append(unetUp(stage_channels[len(stage_channels)-2-i], config.branch_in_channels[len(stage_channels)-2-i], scale_factor, self.is_deconv, self.if_sum_fusion, if_need_channel_change = True, in_size_low_resolution = config.branch_in_channels[len(stage_channels)-1-i], drop_rate = self.Dropout_Rate_UNet[len(stage_channels)-2-i], If_remove_ReLU = self.If_remove_ReLU, If_remove_Norm=self.If_remove_Norm))

                j = j+1
            
            # 如果选了 top most branch:
            if self.branch_choose[-1] == (len(self.stage_channels)-1):
                self.top_branch_channel_change = nn.Sequential(
                    nn.Conv2d(stage_channels[len(self.stage_channels)-1], config.branch_in_channels[len(self.stage_channels)-1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(config.branch_in_channels[len(self.stage_channels)-1]) if not self.If_remove_Norm else nn.Identity(),
                    # inplace=True: 对从上层网络nn.Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量。
                    nn.ReLU(inplace=True) if not self.If_remove_ReLU else nn.Identity(),
                    )

        else:
            # --------------------------------------------------------------------
            # Backbone 各 stage 的 feature 统一进行 channel 的调整
            self.Backbone_channel_changes = []   # 顺序： [stage1_feature, stage2_feature, stage3_feature, stage4_feature]
            # for stage_channel in stage_channels:
            # for i in range(len(stage_channels)):
            for i in self.branch_choose:
                self.Backbone_channel_changes.append(nn.Sequential(
                    nn.Conv2d(stage_channels[i], config.branch_in_channels[i], kernel_size=1, bias=False),
                    nn.BatchNorm2d(config.branch_in_channels[i]) if not self.If_remove_Norm else nn.Identity(),
                    # inplace=True: 对从上层网络nn.Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量。
                    nn.ReLU(inplace=True) if not self.If_remove_ReLU else nn.Identity(),
                    ))

            self.Backbone_channel_changes = nn.ModuleList(self.Backbone_channel_changes) 

        # ==================================================================================
        # 构建 position embedding
        if self.If_use_position_embedding:

            if self.name_position_method == 'Absolute':

                # self.position_embedding = []
                self.pos_Drop_BN = nn.ModuleList()

                for i in self.branch_choose:
                    
                    # patch 的 num 即为该 branch 输入的 feature 总像素数
                    num_patches = int((self.img_size / (2 ** (i+1))) ** 2) 

                    # self.position_embedding.append(
                    #     nn.Parameter(torch.zeros(1, num_patches, config.branch_in_channels[i])),
                    #     )

                    position_embedding = nn.Parameter(torch.zeros(1, num_patches, config.branch_in_channels[i]))
                    setattr(self, f"pos_embed{i}", position_embedding)

                    self.pos_Drop_BN.append(nn.Sequential(
                        nn.Dropout(self.Dropout_Rate_Pos[i]) if self.Dropout_Rate_Pos[i] > 0. else nn.Identity(),
                        nn.BatchNorm2d(config.branch_in_channels[i]),
                        ))

            if self.name_position_method == 'Sinusoid':

                self.position_embedding = nn.ModuleList()
                # self.pos_Drop_BN = nn.ModuleList()

                for i in self.branch_choose:        
                    # patch 的 num 即为该 branch 输入的 feature 总像素数
                    num_patches = int((self.img_size / (2 ** (i+1))) ** 2) # int 向下取整

                    self.position_embedding.append(nn.Sequential(
                        Pos_Embed_Sinusoid(config.branch_in_channels[i], num_patches+1),  # 为了防止 num_patches 不够，向上 +1
                        nn.Dropout(self.Dropout_Rate_Pos[i]) if self.Dropout_Rate_Pos[i] > 0. else nn.Identity(),
                        nn.BatchNorm2d(config.branch_in_channels[i]),
                        ))               


        # ==================================================================================
        # 构建多个 Trans 层
        
        # 构建 4 个 decoder branch
        # stage_key_channels = [8, 16, 32, 64]
        self.trans = nn.ModuleList()
        for i in self.branch_choose:

            dpr = [x.item() for x in torch.linspace(0, config.Drop_path_rate_Trans[i], config.branch_depths[i])]  # stochastic depth decay rule

            self.trans.append(BasicLayer_Share_spatial_reduction(

                block_num=config.branch_depths[i],                     # MSA 的层数

                embedding_dim=config.branch_in_channels[i],      # 输入 Transformer 的 channel 个数。
                key_dim=config.branch_key_channels[i],  

                num_heads=config.branch_num_heads[i],

                Spatial_ratio=config.Spatial_ratios[i],       # 控制各个 branch 中整个 self-attention 中输入 feature 的 spatial reduction
                key_value_ratio = config.key_value_ratios[i], # 控制各个 branch 中 self-attention 中 key 和 value 的 spatial reduction
                mlp_ratio=config.mlp_ratios,
                attn_ratio=config.attn_ratios,

                drop=config.Dropout_Rate_Trans,      # Trans 中使用 dropout 
                drop_path=dpr,   # Trans 中使用 drop path：stochastic depth decay

                one_kv_head = config.one_kv_head, 
                share_kv = config.share_kv,
                If_efficient_attention = config.If_efficient_attention,
                If_attention_scale = config.If_attention_scale,

                norm_cfg=config.norm_cfg,
                act_layer=nn.ReLU))  # 将 nn.ReLU6 换成 nn.ReLU。这里要是 nn.ReLU(inplace=True) 的话为啥会报错？
        
        # ==================================================================================
        # 选择 local, global feature fusion 的方式
        if self.If_Local_GLobal_Fuison:
            if self.Local_Global_fusion_method == 'Sum_fusion':
                # ------------------------------------------
                # 对 local 和 global feature 进行 sum 后进行 channel 调整（相当于一个 line fusion）。
                self.channels_change = nn.ModuleList()
                for i in self.branch_choose:
                    self.channels_change.append(nn.Sequential(
                        nn.Conv2d(config.branch_in_channels[i], config.branch_out_channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(config.branch_out_channels) if not self.If_remove_Norm else nn.Identity(),
                        nn.ReLU(inplace=True) if not self.If_remove_ReLU else nn.Identity(),
                        nn.Dropout2d(self.Dropout_Rate_Local_Global_Fusion[i]) if self.Dropout_Rate_Local_Global_Fusion[i] > 0. else nn.Identity()
                        ))

            elif self.Local_Global_fusion_method == 'Attention_Gate':
                # ------------------------------------------
                self.local_global_Fusions = nn.ModuleList()
                self.local_global_Fusions_drop = nn.ModuleList()

                for i in self.branch_choose:
                    # 利用 attention gate 进行 local global feature fusion
                    self.local_global_Fusions.append(GridAttentionBlock2D(in_channels=config.branch_in_channels[i], 
                                                 gating_channels=config.branch_in_channels[i],
                                                 inter_channels=config.branch_in_channels[i]//2, 
                                                 final_out_channels=config.branch_out_channels,
                                                 mode='concatenation_MultiTrans',
                                                 sub_sample_factor = (1,1))
                                                )
                    self.local_global_Fusions_drop.append(
                                                 nn.Dropout2d(self.Dropout_Rate_Local_Global_Fusion[i]) if self.Dropout_Rate_Local_Global_Fusion[i] > 0. else nn.Identity()
                                                 )
                                                
        else:
            # ------------------------------------------
            # 对 global feature 直接进行 channel 调整（相当于一个 line fusion）。
            self.channels_change = nn.ModuleList()
            for i in self.branch_choose:
                self.channels_change.append(nn.Sequential(
                    nn.Conv2d(config.branch_in_channels[i], config.branch_out_channels, kernel_size=1, bias=False),                    
                    nn.BatchNorm2d(config.branch_out_channels) if not self.If_remove_Norm else nn.Identity(),
                    nn.ReLU(inplace=True) if not self.If_remove_ReLU else nn.Identity(),
                    nn.Dropout2d(self.Dropout_Rate_Local_Global_Fusion[i]) if self.Dropout_Rate_Local_Global_Fusion[i] > 0. else nn.Identity()
                    ))

        if not self.If_use_UNet_decoder:
            # ===================================================================================
            if not self.If_direct_upsampling:
                # ------------------------------------------
                # 计算不同 branch 中输出的 feature 间的 offset map 然后用于上采样
                self.stages_offset = nn.ModuleList()
                # stage 1 与 stage 2、stage 2与 stage 3、stage 3 与 stage 4 间的 offset maps
                for i in range(len(self.branch_choose)-1):   
                        self.stages_offset.append(SFNet_warp_grid(config.branch_in_channels[i], config.branch_in_channels[i]//2))

            # ===================================================================================
            # 各 branch 的 features sum 后进行 line fusion
            # self.linear_fuse = ConvModule(
            #     a = config.branch_out_channels,
            #     b = config.branch_out_channels,
            #     ks = 1,
            #     stride = 1,
            #     pad=0, 
            #     dilation=1,
            #     groups = config.branch_out_channels if self.is_dw else 1,
            #     norm_cfg=self.norm_cfg,
            #     act_cfg=dict(type='ReLU')
            # )
            if self.Multi_branch_concat_fusion:
                self.linear_fuse = nn.Sequential(
                    nn.Conv2d(config.branch_out_channels*len(self.branch_choose), config.branch_out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = config.branch_out_channels if self.is_dw else 1, bias=False),
                    nn.BatchNorm2d(config.branch_out_channels),
                    nn.ReLU(inplace=True) if not self.If_remove_ReLU else nn.Identity(),
                    nn.Dropout2d(self.Dropout_Rate_Multi_branch_fusion) if self.Dropout_Rate_Multi_branch_fusion > 0. else nn.Identity(),   
                    )
            else:
                self.linear_fuse = nn.Sequential(
                    nn.Conv2d(config.branch_out_channels, config.branch_out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = config.branch_out_channels if self.is_dw else 1, bias=False),
                    nn.BatchNorm2d(config.branch_out_channels),
                    nn.ReLU(inplace=True) if not self.If_remove_ReLU else nn.Identity(),
                    nn.Dropout2d(self.Dropout_Rate_Multi_branch_fusion) if self.Dropout_Rate_Multi_branch_fusion > 0. else nn.Identity(),
                    )
        
        # 利用 UNet decoder 进行逐步上采样。
        else:
            self.UNet_decoder = nn.ModuleList()
            j = 0
            for i in range(len(self.branch_choose)-1):

                # 要计算一下不同 branch 间的 scale factor
                scale_factor = (self.branch_choose[len(self.branch_choose)-1-j] - self.branch_choose[len(self.branch_choose)-2-j])*2
                
                self.UNet_decoder.append(unetUp(config.branch_out_channels, config.branch_out_channels, scale_factor, self.is_deconv, self.if_sum_fusion, drop_rate = self.Dropout_Rate_UNet[self.branch_choose[len(self.branch_choose)-2-j]], If_remove_ReLU = self.If_remove_ReLU))
                j = j+1

        # ------------------------------------------------------------------------
        # MMSegmenation 中的 seg head
        if config.If_Deep_Supervision:
            
            # 如果采用 Deep_Supervision 中的 seg_head 的话，在 Multi_branch 的 linear_fuse 后，又多用了一个 3x3 的 fusion。而且 linear fuse 后，少了一个 nn.Dropout2d。
            # self.seg_head = nn.Sequential(
            #             nn.Conv2d(config.branch_out_channels, config.branch_out_channels, kernel_size=3, stride=1, padding=1),
            #             nn.BatchNorm2d(config.branch_out_channels),
            #             nn.ReLU(inplace=True),
            #             nn.Dropout2d(self.Dropout_Rate_SegHead),
            #             nn.Conv2d(config.branch_out_channels, classes, kernel_size=1, stride=1, padding=0, bias=True)
            #             )

            self.seg_head = nn.Sequential(
                        # nn.Conv2d(fam_dim, fam_dim, kernel_size=3, stride=1, padding=1),
                        # nn.BatchNorm2d(fam_dim),
                        # nn.ReLU(),
                        # nn.Dropout2d(self.Dropout_Rate_SegHead),   # 这个也相当与在 multi-branch fusion 后的 Dropout     
                        nn.Conv2d(config.branch_out_channels, classes, kernel_size=1, stride=1, padding=0, bias=True)
                        )

            if self.training:

                if not self.If_in_deep_sup:
                    if not self.If_use_UNet_decoder:
                        self.branches_head = nn.ModuleList()
                        for i in self.branch_choose:
                            # self.branches_head.append(nn.Sequential(
                            #     nn.Conv2d(config.branch_out_channels, config.branch_out_channels, kernel_size=3, stride=1, padding=1),
                            #     nn.BatchNorm2d(config.branch_out_channels),
                            #     nn.ReLU(inplace=True),
                            #     nn.Dropout2d(self.Dropout_Rate_SegHead),
                            #     nn.Conv2d(config.branch_out_channels, classes, kernel_size=1, stride=1, padding=0, bias=True)
                            #     ))

                            self.branches_head.append(nn.Sequential(
                                # nn.Conv2d(config.branch_out_channels, config.branch_out_channels, kernel_size=3, stride=1, padding=1),
                                # nn.BatchNorm2d(config.branch_out_channels),
                                # nn.ReLU(inplace=True),
                                nn.Dropout2d(self.Dropout_Rate_SegHead),
                                nn.Conv2d(config.branch_out_channels, classes, kernel_size=1, stride=1, padding=0, bias=True)
                                ))

                    else:
                        # 如果是用 UNet-deoder, branch[0] 是直接作为 final_feature 不需要再计算 aux loss
                        self.branches_head = nn.ModuleList()
                        for i in self.branch_choose[1:]:
                            # self.branches_head.append(nn.Sequential(
                            #     nn.Conv2d(config.branch_out_channels, config.branch_out_channels, kernel_size=3, stride=1, padding=1),
                            #     nn.BatchNorm2d(config.branch_out_channels),
                            #     nn.ReLU(inplace=True),
                            #     nn.Dropout2d(self.Dropout_Rate_SegHead),
                            #     nn.Conv2d(config.branch_out_channels, classes, kernel_size=1, stride=1, padding=0, bias=True)
                            #     ))

                            self.branches_head.append(nn.Sequential(
                                # nn.Conv2d(config.branch_out_channels, config.branch_out_channels, kernel_size=3, stride=1, padding=1),
                                # nn.BatchNorm2d(config.branch_out_channels),
                                # nn.ReLU(inplace=True),
                                nn.Dropout2d(self.Dropout_Rate_SegHead),
                                nn.Conv2d(config.branch_out_channels, classes, kernel_size=1, stride=1, padding=0, bias=True)
                                ))
                else:
                    self.branches_head = nn.ModuleList()
                    for i in self.branch_choose:
                        # self.branches_head.append(nn.Sequential(
                        #     nn.Conv2d(config.branch_in_channels[i], config.branch_out_channels, kernel_size=3, stride=1, padding=1),
                        #     nn.BatchNorm2d(config.branch_out_channels),
                        #     nn.ReLU(inplace=True),
                        #     nn.Dropout2d(self.Dropout_Rate_SegHead),
                        #     nn.Conv2d(config.branch_out_channels, classes, kernel_size=1, stride=1, padding=0, bias=True)
                        #     ))

                        self.branches_head.append(nn.Sequential(
                            # nn.Conv2d(config.branch_in_channels[i], config.branch_out_channels, kernel_size=3, stride=1, padding=1),
                            # nn.BatchNorm2d(config.branch_out_channels),
                            # nn.ReLU(inplace=True),
                            nn.Dropout2d(self.Dropout_Rate_SegHead),
                            nn.Conv2d(config.branch_in_channels[i], classes, kernel_size=1, stride=1, padding=0, bias=True)
                            ))

        else:
            self.seg_head = nn.Sequential(
                        # nn.Conv2d(fam_dim, fam_dim, kernel_size=3, stride=1, padding=1),
                        # nn.BatchNorm2d(fam_dim),
                        # nn.ReLU(),
                        # nn.Dropout2d(self.Dropout_Rate_SegHead),   # 这个也相当与在 multi-branch fusion 后的 Dropout     
                        nn.Conv2d(config.branch_out_channels, classes, kernel_size=1, stride=1, padding=0, bias=True)
                        )

        if self.If_weight_init:
            self._init_weights()
    # ------------------------------------------------------------------------
    # 进行初始化
    def _init_weights(self):
        self.Backbone_channel_changes.apply(init_weights)
        self.trans.apply(init_weights)

        if self.If_Local_GLobal_Fuison:
            if self.If_direct_sum_fusion:
                self.channels_change.apply(init_weights)
            else:
                self.local_global_Fusions.apply(init_weights)
        else:
            self.channels_change.apply(init_weights)

        if not self.If_direct_upsampling:
            self.stages_offset.apply(init_weights)
        
        self.linear_fuse.apply(init_weights)

        if self.If_Deep_Supervision:
            if self.training:
                self.branches_head.apply(init_weights)

        self.seg_head.apply(init_weights)

        # # 断点加载
        # if isinstance(self.pretrained, str):
        #     logger = get_root_logger()
        #     checkpoint = _load_checkpoint(self.pretrained, logger=logger, map_location='cpu')
        #     if 'state_dict_ema' in checkpoint:
        #         state_dict = checkpoint['state_dict_ema']
        #     elif 'state_dict' in checkpoint:
        #         state_dict = checkpoint['state_dict']
        #     elif 'model' in checkpoint:
        #         state_dict = checkpoint['model']
        #     else:
        #         state_dict = checkpoint
        #     self.load_state_dict(state_dict, False)

    # 如果维度不一致的话，要进行 position embedding 的上采样
    def _pos_embed_reshape(self, pos_embed, patches_H, patches_W, H, W):
        
        # 1, HW, C --->  1, H, W, C ---> 1, C, H, W ---> 1, C, HW ---> 1, HW, C
        pos_embed_reshaped = F.interpolate(
                pos_embed.reshape(1, patches_H, patches_W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear", align_corners=True).reshape(1, -1, H * W).permute(0, 2, 1)
        
        return pos_embed_reshaped

    def forward(self, x, y=None):
        x_size = x.size()

        # 把 single channel 的 slice 扩展为 3 通道的
        # 有的数据集是 grey image
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1) # 变成 3 Channel 的输入。在这里啊！！！把 single channel 的 slice 扩展为 3 通道的。
        
        x = self.layer0(x)
        stage0_feature = x

        x = self.maxpool(x)

        x = self.layer1(x)
        stage1_feature = x

        # print(stage1_feature.size())

        x = self.layer2(x)
        stage2_feature = x

        # print(stage2_feature.size())

        x = self.layer3(x)  # 用于计算中间层 loss
        stage3_feature = x

        # print(stage3_feature.size())

        x = self.layer4(x)
        stage4_feature = x

        # print(stage4_feature.size())
      
        # --------------------------------------------------------------
        stage_ouputs = [stage0_feature, stage1_feature, stage2_feature, stage3_feature, stage4_feature]  # 各特征维度 

        # --------------------------------------------------------------------
        # 在 multi Trans branch 前使用一个 UNet
        if self.If_use_UNet_fusion_stage_features:
            i = 0
            # for j in reversed(range(1, len(stage_ouputs))):
            for j in reversed(range(1 + self.branch_choose[0], len(stage_ouputs))):    # UNet 一直用到 branch_choose 的最高层

                stage_ouputs[j-1], channel_changed = self.UNet_for_stage_features[i](stage_ouputs[j-1], stage_ouputs[j])

                # if i == 0:
                #     stage_ouputs[j] = channel_changed  # 最高层的 stage feature 进行 channel change

                i = i+1

            # 对 top stage feature 进行 channel change
            if self.branch_choose[-1] == (len(self.stage_channels)-1):
                stage_ouputs[len(self.stage_channels)-1] = self.top_branch_channel_change(stage_ouputs[len(self.stage_channels)-1])

            # --------------------------------------------------------------
            # 得到各 stage channel 的 dim 调整
            compress_stage_features = []
           
            for i in self.branch_choose:
                compress_stage_features.append(stage_ouputs[i])

                # print(stage_ouputs[i].size())
                               
        else:
            # --------------------------------------------------------------
            # 进行 backbone 各 stage channel 的 dim 调整
            compress_stage_features = []
            # for i in range(len(stage_ouputs)):
            j = 0
            for i in self.branch_choose:
                compress_stage_features.append(self.Backbone_channel_changes[j](stage_ouputs[i]))
                j = j + 1

        # ----------------------------------------------------------------------------------
        # 加入 deep supervison
        if self.training:
            if self.If_Deep_Supervision:

                if self.If_in_deep_sup:   
                    deep_supervison_outs = []
                    j = 0
                    for i in self.branch_choose:
                        branch_pre = self.branches_head[j](compress_stage_features[j])    # 得到各个 branch 的 logist
                        branch_pre = F.interpolate(branch_pre, x_size[2:], mode='bilinear', align_corners=True)  # 上采样
                        deep_supervison_outs.append(branch_pre)
                        
                        j = j+1      

        # ==================================================================================
        # 加入 position embedding

        if self.If_use_position_embedding:
            if self.If_out_side:

                if self.name_position_method == 'Absolute':
                    
                    j = 0
                    for i in self.branch_choose:
                        
                        # ========================================================================
                        # 减少 reshape
                        patches_H = patches_W = int(self.img_size / (2 ** (i+1)))

                        B, C, H, W = (compress_stage_features[j].size())

                        # 进行 position embedding：1, HW, C
                        pos_embed = getattr(self, f"pos_embed{i}") # 进行 position embedding

                        # 1, p_H*p_W, C --> 1, p_H, p_W, C --> 1, C, p_H, p_W 
                        pos_embed = pos_embed.reshape(1, patches_H, patches_W, -1).permute(0, 3, 1, 2)

                        if patches_H*patches_W != H*W:
                            #  1, C, p_H, p_W --> 1, C, H, W
                            pos_embed = F.interpolate(pos_embed, size=(H, W), mode="bilinear", align_corners=True)

                        # B, C, H, W
                        add_pos_embed = pos_embed + compress_stage_features[j]

                        compress_stage_features[j] = self.pos_Drop_BN[j](add_pos_embed)

                        j = j + 1


                if self.name_position_method == 'Sinusoid':

                    j = 0
                    for i in self.branch_choose:

                        # 通过将 position embedding reshape 到与 feature map 相同的维度，然后相加。以减少反复的 reshape，不然似乎训练时间会大大增加。
                        add_pos_embed = self.position_embedding[j](compress_stage_features[j])

                        compress_stage_features[j] = add_pos_embed
                        
                        j = j + 1
        
        # ----------------------------------------------------------------------------------
        # 进行各个 branch 中 local 和 global 的 feature fusion
        branch_outs = []
        j = 0
        for i in self.branch_choose:

            # =======================
            # 加入 position embedding            
            if self.If_use_position_embedding:
                if not self.If_out_side:
                    if self.name_position_method == 'Absolute':
                                  
                        # ========================================================================
                        # 减少 reshape
                        patches_H = patches_W = int(self.img_size / (2 ** (i+1)))

                        B, C, H, W = (compress_stage_features[j].size())

                        # 进行 position embedding：1, HW, C
                        pos_embed = getattr(self, f"pos_embed{i}") # 进行 position embedding

                        # 1, p_H*p_W, C --> 1, p_H, p_W, C --> 1, C, p_H, p_W 
                        pos_embed = pos_embed.reshape(1, patches_H, patches_W, -1).permute(0, 3, 1, 2)

                        if patches_H*patches_W != H*W:
                            #  1, C, p_H, p_W --> 1, C, H, W
                            pos_embed = F.interpolate(pos_embed, size=(H, W), mode="bilinear", align_corners=True)

                        # B, C, H, W
                        add_pos_embed = pos_embed + compress_stage_features[j]

                        add_pos_embed = self.pos_Drop_BN[j](add_pos_embed)


                    if self.name_position_method == 'Sinusoid':
                        # 通过将 position embedding reshape 到与 feature map 相同的维度，然后相加。以减少反复的 reshape，不然似乎训练时间会大大增加。
                        add_pos_embed = self.position_embedding[j](compress_stage_features[j])
                        
        
            if self.If_use_position_embedding:
                if not self.If_out_side:
                    global_out = self.trans[j](add_pos_embed)      # 4 个 Trans branch 
                else:
                    
                    # print(compress_stage_features[j].size())

                    global_out = self.trans[j](compress_stage_features[j])      # 4 个 Trans branch
            else:
                global_out = self.trans[j](compress_stage_features[j])      # 4 个 Trans branch
            
            
            if self.If_Local_GLobal_Fuison:
                if self.Local_Global_fusion_method == 'Sum_fusion':
                    fuse = global_out + compress_stage_features[j]
                    out_fuse = self.channels_change[j](fuse)  # 输出的 channel 进行调整
                    # out_fuse = fuse
                elif self.Local_Global_fusion_method == 'Attention_Gate':
                    # 利用 attention gate 进行每一个 branch 的 global 与 local feature 的 fusion
                    out_fuse = self.local_global_Fusions[j](compress_stage_features[j], global_out)

                    out_fuse = self.local_global_Fusions_drop[j](out_fuse)
                   
            else:
                out_fuse = self.channels_change[j](global_out)  # 对 Global feature 的 channel 进行调整
            
            branch_outs.append(out_fuse)
            j = j+1

        # ----------------------------------------------------------------------------------
        if not self.If_use_UNet_decoder:
            # 进行不同 branch feature 间的 spsample        
            if not self.If_direct_upsampling:
                # --------------------
                # 计算 stage 1 与 stage 2、stage 2 与 stage 3、stage 3 与 stage 4 间的 offset maps
                stages_warp_grid = []
                for i in range(len(branch_outs)-1):
                    stages_warp_grid.append(self.stages_offset[i](branch_outs[i], branch_outs[i+1]))

                # 利用 offset maps 对各 branch 的 fuse_features 进行上采样。渐进式上采样。
                for i in range(1, len(branch_outs)):
                    for k in reversed(range(i)):
                        branch_outs[i] = F.grid_sample(branch_outs[i], stages_warp_grid[k], align_corners=True)  # 利用网格法进行 scor map 的 upsample
            else:
                # ---------------------
                # 直接上采样
                for i in range(1, len(branch_outs)): 
                    branch_outs[i] = F.interpolate(branch_outs[i], (branch_outs[0].size())[2:], mode='bilinear', align_corners=True)  # align_corners=False

            # ----------------------------------------------------------------------------------
            # 各 branch feature 的 fusion
            if self.Multi_branch_concat_fusion:
                sum_feature = torch.cat(branch_outs, 1)   # 进行 channel 维度的 concat fusion
                final_feature = self.linear_fuse(sum_feature)            
            else:
                sum_feature = sum(branch_outs)    
                final_feature = self.linear_fuse(sum_feature)
        
        # 利用 UNet decoder 进行逐步上采样。
        else:
            i = 0
            for j in reversed(range(1, len(branch_outs))):
                
                branch_outs[j-1], channel_changed = self.UNet_decoder[i](branch_outs[j-1], branch_outs[j])
                i = i+1
            final_feature = branch_outs[0]

        # ----------------------------------------------------------------------------------
        # 加入 deep supervison
        if self.training:
            if self.If_Deep_Supervision:

                if not self.If_in_deep_sup:
                    if not self.If_use_UNet_decoder:
                        deep_supervison_outs = []
                        j = 0
                        for i in self.branch_choose:
                            branch_pre = self.branches_head[j](branch_outs[j])    # 得到各个 branch 的 logist
                            branch_pre = F.interpolate(branch_pre, x_size[2:], mode='bilinear', align_corners=True)  # 上采样
                            deep_supervison_outs.append(branch_pre)
                            
                            j = j+1
                    else:
                        # 如果是用 UNet-deoder, branch[0] 是直接作为 final_feature 不需要再计算 aux loss
                        deep_supervison_outs = []
                        j = 0
                        for i in range(1,len(self.branch_choose)):
                            branch_pre = self.branches_head[j](branch_outs[i])    # 得到各个 branch 的 logist
                            branch_pre = F.interpolate(branch_pre, x_size[2:], mode='bilinear', align_corners=True)  # 上采样
                            deep_supervison_outs.append(branch_pre)
                            
                            j = j+1         

        # -----------------------------------------------------------------------------------
        # 计算 prediction map
        logits = self.seg_head(final_feature)

        logits = F.interpolate(logits, x_size[2:], mode='bilinear', align_corners=True)

        if self.training:
            if self.If_Deep_Supervision:

                return logits, deep_supervison_outs
            else:
                return logits       
        else:
            return logits 

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    