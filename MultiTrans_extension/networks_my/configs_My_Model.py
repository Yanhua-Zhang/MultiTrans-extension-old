import ml_collections

def get_My_Model_V10_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()

    # -------------------------------------------------
    config.branch_choose = [1, 2, 3, 4]             # 将 multi-branch 改为可选。
    config.branch_key_channels = [32, 32, 32, 32, 32]   # [4, 8, 16, 32]。 4 个 branch 中 MSA 的 key/value 的 channel 数
    config.branch_in_channels = [256, 256, 256, 256, 256]              # 从各个 branch 中输入的 channel 个数
    config.branch_out_channels = 256              # 进行各 branch feature fusion 后的 channel 个数

    config.branch_depths = [5, 5, 5, 5, 5]      # 每个 branch 的 MSA 的层数
    config.branch_num_heads = [8, 8, 8, 8, 8]   # 每个 branch 的 MSA head 的个数

    config.Spatial_ratios = [1, 1, 1, 1, 1]         # 控制各个 branch 中整个 self-attention 中输入 feature 的 spatial reductio
    config.key_value_ratios = [1, 1, 1, 1, 1]        # 控制各个 branch 中 self-attention 中 key 和 value 的 spatial reduction

    config.attn_ratios=2      # key_dim 乘以 attn_ratios 是单个 value 的 channel 数
    config.mlp_ratios=2        # 用于计算 MLP 中隐含层的 channel 数

    config.Drop_path_rate_Trans = [0.1, 0.1, 0.1, 0.1, 0.1]  # 是否在 Trans 中使用 drop_path
    config.Dropout_Rate_Trans = 0
    config.Dropout_Rate_SegHead = 0.1  # 这个是 Deep supervision head 中的 dropout
    config.Dropout_Rate_Local_Global_Fusion = [0, 0, 0, 0, 0]
    config.Dropout_Rate_Multi_branch_fusion = 0.1
    config.Dropout_Rate_CNN = [0, 0.3, 0.3, 0.3, 0.3]
    # config.Dropout_Rate_CNN = 0.2
    config.If_backbone_use_Stoch_Depth = False  # backbone 是否使用 Stoch Depth
    config.Dropout_Rate_UNet = [0, 0, 0, 0, 0]
    config.Dropout_Rate_Pos = [0, 0, 0, 0, 0]

    config.one_kv_head = True
    config.share_kv = True
    config.If_efficient_attention = True
    config.If_attention_scale = False

    config.If_use_position_embedding = True
    config.name_position_method = 'Sinusoid'
    config.If_out_side = True

    # -------------------------------------------------
    config.If_Local_GLobal_Fuison = True
    config.Local_Global_fusion_method = 'Attention_Gate'   # 'Sum_fusion' 或 'Attention_Gate'   # 选择 local global feature fusion 的方式
    
    config.If_direct_upsampling = True      # 选择 branch feature 间 upsample 的方式
    config.is_dw = False                    # line fusion layer 是否使用 depth-wise conv
    config.Multi_branch_concat_fusion = False

    config.If_use_UNet_decoder = False    # 是否使用 UNet decoder
    config.is_deconv = False              # UNet decoder 是否使用 Trans Conv
    config.if_sum_fusion = True           # UNet decoder 是否由 concat 改为 sum fusion

    config.If_Deep_Supervision = True
    config.If_in_deep_sup = True

    config.If_remove_ReLU = False
    config.If_remove_Norm = False

    # -------------------------------------------------
    config.backbone_name='resnet50_Deep'
    config.use_dilation=False
    config.is_dw = False
    config.If_pretrained = True

    config.If_use_UNet_fusion_stage_features = True  # backbone 后面是否加一个 UNet 

    # -------------------------------------------------
    config.norm_cfg=dict(type='BN', requires_grad=True)
    # config.act_layer=nn.ReLU6
    config.If_weight_init = False

    # -------------------------------------------------
    config.version = 'V10' 

    return config



