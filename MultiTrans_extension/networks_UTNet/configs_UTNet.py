import ml_collections

def get_My_Model_V10_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    
    config.backbone_name='resnet50'
   
    config.version = 'without_Pretrain'
    

    return config



