
# model settings
bn_norm_cfg = dict(type='SyncBN', requires_grad=True)
gn_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained='torchvision://resnet18',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        frozen_stages=-1,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=bn_norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
    ),
    decode_head=dict(
        type='GAMRFHead',
        # in_channels=(2048, 1024, 512, 256),
        # channels=256,
        # skip_channels=(1024, 512, 256),
        in_channels=(512, 256, 128, 64),
        channels=64,
        skip_channels=(256, 128, 64),
        num_blocks=2,
        flexible=False,
        scales=4,
        dilations=(1, 3, 5),
        order=('conv', 'norm', 'act'),
        dropout_ratio=0.1,
        num_classes=2,
        in_index=(0, 1, 2, 3),
        input_transform="multiple_select",
        norm_cfg=bn_norm_cfg,
        align_corners=True,
        upsample_cfg=dict(scale_factor=2,
                          mode='bilinear',
                          align_corners=True),
        loss_decode=dict(type='DiceLoss', exponent=2),
        att_cfg=dict(style="ours",
                     ca_ratio=1,
                     sa_kernel_size=7,
                     bn=bn_norm_cfg,
                     ca_act=dict(type='ReLU'),
                     sigmoid_act=dict(type='Sigmoid'))
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))