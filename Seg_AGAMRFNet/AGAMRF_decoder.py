import math
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..utils.make_divisible import make_divisible
from mmseg.ops import resize


class ChannelAttention(nn.Module):
    def __init__(self,
                 channels,
                 cfg={"style": "ours",
                      "ca_ratio": 1,
                      "sa_kernel_size": 7,
                      "bn": dict(type='BN', requires_grad=True),
                      "ca_act": dict(type='ReLU'),
                      "sigmoid_act": dict(type='HSigmoid', bias=3.0, divisor=6.0)}):
        super(ChannelAttention, self).__init__()
        self.cfg = cfg
        if cfg["style"] == "ours":
            self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
            self.conv1 = ConvModule(
                in_channels=channels,
                out_channels=make_divisible(channels // cfg["ca_ratio"], 8),
                kernel_size=1,
                stride=1,
                norm_cfg=cfg["bn"],
                act_cfg=cfg["ca_act"])
            self.conv2 = ConvModule(
                in_channels=make_divisible(channels // cfg["ca_ratio"], 8),
                out_channels=channels,
                kernel_size=1,
                stride=1,
                act_cfg=cfg["sigmoid_act"])
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.conv1 = ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                act_cfg=cfg["ca_act"])
            self.conv2 = ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                act_cfg=cfg["sigmoid_act"])

    def transformer_pool(self, x): # spatial_pool
        batch, channel, height, width = x.size()
        input_x = x

        input_x = input_x.view(batch, channel, height * width)# [N, C, H * W]
        input_x = input_x.unsqueeze(1)# [N, 1, C, H * W]

        context_mask = self.conv_mask(x)# [N, 1, H, W]
        context_mask = context_mask.view(batch, 1, height * width)# [N, 1, H * W]
        context_mask = self.softmax(context_mask) # [N, 1, H * W]
        context_mask = context_mask.unsqueeze(-1)# [N, 1, H * W, 1]

        context = torch.matmul(input_x, context_mask)# [N, 1, C, 1]
        context = context.view(batch, channel, 1, 1)# [N, C, 1, 1]

        return context # [N, C, 1, 1]

    def forward(self, x):
        if self.cfg["style"] == "ours":
            output = self.transformer_pool(x)
            output = self.conv1(output)
            output = self.conv2(output)
        else:
            output_avg = self.conv1(self.avg_pool(x))
            output_max = self.conv1(self.max_pool(x))
            output = self.conv2(output_avg + output_max)
        return x * output

class SpatialAttention(nn.Module):
    def __init__(self,
                 channels,
                 cfg={"style": "ours",
                      "ca_ratio": 1,
                      "sa_kernel_size": 3,
                      "bn": dict(type='BN', requires_grad=True),
                      "ca_act": dict(type='ReLU'),
                      "sigmoid_act": dict(type='HSigmoid', bias=3.0, divisor=6.0)}):
        super(SpatialAttention, self).__init__()
        self.cfg = cfg
        if cfg["style"] == "ours":
            self.conv = ConvModule(
                in_channels=channels,
                out_channels=1,
                kernel_size=cfg["sa_kernel_size"],
                padding=cfg["sa_kernel_size"] // 2,
                stride=1,
                norm_cfg=cfg["bn"],
                act_cfg=cfg["sigmoid_act"])
        else:
            self.conv = ConvModule(
                in_channels=2,
                out_channels=1,
                kernel_size=cfg["sa_kernel_size"],
                padding=cfg["sa_kernel_size"] // 2,
                stride=1,
                act_cfg=cfg["sigmoid_act"])

    def forward(self, x):
        if self.cfg["style"] == "ours":
            output = self.conv(x)
        else:
            output_avg = torch.mean(x, dim=1, keepdim=True)
            output_max, _ = torch.max(x, dim=1, keepdim=True)
            output = torch.cat([output_avg, output_max], dim=1)
            output = self.conv(output)

        return x * output

class GMARF_Module(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 skip_inplanes,
                 first_stage=False,
                 scales=4,
                 kernel_size=3,
                 dilations=(1, 3, 5),
                 dilation_index=False,
                 upsample_cfg=dict(
                     scale_factor=2,
                     mode='bilinear',
                     align_corners=False),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type="ReLU"),
                 order=('conv', 'norm', 'act'),
                 flexible=False,
                 att_cfg=None,
                 # soft_threshold_cfg=None,
                 # input_size=80
                 ):
        """Basic block for Res2Net.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(GMARF_Module, self).__init__()
        assert scales > 1, 'scales must be greater than 1 .'
        width = int(math.floor(planes / scales))
        self.first_stage = first_stage
        self.scales = scales
        self.width = width
        self.flexible = flexible
        self.att_cfg = att_cfg
        self.upsample_cfg = upsample_cfg
        self.dilation_index = dilation_index

        if self.first_stage:
            self.identity_downdim = \
                ConvModule(
                    planes + skip_inplanes,
                    planes,
                    kernel_size=1,
                    stride=1,
                    # conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order
                )

            # upsample = nn.Upsample(**upsample_cfg)
            conv = ConvModule(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=1,
                    # conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order
                )

            # if dilation_index:
            #     self.upsample = nn.Sequential(conv)
            # else:
            #     # self.upsample = nn.Sequential(upsample, conv)
            #     self.upsample = nn.Sequential(conv)
            self.upsample_conv = nn.Sequential(conv)

        self.conv_1 = ConvModule(
                (planes + skip_inplanes) if first_stage else inplanes,
                scales * width,
                kernel_size=1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                order=order) # shit-stirrer & regular (inplanes --> scales * width)

        self.convs_2 = nn.ModuleList()
        self.convs_2_dila = nn.ModuleList()
        for i in range(scales - 1):
            self.convs_2.append(
                ConvModule(
                    width,
                    width,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            )

            self.convs_2_dila.append(
                ConvModule(
                    width,
                    width,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=dilations[i],
                    dilation=dilations[i],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order)
            )

        self.conv_3 = ConvModule(
                scales * width,
                scales * width if flexible else planes,
                kernel_size=1,
                stride=1,
                # conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                order=order)# shit-stirrer

        if not flexible:
            self.convs_4 = nn.ModuleList()
            self.convs_4_dila = nn.ModuleList()
            for i in range(scales - 1):
                self.convs_4.append(
                    ConvModule(
                        width,
                        width,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        order=order))

                self.convs_4_dila.append(
                    ConvModule(
                        width,
                        width,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=dilations[i],
                        dilation=dilations[i],
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        order=order))

                self.conv_5 = ConvModule(
                        scales * width,
                        planes,
                        kernel_size=1,
                        stride=1,
                        # conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        order=order) # shit-stirrer & regular (scales * width --> planes)
        if att_cfg is not None:
            self.CA = ChannelAttention(planes, att_cfg)
            self.SA = SpatialAttention(planes, att_cfg)

    def forward(self, skip, x):
        """Forward function."""

        if self.first_stage:
            if self.dilation_index:
                x = self.upsample_conv(x)
            else:
                x = nn.Upsample(size=skip.size()[2:],
                                mode=self.upsample_cfg["mode"],
                                align_corners=self.upsample_cfg["align_corners"])(x)
                x = self.upsample_conv(x)

            x = torch.cat((x, skip), dim=1)
            identity = self.identity_downdim(x)
        else:
            identity = x

        out = self.conv_1(x) # shit-stirrer & regular (inplanes --> scales * width)

        spx = torch.split(out, self.width, 1)
        sp = spx[0].contiguous()
        out = sp
        for i in range(0, self.scales - 1):
            sp = sp + spx[i + 1]
            sp = self.convs_2[i](sp.contiguous())
            sp_dila = self.convs_2_dila[i](sp)
            out = torch.cat((out, sp_dila), 1)
        # spx = torch.split(out, self.width, 1)
        # sp = spx[0].contiguous()
        # out = sp
        # for i in range(1, self.scales): # 1, 2, 3
        #     if i == 1:
        #         sp = spx[i]
        #     else:
        #         sp = spx[i] + sp
        #     sp = self.convs_2[i - 1](sp.contiguous())
        #     # sp_dila = self.convs_2_dila[i - 1](sp)
        #     out = torch.cat((out, sp), 1)

        out = self.conv_3(out) # shit-stirrer

        if not self.flexible:
            spx = torch.split(out, self.width, 1)
            sp = spx[0].contiguous()
            out = sp
            for i in range(0, self.scales - 1):
                sp = sp + spx[i + 1]
                sp = self.convs_4[i](sp.contiguous())
                sp_dila = self.convs_4_dila[i](sp)
                out = torch.cat((out, sp_dila), 1)
            # spx = torch.split(out, self.width, 1)
            # sp = spx[0].contiguous()
            # out = sp
            # for i in range(1, self.scales):  # 1, 2, 3
            #     if i == 1:
            #         sp = spx[i]
            #     else:
            #         sp = spx[i] + sp
            #     sp = self.convs_4[i - 1](sp.contiguous())
            #     # sp_dila = self.convs_4_dila[i - 1](sp)
            #     out = torch.cat((out, sp), 1)

            out = self.conv_5(out) # shit-stirrer & regular (scales * width --> planes)

        if self.att_cfg is not None:
            out = self.CA(out)
            out = self.SA(out)

        out = identity + out

        return out

class GMARF_Block(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 skip_inplanes,
                 num_blocks=2,
                 dilation_index=False,
                 upsample_cfg=dict(
                     scale_factor=2, mode='bilinear', align_corners=False),
                 scales=4,
                 dilations=(1, 3, 5),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 order=('conv', 'norm', 'act'),
                 flexible=False,
                 att_cfg=None,
                 ):
        super(GMARF_Block, self).__init__()
        self.convs_stage = nn.ModuleList()
        for i in range(num_blocks):
            self.convs_stage.append(
                GMARF_Module(
                    inplanes if i == 0 else planes,
                    planes,
                    skip_inplanes,
                    first_stage=True if i == 0 else False,
                    dilations=dilations,
                    dilation_index=dilation_index,
                    upsample_cfg=upsample_cfg,
                    scales=scales,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    order=order,
                    flexible=flexible,
                    att_cfg=att_cfg
                ))

    def forward(self, skip, x):
        """Forward function."""

        out = self.convs_stage[0](skip, x)
        for i in range(1, len(self.convs_stage)):
            out = self.convs_stage[i](skip, out)

        return out

@HEADS.register_module()
class GAMRFHead(BaseDecodeHead):

    def __init__(self,
                 # in_channels=(512, 256, 128, 64),
                 mid_channels=None, #(256, 128, 64)
                 skip_channels=(256, 128, 64),
                 num_blocks=2,
                 dilations_index=(False, False, False, False),
                 upsample_cfg=dict(
                     scale_factor=2, mode='bilinear', align_corners=False),
                 scales=4,
                 dilations=(1, 3, 5),
                 order=('conv', 'norm', 'act'),
                 flexible=False,
                 att_cfg=None,
                 **kwargs):
        super(GAMRFHead, self).__init__(**kwargs)

        if mid_channels is None:
            mid_channels = self.in_channels[1:]

        assert mid_channels[-1] == self.channels
        assert num_blocks >= 0
        assert scales == len(dilations) + 1
        assert len(skip_channels) < len(self.in_index)
        self.num_blocks = num_blocks
        self.stages = nn.ModuleList()

        for i in range(len(skip_channels)):
            self.stages.append(
                GMARF_Block(
                    inplanes=self.in_channels[i],
                    planes=mid_channels[i],
                    skip_inplanes=skip_channels[i],
                    num_blocks=num_blocks,
                    dilation_index=dilations_index[i],
                    upsample_cfg=upsample_cfg,
                    scales=scales,
                    dilations=dilations,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    order=order,
                    flexible=flexible,
                    att_cfg=att_cfg,
                ))

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        x = inputs[::-1]
        output = x[0]

        for i in range(len(self.stages)):
            output = self.stages[i](x[i + 1], output)

        output = self.cls_seg(output)

        return output
