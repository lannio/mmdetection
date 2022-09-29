# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16


from ..builder import NECKS


@NECKS.register_module()
class ATTN_7_FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:        
        type='FPN',
        in_channels=[256, 512, 1024, 2048], # 骨架多尺度特征图输出通道
        out_channels=256,  # 增强后通道输出 # FPN 输出的每个尺度输出特征图通道 # 说明了 5 个输出特征图的通道数都是 256
        start_level=1, # 从输入多尺度特征图的第几个开始计算 # 说明虽然输入是 4 个特征图，但是实际上 FPN 中仅仅用了后面三个
        add_extra_convs='on_input',  # 输出num_outs个多尺度特征图 # 额外输出层的特征图来源 #  说明额外输出的 2 个特征图的来源是骨架网络输出，而不是 FPN 层本身输出又作为后面层的输
        num_outs=5),  # FPN 输出特征图个数 #说明 FPN 模块虽然是接收 3 个特征图，但是输出 5 个特征图
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,   # [256, 512, 1024, 2048]
                 out_channels,  # 256
                 num_outs,      # 3
                 start_level=0, # 1
                 end_level=-1,
                 add_extra_convs=False,     # 'on_input'
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(ATTN_7_FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        

        self.w_relu = nn.ReLU(inplace=True) 
        self.number_f=32
        self.w_conv1 = nn.Conv2d(256,self.number_f,3,1,1,bias=True)
        self.w_conv2 = nn.Conv2d(self.number_f,self.number_f,3,1,1,bias=True) 
        # self.w_conv3 = nn.Conv2d(self.number_f*2,self.number_f,3,1,1,bias=True) 
        # self.w_conv4 = nn.Conv2d(self.number_f*2,self.number_f,3,1,1,bias=True) 
        # self.w_conv5 = nn.Conv2d(self.number_f*2,self.number_f,3,1,1,bias=True) 
        self.w_conv7 = nn.Conv2d(self.number_f*2,64,3,1,1,bias=True) 
        self.w_conv8 = nn.Conv2d(self.number_f*2,64,3,1,1,bias=True) 

        self.w_conv9 = nn.Conv2d(64,256,3,1,1,bias=True) 
        self.w_conv10 = nn.Conv2d(64,256,3,1,1,bias=True) 

        self.p_top = ConvModule(
                    256,
                    256,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=None,
                    act_cfg=None,
                    inplace=False)
        self.p_3_conv = ConvModule(
                    256,
                    16,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=None,
                    act_cfg=None,
                    inplace=False)
        self.p_4_conv = ConvModule(
                    256,
                    16,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=None,
                    act_cfg=None,
                    inplace=False)
        self.p_5_conv = ConvModule(
                    256,
                    16,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=None,
                    act_cfg=None,
                    inplace=False)
        # self.p_6_conv = ConvModule(
        #             256,
        #             16,
        #             1,
        #             stride=1,
        #             padding=0,
        #             conv_cfg=None,
        #             norm_cfg=None,
        #             act_cfg=None,
        #             inplace=False)
        # self.p_7_conv = ConvModule(
        #             256,
        #             16,
        #             1,
        #             stride=1,
        #             padding=0,
        #             conv_cfg=None,
        #             norm_cfg=None,
        #             act_cfg=None,
        #             inplace=False)
        self.level_weight_conv = ConvModule(
                    48,
                    3,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=None,
                    act_cfg=None,
                    inplace=False)
        

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals_pre = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals=[]
        laterals_r=[]
        laterals_n=[]
        for i in range(3):
            x= laterals_pre[i]
            x1 = self.w_conv1(x)
            x2 = self.w_conv2(x1)
            # x3 = self.w_relu(self.w_conv3(torch.cat([x1,x2],1)))
            # x4 = self.w_relu(self.w_conv4(torch.cat([x2,x3],1)))
            # x5 = self.w_relu(self.w_conv5(torch.cat([x3,x4],1)))
            x_r = self.w_conv7(torch.cat([x1,x2],1))
            x_n = self.w_conv8(torch.cat([x1,x2],1))
            # r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 1, dim=1)
            # n1,n2,n3,n4,n5,n6,n7,n8 = torch.split(x_n, 1, dim=1)
            x_r_t = self.w_conv9(x_r)
            x_n_t = self.w_conv10(x_n)

            # x = (x-n1) * (r1+1)
            # x = (x-n2) * (r2+1)
            # x = (x-n3) * (r3+1)
            enhance_image_1 = (x-x_n_t) * (x_r_t)
            # x = (enhance_image_1-n5) * (r5+1)
            # x = (x-n6) * (r6+1)
            # x = (x-n7) * (r7+1)
            # enhance_image = (x-n8) * (r8+1)
            
            # r = torch.cat([(r1+1),(r2+1),(r3+1),(r4+1)],1)
            # n = torch.cat([n1,n2,n3,n4],1)
            laterals.append(enhance_image_1) #torch.Size([16, 256, 80, 80]) //40 20
            # laterals_r.append(r) # torch.Size([16, 8, 80, 80])
            # laterals_n.append(n) # torch.Size([16, 8, 80, 80])
        
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        # outs=[p_0_3,          p_1_4,      p_2_5,      p_3_6,      p_4_7]
        #       60*60*256(8)    40*40*256   20*20*256   10*10*256   5*5*256
        p_0_3, p_1_4, p_2_5  = outs
        p_0_size = p_0_3.shape[2:]
        p_1_size = p_1_4.shape[2:]
        p_2_size = p_2_5.shape[2:]
        # p_3_size = p_3_6.shape[2:]
        # p_4_size = p_4_7.shape[2:]

        p_0_3 = p_0_3
        p_1_3 = self.p_top(p_0_3)
        p_2_3 = self.p_top(p_1_3)
        # p_3_3 = self.p_top(p_2_3)
        # p_4_3 = self.p_top(p_3_3)

        p_1_4 = p_1_4
        p_2_4 = self.p_top(p_1_4)
        # p_3_4 = self.p_top(p_2_4)
        # p_4_4 = self.p_top(p_3_4)
        p_0_4 = F.interpolate(p_1_4, size=p_0_size, **self.upsample_cfg)

        p_2_5 = p_2_5
        # p_3_5 = self.p_top(p_2_5)
        # p_4_5 = self.p_top(p_3_5)
        p_1_5 = F.interpolate(p_2_5, size=p_1_size, **self.upsample_cfg)
        p_0_5 = F.interpolate(p_1_5, size=p_0_size, **self.upsample_cfg)

        # p_3_6 = p_3_6
        # p_4_6 = self.p_top(p_3_6)
        # p_2_6 = F.interpolate(p_3_6, size=p_2_size, **self.upsample_cfg)
        # p_1_6 = F.interpolate(p_2_6, size=p_1_size, **self.upsample_cfg)
        # p_0_6 = F.interpolate(p_1_6, size=p_0_size, **self.upsample_cfg)

        # p_4_7 = p_4_7
        # p_3_7 = F.interpolate(p_4_7, size=p_3_size, **self.upsample_cfg)
        # p_2_7 = F.interpolate(p_3_7, size=p_2_size, **self.upsample_cfg)
        # p_1_7 = F.interpolate(p_2_7, size=p_1_size, **self.upsample_cfg)
        # p_0_7 = F.interpolate(p_1_7, size=p_0_size, **self.upsample_cfg)

        # p_features=[[p_0_3,p_0_4,p_0_5,p_0_6,p_0_7],
        #             [p_1_3,p_1_4,p_1_5,p_1_6,p_1_7],
        #             [p_2_3,p_2_4,p_2_5,p_2_6,p_2_7],
        #             [p_3_3,p_3_4,p_3_5,p_3_6,p_3_7],
        #             [p_4_3,p_4_4,p_4_5,p_4_6,p_4_7],]
        p_features=[[p_0_3,p_0_4,p_0_5],
            [p_1_3,p_1_4,p_1_5],
            [p_2_3,p_2_4,p_2_5]]

        out_feature=[]
        for i in range(3):
                level_weight_3 = self.p_3_conv(p_features[i][0])
                level_weight_4 = self.p_4_conv(p_features[i][1])
                level_weight_5 = self.p_5_conv(p_features[i][2])

                
                level_weight_concat = torch.cat((level_weight_3, level_weight_4, level_weight_5),1)
                level_weight = self.level_weight_conv(level_weight_concat)
                level_weight = F.softmax(level_weight, dim=1)

                level = p_features[i][0] * level_weight[:,0:1,:,:]+\
                        p_features[i][1] * level_weight[:,1:2,:,:]+\
                        p_features[i][2] * level_weight[:,2:,:,:]
                out_feature.append(level) #(torch.Size([16, 256, 80, 80]), torch.Size([16, 256, 40, 40]), torch.Size([16, 8, 20, 20]))
        
        # laterals_r.append(r) # torch.Size([16, 8, 80, 80])
        # laterals_n.append(n) # torch.Size([16, 8, 80, 80])

        # for i in range(3):
        #     out_feature[i]= torch.cat([out_feature[i],laterals_r[i],laterals_n[i]],1) #torch.Size([16, 272, 80, 80]) 40 20

        return tuple(out_feature)
