""" Module containing backbone network and output heads.

Adopted from https://github.com/traveller59/second.pytorch
    PointPillars fork from SECOND.
    Original code written by Alex Lang and Oscar Beijbom, 2018.
    Licensed under MIT License [see LICENSE].
Modified by Christopher Plachetka and Benjamin Sertolli, 2022.
    Licensed under MIT License [see LICENSE].
"""

import numpy as np
import torch
from torch import nn

from deep_learning.models.helper import ZeroPad3d
from torchplus.nn import Empty, Sequential
from torchplus.tools import change_default_args


class Backbone(nn.Module):
    def __init__(self,
                 configured_element_types,
                 num_vertical_layers_per_type,
                 num_classes_per_type=None,
                 classification_active=False,
                 use_norm=True,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_filters=128,
                 use_3d_backbone=False,
                 use_3d_heads=False,
                 target_shape_z=None,  # [nz, ny, nx]
                 head_num_filters=(),
                 num_map_features_head=0,
                 ):
        """
        Module comprising network backbone and output heads.
        The backbone uses a series of downsampling blocks (multiple convs), whose outputs are upsampled (transposed
        conv) and concatenated to form a large feature tensor. This tensor is then fed into multiple output heads: one
        task and regression head for every configured map element type, respectively. If late map fusion is configured,
        the map feature map is concatenated to the tensor before the output heads.


        Args:
            configured_element_types (list): list of element types (lights, poles, signs)
            num_vertical_layers_per_type (dict): specifies the number of vertical (z-dim) layers for each element type
            num_classes_per_type (dict | None): specifies the number of classes for each element type (used for MDD or
                                                subclass classification)
            classification_active (bool): whether to perform subclass classification (only for element detection)
            use_norm (bool): whether to perform BatchNorm
            layer_nums (tuple | list): (downstream) block configuration, specifying number of layers in each block
            layer_strides (tuple | list): strides in the z-dim applied on the first conv layer of each block; basically
                                          determines whether tensor is downsampled (1: no downsampling, 2: downsampling)
            num_filters (tuple | list): block-wise number of filters, applied on every layer of a block (i.e., first
                                        entry specifies number of filters in every layer of the first block, etc.)
            upsample_strides (tuple | list): strides used for upsampling tensor after every downstream block, so that
                                             every upsampled tensor has the same size (in spatial dims)
            num_upsample_filters (tuple | list): number of filters used for upsampling convs (transposed convs); must
                                                 correspond to number of downsampling blocks
            num_input_filters (int): feature dim size of input tensor
            use_3d_backbone (bool): whether to use 3D convs for backbone (more expensive)
            use_3d_heads (bool): whether to use 3D convs for output heads (except for poles)
            target_shape_z (int | None): size of target output shape in the z dimension (required with 3D backbone)
            head_num_filters (tuple | list): specify additional conv layers applied to every output head (expensive, but
                                             useful for late map fusion)
            num_map_features_head: number of map features to concatenate between backbone and heads (only late fusion)
        """
        super().__init__()

        self._configured_element_types = configured_element_types
        self._classification_active = classification_active
        self._same_resolution_in_out = None
        self.num_vertical_layers_per_type = num_vertical_layers_per_type
        self.num_classes_per_type = num_classes_per_type

        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        if use_3d_heads:
            assert use_3d_backbone
            for element_type in configured_element_types:
                if element_type != 'poles':
                    assert num_vertical_layers_per_type[element_type] == target_shape_z

        self.use_3d_backbone = use_3d_backbone
        self.use_3d_heads = use_3d_heads

        if use_3d_backbone:
            BatchNorm = nn.BatchNorm3d
            ZeroPad = ZeroPad3d
            ConvDown = nn.Conv3d
            Conv = nn.Conv3d
            ConvTranspose = nn.ConvTranspose3d
            BatchNormUp = nn.BatchNorm3d
            Block = Sequential
        else:
            BatchNorm = nn.BatchNorm2d
            ZeroPad = nn.ZeroPad2d
            ConvDown = nn.Conv2d
            Conv = nn.Conv2d
            ConvTranspose = nn.ConvTranspose2d
            BatchNormUp = nn.BatchNorm2d
            Block = Sequential

        if use_3d_heads:
            ConvHead = {e_type: nn.Conv3d for e_type in configured_element_types}
            BatchNormHead = {e_type: nn.BatchNorm3d for e_type in configured_element_types}
        else:
            ConvHead = {e_type: nn.Conv2d for e_type in configured_element_types}
            BatchNormHead = {e_type: nn.BatchNorm2d for e_type in configured_element_types}
        ConvHead['poles'] = nn.Conv2d
        BatchNormHead['poles'] = nn.BatchNorm2d

        if use_norm:
            BatchNorm = change_default_args(eps=1e-3, momentum=0.01)(BatchNorm)
            BatchNormUp = change_default_args(eps=1e-3, momentum=0.01)(BatchNormUp)
            for e_type in configured_element_types:
                BatchNormHead[e_type] = change_default_args(eps=1e-3, momentum=0.01)(BatchNormHead[e_type])
            bias = False
        else:
            BatchNorm = Empty
            bias = True

        concat_filter_inputs = [0, 0, 0, 0]

        # Down-sampling blocks
        input_filters = [num_input_filters] + num_filters
        for i in range(len(layer_nums)):
            # Create initial down-sampling layer
            block = Block(
                ZeroPad(1),
                ConvDown(input_filters[i], num_filters[i], 3, stride=layer_strides[i], bias=bias),
                BatchNorm(num_filters[i]),
                nn.ReLU(),
            )

            # Create following "consistent" layers (i.e., no downsampling)
            for _ in range(layer_nums[i]):
                block.add(Conv(num_filters[i], num_filters[i], 3, padding=1, bias=bias))
                block.add(BatchNorm(num_filters[i]))
                block.add(nn.ReLU())

            block_name = f"down_{i+1}"
            self.__setattr__(block_name, block)

        # Up-sampling blocks
        for i in range(len(layer_nums)):
            deconv = Block(
                ConvTranspose(
                    num_filters[i] + concat_filter_inputs[i + 1],
                    num_upsample_filters[i],
                    upsample_strides[i],
                    stride=upsample_strides[i], bias=bias),
                BatchNormUp(num_upsample_filters[i]),
                nn.ReLU(),
            )

            block_name = f"up_{i + 1}"
            self.__setattr__(block_name, deconv)

        # Set num_head_filters as input to the network headers
        num_concat_up_filters = num_upsample_filters[0] + num_upsample_filters[1] + num_upsample_filters[2]
        num_concat_up_filters += num_map_features_head

        # Heads
        if use_3d_backbone and not use_3d_heads:
            num_head_filters = num_concat_up_filters * target_shape_z  # size z-dim
        else:
            num_head_filters = num_concat_up_filters

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.heads_task = nn.ModuleDict()
        self.heads_reg = nn.ModuleDict()
        self.reg_features_lut = {
            'poles': 4,  # x, y, z, diameter
            'signs': 7,  # x, y, z, width, height, yaw_sin, yaw_cos
            'lights': 7
        }

        for element_type in configured_element_types:
            # Set up element type-specific head parameters
            num_classes = 1
            if classification_active:
                num_classes = num_classes_per_type[element_type]

            num_layers = num_vertical_layers_per_type[element_type]

            num_classification_features = num_classes
            num_regression_features = self.reg_features_lut[element_type]

            if not use_3d_heads:
                num_classification_features *= num_layers
                num_regression_features *= num_layers

            if element_type == 'poles' and use_3d_heads:  # for poles always use 2d heads
                num_head_input = num_head_filters * target_shape_z
            else:
                num_head_input = num_head_filters

            num_head_filters_all = [num_head_input] + head_num_filters
            conv_task = Block()
            conv_reg = Block()

            # Additional layers for each head (if specified)
            for i in range(len(head_num_filters)):
                conv_task.add(ConvHead[element_type](num_head_filters_all[i], num_head_filters_all[i + 1], 3, padding=1,
                                                     bias=bias))
                conv_task.add(BatchNormHead[element_type](num_head_filters_all[i + 1]))
                conv_task.add(nn.ReLU())

                conv_reg.add(ConvHead[element_type](num_head_filters_all[i], num_head_filters_all[i + 1], 3, padding=1,
                                                    bias=bias))
                conv_reg.add(BatchNormHead[element_type](num_head_filters_all[i + 1]))
                conv_reg.add(nn.ReLU())

            # Basic head layers
            conv_task.add(ConvHead[element_type](num_head_filters_all[-1], num_classification_features, 1, bias=bias))
            conv_reg.add(ConvHead[element_type](num_head_filters_all[-1], num_regression_features, 1, bias=bias))

            self.heads_task[element_type] = conv_task
            self.heads_reg[element_type] = conv_reg

    def forward(self, x, map_fm=None):

        # Backbone network
        x = self.down_1(x)
        up1 = self.up_1(x)

        x = self.down_2(x)
        up2 = self.up_2(x)

        x = self.down_3(x)
        up3 = self.up_3(x)

        x = torch.cat([up1, up2, up3], dim=1)

        # Merge with map fm (if given)
        if map_fm is not None:
            map_fm = map_fm.permute(0, 4, 3, 2, 1)  # [BS, x, y, z, elem_type] -> [BS, elem_type, z, y, x]
            x = torch.cat((x, map_fm), dim=1)

        if self.use_3d_backbone and not self.use_3d_heads:  # merge z dim into feature dim
            N, C, D, H, W = x.shape
            x = x.view(N, C * D, H, W)

        # Detection head
        # x: [N, C, y(H), x(W)]
        out_dict = {}
        for element_type in self._configured_element_types:
            # element_type = 'lights'
            # Head setup for element type
            out_dict[element_type] = {}
            num_layers = self.num_vertical_layers_per_type[element_type]
            num_classes = self.num_classes_per_type[element_type]
            num_reg_features = self.reg_features_lut[element_type]

            if self.use_3d_heads and element_type == 'poles':
                N, C, D, H, W = x.shape
                x_heads = x.view(N, C * D, H, W)
            else:
                x_heads = x

            # Process output heads for element type
            out_task = self.heads_task[element_type](x_heads)
            out_task = self.sigmoid(out_task)

            out_reg = self.heads_reg[element_type](x_heads)

            if self.use_3d_heads and element_type != 'poles':
                out_task = out_task.permute(0, 1, 2, 4, 3)  # [b, c, nl, ny, nx] -> [b, c, nl, nx, ny]
                out_reg = out_reg.permute(0, 1, 2, 4, 3)
            else:  # 2d heads (or poles)
                # Reverse reversed order during voxelization ([b, c, nx, ny)]
                out_task = out_task.permute(0, 1, 3, 2)
                out_reg = out_reg.permute(0, 1, 3, 2)
                # Reshape to vertical layers
                B, _, X, Y = out_task.shape
                out_task = torch.reshape(out_task, (B, num_classes, num_layers, X, Y))
                out_reg = torch.reshape(out_reg, (B, num_reg_features, num_layers, X, Y))

            out_dict[element_type]['task'] = out_task
            out_dict[element_type]['reg'] = out_reg

        return out_dict
