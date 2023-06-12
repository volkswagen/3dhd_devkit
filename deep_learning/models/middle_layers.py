""" Collection of middle-layer and voxel scatter classes (torch modules).

Adopted from https://github.com/traveller59/second.pytorch
    PointPillars fork from SECOND.
    Original code written by Alex Lang and Oscar Beijbom, 2018.
    Licensed under MIT License [see LICENSE].
Modified by Christopher Plachetka and Benjamin Sertolli, 2022.
    Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn

from deep_learning.models.helper import get_multiple_conv_dim_size, ZeroPad3d
from torchplus import scatter_nd
from torchplus.nn import Empty, Sequential
from torchplus.tools import change_default_args


class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=64):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.

        Args:
            output_shape (list[int] | tuple[int]): Required output shape of features (size: 4)
            num_input_features (int): Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.n_channels_out = num_input_features

    def forward(self, voxel_features, coords, batch_size, map_fm):
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.n_channels_out, self.nx * self.ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.n_channels_out, self.ny, self.nx)

        # Merge with map-prior (if given)
        if map_fm is not None:
            if len(list(map_fm.shape)) == 5:  # [BS, x, y, z, elem_type]
                map_fm = torch.squeeze(map_fm, 3)  # remove z-dim from shape, if it exists
            map_fm = map_fm.permute(0, 3, 2, 1)  # [BS, x, y, elem_type] -> [BS, elem_type, y, x]
            batch_canvas = torch.cat((batch_canvas, map_fm), dim=1)

        return batch_canvas


class VoxelnetScatter(nn.Module):
    def __init__(self, output_shape, num_input_features):
        """
        Variation of PointPillarScatter, which converts list of voxels back into a sparse 3D voxel grid (instead of a
        2D BEV pseudo image) and is used for 3DHDNet.

        Args:
            output_shape (list[int] | tuple[int]): Required output shape of features (size: 4)
            num_input_features (int): Number of input features.
        """
        super().__init__()
        self.name = 'VoxelnetScatter'
        self.voxel_output_shape = output_shape
        self.n_channels_out = num_input_features

    def forward(self, voxel_features, coors, batch_size, map_fm):
        output_shape = [batch_size] + self.voxel_output_shape[1:]
        ret = scatter_nd(coors.long(), voxel_features, output_shape)
        ret = ret.permute(0, 4, 1, 2, 3)  # [BS, z, y, x, features] -> [BS, features, z, y, x]

        # Merge with map-prior (if given)
        if map_fm is not None:
            map_fm = map_fm.permute(0, 4, 3, 2, 1)  # [BS, x, y, z, elem_type] -> [BS, elem_type, z, y, x]
            ret = torch.cat((ret, map_fm), dim=1)

        # N, C, D, H, W = ret.shape

        return ret


class MiddleExtractor(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=(64, ),
                 num_filters_down2=(64, 64)):
        """
        Middle-layer module from VoxelNet.
        First, converts a list of voxels (from an encoder module) back to a sparse 3D voxel grid.
        Then, applies a series of 3D convs to process and downsample the tensor along the z-dimension.
        Finally, merges the remaining z-dim into the feature dimension to create a 2D BEV representation.

        Args:
            output_shape (list[int] | tuple[int]): Required output shape of features (size: 4)
            use_norm: Whether to include BatchNorm.
            num_input_features (int): Number of input features.
            num_filters_down1 (tuple): Number of filters in first downsampling block (note: a non-downsampling layer is
                                       appended using the last filter size)
            num_filters_down2 (tuple): Number of filters in second down-sampling block
        """
        super().__init__()
        self.name = 'MiddleExtractor'
        if use_norm:
            BatchNorm3d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm3d)
            Conv3d = change_default_args(bias=False)(nn.Conv3d)
        else:
            BatchNorm3d = Empty
            Conv3d = change_default_args(bias=True)(nn.Conv3d)
        self.voxel_output_shape = output_shape

        num_filters_down1 = list(num_filters_down1)
        num_filters_down2 = list(num_filters_down2)
        n_layers_total = len(num_filters_down1) + 1 + len(num_filters_down2)

        kernel_sizes = [3] * n_layers_total
        # Stride 2 for down-sampling layers, stride 1 for 'lateral' layer (halves z-dim)
        z_strides = [2] * n_layers_total
        z_strides[len(num_filters_down1)] = 1
        # padding for down-sampling layers (for proper halving), no padding for 'lateral' layer
        z_paddings = [1] * n_layers_total
        z_paddings[len(num_filters_down1)] = 0

        n_channels = num_filters_down1 + [num_filters_down1[-1]] + num_filters_down2

        d_in = num_input_features
        middle_layers = []
        for kernel_size, z_stride, z_pad, d_out in zip(kernel_sizes, z_strides, z_paddings, n_channels):
            middle_layers += [
                ZeroPad3d((1, 1, 1, 1, z_pad, z_pad)),
                Conv3d(d_in, d_out, kernel_size, stride=(z_stride, 1, 1)),
                BatchNorm3d(d_out),
                nn.ReLU()
            ]
            d_in = d_out

        self.middle_conv = Sequential(*middle_layers)

        z_dim_out = get_multiple_conv_dim_size(output_shape[1], kernel_sizes, z_paddings, z_strides)
        self.n_channels_out = n_channels[-1] * z_dim_out

    def forward(self, voxel_features, coors, batch_size, map_fm):
        output_shape = [batch_size] + self.voxel_output_shape[1:]
        ret = scatter_nd(coors.long(), voxel_features, output_shape)
        ret = ret.permute(0, 4, 1, 2, 3)  # [BS, z, y, x, features] -> [BS, features, z, y, x]

        # Merge with map-prior (if given)
        if map_fm is not None:
            map_fm = map_fm.permute(0, 4, 3, 2, 1)  # [BS, x, y, z, elem_type] -> [BS, elem_type, z, y, x]
            ret = torch.cat((ret, map_fm), dim=1)

        ret = self.middle_conv(ret)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret


class MiddleExtractorExtended(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_layers=(),
                 num_filters=(),
                 z_strides=()
                 ):
        """
        Extended version of MiddleExtractor which allows for a more freely configuration of blocks and layers.
        First, converts a list of voxels (from an encoder module) back to a sparse 3D voxel grid.
        Then, applies a series of 3D convs to process and downsample the tensor along the z-dimension.
        Finally, merges the remaining z-dim into the feature dimension to create a 2D BEV representation.

        Args:
            output_shape (list[int] | tuple[int]): Required output shape of features (size: 4)
            use_norm: Whether to include BatchNorm.
            num_input_features (int): Number of input features.
            num_layers (tuple): Block configuration (e.g., (3, 3) -> two blocks with 3 layers each)
            num_filters (tuple): Number of filters in each block (e.g., (32, 64) -> first block: 32 filters for each
                                 layer, second block: 64 filters each)
            z_strides (tuple): z_stride applied in the first layer of each block (2 = downsampling, 1 = no downsampling)
                               Example: (2, 1) -> first block: z_stride 2 for first layer and 1 for consecutive layers,
                               second block: z_stride 1 for each layer
        """
        super().__init__()
        self.name = 'MiddleExtractorExtended'
        num_layers = list(num_layers)
        num_filters = list(num_filters)
        z_strides = list(z_strides)
        assert len(num_layers) == len(num_filters)
        assert len(num_layers) == len(z_strides)

        if use_norm:
            BatchNorm3d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm3d)
            Conv3d = change_default_args(bias=False)(nn.Conv3d)
        else:
            BatchNorm3d = Empty
            Conv3d = change_default_args(bias=True)(nn.Conv3d)
        self.voxel_output_shape = output_shape

        self.middle_layer_blocks = nn.ModuleList()
        d_in = num_input_features

        for block_idx, n_layers in enumerate(num_layers):  # (3, 2, 1)
            block_layers = []

            for layer_idx in range(n_layers):
                d_out = num_filters[block_idx]  # (32, 64, 128)
                # only apply variable z_stride in first layer of each block
                z_stride = z_strides[block_idx] if layer_idx == 0 else 1
                z_pad = 1
                block_layers += [
                    ZeroPad3d((1, 1, 1, 1, z_pad, z_pad)),
                    Conv3d(d_in, d_out, 3, stride=(z_stride, 1, 1)),
                    BatchNorm3d(d_out),
                    nn.ReLU(),
                ]
                d_in = d_out
            self.middle_layer_blocks.append(Sequential(*block_layers))

        z_dim_out = get_multiple_conv_dim_size(output_shape[1], [3] * len(num_layers), [1] * len(num_layers), z_strides)
        self.n_channels_out = num_filters[-1] * z_dim_out

    def forward(self, voxel_features, coors, batch_size, map_fm):
        output_shape = [batch_size] + self.voxel_output_shape[1:]
        ret = scatter_nd(coors.long(), voxel_features, output_shape)
        ret = ret.permute(0, 4, 1, 2, 3)  # [BS, z, y, x, features] -> [BS, features, z, y, x]

        # Merge with map-prior (if given)
        if map_fm is not None:
            map_fm = map_fm.permute(0, 4, 3, 2, 1)  # [BS, x, y, z, elem_type] -> [BS, elem_type, z, y, x]
            ret = torch.cat((ret, map_fm), dim=1)

        for block in self.middle_layer_blocks:
            ret = block(ret)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret
