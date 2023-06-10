""" Collection of voxel feature encoder (VFE) modules.

Adopted from https://github.com/traveller59/second.pytorch
    PointPillars fork from SECOND.
    Original code written by Alex Lang and Oscar Beijbom, 2018.
    Licensed under MIT License [see LICENSE].
Modified by Christopher Plachetka and Benjamin Sertolli, 2022.
    Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F

from deep_learning.models.helper import get_paddings_indicator
from torchplus.nn import Empty
from torchplus.tools import change_default_args


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """ Pillar Feature Net Layer, used as a base layer for all encoders.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_norm (bool): Whether to include BatchNorm.
            last_layer (bool): If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            bias = False
        else:
            BatchNorm1d = Empty
            bias = True

        self.linear = nn.Linear(in_channels, self.units, bias=bias)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        # inputs: [N, 100, 9]. linear: input = 9, output = out_channels, for each of 100 points as additional dimension
        x = self.linear(inputs)
        # x: [N, 100, 32] -> [N, 32, 100], apply norm for each channel, average etc for norm calculated over 100 points
        # 2nd permute: [N, 100, 32]
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)

        if self.last_vfe:
            return x
        else:
            # x_max: [N, 1, 32], [0] for values not indices, max along 100 points, max for ch:0, max for ch:1 etc.
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)  # [N, 1, 32] -> [N, 100, 32], repeat n_point times
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class FeatureNet3DHD(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """ 3DHD Feature Net.
        The network prepares the voxel features and performs forward pass through PFNLayers. Similar to PointPillar's
        PillarFeatureNet, but additionally performs point augmentation in the z dimension.

        Args:
            num_input_features (int): Number of input features, either 3 (x, y, z) or 4 (x, y, z, i).
            use_norm (bool): Whether to include BatchNorm.
            num_filters (list | tuple): Number of features in each of the N PFNLayers.
            with_distance (bool): Whether to include Euclidean distance to points.
            voxel_size (list | tuple): Size of voxels in meter (x, y, z)
            pc_range (list | tuple): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        """

        super().__init__()
        self.name = 'FeatureNet3DHD'
        assert len(num_filters) > 0
        num_input_features += 6
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create FeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need voxel size and x/y/z offset in order to calculate voxel offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self.z_offset = self.vz / 2 + pc_range[2]

    def forward(self, features, num_voxels, coors):
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from voxel center
        f_center = torch.zeros_like(features[:, :, :3])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)
        f_center[:, :, 2] = features[:, :, 2] - (coors[:, 1].float().unsqueeze(1) * self.vz + self.z_offset)

        # Combine feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether voxel was empty. Need to ensure that
        # empty voxels remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = torch.max(features, dim=1, keepdim=True)[0]

        return features.squeeze(dim=1)


class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND and VoxelNet's VoxelFeatureExtractor.

        Args:
            num_input_features (int): Number of input features, either 3 (x, y, z) or 4 (x, y, z, i).
            use_norm (bool): Whether to include BatchNorm.
            num_filters (list | tuple): Number of features in each of the N PFNLayers.
            with_distance (bool): Whether to include Euclidean distance to points.
            voxel_size (list | tuple): Size of voxels, only utilize x and y size.
            pc_range (list | tuple): Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x and y from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)

        # Combine feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = torch.max(features, dim=1, keepdim=True)[0]

        return features.squeeze(dim=1)


class VoxelFeatureExtractor(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(32, 128),
                 with_distance=False):
        """ VoxelFeatureExtractor.
        The network prepares the voxel features and performs forward pass through PFNLayers. Used to build the VoxelNet
        / SECOND model.

        Args:
            num_input_features (int): Number of input features, either 3 (x, y, z) or 4 (x, y, z, i).
            use_norm (bool): Whether to include BatchNorm.
            num_filters (list | tuple): Number of features in each of the N PFNLayers.
            with_distance (bool): Whether to include Euclidean distance to points.
        """
        super().__init__()
        self.name = 'VoxelFeatureExtractor'
        assert len(num_filters) > 0
        num_input_features += 3
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        num_filters = [num_input_features] + list(num_filters)
        filters_pairs = [[num_filters[i], num_filters[i + 1]]
                         for i in range(len(num_filters) - 1)]
        vfe_layers = []
        for _, (i, o) in enumerate(filters_pairs):
            vfe_layers.append(PFNLayer(i, o, use_norm))
        vfe_layers.append(PFNLayer(num_filters[-1], num_filters[-1], use_norm, last_layer=True))
        self.vfe_layers = nn.ModuleList(vfe_layers)

    def forward(self, features, num_voxels, coors):
        # features: [concatenated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concatenated_num_points]
        assert coors is not None
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat([features, features_relative, points_dist], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        for vfe in self.vfe_layers:
            features = vfe(features)
            features *= mask
        # x: [concatenated_num_points, num_voxel_size, 128]
        voxelwise = torch.max(features, dim=1)[0]
        return voxelwise
