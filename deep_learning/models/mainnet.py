""" Module containing the main network, which incorporates encoder, middle layer, and backbone. """

import torch
from torch import nn

from deep_learning.models import encoders as vfe
from deep_learning.models import middle_layers, backbone
from deep_learning.util.lookups import get_all_element_types


class MainNet(nn.Module):
    def __init__(self,
                 encoder_output_shape,
                 batch_size,
                 configured_element_types,
                 num_vertical_layers_per_type,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 num_input_features=4,
                 vfe_class_name="FeatureNet3DHD",
                 vfe_num_filters=(64, 128),
                 with_distance=False,
                 middle_class_name="VoxelnetScatter",
                 middle_num_filters_d1=(64,),
                 middle_num_filters_d2=(64, 64),
                 middle_num_layers=(),
                 middle_num_filters=(),
                 middle_z_strides=(),
                 bb_class_name="Backbone",
                 bb_layer_nums=(3, 5, 5),
                 bb_layer_strides=(2, 2, 2),
                 bb_num_filters=(128, 128, 256),
                 bb_upsample_strides=(1, 2, 4),
                 bb_num_upsample_filters=(256, 256, 256),
                 classification_active=False,
                 num_classes_per_type=None,
                 use_3d_backbone=False,
                 use_3d_heads=False,
                 target_shape_z=None,
                 head_num_filters=(),
                 use_norm=True,
                 map_fm_type=None,
                 map_fm_fusion_type=None,
                 name='mainnet',
                 ):
        """
        Main network, which incorporates encoder, middle layer, and backbone.
        Can be used to build 3DHDNet, PointPillars, and VoxelNet.

        Args:
            encoder_output_shape (list | tuple): output shape of encoder stage in the form of
                                                 [bs, nz, ny, nx, vfe_num_filters[-1]]
            batch_size (int): batch size
            configured_element_types (list | tuple): [Backbone] list of element types (lights, poles, signs)
            num_vertical_layers_per_type (dict:int): [Backbone] dict specifying number of vertical layers (z-dim) for 
                                                     each configured element type
            voxel_size (list | tuple): [Encoder] voxel size in each dimension [x, y, z] in meter
            pc_range (list | tuple): [Encoder] point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
            num_input_features (int): [Encoder] number of input features, either 3 (x, y, z) or 4 (x, y, z, i).
            vfe_class_name (str): [Encoder] FeatureNet3DHD, PillarFeatureNet, or VoxelFeatureExtractor
            vfe_num_filters (list | tuple): [Encoder] layer configuration for encoder
            with_distance (bool): [Encoder] use distance as additional feature for encoder
            middle_class_name (str): [Middle Layer] PointPillarsScatter, VoxelnetScatter, MiddleExtractor, or
                                     MiddleExtractorExtended
            middle_num_filters_d1 (list | tuple): [Middle Layer] Setting for MiddleExtractor
            middle_num_filters_d2 (list | tuple): [Middle Layer] Setting for MiddleExtractor
            middle_num_layers (list | tuple): [Middle Layer] Setting for MiddleExtractorExtended
            middle_num_filters (list | tuple): [Middle Layer] Setting for MiddleExtractorExtended
            middle_z_strides (list | tuple): [Middle Layer] Setting for MiddleExtractorExtended
            bb_class_name (str): [Backbone] Backbone
            bb_layer_nums (list | tuple): [Backbone] specify number of layers in each downstream block
            bb_layer_strides (list | tuple): [Backbone] strides (z-dim) applied on the first conv layer of each block
            bb_num_filters (list | tuple): [Backbone] number of filters for each downstream block
            bb_upsample_strides (list | tuple): [Backbone] strides used to upsample tensor after every downstream block,
                                                so that every upsampled tensor has the same size (in spatial dims)
            bb_num_upsample_filters (list | tuple): [Backbone] number of filters used for upsampling convs (transposed 
                                                    convs); must correspond to number of downsampling blocks
            classification_active (bool): [Backbone] whether to perform subclass classification (only element detection)
            num_classes_per_type (dict): [Backbone] specifies the number of classes for each element type (used for MDD
                                                    or subclass classification)
            use_3d_backbone (bool): [Backbone] whether to use 3D convs for backbone (more expensive)
            use_3d_heads (bool): [Backbone] whether to use 3D convs for output heads (except for poles)
            target_shape_z (int): [Backbone] size of target output shape in the z dimension (required with 3D backbone)
            head_num_filters (list | tuple): [Backbone] specify additional conv layers applied to every output head
            use_norm (bool): whether to use BatchNorm (for full network)
            map_fm_type (str): 'voxels_reg' or 'voxels_lut', everything else -> None
            map_fm_fusion_type (str): 'early' or 'late', everything else -> None
            name (str): some identifiable name (default: mainnet)
        """
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.map_fm_fusion_type = map_fm_fusion_type if map_fm_fusion_type is not None else 'none'

        apply_map_fusion = map_fm_type is not None and map_fm_type.startswith("voxel")
        num_map_features = len(get_all_element_types())
        if map_fm_type == 'voxels_reg' or map_fm_type == 'voxels_lut':
            num_map_features += 7

        # Encoder
        encoder_class_dict = {
            "FeatureNet3DHD": vfe.FeatureNet3DHD,
            "PillarFeatureNet": vfe.PillarFeatureNet,
            "VoxelFeatureExtractor": vfe.VoxelFeatureExtractor
        }
        encoder_class = encoder_class_dict[vfe_class_name]

        if vfe_class_name in ["PillarFeatureNet", "FeatureNet3DHD"]:
            self.voxel_feature_encoder = encoder_class(
                num_input_features,
                use_norm,
                num_filters=vfe_num_filters,
                with_distance=with_distance,
                voxel_size=voxel_size,
                pc_range=pc_range
            )
        elif vfe_class_name == "VoxelFeatureExtractor":
            self.voxel_feature_encoder = encoder_class(
                num_input_features,
                use_norm,
                num_filters=vfe_num_filters,
                with_distance=with_distance)
        else:
            raise ValueError(f"Unknown vfe_class_name '{vfe_class_name}'")

        # Middle layer
        num_middle_input_features = vfe_num_filters[-1]
        if apply_map_fusion and self.map_fm_fusion_type == 'early':
            num_middle_input_features += num_map_features

        middle_class_dict = {
            "PointPillarsScatter": middle_layers.PointPillarsScatter,
            "VoxelnetScatter": middle_layers.VoxelnetScatter,
            "MiddleExtractorExtended": middle_layers.MiddleExtractorExtended,
            "MiddleExtractor": middle_layers.MiddleExtractor,
        }
        middle_layer_class = middle_class_dict[middle_class_name]

        if middle_class_name in ["PointPillarsScatter", "VoxelnetScatter"]:
            self.middle_layer = middle_layer_class(
                output_shape=encoder_output_shape,
                num_input_features=num_middle_input_features
            )
        elif middle_class_name == "MiddleExtractorExtended":
            self.middle_layer = middle_layer_class(
                encoder_output_shape,
                use_norm,
                num_input_features=num_middle_input_features,
                num_layers=middle_num_layers,
                num_filters=middle_num_filters,
                z_strides=middle_z_strides
            )
        elif middle_class_name == 'MiddleExtractor':
            self.middle_layer = middle_layer_class(
                encoder_output_shape,
                use_norm,
                num_input_features=num_middle_input_features,
                num_filters_down1=middle_num_filters_d1,
                num_filters_down2=middle_num_filters_d2
            )
        else:
            raise ValueError(f"Unknown middle_class_name '{middle_class_name}'")
        num_bb_input_filters = self.middle_layer.n_channels_out

        # Backbone
        num_map_features_head = 0
        if apply_map_fusion and self.map_fm_fusion_type == 'late':
            num_map_features_head = num_map_features
        bb_class_dict = {
            "Backbone": backbone.Backbone,
        }
        bb_class = bb_class_dict[bb_class_name]
        self.backbone = bb_class(
            configured_element_types=configured_element_types,
            num_vertical_layers_per_type=num_vertical_layers_per_type,
            target_shape_z=target_shape_z,
            num_classes_per_type=num_classes_per_type,
            classification_active=classification_active,
            use_norm=True,
            layer_nums=bb_layer_nums,
            layer_strides=bb_layer_strides,
            num_filters=bb_num_filters,
            upsample_strides=bb_upsample_strides,
            num_upsample_filters=bb_num_upsample_filters,
            num_input_filters=num_bb_input_filters,
            use_3d_backbone=use_3d_backbone,
            use_3d_heads=use_3d_heads,
            head_num_filters=head_num_filters,
            num_map_features_head=num_map_features_head)

        self.register_buffer("global_step", torch.LongTensor(1).zero_())

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def forward(self, example):
        # Setup
        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]

        map_fm_early = None
        map_fm_late = None
        if 'map_fm' in example:  # Fill early or late fusion placeholder, depending on setting
            if self.map_fm_fusion_type == 'early':
                map_fm_early = example["map_fm"]
            elif self.map_fm_fusion_type == 'late':
                map_fm_late = example["map_fm"]

        batch_size_dev = self.batch_size

        # voxels: [num_voxels, max_num_points_per_voxel, num_input_features]  # e.g., [45042, 96, 4]
        # num_points: [num_voxels]
        # coors: [num_voxels, 4]
        voxel_features = self.voxel_feature_encoder(voxels, num_points, coors)
        # output: [num_voxels, vfe_num_filters[-1]]  # e.g. [45042, 128]

        spatial_features = self.middle_layer(voxel_features, coors, batch_size_dev, map_fm_early)
        # output VoxelNetScatter (3DHDNet): [b, num_bb_input_filters, nz, ny, nx]  # e.g., [2, 128, 30, 200, 140]
        # output other classes (PP, VN): [b, num_bb_input_filters, ny, nx]  # e.g., [2, 128, 200, 140]

        out_dict = self.backbone(spatial_features, map_fm_late)

        return out_dict
