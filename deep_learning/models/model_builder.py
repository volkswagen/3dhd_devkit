""" Functions to build and initialize models. """

import numpy as np
import torch
import torch.nn as nn

from deep_learning.models import mainnet
from dataset.map_deviation import get_deviation_classes


# Module functions #####################################################################################################
########################################################################################################################


def build(model_config: dict, train_config: dict) -> mainnet.MainNet:
    """ Sets up and builds a 3DHDNet (or related) model according to the given configurations. """

    voxel_grid_shape = model_config['fm_shape_hdpc']  # [nx, ny, nz]
    batch_size = train_config['batch_size']
    fm_extent = model_config['fm_extent']

    encoder_output_shape = [batch_size,
                            voxel_grid_shape[2],  # z
                            voxel_grid_shape[1],  # y
                            voxel_grid_shape[0],  # x
                            model_config['vfe_num_filters'][-1]]

    target_shape_z = int(voxel_grid_shape[2] * model_config['target_shape_factor'])

    # pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    pc_range = [fm_extent[0][0], fm_extent[1][0], fm_extent[2][0],
                fm_extent[0][1], fm_extent[1][1], fm_extent[2][1]]

    # Classes placeholder is used for deviation states (MDD) or element type classes (object detection, optional)
    num_classes_per_type = {elem_type: 1 for elem_type in model_config['configured_element_types']}
    if model_config['deviation_detection_model']:
        n_classes = len(get_deviation_classes(False))
        num_classes_per_type = {elem_type: n_classes for elem_type in model_config['configured_element_types']}
    elif model_config['classification_active']:
        if 'poles' in model_config['configured_element_types']:
            num_classes_per_type['poles'] = len(model_config['pole_classes'])

        if 'lights' in model_config['configured_element_types']:
            num_classes_per_type['lights'] = len(model_config['light_classes'])

        if 'signs' in model_config['configured_element_types']:
            num_classes_per_type['signs'] = len(model_config['sign_classes'])

    classification_active = model_config['classification_active'] or model_config['deviation_detection_model']

    num_vertical_layers_per_type = {
        'poles': 1,
        'signs': int(
            np.ceil((model_config['sign_z_max'] - model_config['sign_z_min']) / model_config['sign_z_stride'])),
        'lights': int(
            np.ceil((model_config['light_z_max'] - model_config['light_z_min']) / model_config['light_z_stride']))
    }

    return mainnet.MainNet(
        encoder_output_shape=encoder_output_shape,
        batch_size=batch_size,
        configured_element_types=model_config['configured_element_types'],
        num_vertical_layers_per_type=num_vertical_layers_per_type,
        voxel_size=model_config['voxel_size'],
        pc_range=pc_range,
        vfe_class_name=model_config['vfe_class_name'],
        vfe_num_filters=model_config['vfe_num_filters'],
        middle_class_name=model_config['middle_class_name'],
        middle_num_filters_d1=model_config['middle_num_filters_d1'],
        middle_num_filters_d2=model_config['middle_num_filters_d2'],
        middle_num_layers=model_config['middle_num_layers'],
        middle_num_filters=model_config['middle_num_filters'],
        middle_z_strides=model_config['middle_z_strides'],
        bb_layer_nums=model_config['bb_layer_nums'],
        bb_layer_strides=model_config['bb_layer_strides'],
        bb_num_filters=model_config['bb_num_filters'],
        bb_upsample_strides=model_config['bb_upsample_strides'],
        bb_num_upsample_filters=model_config['bb_num_upsample_filters'],
        classification_active=classification_active,
        num_classes_per_type=num_classes_per_type,
        use_3d_backbone=model_config['use_3d_backbone'],
        use_3d_heads=model_config['use_3d_heads'],
        target_shape_z=target_shape_z,
        head_num_filters=model_config['head_num_filters'],
        use_norm=model_config['use_norm'],
        map_fm_type=model_config['fm_type_map'],
        map_fm_fusion_type=model_config['map_fm_fusion_type'],
    )


def init_from_pretrained(net, pretrained_net_path):
    """ Initializes given model with weights from a pretrained model. """

    # Load pretrained model
    pretrained_dict = torch.load(pretrained_net_path)

    # Init state dict of current net
    net_dict = net.state_dict()

    # Only keep keys of pretrained_dict that match new net_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}

    # Overwrite new net_dict
    net_dict.update(pretrained_dict)

    # Check missing layers
    new_layers = [k for k in net_dict.keys() if k not in pretrained_dict]
    print(f"Missing layers: {new_layers}")

    # Load weights etc.
    net.load_state_dict(net_dict)


def init_weights(module, mode_conv="kaiming_stddev", mode_linear="none", init_bias=False):
    # Init conv layer
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        print(f"Initializing weights for {type(module)}")
        # Set biases to 0
        if init_bias:
            torch.nn.init.zeros_(module.bias)

        if mode_conv == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu')
        elif mode_conv == "kaiming_stddev":
            torch.nn.init.kaiming_normal_(module.weight, a=0.1, nonlinearity='leaky_relu', mode='fan_in')
        elif mode_conv == "xavier":
            torch.nn.init.xavier_uniform_(module.weight)

    elif isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
        print(f"Initializing weights for {type(module)}")
        # Set biases to 0
        if init_bias:
            torch.nn.init.zeros_(module.bias)

        if mode_conv == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu')
        elif mode_conv == "kaiming_stddev":
            torch.nn.init.kaiming_normal_(module.weight, a=0.1, nonlinearity='leaky_relu', mode='fan_in')
        elif mode_conv == "xavier":
            torch.nn.init.xavier_uniform_(module.weight)

    # Init linear layer
    elif isinstance(module, nn.Linear):
        print(f"Initializing weights for {type(module)}")
        # Set biases to 0
        if init_bias:
            torch.nn.init.zeros_(module.bias)

        if mode_linear == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu')
        elif mode_linear == "kaiming_stddev":
            torch.nn.init.kaiming_normal_(module.weight, a=0.1, nonlinearity='leaky_relu', mode='fan_in')
        elif mode_linear == "xavier":
            torch.nn.init.xavier_uniform_(module.weight)


# Test section #########################################################################################################
########################################################################################################################


def main():
    pass


if __name__ == "__main__":
    main()
