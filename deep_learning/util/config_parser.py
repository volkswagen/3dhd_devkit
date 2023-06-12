""" Parses the default configuration and applies automatic settings.
"""
import argparse
import ast
import configparser
from os import path
from pathlib import Path
import math
from distutils.util import strtobool

from utility import system_paths as sys_paths
from deep_learning.util.lookups import *


# Module functions #####################################################################################################
########################################################################################################################


def parse_config(config_file_path, apply_auto_settings=True):
    """ Parses a configuration file.
    Args:
        config_file_path (str): path to config file
        apply_auto_settings (bool): enables automatic settings

    Returns:
        configs (dict:dict): contains model, train, and system settings
    """
    config = configparser.ConfigParser()

    if not path.exists(config_file_path):
        print(config_file_path)
        raise FileNotFoundError("Could not find config file.")

    config.read(config_file_path)

    model_config = config['model']
    train_config = config['train']
    system_config = config['system']

    model_config_dict = {}
    train_config_dict = {}
    system_config_dict = {}

    # Interpret config strings als python params
    for key, val in model_config.items():
        model_config_dict[key] = ast.literal_eval(val)

    for key, val in train_config.items():
        train_config_dict[key] = ast.literal_eval(val)

    for key, val in system_config.items():
        system_config_dict[key] = ast.literal_eval(val)

    # add some auto-generated settings if desired
    if sys_paths.is_system_known() and system_config_dict['auto_configure_paths']:
        system_config_dict.update(sys_paths.get_system_paths())

    configs = {
        'system': system_config_dict,
        'model': model_config_dict,
        'train': train_config_dict
    }

    if apply_auto_settings:
        apply_automatic_settings(configs)
    return configs


def create_argparser_from_dict(config_dict, parser=None):
    """ Creates the argument parser for command line calls.
    Args:
        config_dict (dict): configuration settings
        parser (ArgumentParser): argument parser

    Returns:
        parser (ArgumentParser): argument parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Training experiment configuration.")

    not_supported = []
    # float, tuple, str, dict, NoneType, list, bool, pathlib.WindowsPath, int
    for arg, value in config_dict.items():
        if isinstance(value, bool):  # necessary this way to allow False as argument
            parser.add_argument(f'--{arg}', type=lambda x: bool(strtobool(x)))
        elif isinstance(value, str) or \
                isinstance(value, int) or \
                isinstance(value, float):
            parser.add_argument(f'--{arg}', type=type(value))
        elif isinstance(value, Path):
            parser.add_argument(f'--{arg}', type=Path)
        elif isinstance(value, list) and len(value) > 0 and not any(isinstance(el, list) for el in value):
            # Nested and empty lists are not supported (e.g. fm_extent)
            parser.add_argument(f'--{arg}', nargs='*', type=type(value[0]))
        elif arg == 'fm_extent' or (isinstance(value, list) and len(value) == 0):
            # Accept fm_extent as flattened list, and interpret empty lists as float lists
            parser.add_argument(f'--{arg}', nargs='*', type=float)
        elif arg == 'fm_type_map':
            parser.add_argument(f'--{arg}', type=str)
        else:  # tuple, dict, NoneType
            not_supported.append(arg)

    return parser


def save_config(config_file_path, configs):
    """ Saves the given configs in the specified file.
    Args:
        config_file_path (str|Path): full path of the output file, incl. filename and type (should be .ini)
        configs (dict:dict): nested dictionary of configs to save ('system', 'model' and 'train' configs)
    """
    # Convert all config entries to strings first (mandatory for ConfigParser)
    configs_str = {}
    for sub_config in configs:
        configs_str[sub_config] = {}
        for (k, v) in configs[sub_config].items():
            if isinstance(v, Path) or path.exists(str(v)):
                # put raw string quotes around paths to ignore escape sequences (backslashes)
                configs_str[sub_config][k] = f"r'{v}'"
            elif isinstance(v, str):
                # put quotes around actual strings for correct decoding later
                configs_str[sub_config][k] = f"'{v}'"
            else:
                configs_str[sub_config][k] = str(v)

    # Save the 'string-ified' config
    parser = configparser.ConfigParser()
    parser.read_dict(configs_str)
    with open(config_file_path, 'w', encoding='utf-8') as configfile:
        parser.write(configfile)


def update_dict_from_args(args, config_dict):
    """ Updates a single configuration dictionary with the parameters set via command-line.
    Basically a simplified version of update_configs_from_args used by run_inference and run_evaluation.
    Args:
        args (argparse.Namespace|dict): command-line arguments parsed by argparse
        config_dict (dict): dictionary of configs that args will be checked against
    """
    if isinstance(args, argparse.Namespace):
        args = args.__dict__

    args = {k: v for (k, v) in args.items() if v is not None}  # filter out irrelevant args

    for arg, value in args.items():
        updated = False
        if arg in config_dict:
            config_dict.update({arg: value})
            updated = True
        if not updated:
            raise KeyError(f"CLI argument '{arg}' is not specified in the given config dict ({config_dict.keys()})!")


def update_configs_from_args(args, configs):
    """ Updates the current configuration with the parameters set via command-line.
    Args:
        args (argparse.Namespace|dict): command-line arguments parsed by argparse
        configs (dict:dict): nested dictionary of configs that args will be checked against
    """
    if isinstance(args, argparse.Namespace):
        args = args.__dict__

    args = {k: v for (k, v) in args.items() if v is not None}  # filter out irrelevant args

    # Format special cases
    if 'fm_extent' in args:  # 'unflatten' fm_extent
        args['fm_extent'] = [[args['fm_extent'][i], args['fm_extent'][i+1]]for i in range(0, 6, 2)]
    if 'head_num_filters' in args:
        args['head_num_filters'] = [int(n) for n in args['head_num_filters']]
    if 'fm_type_map' in args and args['fm_type_map'].lower() == 'none':  # convert none string to None type
        args['fm_type_map'] = None

    # Update configs
    for arg, value in args.items():
        updated = False
        for config in configs.values():
            if arg in config:
                config.update({arg: value})
                updated = True
                continue
        if not updated:
            raise KeyError(f"CLI argument '{arg}' is not specified in config. Make sure to declare the parameter in the"
                  f" configuration file first and check if the data type is supported (see create_argparser_fom_dict).")

    # Update auto-generated shapes, because fm_extent or voxel_size might have changed
    apply_automatic_settings(configs)


def apply_automatic_settings(configs):
    """ Automatic settings.
    Args:
        configs (dict:dict): model, train, system configurations
    Returns:
        configs (dict:dict): modified model, train, and system configurations
    """
    if configs['model']['auto_generate_shapes']:
        shapes = auto_generate_shapes(configs['model']['fm_extent'], configs['model']['voxel_size'],
                                      configs['model']['target_shape_factor'])
        configs['model'].update(shapes)

    if configs['model']['set_anchor_layers_like_voxels']:
        anchor_config = set_anchor_layers_like_voxels(configs['model']['fm_extent'], configs['model']['voxel_size'],
                                                      configs['model']['target_shape_factor'])
        configs['model'].update(anchor_config)

    if int(configs['train']['dd_stage']) != 0:
        set_deviation_detection_stage(configs)

    if configs['train']['deviation_detection_task']:
        configs['train']['mlflow_group_name'] = "dd"
    else:
        configs['train']['mlflow_group_name'] = ' + '.join(sorted(configs['model']['configured_element_types']))

    if configs['model']['use_recommended_pole_classes']:
        if configs['model']['classification_active']:
            pole_classes = get_all_pole_classes_er()
        else:
            pole_classes = get_all_pole_classes()
            pole_classes.remove('bollard')

        configs['model']['pole_classes'] = pole_classes

    if configs['model']['use_recommended_light_classes']:
        light_classes = get_all_light_classes()
        configs['model']['light_classes'] = light_classes

    if configs['model']['use_recommended_sign_classes']:
        sign_classes = get_all_sign_classes()
        configs['model']['sign_classes'] = sign_classes


def auto_generate_shapes(fm_extent, voxel_size, target_shape_factor):
    """ Automatically generate feature map shapes (high-density point cloud, map, target).
    Args:
        fm_extent (list[list]): feature map extent as [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        voxel_size (list[float]): [x, y, z] list of voxel sizes (e.g., 0.4 m) in each dimension
        target_shape_factor (float): 1.0 keeps the input size, 0.5 downsamples by half

    Returns:
        dict(int tuple): contains shapes as int tuples as number of voxels per dimension (num_x, num_y, num_z)
    """
    n_dims = 3
    base_shape = [math.ceil(abs(fm_extent[i][0] - fm_extent[i][1]) / voxel_size[i]) for i in range(n_dims)]
    # Apply target shape factor
    target_shape = [int(dim * target_shape_factor) for dim in base_shape[:2]]

    return {'fm_shape_hdpc': tuple(base_shape),
            'fm_shape_map': tuple(base_shape),
            'target_shape': tuple(target_shape)}


def set_anchor_layers_like_voxels(fm_extent, voxel_size, target_shape_factor):
    """ Configures anchor grid generation settings to provide a grid of the same size as the input.
    Args:
        fm_extent (list[list]): feature map extent as [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        voxel_size (list[float]): [x, y, z] list of voxel sizes (e.g., 0.4 m) in each dimension
        target_shape_factor (float): 1.0 keeps the input size, 0.5 downsamples by half

    Returns:
        (dict:float): min, max, and stride settings for anchor placement in the vertical dimension
            (see target_generator.py)
    """
    z_min = fm_extent[2][0]
    z_max = fm_extent[2][1]
    z_stride = voxel_size[2] / target_shape_factor
    print(f"New anchor z_min {z_min}, z_max {z_max}, z_stride {z_stride}")

    return {
        'sign_z_min': z_min,
        'light_z_min': z_min,
        'sign_z_max': z_max,
        'light_z_max': z_max,
        'sign_z_stride': z_stride,
        'light_z_stride': z_stride,
        'pole_z_stride': z_stride
    }


def set_deviation_detection_stage(configs):
    """ Automatic settings for specific stages (MDD-SC (stage 1), MDD-MC (stage 2), MDD-M stage 3)).

    Args:
        configs (dict:dict): model, train, system configurations
    """
    dd_stage = int(configs['train']['dd_stage'])
    if dd_stage == 1:
        configs['train']['deviation_detection_task'] = True     # object vs. deviation detection
        configs['model']['deviation_detection_model'] = False   # specialized MDD network (MDD-M only)
        configs['model']['fm_type_map'] = None                  # enables the additional map input
    elif dd_stage == 2:
        configs['train']['deviation_detection_task'] = True
        configs['model']['deviation_detection_model'] = False
        if configs['model']['fm_type_map'] is None:
            configs['model']['fm_type_map'] = 'voxels_lut'
    elif dd_stage == 3:
        configs['train']['deviation_detection_task'] = True
        configs['model']['deviation_detection_model'] = True
        if configs['model']['fm_type_map'] is None:
            configs['model']['fm_type_map'] = 'voxels_lut'
    else:
        raise ValueError(f"dd_stage '{dd_stage}' is not specified!")
