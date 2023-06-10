""" Lookup tables (LUTs) for various purposes.
"""

from typing import List

import numpy as np


def get_element_type_naming_lut():
    """ LUTs mapping groups and element types.
    """
    group_to_type_lut = {
        'poles': 'Pole',
        'signs': 'TrafficSign',
        'lights': 'TrafficLight'
    }

    type_to_group_lut = {
        'Pole': 'poles',
        'Tree': 'poles',
        'TrafficSign': 'signs',
        'TrafficLight': 'lights',
        'CS_Obstacle_Point': 'poles'
    }

    return group_to_type_lut, type_to_group_lut


def get_class_to_index_lookup(model_config):
    """ Maps an index to a class name and vice versa.
    """
    cls_to_idx_luts = {}
    idx_to_cls_luts = {}

    for element_type in model_config['configured_element_types']:
        cls_to_idx_luts[element_type] = {}
        idx_to_cls_luts[element_type] = {}

        if element_type == 'poles':
            configured_classes = model_config['pole_classes']
            for i_cls, cls in enumerate(configured_classes):
                cls_to_idx_luts[element_type][cls] = i_cls
                idx_to_cls_luts[element_type][i_cls] = cls
        elif element_type == 'lights':
            configured_classes = model_config['light_classes']
            for i_cls, cls in enumerate(configured_classes):
                cls_to_idx_luts[element_type][cls] = i_cls
                idx_to_cls_luts[element_type][i_cls] = cls
        elif element_type == 'signs':
            configured_classes = model_config['sign_classes']
            for i_cls, cls in enumerate(configured_classes):
                cls_to_idx_luts[element_type][cls] = i_cls
                idx_to_cls_luts[element_type][i_cls] = cls
        else:
            cls_to_idx_luts[element_type]['All'] = 0
            idx_to_cls_luts[element_type][0] = 'All'
        # Add more classes here for other element types
        # ...

    return cls_to_idx_luts, idx_to_cls_luts


def get_class_definition_lookup(model_config, include_all=True, include_mean=False):
    """ Creates a LUT for each element type with subclass definitions.

    Class definitions may be augmented with summary classes 'All' and 'Mean'.
    'All' summarizes all objects of all classes (e.g., metrics computation disregards imbalanced class distributions).
    'Mean' contains averaged values over all classes balanced for all classes.
    """

    # Init class definition lookup
    classes_dict = {}
    for element_type in model_config['configured_element_types']:
        configured_classes = []
        if include_all:
            configured_classes += ['All']  # add one class for summarizing all other classes in a single class
        if include_mean:
            configured_classes += ['Mean']

        if model_config['classification_active']:
            if element_type == 'poles':
                configured_classes += model_config['pole_classes']
            elif element_type == 'lights':
                configured_classes += model_config['light_classes']
            elif element_type == 'signs':
                configured_classes += model_config['sign_classes']
            else:
                configured_classes += ['All']
                configured_classes = list(set(configured_classes))  # remove duplicates
            # Add more subclasses here for other element types
            # ...

        classes_dict[element_type] = configured_classes

    return classes_dict


def get_metrics_definition(classification_active: bool):
    """ List of used evaluation metrics.
    """
    metrics = ['recall', 'precision', 'f1']
    if classification_active:
        metrics += ['accuracy']
    return metrics


def get_statistics_definition(classification_active: bool):
    """ List of statistic counters.
    TP: true positive, FP: false positive, FN: false negative
    """
    stats = ['TP', 'FP', 'FN']
    if classification_active:
        stats += ['N']  # correctly classified samples -> accuracy measure
    return stats


def get_error_definition_lookup():
    """ Provides a dict:list of regression features for each element type.
    """
    common = ['distance', 'x_vrf', 'y_vrf']
    error_definitions = {
        'poles': common + ['z_vrf', 'diameter'],
        'signs': common + ['z_vrf', 'width', 'height', 'yaw_vrf'],
        'lights': common + ['z_vrf', 'width', 'height', 'yaw_vrf']
    }

    return error_definitions


def get_all_element_types() -> List[str]:
    """ Provides a list of all implemented element types as group names.
    """
    element_types = ['lights', 'poles', 'signs']
    return element_types


def get_main_size_features() -> dict:
    """ Provides a dict:str defining the width parameter for each element type.
    """
    size_features = {
        'poles': 'diameter',
        'signs': 'width',
        'lights': 'width'
    }
    return size_features


def get_all_run_names() -> List[str]:
    """ List of all trajectory names (= run_names in train, val, and test.json)
    """
    runs = ['OP_1', 'OP_2', 'OP_3', 'OP_4', 'OP_5_a', 'OP_5_b', 'OP_5_c', 'RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_5',
            'RH_6', 'RH_7', 'RH_8', 'RH_9_a', 'RH_9_b', 'RH_10']
    return runs


def dict_list_to_np_array(dict_list, keys_to_consider=None, include_idx=True):
    """ Converts a dict to a numby array
    Args:
        dict_list (dict): data dict to convert
        keys_to_consider (list[str]): list of dict keys to convert
        include_idx (bool): whether to include the entry's index (e.g., for sorting)

    Returns:
        (np.array): matrix with columns being former dict keys and rows being the dict entries
    """
    if keys_to_consider is None:
        keys_to_consider = dict_list[0].keys()

    attributes_vec = [[e[att] for e in dict_list] for att in keys_to_consider]

    if include_idx:
        attributes_vec += [list(range(0, len(dict_list)))]
    return np.column_stack(tuple(attributes_vec))


def init_nested_dict(key_levels: list, init_value) -> dict:
    """ Recursively initializes a multi-level dictionary with the given keys
    Args:
        key_levels (list[list]):
                    list of lists containing the dictionary keys for every level, e.g. key_levels[0] contains keys for
                    the outermost/first-level dictionary, and so on. For conditional keys, a level can also be
                    specified as a dictionary, where the dict keys must match all keys of the previous level
        init_value: any value/object to initialize the last dictionary level with, e.g. {} or 1. Will be copied to avoid
                    referencing the same object across the dictionary

    Returns:
        (dict:dict) Nested dictionary d with values d[key_levels[0]][...][key_levels[-1]] == init_value
    """
    assert isinstance(key_levels, list)
    assert len(key_levels) > 0
    assert not isinstance(key_levels[0], dict)

    if len(key_levels) == 1:
        if hasattr(init_value, 'copy'):
            # copy init_value if possible to avoid shared references
            return {key: init_value.copy() for key in key_levels[0]}
        else:
            return {key: init_value for key in key_levels[0]}
    else:
        if isinstance(key_levels[1], dict):  # use conditional keys
            assert set(key_levels[0]) == set(key_levels[1].keys()), "dict keys must match all keys of previous level"
            ml_dict = {}
            for key in key_levels[0]:
                if len(key_levels) == 2:  # last level left
                    conditional_key_levels = [key_levels[1][key]]
                else:  # more levels left -> append
                    conditional_key_levels = [key_levels[1][key]] + key_levels[2:]
                ml_dict[key] = init_nested_dict(conditional_key_levels, init_value)
            return ml_dict
        else:
            return {key: init_nested_dict(key_levels[1:], init_value) for key in key_levels[0]}


def create_pole_height_lut():
    """ Creates a dict:float containing default height values for each pole class.
    """
    pole_height_lut = {
        'bollard': .5,
        'traffic_light_pole': 3,
        'traffic_sign_pole': 3,
        'billboard_pole': 3,
        'tree': 3,
        'lamppost': 5,
        'telephone_pole': 5,
        'advertising_pillar': 3,
        'flag_pole': 3,
        'gantry_pole': 3,
        'guardrail_pole': .5,
        'pillar': 3,
        'general_pole': 3,
        'delineator': .5,
        'cone': .5,
        'reflector': .5,
        'protection_pole': .5,
    }

    return pole_height_lut


def get_all_pole_classes_er():
    """ Creates a list of all pole classes used for element recognition (detection + classification).
    """
    pole_classes = [
        'protection_pole',
        'traffic_light_pole',
        'traffic_sign_pole',
        'general_pole',
        'lamppost',
        'tree'
    ]
    return pole_classes


def get_all_pole_classes():
    """ Creates a list of all pole classes used for deviation detection. """
    pole_classes = [
        'bollard',
        'traffic_light_pole',
        'traffic_sign_pole',
        'billboard_pole',
        'lamppost',
        'telephone_pole',
        'advertising_pillar',
        'flag_pole',
        'gantry_pole',
        'guardrail_pole',
        'pillar',
        'tree',
        'general_pole'
    ]
    return pole_classes


def get_pole_class_lut():
    """ Creates LUT mapping all available pole classes to those used for classification.
    """
    pole_class_lut = {
        'bollard': 'protection_pole',
        'guardrail_pole': 'protection_pole',
        'traffic_light_pole': 'traffic_light_pole',
        'traffic_sign_pole': 'traffic_sign_pole',
        'billboard_pole': 'general_pole',
        'telephone_pole': 'general_pole',
        'advertising_pillar': 'general_pole',
        'flag_pole': 'general_pole',
        'gantry_pole': 'general_pole',
        'pillar': 'general_pole',
        'general_pole': 'general_pole',
        'lamppost': 'lamppost',
        'tree': 'tree'
    }
    return pole_class_lut


def get_all_light_classes():
    """ Creates a list of all light classes
    """
    light_classes = [
        'people',
        'warning',
        'vehicle'
    ]
    return light_classes


def get_all_sign_classes():
    """ Creates a list of all sign classes
    """
    sign_classes = [
        'circle',
        'rectangle',
        'triangle_up',
        'triangle_down',
        'diamond',
        'arrow',
        'octagon',
    ]
    return sign_classes
