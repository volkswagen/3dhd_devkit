""" Collection of functions to generate plausible map deviations.
"""

import math
from typing import List

import numpy as np

from dataset.point_cloud import PointCloud
from dataset.map_deviation import MapDeviation
from dataset.map_deviation import DeviationTypes as DT
from deep_learning.util import lookups
from deep_learning.preprocessing.map_fmt import limit_yaw_value
from deep_learning.preprocessing.hdpc_modifier import remove_element_from_point_cloud


def generate_map_deviations(map_elements: List[dict], deviation_probs: dict) -> List[MapDeviation]:
    """ Generates list of plausible map deviations from given map elements and settings

    Args:
        map_elements: unordered list of map elements
        deviation_probs: target distribution of INS, DEL, and SUB (deviation probabilities)

    Returns:
        map_deviations: list of generated deviations

    """
    map_deviations = []

    # Set up deviation types and distribution (without VER)
    deviation_types = [DT.DELETION, DT.INSERTION, DT.SUBSTITUTION]
    target_distribution = [deviation_probs[d_type] for d_type in deviation_types]

    # Add VER (must be last to ensure right distribution later)
    deviation_types.append(DT.VERIFICATION)
    target_distribution.append(1 - sum(target_distribution))

    group_to_type_lut, type_to_group_lut = lookups.get_element_type_naming_lut()
    # Sort by type to ensure poles are processed first (due to possible adjacent signs/lights)
    elements_by_type = {e_type: [e for e in map_elements if e['type'] == group_to_type_lut[e_type]]
                        for e_type in ['poles', 'lights', 'signs']}

    for e_type, elements in elements_by_type.items():

        # Set up deviation probability distribution
        deviation_probs_type = target_distribution

        # Note: for lights and signs, INS already exist from adjacent poles INS
        # -> adjust deviation probs for remaining elements to keep target distribution
        if e_type in ['lights', 'signs'] and len(elements) > 0:
            deviation_probs_type = []

            # Get current lights or signs INS distribution (which is already >=0 due to adjacent poles INS)
            num_ins_type = sum([1 for d in map_deviations if d.type_current == e_type
                                and d.deviation_type == DT.INSERTION])
            ins_share = float(num_ins_type) / (len(elements) + num_ins_type)
            remaining_share = 1 - ins_share

            # Adjust probs so that target distribution is still reached as close as possible (on sample level)
            for i, d_type in enumerate(deviation_types[:-1]):
                if d_type == DT.INSERTION:
                    prob = max(target_distribution[i] - ins_share, 0) / remaining_share
                else:  # DEL, SUB
                    prob = target_distribution[i] / remaining_share
                deviation_probs_type.append(prob)

            if sum(deviation_probs_type) > 1:  # Happens in rare cases -> ensure probs sum up to <= 1
                deviation_probs_type = [t / sum(deviation_probs_type) for t in deviation_probs_type]

            deviation_probs_type.append(1 - sum(deviation_probs_type))  # fill up free prob with VER

        for elem in elements:
            # Randomly apply deviation according to target distribution
            deviation_type = np.random.choice(deviation_types, p=deviation_probs_type)

            if deviation_type == DT.DELETION:
                map_deviations.append(MapDeviation(DT.DELETION,
                                                   element_prior=None,
                                                   element_current=elem))

            elif deviation_type == DT.INSERTION:
                map_deviations.append(MapDeviation(DT.INSERTION,
                                                   element_prior=elem,
                                                   element_current=None))
                # For poles, also find and remove all adjacent signs and lights
                if e_type == 'poles':
                    adjacent_elements = get_adjacent_elements(elem,
                                                              elements_by_type['signs'] + elements_by_type['lights'])
                    for adj_elem in adjacent_elements:
                        map_deviations.append(MapDeviation(DT.INSERTION,
                                                           element_prior=adj_elem,
                                                           element_current=None))
                        adj_elem_type = type_to_group_lut[adj_elem['type']]
                        elements_by_type[adj_elem_type].remove(adj_elem)  # remove to prevent duplicate deviations

            elif deviation_type == DT.SUBSTITUTION and e_type in ['lights', 'signs']:
                elem_new = substitute_map_element(elem)
                map_deviations.append(MapDeviation(DT.SUBSTITUTION,
                                                   element_prior=elem_new,
                                                   element_current=elem))

            else:  # label everything else as VER
                map_deviations.append(MapDeviation(DT.VERIFICATION,
                                                   element_prior=elem,
                                                   element_current=elem))

    return map_deviations


def apply_deviations_to_point_cloud(map_deviations: List[MapDeviation], hdpc_pc: PointCloud) -> PointCloud:
    """ Apply given map deviations to point cloud (i.e., remove elements marked as insertion) """
    for deviation in map_deviations:
        if deviation.deviation_type == DT.INSERTION:
            hdpc_pc.points = remove_element_from_point_cloud(deviation.get_prior(), hdpc_pc.points)
    return hdpc_pc


def substitute_map_element(element: dict) -> dict:
    """ Substitutes the given map element by another element type and with random realistic properties.
        Note that only light <-> sign substitutions are supported.
    """
    # Copy basic attributes to new map element
    base_attributes = ['id', 'lat', 'lon', 'x_utm', 'y_utm', 'z_utm', 'x_vrf', 'y_vrf', 'z_vrf', 'valid_flag']
    height_attributes = ['height']
    yaw_attributes = ['yaw_utm', 'yaw_vrf']
    new_elem = {a: element[a] for a in base_attributes + height_attributes + yaw_attributes if a in element}

    # Change type and set random default size
    if element['type'] == 'TrafficLight':
        new_elem['type'] = 'TrafficSign'
        new_elem['width'] = np.random.uniform(0.5, 0.9)
        new_elem['height'] = np.random.uniform(0.3, 0.9)
    elif element['type'] == 'TrafficSign':
        new_elem['type'] = 'TrafficLight'
        new_elem['width'] = np.random.uniform(0.27, 0.33)
        new_elem['height'] = np.random.uniform(0.3, 0.9)
    else:
        raise KeyError(f"Type '{element['type']}' not supported for substitutions!")

    # Randomly adjust vertical position (max. 20cm off)
    z_offset = np.random.uniform(-0.2, 0.2)
    for z_attr in ['z_utm', 'z_vrf']:
        if z_attr in new_elem:
            new_elem[z_attr] += z_offset

    # Adjust orientation
    yaw_offset = 90  # degrees
    for attr in yaw_attributes:
        if attr in new_elem:
            new_elem[attr] = (new_elem[attr] + yaw_offset) % 360
    new_elem = limit_yaw_value([new_elem])[0]
    return new_elem


def get_map_deviations_from_sample_data(map_elements: List[dict], deviation_data: dict) -> List[MapDeviation]:
    """ Generates deviations from the given map elements and (generated) deviation data (instead of generating randomly
        from distribution)
    """
    map_deviations = []
    for elem in map_elements:
        elem_id = str(elem['id'])
        if elem_id in deviation_data:
            deviation_type = deviation_data[elem_id]
            if deviation_type == DT.DELETION.value:
                map_deviations.append(MapDeviation(DT.DELETION,
                                                   element_prior=None,
                                                   element_current=elem))
            elif deviation_type == DT.INSERTION.value:
                map_deviations.append(MapDeviation(DT.INSERTION,
                                                   element_prior=elem,
                                                   element_current=None))
            elif deviation_type == DT.SUBSTITUTION.value:
                elem_new = substitute_map_element(elem)
                map_deviations.append(MapDeviation(DT.SUBSTITUTION,
                                                   element_prior=elem_new,
                                                   element_current=elem))
            else:
                raise KeyError(f"Unknown Deviation Type '{deviation_type}'")
        else:
            map_deviations.append(MapDeviation(DT.VERIFICATION,
                                               element_prior=elem,
                                               element_current=elem))
    return map_deviations


def get_adjacent_elements(pole: dict, other_elements: List[dict]) -> List[dict]:
    """ Determines and returns list of map elements in proximity of a given pole
        (i.e., closer than 0.5 m in x-y-plane)
    """
    dist_threshold = 0.5
    adjacent_elements = []
    pole_x, pole_y = pole['x_utm'], pole['y_utm']
    for elem in other_elements:
        plane_dist = math.sqrt((pole_x - elem['x_utm']) ** 2 + (pole_y - elem['y_utm']) ** 2)
        if plane_dist < dist_threshold:
            adjacent_elements.append(elem)
    return adjacent_elements
