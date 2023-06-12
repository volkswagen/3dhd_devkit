""" Point cloud modifications creating insertions, occlusions, or reducing point density.
"""

from typing import List
from random import random, choice

import numpy as np

from dataset.point_cloud import PointCloud
from dataset.map_deviation import MapDeviation, DeviationTypes
from deep_learning.losses.target_generator import calculate_line_segment, point_to_line_dist
from deep_learning.util import lookups
from deep_learning.util.lookups import create_pole_height_lut


def get_occlusion_types():
    return ['top', 'bottom', 'left', 'right']


def generate_occlusion_setting(map_deviations: List[MapDeviation], occlusion_prob: float):
    """ Sets the occlusion type (top, bottom, left, or right) for a map deviation.

    Args:
        map_deviations (list[MapDeviation]): list of map deviation objects (dataset.map_deviation.pty)
        occlusion_prob (float): probability of the deviation being assigned an occlusion state

    Returns:
        map_deviations (list[MapDeviation]): list of map deviations with assigned occlusion state
    """
    # Save occlusion type as MapDeviation attribute
    for deviation in map_deviations:
        if random() < occlusion_prob and deviation.deviation_type != DeviationTypes.INSERTION:
            occlusion_type = choice(get_occlusion_types())
            deviation.occlusion_type = occlusion_type

    return map_deviations


def get_occlusions_from_sample_data(map_deviations: List[MapDeviation], occlusion_data: dict) -> List[MapDeviation]:
    """ Maps loaded occlusion assignments to elements.

    If occlusion assignments have been generated in advance (loaded by dataset_splitter),
    the occlusion state is here assigned to the respective map deviation.

    Args:
        map_deviations (list[MapDeviation]): list of map deviations
        occlusion_data (dict): element id -> occlusion type

    Returns:
        map_deviations (list[MapDeviation]): list of map deviations with assigned occlusion state
    """
    for deviation in map_deviations:
        elem_id = str(deviation.get_most_recent()['id'])
        if elem_id in occlusion_data:
            occlusion_type = occlusion_data[elem_id]
            if occlusion_type in get_occlusion_types():
                deviation.occlusion_type = occlusion_type
            else:
                raise KeyError(f"Unknown occlusion type '{occlusion_type}'")
    return map_deviations


def apply_point_density(hdpc_pc: PointCloud, point_density: float):
    """ Randomly downsamples a point cloud.

    Args:
        hdpc_pc (PointCloud): point cloud object
        point_density (float): probability of keeping a point

    Returns:
        hdpc_pc (PointCloud): downsampled point cloud
    """
    points = hdpc_pc.points
    keep_indices = np.random.choice(a=[True, False], size=points.shape[0], p=[point_density, 1 - point_density])
    hdpc_pc.points = points[keep_indices]
    return hdpc_pc


def apply_occlusions(map_deviations: List[MapDeviation], hdpc_pc: PointCloud):
    """ Induces partial occlusions into the point cloud.

    Args:
        map_deviations (list[MapDeviation]): list of map deviations
        hdpc_pc (PointCloud) point cloud object

    Returns:

    """
    for deviation in map_deviations:
        if deviation.occlusion_type is not None:
            elem = deviation.get_most_recent()
            hdpc_pc.points = remove_element_from_point_cloud(elem, hdpc_pc.points, deviation.occlusion_type)
    return hdpc_pc


def remove_element_from_point_cloud(element: dict, points: dict, occlusion_type=None):
    """ Removes the whole element or the occluded part from a point cloud.

    Args:
        element (dict): element to be removed
        points (np.ndarray): ndarray of points, keys: x,y,z,intensity
        occlusion_type (str): occlusion state of the element

    Returns:
        points (np.ndarray): ndarray of points without removed element
    """

    _, type_lut = lookups.get_element_type_naming_lut()
    e_type = type_lut[element['type']]

    center = [element['x_vrf'], element['y_vrf'], element['z_vrf']]
    size_x = element[lookups.get_main_size_features()[e_type]] + 0.1
    size_y = size_x
    size_z = element['height'] + 0.1 if 'height' in element else None

    if e_type == 'poles':
        size_x += 0.2
        size_y += 0.2
        size_z = 6.0
        center[2] += size_z / 2
        center[2] += 0.1  # lift pole a little to prevent cutting off ground points
    elif e_type == 'lights':
        size_x *= 1.41  # get rotational object radius for more thorough cropping
        size_y = size_x

    if occlusion_type in ['top', 'bottom']:
        # Halve size_z and move center up or down
        if e_type == 'poles':  # use custom pole heights for part-occlusions
            size_z = create_pole_height_lut()[element['cls']]
            center[2] = element['z_vrf'] + size_z / 2 + 0.1
        size_z /= 2
        up_or_down = 1 if occlusion_type == 'top' else -1
        center[2] += size_z / 2 * up_or_down

    if occlusion_type in ['left', 'right']:
        # poles, lights: halve size_x and move center along x-axis (left: positive, right: negative)
        # signs: halve size_x and size_y, move center to middle between center and edge point
        if e_type == 'signs':
            size_x /= 2
            size_y /= 2
            xy_line = calculate_line_segment(center[0], center[1], element['width'], element['yaw_vrf'])
            edge_point = xy_line[0] if occlusion_type == 'left' else xy_line[1]
            center[0] = (center[0] + edge_point[0]) / 2
            center[1] = (center[1] + edge_point[1]) / 2
        else:
            size_x /= 2
            left_or_right = 1 if occlusion_type == 'left' else -1
            center[0] += size_x / 2 * left_or_right

    # Crop element along orthogonal axes
    extent_x = [center[0] - size_x / 2, center[0] + size_x / 2]
    extent_y = [center[1] - size_y / 2, center[1] + size_y / 2]
    extent_z = [center[2] - size_z / 2, center[2] + size_z / 2]

    mask = (points['x'] > extent_x[0]) * (points['x'] < extent_x[1]) * \
           (points['y'] > extent_y[0]) * (points['y'] < extent_y[1]) * \
           (points['z_over_ground'] > extent_z[0]) * (points['z_over_ground'] < extent_z[1])

    if e_type == 'signs':
        xy_line = calculate_line_segment(center[0], center[1], element['width'], element['yaw_vrf'])
        mask = np.array([False if not pre_masked else point_to_line_dist(np.array([x, y]), xy_line) < 0.1
                         for x, y, pre_masked in zip(points['x'], points['y'], mask)])
    points = points[~mask]

    return points
