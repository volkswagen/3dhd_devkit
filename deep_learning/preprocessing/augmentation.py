""" Data augmentation for both point cloud and map.
"""

import numpy as np

from deep_learning.util import point_cloud_ops as pcops
from deep_learning.util import lookups
from deep_learning.preprocessing.map_fmt import limit_yaw_value
from utility import transformations as trafo


# Module functions #####################################################################################################
########################################################################################################################


def get_pc_crops_per_group(point_cloud, groups, augmentation_settings):
    """ Get local point cloud crops for augmentation groups.

    Args:
        point_cloud (np.ndarray): numpy array of points, keys: (x, y, z, intensity, z_over_ground)
        groups (list[dict]): list of groups, each group comprises a list of (local) pc_crops and elements
        augmentation_settings (dict): augmentation settings

    Returns:
        groups (list[dict]): list of groups, now comprising also the local pc crop
    """

    _, elem_type_lut = lookups.get_element_type_naming_lut()
    size_feature_lut = lookups.get_main_size_features()
    for group in groups:
        group_params = {}
        if augmentation_settings['group_augmentation']:
            # Get maximum extent of group
            x_min = 1000
            x_max = -1000
            y_min = 1000
            y_max = -1000
            width_max = -1000
            for element in group['elements']:
                x = element['x_vrf']
                y = element['y_vrf']

                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x

                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

                elem_type = elem_type_lut[element['type']]
                width = element[size_feature_lut[elem_type]]

                if width > width_max:
                    width_max = width

            # Save group parameters for later removal of points
            dx = x_max - x_min
            dy = y_max - y_min
            square_size = max(dx, dy) + width_max
            if square_size > 3:
                square_size = 3
            if square_size < 1:
                square_size = 1

            x = x_max - dx / 2
            y = y_max - dy / 2
            group_params['x'] = x
            group_params['y'] = y
            group_params['square_size'] = square_size

            # Crop point cloud
            points = pcops.get_crop_2d(point_cloud, x, y, square_size)
            if points['x'].shape[0] == 0:
                group['invalid_flag'] = True
            else:
                group['pc_crops'].append(points)
                group['group_params'] = group_params

        # Implement cropping of individual elements
        else:
            for element in group['elements']:
                pass

    groups = [g for g in groups if 'invalid_flag' not in g.keys()]

    return groups


def retrieve_elements_from_groups(groups):
    """ Extracts map elements from augmentation groups.
    Args:
         groups (list[dict]): list of groups, each group comprises a list of (local) pc_crops and elements

    Returns:
        elements (list[dict]: retrieved list of map elements
    """
    elements = []
    for group in groups:
        elements.extend(group['elements'])

    return elements


def remove_crops_from_point_cloud(point_cloud, groups, augmentation_settings):
    """ Removes local pc crops of augmentation groups from the larger point cloud.

    Args:
        point_cloud (np.ndarray): numpy array of points, keys: (x, y, z, intensity, z_over_ground)
        groups (list[dict]): list of groups, each group comprises a list of (local) pc_crops and elements
        augmentation_settings (dict): augmentation settings

    Returns:
        point_cloud (np.ndarray): modified point cloud with local groups being removed
    """
    for group in groups:
        if augmentation_settings['group_augmentation']:
            group_params = group['group_params']
            point_cloud = pcops.remove_points_2d(point_cloud, group_params['x'], group_params['y'],
                                                 group_params['square_size'])
        else:
            # Implement individual removal here
            pass

    return point_cloud


def insert_crops_into_point_cloud(point_cloud, groups):
    """ Inserts augmented local groups into point cloud.

    Args:
        point_cloud (np.ndarray): numpy array of points, keys: (x, y, z, intensity, z_over_ground)
        groups (list[dict]): list of groups, each group comprises a list of (local) pc_crops and elements

    Returns:
        point_cloud (np.ndarray): point cloud with inserted augmented local pc crops
    """
    for group in groups:
        for pc_crop in group['pc_crops']:
            point_cloud = np.concatenate((point_cloud, pc_crop))

    return point_cloud


def augment_points(points, x_noise, y_noise, z_noise, yaw_noise, scale_noise, random_flip, intensity_noise=None,
                   rot_around_centroid=False):
    """
    Args:
        points (np.ndarray): numpy array of points (local pc crops), keys: (x, y, z, intensity, z_over_ground)
        x_noise (float): shift in x
        y_noise (float): shift in y
        z_noise (float): shift in z
        yaw_noise (float): rotation noise
        scale_noise (float): scaling factor
        random_flip (float): flip probability
        intensity_noise (float): noise to add to intensity values
        rot_around_centroid (bool): enables the rotation around the centroid of points

    Returns:
        points (np.ndarray): augmented local pc crop
        centroid (np.array):  [x_mean, y_mean, z_mean] centroid of local pc crop
    """
    # Translation
    points['x'] += x_noise
    points['y'] += y_noise
    points['z'] += z_noise

    # Rotation
    centroid = None
    if rot_around_centroid:
        x_mean = np.mean(points['x'])
        y_mean = np.mean(points['y'])
        z_mean = np.mean(points['z'])
        centroid = [x_mean, y_mean, z_mean]

    points_xyz = np.row_stack((points['x'], points['y'], points['z']))
    points_xyz = trafo.rotate_points(points_xyz, yaw_noise, centroid=centroid)

    points['x'] = points_xyz[0, :]
    points['y'] = points_xyz[1, :]
    points['z'] = points_xyz[2, :]

    # Random flip
    if random_flip:
        points['y'] = -points['y']

    # Scaling
    points['x'] *= scale_noise
    points['y'] *= scale_noise
    points['z'] *= scale_noise

    if intensity_noise is not None:
        points['intensity'] = np.full_like(points['intensity'], intensity_noise)

    return points, centroid


def augment_map_elements(elements, x_noise, y_noise, z_noise, yaw_noise, scale_noise, random_flip,
                         centroid=None):
    """ Augments map elements.

    Args:
        elements (list[dict]): list of elements to augment
        x_noise (float): shift in x
        y_noise (float): shift in y
        z_noise (float): shift in z
        yaw_noise (float): rotation noise
        scale_noise (float): scaling factor
        random_flip (float): flip probability
        centroid (np.array):  [x_mean, y_mean, z_mean] centroid of local pc crop around which to rotate

    Returns:
         elements (list[dict]): augmented map elements
    """
    x = [e['x_vrf'] for e in elements]
    y = [e['y_vrf'] for e in elements]
    z = [e['z_utm'] for e in elements]

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Translation
    x += x_noise
    y += y_noise
    z += z_noise

    # Rotation
    points_xyz = np.row_stack((x, y, z))
    points_xyz = trafo.rotate_points(points_xyz, yaw_noise, centroid=centroid)

    # Flip
    if random_flip:
        points_xyz[1, :] = -points_xyz[1, :]

    # Scaling
    points_xyz[0, :] *= scale_noise
    points_xyz[1, :] *= scale_noise
    points_xyz[2, :] *= scale_noise

    _, elem_type_lut = lookups.get_element_type_naming_lut()
    size_feature_lut = lookups.get_main_size_features()
    for i, element in enumerate(elements):
        element['x_vrf'] = points_xyz[0, i]
        element['y_vrf'] = points_xyz[1, i]
        element['z_utm'] = points_xyz[2, i]

        elem_type = elem_type_lut[element['type']]
        size_feature = size_feature_lut[elem_type]
        element[size_feature] *= scale_noise

        if 'yaw_vrf' in element:
            element['yaw_vrf'] += yaw_noise
            if random_flip:
                element['yaw_vrf'] *= -1

    elements = limit_yaw_value(elements)

    return elements


def point_dropout(point_cloud, augmentation_settings):
    """ Randomly removes a subset of points.

    Args:
        point_cloud (np.ndarray): point cloud as numpy array, keys: (x, y, z, intensity, z_over_ground)
        augmentation_settings (dict): augmentation settings

    Returns:
        point_cloud (np.ndarray): point cloud with a subset of points removed
    """
    p = augmentation_settings['point_keep_prob']
    keep_indices = np.random.choice(a=[True, False], size=point_cloud.shape[0], p=[p, 1 - p])
    point_cloud = point_cloud[keep_indices]

    return point_cloud


def noise_per_point(point_cloud, augmentation_settings):
    """ Adds a noise (x,y,z) to each point of the point cloud.

    Args:
        point_cloud (np.ndarray): point cloud as numpy array, keys: (x, y, z, intensity, z_over_ground)
        augmentation_settings (dict): augmentation settings

    Returns:
        point_cloud (np.ndarray): point cloud with added noise
    """
    n_points = point_cloud.shape[0]
    std = augmentation_settings['noise_per_point_std']
    x_noise = np.random.normal(0, std, size=n_points)
    y_noise = np.random.normal(0, std, size=n_points)
    z_noise = np.random.normal(0, std, size=n_points)

    point_cloud['x'] += x_noise
    point_cloud['y'] += y_noise
    point_cloud['z'] += z_noise

    return point_cloud


def get_augmentation_groups(map_elements, augmentation_settings):
    """ Creates local groups of elements for joint augmentation.

    Args:
        map_elements (list[dict]): list of map elements to group
        augmentation_settings (dict): augmentation settings as provided by the train config

    Returns:
        groups (list[dict]): list of groups, each group comprises a list of (local) pc_crops and elements
    """
    groups = []
    association_threshold = augmentation_settings['association_threshold']  # threshold (dist) for grouping
    association_consider_z = augmentation_settings['association_consider_z']  # use z-dim for grouping

    # Create matrix for fast distance computation
    x_vec = [e['x_vrf'] for e in map_elements]
    y_vec = [e['y_vrf'] for e in map_elements]
    z_vec = [e['z_utm'] for e in map_elements]
    d_vec = np.zeros(len(x_vec))
    idx_vec = list(range(0, len(map_elements)))  # remember originating index, used as indicator which to keep
    # 0:x, 1:y, 2:z, 3:d, 4:i
    compare_array = np.column_stack((x_vec, y_vec, z_vec, d_vec, idx_vec))
    idx_used = []  # used element indices of original list

    # Iterate elements for grouping
    for i_element, element in enumerate(map_elements):
        if i_element in idx_used:
            continue
        # Create group
        group = {
            'pc_crops': [],
            'elements': []
        }

        group['elements'].append(element)
        idx_used.append(i_element)

        # Get position
        x = element['x_vrf']
        y = element['y_vrf']
        z = element['z_utm']

        # Compute distance to all other elements
        if not association_consider_z:
            d_vec = np.sqrt((x - compare_array[:, 0]) ** 2 + (y - compare_array[:, 1]) ** 2)
        else:
            d_vec = np.sqrt(
                (x - compare_array[:, 0]) ** 2 + (y - compare_array[:, 1]) ** 2 + (z - compare_array[:, 2]) ** 2)

        # Sort array by ascending order
        compare_array[:, 3] = d_vec
        compare_array = compare_array[compare_array[:, 3].argsort()]  # sort b

        for i in range(0, compare_array.shape[0]):
            idx = int(compare_array[i, 4])
            d = compare_array[i, 3]

            if idx in idx_used:
                continue

            # Append to group if close enough
            if d < association_threshold:
                idx_used.append(idx)
                group['elements'].append(map_elements[idx])

            # No close element for grouping left: break
            if d > association_threshold:
                break

        # Append to list of groups
        groups.append(group)

    return groups


def global_augmentation(point_cloud, groups, augmentation_settings):
    """ Globally augments both point cloud and map.

    Applies the same translation, rotation, flipping, and scaling to both point cloud and map elements.

    Args:
        point_cloud (np.ndarray): point cloud as numpy array, keys: (x, y, z, intensity, z_over_ground)
        groups (list[dict]): list of augmentation groups, each group comprises a list of (local) pc_crops and elements
        augmentation_settings (dict): augmentation settings as provided by the train config

    Returns:
        point_cloud (np.ndarray): augmented point cloud
        groups (list[dict]): augmented augmentation groups
    """
    # Get settings
    global_trans_x = augmentation_settings['global_trans_x']
    global_trans_y = augmentation_settings['global_trans_y']
    global_trans_z = augmentation_settings['global_trans_z']
    global_yaw_uni = augmentation_settings['global_yaw_uni']
    global_scaling = augmentation_settings['global_scaling']
    random_flip_prob = augmentation_settings['random_flip_prob']
    equal_intensity_prob = augmentation_settings['equal_intensity_prob']

    # Get global noise
    x_noise = np.random.uniform(-global_trans_x, global_trans_x)
    y_noise = np.random.uniform(-global_trans_y, global_trans_y)
    z_noise = np.random.uniform(-global_trans_z, global_trans_z)
    yaw_noise = np.random.uniform(-global_yaw_uni, global_yaw_uni)
    scale_noise = np.random.uniform(1 - global_scaling, 1 + global_scaling)
    random_flip = np.random.choice([True, False], p=[random_flip_prob, 1 - random_flip_prob])

    # Set all intensities to 0 (may let the network focus more on geometric features)
    apply_equal_intensity = np.random.choice([True, False], p=[equal_intensity_prob, 1 - equal_intensity_prob])
    i_max = 0.0  # 0.5
    intensity_noise = np.random.uniform(0, i_max) if apply_equal_intensity else None

    # Transform point cloud
    point_cloud, _ = augment_points(
        point_cloud, x_noise, y_noise, z_noise, yaw_noise, scale_noise, random_flip, intensity_noise)

    # Transform map elements
    for group in groups:
        group['elements'] = augment_map_elements(
            group['elements'], x_noise, y_noise, z_noise, yaw_noise, scale_noise, random_flip)

    return point_cloud, groups


def augment_groups(groups, augmentation_settings):
    """ Augmentation of local augmentation groups.

    Args:
        groups (list[dict]): list of augmentation groups, each group comprises a list of (local) pc_crops and elements
        augmentation_settings (dict): augmentation settings as provided by the train config

    Returns:
        groups (list[dict]): locally augmented groups
    """
    # Get settings
    local_t_center = augmentation_settings['local_t_center']
    local_yaw_uni = augmentation_settings['local_yaw_uni']
    local_scaling = augmentation_settings['local_scaling']
    random_flip_prob = augmentation_settings['random_flip_prob']

    # Augment groups
    for group in groups:
        if augmentation_settings['group_augmentation']:
            # One noise per group
            x_noise = np.random.normal(local_t_center)
            y_noise = np.random.normal(local_t_center)
            z_noise = np.random.normal(local_t_center)
            scale_noise = np.random.uniform(1 - local_scaling, 1 + local_scaling)
            yaw_noise = np.random.uniform(-local_yaw_uni, local_yaw_uni)

            random_flip = np.random.choice([True, False], replace=False, p=[random_flip_prob, 1 - random_flip_prob])

            # Transform point cloud
            group['pc_crops'][0], centroid = augment_points(group['pc_crops'][0], x_noise, y_noise, z_noise,
                                                            yaw_noise, scale_noise, random_flip,
                                                            rot_around_centroid=True)

            # Transform map elements
            group['elements'] = augment_map_elements(group['elements'], x_noise, y_noise, z_noise, yaw_noise,
                                                     scale_noise, random_flip, centroid)

        else:
            # implement individual augmentation here
            pass

    return groups
