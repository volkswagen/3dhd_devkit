""" Visualization of anchors.
"""

from typing import List

import numpy as np

from dataset.map_deviation import get_deviation_classes
from deep_learning.losses import target_generator
from visualization import visualization_elements as vis_elements
from visualization.visualization_elements import get_element_color


# Module functions #####################################################################################################
########################################################################################################################


def visualize_anchor_grid_3d(element_types: List[str], target_dict: dict,
                             target_gen: target_generator.TargetGenerator, unnorm=True, dd_model=False):
    """ Visualizes anchor grids (poles, signs, lights) with matching states (match, don't care, no match).

    Match: anchor contains part of a real world object
    Don't care: anchor is insufficiently overlapping, exclude for loss computation
    No match: anchor is not overlapping with a real-world object

    Args:
        element_types (list[str]): list of element types for which to visualize the anchor grid
        target_dict (dict:tensor): target dictionary containing target tensors per element type
        target_gen (losses.TargetGenerator):
        unnorm (bool): True if anchors should be displayed according to their regression features
        dd_model (bool): True if MDD-M anchors (stage 3) are to be visualized (evaluation state)
    """
    for element_type in element_types:
        anc = target_dict[element_type]['anc']
        task = target_dict[element_type]['task']
        reg = target_dict[element_type]['reg']
        mask = target_dict[element_type]['mask']
        norm_data = target_gen.get_norm_data(element_type)

        if element_type == 'poles':
            visualize_pole_anchor_grid_3d(anc, task, reg, mask, norm_data, unnorm, dd_model)
        elif element_type == 'signs':
            visualize_sign_anchor_grid_3d(anc, task, reg, mask, norm_data, unnorm, dd_model)
        elif element_type == 'lights':
            visualize_light_anchor_grid_3d(anc, task, reg, mask, norm_data, unnorm, dd_model)
        else:
            raise ValueError(f"Not defined for element type '{element_type}'")


def visualize_pole_anchor_grid_3d(anc, task, reg, mask, norm_data, unnorm=True, dd_model=False):
    """ Visualizes poles anchors using Mayavi in 3D. Can be overlaid with the point cloud.

    Args:
        anc (float tensor): tensor containing anchor positions, rest is zeros, [R, Z, X, Y]
        task (float tensor): task tensor [C, Z, X, Y], C containing score values (e.g., class scores)
        reg (float tensor): regression tensor [R, Z, X, Y] with encoded regression features in R
        mask (float tensor): mask tensor [1, Z, X, Y] with 0 being "don't care"
        norm_data (dict:float): data for unnormalizing anchor regression features
        unnorm (bool): True if anchors should be displayed according to their regression features (overlapping matches)
        dd_model: True if MDD-M anchors (stage 3) are to be visualized (with evaluation state)

    Returns:
        visus_poles (list[visualization_elements.Pole]): list of pole visualization objects
    """

    dev_classes = get_deviation_classes(include_aggregate_classes=False)
    has_none_class = task.shape[0] == len(dev_classes) + 1

    # Plot anchor if single class is set
    if has_none_class:
        task_max = np.max(task[:-1, :, :, :], axis=0, keepdims=True)
    else:
        task_max = np.max(task, axis=0, keepdims=True)

    matched = (task_max[0] == 1)
    # unmatched = (task_max[0] == 0)
    excluded = (mask[0] == 0)

    matched_indices = matched.nonzero()  # shape: [0, N_lay, N_x, N_y]
    # unmatched_indices = unmatched.nonzero()  # [0, N_lay, N_x, N_y]
    excluded_indices = excluded.nonzero()  # [0, N_lay, N_x, N_y]

    color_map = {
        'matched': (0, 1, 0),
        'unmatched': (1, 0, 0),
        'excluded': (0, 1, 1)
    }

    tags = ['matched', 'excluded']
    groups = [matched_indices, excluded_indices]
    colors = [color_map['matched'], color_map['excluded']]

    # Plot anchors
    visus_poles = []
    for indices, color, tag in zip(groups, colors, tags):
        num_entries = len(indices[0])

        for i_entry in range(0, num_entries):
            i_l = indices[0][i_entry]
            i_x = indices[1][i_entry]
            i_y = indices[2][i_entry]

            # Visualize anchors only by their default position without including regression features
            if not unnorm:
                pole_data = {
                    'x_vrf': anc[0][i_l][i_x][i_y],
                    'y_vrf': anc[1][i_l][i_x][i_y],
                    'z_vrf': anc[2][i_l][i_x][i_y],
                    'diameter': norm_data['pole_default_diameter']
                }
            # Visualizes anchors with set regression features (matching anchors overlap).
            else:
                if tag == 'matched':
                    pole_data = {
                        'x_vrf': reg[0][i_l][i_x][i_y],
                        'y_vrf': reg[1][i_l][i_x][i_y],
                        'z_vrf': reg[2][i_l][i_x][i_y],
                        'diameter': reg[3][i_l][i_x][i_y] + 0.05
                    }
                    norm_data['anchor_x'] = anc[0][i_l][i_x][i_y]
                    norm_data['anchor_y'] = anc[1][i_l][i_x][i_y]
                    norm_data['anchor_z'] = anc[2][i_l][i_x][i_y]

                    pole_data = target_generator.unnormalize_pole_data(pole_data, norm_data)
                else:
                    pole_data = {
                        'x_vrf': anc[0][i_l][i_x][i_y],
                        'y_vrf': anc[1][i_l][i_x][i_y],
                        'z_vrf': anc[2][i_l][i_x][i_y],
                        'diameter': norm_data['pole_default_diameter']
                    }

            if dd_model and tag == 'matched':
                cls_idx = np.argmax(task[:, i_l, i_x, i_y])
                color = get_element_color(deviation_type=dev_classes[cls_idx])

            vis_pole = vis_elements.Pole(pole_data['x_vrf'], pole_data['y_vrf'], pole_data['z_vrf'],
                                         radius=pole_data['diameter'] / 2,
                                         color=color)
            vis_pole.plot()
            visus_poles.append(vis_pole)

    return visus_poles


def visualize_sign_anchor_grid_3d(anc, task, reg, mask, norm_data, unnorm=True, dd_model=False):
    """ Visualizes sign anchors using Mayavi in 3D. Can be overlaid with the point cloud.

    Args:
        anc (float tensor): tensor containing anchor positions, rest is zeros, [R, Z, X, Y]
        task (float tensor): task tensor [C, Z, X, Y], C containing score values (e.g., class scores)
        reg (float tensor): regression tensor [R, Z, X, Y] with encoded regression features in R
        mask (float tensor): mask tensor [1, Z, X, Y] with 0 being "don't care"
        norm_data (dict:float): data for unnormalizing anchor regression features
        unnorm (bool): True if anchors should be displayed according to their regression features (overlapping matches)
        dd_model: True if MDD-M anchors (stage 3) are to be visualized (with evaluation state)

    Returns:
        visus_signs (list[visualization_elements.TrafficSign]): list of sign visualization objects
    """
    # Settings
    dev_classes = get_deviation_classes(include_aggregate_classes=False)
    has_none_class = task.shape[0] == len(dev_classes) + 1

    # Plot anchor if single class is set
    if has_none_class:
        task_max = np.max(task[:-1, :, :, :], axis=0, keepdims=True)
    else:
        task_max = np.max(task, axis=0, keepdims=True)

    matched = (task_max[0] == 1)
    # unmatched = (task_max[0] == 0)
    excluded = (mask[0] == 0)

    matched_indices = matched.nonzero()  # shape: [0, N_lay, N_x, N_y]
    # unmatched_indices = unmatched.nonzero()  # [0, N_lay, N_x, N_y]
    excluded_indices = excluded.nonzero()  # [0, N_lay, N_x, N_y]

    color_map = {
        'matched': (0, 1, 0),
        'unmatched': (1, 0, 0),
        'excluded': (0, 1, 1)
    }

    tags = ['matched', 'excluded']
    groups = [matched_indices, excluded_indices]
    colors = [color_map['matched'], color_map['excluded']]

    # Plot anchors
    visus_signs = []
    for indices, color, tag in zip(groups, colors, tags):
        num_entries = len(indices[0])

        for i_entry in range(0, num_entries):
            i_l = indices[0][i_entry]
            i_x = indices[1][i_entry]
            i_y = indices[2][i_entry]

            if not unnorm:
                sign_data = {
                    'x_vrf': anc[0][i_l][i_x][i_y],
                    'y_vrf': anc[1][i_l][i_x][i_y],
                    'z_vrf': anc[2][i_l][i_x][i_y],
                    'width': norm_data['sign_default_width'],
                    'height': norm_data['sign_default_height'],
                    'yaw_vrf': norm_data['sign_default_yaw']
                }
            else:
                # Only unnorm matched anchors
                if tag == 'matched':
                    sign_data = {
                        'x_vrf': reg[0][i_l][i_x][i_y],
                        'y_vrf': reg[1][i_l][i_x][i_y],
                        'z_vrf': reg[2][i_l][i_x][i_y],
                        'width': reg[3][i_l][i_x][i_y] + 0.05,
                        'height': reg[4][i_l][i_x][i_y] + 0.05,
                        'yaw_sin': reg[5][i_l][i_x][i_y],
                        'yaw_cos': reg[6][i_l][i_x][i_y]
                    }
                    norm_data['anchor_x'] = anc[0][i_l][i_x][i_y]
                    norm_data['anchor_y'] = anc[1][i_l][i_x][i_y]
                    norm_data['anchor_z'] = anc[2][i_l][i_x][i_y]

                    sign_data = target_generator.unnormalize_sign_data(sign_data, norm_data)
                else:
                    sign_data = {
                        'x_vrf': anc[0][i_l][i_x][i_y],
                        'y_vrf': anc[1][i_l][i_x][i_y],
                        'z_vrf': anc[2][i_l][i_x][i_y],
                        'width': norm_data['sign_default_width'],
                        'height': norm_data['sign_default_height'],
                        'yaw_vrf': norm_data['sign_default_yaw']
                    }

            if dd_model and tag == 'matched':
                cls_idx = np.argmax(task[:, i_l, i_x, i_y])
                color = get_element_color(deviation_type=dev_classes[cls_idx])

            vis_sign = vis_elements.TrafficSign(sign_data['x_vrf'], sign_data['y_vrf'], sign_data['z_vrf'],
                                                cls=None,
                                                width=sign_data['width'],
                                                height=sign_data['height'],
                                                yaw=sign_data['yaw_vrf'],
                                                color=color)
            vis_sign.plot()
            visus_signs.append(vis_sign)

    return visus_signs


def visualize_light_anchor_grid_3d(anc, task, reg, mask, norm_data, unnorm=True, dd_model=False):
    """ Visualizes light anchors using Mayavi in 3D. Can be overlaid with the point cloud.

    Args:
        anc (float tensor): tensor containing anchor positions, rest is zeros, [R, Z, X, Y]
        task (float tensor): task tensor [C, Z, X, Y], C containing score values (e.g., class scores)
        reg (float tensor): regression tensor [R, Z, X, Y] with encoded regression features in R
        mask (float tensor): mask tensor [1, Z, X, Y] with 0 being "don't care"
        norm_data (dict:float): data for unnormalizing anchor regression features
        unnorm (bool): True if anchors should be displayed according to their regression features (overlapping matches)
        dd_model: True if MDD-M anchors (stage 3) are to be visualized (with evaluation state)

    Returns:
        visus_lights (list[visualization_elements.TrafficLight]): list of light visualization objects
    """
    # Settings
    dev_classes = get_deviation_classes(include_aggregate_classes=False)
    has_none_class = task.shape[0] == len(dev_classes) + 1

    # Plot anchor if single class is set
    if has_none_class:
        task_max = np.max(task[:-1, :, :, :], axis=0, keepdims=True)
    else:
        task_max = np.max(task, axis=0, keepdims=True)

    matched = (task_max[0] == 1)
    # unmatched = (task_max[0] == 0)
    excluded = (mask[0] == 0)

    matched_indices = matched.nonzero()  # shape: [0, N_lay, N_x, N_y]
    # unmatched_indices = unmatched.nonzero()  # [0, N_lay, N_x, N_y]
    excluded_indices = excluded.nonzero()  # [0, N_lay, N_x, N_y]

    color_map = {
        'matched': (0, 1, 0),
        'unmatched': (1, 0, 0),
        'excluded': (0, 1, 1)
    }

    tags = ['matched', 'excluded']
    groups = [matched_indices, excluded_indices]
    colors = [color_map['matched'], color_map['excluded']]

    # Plot anchors
    visus_lights = []
    for indices, color, tag in zip(groups, colors, tags):
        num_entries = len(indices[0])

        for i_entry in range(0, num_entries):
            i_l = indices[0][i_entry]
            i_x = indices[1][i_entry]
            i_y = indices[2][i_entry]

            if not unnorm:
                light_data = {
                    'x_vrf': anc[0][i_l][i_x][i_y],
                    'y_vrf': anc[1][i_l][i_x][i_y],
                    'z_vrf': anc[2][i_l][i_x][i_y],
                    'width': norm_data['light_default_width'],
                    'height': norm_data['light_default_height'],
                    'yaw_vrf': norm_data['light_default_yaw']
                }
            else:
                # Only unnorm matched anchors
                if tag == 'matched':
                    light_data = {
                        'x_vrf': reg[0][i_l][i_x][i_y],
                        'y_vrf': reg[1][i_l][i_x][i_y],
                        'z_vrf': reg[2][i_l][i_x][i_y],
                        'width': reg[3][i_l][i_x][i_y] + 0.05,
                        'height': reg[4][i_l][i_x][i_y] + 0.05,
                        'yaw_sin': reg[5][i_l][i_x][i_y],
                        'yaw_cos': reg[6][i_l][i_x][i_y]
                    }
                    norm_data['anchor_x'] = anc[0][i_l][i_x][i_y]
                    norm_data['anchor_y'] = anc[1][i_l][i_x][i_y]
                    norm_data['anchor_z'] = anc[2][i_l][i_x][i_y]

                    light_data = target_generator.unnormalize_light_data(light_data, norm_data)
                else:
                    light_data = {
                        'x_vrf': anc[0][i_l][i_x][i_y],
                        'y_vrf': anc[1][i_l][i_x][i_y],
                        'z_vrf': anc[2][i_l][i_x][i_y],
                        'width': norm_data['light_default_width'],
                        'height': norm_data['light_default_height'],
                        'yaw_vrf': norm_data['light_default_yaw']
                    }

            if dd_model and tag == 'matched':
                cls_idx = np.argmax(task[:, i_l, i_x, i_y])
                color = get_element_color(deviation_type=dev_classes[cls_idx])

            vis_light = vis_elements.TrafficLight(light_data['x_vrf'], light_data['y_vrf'], light_data['z_vrf'],
                                                  width=light_data['width'],
                                                  depth=light_data['width'],
                                                  height=light_data['height'],
                                                  yaw=light_data['yaw_vrf'],
                                                  color=color)
            vis_light.plot()
            visus_lights.append(vis_light)

    return visus_lights
