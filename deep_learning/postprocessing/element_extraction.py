""" Post-processing for the object detection DNN.
"""

import numpy as np

from dataset.map_deviation import get_deviation_classes
from deep_learning.losses import target_generator
from deep_learning.util import lookups


def convert_grid_to_element_lists(out_dict, target_dict, model_config):
    """ Converts the output grid to lists of map elements (object detection DNN).

    Args:
        out_dict (dict:tensor): contains output tensors of network per element type
        target_dict (dict:tensor): contains target tensors per element type
        model_config (dict): contains model settings

    Returns:
        nms_filtered_lists(dict:list) contains lists of predictions (elements) for element types
    """

    element_types = model_config['configured_element_types']
    _, idx_to_cls_luts = lookups.get_class_to_index_lookup(model_config)
    group_to_type_lut, _ = lookups.get_element_type_naming_lut()
    dev_classes = get_deviation_classes(include_aggregate_classes=False)
    predictions_lists = {}

    # Get values for unnormalization
    norm_data = {el: target_generator.get_norm_data(model_config, el) for el in element_types}

    # Generate object lists
    for e_type in element_types:
        # Init object list
        predictions_lists[e_type] = []

        # Get layers and reshape predictions (task + reg)
        task = out_dict[e_type]['task']   # [batch, classes, L, Z, X, Y]
        reg = out_dict[e_type]['reg']     # [batch, C, Z, X, Y], C is depending on the element_type
        anc = target_dict[e_type]['anc']

        task = np.moveaxis(task, 1, -1)  # -> [batch, L, X, Y, C] for easier iteration
        reg = np.moveaxis(reg, 1, -1)
        anc = np.moveaxis(anc, 1, -1)

        # Create array of scores and cell indices for sorting, filtering, and iterating
        task_layer_max = np.max(task, axis=4, keepdims=True)
        indices = np.indices(task_layer_max.shape)  # -> [index, batch, Z, X, Y, C]
        grid_indices = np.column_stack((task_layer_max.flatten(),   # score
                                        indices[0].flatten(),       # batch
                                        indices[1].flatten(),       # Z
                                        indices[2].flatten(),       # X
                                        indices[3].flatten()))      # Y

        # Sort by first column (score)
        # [N, 5(score, b, z, x, y)]
        grid_indices = grid_indices[grid_indices[:, 0].argsort()[::-1]]  # [::-1] for reverse (descending)

        # Consider only first 5000 entries for speed
        grid_indices = grid_indices[0:5000, :]

        # Retrieve elements from grid
        for row in range(0, grid_indices.shape[0]):
            score = grid_indices[row, 0]
            if score < .05:
                break

            ib = int(grid_indices[row, 1])
            il = int(grid_indices[row, 2])
            ix = int(grid_indices[row, 3])
            iy = int(grid_indices[row, 4])

            cls = None
            dev_type = None
            if model_config['deviation_detection_model']:
                cls_idx = np.argmax(task[ib, il, ix, iy, :])
                dev_type = dev_classes[cls_idx]
            elif model_config['classification_active']:
                cls_idx = np.argmax(task[ib, il, ix, iy, :])
                cls = idx_to_cls_luts[e_type][cls_idx]

            element_data = {
                'type': group_to_type_lut[e_type],
                'score': score,
                'class': cls,
                'x_vrf': reg[ib][il][ix][iy][0],
                'y_vrf': reg[ib][il][ix][iy][1],
                'z_vrf': reg[ib][il][ix][iy][2]
            }

            if dev_type is not None:
                element_data['dev_type'] = dev_type

            if e_type == 'poles':
                element_data.update({
                    'diameter': reg[ib][il][ix][iy][3]
                })
            elif e_type == 'signs':
                element_data.update({
                    'width': reg[ib][il][ix][iy][3],
                    'height': reg[ib][il][ix][iy][4],
                    'yaw_sin': reg[ib][il][ix][iy][5],
                    'yaw_cos': reg[ib][il][ix][iy][6],
                })
            elif e_type == 'lights':
                element_data.update({
                    'width': reg[ib][il][ix][iy][3],
                    'height': reg[ib][il][ix][iy][4],
                    'yaw_sin': reg[ib][il][ix][iy][5],
                    'yaw_cos': reg[ib][il][ix][iy][6],
                })

            # Unnormalize
            norm_data[e_type]['anchor_x'] = anc[ib][il][ix][iy][0]
            norm_data[e_type]['anchor_y'] = anc[ib][il][ix][iy][1]
            norm_data[e_type]['anchor_z'] = anc[ib][il][ix][iy][2]
            element_data = target_generator.unnormalize_element_data(e_type, element_data, norm_data[e_type])

            predictions_lists[e_type].append(element_data)

    return predictions_lists


def nms(predictions_lists, configs, thresholds: dict = None):
    """ Non-maximum-suppression (NMS) to filter overlapping elements.

    Args:
        predictions_lists (dict:list): contains list of predicted elements per element type
        configs (dict:dict): contains all model, train, and system settings
        thresholds (dict): thresholds (either per element type for object detection or per type and evaluation state)

    Returns:
        nms_filtered_lists (dict:list): contains nms filtered list of elements per element type
    """

    # Get default thresholds of none are available (otherwise obtained from validation)

    if thresholds is None:
        thresholds = {e_type: configs['train']['threshold_score'] for e_type in predictions_lists.keys()}
    nms_filtered_lists = {}

    # Filter predictions for each type
    for e_type, predictions in predictions_lists.items():
        nms_filtered_lists[e_type] = []
        # Only take what's needed for NMS filtering
        attributes = ['score', 'x_vrf', 'y_vrf', 'z_vrf']
        if e_type == 'lights':
            attributes += ['width', 'height']
        elif e_type == 'poles':
            attributes += ['diameter']
        elif e_type == 'signs':
            attributes += ['width', 'height', 'yaw_vrf']

        candidates = lookups.dict_list_to_np_array(predictions, attributes, include_idx=True)
        keep_indices = []

        # Filter by threshold
        score_mask = np.argwhere(candidates[:, 0] >= thresholds[e_type]).reshape(-1)
        candidates = candidates[score_mask]

        # Sort predictions by score
        candidates = candidates[candidates[:, 0].argsort()[::-1]]  # [::-1] reverses to descending order
        candidates = candidates[:1000]  # consider only first 1000

        # NMS
        while candidates.shape[0] > 0:
            # Take first candidate with the highest score
            candidate = candidates[0]
            keep_indices.append(candidate[-1])  # append original index to keep

            # Compare with other candidates
            candidates = candidates[1:]
            dx = candidates[:, 1] - candidate[1]
            dy = candidates[:, 2] - candidate[2]
            # dz = candidates[:, 3] - candidate[3]

            # Filter out close candidates
            if e_type == 'lights':
                # get xy overlaps
                d_xy = np.sqrt(dx ** 2 + dy ** 2)
                iosas = [target_generator.iosa_two_circles(dist, size, candidate[4]) for dist, size in
                        zip(d_xy, candidates[:, 4])]

                # get z overlaps
                z_overlaps = [target_generator.line_overlap(p1=z, len1=height, p2=candidate[3], len2=candidate[5])
                              for z, height in zip(candidates[:, 3], candidates[:, 5])]

                # create mask
                light_iosa_threshold = configs['model']['light_iosa_threshold']
                light_z_foreground_threshold = configs['model']['light_z_foreground_threshold']
                mask = [not (iosa >= light_iosa_threshold and z_overlap >= light_z_foreground_threshold) for
                        iosa, z_overlap in zip(iosas, z_overlaps)]

            elif e_type == 'poles':
                mask = np.sqrt(dx ** 2 + dy ** 2) > configs['train']['threshold_association_distance']

            elif e_type == 'signs':
                # Get xy distance to line
                line_seg = target_generator.calculate_line_segment(candidate[1], candidate[2], candidate[4],
                                                                   candidate[6])
                dists = [target_generator.point_to_line_dist(np.array([x, y]), line_seg) for x, y in
                         zip(candidates[:, 1], candidates[:, 2])]

                # Get z overlaps
                z_overlaps = [target_generator.line_overlap(p1=z, len1=height, p2=candidate[3], len2=candidate[5])
                              for z, height in zip(candidates[:, 3], candidates[:, 5])]

                # Create mask
                dist_threshold = configs['model']['sign_distance_threshold']
                sign_z_foreground_threshold = configs['model']['sign_z_foreground_threshold']
                mask = [not (dist <= dist_threshold and z_overlap >= sign_z_foreground_threshold) for
                        dist, z_overlap in zip(dists, z_overlaps)]

            else:
                raise ValueError(f"Not defined for element_type '{e_type}'")

            candidates = candidates[mask]

        # Keep only valid entries
        keep_indices = [int(i) for i in keep_indices]
        nms_filtered_lists[e_type] = [predictions[i] for i in keep_indices]

    return nms_filtered_lists
