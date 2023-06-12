""" Post-processing for the deviation detection DNN.
"""

import numpy as np

from dataset.map_deviation import get_deviation_classes, DeviationTypes, MapDeviation
from deep_learning.losses import target_generator
from deep_learning.postprocessing.element_extraction import nms
from deep_learning.util import lookups
from deep_learning.util.lookups import init_nested_dict


def lut_based_post_processing(out_dict, target_dict, configs, map_lut):
    """ Retrieves lists of predicted deviations from the network output for MDD-M (stage 3).

    The post-processing is based on a generated lookup table (LUT) during map encoding
    (deeplearning.preprocessing.map_fmt).

    Args:
        out_dict (dict:tensor): contains output tensors of network per element type
        target_dict (dict:tensor): contains target tensors per element type
        configs (dict:dict): contains all model, train, and system settings:
        map_lut (dict:dict: contains LUT information (map_lut, conflicting_elements, stats) per element type

    Returns:
        predictions_lists_nms (dict:list): contains list of predicted deviations per element type
    """

    # Initialize
    element_types = configs['model']['configured_element_types']
    group_to_type_lut, type_to_group_lut = lookups.get_element_type_naming_lut()
    dev_classes = get_deviation_classes(include_aggregate_classes=False)

    predictions_list_lut = []  # predictions for existing map hypotheses
    predictions_lists_deletions = {}
    predictions_lists_lut = {}
    predictions_voxels = []     # only for visualization purposes
    for e_type in element_types:
        predictions_lists_deletions[e_type] = []

    # Get values for unnormalization
    norm_data = {el: target_generator.get_norm_data(configs['model'], el) for el in element_types}

    # Indices (channels in tensors) for deviation states
    i_cor = dev_classes.index(DeviationTypes.VERIFICATION.value)
    i_ins = dev_classes.index(DeviationTypes.INSERTION.value)
    i_sub = dev_classes.index(DeviationTypes.SUBSTITUTION.value)
    i_del = dev_classes.index(DeviationTypes.DELETION.value)

    map_lut = map_lut['lut']
    for e_type in map_lut:
        task = out_dict[e_type]['task']  # [batch, scores, Z, X, Y]
        reg = out_dict[e_type]['reg']
        anc = target_dict[e_type]['anc']

        # For substitutions: the head of the element type present in the point cloud predicts the substitution
        # E.g., light in map, sign in point cloud -> get the substitution score from the sign head
        task_sub = None
        reg_sub = None
        if e_type == 'signs':
            task_sub = out_dict['lights']['task']
            reg_sub = out_dict['lights']['reg']
        elif e_type == 'lights':
            task_sub = out_dict['signs']['task']
            reg_sub = out_dict['signs']['reg']

        task = np.moveaxis(task, 1, -1)  # -> [batch, Z, X, Y, C], C: channel
        reg = np.moveaxis(reg, 1, -1)
        anc = np.moveaxis(anc, 1, -1)
        if task_sub is not None:
            task_sub = np.moveaxis(task_sub, 1, -1)
            reg_sub = np.moveaxis(reg_sub, 1, -1)

        # First, for evaluation states: verification, insertion, substitution
        # Get predictions for existing map hypotheses using the LUT generated during map feature map creation
        # Query voxels for each element in LUT to obtain a single prediction
        for elem_id in map_lut[e_type]:
            ib = 0  # batch index, always zero for evaluation
            voxels = map_lut[e_type][elem_id]  # list of (x, y, z)

            # Poles: only predicted in 2D, map vertical voxels to BEV
            if e_type == 'poles':
                voxels = [(voxel[0], voxel[1], 0) for voxel in voxels]
                voxels = list(set(voxels))  # remove duplicates

            score_dict = {'VER': [], 'INS': [], 'SUB': []}  # dict of scores (originating from multiple voxels)
            preds_dict = {'VER': [], 'INS': [], 'SUB': []}  # dict of predictions

            for _, voxel in enumerate(voxels):
                # Collect prediction from each voxel predicting the element
                ix = voxel[0]
                iy = voxel[1]
                iz = voxel[2]

                if e_type == 'poles':
                    s_cor = task[ib][iz][ix][iy][i_cor]       # VER
                    s_ins = task[ib][iz][ix][iy][i_ins]       # INS
                    s_sub = 0.0
                else:
                    s_cor = task[ib][iz][ix][iy][i_cor]       # VER
                    s_ins = task[ib][iz][ix][iy][i_ins]       # INS
                    s_sub = task_sub[ib][iz][ix][iy][i_sub]   # SUB

                scores = [s_cor, s_ins, s_sub]
                s_max = np.max(scores)          # maximum score
                i_argmax = np.argmax(scores)    # state index

                # Create prediction
                p_type = e_type     # p_type: element type of prediction, differs for SUBs
                reg_data = reg
                dev_type = None
                if i_argmax == 0:
                    dev_type = DeviationTypes.VERIFICATION.value
                elif i_argmax == 1:
                    dev_type = DeviationTypes.INSERTION.value
                elif i_argmax == 2:
                    dev_type = DeviationTypes.SUBSTITUTION.value
                    reg_data = reg_sub
                    p_type = 'signs' if e_type == 'lights' else 'lights'

                norm_data[p_type]['anchor_x'] = anc[ib][iz][ix][iy][0]
                norm_data[p_type]['anchor_y'] = anc[ib][iz][ix][iy][1]
                norm_data[p_type]['anchor_z'] = anc[ib][iz][ix][iy][2]
                pred = {
                    'id': elem_id,
                    'type': group_to_type_lut[p_type],
                    'score': s_max,
                    'class': None,
                    'dev_type': dev_type,
                }
                pred = extract_regression_data_from_grid(pred, p_type, iz, ix, iy, reg_data, norm_data)
                score_dict[dev_type].append(s_max)
                preds_dict[dev_type].append(pred)

                voxel_pred = {
                    'id': elem_id,
                    'type': group_to_type_lut[p_type],
                    'voxel': voxel,
                    'scores': scores
                }
                voxel_pred = extract_regression_data_from_grid(voxel_pred, p_type, iz, ix, iy, reg_data, norm_data)
                predictions_voxels.append(voxel_pred)

            # Create final prediction considering the voxel predictions for this element
            states = ['VER', 'INS', 'SUB']
            votes = []
            max_scores = []
            for s in states:
                votes.append(len(score_dict[s]))
                if score_dict[s]:
                    max_scores.append(np.max(score_dict[s]))
                else:
                    max_scores.append(0)

            # i_argmax = np.argmax(votes)       # majority vote
            i_argmax = np.argmax(max_scores)    # maximum score

            # Select evaluation state based on highest score of the set of predicting voxels
            dev_type = None
            if i_argmax == 0:
                dev_type = DeviationTypes.VERIFICATION.value
            elif i_argmax == 1:
                dev_type = DeviationTypes.INSERTION.value
            elif i_argmax == 2:
                dev_type = DeviationTypes.SUBSTITUTION.value

            # Compute average of all voxel predictions to refine the element's bounding shape
            attributes = ['x_vrf', 'y_vrf', 'z_vrf', 'diameter', 'height', 'width', 'score', 'yaw_vrf']
            estimations = {}
            for a in attributes:
                estimations[a] = []

            for p in preds_dict[dev_type]:
                for a in attributes:
                    if a in p.keys():
                        estimations[a].append(p[a])

            # Compute standard average. For yaw: calc via real and imaginary components (e.g., 1° vs. 359°)
            for a in attributes:
                if len(estimations[a]) > 0:
                    if a != 'yaw_vrf':
                        estimations[a] = np.average(estimations[a])  # average
                    else:
                        real = np.average(np.cos(np.radians(estimations[a])))
                        imag = np.average(np.sin(np.radians(estimations[a])))
                        estimations[a] = np.degrees(np.arctan2(imag, real))

            pred = preds_dict[dev_type][0]
            for a in attributes:
                pred[a] = estimations[a]

            predictions_list_lut.append(pred)

        # Second: get deletions (predictions for elements missing in the map)
        # Create array of scores and cell indices for sorting, filtering, and iterating
        task_max = np.max(task, axis=4)
        task_argmax = np.argmax(task, axis=4)
        indices = np.indices(task_max.shape)  # -> [index, batch, L, X, Y]
        deletions_max = (task_argmax == i_del).flatten()  # check for which voxels deletion score is highest

        grid_indices = np.column_stack((task_max.flatten()[deletions_max],    # score
                                        indices[0].flatten()[deletions_max],        # batch
                                        indices[1].flatten()[deletions_max],        # Z
                                        indices[2].flatten()[deletions_max],        # X
                                        indices[3].flatten()[deletions_max]))       # Y

        # Sort by first column (score) -> get deletions with the highest score
        # [N, 5(score, b, z, x, y)]
        grid_indices = grid_indices[grid_indices[:, 0].argsort()[::-1]]  # [::-1] for reverse (descending)

        # Consider only first 5000 entries for speed
        grid_indices = grid_indices[0:5000, :]
        for row in range(0, grid_indices.shape[0]):
            score = grid_indices[row, 0]
            if score < .05:
                break

            ib = int(grid_indices[row, 1])
            iz = int(grid_indices[row, 2])
            ix = int(grid_indices[row, 3])
            iy = int(grid_indices[row, 4])

            pred = {
                'type': group_to_type_lut[e_type],
                'score': score,
                'class': None,
                'dev_type': DeviationTypes.DELETION.value,
            }

            norm_data[e_type]['anchor_x'] = anc[ib][iz][ix][iy][0]
            norm_data[e_type]['anchor_y'] = anc[ib][iz][ix][iy][1]
            norm_data[e_type]['anchor_z'] = anc[ib][iz][ix][iy][2]
            pred = extract_regression_data_from_grid(pred, e_type, iz, ix, iy, reg, norm_data)
            predictions_lists_deletions[e_type].append(pred)

    # Apply NMS filtering to deletions
    predictions_lists_deletions = nms(predictions_lists_deletions, configs)

    # Resort LUT predictions based on current element type (LUT is sorted by prior type) for NMS
    for e_type in element_types:
        predictions_lists_lut[e_type] = [d for d in predictions_list_lut if
                                            type_to_group_lut[d['type']] == e_type]

    # Third: combine the list for VER, INS, SUB with the list of DEL
    predictions_lists_nms = predictions_lists_lut
    for e_type in element_types:
        predictions_lists_nms[e_type].extend(predictions_lists_deletions[e_type])

    return predictions_lists_nms


def extract_regression_data_from_grid(element_data, element_type, iz, ix, iy, reg, norm_data):
    """ Unnormalize object data retrieved from the network's regression output.

    Args:
        element_data (dict): data of the predicted element
        element_type (str): type of the element
        iz (int): z-coordinate of voxel
        ix (int): x-coordinate
        iy (int): y-coodrinate
        reg (tensor): predicted regression output of network [batch, Z, X, Y, channel]
        norm_data(dict:dict): contains data for unnormalization per element type

    Returns:
        element_data (dict): extended and unnormalized element data
    """
    ib = 0
    element_data.update({
        'x_vrf': reg[ib][iz][ix][iy][0],
        'y_vrf': reg[ib][iz][ix][iy][1],
        'z_vrf': reg[ib][iz][ix][iy][2]
    })

    if element_type == 'poles':
        element_data.update({
            'diameter': reg[ib][iz][ix][iy][3]
        })
    elif element_type == 'signs':
        element_data.update({
            'width': reg[ib][iz][ix][iy][3],
            'height': reg[ib][iz][ix][iy][4],
            'yaw_sin': reg[ib][iz][ix][iy][5],
            'yaw_cos': reg[ib][iz][ix][iy][6],
        })
    elif element_type == 'lights':
        element_data.update({
            'width': reg[ib][iz][ix][iy][3],
            'height': reg[ib][iz][ix][iy][4],
            'yaw_sin': reg[ib][iz][ix][iy][5],
            'yaw_cos': reg[ib][iz][ix][iy][6],
        })

    # Unnormalize
    element_data = target_generator.unnormalize_element_data(element_type, element_data, norm_data[element_type])

    return element_data


def create_deviation_predictions_lut(predictions_lists: dict, gt_deviations: dict, configs: dict):
    """ Converts MDD-M (stage 3) predictions into deviation objects.

    Deviation objects comprise a prior (examined) element (input to the net) and the actual
    element found in the sensor data.

    Args:
        predictions_lists (dict:list): list of predicted elements with evaluation state per element type
        gt_deviations (dict:list): list of ground truth (GT) deviations per element type
        configs:

    Returns:

    """
    element_types = configs['model']['configured_element_types']
    # group_to_type_lut, _ = lookups.get_element_type_naming_lut()
    deviations_lists = init_nested_dict([element_types], [])

    # GT map deviations providing a prior and current element
    gt_deviations_all = []
    for elem_type in element_types:
        gt_deviations_all.extend(gt_deviations[elem_type])
    elements_prior = [d.get_prior() for d in gt_deviations_all if d.get_prior() is not None]

    deviation_predictions = []
    for e_type, predictions in predictions_lists.items():
        for pred in predictions:
            # DELETIONS: create deviation right away, no map-prior needed
            if pred['dev_type'] == DeviationTypes.DELETION.value:
                dev_pred = MapDeviation(DeviationTypes.DELETION,
                                        element_prior=None,
                                        element_current=pred,
                                        is_prediction=True)

            # VERIFICATIONS, INSERTIONS, SUBSTITUTIONS
            else:
                # Find respective prior element
                e = [e for e in elements_prior if e['id'] == pred['id']]
                if not e:
                    raise KeyError("[Error] post processing: element not found in list of prior elements.")
                e = e[0]  # only one element with this id can be found

                # INS: no current element
                # VER: current element is prediction
                # SUB: current element is prediction
                if pred['dev_type'] == DeviationTypes.INSERTION.value:
                    dev_pred = MapDeviation(DeviationTypes.INSERTION, element_prior=e, element_current=None,
                                            is_prediction=True)
                elif pred['dev_type'] == DeviationTypes.VERIFICATION.value:
                    dev_pred = MapDeviation(DeviationTypes.VERIFICATION, element_prior=e, element_current=pred,
                                            is_prediction=True)
                elif pred['dev_type'] == DeviationTypes.SUBSTITUTION.value:
                    dev_pred = MapDeviation(DeviationTypes.SUBSTITUTION, element_prior=e, element_current=pred,
                                            is_prediction=True)
                else:
                    raise ValueError(f"Unknown deviation type {pred['dev_type']}")

            deviation_predictions.append(dev_pred)

    # Sort predictions based on "current" type (required for evaluation, e.g., of substitutions)
    for e_type in element_types:
        deviations_lists[e_type] = [d for d in deviation_predictions if d.type_current == e_type]

    return deviations_lists
