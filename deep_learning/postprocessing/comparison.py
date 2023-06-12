""" Post-processing for MDD-SC (stage 1) and MDD-MC (stage 2).
"""

import math
import numpy as np

from dataset.map_deviation import MapDeviation, DeviationTypes
from deep_learning.losses import target_generator
from deep_learning.util import lookups
from deep_learning.util.lookups import init_nested_dict


def compare_predictions_with_map_prior(predictions_lists: dict, gt_deviations: dict, configs: dict):
    """ Compare predicted and examined map elements to identify deviations (object detection DNN).

    Args:
        predictions_lists (dict:list): list of predicted elements per element type
        gt_deviations (dict:list): list of GT deviations comprising prior (examined) and current element
        configs (dict:dict): contains model, tran, and system configurations

    Returns:
        deviations_lists (dict:list) list of predicted deviations per element type
    """
    # Initialize
    element_types = predictions_lists.keys()
    deviations_lists = init_nested_dict([element_types], [])
    deviations = []
    _, type_lut = lookups.get_element_type_naming_lut()
    relevant_dims = ['x_vrf', 'y_vrf', 'z_vrf']  # for distance map calculations
    # Get all deviations
    gt_deviations_all = []
    for e_type in element_types:
        gt_deviations_all += gt_deviations[e_type]

    for e_type, predictions in predictions_lists.items():
        # Get all prior map elements per type
        elements_prior_type = [d.get_prior() for d in gt_deviations_all if d.get_prior() is not None
                               and d.type_prior == e_type]

        # Convert examined elements to np array for distance map calculation
        prior_candidates = lookups.dict_list_to_np_array(elements_prior_type, relevant_dims, include_idx=True)

        for pred in predictions:
            if len(elements_prior_type) == 0:
                dev_pred = MapDeviation(DeviationTypes.DELETION, element_prior=None, element_current=pred,
                                        is_prediction=True)
            else:
                # Find closest gt_element and check association criteria
                d_vec = np.sqrt(sum([(prior_candidates[:, i] - pred[dim]) ** 2 for i, dim in enumerate(relevant_dims)]))
                i_min = np.argmin(d_vec)
                element_prior: dict = elements_prior_type[i_min]

                is_match = check_for_match(e_type, element_prior, pred, configs['model'], configs['train'])
                already_confirmed = 'confirmed' in element_prior.keys() and element_prior['confirmed']

                if is_match and not already_confirmed:
                    pred['id'] = element_prior['id']
                    dev_pred = MapDeviation(DeviationTypes.VERIFICATION, element_prior=element_prior, element_current=pred,
                                            is_prediction=True)
                    element_prior['confirmed'] = True
                else:
                    dev_pred = MapDeviation(DeviationTypes.DELETION, element_prior=None, element_current=pred,
                                            is_prediction=True)

            deviations.append(dev_pred)

        # For all gt_elements without 'confirmed' flag -> INSERTION
        unconfirmed_elements = [e for e in elements_prior_type if 'confirmed' not in e.keys() or not e['confirmed']]
        for elem in unconfirmed_elements:
            e_type = type_lut[elem['type']]
            elem['class'] = elem['cls'] if e_type == 'poles' else 'All'
            dev_pred = MapDeviation(DeviationTypes.INSERTION, element_prior=elem, element_current=None,
                                    is_prediction=True)
            deviations.append(dev_pred)

    # SUBSTITUTION: cross-check (different elem_types) all deletions and insertions based on distance
    deletions = [d for d in deviations if d.deviation_type == DeviationTypes.DELETION]
    insertions = [d for d in deviations if d.deviation_type == DeviationTypes.INSERTION]

    for deletion in deletions:
        del_elem = deletion.get_current()
        insertions_sub = [i for i in insertions if i.get_prior()['type'] != deletion.get_current()['type']]
        if len(insertions_sub) == 0:
            continue

        insertions_sub_elems = [i.get_prior() for i in insertions_sub]
        subst_candidates = lookups.dict_list_to_np_array(insertions_sub_elems, relevant_dims, include_idx=True)

        # Find the closest insertion and check for association distance
        d_vec = np.sqrt(sum([(subst_candidates[:, i] - del_elem[dim]) ** 2 for i, dim in enumerate(relevant_dims)]))
        if np.min(d_vec) < configs['train']['threshold_association_distance']:
            # Matched insertion and deletion -> create new substitution and delete old deviations
            i_min = np.argmin(d_vec)
            subst_insertion: MapDeviation = insertions_sub[i_min]
            del_elem['id'] = subst_insertion.get_prior()['id']  # set ID of substituted element
            substitution = MapDeviation(DeviationTypes.SUBSTITUTION, element_prior=subst_insertion.get_prior(),
                                        element_current=del_elem, is_prediction=True)
            deviations.append(substitution)
            deviations.remove(subst_insertion)
            insertions.remove(subst_insertion)
            deviations.remove(deletion)

    for e_type in element_types:
        deviations_lists[e_type] = [d for d in deviations if d.type_current == e_type]

    return deviations_lists


def check_for_match(element_type, gt_element, prediction, model_config, train_config):
    """Checks a given prediction and ground-truth element for element-specific matching criteria.
        Returns True if it matches criteria, else False.
    """
    if element_type == 'lights':  # iosa + z_overlap
        d_xy = math.sqrt((gt_element['x_vrf'] - prediction['x_vrf']) ** 2 +
                         (gt_element['y_vrf'] - prediction['y_vrf']) ** 2)
        iosa = target_generator.iosa_two_circles(d_xy, gt_element['width'], prediction['width'])

        z_overlap = target_generator.line_overlap(p1=gt_element['z_vrf'], len1=gt_element['height'],
                                                  p2=prediction['z_vrf'], len2=prediction['height'])

        is_match = iosa >= model_config['light_iosa_threshold'] and z_overlap >= model_config[
            'light_z_foreground_threshold']

    elif element_type == 'poles':  # xy_distance
        d_xy = math.sqrt((gt_element['x_vrf'] - prediction['x_vrf']) ** 2 +
                         (gt_element['y_vrf'] - prediction['y_vrf']) ** 2)
        # iosa = target_generator.iosa_two_circles(d_xy, gt_element['diameter'], prediction['diameter'])
        # is_match = iosa >= model_config['pole_iosa_threshold']
        is_match = d_xy < train_config['threshold_association_distance']

    elif element_type == 'signs':  # xy_distance to line + z_overlap
        line_seg = target_generator.calculate_line_segment(gt_element['x_vrf'], gt_element['y_vrf'],
                                                           gt_element['width'], gt_element['yaw_vrf'])
        p = np.array([prediction['x_vrf'], prediction['y_vrf']])
        dist = target_generator.point_to_line_dist(p, line_seg)

        z_overlap = target_generator.line_overlap(p1=gt_element['z_vrf'], len1=gt_element['height'],
                                                  p2=prediction['z_vrf'], len2=prediction['height'])

        is_match = dist <= model_config['sign_distance_threshold'] \
                   and z_overlap >= model_config['sign_z_foreground_threshold']

    else:
        raise ValueError(f"Not defined for element_type '{element_type}'")

    return is_match
