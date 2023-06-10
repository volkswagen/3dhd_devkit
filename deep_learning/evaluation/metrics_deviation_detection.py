""" Compute evaluation metrics for the deviation detection task.
"""

import numpy as np

from dataset.map_deviation import DeviationTypes, get_deviation_classes
from deep_learning.postprocessing.comparison import check_for_match
from deep_learning.util import lookups
from deep_learning.evaluation.metrics_shared import calculate_metrics_from_statistics
from deep_learning.util.lookups import init_nested_dict


def compute_metrics_id_based(predictions_lists: dict, gt_deviations: dict, configs: dict):
    """ Computes metrics (f1, recall precision) based on element ids if applicable.

    This function compares obtained predictions with the corresponding GT to identify true positives (TPs),
    false positives (FPs), and false negatives (FPn) for each evaluation state (VER, DEL, INS, SUB).
    The argument gt_elements comprise lists of deviations. Each deviation contains a prior (examined)
    and a current element (GT reflecting the sensor data).

    Structure of analysis output:
        analysis[element_type]['attributional_errors'][classes]
        analysis[element_type]['detection_statistics'][classes]
        analysis[element_type]['metrics'][classes]
        analysis[element_type]['gt_objects']
        analysis[element_type]['predictions']
        analysis[element_type]['confusion_matrix']
    with 'classes' corresponding to evaluation states (VER, INS, DEL, SUB, Deviating, With-Prior).
    'Deviating' ('DEV') summarizes INS, DEL, SUB.
    'With-Prior' summarizes INS, SUB, VER.
    'All' summarizes VER, INS, DEL, and SUB.

    Args:
        predictions_lists (dict:list): maps element type to list of predictions
        gt_deviations: (dict:list): list of GT deviations per element type
        configs (dict:dict): contains model, train, and system configs

    Returns:
        analysis (dict:dict): contains analyzed data, e.g., metrics, errors, and evaluated predictions
    """
    # Initialize
    model_config, train_config = configs['model'], configs['train']
    configured_element_types = predictions_lists.keys()

    metrics_lookup = lookups.get_metrics_definition(classification_active=False)
    statistics_lookup = lookups.get_statistics_definition(classification_active=False)
    errors_lookup = lookups.get_error_definition_lookup()

    deviation_classes = [d for d in DeviationTypes]
    deviation_classes_with_agg = get_deviation_classes()

    # error accumulator and counters for associated predictions (per elem_type and class)
    error_statistics = init_nested_dict([configured_element_types, deviation_classes_with_agg], {})
    # contains averaged errors for one sample (per elem_type and class)
    attributional_errors = init_nested_dict([configured_element_types, deviation_classes_with_agg], {})
    # counters for TP, FP, FN (per elem_type and class)
    detection_statistics = init_nested_dict(
        [configured_element_types, deviation_classes_with_agg, statistics_lookup], 0)
    # computed metrics for one sample (per elem_type and class)
    metrics = init_nested_dict([configured_element_types, deviation_classes_with_agg, metrics_lookup], 0.0)

    # Initialize final layer of error dicts and confusion matrix (cannot be handled by init function)
    for element_type in configured_element_types:
        # Error dicts
        for cl in deviation_classes_with_agg:
            for error in errors_lookup[element_type]:
                error_statistics[element_type][cl][error] = []
                attributional_errors[element_type][cl][error] = []

    # Get all GT elements (get_most_recent() provides either the prior or current element for each deviation)
    gt_elements_all = []
    for e_type in configured_element_types:
        gt_elements_all += [e.get_most_recent() for e in gt_deviations[e_type]]

    # Iterate over all element types and predictions and compare with GT
    for e_type, predictions_type in predictions_lists.items():
        # GT elements and predictions are sorted by type of the current element
        gt_deviations_type = gt_deviations[e_type]
        errors = errors_lookup[e_type]

        for deviation_cls in deviation_classes:
            predictions_dev = [p for p in predictions_type if p.deviation_type == deviation_cls]
            predictions_cur = [p.get_most_recent() for p in predictions_dev]    # currently predicted elements

            gt_deviations_cls = [d for d in gt_deviations_type if d.deviation_type == deviation_cls]
            gt_elements_cur = [d.get_most_recent() for d in gt_deviations_cls]  # current elements (GT in sensor data)

            dev_classes = ['All', deviation_cls.value]
            dev_classes += ['Deviating'] if deviation_cls != DeviationTypes.VERIFICATION else []
            dev_classes += ['With-Prior'] if deviation_cls != DeviationTypes.DELETION else []

            relevant_dims = ['x_vrf', 'y_vrf', 'z_vrf']
            gt_candidates = lookups.dict_list_to_np_array(gt_elements_cur, relevant_dims, include_idx=True)

            # Check if GT elements are available
            gt_available = True if len(gt_elements_cur) > 0 else False

            # Iterate over all predictions
            for prediction in predictions_cur:
                # Sanity check: find prediction (with prior) in GT elements
                if not deviation_cls == DeviationTypes.DELETION:
                    match = [e for e in gt_elements_all if e['id'] == prediction['id']]
                    if not match:
                        raise KeyError("[Error] evaluation: prediction ID not found!")

                # No GT available -> set all predictions to FP
                if not gt_available:
                    prediction['eval_class'] = 'FP'
                    for dev_cls in dev_classes:
                        detection_statistics[e_type][dev_cls]['FP'] += 1
                    continue

                # Associate with GT elements and check if GT matches
                gt_match = False
                gt_element = {}
                d_gt = None

                # DELETIONS: apply association metrics
                if deviation_cls == DeviationTypes.DELETION:
                    d_vec = np.sqrt(
                        sum([(gt_candidates[:, i] - prediction[dim]) ** 2 for i, dim in enumerate(relevant_dims)]))
                    i_min = np.argmin(d_vec)
                    d_gt = d_vec[i_min]
                    gt_element: dict = gt_elements_cur[i_min]
                    is_match = check_for_match(e_type, gt_element, prediction, model_config, train_config)
                    already_mapped = 'mapped_flag' in gt_element.keys() and gt_element['mapped_flag']
                    if is_match and not already_mapped:
                        gt_match = True

                # INSERTIONS, SUBSTITUTIONS, VERIFICATIONS: use id
                else:
                    # Try to find respective ID in current GT elements (most recent)
                    match = [e for e in gt_elements_cur if e['id'] == prediction['id']]
                    if match:
                        gt_match = True
                        gt_element = match[0]
                        d_gt = np.sqrt(
                            sum([(gt_element[dim] - prediction[dim]) ** 2 for i, dim in enumerate(relevant_dims)]))

                # Association found -> TP
                if gt_match:
                    # Set TP counter
                    for dev_cls in dev_classes:
                        detection_statistics[e_type][dev_cls]['TP'] += 1

                    # Set errors and statistics (if classification configured: for 'All' and gt class
                    # Ignore insertions for errors and accuracy since predictions are gt_elements
                    if deviation_cls != DeviationTypes.INSERTION:
                        for dev_cls in dev_classes:
                            for err in errors:
                                if err == 'distance':
                                    val = d_gt
                                elif err == 'yaw_vrf':
                                    val = 180 - abs(abs(prediction[err] - gt_element[err]) - 180)
                                else:
                                    val = abs(prediction[err] - gt_element[err])
                                attributional_errors[e_type][dev_cls][err].append(val)
                                error_statistics[e_type][dev_cls][err].append(val)

                    prediction['eval_class'] = 'TP'
                    gt_element['mapped_flag'] = True

                # No association found -> FP
                else:
                    prediction['eval_class'] = 'FP'
                    # Sort FP into corresponding state
                    for dev_cls in dev_classes:
                        detection_statistics[e_type][dev_cls]['FP'] += 1

            # Set FN for unmapped GT elements (or those lost during lut creation)
            for gt_element in gt_elements_cur:
                if not 'mapped_flag' in gt_element.keys():
                    gt_element['eval_class'] = 'FN'
                if 'excluded_from_lut' in gt_element.keys():
                    gt_element['eval_class'] = 'FN'

    # Compute average for errors
    for e_type in attributional_errors.keys():
        for dev_cls in deviation_classes_with_agg:
            # Get errors for element type and class
            errors_dict = attributional_errors[e_type][dev_cls]

            # Average errors
            for error_type, deltas in errors_dict.items():
                if not deltas:
                    error = None
                else:
                    error = np.mean(deltas)
                attributional_errors[e_type][dev_cls][error_type] = error  # substitute list by mean

    # Count FN

    for e_type, deviations in gt_deviations.items():
        for d in deviations:
            gt_type = d.type_current
            element = d.get_most_recent()

            dev_classes = ['All', d.deviation_type.value]
            dev_classes += ['Deviating'] if d.deviation_type != DeviationTypes.VERIFICATION else []
            dev_classes += ['With-Prior'] if d.deviation_type != DeviationTypes.DELETION else []

            # if 'mapped_flag' not in element.keys() and elem_type in detection_statistics:
            if 'eval_class' in element.keys() and gt_type in detection_statistics:
                if element['eval_class'] == 'FN':
                    for dev_cls in dev_classes:
                        detection_statistics[e_type][dev_cls]['FN'] += 1

    # Compute metrics
    for e_type in configured_element_types:
        for dev_cls in detection_statistics[e_type].keys():
            metrics[e_type][dev_cls] = calculate_metrics_from_statistics(
                detection_statistics[e_type][dev_cls], classification_active=False)

    # Resort results to final analysis dict
    analysis = init_nested_dict([configured_element_types], {})

    for e_type in configured_element_types:
        analysis[e_type]['attributional_errors'] = attributional_errors[e_type]
        analysis[e_type]['error_statistics'] = error_statistics[e_type]
        analysis[e_type]['metrics'] = metrics[e_type]
        analysis[e_type]['detection_statistics'] = detection_statistics[e_type]
        analysis[e_type]['gt_objects'] = gt_deviations[e_type]
        analysis[e_type]['predictions'] = predictions_lists[e_type]
        analysis[e_type]['confusion_matrix'] = None

    return analysis


def compute_metrics_from_accumulation_deviation_detection(model_config, stats_acc, errors_acc):
    """ Computes detection and error metrics based on accumulated detection statistics (TP etc. in stats_acc).

    Args:
        model_config (dict): contains model configuration
        stats_acc (dict:dict): contains accumulated TP, FP, FN statistics for metrics computation
        errors_acc: (dict:dict) contains accumulated errors (for elements and samples) for averaging

    Returns:
        metrics (dict:dict): contains metrics for each element type
        errors (dict:dict): contains errors for each element type
    """

    element_types = model_config['configured_element_types']
    errors_def = lookups.get_error_definition_lookup()
    metrics_def = lookups.get_metrics_definition(classification_active=False)
    deviation_classes = get_deviation_classes()

    # Container for final recall / precision / f1 values (class specific)
    metrics = init_nested_dict([element_types, deviation_classes, metrics_def], 0)
    errors = init_nested_dict([element_types, deviation_classes], {})

    # Compute overall recall, precision, f1
    for element_type in element_types:
        # Init final layer of errors (cannot be handled by init function)
        for dev_cls in deviation_classes:
            for error in errors_def[element_type]:
                errors[element_type][dev_cls][error] = 0

        for dev_cls in deviation_classes:
            # Compute metrics and errors class wise
            metrics[element_type][dev_cls] = calculate_metrics_from_statistics(
                stats_acc[element_type][dev_cls], classification_active=False)

            for error in errors_def[element_type]:
                values = errors_acc[element_type][dev_cls][error]
                if values:
                    errors[element_type][dev_cls][error] = np.mean(values)
                else:
                    errors[element_type][dev_cls][error] = None

    return metrics, errors
