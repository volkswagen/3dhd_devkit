""" Compute evaluation metrics for the object detection task.
"""

import numpy as np

from deep_learning.postprocessing.comparison import check_for_match
from deep_learning.util import lookups
from deep_learning.evaluation.metrics_shared import calculate_metrics_from_statistics
from deep_learning.util.lookups import init_nested_dict


def compute_metrics_element_detection(predictions_lists, gt_elements, model_config, train_config):
    """ Computes metrics (f1, recall precision) for the object detection task.

    This function compares obtained predictions with the corresponding GT elements to identify true positives (TPs),
    false positives (FPs), and false negatives (FPs). The argument gt_elements comprise lists of GT elements
    present in the sensor data.

    Structure of analysis output:
        analysis[element_type]['attributional_errors'][classes]
        analysis[element_type]['detection_statistics'][classes]
        analysis[element_type]['metrics'][classes]
        analysis[element_type]['gt_objects']
        analysis[element_type]['predictions']
    with 'classes' corresponding to subclasses of objects, e.g., 'tree', 'lamppost', etc. for poles.
    'All' summarizes all elements of a type (e.g., poles) disregarding their subclass.
    'Mean' computes the average over all classes.

    Args:
        predictions_lists (dict:list): maps element type to list of predictions
        gt_elements: (dict:list): maps element type to lists of elements (deviations)
        model_config (dict): contains the model configuration
        train_config (dict): contains the training configuration

    Returns:
        analysis (dict:dict): contains analyzed data, e.g., metrics, errors, and evaluated predictions
    """

    # Init analysis dicts
    configured_element_types = predictions_lists.keys()
    classes_dict = lookups.get_class_definition_lookup(model_config, include_mean=True)
    classes_dict_conf = lookups.get_class_definition_lookup(model_config, include_all=False, include_mean=True)
    classification_active = model_config['classification_active']

    metrics_lookup = lookups.get_metrics_definition(classification_active)
    statistics_lookup = lookups.get_statistics_definition(classification_active)
    errors_lookup = lookups.get_error_definition_lookup()

    # Error accumulator and counters for associated predictions (per elem_type and class)
    error_statistics = init_nested_dict([configured_element_types, classes_dict], {})
    # Contains averaged errors for one sample (per elem_type and class)
    attributional_errors = init_nested_dict([configured_element_types, classes_dict], {})
    # Counters for TP, FP, FN (per elem_type and class)
    detection_statistics = init_nested_dict([configured_element_types, classes_dict, statistics_lookup], 0)
    # Computed metrics for one sample (per elem_type and class)
    metrics = init_nested_dict([configured_element_types, classes_dict, metrics_lookup], 0.0)
    # elem_type -> GT class -> predicted class
    confusion_matrices = init_nested_dict([configured_element_types, classes_dict_conf], {})

    # Initialize final layer of error dicts (cannot be handled by init function)
    for element_type in configured_element_types:
        for cl in classes_dict[element_type]:
            for error in errors_lookup[element_type]:
                error_statistics[element_type][cl][error] = []
                attributional_errors[element_type][cl][error] = []
        for cl_gt in classes_dict_conf[element_type]:
            for cl_pred in classes_dict_conf[element_type]:
                confusion_matrices[element_type][cl_gt][cl_pred] = 0

    # Iterate over all element types and predictions and compare with GT
    for element_type, predictions in predictions_lists.items():
        gt_elements_type = gt_elements[element_type]
        errors = errors_lookup[element_type]
        relevant_dims = ['x_vrf', 'y_vrf', 'z_vrf']
        gt_candidates = lookups.dict_list_to_np_array(gt_elements_type, relevant_dims, include_idx=True)

        # Check if GT elements are available
        gt_available = True
        if gt_candidates.size == 0:
            # print("[Warning] evaluation: no GT objects for {}, size gt: {}.".format(element_type, num_elements))
            gt_available = False

        # Iterate over all predictions
        for prediction in predictions:
            # No GT available -> set all predictions to FP
            if not gt_available:
                prediction['eval_class'] = 'FP'
                classes = list({'All', prediction['class']}) if classification_active else classes_dict[element_type]
                for cl in classes:
                    detection_statistics[element_type][cl]['FP'] += 1
                continue

            # Find closest GT element
            d_vec = np.sqrt(sum([(gt_candidates[:, i] - prediction[dim]) ** 2 for i, dim in enumerate(relevant_dims)]))
            i_min = np.argmin(d_vec)
            d_min = d_vec[i_min]
            gt_element: dict = gt_elements_type[i_min]  # specify as dict to stop pycharm from complaining

            # Check for match
            is_match = check_for_match(element_type, gt_element, prediction, model_config, train_config)
            already_mapped = 'mapped_flag' in gt_element.keys() and gt_element['mapped_flag']

            # Association found -> TP
            if is_match and not already_mapped:
                classes = ['All']
                classes_match = True

                if classification_active:
                    if element_type == 'poles':
                        cl_gt = gt_element['cls']  # ground truth class
                    elif element_type == 'signs':
                        cl_gt = gt_element['shape']
                    elif element_type == 'lights':
                        cl_gt = gt_element['cls']
                    else:
                        raise ValueError(f"Not defined for element_type '{element_type}'")

                    cl_pred = prediction['class']
                    classes_match = (cl_gt == cl_pred)

                    # Set confusion matrix
                    confusion_matrices[element_type][cl_gt][cl_pred] += 1
                    classes = list({'All', cl_gt})

                # Set errors and statistics (if classification configured: for 'All' and gt class
                for cl in classes:
                    for err in errors:
                        if err == 'distance':
                            val = d_min
                        elif err == 'yaw_vrf':
                            val = 180 - abs(abs(prediction[err] - gt_element[err]) - 180)
                        else:
                            val = abs(prediction[err] - gt_element[err])
                        attributional_errors[element_type][cl][err].append(val)
                        error_statistics[element_type][cl][err].append(val)

                    # Set TP counter
                    detection_statistics[element_type][cl]['TP'] += 1

                    # Set ACC counter N
                    if classes_match and classification_active:
                        detection_statistics[element_type][cl]['N'] += 1

                prediction['eval_class'] = 'TP'
                gt_element['mapped_flag'] = True

            # No association found -> FP
            else:
                prediction['eval_class'] = 'FP'
                # Sort FP into corresponding class if class is predicted.
                if classification_active:
                    classes = list({'All', prediction['class']})

                # Else: Count one FP for all classes, will downgrade the precision
                else:
                    classes = classes_dict[element_type]

                # Increase FP counter
                for cl in classes:
                    detection_statistics[element_type][cl]['FP'] += 1

    # Compute average for errors
    for element_type in attributional_errors.keys():
        for cl in attributional_errors[element_type].keys():
            # Get errors for element type and class
            errors_dict = attributional_errors[element_type][cl]
            # Average errors
            for error_type, deltas in errors_dict.items():
                if not deltas:
                    error = None
                else:
                    error = np.mean(deltas)
                attributional_errors[element_type][cl][error_type] = error  # substitute list by mean

    # Count FN
    _, type_to_group_lut = lookups.get_element_type_naming_lut()
    for element_type, elements in gt_elements.items():
        for element in elements:
            # Map naming convention 'TrafficSign' -> 'signs'
            element_type = type_to_group_lut[element['type']]

            classes = ['All']
            if classification_active:
                if element_type == 'poles':
                    cl_gt = element['cls']
                elif element_type == 'lights':
                    cl_gt = element['cls']
                elif element_type == 'signs':
                    cl_gt = element['shape']
                else:
                    cl_gt = 'All'
                classes = list({'All', cl_gt})

            for cl in classes:
                if 'mapped_flag' not in element.keys() and element_type in detection_statistics:
                    detection_statistics[element_type][cl]['FN'] += 1

    # Compute metrics
    for element_type in configured_element_types:
        for cl in detection_statistics[element_type].keys():
            metrics[element_type][cl] = calculate_metrics_from_statistics(detection_statistics[element_type][cl],
                                                                          classification_active)

    # Resort results to final analysis dict
    analysis = init_nested_dict([configured_element_types], {})
    for element_type in configured_element_types:
        analysis[element_type]['attributional_errors'] = attributional_errors[element_type]
        analysis[element_type]['error_statistics'] = error_statistics[element_type]

        analysis[element_type]['metrics'] = metrics[element_type]
        analysis[element_type]['detection_statistics'] = detection_statistics[element_type]

        analysis[element_type]['gt_objects'] = gt_elements[element_type]
        analysis[element_type]['predictions'] = predictions_lists[element_type]
        analysis[element_type]['confusion_matrix'] = confusion_matrices[element_type]

    return analysis


def compute_metrics_from_accumulation_element_detection(model_config, stats_acc, errors_acc):
    """ Computes detection and error metrics based on accumulated detection statistics (TP etc. in stats_acc).

    Args:
        model_config (dict): contains model configuration
        stats_acc (dict:dict): contains accumulated TP, FP, FN statistics for metrics computation
        errors_acc: (dict:dict) contains accumulated errors (for elements and samples) for averaging

    Returns:
        metrics (dict:dict): nested dict containing metrics (f1, recall, precision) for each element type and class
        errors (dict:dict): nested dict containing type-specific regression errors for each element type and class
    """

    element_types = model_config['configured_element_types']
    classification_active = model_config['classification_active']
    classes_lookup = lookups.get_class_definition_lookup(model_config, include_mean=classification_active)
    errors_lookup = lookups.get_error_definition_lookup()
    metrics_def = lookups.get_metrics_definition(classification_active)

    # Container for final recall / precision / f1 values (class specific)
    metrics = init_nested_dict([element_types, classes_lookup, metrics_def], 0)
    errors = init_nested_dict([element_types, classes_lookup], {})

    # Compute overall recall, precision, f1
    for element_type in element_types:
        # Get configured classes
        configured_classes = classes_lookup[element_type]

        # Init final layer of errors (cannot be handled by init function)
        for cl in configured_classes:
            for error in errors_lookup[element_type]:
                errors[element_type][cl][error] = 0

        # Setup valid counters for counting entries that are not none (used for mean computation)
        valid_entry_ctrs = init_nested_dict([metrics_def + errors_lookup[element_type]], 0)

        # Compute metrics and errors class wise
        for cl in configured_classes:
            if cl == 'Mean':  # skip summary classes
                continue

            metrics[element_type][cl] = calculate_metrics_from_statistics(stats_acc[element_type][cl],
                                                                          classification_active)

            for error in errors_lookup[element_type]:
                values = errors_acc[element_type][cl][error]
                if values:
                    errors[element_type][cl][error] = np.mean(values)
                else:
                    errors[element_type][cl][error] = None

            # Compute mean by summing all classes into "mean" class
            if cl == 'All':  # skip summary classes
                continue

            if classification_active:
                # Metrics
                for metric in metrics_def:
                    value = metrics[element_type][cl][metric]
                    if value is not None:
                        metrics[element_type]['Mean'][metric] += value
                        valid_entry_ctrs[metric] += 1

                # Errors
                for error in errors_lookup[element_type]:
                    value = errors[element_type][cl][error]
                    if value is not None:
                        errors[element_type]['Mean'][error] += value
                        valid_entry_ctrs[error] += 1
        # Class loop end

        if classification_active:
            # Final mean computation
            for metric in metrics_def:
                if valid_entry_ctrs[metric] > 0:
                    metrics[element_type]['Mean'][metric] /= valid_entry_ctrs[metric]
                else:
                    metrics[element_type]['Mean'][metric] = None

            for error in errors_lookup[element_type]:
                if valid_entry_ctrs[error] > 0:
                    errors[element_type]['Mean'][error] /= valid_entry_ctrs[error]
                else:
                    errors[element_type]['Mean'][error] = None
        # Element type loop end

    return metrics, errors
