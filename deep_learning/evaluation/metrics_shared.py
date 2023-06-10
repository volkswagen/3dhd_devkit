""" Provides shared functions for metrics computation (deviation and object detection tasks)
"""

from dataset.map_deviation import get_deviation_classes


def filter_by_threshold_only(predictions_lists, configs, thresholds: dict = None):
    """ Filters predictions by thresholds.

    For deviation detection (MDD-M (stage 3)), 4 thresholds for each element type are available.
    For object detection (MDD-SC (stage 1), MDD-MC (stage 2)), 1 threshold for each element type is available.

    Args:
        predictions_lists (dict:list): contains predictions (list) for each element type (key)
        configs (dict:dict): contains model, train, and system configurations
        thresholds (dict:dict | dict:float): contains thresholds for each evaluation state or element type, respectively

    Returns:
        filtered_lists (dict:list): contains filtered predictions (list) for each element type (key)
    """
    # Get default threshold from config if none other is provided
    if thresholds is None:
        thresholds = {e_type: configs['train']['threshold_score'] for e_type in predictions_lists.keys()}
    filtered_lists = {}
    deviation_types = get_deviation_classes(False)

    for element_type, predictions in predictions_lists.items():
        # Stage 3: allow separate thresholds for deviation types as well
        if configs['model']['deviation_detection_model']:
            thresholds_type = thresholds[element_type]
            if isinstance(thresholds_type, float):
                thresholds_type = {dev_type: thresholds_type for dev_type in deviation_types}

            # Filter predictions
            filtered_lists[element_type] = []
            for dev_type, threshold in thresholds_type.items():
                filtered_lists[element_type] += [p for p in predictions if
                                                 p['dev_type'] == dev_type and p['score'] >= threshold]

        # Stage 1 and 2:
        else:
            filtered_lists[element_type] = [p for p in predictions if p['score'] >= thresholds[element_type]]

    return filtered_lists


def calculate_metrics_from_statistics(statistics_dict, classification_active):
    """ Calculates precision, recall, f1, and (classification) accuracy from TP, FP, and FN statistics.

    Args:
        statistics_dict (dict:int): contains TP, FP and FN counters
        classification_active (bool): True if classification is active and accuracy has to be computed

    Returns:
        metrics (dict:float): contains precision, recall, f1, and accuracy
    """

    TP = statistics_dict['TP']
    FP = statistics_dict['FP']
    FN = statistics_dict['FN']
    N = statistics_dict['N'] if classification_active else None  # N is total count of objects (subclass specific)

    # Precision can be calculated anyway: no GT objects -> all are considered as FP
    if (TP + FP) == 0:
        pre = None
        # print("[Warning] eval: invalid precision.")
    else:
        pre = TP / (TP + FP)

    # No GT objects present:
    if (TP + FN) == 0:
        rec = None
        # print("[Warning] eval: invalid recall.")
    else:
        rec = TP / (TP + FN)

    if rec is not None and pre is not None:
        f1 = 2 * (pre * rec) / (pre + rec + 10 ** -6)
    else:
        f1 = None

    if TP == 0 or not classification_active:
        acc = None
    else:
        acc = N / TP

    metrics = {
        'precision': pre,
        'recall': rec,
        'f1': f1,
    }
    if classification_active:
        metrics['accuracy'] = acc

    return metrics
