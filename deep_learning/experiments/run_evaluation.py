""" Script to evaluate model predictions with regard to confidence thresholds (PR-curves, optimize thresholds, etc.)"""

import sys
sys.path.append("../..")

import copy
import pickle
import pprint
from multiprocessing import Pool
from time import time

import numpy as np
from tqdm import tqdm
from mlflow.exceptions import MlflowException

from dataset import helper
from dataset.map_deviation import get_deviation_classes
from deep_learning.evaluation import metrics_deviation_detection
from deep_learning.evaluation import metrics_element_detection
from deep_learning.evaluation import metrics_shared
from deep_learning.postprocessing import comparison
from deep_learning.postprocessing import deviation_extraction
from deep_learning.util import config_parser, lookups
from deep_learning.util.tracking import ExperimentTracker
from utility.logger import set_logger
from utility.system_paths import get_system_paths


# Module functions #####################################################################################################
########################################################################################################################

class PrintDict(dict):
    """ Wrapper class for dictionaries for fancy string formatting (recommended if dict contains floats). """
    def __str__(self):
        return str({k: round(v, 2) if isinstance(v, float) else v for k, v in self.items()})


def evaluate_threshold(threshold, configs, predictions, elem_types=None, norm_conf_mat=False, print_results=True):
    """ Evaluates given predictions using a single threshold configuration.

    Args:
        threshold (float | dict:float | dict:dict): confidence threshold, either as single value applied to all element
                                                    types or as a dictionary specifying the threshold for each element
                                                    type separately (MDD-M: additional level for deviation classes)
        configs (dict:dict): configuration dictionary (comprising system, model, and train config)
        predictions (list[dict]): list of predictions, as generated by run_inference script
        elem_types (None | list[str]): list of configured element types (lights, poles, signs);
                                       if None, it is obtained from configs (configured_element_types)
        norm_conf_mat: if true, normalizes confusion matrix, else stores absolute counts
        print_results: if true, show evaluation results on console (and save to log file if a logger is active)

    Returns:
        metrics (dict:dict): nested dict containing metrics (f1, recall, precision) for each element type and class
        errors (dict:dict): nested dict containing type-specific regression errors for each element type and class
        eval_results (list[dict]): list of evaluation results (metrics, errors, etc.) for each sample
        confusion_acc (dict:dict): nested dict containing confusion matrix for each element type

    """

    # Step 1: Setup
    t_start = time()
    model_config, train_config = configs['model'], configs['train']
    deviation_detection_task = train_config['deviation_detection_task']
    deviation_detection_model = model_config['deviation_detection_model']
    preds = predictions  # copy.deepcopy(predictions)

    if elem_types is None:
        elem_types = model_config['configured_element_types']
    if threshold is None:
        threshold = train_config['threshold_score']
    if isinstance(threshold, float):  # Apply given threshold to all element types
        threshold = {elem_type: threshold for elem_type in elem_types}

    # _cm = confusion matrix
    configured_classes = lookups.get_class_definition_lookup(model_config, include_all=True)
    configured_classes_cm = lookups.get_class_definition_lookup(model_config, include_all=False)
    if deviation_detection_task:
        dev_classes = get_deviation_classes(include_aggregate_classes=True)
        dev_classes_cm = get_deviation_classes(include_aggregate_classes=False)
        configured_classes = {e_type: dev_classes for e_type in elem_types}
        configured_classes_cm = {e_type: dev_classes_cm for e_type in elem_types}

    errors_lookup = lookups.get_error_definition_lookup()
    stats_def = lookups.get_statistics_definition(model_config['classification_active'])
    metrics_def = lookups.get_metrics_definition(model_config['classification_active'])

    if deviation_detection_task:
        dev_classes = get_deviation_classes(include_aggregate_classes=True)
        configured_classes = {e_type: dev_classes for e_type in elem_types}
    eval_results = []  # list saving eval results for SampleViewer

    stats_acc = lookups.init_nested_dict([elem_types, configured_classes, stats_def], 0)
    confusion_acc = lookups.init_nested_dict([elem_types, configured_classes_cm], {})
    errors_acc = lookups.init_nested_dict([elem_types, configured_classes], {})
    for elem_type in elem_types:
        for cl in configured_classes[elem_type]:
            for error in errors_lookup[elem_type]:
                errors_acc[elem_type][cl][error] = []
        for cl_cm in configured_classes_cm[elem_type]:
            for cl_pred in configured_classes_cm[elem_type]:
                confusion_acc[elem_type][cl_cm][cl_pred] = 0

    # Step 2: Parse through predictions
    for prediction in preds:  # tqdm(preds):
        gt_elements = copy.deepcopy(prediction['gt_objects'])
        predictions_lists = prediction['predictions_lists']

        predictions_lists_nms = metrics_shared.filter_by_threshold_only(predictions_lists, configs, threshold)
        if deviation_detection_task:
            if deviation_detection_model:  # Stage 3 (MDD-M)
                deviation_lists = deviation_extraction.create_deviation_predictions_lut(
                    predictions_lists_nms, gt_elements, configs)
            else:  # Stage 1 or 2 (MDD-SC, MDD-MC)
                deviation_lists = comparison.compare_predictions_with_map_prior(
                    predictions_lists_nms, gt_elements, configs)
            analysis = metrics_deviation_detection.compute_metrics_id_based(
                deviation_lists, gt_elements, configs)
        else:  # Object detection
            analysis = metrics_element_detection.compute_metrics_element_detection(
                predictions_lists_nms, gt_elements, model_config, train_config)

        eval_result = {
            'sample_id': prediction['sample_id'],
            'run_name': prediction['run_name'],
            'analysis': analysis,  # Contains nms filtered predictions and gt elements
        }
        eval_results.append(eval_result)

        # Append to accumulators
        for elem_type in elem_types:
            for cl in configured_classes[elem_type]:
                for stat in stats_def:
                    stats_acc[elem_type][cl][stat] += analysis[elem_type]['detection_statistics'][cl][stat]

                for error in errors_lookup[elem_type]:
                    errors_acc[elem_type][cl][error] += analysis[elem_type]['error_statistics'][cl][error]

            # Add to confusion matrices: only for object detection
            if not deviation_detection_task:
                for cl in configured_classes_cm[elem_type]:
                    for cl_pred in configured_classes_cm[elem_type]:
                        confusion_acc[elem_type][cl][cl_pred] += analysis[elem_type]['confusion_matrix'][cl][cl_pred]
        # End prediction loop

    # PRC / single score evaluation
    if deviation_detection_task:
        metrics, errors = metrics_deviation_detection.compute_metrics_from_accumulation_deviation_detection(model_config, stats_acc, errors_acc)
    else:
        metrics, errors = metrics_element_detection.compute_metrics_from_accumulation_element_detection(model_config, stats_acc, errors_acc)

    # Use summary class 'All' for computing the prc curve
    # Norm confusion matrix
    if not deviation_detection_task and norm_conf_mat:
        for elem_type in elem_types:
            for cls in configured_classes_cm[elem_type]:
                n_tp = 0
                for cls_pred in configured_classes_cm[elem_type]:
                    n_tp += confusion_acc[elem_type][cls][cls_pred]

                for cls_pred in configured_classes_cm[elem_type]:
                    val = confusion_acc[elem_type][cls][cls_pred]
                    val = val / n_tp if n_tp != 0 else 0
                    val = float(f"{val:.3f}")
                    confusion_acc[elem_type][cls][cls_pred] = val

    if print_results:
        for elem_type in elem_types:
            if deviation_detection_task:
                prc_cls = ['VER', 'Deviating']
            else:
                prc_cls = ['All']
                if model_config['classification_active']:
                    prc_cls.append('Mean')

            for cls in prc_cls:
                print_str = f"{cls}: {elem_type}: "
                for metric in metrics_def:
                    value = metrics[elem_type][cls][metric]
                    value_str = f"{value:04.3f}" if value is not None else "n/a"
                    print_str += f"{metric}: {value_str} | "
                if isinstance(threshold[elem_type], dict):
                    thres = PrintDict(threshold[elem_type])
                else:
                    thres = round(threshold[elem_type], 2)

                print_str += f"threshold: {thres} | time: {time() - t_start:.2f} s"
                print(print_str)

    return metrics, errors, eval_results, confusion_acc


def get_best_setting(prc_data, partition, configs, tracker=None):
    """ Determines the best (valid) threshold configuration (by F1-score) from the given PRC data and logs the PR curves
        to the given tracker, if specified.

    Args:
        prc_data (dict): contains metrics for each tested threshold (as stored in prc.pickle by run_prc_evaluation)
        partition (str): dataset partition (train, val, or test)
        configs (dict:dict): configuration dictionary (comprising system, model, and train config)
        tracker (ExperimentTracker | None): if specified, PR curves will be logged to the tracker

    Returns:
        best_setting (dict:dict): contains the best metrics and corresponding threshold for each element type and class
    """

    # Setup
    deviation_detection_task = configs['train']['deviation_detection_task']
    deviation_detection_model = configs['model']['deviation_detection_model']
    elem_types = [e for e in prc_data if e != 'thresholds']
    best_setting = {e: {} for e in elem_types}
    valid_settings = {e: {} for e in elem_types}

    for elem_type in elem_types:
        if deviation_detection_task:
            classes = get_deviation_classes()
            classes.remove('Deviating')
            classes = ['Deviating'] + classes  # move Deviating to first pos
        else:
            classes = list(prc_data[elem_type].keys())

        for cls in classes:
            settings_list = prc_data[elem_type][cls].items()
            settings = {metric: np.array(values) for metric, values in settings_list}
            settings['threshold'] = np.array(prc_data['thresholds'])

            # Remove invalid entries
            valid = [settings['recall'][i] is not None and settings['precision'][i] is not None
                     for i in range(len(settings['recall']))]
            valid_settings[elem_type][cls] = {k: v[valid] for k, v in settings.items()}

            if len(valid_settings[elem_type][cls]['threshold']) == 0:
                print(f"[Warning] no valid prc data for class '{cls}' of type '{elem_type}'!")
                continue

            # Get class to use for optimizing thresholds (depends on the detection system used)
            if deviation_detection_task:
                if deviation_detection_model:  # MDD-M allows for type-individual thresholds
                    main_class = cls
                else:
                    main_class = 'Deviating'  # MDD-SC, MDD-MC
            else:
                main_class = 'All'  # object detection

            idx_best = np.argmax(valid_settings[elem_type][main_class]['f1'])
            best_setting[elem_type][cls] = {k: v[idx_best] for k, v in valid_settings[elem_type][cls].items()}

    if tracker is not None:
        tracker.log_pr_curve(partition, valid_settings, log_single_curves=deviation_detection_task)

    print("Best settings:")
    for e_type in best_setting.keys():
        print(f"\t{e_type}:")
        for cls in best_setting[e_type].keys():
            print(f"\t\t{cls}: {PrintDict(best_setting[e_type][cls])}")

    return best_setting


# Script section #######################################################################################################
########################################################################################################################


def run_prc_evaluation(configs, partition, net_dir, element_types=None, thresholds=None, save_results=True,
                       early_stop=False, multiproc=False, num_workers=4):
    """ Loads and evaluates predictions using the given thresholds to create a PR curve and find the optimal threshold
        setting (typically used on validation set).

    Args:
        configs (dict:dict): configuration dictionary (comprising system, model, and train config)
        partition (str): dataset partition (train, val, or test)
        net_dir (str): which model checkpoint to load
        element_types (list[str]): list of element types to consider (lights, poles, signs) (default: get from config)
        thresholds (list[float] | ndarray): list of thresholds to evaluate (default: np.arange(.05, .9, .01))
        save_results (bool): save PRC data (prc.pickle) and log to configured experiment trackers
        early_stop (bool): stops PRC evaluation, if F1 (of first element type) starts to decrease (only if no multiproc)
        multiproc (bool): use multiprocessing (speed up evaluation)
        num_workers (int): number of workers (for multiprocessing)

    Returns:
        best_setting (dict:dict): result from get_best_setting, which can be further used for final test set evaluation

    """
    # Set up parameters
    system_config, model_config, train_config = configs['system'], configs['model'], configs['train']
    log_dir = system_config['log_dir']
    experiment = system_config['experiment']
    deviation_detection_task = train_config['deviation_detection_task']

    if element_types is None:
        element_types = model_config['configured_element_types']

    if thresholds is None:
        thresholds = np.arange(.01, .9, .01)

    print(f"Element types to evaluate: {element_types}")
    print(f"Thresholds to evaluate: {thresholds}")

    # Set up metrics and container to store data for PR curve
    metrics_def = lookups.get_metrics_definition(model_config['classification_active'])

    # Deviation detection: log all prc data for all evaluation states
    if deviation_detection_task:
        prc_cls = get_deviation_classes(include_aggregate_classes=True)
    # Object detection: log for summarizing classes 'All' and 'Mean'
    else:
        prc_cls = ['All']
        if model_config['classification_active']:
            prc_cls.append('Mean')

    prc_data = {}
    for element_type in element_types:
        prc_data['thresholds'] = []
        prc_data[element_type] = lookups.init_nested_dict([prc_cls, metrics_def], [])

    # Load predictions
    predictions_root = log_dir / experiment / partition / net_dir
    predictions_path = predictions_root / 'predictions.pickle'
    with open(predictions_path, 'rb') as f:
        predictions = pickle.load(f)

    # Iterate over thresholds
    if multiproc:
        print("Using multiprocessing")
        multi_processor = helper.SampleProcessor(evaluate_threshold, configs=configs, predictions=predictions,
                                                 elem_types=element_types, norm_conf_mat=False, print_results=True)
        pool = Pool(num_workers)
        loader = pool.imap(multi_processor, thresholds, chunksize=1)

        # Process samples
        loader = tqdm(loader, total=len(thresholds))
        for i, (metrics, _, _, _) in enumerate(loader):
            # Get prc data for configures prc classes
            prc_data['thresholds'].append(thresholds[i])
            for elem_type in element_types:
                for cls in prc_cls:
                    for metric in metrics_def:
                        prc_data[elem_type][cls][metric].append(metrics[elem_type][cls][metric])
        pool.close()
    else:
        # Counters for early stop mechanism
        prev_f1 = -1
        consecutive_declines = 0

        for _, threshold in enumerate(thresholds):
            metrics, _, _, _ = evaluate_threshold(threshold, configs, predictions, element_types, False,
                                                       print_results=True)
            # Get prc data for configures prc classes
            prc_data['thresholds'].append(threshold)
            for elem_type in element_types:
                for cls in prc_cls:
                    for metric in metrics_def:
                        prc_data[elem_type][cls][metric].append(metrics[elem_type][cls][metric])

            if early_stop:
                f1 = metrics[element_types[0]]['All']['f1']
                consecutive_declines = 0 if f1 >= prev_f1 else consecutive_declines + 1
                prev_f1 = f1
                if consecutive_declines >= 2:
                    print("Stopping early ...")
                    break

    tracker = None
    if save_results:
        print("Saving curve data...")
        target_dir = log_dir / experiment / partition / net_dir
        helper.save_data_as_pickle(prc_data, target_dir, 'prc')
        try:
            tracker = ExperimentTracker.from_config(configs)
        except MlflowException as e:
            print("Warning logging: experiment not found in mlruns for export.")
            print(f"MLflow exception message: {e}")

    # Determine best setting (by f1-score) and log pr-curve (if specified)
    best_setting = get_best_setting(prc_data, partition, configs, tracker)

    print(f"Best setting: {best_setting}")
    return best_setting


def run_single_evaluation(configs, partition, net_dir, element_types=None, thresholds=None, from_best_setting=None,
                          save_results=True):
    """ Loads and evaluates predictions using a single threshold configuration. Typically used on the test set using the
        PRC-optimized thresholds found on the validation set.

    Args:
        configs (dict:dict): configuration dictionary (comprising system, model, and train config)
        partition (str): dataset partition (train, val, or test)
        net_dir (str): which model checkpoint to load
        element_types (list[str]): list of element types to consider (lights, poles, signs) (default: get from config)
        thresholds (float | dict:float | dict:dict): confidence threshold, either as single value applied to all element
                                                     types or as a dictionary specifying the threshold for each element
                                                     type separately (MDD-M: additional level for deviation classes)
        from_best_setting (dict:dict): result from get_best_setting used to set the thresholds accordingly (convenience
                                       setting which overrides thresholds)
        save_results (bool): save results to pickle files and log to tracker using the 'best_setting' tag
    """

    # Set up parameters
    system_config, model_config, train_config = configs['system'], configs['model'], configs['train']
    log_dir = system_config['log_dir']
    experiment = system_config['experiment']

    if element_types is None:
        element_types = model_config['configured_element_types']

    if from_best_setting is not None:
        if model_config['deviation_detection_model']:  # Separately optimized thresholds for element AND deviation types
            dev_types = get_deviation_classes(False)
            thresholds = {elem_type: {} for elem_type in element_types}
            for elem_type in element_types:
                thresholds[elem_type] = {dev_type: round(from_best_setting[elem_type][dev_type]['threshold'], 2)
                                         for dev_type in dev_types if dev_type in from_best_setting[elem_type].keys()}

        else:  # Separately optimized thresholds for element types, but not deviation types
            thresholds = {elem_type: from_best_setting[elem_type]['All']['threshold'] for elem_type in element_types}

        print(f"Setting threshold to {PrintDict(thresholds)}")

    elif thresholds is None:
        thresholds = train_config['threshold_score']

    print(f"Element types to evaluate: {element_types}")
    print(f"Threshold(s) to evaluate: {PrintDict(thresholds)}")

    # Load predictions
    predictions_root = log_dir / experiment / partition / net_dir
    predictions_path = predictions_root / 'predictions.pickle'
    with open(predictions_path, 'rb') as f:
        predictions = pickle.load(f)

    # Parse predictions
    print(f"{partition}: Processing threshold {PrintDict(thresholds)}")
    metrics, errors, eval_results, confusion_acc = evaluate_threshold(thresholds, configs, predictions, element_types,
                                                                      True, True)

    # Format values
    configured_classes = lookups.get_class_definition_lookup(model_config, include_all=True)
    if configs['train']['deviation_detection_task']:
        dev_classes = get_deviation_classes()
        configured_classes = {e_type: dev_classes for e_type in element_types}

    metrics_original = copy.deepcopy(metrics)
    for element_type in element_types:
        for cls in configured_classes[element_type]:
            # in cm
            for k, v in errors[element_type][cls].items():
                if v is None:
                    continue
                if 'yaw' not in k:
                    errors[element_type][cls][k] = round(v * 100, 1)
                else:
                    errors[element_type][cls][k] = round(v, 1)

            for k, v in metrics[element_type][cls].items():
                metrics[element_type][cls][k] = round(v, 2) if v is not None else v

    print("Evaluation summary:")
    pprint.pprint(f"Confusion matrix: {confusion_acc}")
    pprint.pprint(metrics)
    pprint.pprint(errors)

    # Save values
    if save_results:
        print("Saving summary and eval results data...")
        try:
            tracker = ExperimentTracker.from_config(configs)
            tracker.log_best_setting(partition, metrics_original, thresholds)
        except MlflowException as e:
            print("Warning logging: experiment not found in mlruns for export.")
            print(f"MLflow exception message: {e}")

        target_dir = log_dir / experiment / partition / net_dir
        helper.save_data_as_pickle(metrics, target_dir, 'metrics')
        helper.save_data_as_pickle(errors, target_dir, 'errors')
        helper.save_data_as_pickle(confusion_acc, target_dir, 'confusion')
        helper.save_data_as_pickle(eval_results, target_dir, 'eval_results')


def run_get_best_setting(configs, partition, net_dir, save_results=False):
    """ Loads prc data from given partition (if it exists) and extracts the best threshold setting.

    Args:
        configs (dict:dict): configuration dictionary (comprising system, model, and train config)
        partition (str): dataset partition (train, val, or test)
        net_dir (str): which model checkpoint to load
        save_results (bool): save prc data to experiment trackers (default: False)

    Returns:
        best_setting (dict:dict): result from get_best_setting, which can be used for final test set evaluation

    """
    prc_path = configs['system']['log_dir'] / configs['system']['experiment'] / partition / net_dir / 'prc.pickle'
    prc_data = helper.load_data_from_pickle(prc_path)
    # pprint.pprint(prc_data)
    tracker = None
    if save_results:
        tracker = ExperimentTracker.from_config(configs)

    best_setting = get_best_setting(prc_data, partition, configs, tracker)
    return best_setting


def run_evaluation():
    """ Main function for PRC evaluation. It is recommended to evaluate the validation set first (PR curves), because
        for the test set (and training set), the best thresholds are acquired from the validation set PRC evaluation.
        Note that the experiments, partitions, and net_dirs used can also be configured via command-line.

    Settings:
        thresholds (list[float] | ndarray): list of thresholds to evaluate (only used for PRC evaluation of val set)
        save_results (bool): save all PRC and threshold evaluation results as pickle files and log to trackers
        early_stop (bool): stops PRC evaluation, if F1 (of first element type) starts to decrease (only if no multiproc)
        multiproc (bool): use multiprocessing to speed up evaluation (using 4 workers as default)
        cfg (dict:list): combinations of models and datasets to evaluate (can be configured via CL args)

    """

    ##################
    # Settings
    thresholds = np.arange(0.01, 0.9, .01)
    save_results = True
    early_stop = False
    multiproc = True

    # Default settings
    cfg = {
        'experiments': [
            'dd-s3-60m',
            'dd-s2-60m',
            'dd-s1-60m',
            'er-lps-3dhdnet-60m'
        ],
        'partitions': [
            'val',
            'test'
        ],
        'net_dirs': ['checkpoint']
    }

    # Override settings with any given command-line parameters
    parser = config_parser.create_argparser_from_dict(cfg)
    args = parser.parse_args()
    config_parser.update_dict_from_args(args, cfg)
    #################

    log_dir = get_system_paths()['log_dir']
    t_start = time()

    for experiment in cfg['experiments']:
        for partition in cfg['partitions']:
            for net_dir in cfg['net_dirs']:
                # Read config file
                config_path = log_dir / experiment / 'config.ini'
                configs = config_parser.parse_config(config_path, False)

                # Overwrite config file run name (if run is copied by hand)
                configs['system']['experiment'] = experiment

                log_file = configs['system']['log_dir'] / experiment / f"eval_log_{partition}.txt"
                set_logger(log_file)

                print(f"Running: {experiment}, {partition}, {net_dir}")

                if 'val' in partition:
                    # Compute PR curves for validation set, then evaluate on best thresholds found and save results
                    best_setting = run_prc_evaluation(configs, partition, net_dir, thresholds=thresholds,
                                                      save_results=save_results, early_stop=early_stop, multiproc=multiproc)
                    run_single_evaluation(configs, partition, net_dir, from_best_setting=best_setting,
                                          save_results=save_results)

                elif 'test' in partition or 'train' in partition:
                    # Get best setting from validation set, then evaluate on test set and save results
                    best_setting = run_get_best_setting(configs, 'val', net_dir)
                    run_single_evaluation(configs, partition, net_dir, from_best_setting=best_setting,
                                          save_results=save_results)

    print(f"Time taken: {time() - t_start:.2f} seconds")


# Run section ##########################################################################################################
########################################################################################################################


def main():
    run_evaluation()


if __name__ == "__main__":
    main()
