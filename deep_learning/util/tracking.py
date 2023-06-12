""" Module containing class and functions for managing experiment tracking with tensorboard and mlflow """

from pathlib import Path
from typing import List, Union, Optional

import mlflow
import yaml
from torch.utils.tensorboard import SummaryWriter

from dataset import map_deviation
from deep_learning.util import lookups


class ExperimentTracker:
    """
    Manages logging to tracking frameworks like mlflow and tensorboard
    """

    def __init__(self,
                 tracking_backend: List[str],
                 log_dir: Union[Path, str],
                 experiment_name: str,
                 mlflow_run_id: Optional[str] = None):
        """
        Args:
            tracking_backend: list of tracking backends to use (options: 'tensorboard', 'mlflow')
            log_dir: root directory where all logs are stored
            experiment_name: name of current experiment
            mlflow_run_id: unique run_id provided by mlflow (needed to log to an existing mlflow run)
        """
        self.mlflow = 'mlflow' in tracking_backend
        self.tensorboard = 'tensorboard' in tracking_backend
        self.experiment_dir = Path(log_dir) / experiment_name
        self.log_dir = Path(log_dir)

        self.writers = {}  # SummaryWriters will be lazy-instantiated on demand

        if self.mlflow:
            assert mlflow_run_id is not None
            resume_mlflow_run(log_dir, mlflow_run_id)

    @classmethod
    def from_config(cls, configs):
        """ Alternative constructor to conveniently set up the tracker from configs. """
        return cls(configs['train']['tracking_backend'],
                   configs['system']['log_dir'],
                   configs['system']['experiment'],
                   configs['system']['mlflow_run_id'])

    def _check_partition(self, partition: str):
        """ Checks if a tensorboard writer already exists for the given partition, and creates one if not """
        if self.tensorboard and partition not in self.writers:
            writer_dir = self.experiment_dir / partition
            writer_dir.mkdir(parents=True, exist_ok=True)
            self.writers[partition] = SummaryWriter(log_dir=str(writer_dir), flush_secs=10)

    def log_metric(self, partition: str, tag: str, value, it: int = 0):
        """ Logs a singular value at the given partition, tag, and with the given it(eration). """
        self._check_partition(partition)
        if self.mlflow:
            mlflow.log_metric(f"{partition}/{tag}", value, it)
        if self.tensorboard:
            self.writers[partition].add_scalar(tag, value, it)

    def log_losses(self, partition, total_loss, loss_dict, it=0):
        """ Logs the given losses (total_loss and loss_dict) to the specified tracking backends.

        Args:
            partition (str): which partition is used (train, val, or test)
            total_loss (float): total loss (single value)
            loss_dict (dict): dict comprising specific losses for each element type and loss type (e.g. focal loss)
            it (int): iteration number used to track progress of specific values (typically for training)
        """

        self.log_metric(partition, 'loss_total', total_loss, it)
        multitask = len(loss_dict.keys()) > 1

        for elem_type, elem_loss_dict in loss_dict.items():
            type_str = f"{elem_type}/" if multitask else ''
            for key, loss in elem_loss_dict.items():
                self.log_metric(partition, f'losses/{type_str}{key}', loss, it)

    def log_errors_and_metrics(self, partition, errors, metrics, model_config, deviation_detection, it=0):
        """ Logs the given errors and metrics evaluations.

        Args:
            partition (str): which partition is used (train, val, or test)
            errors (dict): dict containing measured regression errors for each element type
            metrics (dict): dict containing computed metrics for each element type
            model_config (dict): model settings
            deviation_detection (bool): whether deviation detection (MDD) is active
            it (int): iteration number used to track progress of specific values (typically for training)
        """

        element_types = model_config['configured_element_types']

        if deviation_detection:
            dev_classes = map_deviation.get_deviation_classes()
            configured_classes = {e_type: dev_classes for e_type in element_types}
        else:
            log_mean = model_config['classification_active']
            configured_classes = lookups.get_class_definition_lookup(model_config, include_mean=log_mean)

        for e_type in element_types:
            type_str = f"{e_type}/" if len(element_types) > 1 else ''

            for cl in configured_classes[e_type]:
                for metric, value in metrics[e_type][cl].items():
                    if value is not None:  # Only add if value is not None
                        self.log_metric(partition, f'metrics/{type_str}{metric}/{cl}', value, it)

                for error, value in errors[e_type][cl].items():
                    if value is not None:
                        self.log_metric(partition, f'errors/{type_str}{error}/{cl}', value, it)

    def log_plot(self, partition, tag, fig, it=0):
        """ Logs the given plot/figure. Note: For space efficiency, the plot is only logged in one backend (preferably
            tensorboard).

        Args:
            partition (str): which partition is used (train, val, or test)
            tag (str): descriptive tag to log the plot as
            fig (matplotlib.figure.Figure): plot to save
            it (int): iteration number used to track progress of specific values (typically for training)
        """

        self._check_partition(partition)

        # Log plots only once (preferably on tensorboard, because more space-efficient)
        if self.tensorboard:
            self.writers[partition].add_figure(tag, fig, it)
        elif self.mlflow:
            # mlflow.log_figure(fig, f"{partition}/{tag}/{it:06d}.png")
            save_dir = get_artifact_path() / partition / tag
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / f"{it:06d}.png")

    def log_config(self, configs):
        """ Logs general configuration in mlflow. If mlflow is not configured, this function does nothing.

        Args:
            configs (dict:dict): configuration dictionary (comprising system, model, and train config)
        """

        if not self.mlflow:  # this function only supports mlflow
            return

        # Set up parameters that should be logged
        model_params = configs['model'].keys()
        # Log only parameters of used element types
        for element_type, keyword in [('poles', 'pole'), ('signs', 'sign'), ('lights', 'light')]:
            if element_type not in configs['model']['configured_element_types']:
                model_params = [k for k in model_params if not k.startswith(keyword)]

        train_params = configs['train'].keys()
        system_params = configs['system'].keys()

        params_to_log = {'model': model_params,
                         'train': train_params,
                         'system': system_params}

        for config_type, config in configs.items():
            for key in params_to_log[config_type]:
                if key in config.keys():
                    mlflow.log_param(f"{config_type}/{key}", config[key])
                else:
                    print(f"[Warning] mlflow tracking: {config_type} configuration '{key}' does not exist!")

    def log_pr_curve(self, partition, valid_settings, log_single_curves=False):
        """ Logs PR curves in numerical format (i.e., not as figure) for custom visualization by tracking backends.

        Args:
            partition (str): which partition is used (train, val, or test)
            valid_settings (dict:dict): contains valid (i.e., not None) metrics for each evaluated element type, class,
                                        and threshold
            log_single_curves (bool): if True, also logs singular metrics curves (x: threshold, y: precision/recall/f1)
        """
        element_types = list(valid_settings.keys())

        for e_type in element_types:
            type_str = f"{e_type}/" if len(element_types) > 1 else ''
            for cls in valid_settings[e_type].keys():

                # Log pr curve values as 'per mil', because steps (x-axis) have to be integers
                precision_per_mil = [round(prec * 1000) for prec in valid_settings[e_type][cls]['precision']]
                recall_per_mil = [int(round(rec * 1000)) for rec in valid_settings[e_type][cls]['recall']]
                for prec, rec in zip(precision_per_mil, recall_per_mil):
                    self.log_metric(partition, f"eval/pr_curve_per_mil/{type_str}{cls}", prec, rec)

                if log_single_curves:
                    threshold_per_cent = [int(round(thr * 100)) for thr in valid_settings[e_type][cls]['threshold']]
                    for prec, thr in zip(precision_per_mil, threshold_per_cent):
                        self.log_metric(partition, f"eval/precision_thr_per_mil/{type_str}{cls}", prec, thr)
                    for rec, thr in zip(recall_per_mil, threshold_per_cent):
                        self.log_metric(partition, f"eval/recall_thr_per_mil/{type_str}{cls}", rec, thr)

                    f1_per_mil = [round(f1 * 1000) for f1 in valid_settings[e_type][cls]['f1']]
                    for f1, thr in zip(f1_per_mil, threshold_per_cent):
                        self.log_metric(partition, f"eval/f1_thr_per_mil/{type_str}{cls}", f1, thr)

    def log_best_setting(self, partition, metrics, thresholds):
        """ Logs metrics, measured using a specific threshold setting, under the tag 'best_setting'.

        Args:
            partition (str): which partition is used (train, val, or test)
            metrics (dict:dict): contains the metrics of every configured element type and class, using a specific
                                 threshold setting
            thresholds (float | dict:float | dict:dict): contains the corresponding thresholds applied for every element type and class
        """

        element_types = list(metrics.keys())

        for e_type in element_types:
            type_str = f"{e_type}/" if len(element_types) > 1 else ''
            for cls in metrics[e_type].keys():
                # Log metrics
                for metric, val in metrics[e_type][cls].items():
                    if val is None:
                        continue
                    self.log_metric(partition, f"eval/best_setting/{type_str}{metric}/{cls}", val)

                # Log corresponding threshold for reproducibility
                if isinstance(thresholds, float):
                    threshold = thresholds
                elif isinstance(thresholds[e_type], float):
                    threshold = thresholds[e_type]
                elif cls in thresholds[e_type]:
                    threshold = thresholds[e_type][cls]
                else:
                    continue
                self.log_metric(partition, f"eval/best_setting/{type_str}threshold/{cls}", threshold)


def start_new_mlflow_run(log_dir, mlflow_group_name, mlflow_run_name):
    """ Starts a new mlflow run and returns the mlflow experiment id and run id.

    Args:
        log_dir (str | Path): root directory where all logs are stored (parent directory of 'mlruns' folder)
        mlflow_group_name (str): name of the experiment group ('experiment' in mlflow terms)
        mlflow_run_name (str): name of the current experiment ('run' in mlflow terms)
    """

    if mlflow.active_run() is not None:
        mlflow.end_run()
    log_dir = Path(log_dir) / 'mlruns'
    mlflow.set_tracking_uri(f"file:{str(log_dir)}")
    mlflow.set_experiment(mlflow_group_name)
    active_run = mlflow.start_run(run_name=mlflow_run_name)
    return active_run.info.experiment_id, active_run.info.run_id


def resume_mlflow_run(log_dir, run_id):
    """ Resumes an existing mlflow run (meaning that subsequent logs will be stored under this run).

    Args:
        log_dir: root directory where all logs are stored (parent directory of 'mlruns' folder)
        run_id: unique run_id referring to the experiment to resume (provided by mlflow)
    """

    # Safety check
    active_run = mlflow.active_run()
    if active_run is not None:
        if active_run.info.run_id == run_id:
            return
        else:
            mlflow.end_run()

    log_dir = Path(log_dir) / 'mlruns'
    mlflow.set_tracking_uri(f"file:{str(log_dir)}")
    mlflow.start_run(run_id=run_id)


def clean_mlflow_path(path):
    """ Removes 'file:' prefix from paths found in mlflow config files. """
    prefix = r'file:'
    if path.startswith(prefix):
        path = path[len(prefix):]
    return Path(path)


def get_artifact_path():
    """ Returns path where mlflow artifacts (images etc.) are currently being logged. """
    return clean_mlflow_path(mlflow.get_artifact_uri())


def set_artifact_path(path):
    """ Sets the mlflow artifact path (where images etc. are going to be stored) to the given path. """
    # Set the artifact path by updating the mlflow run meta-file
    meta_file = clean_mlflow_path(mlflow.get_tracking_uri())
    meta_file = Path(meta_file) / mlflow.active_run().info.experiment_id / mlflow.active_run().info.run_id / 'meta.yaml'
    with open(meta_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    data['artifact_uri'] = f"file:{path}"
    with open(meta_file, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)
