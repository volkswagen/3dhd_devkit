""" Script to set up and start a new experiment """

import sys
sys.path.append("..")
sys.path.append("../..")

import os
from time import time
from pathlib import Path

import torch.multiprocessing as mp

from deep_learning import root_dir
from deep_learning.experiments.run_evaluation import run_prc_evaluation, run_single_evaluation, run_get_best_setting
from deep_learning.experiments.run_inference import run_inference
from deep_learning.training import trainer
from deep_learning.util import config_parser, tracking
from utility.logger import set_logger


# Module functions #####################################################################################################
########################################################################################################################


def setup_config_and_logging(config_file):
    """ Sets up the configuration (from the given file path and args), and proper logging functionality (select a unique
    experiment name, start console logger and mlflow tracking, save configuration in log folder).

    Args:
        config_file (str | Path): full path to a configuration (.ini) file

    Returns:
        configs (dict:dict): the parsed and updated experiment configuration
    """

    # Parse config file and command-line arguments (default settings are default_config.ini, overwrite settings)
    configs = config_parser.parse_config(config_file)

    # Read command line arguments and update config correspondingly
    configs_merged = {}
    for config in configs.values():
        configs_merged.update(config)
    parser = config_parser.create_argparser_from_dict(configs_merged)
    args = parser.parse_args()
    config_parser.update_configs_from_args(args, configs)
    system_config, _, train_config = configs['system'], configs['model'], configs['train']
    system_config['config_file_path'] = config_file

    # Set up available logging dir and start console logging
    preferred_experiment_name = system_config['experiment']
    experiment_name = preferred_experiment_name
    current_log_dir = system_config['log_dir'] / experiment_name

    idx = 1
    while current_log_dir.exists():
        print(f"[Log setup] {current_log_dir} already exists!")
        experiment_name = f"{preferred_experiment_name}-{idx}"
        current_log_dir = system_config['log_dir'] / experiment_name
        idx += 1

    system_config['experiment'] = experiment_name
    system_config['log_file_path'] = current_log_dir / 'console_log.txt'
    set_logger(system_config['log_file_path'])
    print(f"[Log setup] Will save this experiment as '{experiment_name}'")

    # Set up and start mlflow tracking, if configured
    if 'mlflow' in train_config['tracking_backend']:
        exp_id, run_id = tracking.start_new_mlflow_run(system_config['log_dir'], train_config['mlflow_group_name'],
                                                       experiment_name)
        system_config['mlflow_experiment_id'] = exp_id
        system_config['mlflow_run_id'] = run_id
        tracking.set_artifact_path(current_log_dir)

    # Ensure that val partition will be evaluated first (necessary to get correct threshold to use for other partitions)
    if 'val' in train_config['inference_partitions'] and len(train_config['inference_partitions']) > 1:
        train_config['inference_partitions'].remove('val')
        train_config['inference_partitions'] = ['val'] + train_config['inference_partitions']

    config_parser.save_config(current_log_dir / 'config.ini', configs)
    return configs


def run_train(configs):
    """ Starts training using the given configuration (using DDP if configured). """
    system_config = configs['system']

    if system_config['ddp']:
        os.environ['MASTER_PORT'] = '7000'
        os.environ['MASTER_ADDR'] = 'localhost'
        mp.spawn(trainer.train, nprocs=system_config['num_gpus'], args=(configs,))
    else:
        trainer.train(system_config['train_device'], configs)


def main():
    """ Starts a new experiment using the default config and given command line parameters.
        For convenience, it also includes model inference and evaluation after training.
    """
    t_start = time()

    config_file = os.path.join(root_dir(), "configurations", "default_config.ini")
    configs = setup_config_and_logging(config_file)

    run_train(configs)
    print(f"Time taken for training: {time() - t_start}")

    checkpoint = 'checkpoint'
    for partition in configs['train']['inference_partitions']:
        print(f"Starting inference on {partition} partition...")
        run_inference(configs, partition, checkpoint)

    for partition in configs['train']['inference_partitions']:
        print(f"Starting evaluation on {partition} partition...")
        if partition == 'val':
            best_setting = run_prc_evaluation(configs, partition, checkpoint)
        else:
            # run_prc_evaluation(configs, partition, checkpoint)
            best_setting = run_get_best_setting(configs, 'val', checkpoint)

        run_single_evaluation(configs, partition, checkpoint, from_best_setting=best_setting)

    print(f"Time taken (total): {time() - t_start}")


if __name__ == '__main__':
    main()
