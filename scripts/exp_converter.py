
from utility.system_paths import get_system_paths
from os import path
from enum import Enum

import shutil
import pickle

from deep_learning.util import config_parser

# Script section #######################################################################################################
########################################################################################################################

def run_convert_exp():
    exp_name = 'dd-s3-60m'
    suffix = 'cnv'
    exp_name_cnv = exp_name + "-" + suffix
    paths = get_system_paths()
    log_dir = paths['log_dir']

    source = log_dir / exp_name
    target = log_dir / exp_name_cnv
    if not path.exists(target):
        shutil.copytree(source, target)

    # Modify config.ini
    config_path = target / 'config.ini'

    # Load predictions
    partitions = ['val', 'test']
    net_dir = 'checkpoint'
    for part in partitions:
        predictions_root = log_dir / exp_name_cnv / part / net_dir
        predictions_path = predictions_root / 'predictions.pickle'
        with open(predictions_path, 'rb') as f:
            predictions = pickle.load(f)

    ##################
    # Override settings
    configs = config_parser.parse_config(config_path, False)
    configs['system']['experiment'] = configs['system'].pop('run_name')
    configs['model']['light_default_width'] = configs['model'].pop('light_default_size')
    configs['model']['element_size_factor'] = configs['model'].pop('threshold_factor')
    configs['train']['load_generated_deviations'] = configs['train'].pop('load_presaved_deviations')
    configs['train']['generated_deviations_setting'] = configs['train'].pop('presaved_deviations_setting')
    configs['model']['pole_iosa_threshold'] = configs['model'].pop('pole_iou_threshold')
    configs['model']['light_iosa_threshold'] = configs['model'].pop('light_iou_threshold')
    config_parser.save_config(config_path, configs)


# Run section ##########################################################################################################
########################################################################################################################


def main():
    run_convert_exp()




if __name__ == "__main__":
    main()
