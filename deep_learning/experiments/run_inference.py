""" Script to run inference on a dataset (typically validation or test set) using a trained model """

import sys

sys.path.append("../..")

import os
from time import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from dataset import helper
from deep_learning.losses import losses, target_generator, weights
from deep_learning.models import model_builder
from deep_learning.training import dataset_loader, dataset_splitter
from deep_learning.util import config_parser, util
from deep_learning.postprocessing import deviation_extraction, element_extraction
from deep_learning.preprocessing import preprocessor
from utility.system_paths import get_system_paths
from utility.logger import set_logger


# Module functions #####################################################################################################
########################################################################################################################


def setup_test_dataset(configs: dict, partition: str) -> DataLoader:
    """ Sets up dataloader for inference. """
    system_config, model_config, train_config = configs['system'], configs['model'], configs['train']

    # Configure dataset
    print("Configuring dataset...")
    splitter = dataset_splitter.MddDatasetSplitter(system_config['dataset_dir'], partitions=[partition],
                                                   version=train_config['dataset_version'])
    splitter.load_partitions()
    if train_config['deviation_detection_task'] and train_config['load_generated_deviations']:
        splitter.load_generated_deviations_and_occlusions(train_config['generated_deviations_setting'],
                                                          train_config['occlusion_prob'])
    print("Configuring dataset... done!")

    # Setup preprocessor
    print("Configuring preprocessor and target generator...")
    pre = preprocessor.Preprocessor(configs)

    pre.configure_hdpc_fm(shape=model_config['fm_shape_hdpc'],
                          fm_type=model_config['fm_type_hdpc'],
                          load_lvl=train_config['load_lvl_hdpc'])

    pre.configure_map_fm(shape=model_config['fm_shape_map'],
                         fm_type=model_config['fm_type_map'],
                         load_lvl=train_config['load_lvl_map'])

    target_gen = target_generator.TargetGenerator(model_config)
    print("Configuring preprocessor and target generator ... done!")

    print("Creating dataset...")
    mdd_dataset_test = dataset_loader.MddDataset(
        splitter.partition[partition], pre, target_gen, augment=False)

    dataloader = DataLoader(mdd_dataset_test,
                            num_workers=system_config['num_workers'],
                            shuffle=False,
                            pin_memory=False,
                            batch_size=1,
                            collate_fn=dataset_loader.merge_second_batch)
    print("Creating dataset... done!")

    return dataloader


# Script section #######################################################################################################
########################################################################################################################


def run_inference(configs, partition='test', net_dir='checkpoint', n_samples=None):
    """ Performs inference on the given dataset partition using the given model and saves results as a pickle file.

    Args:
        configs (dict:dict): configuration dictionary (comprising system, model, and train config)
        partition (str): dataset partition to process (default: 'test')
        net_dir (str): which model checkpoint to load (default: 'checkpoint')
        n_samples (None | int): limit number of samples to process; if None, the full partition is processed
    """

    system_config, model_config, train_config = configs['system'], configs['model'], configs['train']
    log_dir = system_config['log_dir']
    experiment = system_config['experiment']

    # Set batch_size to 1 in train_config to correctly build the model
    train_config['batch_size'] = 1

    dataloader = setup_test_dataset(configs, partition=partition)

    # Set up model
    net_path = os.path.join(log_dir, experiment, 'train', net_dir, 'net.pt')
    print(f"Loading model: {net_path}")
    with torch.no_grad():
        net = model_builder.build(model_config, train_config)

    if system_config['multi_gpu'] or system_config['ddp']:
        net = torch.nn.DataParallel(net, device_ids=[0])

    # Load loss
    weight_dict = train_config['weight_dict']
    weight_dict_classes = train_config['weight_dict_classes']
    criterion = losses.get_criterion(
        model_config['loss_type'], weight_dict, weight_dict_classes, train_config['use_class_specific_weights'])
    criterion = criterion.to(system_config['train_device'])

    # Restore model parameters
    state_dict = torch.load(net_path)
    net.load_state_dict(state_dict)

    # Set to evaluation mode
    net.to(system_config['train_device'])
    net.eval()

    # Run inference
    it_test = 0
    predictions = []
    timings = {
        't_load': [],
        't_trans': [],
        't_net': [],
        't_loss': [],
        't_post': [],
        't_per_sample_no_nms': [],
        't_iteration': []
    }
    amp_context = autocast() if system_config['use_amp'] else nullcontext()
    deviation_detection_model = model_config['deviation_detection_model']
    print("Starting inference.")
    ts_end = time()
    for example_batched in dataloader:
        it_test += 1

        if n_samples is not None and (it_test % n_samples == 0):
            break

        torch.cuda.synchronize()
        ts_start = time()

        # Transfer to GPU
        example_torch = util.example_convert_to_torch(example_batched, torch.float32, system_config['train_device'])

        torch.cuda.synchronize()
        ts_transfer = time()

        target_dict = example_torch['target_dict']

        # Apply model on input
        with torch.no_grad():
            with amp_context:
                out_dict = net(example_torch)

        torch.cuda.synchronize()
        ts_net = time()

        # Calculate validation loss
        norm_factors = weights.get_loss_norm_by_elements(example_batched['gt_objects'])
        total_loss, loss_dict, focal_loss2d_dict = criterion(out_dict, target_dict, norm_factors=norm_factors)

        for element_type in model_config['configured_element_types']:
            for key, loss in loss_dict[element_type].items():
                loss_dict[element_type][key] = loss.item()

        torch.cuda.synchronize()
        ts_loss = time()

        # Detach output and append to predictions
        total_loss = total_loss.detach().cpu()
        for element_type in model_config['configured_element_types']:
            for k, v in out_dict[element_type].items():
                out_dict[element_type][k] = v.detach().cpu().numpy()

            for k, v in target_dict[element_type].items():
                target_dict[element_type][k] = v.detach().cpu().numpy()

            focal_loss2d_dict[element_type] = focal_loss2d_dict[element_type].detach().cpu().numpy()

        total_loss = total_loss.item()

        # Create predictions
        gt_objects = example_batched['gt_objects'][0]

        ts_before_nms = time()
        train_config['threshold_score'] = 0.01  # set minimal score threshold for NMS (i.e., perform NMS only)
        # Get predictions from grid
        if deviation_detection_model:
            map_lut = example_batched['map_lut'][0]
            predictions_lists_nms = deviation_extraction.lut_based_post_processing(
                out_dict, target_dict, configs, map_lut)
        else:
            predictions_lists = element_extraction.convert_grid_to_element_lists(
                out_dict, target_dict, model_config)
            predictions_lists_nms = element_extraction.nms(predictions_lists, configs)

        prediction = {
            'predictions_lists': predictions_lists_nms,
            'loss': total_loss,
            'sample_id': example_batched['sample_ids'][0],
            'run_name': example_batched['run_names'][0],
            'gt_objects': gt_objects,
        }
        predictions.append(prediction)

        t_iteration = time() - ts_start
        t_per_sample_no_nms = (ts_before_nms - ts_start) / train_config['batch_size']
        print(f"{partition.capitalize()} Set. iteration: {it_test}, loss: {total_loss:.3f}, "
              f"t_iteration: {t_iteration:.3f}, t_per_sample_no_nms: {t_per_sample_no_nms:.3f}")

        timings['t_load'].append(ts_start - ts_end)
        timings['t_trans'].append(ts_transfer - ts_start)
        timings['t_net'].append(ts_net - ts_transfer)
        timings['t_loss'].append(ts_loss - ts_net)
        timings['t_post'].append(t_iteration - ts_before_nms)
        timings['t_per_sample_no_nms'].append(t_per_sample_no_nms)
        timings['t_iteration'].append(t_iteration)
        ts_end = time()

    print("Timings average:")
    for timing_type, _ in timings.items():
        timing = round(float(np.mean(timings[timing_type])), 4)
        print(f"{timing_type}: {timing} s")

    print("Saving predictions to pickle.")
    target_dir = log_dir / experiment / partition / net_dir
    helper.save_data_as_pickle(predictions, target_dir, 'predictions')
    print("Done")


# Run section ##########################################################################################################
########################################################################################################################


def main():
    """ Perform data inference using the specified models and partition partitions.
        Includes logging and saving results.

    Settings:
        log_progress (bool): log progress to a text file if true
        cfg (dict:list): combinations of models and datasets to perform inference on (can be configured via CL args)
    """
    ##################
    # Settings
    log_progress = True
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
    ##################

    # Override settings with any given command-line parameters
    parser = config_parser.create_argparser_from_dict(cfg)
    args = parser.parse_args()
    config_parser.update_dict_from_args(args, cfg)

    log_dir = get_system_paths()['log_dir']
    for experiment in cfg['experiments']:
        for partition in cfg['partitions']:
            for net_dir in cfg['net_dirs']:
                # Read config file
                default_config_path = log_dir / experiment / 'config.ini'
                configs = config_parser.parse_config(default_config_path, False)

                # Overwrite config file run name (if run is copied by hand)
                configs['system']['experiment'] = experiment

                if log_progress:
                    log_file = configs['system']['log_dir'] / experiment / f"inference_log_{partition}.txt"
                    set_logger(log_file)

                print(f"Running: {experiment}, {partition}, {net_dir}")
                run_inference(configs, partition=partition, net_dir=net_dir)


if __name__ == "__main__":
    main()
