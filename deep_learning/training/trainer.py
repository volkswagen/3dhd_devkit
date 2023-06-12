""" Module comprising main training procedure. """

import os
from contextlib import nullcontext
from multiprocessing import Manager
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import helper, map_deviation
from dataset_api.pc_reader import read_binary_hdpc_file
from deep_learning.evaluation import metrics_deviation_detection, metrics_element_detection
from deep_learning.losses import losses, target_generator, weights
from deep_learning.models import model_builder
from deep_learning.postprocessing import comparison, deviation_extraction, element_extraction
from deep_learning.preprocessing import preprocessor
from deep_learning.training import dataset_loader, dataset_splitter
from deep_learning.util import lookups, sample_filter, tracking, util
from utility.logger import set_logger
from visualization import visualization_evaluation as vis_eval


# Configuration ########################################################################################################
########################################################################################################################

def configure_pipeline(configs):
    """ Sets up dataloaders according to configs, including (optional) sample filter, preprocessor, and target generator

    Args:
        configs (dict:dict): configuration dictionary (comprising system, model, and train config)

    Returns:
        dataloaders (dict:Dataloader): dict containing dataloaders for train and val partition

    """
    model_config, system_config, train_config = configs['model'], configs['system'], configs['train']

    # Configure dataset
    print("Configuring dataset...")
    partitions_to_load = ['train', 'val']
    splitter = dataset_splitter.MddDatasetSplitter(system_config['dataset_dir'], partitions=partitions_to_load,
                                                   version=train_config['dataset_version'])
    splitter.load_partitions()
    if train_config['deviation_detection_task'] and train_config['load_generated_deviations']:
        splitter.load_generated_deviations_and_occlusions(train_config['generated_deviations_setting'],
                                                          train_config['occlusion_prob'])

    print("Filtering dataset...")
    t_start = time()
    for partition in partitions_to_load:
        samples_filtered = sample_filter.filter_samples_by_num_elements_by_configs(splitter.partition[partition],
                                                                                   configs)
        if system_config['ddp']:
            # Dataset must be evenly distributed across gpus (otherwise ddp will desync and freeze at end)
            overflow = len(samples_filtered) % system_config['world_size']
            if overflow != 0:
                samples_filtered = samples_filtered[:-overflow]

        splitter.partition[partition] = samples_filtered
    print(f"Time taken for filtering: {time() - t_start:.2f} s")

    # Load all pc tiles if configured
    map_meta_data_path = os.path.join(system_config['map_metadata_dir'], 'HDPC_TileDefinition.json')
    tile_shapes = helper.load_from_json(map_meta_data_path)
    pc_tiles = None
    if train_config['load_lvl_hdpc'] == 'online' and train_config['prefetch_pc_tiles']:
        print("Prefetching tiles...")
        manager = Manager()
        pc_tiles = manager.dict()
        for tile in tqdm(tile_shapes):
            # print(f"Online training. Loading point cloud tile: {tile['name']} ")
            path = os.path.join(system_config['hdpc_tiles_dir'], tile['name'] + '.bin')
            pc_tiles[tile['name']] = read_binary_hdpc_file(path, unnorm=False)

    # Setup preprocessor
    print("Configuring preprocessor and anchor generator...")
    pre = preprocessor.Preprocessor(configs)

    pre.configure_hdpc_fm(shape=model_config['fm_shape_hdpc'],
                          fm_type=model_config['fm_type_hdpc'],
                          load_lvl=train_config['load_lvl_hdpc'],
                          pc_tiles=pc_tiles)

    pre.configure_map_fm(shape=model_config['fm_shape_map'],
                         fm_type=model_config['fm_type_map'],
                         load_lvl=train_config['load_lvl_map'])

    target_gen = target_generator.TargetGenerator(model_config)

    print("Setting up dataloaders...")
    # Create datasets
    mdd_dataset_train = dataset_loader.MddDataset(splitter.partition['train'], pre, target_generator=target_gen,
                                                  augment=True, log=system_config['log_file_path'])
    mdd_dataset_val = dataset_loader.MddDataset(splitter.partition['val'], pre, target_generator=target_gen,
                                                augment=False, log=system_config['log_file_path'])

    # Create sampler
    if system_config['ddp']:
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            mdd_dataset_train,
            num_replicas=system_config['world_size'],
            rank=system_config['rank']
        )
        sampler_eval = torch.utils.data.distributed.DistributedSampler(
            mdd_dataset_val,
            num_replicas=system_config['world_size'],
            rank=system_config['rank']
        )
    else:
        sampler_train, sampler_eval = None, None

    pin_memory = system_config['ddp'] or system_config['multi_gpu']  # Speeds up data transfer CPU -> GPU

    dataloader_train = DataLoader(mdd_dataset_train,
                                  batch_size=train_config['batch_size'],
                                  collate_fn=dataset_loader.merge_second_batch,
                                  num_workers=system_config['num_workers'],
                                  shuffle=sampler_train is None,
                                  pin_memory=pin_memory,
                                  sampler=sampler_train,
                                  drop_last=True)

    dataloader_val = DataLoader(mdd_dataset_val,
                                batch_size=train_config['batch_size'],
                                collate_fn=dataset_loader.merge_second_batch,
                                num_workers=system_config['num_workers'],
                                shuffle=sampler_eval is None,
                                pin_memory=pin_memory,
                                sampler=sampler_eval,
                                drop_last=True)

    dataloaders = {'train': dataloader_train, 'val': dataloader_val}

    print("Data pipeline setup done!")
    return dataloaders


def configure_model(configs):
    """ Sets up and returns the model, optimizer, and loss criterion, according to the given config. If specified,
        the model is placed on a GPU (as single-GPU or part of a DP or DDP setup). Also, weights are initialized
        randomly or, if specified, from a pretrained model.

    Args:
        configs (dict:dict): configuration dictionary (comprising system, model, and train config)

    Returns:
        (net, optimizer, criterion): the network/model, optimizer, and loss criterion, to use in a pytorch pipeline
    """
    model_config, system_config, train_config = configs['model'], configs['system'], configs['train']

    # Build model
    print("Building model...")
    device = system_config['train_device']  # e.g., "cuda:0"
    net = model_builder.build(model_config, train_config)

    # DDP or Multi-GPU: set up model on specified GPU, otherwise keep on CPU
    if system_config['ddp']:
        rank = system_config['rank']  # e.g., 0
        if system_config['use_batch_sync']:
            print("Using batch sync!")
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net.to(device)
        net = nn.parallel.DistributedDataParallel(net, device_ids=[rank])
        print("Using ", system_config['num_gpus'], "GPUs for training.")
    else:
        # Multiple-GPU
        if system_config['multi_gpu'] and torch.cuda.device_count() > 1:
            print("Using ", len(system_config['train_device_ids']), "GPUs for training.")
            net = torch.nn.DataParallel(net, device_ids=system_config['train_device_ids'])
            net.to(device)
        else:
            print(f"Placing network on: {system_config['train_device']}")
            net.to(device)

    # Init model
    print("Initializing model weights...")
    net.apply(model_builder.init_weights)
    # Pretrained layers if configured
    if train_config['init_mode'] == 'pretrained':
        pretrained_net = train_config['experiment_name_pretrained']
        print(f"Model config: Init from pretrained model: {pretrained_net}")
        pretrained_net_path = os.path.join(system_config['log_dir'], pretrained_net, 'train', 'checkpoint', 'net.pt')
        model_builder.init_from_pretrained(net, pretrained_net_path)

    # Optimizer
    print("Setting up optimizer and loss...")
    if train_config['optimizer'] == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=train_config['learning_rate'],
                               weight_decay=train_config['weight_decay'])
    else:
        raise ValueError(f"Optimizer name '{train_config['optimizer']}' is not defined!")

    # Criterion
    weight_dict = train_config['weight_dict']
    weight_dict_classes = train_config['weight_dict_classes']
    criterion = losses.get_criterion(model_config['loss_type'], weight_dict, weight_dict_classes,
                                     train_config['use_class_specific_weights'])
    criterion = criterion.to(system_config['train_device'])

    print("Model setup done!")
    return net, optimizer, criterion


# Main #################################################################################################################
########################################################################################################################


def train(gpu, configs):
    """
    Starts a full training procedure on the specified device (gpu), according to the given configs. Also includes
    data pipeline setup and validation set evaluation. Note that this function only covers a single training process,
    so for multiprocessing (including DDP), this function needs to be called in a separate thread on every device
    (e.g., using multiprocessing.spawn).

    Args:
        gpu (int): device id (0 -> main device/process)
        configs (dict:dict): configuration dictionary (comprising system, model, and train settings)
    """

    t_start = time()

    train_config = configs['train']
    model_config = configs['model']
    system_config = configs['system']

    set_logger(system_config['log_file_path'])

    element_types = model_config['configured_element_types']

    # Setup process group (if DDP is configured)
    if system_config['ddp']:
        print("DDP configured.")
        print(f"Spawning process for gpu: {gpu}")
        print("Initializing process group...")

        system_config['rank'] = gpu
        system_config['train_device'] = torch.device(f"cuda:{gpu}")
        system_config['world_size'] = system_config['num_gpus'] * system_config['num_nodes']
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=system_config['world_size'],
            rank=system_config['rank']
        )

    is_main_process = not (system_config['ddp'] and (gpu != 0))

    # Configure model
    net, optimizer, criterion = configure_model(configs)

    if is_main_process:
        print(net)

    # Configure data loaders
    data_loaders = configure_pipeline(configs)
    for num, loader in data_loaders.items():
        print(f"Dataset size {num}: {len(loader.dataset)}")

    # Configure experiment tracking
    tracker: tracking.ExperimentTracker = None
    if is_main_process:  # Track data only for first GPU/process to prevent repeated logging
        tracker = tracking.ExperimentTracker.from_config(configs)
        tracker.log_config(configs)

    exp_dir_train = Path(system_config['log_dir']) / system_config['experiment'] / 'train'

    # Preparation for epochs
    it_train = 0
    if train_config['init_mode'] == 'pretrained':
        it_train = train_config['start_iteration']

    it_val = 0
    log_val_loss = []
    stop_condition = False

    running_total_loss = 0
    running_loss_dict = {}

    timings = {
        't_load': [],  # Time to wait for dataloader to load batch
        't_trans': [],  # Time to transfer data to GPU
        't_net': [],  # Time to propagate data through model
        't_loss': [],  # Time taken for loss, backprop, and moving data back to CPU
        't_iteration': []  # t_load + t_trans + t_net + t_loss
    }

    print(f"Process {gpu}: Starting to train.")

    # For accurate timings: uncomment torch.cuda.synchronize() before timing measurements
    for i_epoch in range(1, train_config['num_epochs'] + 1):
        # Uses automatic mixed precision to speed up training, if configured
        if system_config['use_amp']:
            amp_context = autocast()
            scaler = torch.cuda.amp.grad_scaler.GradScaler()
        else:
            amp_context = nullcontext()
            scaler = None

        if system_config['ddp']:
            data_loaders['train'].sampler.set_epoch(i_epoch - 1)

        ts_end_it = time()
        for _, example_batched in enumerate(data_loaders['train']):

            # Stop training if a max. number of iterations is configured (not default)
            if train_config['iteration_stop_condition'] and it_train == train_config['num_iterations']:
                stop_condition = True
                print(f"Training done (reached {train_config['num_iterations']} iterations).")
                break

            # torch.cuda.synchronize()
            ts_start_it = time()
            it_train += 1

            example_torch = util.example_convert_to_torch(example_batched, torch.float32, system_config['train_device'])
            target_dict = example_torch['target_dict']

            # torch.cuda.synchronize()
            ts_net_1 = time()

            # Zero the parameter gradients
            optimizer.zero_grad()

            with amp_context:
                # Apply model on input
                out_dict = net(example_torch)

                # torch.cuda.synchronize()
                ts_net_2 = time()

                # Calculate loss and backpropagation if training
                norm_factors = weights.get_loss_norm_by_elements(example_batched['gt_objects'])
                total_loss, loss_dict, focal_loss2d_dict = criterion(out_dict, target_dict, norm_factors)

            if system_config['use_amp']:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            # Detach losses, output, and target
            total_loss, loss_dict, focal_loss2d_dict, out_dict, target_dict = \
                detach_loss_and_output(total_loss, loss_dict, focal_loss2d_dict, out_dict, target_dict)

            # Average loss
            running_total_loss += total_loss
            running_loss_dict = add_to_running_dict(loss_dict, running_loss_dict, element_types)

            # torch.cuda.synchronize()
            ts_per_sample = time()

            # Skip all other GPU processes
            if not is_main_process:
                ts_end_it = time()
                continue

            # Timings
            t_load = ts_start_it - ts_end_it    # beginning of new iteration - end of last iteration
            t_trans = ts_net_1 - ts_start_it
            t_net = ts_net_2 - ts_net_1
            t_loss = ts_per_sample - ts_net_2
            t_iteration = ts_per_sample - ts_end_it     # end of last - to new sample timestamp

            # Log timings
            if it_train > 10:
                timings['t_load'].append(t_load)
                timings['t_trans'].append(t_trans)
                timings['t_net'].append(t_net)
                timings['t_loss'].append(t_loss)
                timings['t_iteration'].append(t_iteration)

            # Log basic progress to console
            if it_train < 100 or it_train % 10 == 0:
                print(f"GPU {gpu}. Train {system_config['experiment']}. epoch: {i_epoch}, iteration : {it_train}, "
                      f"loss: {total_loss:7.3f}, t_iteration: {t_iteration:06.3f} | t_load: {t_load:06.3f}, "
                      f"t_trans: {t_trans:06.3f}, t_net: {t_net:06.3f}, t_loss: {t_loss:06.3f}, "
                      f"BS: {train_config['batch_size']}")

            # Log detailed progress (images, metrics, etc.)
            if it_train % train_config['train_log_interval'] == 0:
                # Convert to numpy for further analysis
                out_dict = convert_to_numpy(out_dict, element_types)
                target_dict = convert_to_numpy(target_dict, element_types)

                # Log images
                for e_type in element_types:
                    fig = vis_eval.vis_prediction_target_loss(
                        out_dict[e_type], target_dict[e_type], focal_loss2d_dict[e_type], show_plot=False)
                    tag = f"visualization/target_output_mask_loss/{e_type}"
                    tracker.log_plot('train', tag, fig, it_train)

                # Metrics computation
                predictions_lists_nms, analysis = get_predictions_and_analysis_from_net_output(configs, example_batched,
                                                                                               out_dict, target_dict)

                # Create plot to integrate into TensorBoard
                for e_type in element_types:
                    fig = vis_eval.vis_nms_evaluation(predictions_lists_nms, analysis, e_type,
                                                      model_config['fm_extent'], show_plot=False)
                    tag = f"visualization/{e_type}"
                    tracker.log_plot('train', tag, fig, it_train)

                t_eval = time() - ts_per_sample

                # Log errors and metrics
                errors, metrics = get_errors_and_metrics_from_analysis(analysis, element_types)
                tracker.log_errors_and_metrics('train', errors, metrics, model_config,
                                               train_config['deviation_detection_task'], it_train)

                log_errors_and_metrics_console(errors, metrics, element_types, t_eval, it_train, i_epoch, cl='All')

                # Log average losses and reset counters
                avg_total_loss = running_total_loss / train_config['train_log_interval']
                running_loss_dict = average_running_dict(running_loss_dict, element_types)
                tracker.log_losses('train', avg_total_loss, running_loss_dict, it_train)
                running_total_loss = 0
                running_loss_dict = {}

            # Test current model on a subset of the validation set
            if (train_config['validation_mode'] in ['interval', 'both']) \
                    and (it_train % train_config['validation_interval'] == 0):
                net.eval()  # Use batch_norm (running mean, var), deactivate dropout
                total_loss_val, it_val = run_validation(net, criterion, data_loaders['val'], configs, tracker, it_val,
                                                        it_train=it_train)
                net.train()  # Set back to training state

                # Check validation log for early stopping and saving to opt (if early stopping is configured)
                log_val_loss.append(total_loss_val)
                stop_condition = check_validation_log(log_val_loss, net, exp_dir_train, train_config)
                if stop_condition:
                    break

                print("*************** Training ***************")

            # Save model to checkpoint
            if train_config['checkpoint_interval'] > 0 and it_train % train_config['checkpoint_interval'] == 0:
                save_model(net, exp_dir_train)

            ts_end_it = time()

        # End of epoch
        # Break if iteration stop condition or early stopping condition (during interval validation) is reached
        if stop_condition:
            break

        # Save to checkpoint after each epoch
        if is_main_process and (i_epoch == train_config['num_epochs']):
            save_model(net, exp_dir_train)

        # Save epoch checkpoints
        if is_main_process and (i_epoch % train_config['epoch_checkpoint_interval'] == 0):
            save_model(net, exp_dir_train, checkpoint_name=f"epoch_{i_epoch}")

    if is_main_process:
        print("Timings:")
        for timing_type in timings:
            timing = round(float(np.median(timings[timing_type])), 4)
            print(f"{timing_type}: {timing}")
            tracker.log_metric('train', timing_type, timing, 0)

        t_total_hours = round((time() - t_start) / 3600.0, 2)
        tracker.log_metric('train', 't_total_hours', t_total_hours, 0)

    print(f"[GPU {gpu}] Finished training.")

    if system_config['ddp']:
        dist.destroy_process_group()


def run_validation(net, criterion, data_loader_val, configs, tracker, it_val, it_train):
    """ Performs a full validation (sub-)set inference and evaluation, according to the configuration. Intended to be
        run as part of the training process.

    Args:
        net (nn.Module): the torch model to validate
        criterion (nn.Module): loss criterion
        data_loader_val (DataLoader): dataloader for validation set
        configs (dict:dict): configuration dictionary (comprising system, model, and train config)
        tracker (tracking.ExperimentTracker): tracker object used to log plots, metrics, losses, etc.
        it_val (int): global counter for validation iterations (for logging purposes)
        it_train (int): current iteration of training process (for logging purposes)

    Returns:
        averaged_total_loss (float): average total loss per validation iteration
        it_val (int): updated validation iteration counter

    """

    print("*************** Validation ***************")

    system_config, model_config, train_config = configs['system'], configs['model'], configs['train']
    element_types = model_config['configured_element_types']
    deviation_detection_task = train_config['deviation_detection_task']

    running_total_loss = 0
    running_loss_dict = {}

    # Init accumulation containers
    if deviation_detection_task:
        dev_classes = map_deviation.get_deviation_classes()
        classes_lookup = {e_type: dev_classes for e_type in element_types}
    else:
        classes_lookup = lookups.get_class_definition_lookup(model_config)
    errors_lookup = lookups.get_error_definition_lookup()
    statistics_def = lookups.get_statistics_definition(model_config['classification_active'])

    stats_acc = lookups.init_nested_dict([element_types, classes_lookup, statistics_def], 0)
    errors_acc = lookups.init_nested_dict([element_types, classes_lookup], {})

    for e_type in element_types:
        for cl in classes_lookup[e_type]:
            for error in errors_lookup[e_type]:
                errors_acc[e_type][cl][error] = []

    # Validate network
    print(f"Net status eval: {not net.training}")
    it = 0  # Validation loop iteration counter
    ts_end_it = time()
    t_last_it = 0
    amp_context = autocast() if system_config['use_amp'] else nullcontext()
    for i_batch, example_batched in enumerate(data_loader_val):
        it += 1
        it_val += 1

        # Transfer to GPU
        torch.cuda.synchronize()
        ts_start = time()

        example_torch = util.example_convert_to_torch(example_batched, torch.float32, system_config['train_device'])

        target_dict = example_torch['target_dict']

        # Apply model on input
        with torch.no_grad():
            with amp_context:
                out_dict = net(example_torch)

        # Calculate validation loss
        norm_factors = weights.get_loss_norm_by_elements(example_batched['gt_objects'])
        total_loss, loss_dict, focal_loss2d_dict = criterion(out_dict, target_dict, norm_factors=norm_factors)

        # Detach
        total_loss, loss_dict, focal_loss2d_dict, out_dict, target_dict = \
            detach_loss_and_output(total_loss, loss_dict, focal_loss2d_dict, out_dict, target_dict)

        running_total_loss += total_loss
        running_loss_dict = add_to_running_dict(loss_dict, running_loss_dict, element_types)

        torch.cuda.synchronize()

        time_per_sample = (time() - ts_start) / train_config['batch_size']
        if it < 10 or it % 10 == 0:
            print(f"Val. iteration: {it}, loss: {total_loss:7.3f}, t_per_sample: {time_per_sample:06.3f}, "
                  f"t_last_iteration: {t_last_it:06.3f}")

        # Convert output to numpy for plotting
        out_dict = convert_to_numpy(out_dict, element_types)
        target_dict = convert_to_numpy(target_dict, element_types)

        # Log images
        if (i_batch + 1) % train_config['validation_log_interval'] == 0 or (i_batch + 1) == 1:
            for e_type in element_types:
                fig = vis_eval.vis_prediction_target_loss(out_dict[e_type], target_dict[e_type],
                                                          focal_loss2d_dict[e_type], show_plot=False)
                tag = f"visualization/target_output_mask_loss/{e_type}"
                tracker.log_plot('val', tag, fig, it_val)

        #############################
        # Metrics computation

        predictions_lists_nms, analysis = get_predictions_and_analysis_from_net_output(configs, example_batched,
                                                                                       out_dict, target_dict)

        # Create plot to integrate into TensorBoard
        if it_val % train_config['validation_log_interval'] == 0:
            for e_type in element_types:
                fig = vis_eval.vis_nms_evaluation(predictions_lists_nms, analysis, e_type, model_config['fm_extent'],
                                                  show_plot=False)
                tag = f"visualization/{e_type}"
                tracker.log_plot('val', tag, fig, it_val)

        # Accumulate metrics and errors
        for e_type in element_types:
            detection_statistics = analysis[e_type]['detection_statistics']
            error_statistics = analysis[e_type]['error_statistics']

            for cl in classes_lookup[e_type]:
                # Add metrics (TP, FP, FN, (N))
                for stat in statistics_def:
                    stats_acc[e_type][cl][stat] += detection_statistics[cl][stat]

                # Add errors
                for error in errors_lookup[e_type]:
                    errors_acc[e_type][cl][error].extend(error_statistics[cl][error])

        # Log sample based metrics and errors to console
        errors, metrics = get_errors_and_metrics_from_analysis(analysis, element_types)
        if it < 10 or it % 10 == 0:
            log_errors_and_metrics_console(errors, metrics, element_types, is_val=True, cl='All')

        #############################
        # Check stop condition
        if (train_config['validation_mode'] == 'interval') and (it >= train_config['num_validation_batches']):
            break

        t_last_it = time() - ts_end_it
        ts_end_it = time()

    # Average loss and log
    averaged_total_loss = running_total_loss / it

    running_loss_dict = average_running_dict(running_loss_dict, element_types)

    tracker.log_losses('val', averaged_total_loss, running_loss_dict, it_train)

    # Compute final validation results based on accumulation
    if deviation_detection_task:
        metrics, errors = metrics_deviation_detection.compute_metrics_from_accumulation_deviation_detection(
            model_config, stats_acc, errors_acc)
    else:
        metrics, errors = metrics_element_detection.compute_metrics_from_accumulation_element_detection(
            model_config, stats_acc, errors_acc)

    # Log metrics and errors
    tracker.log_errors_and_metrics('val', errors, metrics, model_config, deviation_detection_task, it_train)
    print("Val. Accumulated values:")
    log_errors_and_metrics_console(errors, metrics, element_types, t_eval=None, it=None, is_val=True)

    return averaged_total_loss, it_val


# Logging ##############################################################################################################
########################################################################################################################

def log_errors_and_metrics_console(errors, metrics, element_types, t_eval=None, it=None, i_epoch=None, is_val=False,
                                   cl='All'):
    """ This function logs errors and metrics to the output console.

    Args:
        errors (dict): dict containing attributional (regression) errors from analysis for each element type
        metrics (dict): dict containing metrics from analysis for each element type
        element_types (list[str]): list of configured element types (lights, poles, signs)
        t_eval (float | None): time taken for evaluation (in seconds)
        it (int | None): iteration index (1-indexed)
        i_epoch (int | None): epoch index (1-indexed)
        is_val (bool): True for validation set evaluation, False: training set
        cl (str): which class to report from MDD or subclass classification; default: 'All'
    """

    for e_type in element_types:
        errors_str = ""
        metrics_str = ""

        # Compose errors report string
        for error, value in errors[e_type][cl].items():
            # value is None if no GT was available for element type
            if value is not None:
                errors_str += f"{error}: {value:06.3f} | "
            else:
                errors_str += f"{error}: n/a | "

        # Compose metrics report string
        for metric, value in metrics[e_type][cl].items():
            if value is not None:
                metrics_str += f"{metric}: {value:06.3f} | "
            else:
                metrics_str += f"{metric}: n/a | "

        # Compose prefix string containing basic info
        if is_val:
            prefix_str = f"Val. [{e_type}] "
        else:
            prefix_str = f"Train. [{e_type}] "

        if i_epoch and it and t_eval:
            prefix_str += f"epoch: {i_epoch}, iteration : {it}, time_eval: {t_eval:06.3f} | "

        print(prefix_str + metrics_str + errors_str)


# Helper ###############################################################################################################
########################################################################################################################


def detach_loss_and_output(total_loss, loss_dict, focal_loss_2d_dict, out_dict, target_dict):
    """ Detaches loss and output tensors from pytorch graph and moves to CPU for further usage. """
    total_loss = total_loss.detach().cpu().item()
    for e_type, loss in loss_dict.items():
        for k, v in loss.items():
            loss[k] = v.detach().cpu().item()

    for e_type, out in out_dict.items():
        for k, v in out.items():
            out[k] = v.detach().cpu()

    for e_type, target in target_dict.items():
        for k, v in target.items():
            target[k] = v.detach().cpu()

    for e_type, loss in focal_loss_2d_dict.items():
        if loss is not None:
            focal_loss_2d_dict[e_type] = loss.detach().cpu()

    return total_loss, loss_dict, focal_loss_2d_dict, out_dict, target_dict


def save_model(net, run_dir, checkpoint_name='checkpoint'):
    """ Saves model checkpoint <net> in <run_dir> folder as <checkpoint_name>.pt """
    print(f"Saving model to {checkpoint_name}.")
    path = os.path.join(run_dir, checkpoint_name)
    if not os.path.isdir(path):
        Path(path).mkdir(exist_ok=True, parents=True)
    path = os.path.join(path, 'net.pt')
    torch.save(net.state_dict(), path)


def check_validation_log(val_loss_log, net, run_dir, train_config):
    """ Checks validation loss development for early stopping indicator (i.e., loss stops improving for n consecutive
        iterations) and returns bool indicating whether the early stopping condition applies. Also, saves model
        checkpoint as opt.pt if the loss has improved.
    """
    n_no_improvement = 0
    minimum = min(val_loss_log)
    stop_cond = False
    max_failed_improvements = train_config['max_failed_improvements']
    log_size = len(val_loss_log)
    if log_size < max_failed_improvements:
        max_failed_improvements = log_size

    # Check if loss has improved to save model
    if val_loss_log[-1] <= minimum:
        print("Validation: Loss improved.")
        save_model(net, run_dir, checkpoint_name="opt")

    # Check for early stopping condition
    for i in range(1, max_failed_improvements + 1):
        loss = val_loss_log[-i]
        if not loss <= minimum:
            n_no_improvement += 1
        else:
            break  # latest loss is minimum

    if n_no_improvement > 0:
        print(f"Validation: Loss not improved for {n_no_improvement}/{max_failed_improvements} times.")

    if train_config['early_stopping_active'] and (n_no_improvement == max_failed_improvements):
        stop_cond = True

    return stop_cond


def convert_to_numpy(out_dict, element_types):
    """ Returns a numpy-converted version of the net output for further usage. Make sure to detach and move out_dict to
        the CPU before to make out_dict accessible (e.g., using detach_loss_and_output()).
    """
    for e_type in element_types:
        for k, v in out_dict[e_type].items():
            v = np.array(v)
            out_dict[e_type][k] = v

    return out_dict


def extract_first_sample(data_dict, element_types):
    """ Extracts and returns the first sample of a batch-sized data dict (i.e., out_dict or target_dict). """
    for e_type in element_types:
        for k, v in data_dict[e_type].items():
            v = v[0, :, :, :]  # [B, C*L, X, Y]
            data_dict[e_type][k] = np.expand_dims(v, axis=0)

    return data_dict


def add_to_running_dict(values_dict, running_dict, element_types):
    """ Iteratively appends values from values_dict to running_dict accumulator and returns the updated running_dict.

    Args:
        values_dict (dict): dict containing subdicts of singular values for every element type
        running_dict (dict): dict accumulating values of values_dict in corresponding lists; if empty, it will be
                             initialized analogous to values_dict
        element_types (list[str]): list of configured element types (lights, poles, signs)

    """
    for e_type in element_types:
        for key, value in values_dict[e_type].items():
            # Init running loss dict if not existing
            if e_type not in running_dict:
                running_dict[e_type] = {}

            # Skip None values (no GT available)
            if not value:
                continue
            # Add running loss (key == name of loss)
            if key in running_dict[e_type]:
                running_dict[e_type][key].append(value)
            else:
                running_dict[e_type][key] = [value]

    return running_dict


def average_running_dict(running_dict, element_types):
    """ Computes mean of each accumulated list in running_dict in-place and returns the updated running_dict. """
    for e_type in element_types:
        # Average loss
        for key, values in running_dict[e_type].items():
            running_dict[e_type][key] = np.mean(values)

    return running_dict


def get_predictions_and_analysis_from_net_output(configs, example_batched, out_dict, target_dict):
    """ Computes and returns the NMS-filtered predictions (dict:list) and the corresponding analysis dict, containing
        metrics, errors, etc. Note that only the first sample of the given batch is considered.

    Args:
        configs (dict:dict): configuration dictionary (comprising system, model, and train settings)
        example_batched (dict:list): batch provided by dataloader, which may contain multiple samples
        out_dict (dict): net output; note that tensors must be in numpy format (see convert_to_numpy())
        target_dict (dict): training targets; note that tensors must be in numpy format (see convert_to_numpy())

    """
    model_config, train_config = configs['model'], configs['train']
    element_types = model_config['configured_element_types']
    deviation_detection_task = train_config['deviation_detection_task']
    deviation_detection_model = model_config['deviation_detection_model']

    # Use only first sample of batch
    out_dict = extract_first_sample(out_dict, element_types)
    target_dict = extract_first_sample(target_dict, element_types)
    gt_elements = example_batched['gt_objects'][0]

    # Get predictions from grid
    if deviation_detection_model:
        map_lut = example_batched['map_lut'][0]
        predictions_lists_nms = deviation_extraction.lut_based_post_processing(
            out_dict, target_dict, configs, map_lut)
    else:
        predictions_lists = element_extraction.convert_grid_to_element_lists(
            out_dict, target_dict, model_config)
        predictions_lists_nms = element_extraction.nms(predictions_lists, configs)

    # Compute metrics
    if deviation_detection_task:
        if deviation_detection_model:
            deviation_lists = deviation_extraction.create_deviation_predictions_lut(
                predictions_lists_nms, gt_elements, configs)
        else:
            deviation_lists = comparison.compare_predictions_with_map_prior(
                predictions_lists_nms, gt_elements, configs)
        analysis = metrics_deviation_detection.compute_metrics_id_based(
            deviation_lists, gt_elements, configs)
    else:  # object detection
        analysis = metrics_element_detection.compute_metrics_element_detection(
            predictions_lists_nms, gt_elements, model_config, train_config)

    return predictions_lists_nms, analysis


def get_errors_and_metrics_from_analysis(analysis, element_types):
    """ Extracts and returns the errors and metrics from the given analysis dict. """
    errors = {}
    metrics = {}
    for e_type in element_types:
        errors[e_type] = analysis[e_type]['attributional_errors']
        metrics[e_type] = analysis[e_type]['metrics']

    # Errors and metrics: contain class-specific evaluations
    return errors, metrics


# Testing ##############################################################################################################
########################################################################################################################


def test():
    pass


if __name__ == "__main__":
    test()
