""" Module to generate sample point clouds """

import sys
sys.path.append("..")

import math
import os
from multiprocessing import Manager
from pathlib import Path
from time import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset.helper import load_from_json, save_data_to_json
from dataset_api.pc_reader import read_binary_hdpc_file
from deep_learning import configurations
from deep_learning.preprocessing.preprocessor import Preprocessor
from deep_learning.util import config_parser
from deep_learning.util.lookups import get_all_run_names
from utility.logger import set_logger

# Class definitions ####################################################################################################
########################################################################################################################


class SampleDataset(Dataset):
    """ Used to iterate over list of samples via torch Dataloader """

    def __init__(self, samples, func, **kwargs):
        """
        Args:
            samples (list[dict]): List of samples to iterate over
            partition (str): train, val, or test partition of samples
            func: Function to call in every iteration (must take a sample as first argument)
            kwargs: all other arguments needed to pass onto the given function
        """
        self.samples = samples
        self.func = func
        self.args = kwargs

    def __len__(self):
        # return number of samples
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[int(idx)]
        return self.func(sample, **self.args)


def save_hdpc(sample: dict, preprocessor: Preprocessor, skip_existing: bool, log_file: Path):
    """ Saves a single sample point cloud (can be used for SampleDataset) """
    set_logger(log_file)

    output_file = os.path.join(
        preprocessor.samples_dir, sample['partition'], "HDPC", sample['run_name'] + "_" + str(sample['ID']).zfill(7) + "_hdpc")

    if skip_existing and os.path.exists(f"{output_file}.npz"):
        try:
            data = np.load(f"{output_file}.npz")  # load existing file to test if it is corrupted
            _ = data[data.files[0]]  # access file content to check for corruption
            print(f"Skipped, file already exists: {output_file}")
            return {'ID': sample['ID'],
                    'run_name': sample['run_name'],
                    'valid': True}
        except (IOError, ValueError, LookupError) as err:
            print(f"[{type(err)}] {err}")
            print(f"Corrupted file: {output_file}.npz")

    # Load data according to preprocessor configuration
    preprocessor.load_hdpc(sample, vrf=True)
    hd_pc = preprocessor.get_hdpc_pc()
    hd_pc = hd_pc.points
    num_points = len(hd_pc)
    valid = True
    if num_points < 100000:
        print(f"Invalid sample found [sample {(sample['run_name'], sample['ID'])}]. Not enough points: {num_points}")
        valid = False

    # Save in int16 format
    dtype = np.dtype([
        ("x", np.int16),
        ("y", np.int16),
        ("z", np.int16),
        ("intensity", np.int16),
        ("is_ground", np.bool_)
    ])
    hd_pc_out = np.zeros(len(hd_pc), dtype=dtype)  # Allocate object

    hd_pc_out['x'] = np.around(hd_pc['x'] * 100)
    hd_pc_out['y'] = np.around(hd_pc['y'] * 100)
    hd_pc_out['z'] = np.around(hd_pc['z'] * 100)
    hd_pc_out['intensity'] = np.around(hd_pc['intensity'] * 100)
    hd_pc_out['is_ground'] = hd_pc['is_ground']

    np.savez_compressed(output_file, hd_pc_out)
    print(f"Saved to file: {output_file}")

    return {'ID': sample['ID'],
            'run_name': sample['run_name'],
            'valid': valid}


# Script section #######################################################################################################
########################################################################################################################


def run_export_pc(partitions=None, runs_to_load=None, skip_existing=False):
    """ Generate and save sample point clouds for the given dataset partitions.

    Args:
        partitions (list[str]): list of partitions to process (train, val, test). Uses all partitions per default.
        runs_to_load (list[str]): list of runs (trajectories) to process. Uses all available runs per default.
        skip_existing: skip pc generation for already existing sample point clouds (default: False).

    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False

    if partitions is None:
        partitions = ['train', 'val', 'test']
    if runs_to_load is None:
        runs_to_load = get_all_run_names()


    # Use config file to take arguments
    default_config_path = os.path.join(configurations.root_dir(), "default_config.ini")

    configs = config_parser.parse_config(default_config_path)
    system_config, model_config, _ = configs['system'], configs['model'], configs['train']

    # Adjust config
    model_config['fm_extent'] = [[-40.0, 100.0], [-40.0, 40.0], [-10000.0, 10000.0]]
    model_config['load_lvl_hdpc'] = 'online'
    model_config['load_lvl_map'] = 'online'

    log_file = system_config['log_dir'] / f"log_export_pc_{'_'.join(partitions)}.txt"
    if log_file.is_file():
        log_file.unlink()  # Remove old log
    set_logger(log_file)

    samples_dir = system_config['samples_dir']
    dataset_dir = system_config['dataset_dir']
    shape_input = model_config['fm_shape_hdpc']
    num_workers = system_config['num_workers']

    print("Prefetching tiles...")
    manager = Manager()
    pc_tiles = manager.dict()
    map_meta_data_path = os.path.join(system_config['map_metadata_dir'], 'HDPC_TileDefinition.json')
    tile_shapes = load_from_json(map_meta_data_path)
    for tile in tile_shapes:
        print(f"Loading point cloud tile: {tile['name']} ")
        path = os.path.join(system_config['hdpc_tiles_dir'], tile['name'] + '.bin')
        pc_tiles[tile['name']] = read_binary_hdpc_file(path, unnorm=False)

    # Configure reader
    pre = Preprocessor(configs)
    pre.configure_hdpc_fm(shape_input, fm_type='voxels', load_lvl='online', pc_tiles=pc_tiles)

    # Create sample directory
    for partition in partitions:
        target_dir = os.path.join(samples_dir, partition, "HDPC")
        Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Process partitions
    for partition in partitions:
        print(f"Processing {partition} partition")
        json_path = os.path.join(dataset_dir, partition + ".json")

        samples = load_from_json(json_path)

        # Filter samples by selected runs
        for sample in samples:
            sample['partition'] = partition

        print(f"Found {len(samples)} samples from {runs_to_load}")

        # Create dataset and iterate over samples (saving sample point clouds each time)
        saver_dataset = SampleDataset(samples, save_hdpc, preprocessor=pre, skip_existing=skip_existing,
                                      log_file=log_file)
        batch_size = 10
        data_loader = DataLoader(saver_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        t_start = time()
        invalid_samples = []
        iterator = tqdm(data_loader, total=math.ceil(len(samples) / batch_size))
        for ret in iterator:
            for ID, run_name, valid in zip(ret['ID'], ret['run_name'], ret['valid']):
                if not valid:
                    invalid_samples += [s for s in samples if s['run_name'] == run_name and s['ID'] == ID]

        print(f"Time taken: {time() - t_start:.2f} seconds")

        # Save list of invalid samples (if any found)
        if len(invalid_samples) > 0:
            print(f"Found {len(invalid_samples)} invalid samples. Saving as {partition}_invalid.json ...")
            path = os.path.join(dataset_dir, f"{partition}_invalid")
            save_data_to_json(invalid_samples, path, f"{partition}_invalid.json")

        print(f'Completed {partition} partition.')


# Run section #########################################################################################################
########################################################################################################################


def main():
    run_export_pc(skip_existing=True)


if __name__ == "__main__":
    main()
