""" Module to filter and save map elements for each sample """

import sys

sys.path.append("..")

import os
import os.path
from multiprocessing import Pool
from pathlib import Path
from typing import Union

from tqdm import tqdm

from dataset.helper import SampleProcessor, load_from_json, save_data_to_json
from dataset_api.map_reader import MapReader
from utility.system_paths import get_system_paths

# Module functions #####################################################################################################
########################################################################################################################


def generate_map_elements_sample(
        sample: dict, samples_dir: Union[str, Path], radius: float, map_reader: MapReader, skip_existing: bool):
    """ Obtains all map elements around a given sample and saves element list as json

    Returns:
        1 if file was (purposefully) skipped because it already exists, and 0 if the element list was newly generated
        and saved
    """
    filename = sample['run_name'] + "_" + str(sample['ID']).zfill(7) + '_elements.json'
    output_dir = os.path.join(samples_dir, sample['partition'], 'MapElements')
    output_file = os.path.join(output_dir, filename)
    if skip_existing and os.path.exists(output_file):
        return 1
    # filename = str(sample['ID']).zfill(7) + '_elements.json'
    # output_file = os.path.join(dataset_dir, sample['run_name'], 'MapElements_w_cs_obstacles_point', filename)
    # # filename = sample['run_name'] + "_" + str(sample['ID']).zfill(7) + '_elements.json'
    # output_file = os.path.join(samples_dir, sample['partition'], 'MapElements', filename)
    # if skip_existing and os.path.exists(output_file):
    #     return 1

    pose = [sample['loc_data']['Latitude_deg'], sample['loc_data']['Longitude_deg']]
    elements = map_reader.filter_map_elements_by_location(pose, radius=radius)

    # print(f"Saved file {output_file}")
    save_data_to_json(elements, output_dir, filename)

    return 0


# Script section #######################################################################################################
########################################################################################################################


def run_generate_elements_dataset():
    """ Generates map element dataset for the specified partitions and runs, which comprises a list of map elements for
        each sample stored in a json file.

    Settings:
        use_multi_processing (bool): true enables multiprocessing (faster)
        num_workers (int): number of workers for multiprocessing
        chunk_size (int): chunk size for multiprocessing

        partitions (list[str]): list of dataset partitions to process (train, val, test)
        runs (list[str]): list of runs (trajectories) to process
        skip_existing (bool): if true, samples with existing map element json files are skipped
        e_types (list[str]): list of element types to consider (lights, poles, signs)
        radius (float): radius used to obtain map elements around the position of a sample [m]
    """

    ##################
    # Settings
    use_multiprocessing = True
    num_workers = 32
    chunk_size = 16

    partitions = ['train', 'val', 'test']
    skip_existing = False
    e_types = ["lights", "poles", "signs"]

    radius = 100  # meter
    ##################

    paths = get_system_paths()
    samples_dir = paths['samples_dir']
    dataset_dir = paths['dataset_dir']
    map_json_base_dir = paths['map_base_dir']

    # Create target directories
    for partition in partitions:
        elements_dir = samples_dir / partition / 'MapElements'
        elements_dir.mkdir(parents=True, exist_ok=True)

    # Set up MapReader
    samples = []
    map_reader = MapReader(map_json_base_dir, e_types)
    map_reader.read_map_json_files()

    # Load list of samples
    for part in partitions:
        samples_part = load_from_json(os.path.join(dataset_dir, part + '.json'))
        for sample in samples_part:
            sample['partition'] = part
        samples += samples_part
    print(f"Found {len(samples)} samples in partitions {partitions}.")

    if not use_multiprocessing:
        # Use a simple for-loop
        for sample in tqdm(samples):
            generate_map_elements_sample(sample, samples_dir, radius, map_reader, skip_existing)

    else:
        print(f"Using multiprocessing with {num_workers} workers and chunksize {chunk_size}")

        sample_processor = SampleProcessor(generate_map_elements_sample, samples_dir=samples_dir, radius=radius,
                                           map_reader=map_reader, skip_existing=skip_existing)
        pool = Pool(num_workers)
        loader = pool.imap_unordered(sample_processor, samples, chunksize=chunk_size)

        # Process samples and show progress bar
        for _ in tqdm(loader, total=len(samples)):
            pass
        pool.close()

    print("Finished.")

# Run section ##########################################################################################################
########################################################################################################################


def main():
    run_generate_elements_dataset()


if __name__ == "__main__":
    main()
