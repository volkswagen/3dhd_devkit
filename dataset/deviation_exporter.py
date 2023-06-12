""" Module generating and loading deviation and occlusion json files needed for validation and test inference.
"""

import sys
sys.path.append("..")

import os
import os.path
import copy

from multiprocessing import Pool
from time import time
from tqdm import tqdm

from dataset.helper import SampleProcessor, load_from_json, save_data_to_json
from dataset.map_deviation import DeviationTypes
from dataset.deviation_generator import generate_map_deviations
from dataset_api.map_reader import MapReader
from deep_learning.preprocessing.hdpc_modifier import generate_occlusion_setting
from utility.system_paths import get_system_paths


# Module functions #####################################################################################################
########################################################################################################################


def load_generated_deviations_and_occlusions(samples, partition, save_dir, deviations_setting, occlusion_prob):
    """ Loads generated json files containing deviations and occlusion data.

    Args:
        samples (list[dict]): samples for which to apply deviation and occlusion data
        partition (str): dataset partition to which samples belong (train, val, or test)
        save_dir (str | Path): directory where to find json files
        deviations_setting (str): specifies file which deviation setting to load (e.g., 10-10-5)
        occlusion_prob (float): specifies file name of occlusion data to load
    """
    # List files in dir
    files = os.listdir(save_dir)

    # Load deviations
    deviation_filename = f"{partition}_deviations_{deviations_setting}.json"
    if deviation_filename not in files:
        error_str = f"Could not find pre-saved deviations for {partition} partition at {save_dir}/{deviation_filename}"
        raise FileNotFoundError(error_str)

    samples_deviations = load_from_json(os.path.join(save_dir, deviation_filename))

    print(f"[{partition}] Deviation settings: {samples_deviations['settings']}")
    # Enrich samples with deviation info
    t_start = time()
    for sample in samples:
        matching_samples = [v for k, v in samples_deviations.items() if
                            k == f"{sample['run_name']} {sample['ID']}"]
        if len(matching_samples) == 0:
            error_str = f"No deviation data for sample {(sample['run_name'], sample['ID'])} in {save_dir}/" \
                        f"{deviation_filename}!"
            raise LookupError(error_str)
        sample['deviation_data'] = matching_samples[0]
    print(f"Matched deviations to samples ({time() - t_start:.2f} seconds)")

    # Load occlusions
    if occlusion_prob > 0.0:
        occlusion_filename = f"{partition}_occlusions_{round(occlusion_prob * 100)}.json"
        if occlusion_filename not in files:
            raise FileNotFoundError(
                f"Could not find pre-saved occlusions for {partition} partition at {save_dir}/{occlusion_filename}")

        samples_occlusions = load_from_json(os.path.join(save_dir, occlusion_filename))

        # Enrich samples with occlusion info
        t_start = time()
        for sample in samples:
            matching_samples = [v for k, v in samples_occlusions.items() if
                                k == f"{sample['run_name']} {sample['ID']}"]
            if len(matching_samples) == 0:
                print(f"[Warning] No deviation data for sample {sample['run_name'], sample['ID']}")
            sample['occlusion_data'] = matching_samples[0]
        print(f"Matched occlusions to samples ({time() - t_start:.2f} seconds)")

    return samples


def sample_generate_deviations_and_occlusions_for_export(sample, map_reader, deviation_probs, occlusion_prob):
    """ Generates deviations and occlusions for a single sample.

    Note that deviation and occlusion types are only exported for affected elements.
    I.e., verifications are not exported.

    Args:
        sample (dict): sample (id, lat, lon...) for which to generate deviations
        map_reader (map_reader.MapReader): map reader object providing map elements
        deviation_probs (dict): deviation probabilities for INS, DEL, and SUB
        occlusion_prob (float): occlusion probability

    Returns:
        sample_name (str): unique sample name (run_name and sample id)
        deviation_data (dict): maps element_id to deviation type (INS, DEL, SUB)
        occlusion_data (dict): maps element_id to occlusion type
    """

    global_pose_wgs = (sample['loc_data']['Latitude_deg'],
                       sample['loc_data']['Longitude_deg'],
                       sample['loc_data']['Yaw_deg'])
    map_elements_wgs = copy.deepcopy(map_reader.filter_map_elements_by_location(global_pose_wgs, radius=100))

    # Bollards are generally not used, thus skipped for runtime reasons
    map_elements_wgs = [e for e in map_elements_wgs if not (e['type'] == 'Pole' and e['cls'] == 'bollard')]

    deviations = generate_map_deviations(map_elements_wgs, deviation_probs)
    deviations = generate_occlusion_setting(deviations, occlusion_prob)

    # Only save element id and deviation type
    deviation_data = {}
    occlusion_data = {}
    for deviation in deviations:
        elem_id = int(deviation.get_most_recent()['id'])
        if deviation.deviation_type != DeviationTypes.VERIFICATION:  # do not save verifications (default case)
            deviation_data[elem_id] = deviation.deviation_type.value
        if deviation.occlusion_type is not None:
            occlusion_data[elem_id] = deviation.occlusion_type

    sample_name = f"{sample['run_name']} {sample['ID']}"

    return sample_name, deviation_data, occlusion_data


def get_deviation_setting(setting_name, ins_prob, del_prob, sub_prob):
    """ Creates a setting dict based on arguments (deviation probabilities).

    Args:
        setting_name (str): name of the setting that will be saved in the json file name
        ins_prob (float): probability of an element turned into an insertion (INS)
        del_prob (float): probability of an element turned into a deletion (DEL)
        sub_prob (float): probability of an element turned into a substitution (SUB)

    Returns:
        setting (dict): deviation setting containing name and probabilities.
    """
    deviation_setting_name = setting_name
    deviation_probs = {
        DeviationTypes.INSERTION: ins_prob,
        DeviationTypes.DELETION: del_prob,
        DeviationTypes.SUBSTITUTION: sub_prob
    }

    setting = {
        'deviation_setting_name': deviation_setting_name,
        'deviation_probs': deviation_probs
    }

    return setting


# Script section #######################################################################################################
########################################################################################################################


def run_generate_occlusions(occlusion_probs, partitions, use_multiprocessing=True, num_workers=8, chunk_size=8):
    """ Generates and exports occlusions as json files.

   Args:
        occlusion_probs (list[float]): list of occlusion settings (probability of an element becoming occluded)
        partitions (list[str]): dataset partitions (e.g., 'val', 'test') for which to generate and export occlusions
        use_multiprocessing (bool): true enables multiprocessing (faster)
        num_workers (int): number of workers for multiprocessing
        chunk_size (int): chunk size for multiprocessing
    """

    # Occlusions are generated independently from deviations, hence no deviations needed
    # Set of probabilities to zero as we only need occlusions
    deviation_probs = {
        DeviationTypes.INSERTION: 0,
        DeviationTypes.DELETION: 0,
        DeviationTypes.SUBSTITUTION: 0
    }

    paths = get_system_paths()
    dataset_dir = paths['dataset_dir']

    for occlusion_prob in occlusion_probs:
        for part in partitions:
            # Load partition samples
            print(f"Processing {part} partition ...")
            json_path = os.path.join(dataset_dir, part + '.json')
            samples = load_from_json(json_path)
            print(f"Found {len(samples)} samples.")

            occlusion_export_dict = {}

            # Generate occlusions
            if not use_multiprocessing:
                # use a simple for-loop
                for sample in tqdm(samples):
                    sample_name, _, occlusion_data = sample_generate_deviations_and_occlusions_for_export(
                        sample, dataset_dir, deviation_probs, occlusion_prob)
                    occlusion_export_dict[sample_name] = occlusion_data

            else:
                print(f"Using multiprocessing with {num_workers} workers and chunksize {chunk_size}")

                sample_processor = SampleProcessor(sample_generate_deviations_and_occlusions_for_export,
                                                   dataset_dir=dataset_dir,
                                                   deviation_probs=deviation_probs,
                                                   occlusion_prob=occlusion_prob)
                pool = Pool(num_workers)
                loader = pool.imap_unordered(sample_processor, samples, chunksize=chunk_size)

                # Process samples and show progress bar
                for sample_name, _, occlusion_data in tqdm(loader, total=len(samples)):
                    occlusion_export_dict[sample_name] = occlusion_data
                pool.close()

            # Export as json
            output_filename = f"{part}_occlusions_{round(occlusion_prob * 100)}.json"
            print(f"Saving {output_filename} ...")
            save_data_to_json(occlusion_export_dict, dataset_dir, output_filename)

        print("Finished.")


def run_generate_deviations(deviation_settings, partitions, use_multiprocessing=True, num_workers=8, chunk_size=8):
    """ Generates and exports deviations as json files.

    Args:
        deviation_settings (list[dict]): list of deviation settings obtained by get_deviation_settings(...)
        partitions (list[str]): dataset partitions (e.g., 'val', 'test') for which to generate and export deviations
        use_multiprocessing (bool): true enables multiprocessing (faster)
        num_workers (int): number of workers for multiprocessing
        chunk_size (int): chunk size for multiprocessing
    """
    paths = get_system_paths()
    dataset_dir = paths['dataset_dir']
    map_reader = MapReader(paths['map_base_dir'], ["lights", "poles", "signs"])
    map_reader.read_map_json_files()

    for setting in deviation_settings:
        # Read settings
        deviation_setting_name = setting['deviation_setting_name']
        deviation_probs = setting['deviation_probs']

        # Generate deviations
        for part in partitions:
            # Load partition samples for which to generate deviations
            print(f"Processing {part} partition ...")
            json_path = os.path.join(dataset_dir, part + '.json')
            samples = load_from_json(json_path)
            print(f"Found {len(samples)} samples.")

            deviation_export_dict = {'settings': {k.value: v for k, v in deviation_probs.items()}}

            # Generate deviations
            if not use_multiprocessing:
                # Use a simple for-loop
                for sample in tqdm(samples):
                    sample_name, deviation_data, _ = sample_generate_deviations_and_occlusions_for_export(
                        sample, dataset_dir, deviation_probs, occlusion_prob=0.0)

                    deviation_export_dict[sample_name] = deviation_data

            else:
                print(f"Using multiprocessing with {num_workers} workers and chunksize {chunk_size}")

                sample_processor = SampleProcessor(sample_generate_deviations_and_occlusions_for_export,
                                                   deviation_probs=deviation_probs, occlusion_prob=0.0,
                                                   map_reader=map_reader)
                pool = Pool(num_workers)
                loader = pool.imap_unordered(sample_processor, samples, chunksize=chunk_size)

                # Process samples and show progress bar
                for sample_name, deviation_data, _ in tqdm(loader, total=len(samples)):
                    deviation_export_dict[sample_name] = deviation_data
                pool.close()

            # Export deviations as json files
            output_filename = f"{part}_deviations_{deviation_setting_name}.json"
            print(f"Saving {output_filename} ...")
            save_data_to_json(deviation_export_dict, dataset_dir, output_filename)

    print("Finished.")


# Run section ##########################################################################################################
########################################################################################################################


def main():
    """ Generates and exports map deviations as json files (for validation and test set).

    E.g., element_id 1234 -> DEL etc.
    Also, occlusion settings are defined (element_id -> bottom, right, left, top occlusion).
    E.g., 10-10-5 specifies the deviation occurrence probability for deletions, insertions, substitutions, respectively.

    Settings:
        use_multi_processing (bool): true enables multiprocessing (faster)
        num_workers (int): number of workers for multiprocessing
        chunk_size (int): chunk size for multiprocessing
    """
    ##################
    # Settings
    use_multiprocessing = True
    num_workers = 8
    chunk_size = 8
    ##################

    # Define probability settings
    # Arguments: occurrence probability name, deletion probability, insertion probability, substitution probability
    deviation_settings = [
        get_deviation_setting('10-10-5', 0.1, 0.1, 0.05),
        get_deviation_setting('2-2-1', 0.02, 0.02, 0.01),
        get_deviation_setting('30-30-15', 0.3, 0.3, 0.15),
    ]

    # Define series of json files with different occlusion probability settings
    occlusion_probs = [0.0, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    partitions = ['val', 'test']

    run_generate_deviations(deviation_settings, partitions, use_multiprocessing, num_workers, chunk_size)
    run_generate_occlusions(occlusion_probs, partitions, use_multiprocessing, num_workers, chunk_size)


if __name__ == "__main__":
    main()
