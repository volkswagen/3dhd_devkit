""" Filters samples prior to training or inference.
"""
from multiprocessing import Pool

from tqdm import tqdm
import utm

from dataset.helper import SampleProcessor, load_from_json
from deep_learning.preprocessing import map_fmt
from dataset_api.map_reader import MapReader


def filter_sample_by_num_elements(sample, samples_dir, load_lvl_map, fm_extent, pole_classes, element_type,
                                  min_number_of_elements, map_reader=None):
    """ Filters a sample according to the number of elements contained in that sample.
    Works for both load levels ('online' or 'generated'), which 'generated' providing a speed advantage.
    Generated: Sample crops (point cloud and map) are generated prior to training and inference.
    Online: Sample crops are created on-the-fly.

    Args:
        sample (dict): sample for which to test validity
        samples_dir (str|Path): directory of generated dataset
        load_lvl_map (str): online or generated
        fm_extent (list[list]): feature map extent as [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        pole_classes (list[str]): list of desired pole classes for filtering
        element_type (str): element type to use for filtering
        min_number_of_elements (int): number of required elements to consider a sample as valid
        map_reader (map_reader.MapReader): reader object for online load level

    Returns:

    """
    # Get map elements depending on load level
    if load_lvl_map == 'generated':
        filename = f"{sample['run_name']}_{str(sample['ID']).zfill(7)}_elements.json"
        map_elements_file = samples_dir / sample['partition'] / 'MapElements' / filename
        map_elements_wgs = load_from_json(map_elements_file)

    elif load_lvl_map == 'online':
        pose = [sample['loc_data']['Latitude_deg'], sample['loc_data']['Longitude_deg']]
        map_elements_wgs = map_reader.filter_map_elements_by_location(pose, radius=100)
    else:
        raise ValueError("[Error] sample filter: unknown load level specified!")

    # Filter by element type
    if element_type != 'Any':
        map_elements_wgs = [e for e in map_elements_wgs if e['type'] == element_type]

    # Filter out irrelevant pole classes and merge stacked signs for correct element count later
    if element_type in ['Pole', 'Any']:
        map_elements_wgs = [e for e in map_elements_wgs if not (e['type'] == 'Pole' and e['cls'] not in pole_classes)]
        # Sometimes trees exists whereas no poles are present
        map_elements_wgs = [e for e in map_elements_wgs if not (e['type'] == 'Pole' and e['cls'] == 'tree')]
    if element_type in ['TrafficSign', 'Any'] and min_number_of_elements > 1:
        lower_shapes = ['rectangle']
        width_threshold = 0.1
        map_elements_wgs = map_fmt.join_vertically_stacked_signs(map_elements_wgs, lower_shapes, lower_shapes,
                                                                 width_threshold=width_threshold)

    # Transform to vrf and filter by extent
    lat = sample['loc_data']['Latitude_deg']
    lon = sample['loc_data']['Longitude_deg']
    pos_x, pos_y, _, _ = utm.from_latlon(lat, lon)
    position_utm = [pos_x, pos_y, 0]
    orientation_utm = [0, 0, sample['loc_data']['Yaw_deg']]
    map_elements_vrf = map_fmt.transform_elements_wgs_to_vrf(map_elements_wgs, position_utm, orientation_utm)
    map_elements = map_fmt.filter_elements_by_fm_extent(map_elements_vrf, fm_extent)

    # Filter by number of elements
    valid = False
    if len(map_elements) >= min_number_of_elements:
        valid = True
    return sample, valid


def filter_samples_by_num_elements(samples, configs, element_type, min_number_of_elements, num_workers=4,
                                   chunk_size=4, show_progress_bar=True):
    """ Tests a list of elements if filtering conditions apply. Invalid samples are excluded.
    Args:
        samples (list[dict]): list of samples to test for filtering (from train, val, or test.json)
        configs (dict:dict): contains model, train, and system settings
        element_type (str): element type by which to filter
        min_number_of_elements (int): number of required elements to consider a sample as valid
        num_workers (int): number of workers for multiprocessing
        chunk_size (int): chunk size for multiprocessing
        show_progress_bar (bool): enables progress bar

    Returns:
        samples_filtered (list[dict]) filtered list of elements
    """

    model_config, system_config, train_config = configs['model'], configs['system'], configs['train']
    samples_filtered = []
    # configure map reader for online loading
    map_reader = None
    if train_config['load_lvl_map'] == 'generated':
        print("Sample filter: generated dataset configured.")
    elif train_config['load_lvl_map'] == 'online':
        print("Sample filter: online dataset configured.")
        e_types = model_config['configured_element_types']
        map_reader = MapReader(system_config['map_base_dir'], e_types)
        map_reader.read_map_json_files()
    else:
        raise ValueError("[Error] sample filter: unknown load level specified!")

    # Init
    sample_processor = SampleProcessor(filter_sample_by_num_elements,
                                       samples_dir=system_config['samples_dir'],
                                       load_lvl_map=train_config['load_lvl_map'],
                                       fm_extent=model_config['fm_extent'],
                                       pole_classes=model_config['pole_classes'],
                                       element_type=element_type,
                                       min_number_of_elements=min_number_of_elements,
                                       map_reader=map_reader)
    pool = Pool(num_workers)
    loader = pool.imap_unordered(sample_processor, samples, chunksize=chunk_size)

    # Process samples
    if show_progress_bar:
        loader = tqdm(loader, total=len(samples))
    for sample, valid in loader:
        if valid:
            samples_filtered.append(sample)
    pool.close()
    return samples_filtered


def filter_samples_by_num_elements_by_configs(samples, configs):
    """ Filters loaded samples according to configuration.

    Args:
        samples (list[dict]): list of samples to test for filtering (from train, val, or test.json)
        configs (dict:dict): contains model, train, and system settings

    Returns:
        samples (list[dict]) filtered list of elements
    """
    min_elements_per_type = {
        "TrafficLight": configs['train']['lights_per_sample_min'],
        "Pole": configs['train']['poles_per_sample_min'],
        "TrafficSign": configs['train']['signs_per_sample_min'],
        "Any": configs['train']['any_elements_per_sample_min']
    }

    num_workers = 6  # system_config['num_workers']
    chunk_size = 32

    # consecutively filter by each given element type
    for e_type, min_elements in min_elements_per_type.items():
        if min_elements == 0:
            continue
        samples = filter_samples_by_num_elements(samples, configs, e_type, min_elements, num_workers, chunk_size,
                                                 show_progress_bar=False)

    return samples
