""" Loads and preprocesses point cloud and map data.
"""

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from deep_learning.preprocessing.preprocessor import Preprocessor
from deep_learning.losses.target_generator import TargetGenerator
from utility.logger import set_logger


# Class definitions ####################################################################################################
########################################################################################################################


class MddDataset(Dataset):
    """ Dataset class as input to PyTorch dataloaders.
    """

    def __init__(self, samples, preprocessor: Preprocessor, target_generator: TargetGenerator,
                 augment=False, log=None, include_loaded_pc=False, create_feature_maps=True):
        """
        Args:
            samples (list[dict]): list of samples ({train, val, test}.json)
            preprocessor (Preprocessor): preprocessor object
            target_generator (TargetGenerator): target generator object
            augment (bool): enables data augmentation
            log (Path): logging path, logs console output to text file
            include_loaded_pc (bool): include the loaded point cloud into output (unprocessed)
            create_feature_maps (bool): processes point cloud and map into network inputs (feature maps)
        """

        self.samples = samples
        self.preprocessor = preprocessor
        self.augment = augment
        self.target_gen = target_generator
        self.log = log
        self.include_loaded_pc = include_loaded_pc
        self.create_feature_maps = create_feature_maps

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return self.load_sample(sample)

    def load_sample(self, sample):
        """ Loads and pre-processes a sample.

        Keys of process_sample:

        sample_id (int): ID of loaded sample
        run_name (str): trajectory name (e.g. OP_1) of sample
        gt_map_objects (dict:list): contains GT elements (object detection) or deviations (deviation detection)
        voxels (float tensor): [M, max_points, ndim] with [M voxels, max points per voxel (96), number dimensions (4)].
        coordinates (int tensor): [M, 3] with M voxels with x, y, z coordinates
        num_points_per_voxel (int tensor): [M] with M counters of points for each voxel
        map_fm (float tensor): [X, Y, Z, num_map_features] map representation
        map_lut (dict:list): list of element IDs with matched voxels as list of tuples (type->id->list of tuples)
        hdpc_pc (PointCloud): loaded unprocessed point cloud
        target_dict (dict:tensor): list of target tensors (task, reg, anc, mask) per element type

        Args:
            sample (dict): sample to load

        Returns:
            process_sample (dict): loaded and pre-processed sample
        """
        if self.log is not None:
            set_logger(self.log)

        # Load data according to preprocessor configuration
        # PC is already transformed to VRF during initial loading
        self.preprocessor.load(sample)

        # Transform and filter map elements
        self.preprocessor.transform_elements_to_vrf(sample)
        # Use pc_bbox for filtering, smaller target fm extent is cropped after augmentation
        self.preprocessor.filter_elements_by_pc_bbox()

        # Data augmentation: point cloud and GT map elements
        if self.augment:
            self.preprocessor.augment_data()

        # Crop data to configured fm extent after augmentation
        self.preprocessor.crop_pc_to_fm_extent()
        self.preprocessor.filter_elements_by_fm_extent()
        self.preprocessor.update_elements_ground_level()

        # Correct orientation values
        self.preprocessor.limit_orientation_values()

        # Remove elements for which no point data is available (safety measure)
        self.preprocessor.filter_elements_without_point_data()

        process_sample = {'sample_id': sample['ID'],
                          'run_name': sample['run_name']}

        # Get GT map objects: deviations (deviation detection) or map elements in sensor data (object detection)
        # See deep_learning.training.data_splitter for loading of priorly generated map deviations
        if self.preprocessor.deviation_detection_task:
            self.preprocessor.generate_deviations()
            process_sample['gt_objects'] = self.preprocessor.get_map_deviations()
        else:
            process_sample['gt_objects'] = self.preprocessor.get_map_elements()

        # Voxelize point cloud and encode map
        if self.create_feature_maps:
            hdpc_fm, map_fm, map_lut = self.preprocessor.create_feature_maps()
            process_sample['voxels'] = hdpc_fm['voxels']
            process_sample['num_points'] = hdpc_fm['num_points']
            process_sample['coordinates'] = hdpc_fm['coordinates']

            if map_fm is not None:
                process_sample['map_fm'] = map_fm

            if map_lut is not None:
                process_sample['map_lut'] = map_lut

        if self.include_loaded_pc:
            # Use point cloud to dict conversion for including PCs in samples when using PyTorch's dataloader
            # process_sample['hdpc_pc'] = self.preprocessor.get_hdpc_pc().convert_to_dict()
            process_sample['hdpc_pc'] = self.preprocessor.get_hdpc_pc()

        # Get training targets
        if self.target_gen is not None:
            self.target_gen.set_map_objects(process_sample['gt_objects'])
            target_dict, gt_map_objects = self.target_gen.get_targets()  # according to initial configuration
            process_sample['target_dict'] = target_dict
            process_sample['gt_objects'] = gt_map_objects

        return process_sample


def merge_second_batch(batch_list, _unused=False):
    """ Collate function merging a second batch for PyTorch's DataLoader.
    Not all keys in process_sample can be batched automatically.
    Hence, a custom function is required defining data is process_sample is batched.

    Args:
        batch_list (list[dict]): list of batched process_sample

    Returns:
        ret (dict): batched process_sample containing stacked tensors or lists
    """
    # Initialize
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}

    # Build network input tensors
    for key, elems in example_merged.items():
        if key in [
            'voxels',
            'num_points',
        ]:
            ret[key] = np.concatenate(elems, axis=0)

        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)),
                    mode='constant',
                    constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        elif key == 'map_fm':
            ret[key] = np.stack(elems, axis=0)

    # Include GT for evaluation, sample ID, run_name as lists
    gt_list, id_list, run_list, lut_list, pc_list = [], [], [], [], []
    for ib in range(0, len(batch_list)):
        gt_list.append(batch_list[ib]['gt_objects'])
        id_list.append(batch_list[ib]['sample_id'])
        run_list.append(batch_list[ib]['run_name'])

        del batch_list[ib]['gt_objects']
        del batch_list[ib]['voxels']
        del batch_list[ib]['num_points']
        del batch_list[ib]['coordinates']

        if 'map_fm' in batch_list[ib]:
            del batch_list[ib]['map_fm']
        if 'map_lut' in batch_list[ib]:
            lut_list.append(batch_list[ib]['map_lut'])
            del batch_list[ib]['map_lut']
        if 'hdpc_pc' in batch_list[ib]:
            pc_list.append(batch_list[ib]['hdpc_pc'])
            del batch_list[ib]['hdpc_pc']

    ret['gt_objects'] = gt_list
    ret['sample_ids'] = id_list
    ret['run_names'] = run_list
    if lut_list:
        ret['map_lut'] = lut_list
    if pc_list:
        ret['pc_list'] = pc_list

    # Add target dict: Normal batching function
    data_batched = torch.utils.data.dataloader.default_collate(batch_list)
    ret['target_dict'] = data_batched['target_dict']

    return ret


# Module functions #####################################################################################################
########################################################################################################################


# Test section #########################################################################################################
########################################################################################################################


def main():
    pass


if __name__ == "__main__":
    main()
