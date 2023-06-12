""" Viewer for visualizing predictions or samples.
"""

import os

import mayavi.mlab as mlab
import numpy as np
import utm

import dataset.deviation_generator as dd
import deep_learning
import deep_learning.preprocessing.hdpc_modifier as dd_pc
import visualization.visualization_anchors as vis_anchors
import visualization.visualization_basics as vis_basics
import visualization.visualization_evaluation as vis_eval
import visualization.visualization_map_fm as vis_map_fm
import visualization.visualization_pp as vis_pp
from dataset import helper, map_deviation
from deep_learning.losses.target_generator import TargetGenerator
from deep_learning.preprocessing.map_fmt import MapFMT
from deep_learning.preprocessing.preprocessor import Preprocessor
from deep_learning.training.dataset_loader import MddDataset
from deep_learning.training.dataset_splitter import MddDatasetSplitter
from deep_learning.util import config_parser, lookups
from utility.system_paths import get_system_paths
from visualization.visualization_elements import vis_map_elements


# Class definitions ####################################################################################################
########################################################################################################################


class SampleViewer:
    """ Visualizes predictions or samples.

    The viewer visualizes obtained network predictions as evaluated network output.
    Alternatively, the viewer only displays samples (point cloud crops with map elements as network input).
    The viewer utilizes the pre-processing chain as used for training and inference.

    To visualize predictions, the execution of run_evaluation.py for an experiment is required.
    """

    def __init__(self,
                 partition,
                 config_file_path,
                 predictions=None,
                 eval_results=None,
                 load_voxelized_pc=False,
                 show_removed_points=False,
                 show_anchors=False,
                 show_map_fm=False,
                 show_map_lut=False,
                 show_augmentation=False,
                 load_lvl='online',
                 deviation_detection_task=None,
                 load_generated_deviations=False,
                 fm_type_map=None,
                 fm_extent=None,
                 color_by_eval_result=True):
        """
        Args:
            partition (str): current dataset partition (train, val, or test)
            config_file_path (str | Path): path to configuration file path.
                Stored config.ini contained in an experiment's log folder, or the default configuration file.
            predictions (dict:list): contains lists of predictions per element type
            eval_results (list:dict): evaluation results as saved by run_evaluation.py
            load_voxelized_pc (bool): whether to voxelize the point cloud during pre-processing
            show_removed_points (bool): displays removed points lost due to voxelization (requires load_voxelized_pc)
            show_anchors (bool): displays the anchor grid
            show_map_fm (bool): displays the map feature map (map representation)
            show_map_lut (bool): displays the map lookup table (element id -> list of voxels)
            show_augmentation (bool): activates augmentation during pre-processing
            load_lvl (str): load level, 'generated' or 'online'
            deviation_detection_task (bool): True: deviation detection task, False: object detection task
            load_generated_deviations (bool): whether to load the fixed assignment of elements to deviation states
            fm_type_map (str): type of map representations (only 'voxels_lut' provided)
            fm_extent (list[list]): feature map extent as [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        """

        # If predictions are provided for visualization, get unchanged config.ini experiment log (no auto settings)
        apply_auto_settings = False if predictions is not None else True
        configs = config_parser.parse_config(config_file_path, apply_auto_settings)

        # Settings according to config, update shapes if fm_extent is changed (view_samples)
        system_config, model_config, train_config = configs['system'], configs['model'], configs['train']
        if fm_extent is not None:
            configs['model']['fm_extent'] = fm_extent
            shapes = config_parser.auto_generate_shapes(configs['model']['fm_extent'], configs['model']['voxel_size'],
                                                        configs['model']['target_shape_factor'])
            configs['model'].update(shapes)

        self.configs = configs
        self.dataset_dir = system_config['dataset_dir']
        self.partition = partition
        self.predictions = predictions
        self.eval_results = eval_results
        self.load_voxelized_pc = load_voxelized_pc
        self.show_removed_points = show_removed_points  # Highlight points that were removed during voxelization
        self.show_anchors = show_anchors
        self.show_map_fm = show_map_fm
        self.show_map_lut = show_map_lut
        self.show_augmentation = show_augmentation
        self.configured_element_types = model_config['configured_element_types']
        self.classification_active = model_config['classification_active']
        self.deviation_detection_task = train_config['deviation_detection_task']
        self.load_generated_deviations = load_generated_deviations
        self.occlusion_prob = train_config['occlusion_prob']
        self.dd_point_density = train_config['dd_point_density']
        self.deviations_setting = train_config['generated_deviations_setting']
        self.color_by_eval_result = color_by_eval_result

        # Manually set deviation detection task if desired (view_samples(...))
        if deviation_detection_task is not None:
            self.deviation_detection_task = deviation_detection_task
            train_config['deviation_detection_task'] = deviation_detection_task

        # Overwrite config file (if map_fm visualization is desired)
        if fm_type_map is not None:
            if fm_type_map.lower() == 'none':
                fm_type_map = None
            model_config['fm_type_map'] = fm_type_map

        # Private members
        self._target_gen: TargetGenerator = TargetGenerator(model_config) if self.show_anchors else None

        self._samples: list = None
        self._load_dataset()        # load samples
        self._hdpc_pc = None
        self._hdpc_voxels = None    # voxelized point cloud
        self._target_dict = None    # loaded target dictionary
        self._gt_objects = None     # ground-truth objects

        fm_shape_hdpc = model_config['fm_shape_hdpc']
        fm_type_hdpc = model_config['fm_type_hdpc']
        fm_shape_map = model_config['fm_shape_map']
        fm_map_type = model_config['fm_type_map']
        self._fm_extent = model_config['fm_extent']

        train_config['load_lvl_hdpc'] = load_lvl
        train_config['load_lvl_map'] = load_lvl
        model_config['z_encoding_hdpc'] = 'global'

        self._map_fm = None
        self._map_fmt = None
        self._map_lut = None

        # Separate map fmt for visualize_predictions if visualization is not generated from generated deviations
        if predictions is not None and show_map_fm:
            self._map_fmt = MapFMT()
            self._map_fmt.set_shape_and_extent(fm_shape_map, self._fm_extent)

        # Configure preprocessor for reading
        self._prepro = Preprocessor(configs)
        self._prepro.configure_hdpc_fm(fm_shape_hdpc, fm_type_hdpc, load_lvl=load_lvl)
        self._prepro.configure_map_fm(fm_shape_map, fm_map_type, load_lvl=load_lvl)
        self._dataset = MddDataset(self._samples, self._prepro, self._target_gen, self.show_augmentation,
                                   include_loaded_pc=True,
                                   create_feature_maps=self.load_voxelized_pc or self.show_map_fm)

    def _load_dataset(self):
        """ Loads samples from train, val, and test.json files.
        """

        # Use MddDatasetSplitter to load list of samples properly
        splitter = MddDatasetSplitter(self.dataset_dir, [self.partition], self.configs['train']['dataset_version'])
        splitter.load_partitions()

        if self.load_generated_deviations and self.deviation_detection_task and self.partition in ['val', 'test']:
            print("Loading previously saved deviations and occlusions ...")
            splitter.load_generated_deviations_and_occlusions(self.deviations_setting, self.occlusion_prob)

        samples = splitter.partition[self.partition]

        # Prevent preprocessor from generating unwanted random deviations by filling samples with empty deviation data
        if self.predictions and self.deviation_detection_task and not self.load_generated_deviations:
            for sample in samples:
                sample['deviation_data'] = {}

        self._samples = samples

    def _find_sample(self, idx=None, sample_id=None, run_name=None):
        """ Finds a sample within the list of samples.

        Either simply the list index is provided or both sample id and trajectory name (run_name) from which
        the sample originates.

        Args:
            idx (int): index of a sample within the total list.
            sample_id (int): id of the sample (only unique within a trajectory)
            run_name (str): name of source trajectory for that specific sample

        Returns:
            sample (dict): found sample
        """

        # Use index
        if idx is not None:
            sample = self._samples[idx]
        # Use id and run name to find sample
        else:
            matched_samples = [s for s in self._samples if s['ID'] == sample_id and s['run_name'] == run_name]
            if len(matched_samples) != 1:
                raise LookupError(f"Found {len(matched_samples)} samples that match ID {sample_id} and run {run_name}!")
            sample = matched_samples[0]
        return sample

    def _load_sample_vrf(self, idx=None, sample_id=None, run_name=None):
        """ Loads a sample in the vehicle reference frame (VRF) including pre-processing.

        A sample is provided either by list index or by sample id and run name (source trajectory).

        Args:
           idx (int): index of a sample within the total list.
           sample_id (int): id of the sample (only unique within a trajectory)
           run_name (str): name of source trajectory for that specific sample
        """
        sample = self._find_sample(idx, sample_id, run_name)
        idx_str = f"{idx} " if idx is not None else ''
        print(f"Loading sample {idx_str}{sample['run_name'], sample['ID']}")

        # Preprocessor configured to load generated or dynamically generated deviations
        processed_sample = self._dataset.load_sample(sample)

        self._gt_objects = processed_sample['gt_objects']
        if isinstance(self._gt_objects, dict):
            self._gt_objects = []
            for _, sublist in processed_sample['gt_objects'].items():
                self._gt_objects += sublist
        self._hdpc_pc = processed_sample['hdpc_pc']
        self._hdpc_voxels = processed_sample['voxels'] if self.load_voxelized_pc else None
        self._target_dict = processed_sample['target_dict'] if self.show_anchors else None

        if 'map_fm' in processed_sample:
            self._map_fm = processed_sample['map_fm']
        if 'map_lut' in processed_sample:
            self._map_lut = processed_sample['map_lut']

    def _load_sample_utm(self, idx=None, sample_id=None, run_name=None):
        """ Loads a sample in UTM reference frame (without pre-processing).

        A sample is provided either by list index or by sample id and run name (source trajectory).

        Args:
           idx (None | int): index of a sample within the total list.
           sample_id (int): id of the sample (only unique within a trajectory)
           run_name (str): name of source trajectory for that specific sample
        """
        # Set load level to 'online' to load unprocessed data in UTM coordinates
        sample = self._find_sample(idx, sample_id, run_name)
        self._prepro._hdpc_load_lvl = 'online'

        # Load HDPC
        self._prepro.load_hdpc(sample, vrf=False)
        hdpc_pc = self._prepro.get_hdpc_pc()

        # Load map elements
        self._prepro.load_map(sample)
        self._hdpc_pc = hdpc_pc
        self._hdpc_voxels = None
        self._gt_objects = self._prepro.get_map_elements()

    def _visualize_pc(self):
        """ Visualizes a loaded and possible pre-processed (voxelized) point cloud.
        """
        # Visualize voxelized point cloud
        if self.load_voxelized_pc:
            vis_pp.vis_pc_from_voxels(self._hdpc_voxels)
            if self.show_removed_points:
                vis_pp.vis_removed_points(self._hdpc_pc, self._hdpc_voxels)
        # Visualize unprocessed point cloud
        else:
            vis_basics.vis_pc(self._hdpc_pc.points, use_z_over_ground=True)

        if self.show_anchors:
            vis_anchors.visualize_anchor_grid_3d(
                self.configured_element_types, self._target_dict, self._target_gen, unnorm=False)

        if self.show_map_fm:
            vis_map_fm.vis_map_fm(self._map_fm, self._fm_extent)

        if self.show_map_lut:
            vis_map_fm.vis_map_lut(self._map_lut, self._map_fm, self._fm_extent)

    def sort_eval_results(self, sorting, element_type=None, metric=None, cls='All'):
        """ Sorts evaluation results by various methods.
        Args:
            sorting (str): specifies the sorting method (options: 'highest' (default), 'lowest', 'random', None)
            element_type (str): element type by which to sort results
            metric (str): metric or stats by which to sort (f1, precision, recall, TP, FP, FN...)
            cls (str): class by which to sort, e.g., 'All' or an evaluation state (VER, DEL, INS, SUB...)
        """
        if element_type is None or element_type not in ['lights', 'poles', 'signs']:
            element_type = self.configured_element_types[0]
        if sorting is None:
            print("No sorting applied.")
            return
        if sorting == 'random':
            print("Sorting eval_results randomly.")
            np.random.seed(42)
            np.random.shuffle(self.eval_results)
            return

        # Sort by metric if provided
        print(f"Sorting eval_results by {sorting} {metric} for {element_type} ({cls}).")
        valid_metrics = lookups.get_metrics_definition(self.classification_active)
        valid_statistics = lookups.get_statistics_definition(self.classification_active)
        if metric in valid_metrics + valid_statistics:
            stat_key = 'metrics' if metric in valid_metrics else 'detection_statistics'
            reverse = True  # true=descending -> best
            if sorting == 'lowest':
                reverse = False

            # Filter out None entries to prevent sorting error
            eval_results = [e for e in self.eval_results if
                            e['analysis'][element_type][stat_key][cls][metric] is not None]
            self.eval_results = sorted(
                eval_results, key=lambda e: e['analysis'][element_type][stat_key][cls][metric], reverse=reverse)

    def filter_eval_results(
            self, element_types=None, metric='f1', cls='All', comparator='>=', threshold=0, predefined_list=None):
        """ Filters evaluation results according to the comparator and a metric.

        Args:
            element_types (None | list[str]): element types to consider
            metric (str): metric (or statistic) by which to filter (e.g., recall, FP...)
            cls (str): class by which to filter (e.g., for deviation detection: VER, INS...)
            comparator (str): specifies the comparison function (options: <, <=, >, >=)
            threshold (float): threshold value used for filtering (w.r.t. comparator)
            predefined_list (None | list[tuple]): specific list of samples to consider
        """

        if element_types is None:
            element_types = self.configured_element_types
        compare = {
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
        }
        assert comparator in compare

        valid_metrics = lookups.get_metrics_definition(self.classification_active)
        valid_statistics = lookups.get_statistics_definition(self.classification_active)
        assert metric in valid_metrics + valid_statistics
        stat_key = 'metrics' if metric in valid_metrics else 'detection_statistics'

        if predefined_list is not None and len(predefined_list) > 0:
            print("Filtering eval_results by predefined list ...")
            # Filter eval results by specified list of samples
            # Each entry of list must be a tuple in the form (<run_name[str]>, <sample_id[int]>)
            eval_results = []
            for sample in predefined_list:
                eval_results += [e for e in self.eval_results if (e['run_name'], e['sample_id']) == sample]
            self.eval_results = eval_results

        print(f"Filtering eval_results: {', '.join(element_types)} {metric} ({cls}) {comparator} {threshold} ...")
        eval_results = []
        len_old = len(self.eval_results)

        for eval_result in self.eval_results:
            keep_sample = True
            for e_type in element_types:
                metrics = eval_result['analysis'][e_type][stat_key][cls]
                value = metrics[metric]
                if value is None or not compare[comparator](value, threshold):
                    keep_sample = False

            if keep_sample:
                eval_results.append(eval_result)

        self.eval_results = eval_results
        print(f"Filtering finished! Keeping {len(eval_results)} of {len_old} samples.")

    def visualize_predictions(self, element_types=None, show_2d=True, show_unfiltered_preds=False,
                              score_threshold=.0, show_gt=False, show_predictions=True):
        """ Main function visualizing predictions (post-processed network output).

        Args:
            element_types (None | list[str]): configured or provided list of element types
            show_2d (bool): enables 2D visualization of predictions
            show_unfiltered_preds (bool): shows unfiltered predictions (before non-maximum-suppression)
            score_threshold (float): threshold above which to show unfilterd predictions
            show_gt (bool): display gt objects (with association state to visualize false negatives (FNs))
            show_predictions (bool): enables prediction visualization (TPs, FPs)
        """
        if element_types is None:
            element_types = self.configured_element_types

        # Print general detection statistics
        print("Total statistics:")
        for el_type in element_types:
            print(f"\t{el_type}:")
            classes = map_deviation.get_deviation_classes(False) if self.deviation_detection_task else ['All']
            for cls in classes:
                print(f"\t\t{cls}:")
                for stat in ['TP', 'FP', 'FN']:
                    count = sum([e['analysis'][el_type]['detection_statistics'][cls][stat] for e in
                                 self.eval_results])
                    print(f"\t\t\t{stat}: {count}")

        # Iterate over samples
        for eval_result in self.eval_results:
            # Print sample info
            print('=' * 50)
            for e_type in element_types:
                results_type = eval_result['analysis'][e_type]

                metric_keys = ['f1', 'recall', 'precision']
                statistics_keys = ['TP', 'FP', 'FN']
                error_keys = ['distance']
                classes_to_show = ['All']
                if self.deviation_detection_task:
                    classes_to_show = ['Deviating', 'VER']

                if results_type['metrics']['All']['f1'] is None:
                    continue

                # Generate info string
                for cls in classes_to_show:
                    info_str = f"[{e_type: <6} {cls + ']': <10} "

                    metrics_cls = results_type['metrics'][cls]
                    for metric in metric_keys:
                        info_str += f"{metric}: "
                        info_str += f"{metrics_cls[metric]:.2f}, " if metrics_cls[metric] is not None else "n/a, "

                    stats_cls = results_type['detection_statistics'][cls]
                    for stat in statistics_keys:
                        info_str += f"{stat}s: "
                        info_str += f"{stats_cls[stat]:2}, " if stats_cls[stat] is not None else "n/a, "

                    errors_cls = results_type['attributional_errors'][cls]
                    for error in error_keys:
                        if error in errors_cls:
                            info_str += f"error_{error}: "
                            info_str += f"{errors_cls[error]:.3f}, " if errors_cls[error] is not None else "n/a, "

                    print(info_str)

            # Get prediction lists for evaluation result, self.predictions is loaded in view_predictions(...)
            prediction_lists = [p for p in self.predictions if p['sample_id'] == eval_result['sample_id']
                                and p['run_name'] == eval_result['run_name']][0]['predictions_lists']

            # 2D visualization
            if show_2d:
                for e_type in element_types:
                    vis_eval.vis_nms_evaluation(
                        prediction_lists, eval_result['analysis'], e_type, self._fm_extent, show_plot=True)

            # 3D visualization
            gt_objects = []
            for e_type in element_types:
                gt_objects += eval_result['analysis'][e_type]['gt_objects']

            self._load_sample_vrf(sample_id=eval_result['sample_id'], run_name=eval_result['run_name'])

            # Reapply deviations if not loaded from stored assignment (load_generated_deviations is False)
            # Deviations are also stored as gt_objects in analysis, provides same result as loading the fixed assignment
            if not self.load_generated_deviations:
                if self.deviation_detection_task:
                    print("Reapplying deviations to point cloud.")
                    self._hdpc_pc = dd_pc.apply_point_density(self._hdpc_pc, self.dd_point_density)
                    self._hdpc_pc = dd.apply_deviations_to_point_cloud(gt_objects, self._hdpc_pc)
                    self._hdpc_pc = dd_pc.apply_occlusions(gt_objects, self._hdpc_pc)

                if self.show_map_fm:
                    map_fm_elements = gt_objects
                    if self.deviation_detection_task:
                        map_fm_elements = [d.get_prior() for d in gt_objects if d.get_prior() is not None]
                    self._map_fmt.set_map_elements(map_fm_elements)
                    map_fm_keys = ['map_fm_element_types',
                                   'pole_iosa_threshold',
                                   'sign_distance_threshold',
                                   'sign_z_foreground_threshold',
                                   'sign_default_height',
                                   'sign_default_width',
                                   'light_default_height',
                                   'light_default_width',
                                   'pole_default_diameter',
                                   'element_size_factor']
                    map_fm_settings = {key: self.configs['model'][key] for key in map_fm_keys}
                    if self.configs['model']['fm_type_map'] == 'voxels_lut':
                        self._map_fm, self._map_lut = self._map_fmt.create_feature_map_voxels_lut(
                                        self.configs['model']['map_fm_element_types'],
                                        self.configs['model']['fm_type_map'],
                                        map_fm_settings)

            # Create point plot
            mlab.figure(bgcolor=(1, 1, 1))
            self._visualize_pc()

            # Deviation detection task: gt_objects contains deviations
            # -> comprising "prior" (examined) elements and "current" elements in sensor data
            # Object detection task: gt_objects contain elements in sensor data
            for e_type in element_types:
                if self.deviation_detection_task:
                    gt_objects_type = [d for d in gt_objects if d.type_current == e_type]
                else:
                    gt_objects_type = eval_result['analysis'][e_type]['gt_objects']

                predictions = eval_result['analysis'][e_type]['predictions']

                if show_unfiltered_preds:
                    if self.deviation_detection_task:
                        raise ValueError("Unfiltered predictions not supported with deviation detection! (set "
                                         "show_unfiltered_preds to False when deviation_detection_task is active)")
                    preds_unfiltered_all = prediction_lists[e_type]
                    preds_unfiltered = [p for p in preds_unfiltered_all if
                                        p['score'] > score_threshold or 'eval_class' in p.keys()]
                    # filter out confirmed predictions (otherwise it will be overdrawn in visualization)
                    pred_ids = [[p['x_vrf'], p['y_vrf']] for p in predictions]
                    preds_unfiltered = [p for p in preds_unfiltered if [p['x_vrf'], p['y_vrf']] not in pred_ids]

                    predictions += preds_unfiltered
                    print(f"Score threshold ({score_threshold}): kept {len(predictions)} of {len(preds_unfiltered_all)}"
                          f" predictions")

                if show_gt:
                    vis_map_elements(
                        gt_objects_type, vrf=True, is_prediction=False, is_evaluation=True, display_class=False)

                if show_predictions:
                    vis_map_elements(predictions, vrf=True, is_prediction=True, is_evaluation=True,
                                     color_by_eval_result=self.color_by_eval_result)

            mlab.show()
        print("Congratulations, you saw all predictions!")

    def visualize_sample_vrf(self, idx=None, sample_id=None, run_name=None, element_types=None):
        """ Loads and visualizes a sample in the vehicle reference frame (VRF).

        A sample is provided either by list index or by sample id and run name (source trajectory).

        Args:
           idx (int): index of a sample within the total list.
           sample_id (int): id of the sample (only unique within a trajectory)
           run_name (str): name of source trajectory for that specific sample
           element_types (list[str]) configured element types
        """
        if element_types is None:
            element_types = self.configured_element_types
        # Map elements are transformed to VRF using prepro
        self._load_sample_vrf(idx=idx, sample_id=sample_id, run_name=run_name)

        mlab.figure(bgcolor=(1, 1, 1))
        self._visualize_pc()

        _, elem_type_lut = lookups.get_element_type_naming_lut()
        # Deviation detection: visualize prior (examined, deviating) set of elements (instead of "current" sensor data)
        if self.deviation_detection_task:
            selected_elements = [e for e in self._gt_objects if e.type_prior in element_types]
        else:
            selected_elements = [e for e in self._gt_objects if elem_type_lut[e['type']] in element_types]

        # Deviation detection: set color_by_dev_state=False to color examined elements in black
        vis_map_elements(selected_elements, vrf=True, display_class=False, color_by_dev_state=True)
        mlab.show()

    def visualize_sample_utm(self, sample_id, run_name, element_types=None):
        """ Loads and visualizes a sample in the UTM reference frame (unprocessed).

        A sample is provided either by list index or by sample id and run name (source trajectory).

        Args:
           sample_id (int): id of the sample (only unique within a trajectory)
           run_name (str): name of source trajectory for that specific sample
           element_types (list[str]) configured element types
        """
        if element_types is None:
            element_types = self.configured_element_types
        self._load_sample_utm(idx=None, sample_id=sample_id, run_name=run_name)

        # Normalize for visualization purposes (Mayavi cannot handle too large value ranges)
        x_mean = np.mean(self._hdpc_pc.points['x'])
        y_mean = np.mean(self._hdpc_pc.points['y'])
        z_mean = np.mean(self._hdpc_pc.points['z'])

        self._hdpc_pc.points['x'] -= x_mean
        self._hdpc_pc.points['y'] -= y_mean
        self._hdpc_pc.points['z'] -= z_mean

        for e in self._gt_objects:
            lat = e['lat']
            lon = e['lon']
            pos_utm = utm.from_latlon(lat, lon)
            x = pos_utm[0]
            y = pos_utm[1]
            e['x_utm'] = x - x_mean
            e['y_utm'] = y - y_mean
            e['z_utm'] -= z_mean

        mlab.figure()
        vis_basics.vis_pc(self._hdpc_pc.points, use_z_over_ground=False)

        # Separate map elements
        _, elem_type_lut = lookups.get_element_type_naming_lut()
        selected_elements = [e for e in self._gt_objects if elem_type_lut[e['type']] in element_types]
        vis_map_elements(selected_elements, vrf=False)

        mlab.show()


# Script section #######################################################################################################
########################################################################################################################

def run_view_predictions():
    """ Configures the viewer to visualize network predictions.

    Visualization of an experiment requires the prior execution of run_inference.py and run_evaluation.py.

    For element recognition, elements are per default colored according to their evaluation result (TP green,
    FP cyan, FN red, and unassociated ground truth (GT) elements in gray.

    For deviation detection, VER, INS, DEL, SUB are depicted in green, red, orange, and blue, respectively.
    Unassociated GT elements occur in gray.
    Here, the evaluation result is indicated by the text labels (green TP, red FN or FP).

    Mind to set sort_class='All', show_map_fm=False and show_map_lut=False
    when visualizing an element recognition experiment.
    """

    # Settings
    log_dir = get_system_paths()['log_dir']     # configured log directory containing experiments
    experiment_name = 'er-lps-3dhdnet-60m'         # dd-s3-60m, name of experiment to visualize
    net_path = 'checkpoint'
    # Sorting
    partition = 'test'              # dataset partition, train, val, or test
    sorting = 'highest'             # method for sorting evaluation results: 'highest', 'lowest', 'random', None
    metric = 'TP'                   # metric by which to sort: precision, recall, f1, TP, FP, FN
    sort_class = 'All'              # selected class for sorting. MDD: VER, DEL, INS, SUB, All, Deviating, With-Prior
    sort_element_type = 'signs'     # 'signs', 'lights', 'poles' (anything else -> first type in config)
    # Filter
    use_filter = False               # enables the filtering of evaluation results
    filter_element_types = None     # element types to keep during filtering (if None, get from config)
    filter_metric = 'f1'            # precision, recall, f1, TP, FP, FN
    filter_cls = 'Deviating'        # class used for filtering, for DD: VER, DEL, INS, SUB, All, Deviating, With-Prior
    comparator = '>='               # comparator for filtering '<', '<=', '>=', '>', '=='
    filter_threshold = 0            # filter threshold below which to suppress objects
    samples = None                  # Set to None to disable sample filter (fixed list of samples)
    # samples = [                   # Uncomment to enable sample filter (fixed list of samples)
    #     ('OP_5_a', 1922),
    #            ]

    # Visualization settings
    load_generated_deviations = False    # False: regenerate deviations from those seen during inference (faster)
    element_types = None                # (list[str]) element types to visualize
    load_voxelized_pc = True            # enables voxelization
    show_removed_points = False         # display points lost during voxelization (only if load_voxelized_pc)
    show_anchors = False                # display the anchor grids
    show_map_fm = False                  # display the generated map representation (only for S2=MDD-MC & S3=MDD-M)
    show_map_lut = False                # display the map lookup table (MDD-M), requires show_map_fm = True
    show_unfiltered_preds = False       # display unfiltered predictions (before NMS, only for element recognition)
    score_threshold = 0.25              # threshold above which to display predictions
    show_2d = False                     # show predictions in 2D
    load_lvl = 'online'                 # load level ('online': without prior sample generation, or 'generated')
    show_augmentation = False           # enables data augmentation during pre-processing
    show_gt = True                      # show GT objects (for unassociated FN elements)
    show_predictions = True             # show predictions
    color_by_eval_result = True        # colors predictions according to their evaluation result (TP, FP, FN)
    ###############################

    print("Viewing predictions.")
    print(f"Visualizing experiment '{experiment_name}'.")
    config_file_path = os.path.join(log_dir, experiment_name, 'config.ini')
    predictions = helper.load_data_from_pickle(
        os.path.join(log_dir, experiment_name, partition, net_path, 'predictions.pickle'))
    eval_results = helper.load_data_from_pickle(
        os.path.join(log_dir, experiment_name, partition, net_path, 'eval_results.pickle'))

    # Setup viewer
    viewer = SampleViewer(partition, config_file_path,
                          predictions,
                          eval_results,
                          load_voxelized_pc=load_voxelized_pc,
                          show_removed_points=show_removed_points,
                          show_anchors=show_anchors,
                          show_map_fm=show_map_fm,
                          show_map_lut=show_map_lut,
                          show_augmentation=show_augmentation,
                          load_lvl=load_lvl,
                          load_generated_deviations=load_generated_deviations,
                          color_by_eval_result=color_by_eval_result)
    # Filter
    if use_filter:
        viewer.filter_eval_results(element_types=filter_element_types,
                                   metric=filter_metric,
                                   cls=filter_cls,
                                   comparator=comparator,
                                   threshold=filter_threshold,
                                   predefined_list=samples)
    # Sort
    viewer.sort_eval_results(sorting, element_type=sort_element_type, metric=metric, cls=sort_class)

    # Visualize
    viewer.visualize_predictions(show_2d=show_2d,
                                 show_unfiltered_preds=show_unfiltered_preds,
                                 score_threshold=score_threshold,
                                 element_types=element_types,
                                 show_gt=show_gt,
                                 show_predictions=show_predictions)


def run_view_samples():
    """ Configures the viewer to visualize network inputs (point cloud and map).

    Visualization is provided for the vehicle reference frame (VRF) including pre-processing
    and in UTM (global coordinates) without pre-processing.
    """
    # Settings
    partition = 'train'  # dataset partition to load (train, val, test.json)
    config_file_name = "default_config.ini"     # name of the config file
    config_file_path = os.path.join(deep_learning.root_dir(), "configurations", config_file_name)

    load_lvl = 'online'                 # load level ('online': without prior sample generation, or 'generated')
    deviation_detection_task = False     # enables deviation task, displays elements in map prior
    load_generated_deviations = True    # True: loads fixed deviation assignments
    show_vrf = True                     # pre-process and visualize sample in VRF
    show_map_fm = False                  # create and visualize map feature map
    show_map_lut = False                # visualize map lookup table (sets of voxels matching to an element)
    show_anchors = False                # display the anchor grids
    show_augmentation = False           # enables data augmentation for visualization

    fm_type_map = 'voxels_lut'      # type of map feature map to create (only 'voxel_lut' provided)
    if not show_map_fm:
        fm_type_map = None

    fm_extent = [[-10, 50.8], [-20, 20], [-2, 7.6]]     # [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    ###############################

    print("Viewing samples.")
    # Init viewer
    viewer = SampleViewer(partition, config_file_path,
                          load_voxelized_pc=True,
                          show_removed_points=False,
                          show_anchors=show_anchors,
                          show_map_fm=show_map_fm,
                          show_map_lut=show_map_lut,
                          show_augmentation=show_augmentation,
                          load_lvl=load_lvl,
                          deviation_detection_task=deviation_detection_task,
                          fm_type_map=fm_type_map,
                          load_generated_deviations=load_generated_deviations,
                          fm_extent=fm_extent)

    # samples = [('OP_5_a', 6047)]    # fixed list of samples to visualize
    samples = []                  # uncomment to simply iterate by list index
    if samples:
        for run, sample_id in samples:
            if show_vrf:
                viewer.visualize_sample_vrf(sample_id=sample_id, run_name=run, element_types=None)
            else:
                viewer.visualize_sample_utm(sample_id=sample_id, run_name=run, element_types=None)
    else:
        for idx in range(0, 50000, 20):
            viewer.visualize_sample_vrf(idx=idx, element_types=None)


# Run section #########################################################################################################
########################################################################################################################


def main():
    run_view_predictions()  # visualize network outputs
    # run_view_samples()    # visualize network inputs


if __name__ == "__main__":
    main()
