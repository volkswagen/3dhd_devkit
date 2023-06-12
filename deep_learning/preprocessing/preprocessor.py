""" Pre-processor preparing map and point cloud as network inputs.
"""

import copy
import utm

from dataset_api import pc_reader
from dataset_api import map_reader
from dataset import helper
from dataset import point_cloud

from deep_learning.util import lookups
from deep_learning.util import point_cloud_ops as pcops
from deep_learning.preprocessing import augmentation as aug
from deep_learning.preprocessing import hdpc_fmt
from deep_learning.preprocessing import map_fmt
import deep_learning.preprocessing.hdpc_modifier as dd_pc
import dataset.deviation_generator as dd
from dataset.map_deviation import DeviationTypes

# Class definitions ####################################################################################################
########################################################################################################################


class Preprocessor:
    """ Loads and preprocesses high density point clouds (HDPCs) and map data. """

    def __init__(self, configs):
        """
        Args:
            configs (dict:dict): configuration dictionary (comprising system, model, and train config)
        """
        system_config, model_config, train_config = configs['system'], configs['model'], configs['train']

        # Public members
        self.hdpc_tiles_dir = system_config['hdpc_tiles_dir']   # directory of HDPC tiles
        self.map_metadata = system_config['map_metadata_dir']   # directory of PC tile definitions
        self.map_base_dir = system_config['map_base_dir']       # directory of map data (json files)
        self.dataset_dir = system_config['dataset_dir']         # directory of dataset files (e.g., train.json)
        self.samples_dir = system_config['samples_dir']         # directory of generated samples

        self.deviation_detection_task = train_config['deviation_detection_task']
        self.fm_extent = model_config['fm_extent']
        self.configured_element_types = model_config['configured_element_types']
        self.pole_classes = model_config['pole_classes']
        self.light_classes = model_config['light_classes']
        self.sign_classes = model_config['sign_classes']

        # Private members
        # Set by configure functions
        self._shape_hdpc = None
        self._shape_map = None
        self._fm_type_hdpc = None
        self._fm_type_map = None

        # Setup data loaders
        self._hdpc_reader = None
        self._map_reader = None

        self._hdpc_pc: point_cloud.PointCloud = None  # loaded hdpc point cloud
        self._map_elements = None  # loaded map elements

        self._hdpc_load_lvl = 'online'
        self._map_load_lvl = 'online'
        self._z_encoding = model_config['z_encoding_hdpc']

        self._hdpc_fm = None    # computed hdpc feature map (voxelized point cloud before encoding)
        self._map_fm = None     # computed map feature map (encoded map)
        self._map_lut = None    # computed map LUT (element ids -> voxels) during map encoding (MDD-M (stage 3))

        # feature map transformers (fmt)
        self._hdpc_fmt: hdpc_fmt.HdpcFMT = None
        self._map_fmt: map_fmt.MapFMT = None

        # HDPC preprocessing settings (subset of model_config)
        hdpc_preproc_keys = ['voxel_size', 'max_num_voxels', 'max_points_per_voxel', 'min_points_per_voxel']
        self._hdpc_preprocessing_settings = {key: model_config[key] for key in hdpc_preproc_keys}

        # Map encoding settings
        self._map_fm_element_types = model_config['map_fm_element_types']
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
        self._map_fm_settings = {key: model_config[key] for key in map_fm_keys}
        self._join_stacked_signs_shapes = model_config['join_stacked_signs_shapes']
        self._join_stacked_signs_width_threshold = model_config['join_stacked_signs_width_threshold']

        # Sample filter
        self._min_elements_per_type = {
            "lights": train_config['lights_per_sample_min'],
            "poles": train_config['poles_per_sample_min'],
            "signs": train_config['signs_per_sample_min']
        }

        # Augmentation
        aug_keys = ['local_augmentation',
                    'global_augmentation',
                    'global_trans_x',
                    'global_trans_y',
                    'global_trans_z',
                    'global_yaw_uni',
                    'global_scaling',
                    'random_flip_prob',
                    'equal_intensity_prob',
                    'association_consider_z',
                    'association_threshold',
                    'group_augmentation',
                    'local_t_center',
                    'local_yaw_uni',
                    'local_scaling',
                    'point_dropout',
                    'point_keep_prob',
                    'noise_per_point',
                    'noise_per_point_std']

        self._augmentation_settings = {}
        for key in aug_keys:
            self._augmentation_settings[key] = train_config[key]

        # Deviation detection settings
        self._deviation_probs = {
            DeviationTypes.INSERTION: train_config['insertion_prob'],
            DeviationTypes.DELETION: train_config['deletion_prob'],
            DeviationTypes.SUBSTITUTION: train_config['substitution_prob']
        }
        self._occlusion_prob = train_config['occlusion_prob']
        self._dd_point_density = train_config['dd_point_density']
        self._deviation_data = None
        self._occlusion_data = None
        self._map_deviations = None

        # Classification settings
        self.classification_active = model_config['classification_active']

    def get_hdpc_pc(self):
        """ Provides the loaded point cloud.
        """
        return self._hdpc_pc

    def get_map_elements(self):
        """ Provides map elements of configured types.
        """
        configured_map_elements = self.filter_elements_by_types(self.configured_element_types)
        return configured_map_elements

    def get_map_deviations(self):
        """ Provides map deviations of configured types.
        """
        configured_map_deviations = [d for d in self._map_deviations if d.type_current in self.configured_element_types]
        return configured_map_deviations

    def configure_hdpc_fm(self, shape, fm_type='voxels', load_lvl='online', pc_tiles=None):
        """ Configures the point cloud voxelization (hdpc feature map)

        Args:
            shape (list[int]): [X, Y, Z] number of voxels in respective dimensions
            fm_type (str): 'voxels' to enable voxelization
            load_lvl (st): 'online' creates pc crops on-the-fly, 'generated' loads samples created in advance
            pc_tiles (None | dict:dict): point cloud tiles, name->point cloud (as dict)
        """
        self._shape_hdpc = shape
        self._fm_type_hdpc = fm_type
        self._hdpc_load_lvl = load_lvl
        self._hdpc_fmt = hdpc_fmt.HdpcFMT()     # HDPC feature map transformer (voxelization)
        self._hdpc_reader = pc_reader.HDPCReader(
            self.hdpc_tiles_dir, map_meta_data_dir=self.map_metadata, pc_tiles=pc_tiles)

    def configure_map_fm(self, shape, fm_type, load_lvl='online'):
        """ Configures the map encoding (map feature map)

        Args:
            shape (list[int]): [X, Y, Z] number of voxels in respective dimensions
            fm_type (str | None): 'voxels_lut' to enable map encoding
            load_lvl: 'online' creates provides on-the-fly, 'generated' loads map elements created in advance
        """
        self._shape_map = shape
        self._fm_type_map = fm_type
        if fm_type is not None and fm_type.lower() == 'none':
            self._fm_type_map = None
        self._map_load_lvl = load_lvl
        self._map_reader = map_reader.MapReader(self.map_base_dir, self.configured_element_types)
        self._map_fmt = map_fmt.MapFMT()    # map feature map transformer (map encoding)
        self._map_fmt.set_shape_and_extent(self._shape_map, self.fm_extent)
        if self._map_load_lvl == 'online':
            self._map_reader.read_map_json_files()  # read all map elements once

    def load(self, sample):
        """ Loads point cloud and map for a sample.
        """
        # Read HDPC if configured
        self.load_hdpc(sample)

        # Read map elements
        self.load_map(sample)

    def load_hdpc(self, sample, vrf=True):
        """ Loads a point cloud according to the configuration (load level).

        Args:
            sample (dict): sample to be loaded (e.g., pose on the map)
            vrf (bool): enables the transformation into the vehicle reference frame (VRF)
        """

        global_pose_wgs = (sample['loc_data']['Latitude_deg'],
                           sample['loc_data']['Longitude_deg'],
                           sample['loc_data']['Yaw_deg'])

        # Create point cloud crop for all point cloud tiles
        if self._hdpc_load_lvl == 'online':
            # Crop larger extent for augmentation, target extent is cropped from that
            pc_crop_extent = [[-40.0, 100.0], [-40.0, 40.0], [-10000.0, 10000.0]]
            # Extent initial crop extent before augmentation; consider all points in vertical dimension (better norm)
            # pc_crop_extent = copy.deepcopy(self.fm_extent)
            # for dim, extend in zip([0, 1, 2], [10, 20, 10000]):
            #     pc_crop_extent[dim] = [pc_crop_extent[dim][0] - extend, pc_crop_extent[dim][1] + extend]
            self._hdpc_pc = self._hdpc_reader.read_point_cloud(global_pose_wgs,
                                                               pc_crop_extent=pc_crop_extent,
                                                               z_encoding=self._z_encoding,
                                                               vrf=vrf)
        # Load a previously generated point cloud crop (faster)
        elif self._hdpc_load_lvl == 'generated':
            # Set up file path
            file_name = f"{sample['run_name']}_{str(sample['ID']).zfill(7)}_hdpc.npz"
            file_path = self.samples_dir / sample['partition'] / 'HDPC' / file_name

            if not file_path.exists():
                help_str = f"File {file_path} does not exist. Make sure to use load_lvl 'online' or generate the " \
                           f"point clouds before using dataset/hdpc_generator.py"
                raise FileNotFoundError(help_str)

            # Use static function to load sample point cloud (generated)
            self._hdpc_pc = pc_reader.read_sample_point_cloud(file_path, z_encoding=self._z_encoding)

        else:
            error_str = f"Unspecified hdpc_load_lvl '{self._hdpc_load_lvl}'! Choose from ['online', 'generated']."
            raise ValueError(error_str)

    def load_map(self, sample):
        """ Loads map elements for a sample.

        Args:
            sample (dict): sample to be loaded (e.g., pose on the map)
        """

        global_pose_wgs = (sample['loc_data']['Latitude_deg'],
                           sample['loc_data']['Longitude_deg'],
                           sample['loc_data']['Yaw_deg'])

        # Load map elements from entire map within region of interest (ROI)
        if self._map_load_lvl == 'online':
            map_elements = copy.deepcopy(self._map_reader.filter_map_elements_by_location(global_pose_wgs, radius=100))

        # Load a previously generated map element list for ROI (also within radius=100)
        elif self._map_load_lvl == 'generated':
            file_name = f"{sample['run_name']}_{str(sample['ID']).zfill(7)}_elements.json"
            file_path = self.samples_dir / sample['partition'] / 'MapElements' / file_name

            if not file_path.exists():
                help_str = f"File {file_path} does not exist. Make sure to use load_lvl 'online' or generate the " \
                           f"map elements before using dataset/map_generator.py"
                raise FileNotFoundError(help_str)

            map_elements = helper.load_from_json(file_path)

        else:
            error_str = f"Unspecified map_load_lvl '{self._map_load_lvl}'! Choose from ['online', 'generated']."
            raise ValueError(error_str)

        # Joint pre-processing steps
        # Keep only signs that are 'fused' (fusing two sides of the same sign modeled with two rectangles)
        map_elements = [e for e in map_elements if 'two_sided' not in e.keys()]

        # Map all available pole classes to those selected in case of element recognition (active classification)
        if self.classification_active:
            pole_class_lut = lookups.get_pole_class_lut()
            for e in map_elements:
                if e['type'] == 'Pole':
                    e['cls'] = pole_class_lut[e['cls']]

        map_elements = [e for e in map_elements if not (e['type'] == 'Pole' and e['cls'] not in self.pole_classes)]
        map_elements = map_fmt.join_vertically_stacked_signs(map_elements, self._join_stacked_signs_shapes,
                                                             self._join_stacked_signs_shapes,
                                                             width_threshold=self._join_stacked_signs_width_threshold)

        # Initial element filtering
        self._map_elements = map_elements
        allowed_element_types = list(set().union(self.configured_element_types, self._map_fm_element_types))
        self._map_elements = self.filter_elements_by_types(allowed_element_types)

        # Deviation and occlusion data may have been loaded when loading samples through data splitter
        self._deviation_data = sample['deviation_data'] if 'deviation_data' in sample else None
        self._occlusion_data = sample['occlusion_data'] if 'occlusion_data' in sample else None

    def transform_elements_to_vrf(self, sample):
        """ Transforms map elements from UTM into the vehicle reference frame (VRF).

        Args:
            sample (dict): contains vehicle pose in WGS84 on the map for transformation of elements into the VRF.
        """
        lat = sample['loc_data']['Latitude_deg']
        lon = sample['loc_data']['Longitude_deg']
        yaw = sample['loc_data']['Yaw_deg']

        pos_utm = utm.from_latlon(lat, lon)
        position_utm = [pos_utm[0], pos_utm[1], 0]  # [x_utm, y_utm, z_utm]
        orientation_utm = [0, 0, yaw]

        # Normalize element height using the point cloud's estimated ground level
        ground_level = self._hdpc_pc.ground_level_estimate
        if ground_level is None:
            # No points available: set z_utm to 0, otherwise: use a single point cloud for normalization
            if len(self._hdpc_pc.points) == 0:
                position_utm[2] = 0
            else:
                position_utm[2] = self._hdpc_pc.points['z'][0] - self._hdpc_pc.points['z_over_ground'][0]
        else:
            position_utm[2] = ground_level

        self._map_elements = map_fmt.transform_elements_wgs_to_vrf(self._map_elements, position_utm, orientation_utm)

    def update_elements_ground_level(self):
        """ Updates z-position (normalization) of map elements using the point cloud's estimated ground level.
        """
        ground_level = self._hdpc_pc.ground_level_estimate
        if ground_level is None:
            print("Warning: No ground level estimate, cannot update map elements!")
            return None

        for elem in self._map_elements:
            elem['z_vrf'] = elem['z_utm'] - ground_level

    def limit_orientation_values(self):
        """ Limits the orientation of signs to a 180° value range.
        """
        self._map_elements = map_fmt.limit_yaw_value(self._map_elements)

    def crop_pc_to_fm_extent(self):
        """ Crops the point cloud to the specified feature map extent.
        """
        if self._hdpc_pc.bbox != self.fm_extent:
            self._hdpc_pc.crop_to_extent(self.fm_extent)

    def filter_elements_by_fm_extent(self):
        """ Filters map elements to be inside the specified feature map extent.
        """
        self._map_elements = map_fmt.filter_elements_by_fm_extent(self._map_elements, self.fm_extent)

    def filter_elements_by_pc_bbox(self):
        """ Filters map elements by the point cloud's bounding box.
        """
        self._map_elements = map_fmt.filter_elements_by_fm_extent(self._map_elements, self._hdpc_pc.bbox)

    def augment_data(self):
        """ Augments both point cloud and map for training.
        """
        hdpc_points = self._hdpc_pc.points
        map_elements = self._map_elements
        if self._fm_type_map is None or 'voxels' not in self._fm_type_map:
            map_elements = self.get_map_elements()  # small speedup by only considering configured element types

        # Group elements into augmentation groups
        groups = aug.get_augmentation_groups(map_elements, self._augmentation_settings)

        # 1) Local: elements and their respective local point clouds
        if self._augmentation_settings['local_augmentation']:
            # Get pc crops per group or element (depending on settings)
            groups = aug.get_pc_crops_per_group(hdpc_points, groups, self._augmentation_settings)
            # Remove points of local groups before augmentation
            hdpc_points = aug.remove_crops_from_point_cloud(hdpc_points, groups, self._augmentation_settings)
            # Augment a group as entity or single elements within a group separately
            groups = aug.augment_groups(groups, self._augmentation_settings)
            # Insert augmented point clouds
            hdpc_points = aug.insert_crops_into_point_cloud(hdpc_points, groups)

        # 2) Global: translation and rotation of point cloud and GT
        hdpc_points, groups = aug.global_augmentation(hdpc_points, groups, self._augmentation_settings)

        # 3) Point dropout: randomly remove a subset of points
        if self._augmentation_settings['point_dropout']:
            hdpc_points = aug.point_dropout(hdpc_points, self._augmentation_settings)

        # 4) Noise per point
        if self._augmentation_settings['noise_per_point']:
            hdpc_points = aug.noise_per_point(hdpc_points, self._augmentation_settings)

        # Set data for further processing
        self._hdpc_pc.points = hdpc_points
        self._map_elements = aug.retrieve_elements_from_groups(groups)

    def create_feature_maps(self):
        """ Transforms point cloud and map into feature map as network inputs.

        Returns:
            self._hdpc_fm (dict:tensor): contains voxels, num_points, and coordinates tensors (see data_loader.py)
            self._map_fm (float tensor): (float tensor): [X, Y, Z, num_map_features] map representation
            self._map_lut (dict:list): list of element IDs with matched voxels as list of tuples (type->id->list of tuples)
        """
        # Transform HCPC into feature map if configured
        if self._fm_type_hdpc is not None:
            self._hdpc_fmt.set_point_cloud_data(self._hdpc_pc)
            # Voxelization
            if self._fm_type_hdpc == 'voxels':
                self._hdpc_fm = self._hdpc_fmt.create_feature_map_voxels(self._shape_hdpc,
                                                                         self.fm_extent,
                                                                         self._hdpc_preprocessing_settings)

        # Transform map elements into map feature map if configured
        if self._fm_type_map is not None:
            # Deviation detection task: deviations comprise prior (examined) and current elements (in sensor data)
            # Use examined set to generate map feature map
            if self.deviation_detection_task:
                fm_map_elements = [d.get_prior() for d in self._map_deviations]
                fm_map_elements = [e for e in fm_map_elements if e is not None]
            else:
                raise ValueError("Map representation is only applied in deviation detection! (fm_type_map is specified, but deviation_detection_task is not active)")

            # Filter map elements
            fm_map_elements = self.filter_elements_by_types(self._map_fm_element_types, fm_map_elements)
            self._map_fmt.set_map_elements(fm_map_elements)
            # Create configured map representation
            if self._fm_type_map == 'voxels_lut':
                self._map_fm, self._map_lut = self._map_fmt.create_feature_map_voxels_lut(
                    self._map_fm_element_types,
                    self._fm_type_map,
                    self._map_fm_settings)

        return self._hdpc_fm, self._map_fm, self._map_lut

    def filter_elements_by_types(self, configured_element_types, map_elements=None) -> list:
        """ Filters provided map elements according to the configured types.

        Args:
            configured_element_types (list[str]): configured element types
            map_elements (list[dict]: list of map elements

        Returns:
            filtered_map_elements (list[dict]): filtered list of map elements
        """

        # Note that the filtered list is returned, not set internally
        if map_elements is None:
            map_elements = self._map_elements

        group_to_type_lut, _ = lookups.get_element_type_naming_lut()  # maps 'signs' -> 'TrafficSign'
        allowed_element_types = [group_to_type_lut[el_type] for el_type in configured_element_types]

        filtered_map_elements = [e for e in map_elements if e['type'] in allowed_element_types]
        return filtered_map_elements

    def filter_elements_without_point_data(self):
        """ Filters elements without point data in proximity (e.g., bad cropping)
        """
        for element in self._map_elements:
            x = element['x_vrf']
            y = element['y_vrf']

            element['valid_flag'] = True
            # Get points within square_size=1 meter around x-y-coordinate of map element
            points = pcops.get_crop_2d(self._hdpc_pc.points, x, y, square_size=1)
            if points['x'].shape[0] == 0:
                element['valid_flag'] = False

        self._map_elements = [e for e in self._map_elements if e['valid_flag']]

    def generate_deviations(self):
        """ Generates map deviations.

        Deviations may be simulated dynamically (e.g., during training as part of the data augmentation
        strategy), or priorly deviation assignment (element id -> deviation state) may be loaded.
        Similar applies for partial occlusions of map elements (left, top, bottom, right).

        Note: Priorly generated map deviations are loaded along with samples using deep_learning.training.data_splitter.

        """
        # Generate deviations (or load pre-generated assignments)
        if self._deviation_data is not None:
            # Generate deviations based on loaded map elements and loaded deviation data
            self._map_deviations = dd.get_map_deviations_from_sample_data(self._map_elements, self._deviation_data)
        else:
            # Dynamically generate map deviations on-the-fly
            self._map_deviations = dd.generate_map_deviations(self._map_elements, self._deviation_probs)

        # Generate occlusions (or load pre-generated setting)
        if self._occlusion_data is not None:
            self._map_deviations = dd_pc.get_occlusions_from_sample_data(self._map_deviations, self._occlusion_data)
        else:
            self._map_deviations = dd_pc.generate_occlusion_setting(self._map_deviations, self._occlusion_prob)

        # Apply deviations and modifications to point cloud
        self._hdpc_pc = dd_pc.apply_point_density(self._hdpc_pc, self._dd_point_density)
        self._hdpc_pc = dd.apply_deviations_to_point_cloud(self._map_deviations, self._hdpc_pc)
        self._hdpc_pc = dd_pc.apply_occlusions(self._map_deviations, self._hdpc_pc)
