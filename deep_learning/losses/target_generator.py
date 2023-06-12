""" Generates the training targets.
"""

import math as m
import numpy as np

from dataset.map_deviation import MapDeviation, get_deviation_classes
from deep_learning.util import lookups
from utility import transformations


# Class definitions ####################################################################################################
########################################################################################################################


class TargetGenerator:
    """ Generate targets for object and deviation detection tasks.
    """
    def __init__(self, model_config):
        """
        Target generation is based on provided gt_map_objects (ground truth), which are:
        object detection task:
        -> lists of map elements present in the sensor data
        deviation detection task:
        -> lists of deviations (comprising prior (examined) and current elements (in sensor data))

        Args:
            model_config (dict): model configuration
        """

        self.gt_map_objects = None    # in vehicle reference frame (VRF)
        self.deviation_detection_model = model_config['deviation_detection_model']  # MDD-M (stage 3) net
        self.configured_element_types = model_config['configured_element_types']
        self.classification_active = model_config['classification_active']

        self.target_type = model_config['target_type']
        self.target_shape = model_config['target_shape']
        self.fm_extent = model_config['fm_extent']

        # Poles
        self.pole_default_diameter = model_config['pole_default_diameter']
        self.pole_iosa_threshold = model_config['pole_iosa_threshold']

        # Signs
        self.sign_default_height = model_config['sign_default_height']
        self.sign_default_width = model_config['sign_default_width']
        self.sign_default_yaw = model_config['sign_default_yaw']
        self.sign_distance_threshold = model_config['sign_distance_threshold']
        self.sign_z_foreground_threshold = model_config['sign_z_foreground_threshold']
        self.sign_z_background_threshold = model_config['sign_z_background_threshold']

        # Lights
        self.light_default_height = model_config['light_default_height']
        self.light_default_width = model_config['light_default_width']
        self.light_default_yaw = model_config['light_default_yaw']
        self.light_iosa_threshold = model_config['light_iosa_threshold']
        self.light_z_foreground_threshold = model_config['light_z_foreground_threshold']
        self.light_z_background_threshold = model_config['light_z_background_threshold']

        self.z_max = {
            'signs': model_config['sign_z_max'],
            'lights': model_config['light_z_max']
        }
        self.z_min = {
            'signs': model_config['sign_z_min'],
            'lights': model_config['light_z_min']
        }
        self.z_stride = {
            'signs': model_config['sign_z_stride'],
            'lights': model_config['light_z_stride'],
            'poles': model_config['pole_z_stride']
        }

        self.regression_features = {
            'lights': ['x_vrf', 'y_vrf', 'z_vrf', 'width', 'height', 'yaw_sin', 'yaw_cos'],
            'poles': ['x_vrf', 'y_vrf', 'z_vrf', 'diameter'],
            'signs': ['x_vrf', 'y_vrf', 'z_vrf', 'width', 'height', 'yaw_sin', 'yaw_cos']
        }

        # Compute cell sizes and extent mins
        self.cell_sizes = get_cell_sizes(model_config)
        fm_extent = model_config['fm_extent']
        extent_x_min = fm_extent[0][0]
        extent_y_min = fm_extent[1][0]
        self.extent_mins = [extent_x_min, extent_y_min]

        # Get lookup tables mapping class to index if classification is configured
        self._cls_to_idx_luts = None
        if self.classification_active:
            cls_to_idx_luts, _ = lookups.get_class_to_index_lookup(model_config)
            self._cls_to_idx_luts = cls_to_idx_luts

    def set_map_element_types(self, element_types):
        self.configured_element_types = element_types

    def set_map_objects(self, gt_map_objects):
        self.gt_map_objects = gt_map_objects

    def get_targets(self):
        """ Provides the configured target.

        Returns:
            target_dict (dict:dict): contains target tensors (task, regression, anchors, mask) for each element type
            gt_map_objects (dict:list): contains GT elements (object detection) or deviations (deviation detection)
        """
        if self.target_type == "anchors":
            target_dict, gt_map_objects = self._get_targets_anchors()
        else:
            raise ValueError(f"Undefined target_type {self.target_type}!")

        return target_dict, gt_map_objects

    def _get_targets_anchors(self):
        """ Provides the created targets based on anchors.

        task: task target tensor (object detection or deviation detection, scores for focal loss)
        reg: regression target tensor
        anc: anchor grid (required for unnormalization)
        mask: mask tensor (required for loss computation to implement the "don't care" state)

        All tensors are shaped as: [num_classes, num_voxels_z, num_voxels_x, num_voxels_y].

        Returns:
            target_dict (dict:dict): contains target tensors (task, reg, anc, mask) for each element type
            gt_map_objects (dict:list): contains GT elements (object detection) or deviations (deviation detection)

        """
        # Sort map elements into separate lists
        deviation_detection_task = len(self.gt_map_objects) > 0 and isinstance(self.gt_map_objects[0], MapDeviation)
        obj_lists = {}          # objects for targets: deviations (MDD-M (stage 3)) or elements present in sensor data
        gt_map_objects = {}     # sorted by "current" element type

        for e_type in self.configured_element_types:
            # Map type naming conventions
            type_names, _ = lookups.get_element_type_naming_lut()

            # Sort objects based on "current" type (found in sensor)
            # Deviation detection task
            # MDD-M (stage 3)
            if self.deviation_detection_model:
                # Use deviation for target generation (e.g., set target for INS state)
                obj_lists[e_type] = [d for d in self.gt_map_objects if d.type_current == e_type]
                gt_map_objects[e_type] = obj_lists[e_type]

            # MDD-SC (stage 1) or MDD-MC (stage 2)
            elif deviation_detection_task:
                # Use "current" element in sensor data from deviation object for target generation
                obj_lists[e_type] = [
                    d.get_current() for d in self.gt_map_objects if d.get_current() is not None and d.type_current == e_type]

                gt_map_objects[e_type] = [d for d in self.gt_map_objects if d.type_current == e_type]

            # Object detection task
            else:
                # Elements for target generation are only those found in sensor data
                obj_lists[e_type] = [e for e in self.gt_map_objects if e['type'] == type_names[e_type]]
                gt_map_objects[e_type] = obj_lists[e_type]

        # Create anchor grids
        target_dict = {el_type: {} for el_type in self.configured_element_types}

        if 'lights' in self.configured_element_types:
            target_dict['lights'] = self.generate_targets_lights(obj_lists['lights'])

        if 'poles' in self.configured_element_types:
            target_dict['poles'] = self.generate_targets_poles(obj_lists['poles'])

        if 'signs' in self.configured_element_types:
            target_dict['signs'] = self.generate_targets_signs(obj_lists['signs'])

        return target_dict, gt_map_objects

    def get_anchor_grid(self, elem_type, num_layers, use_z_dim=True):
        """ Generates a tensor (for target_dict) with all anchor positions.
        R: regression channel [x, y, z].
        L: number of vertical layers (voxels in z-dimension).

        Args:
            elem_type (str): element type for which to create the anchor grid
            num_layers (int): number of vertical layers in the anchor grid (z-dimension)
            use_z_dim (bool): create the vertical dimension

        Returns:
            anc (tensor): tensor containing anchor positions, rest is zeros [R, L, X, Y]
        """

        # Init
        target_shape = self.target_shape
        cell_sizes = self.cell_sizes
        ndims = 3
        anc = np.zeros((ndims, num_layers, target_shape[0], target_shape[1]))

        # Indices to x,y coordinates in VRF
        indices_x = np.indices(target_shape)[0]  # [X, Y]
        indices_y = np.indices(target_shape)[1]  # [X, Y]
        indices_z = np.indices([num_layers])[0]  # [Z] vector

        # Compute anchor positions from indices
        anchors_x = (cell_sizes[0] / 2) + (indices_x * cell_sizes[0]) + self.extent_mins[0]  # matrix, 0: x, 1: y
        anchors_y = (cell_sizes[1] / 2) + (indices_y * cell_sizes[1]) + self.extent_mins[1]

        # x,y values
        for i_lay in range(0, num_layers):
            anc[0, i_lay, :, :] = anchors_x
            anc[1, i_lay, :, :] = anchors_y

        # z values
        if use_z_dim:
            anchors_z = (self.z_stride[elem_type] / 2) + (indices_z * self.z_stride[elem_type]) + self.z_min[elem_type]
            for ix, iy in np.ndindex(target_shape):
                anc[2, :, ix, iy] = anchors_z

        return anc

    def generate_targets_lights(self, lights):
        """ Generates the light targets (anchor based).

        Args:
            lights (list[MapDeviation] | list[dict]): light deviations or light elements in sensor data

        Returns:
            target_dict_lights (dict:tensor): contains task, regression, mask, and anchor tensors
        """
        # Init
        e_type = 'lights'
        target_shape = self.target_shape
        z_max = self.z_max[e_type]
        z_min = self.z_min[e_type]
        z_stride = self.z_stride[e_type]

        num_classes = 1
        num_layers = int(np.ceil((z_max - z_min) / z_stride))
        num_reg_features = len(self.regression_features[e_type])  # dx, dy, dz, (dh, dw, yaw_sin, yaw_cos)

        if self.classification_active:
            num_classes = len(self._cls_to_idx_luts[e_type].keys())
        dev_classes = get_deviation_classes(include_aggregate_classes=False)
        if self.deviation_detection_model:
            num_classes = len(dev_classes)

        # Tensor shapes: [num_classes, num_voxels_z, num_voxels_x, num_voxels_y]
        task = np.zeros((num_classes, num_layers, target_shape[0], target_shape[1]))
        reg = np.zeros((num_reg_features, num_layers, target_shape[0], target_shape[1]))
        mask = np.ones((1, num_layers, target_shape[0], target_shape[1]))
        anc = self.get_anchor_grid(e_type, num_layers, use_z_dim=True)

        # Start matching elements to anchor grid
        for light in lights:
            i_cls = 0
            if self.deviation_detection_model:
                deviation: MapDeviation = light
                i_cls = dev_classes.index(deviation.deviation_type.value)
                light = deviation.get_most_recent()
            elif self.classification_active:
                cls = light['cls']
                i_cls = self._cls_to_idx_luts[e_type][cls]

            x_vrf = light['x_vrf']
            y_vrf = light['y_vrf']
            z_vrf = light['z_vrf']

            width = light['width']
            height = light['height']
            yaw_vrf = light['yaw_vrf']

            # Create distances maps
            dx_map = x_vrf - anc[0, :, :, :]
            dy_map = y_vrf - anc[1, :, :, :]
            dz_map = z_vrf - anc[2, :, :, :]

            # Get candidates for matching
            # dx_map: [L, X, Y] with L being the number of vertical layers (voxels in z-dimension)
            # Get all anchors within radius of the light for match testing

            thres_dist = max(width, self.cell_sizes[0]) / 2     # distance threshold
            thres_height = height + self.light_default_height   # height threshold
            match_map = np.logical_and(abs(dx_map) < thres_dist, abs(dy_map) < thres_dist, abs(dz_map) < thres_height)
            indices_map = match_map.nonzero()  # get indices where condition is satisfied

            # Find the closest anchor to ensure match despite further conditions
            dmap = np.sqrt(dx_map ** 2 + dy_map ** 2 + dz_map ** 2)
            indices_min = np.unravel_index(dmap.argmin(), dmap.shape)
            il = indices_min[0]  # layer index
            ix = indices_min[1]
            iy = indices_min[2]
            indices = [il, ix, iy, i_cls]

            # Set object data for matched anchors
            object_data = {
                'x_vrf': x_vrf,
                'y_vrf': y_vrf,
                'z_vrf': z_vrf,
                'width': np.log(width / self.light_default_width),
                'height': np.log(height / self.light_default_height),
                'yaw_sin': m.sin(m.radians(yaw_vrf)),
                'yaw_cos': m.cos(m.radians(yaw_vrf))
            }

            task, reg, mask = self.set_element_anchor(
                task, reg, mask, anc, indices, object_data, e_type)

            # Test matching candidates to refine matches
            n_matches = 0
            for il, ix, iy, in zip(indices_map[0], indices_map[1], indices_map[2]):
                anchor_x = anc[0, il, ix, iy]
                anchor_y = anc[1, il, ix, iy]
                anchor_z = anc[2, il, ix, iy]

                # Get iosa for circular overlap with anchors in x/y-dimension
                dist = m.sqrt((anchor_x - x_vrf) ** 2 + (anchor_y - y_vrf) ** 2)
                iosa = iosa_two_circles(dist, width, self.light_default_width)

                z_overlap = line_overlap(p1=z_vrf, len1=height, p2=anchor_z, len2=self.light_default_height)
                indices = [il, ix, iy, i_cls]

                # Match condition, rest remains background
                if iosa >= self.light_iosa_threshold and z_overlap >= self.light_z_foreground_threshold:
                    task, reg, mask = self.set_element_anchor(task, reg, mask, anc, indices, object_data, e_type)
                    n_matches += 1

                # Don't care state with an insufficient z-overlap
                elif iosa > 0 and (self.light_z_background_threshold <= z_overlap):
                    # Set only to don't care if not already matched (the closest condition checked above)
                    already_matched = np.max(task[:, il, ix, iy])
                    if not already_matched:
                        mask[0, il, ix, iy] = 0

        target_dict_lights = {
            'task': task,
            'reg': reg,
            'mask': mask,
            'anc': anc
        }

        return target_dict_lights

    def generate_targets_poles(self, poles):
        """ Generates the pole targets (anchor based).

        Args:
            poles (list[MapDeviation] | list[dict]): pole deviations or pole elements in sensor data

        Returns:
            target_dict_poles (dict:tensor): contains task, regression, mask, and anchor tensors
        """

        # Init tensors
        e_type = 'poles'
        target_shape = self.target_shape
        default_diameter = self.pole_default_diameter  # pole anchor diameter
        num_classes = 1
        if self.classification_active:
            num_classes = len(self._cls_to_idx_luts[e_type].keys())

        dev_classes = get_deviation_classes(include_aggregate_classes=False)
        if self.deviation_detection_model:
            num_classes = len(dev_classes)

        num_reg_features = len(self.regression_features[e_type])  # x, y, z, d
        num_layers = 1  # always 1 since we basically ignore z-dim for poles

        # Tensor shapes: [num_classes, num_voxels_z, num_voxels_x, num_voxels_y]
        task = np.zeros((num_classes, num_layers, target_shape[0], target_shape[1]))
        reg = np.zeros((num_reg_features, num_layers, target_shape[0], target_shape[1]))
        mask = np.ones((1, num_layers, target_shape[0], target_shape[1]))
        anc = self.get_anchor_grid(e_type, num_layers, use_z_dim=False)

        # Match poles into anchor grid
        for pole in poles:
            i_cls = 0
            if self.deviation_detection_model:
                deviation: MapDeviation = pole
                i_cls = dev_classes.index(deviation.deviation_type.value)
                pole = deviation.get_most_recent()
            elif self.classification_active:
                cls = pole['cls']
                i_cls = self._cls_to_idx_luts[e_type][cls]

            # Compute offsets to anchors
            x_vrf = pole['x_vrf']
            y_vrf = pole['y_vrf']
            z_vrf = pole['z_vrf']
            d = pole['diameter']
            dx_map = x_vrf - anc[0, 0, :, :]
            dy_map = y_vrf - anc[1, 0, :, :]

            # Get candidates for IoSA test
            thres_dist = max(d, self.cell_sizes[0])
            match_map = np.logical_and(abs(dx_map) < thres_dist, abs(dy_map) < thres_dist)
            indices_map = match_map.nonzero()  # get indices where condition is satisfied

            # Find closest anchor to ensure match despite IoSA
            dmap = np.sqrt((dx_map ** 2) + (dy_map ** 2))
            indices_min = np.unravel_index(dmap.argmin(), dmap.shape)
            il = 0
            ix = indices_min[0]
            iy = indices_min[1]
            indices = [il, ix, iy, i_cls]

            # Set object data for matching anchors
            object_data = {
                'x_vrf': x_vrf,
                'y_vrf': y_vrf,
                'z_vrf': z_vrf,
                'diameter': np.log(d / self.pole_default_diameter)
            }
            task, reg, mask = self.set_element_anchor(
                task, reg, mask, anc, indices, object_data, e_type)

            # Find the closest anchors by IoSA match
            for ix, iy in zip(indices_map[0], indices_map[1]):
                anchor_x = anc[0, il, ix, iy]
                anchor_y = anc[1, il, ix, iy]

                dist = m.sqrt((anchor_x - x_vrf) ** 2 + (anchor_y - y_vrf) ** 2)
                iosa = iosa_two_circles(dist, d, default_diameter)  # compute IoSA between anchor and pole

                if iosa > self.pole_iosa_threshold:
                    # Matched candidates
                    indices = [il, ix, iy, i_cls]
                    task, reg, mask = self.set_element_anchor(task, reg, mask, anc, indices, object_data, e_type)
                elif iosa > 0:
                    # Ambiguous candidates: "don't care state" with insufficient overlap
                    already_matched = np.max(task[:, il, ix, iy])
                    if not already_matched:
                        mask[0, il, ix, iy] = 0

        target_dict_poles = {
            'task': task,
            'reg': reg,
            'mask': mask,
            'anc': anc
        }

        return target_dict_poles

    def generate_targets_signs(self, signs):
        """ Generates the sign targets (anchor based).

        Args:
            signs (list[MapDeviation] | list[dict]): sign deviations or sign elements in sensor data

        Returns:
            target_dict_signs (dict:tensor): contains task, regression, mask, and anchor tensors
        """

        # Init tensors
        e_type = 'signs'
        target_shape = self.target_shape
        z_max = self.z_max[e_type]
        z_min = self.z_min[e_type]
        z_stride = self.z_stride[e_type]
        dist_threshold = self.sign_distance_threshold  # distance threshold for line segment distance in xy plane

        # if self.classification_active:
        #     num_classes = len(self._cls_to_idx_luts[e_type].keys())
        num_classes = 1
        num_reg_features = len(self.regression_features[e_type])  # dx, dy, dz, (dh, dw, yaw_sin, yaw_cos)
        num_layers = int(np.ceil((z_max - z_min) / z_stride))

        if self.classification_active:
            num_classes = len(self._cls_to_idx_luts[e_type].keys())
        dev_classes = get_deviation_classes(include_aggregate_classes=False)
        if self.deviation_detection_model:
            num_classes = len(dev_classes)

        # Tensor shapes: [num_classes, num_voxels_z, num_voxels_x, num_voxels_y]
        task = np.zeros((num_classes, num_layers, target_shape[0], target_shape[1]))
        reg = np.zeros((num_reg_features, num_layers, target_shape[0], target_shape[1]))
        mask = np.ones((1, num_layers, target_shape[0], target_shape[1]))
        anc = self.get_anchor_grid(e_type, num_layers, use_z_dim=True)

        # Start matching
        for sign in signs:
            i_cls = 0
            if self.deviation_detection_model:
                deviation: MapDeviation = sign
                i_cls = dev_classes.index(deviation.deviation_type.value)
                sign = deviation.get_most_recent()
            elif self.classification_active:
                cls = sign['shape']
                i_cls = self._cls_to_idx_luts[e_type][cls]

            x_vrf = sign['x_vrf']
            y_vrf = sign['y_vrf']
            z_vrf = sign['z_vrf']
            width = sign['width']
            height = sign['height']
            yaw_vrf = sign['yaw_vrf']

            # Create distances maps
            dx_map = x_vrf - anc[0, :, :, :]
            dy_map = y_vrf - anc[1, :, :, :]
            dz_map = z_vrf - anc[2, :, :, :]

            # Get candidates matching test
            # dx_map: [L, X, Y]
            thres_dist = width / 2  # get all anchors within radius of the sign for match testing
            thres_height = height + self.sign_default_height
            match_map = np.logical_and(abs(dx_map) < thres_dist, abs(dy_map) < thres_dist, abs(dz_map) < thres_height)
            indices_map = match_map.nonzero()  # get indices where condition is satisfied

            # Find the closest anchor to ensure match despite further conditions
            dmap = np.sqrt(dx_map ** 2 + dy_map ** 2 + dz_map ** 2)
            indices_min = np.unravel_index(dmap.argmin(), dmap.shape)
            il = indices_min[0]  # layer index
            ix = indices_min[1]
            iy = indices_min[2]
            indices = [il, ix, iy, i_cls]

            # Set object data for matching anchors
            object_data = {
                'x_vrf': x_vrf,
                'y_vrf': y_vrf,
                'z_vrf': z_vrf,
                'width': np.log(width / self.sign_default_width),
                'height': np.log(height / self.sign_default_height),
                'yaw_sin': m.sin(m.radians(yaw_vrf * 2)),  # map from [-90, 90] to [-180, 180] to get closed space
                'yaw_cos': m.cos(m.radians(yaw_vrf * 2))
            }

            task, reg, mask = self.set_element_anchor(
                task, reg, mask, anc, indices, object_data, e_type)

            # Refine matches
            line_seg = calculate_line_segment(x_vrf, y_vrf, width, yaw_vrf)
            match_ctr = 0
            for il, ix, iy, in zip(indices_map[0], indices_map[1], indices_map[2]):
                anchor_x = anc[0, il, ix, iy]
                anchor_y = anc[1, il, ix, iy]
                anchor_z = anc[2, il, ix, iy]

                p = np.array([anchor_x, anchor_y])
                dist = point_to_line_dist(p, line_seg)
                z_over = line_overlap(p1=z_vrf, len1=height, p2=anchor_z, len2=self.sign_default_height)
                indices = [il, ix, iy, i_cls]

                # Match condition, rest remains background
                if dist <= dist_threshold and z_over >= self.sign_z_foreground_threshold:
                    task, reg, mask = self.set_element_anchor(task, reg, mask, anc, indices, object_data, e_type)
                    match_ctr += 1
                # Don't care
                elif dist <= dist_threshold \
                        and (self.sign_z_background_threshold <= z_over <= self.sign_z_foreground_threshold):
                    # Set only to don't care if not already matched (the closest condition checked above)
                    already_matched = np.max(task[:, il, ix, iy])
                    if not already_matched:
                        mask[0, il, ix, iy] = 0

        target_dict_signs = {
            'task': task,
            'reg': reg,
            'mask': mask,
            'anc': anc
        }

        return target_dict_signs

    def set_element_anchor(self, task, reg, mask, anc, indices, object_data, element_type):
        """ Sets the target data for matching anchors.

        Args:
            task (tensor): task tensor
            reg (tensor): regression tensor
            mask (tensor): mask tensor ("don't care")
            anc (tensor): anchor tensor (anchor positions)
            indices (list[int]): indices of anchor to set (iz, ix, iy, icls)
            object_data (dict): data of element to set
            element_type (str): type of the current element

        Returns:
            task (tensor): set task tensor
            reg (tensor): set regression tensor
            mask (tensor): set mask tensor ("don't care")
        """
        # Task and mask
        il = indices[0]     # layer l is vertical z-dimension
        ix = indices[1]
        iy = indices[2]
        icls = indices[3]

        anchor_x = anc[0, il, ix, iy]
        anchor_y = anc[1, il, ix, iy]
        anchor_z = anc[2, il, ix, iy]
        task[icls, il, ix, iy] = 1
        mask[0, il, ix, iy] = 1  # is already one...

        # Regression targets
        reg[0, il, ix, iy] = (object_data['x_vrf'] - anchor_x) / self.cell_sizes[0]
        reg[1, il, ix, iy] = (object_data['y_vrf'] - anchor_y) / self.cell_sizes[1]
        reg[2, il, ix, iy] = (object_data['z_vrf'] - anchor_z) / self.z_stride[element_type]

        idx_offset = 3  # x_vrf, y_vrf, z_vrf
        additional_keys = self.regression_features[element_type][idx_offset:]
        for i, key in enumerate(additional_keys):
            reg[idx_offset + i, il, ix, iy] = object_data[key]

        return task, reg, mask

    def get_norm_data(self, e_type):
        """ Providing normalization data.

        Args:
            e_type (str): type of the element for which to get normalization data

        Returns:
            norm_data (dict:float): normalization data
        """
        if e_type == 'poles':
            norm_data = {
                'cell_sizes': self.cell_sizes,
                'pole_default_diameter': self.pole_default_diameter,
                'pole_z_stride': self.z_stride[e_type]
            }
        elif e_type == 'signs':
            norm_data = {
                'cell_sizes': self.cell_sizes,
                'sign_default_height': self.sign_default_height,
                'sign_default_width': self.sign_default_width,
                'sign_default_yaw': self.sign_default_yaw,
                'sign_z_stride': self.z_stride[e_type]
            }
        elif e_type == 'lights':
            norm_data = {
                'cell_sizes': self.cell_sizes,
                'light_default_height': self.light_default_height,
                'light_default_width': self.light_default_width,
                'light_default_yaw': self.light_default_yaw,
                'light_z_stride': self.z_stride[e_type]
            }
        else:
            raise TypeError(f"Function not defined for element_type '{e_type}'")
        return norm_data


# Module functions #####################################################################################################
########################################################################################################################

def get_cell_sizes(model_config):
    """ Computes the cell sizes in the x-y-plane.

    Args:
        model_config (dict): model configuration

    Returns:
        cell_sizes (list[float]): cell size in x- and in y-dimension
    """
    fm_extent = model_config['fm_extent']
    target_shape = model_config['target_shape']

    extent_x_min = fm_extent[0][0]
    extent_x_max = fm_extent[0][1]
    extent_y_min = fm_extent[1][0]
    extent_y_max = fm_extent[1][1]

    cell_size_x = (extent_x_max - extent_x_min) / target_shape[0]
    cell_size_y = (extent_y_max - extent_y_min) / target_shape[1]
    cell_sizes = [cell_size_x, cell_size_y]

    return cell_sizes


def get_norm_data(model_config, e_type):
    """ Provides normalization data.

    Args:
        model_config (dict): model configuration
        e_type (str): element type

    Returns:

    """
    cell_sizes = get_cell_sizes(model_config)
    if e_type == 'poles':
        norm_data = {
            'cell_sizes': cell_sizes,
            'pole_default_diameter': model_config['pole_default_diameter'],
            'pole_z_stride': model_config['pole_z_stride']
        }
    elif e_type == 'signs':
        norm_data = {
            'cell_sizes': cell_sizes,
            'sign_default_height': model_config['sign_default_height'],
            'sign_default_width': model_config['sign_default_width'],
            'sign_default_yaw': model_config['sign_default_yaw'],
            'sign_z_stride': model_config['sign_z_stride']
        }
    elif e_type == 'lights':
        norm_data = {
            'cell_sizes': cell_sizes,
            'light_default_height': model_config['light_default_height'],
            'light_default_width': model_config['light_default_width'],
            'light_default_yaw': model_config['light_default_yaw'],
            'light_z_stride': model_config['light_z_stride']
        }
    else:
        raise TypeError(f"Function not defined for element_type '{e_type}'")
    return norm_data


def unnormalize_element_data(e_type, element_data, norm_data):
    """ Unnormalizes an element (e.g., prediction)

    Args:
        e_type (str): element type
        element_data (dict): predicted or GT element
        norm_data (dict): data for unnormalization

    Returns:

    """
    if e_type == 'lights':
        return unnormalize_light_data(element_data, norm_data)
    elif e_type == 'poles':
        return unnormalize_pole_data(element_data, norm_data)
    elif e_type == 'signs':
        return unnormalize_sign_data(element_data, norm_data)
    else:
        raise ValueError(f"Not defined for element_type '{e_type}'")


def unnormalize_pole_data(pole_data, norm_data):
    """ Reverts the normalization for poles.
    """

    pole_data['x_vrf'] = (pole_data['x_vrf'] * norm_data['cell_sizes'][0]) + norm_data['anchor_x']
    pole_data['y_vrf'] = (pole_data['y_vrf'] * norm_data['cell_sizes'][1]) + norm_data['anchor_y']
    pole_data['z_vrf'] = (pole_data['z_vrf'] * norm_data['pole_z_stride']) + norm_data['anchor_z']

    pole_data['diameter'] = np.exp(pole_data['diameter']) * norm_data['pole_default_diameter']

    return pole_data


def unnormalize_sign_data(sign_data, norm_data):
    """ Reverts normalization for signs.
    """

    sign_data['x_vrf'] = (sign_data['x_vrf'] * norm_data['cell_sizes'][0]) + norm_data['anchor_x']
    sign_data['y_vrf'] = (sign_data['y_vrf'] * norm_data['cell_sizes'][1]) + norm_data['anchor_y']
    sign_data['z_vrf'] = (sign_data['z_vrf'] * norm_data['sign_z_stride']) + norm_data['anchor_z']

    sign_data['width'] = np.exp(sign_data['width']) * norm_data['sign_default_width']
    sign_data['height'] = np.exp(sign_data['height']) * norm_data['sign_default_height']

    # reconstruct angle from sin and cos
    sign_data['yaw_vrf'] = m.degrees(m.atan2(sign_data['yaw_sin'], sign_data['yaw_cos'])) / 2
    sign_data.pop('yaw_sin')
    sign_data.pop('yaw_cos')

    return sign_data


def unnormalize_light_data(light_data, norm_data):
    """ Reverts normalization for lights.
    """

    light_data['x_vrf'] = (light_data['x_vrf'] * norm_data['cell_sizes'][0]) + norm_data['anchor_x']
    light_data['y_vrf'] = (light_data['y_vrf'] * norm_data['cell_sizes'][1]) + norm_data['anchor_y']
    light_data['z_vrf'] = (light_data['z_vrf'] * norm_data['light_z_stride']) + norm_data['anchor_z']

    light_data['width'] = np.exp(light_data['width']) * norm_data['light_default_width']
    light_data['height'] = np.exp(light_data['height']) * norm_data['light_default_height']

    # reconstruct angle from sin and cos
    light_data['yaw_vrf'] = m.degrees(m.atan2(light_data['yaw_sin'], light_data['yaw_cos'])) % 360
    light_data.pop('yaw_sin')
    light_data.pop('yaw_cos')

    return light_data


def line_overlap(p1, len1, p2, len2, norm=True):
    """ Get overlap of two lines.

    Calculates the overlap of two 1D line segments defined by center (p)osition and (len)gth of the two lines.
    If norm is set, we normalize to a value range of 0...1.

    Args:
        p1 (float): center position of first line
        len1 (float): length of first line
        p2 (float): "
        len2 (float):´"
        norm (bool): normalize data to (0...1)

    Returns:
        overlap (float): overlap of both line segments
    """

    min1 = p1 - len1 / 2
    max1 = p1 + len1 / 2
    min2 = p2 - len2 / 2
    max2 = p2 + len2 / 2
    overlap = max(0, min(max1, max2) - max(min1, min2))
    if norm:
        smaller_line = min(len1, len2)
        overlap /= smaller_line

    return overlap


def calculate_line_segment(x_vrf, y_vrf, width, yaw):
    """ Computes a line segment for signs in the x-y-plane.

    Args:
        x_vrf (float): x-position of a sign
        y_vrf (float): y-position
        width (float): sign width
        yaw (float): sign yaw

    Returns:
        line_seg (list[np.array]): [[x,y], [x,y]] line segment as start and end point (2D)
    """
    # Define line segment in xy plane for GT sign
    a = np.array([x_vrf - width / 2, y_vrf, 0])
    b = np.array([x_vrf + width / 2, y_vrf, 0])
    seg_points = np.column_stack((a, b))  # 2 x 3
    centroid = (x_vrf, y_vrf, 0)
    # seg_points: [3, N(points)]
    seg_points = transformations.rotate_points(seg_points, yaw, centroid=centroid)
    seg_points = seg_points[0:2, :]  # take only x and y coordinates
    line_seg = [seg_points[:, 0], seg_points[:, 1]]

    return line_seg


def point_to_line_dist(point, line):
    """ Computes the shortest distance of a point to a line.

    Code is based on:
    https://stackoverflow.com/questions/27161533/find-the-shortest-distance-between-a-point-and-line-segments-not-line

    To calculate the closest distance to a line segment, we first need to check
    if the point projects onto the line segment.  If it does, then we calculate
    the orthogonal distance from the point to the line.
    If the point does not project to the line segment, we calculate the
    distance to both endpoints and take the shortest distance.

    Args:
        point (np.array): [x,y] array describing the point
        line (list[np.array]): [[x,y], [x,y]] line segment as start and end point (2D)

    Returns:
        dist (float): shortest distance of the point (either to line segment or to respective edge points)
    """

    # Unit vector
    unit_line = line[1] - line[0]
    norm_unit_line = unit_line / np.linalg.norm(unit_line)

    # Compute the perpendicular distance to the theoretical infinite line
    segment_dist = (
            np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) /
            np.linalg.norm(unit_line)
    )

    diff = (
            (norm_unit_line[0] * (point[0] - line[0][0])) +
            (norm_unit_line[1] * (point[1] - line[0][1]))
    )

    x_seg = (norm_unit_line[0] * diff) + line[0][0]
    y_seg = (norm_unit_line[1] * diff) + line[0][1]

    endpoint_dist = min(
        np.linalg.norm(line[0] - point),
        np.linalg.norm(line[1] - point)
    )

    # Decide if the intersection point falls on the line segment
    lp1_x = line[0][0]  # line point 1 x
    lp1_y = line[0][1]  # line point 1 y
    lp2_x = line[1][0]  # line point 2 x
    lp2_y = line[1][1]  # line point 2 y
    is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
    is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y

    # Return distance to segment or distance to endpoint
    if is_betw_x and is_betw_y:
        dist = segment_dist
    else:
        dist = endpoint_dist

    return dist


def iosa_two_circles(d, d1, d2):
    """ Computes the area of intersection of two circles (normalized to the smaller circle area).

    Args:
        d (float): distance separating both circle centers
        d1 (float): diameter of circle 1
        d2 (float): diameter of circle 2

    Returns:
        iosa (float): intersection over smaller area
    """

    R = d1 / 2
    r = d2 / 2

    a1 = m.pi * R ** 2
    a2 = m.pi * r ** 2

    if d <= abs(R - r):
        # One circle is entirely enclosed in the other.
        a_intersect = np.pi * min(R, r) ** 2
    elif d >= r + R:
        # The circles don't overlap at all.
        a_intersect = 0
    else:
        r2, R2, d2 = r ** 2, R ** 2, d ** 2
        alpha = np.arccos((d2 + r2 - R2) / (2 * d * r))
        beta = np.arccos((d2 + R2 - r2) / (2 * d * R))

        a_intersect = (r2 * alpha + R2 * beta -
                       0.5 * (r2 * np.sin(2 * alpha) + R2 * np.sin(2 * beta)))

    # Choose the smaller circle for norm
    a = min([a1, a2])
    iosa = a_intersect / a

    return iosa
