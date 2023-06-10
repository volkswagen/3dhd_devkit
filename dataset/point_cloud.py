""" Module containing point cloud class specifications """

from abc import ABC
import numpy as np

# Class definitions ####################################################################################################
########################################################################################################################


class PointCloud(ABC):
    """
    Base Class for Point Cloud data as standard interface
    """
    points = None
    dtype = None
    bbox = None

    def __init__(self, points, vrf=False, z_encoding='global'):
        """ Base class constructor """

        self.vrf = vrf  # indicating if points are in vehicle reference coordinate system
        self.z_encoding = z_encoding  # defines the normalization method regarding the z-dimension

        self.points = np.zeros(len(points), dtype=self.dtype)
        self.points['x'] = points['x']
        self.points['y'] = points['y']
        self.points['z'] = points['z']
        self.points['intensity'] = points['intensity']
        self.points['is_ground'] = points['is_ground']

        self.ground_level_estimate = None
        self.normalize_z_over_ground()

        # Determine actual bounding box
        x_min, x_max = np.min(self.points['x']), np.max(self.points['x'])
        y_min, y_max = np.min(self.points['y']), np.max(self.points['y'])
        z_min, z_max = -10000, 10000  # not limited
        self.bbox = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

        for field in self.dtype.names:
            self._add_property(field)

    def _add_property(self, name):
        """ Add a field from points to class as attribute

        Args:
            name: name of the field
        """

        def fget(self):
            return self.points[name]

        def fset(self, value):
            self.assertWriteMode()
            self._writer.set_dimension(name, value)

        setattr(self.__class__, name, property(fget, fset, None, None))

    def get_xyz(self):
        """ Returns split arrays for x, y, and z """
        return self.points['x'], self.points['y'], self.points['z_over_ground']

    def get_intensity(self):
        """ Returns intensity value array """
        return self.points['intensity']

    def get_point_bounds(self):
        """ Returns point cloud bounds as [[x_min, x_max], [y_min, y_max], [z_min, z_max]] """
        x_minmax = [np.amin(self.points['x']), np.amax(self.points['x'])]
        y_minmax = [np.amin(self.points['y']), np.amax(self.points['y'])]
        z_minmax = [np.amin(self.points['z_over_ground']), np.amax(self.points['z_over_ground'])]
        return [x_minmax, y_minmax, z_minmax]

    def crop_to_extent(self, extent, crop_z=True):
        """ Crop PC to the given extent.

        Args:
            extent (list[list]): extent used to crop the PC as [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            crop_z: if false, PC will only be cropped in x and y dim (in this case, extent does not need [z_min, z_max])
        """
        x_min, x_max = extent[0]
        y_min, y_max = extent[1]

        # Find points inside constrained area
        xy_mask = (self.points['x'] > x_min) * (self.points['x'] < x_max) * \
                  (self.points['y'] > y_min) * (self.points['y'] < y_max)
        self.points = self.points[xy_mask]

        # Filter height
        if crop_z:
            z_min, z_max = extent[2]
            self.normalize_z_over_ground()
            z_mask = (self.points['z_over_ground'] > z_min) * (self.points['z_over_ground'] < z_max)
            self.points = self.points[z_mask]
        else:
            z_min, z_max = -10000, 10000  # i.e., not limited

        # Apply filter to saved points
        self.bbox = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

    def get_ground_level_estimate_global(self):
        """
        Rough estimation of a global round level by averaging all ground points.

        Returns:
             z_average (float): averaged z position of ground points
        """
        if len(self.points) == 0:
            print("Warning: point cloud is empty!")
            return None
        # Compute average
        valid_class = self.points['is_ground']
        if len(self.points['z'][valid_class]) == 0:
            print("Warning: no valid ground points found!")
            return None
        z_average = np.average(self.points['z'][valid_class])

        return z_average

    def normalize_z_over_ground(self):
        """ Normalizes z_over_ground attribute across the point cloud according to the specified z_encoding """
        if len(self.points) == 0:
            print("Warning: point cloud is empty!")
            return

        # Use global normalization by ground classified points
        if self.z_encoding == 'global':
            # Averaged z value of ground points
            self.ground_level_estimate = self.get_ground_level_estimate_global()
            if self.ground_level_estimate is not None:
                self.points['z_over_ground'] = self.points['z'] - self.ground_level_estimate

        else:
            # normal z position
            self.points['z_over_ground'] = self.points['z']

    def convert_to_dict(self):
        pc = {'x': np.array(self.points['x'], dtype=self.points['x'].dtype),
              'y': np.array(self.points['y'], dtype=self.points['y'].dtype),
              'z': np.array(self.points['z'], dtype=self.points['z'].dtype),
              'z_over_ground': np.array(self.points['z'], dtype=self.points['z'].dtype),
              'intensity': np.array(self.points['intensity'], dtype=self.points['intensity'].dtype),
              'is_ground': np.array(self.points['is_ground'], dtype=np.bool_)}

        return pc


class UTMPointCloud(PointCloud):
    """
    A point cloud is implemented as a list of measurements. This class provides methods to filter measurements.
    Uses Float64 to store UTM coordinates.
    """
    dtype = np.dtype([
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("z_over_ground", np.float64),
        ("intensity", np.float32),
        ("is_ground", np.bool_),
    ])

    def __init__(self, points, z_encoding='global'):
        """ Constructor, parsing input into internal point cloud data structure

        Args:
            points: points numpy dict, containing x, y, z, intensity, is_ground, and z_over_ground arrays
            z_encoding (str): method used to normalize the z-dimension to the local ground level
        """
        # Call Baseclass constructor
        super().__init__(points, vrf=False, z_encoding=z_encoding)


class VRFPointCloud(PointCloud):
    """ Subclass of PointCloud intended for VRF coordinates, stored in float32 (instead of float64 for UTM)"""

    dtype = np.dtype([
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("z_over_ground", np.float32),
        ("intensity", np.float32),
        ("is_ground", np.bool_)
    ])

    def __init__(self, points, z_encoding='global'):
        """ Constructor, parsing input into internal point cloud data structure

        Args:
            points: points numpy dict, containing x, y, z, intensity, is_ground, and z_over_ground arrays
            z_encoding (str): method used to normalize the z-dimension to the local ground level
        """
        super().__init__(points, vrf=True, z_encoding=z_encoding)


# Module functions #####################################################################################################
########################################################################################################################


def convert_point_cloud_dict_to_numpy(point_cloud_dict, speedup=False):
    """ Convert PC dictionary into a more efficient (dict-like) numpy array

    Args:
        point_cloud_dict: point cloud dictionary
        speedup: use int32 encoding if true, else float64

    Returns:
        points: PC encoded as numpy array

    """
    if speedup:
        dtype = np.dtype([
            ("x", np.int32),
            ("y", np.int32),
            ("z", np.int32),
            ("intensity", np.int32),
            ("is_ground", np.bool_)
        ])
    else:
        dtype = np.dtype([
            ("x", np.float64),
            ("y", np.float64),
            ("z", np.float64),
            ("intensity", np.float32),
            ("is_ground", np.bool_)
        ])

    num_points = len(point_cloud_dict['is_ground'])
    points = np.zeros(num_points, dtype=dtype)  # allocate object
    points['x'] = point_cloud_dict['x']
    points['y'] = point_cloud_dict['y']
    points['z'] = point_cloud_dict['z']
    points['intensity'] = point_cloud_dict['intensity']
    points['is_ground'] = point_cloud_dict['is_ground']

    return points
