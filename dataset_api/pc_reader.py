""" Reads high-density point clouds in the 3DHD CityScenes dataset. """

import math
import os
import copy

import numpy as np
import utm
from shapely.geometry import Polygon, box

from dataset import point_cloud as pc
from dataset import helper
from utility import transformations

# Module functions #####################################################################################################
########################################################################################################################


class HDPCReader:
    """ Reads and extracts a point cloud crop (ROI) from binary point cloud tiles.
    """

    def __init__(self, hdpc_tile_dir, map_meta_data_dir, pc_tiles=None):
        r"""
        Note: pc_tiles (see below) contains prefetched tiles loaded by deep_learning.training.trainer

        hdpc = high-density point cloud.
        Example directories:
        C:\3DHD_CityScenes\HD_PointCloud_Tiles
        C:\3DHD_CityScenes\HD_Map_MetaData

        Args:
            hdpc_tile_dir (str|Path): directory of point cloud tiles
            map_meta_data_dir (str|Path): directory of point cloud metadata (tile locations in UTM)
            pc_tiles: (dict[dict]): maps tile name to point cloud dict (x, y, z, intensity...)
        """
        self.hdpc_tile_dir = hdpc_tile_dir
        self.map_meta_data_path = os.path.join(map_meta_data_dir, 'HDPC_TileDefinition.json')
        self.tile_shapes = helper.load_from_json(self.map_meta_data_path)
        self.pc_tiles = pc_tiles

    def read_point_cloud(self, global_pose_wgs, max_extent: float = None, pc_crop_extent: list = None,
                         z_encoding='global', vrf=True):
        """ Reads point cloud tiles and returns a point cloud crop of specified location and extent.

        To speed up the cropping process (crop from available tiles to generate a sample point cloud),
        we operate in a scaled and shifted coordinate system, allowing for the usage of int32 instead of float64.
        Point cloud (PC) tiles in 3DHD CityScenes are stored in coordinates scaled by 1000, with offsets of
        500000 and 5000000 applied to x and y coordinates, respectively. Thus, the provided ROI (global_pose and
        pc_crop_extent) is first transformed into the scaled integer coordinate system for cropping, with the
        obtained crop eventually transformed to UTM (requiring float64) or the vehicle reference frame (VRF, float32).
        All values are provided in meter or degree.

        VRF: right-handed coordinate system, x forward, y left, z up.

        Args:
            global_pose_wgs (tuple): [lat, lon, yaw] global pose on the map in WGS84
            max_extent (float): maximum extent of the point cloud crop (used if no pc_crop_extent is provided)
            pc_crop_extent (list[list]): definition of the crop extent [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            z_encoding (str): defines the method by which to normalize the z-dimension
            vrf: (bool): true if point cloud crop is to be transformed into VRF

        Returns:
            hd_pc (point_cloud.PointCloud): created point cloud object
        """

        assert (max_extent is None) != (pc_crop_extent is None)  # exactly one of the two must be specified
        if pc_crop_extent is None:
            pc_crop_extent = [[max_extent, max_extent], [max_extent, max_extent], [max_extent, max_extent]]

        # Convert position to utm
        position_utm = utm.from_latlon(global_pose_wgs[0], global_pose_wgs[1])
        position_utm = np.array([position_utm[0], position_utm[1]])   # convert tuple to list

        # Calculate radius of the smallest circle encompassing the fm_extent (crop rectangle later with any orientation)
        x_max = max([abs(v) for v in pc_crop_extent[0]])  # max absolute value in x-dim
        y_max = max([abs(v) for v in pc_crop_extent[1]])
        d_ext = math.sqrt(x_max ** 2 + y_max ** 2)  # max diagonal length of fm_extent (from center)
        # Box: box module (from shapely), rectangle as shape geometry
        roi = box(position_utm[0] - d_ext, position_utm[1] - d_ext, position_utm[0] + d_ext, position_utm[1] + d_ext)

        # Find tiles intersecting ROI
        dtype_input = np.dtype([
            ("x", np.int32),
            ("y", np.int32),
            ("z", np.int32),
            ("intensity", np.int32),
            ("is_ground", np.bool_),
        ])
        scale = 1000
        offset_x = 500000
        offset_y = 5000000

        in_pts = np.array([], dtype=dtype_input)  # points from all tiles intersecting ROI
        # Iterate all tiles to check for intersection with ROI
        for tile in self.tile_shapes:
            # Create polygon object from map meta data
            points_utm = tile['points_utm']
            poly = Polygon(points_utm)  # points: corner points of sector shapes

            # Check for intersection with ROI, keep points in int32 for speed
            if poly.intersects(roi):
                if not self.pc_tiles:
                    inpc = read_binary_hdpc_file(os.path.join(self.hdpc_tile_dir, tile['name'] + '.bin'), unnorm=False)
                else:
                    inpc = self.pc_tiles[tile['name']]
                inpc = pc.convert_point_cloud_dict_to_numpy(inpc, speedup=True)
                # Convert to numpy
                in_pts = np.concatenate((in_pts, inpc))
                # Cleanup
                del inpc

        d_ext = np.int32(d_ext * scale)
        position_utm_scaled = copy.copy(position_utm)
        for dim, offset in zip([0, 1], [offset_x, offset_y]):
            position_utm_scaled[dim] = (position_utm[dim] - offset) * scale
        position_utm_scaled = position_utm_scaled.astype(np.int32)

        x_minmax = [(position_utm_scaled[0] - d_ext), (position_utm_scaled[0] + d_ext)]   # bounds for x
        y_minmax = [(position_utm_scaled[1] - d_ext), (position_utm_scaled[1] + d_ext)]   # bounds for y

        # First determine points valid by x
        valid_x = np.logical_and(in_pts['x'] >= x_minmax[0], in_pts['x'] <= x_minmax[1])
        in_pts = in_pts[valid_x]
        # Filter valid x points by bounds for y
        valid_y = np.logical_and(in_pts['y'] >= y_minmax[0], in_pts['y'] <= y_minmax[1])
        points = in_pts[valid_y]
        num_points = len(points)

        # Create point cloud object. VRF: float32, else is UTM with float64
        if vrf:
            # Transform point cloud into vehicle coordinate frame and create point cloud object
            # coords in float64 for transformation
            coords = np.array([points['x'], points['y'], points['z']], dtype=np.float64)
            coords[0, :] = (coords[0, :] / scale) + offset_x
            coords[1, :] = (coords[1, :] / scale) + offset_y
            coords[2, :] = (coords[2, :] / scale)

            # For exact reproduction of published experiments: use cm precision
            pos_utm = np.array([position_utm[0], position_utm[1], 0], dtype=np.float64)
            coords = transformations.transform_points_utm_to_vrf(coords, pos_utm, [0, 0, global_pose_wgs[2]])
            for dim in [0, 1, 2]:
                coords[dim, :] = np.float32(np.around(coords[dim, :] * 100) / 100)

            dtype_vrf = np.dtype([
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("intensity", np.float32),
                ("is_ground", np.bool_),
            ])
            points_vrf = np.zeros(num_points, dtype=dtype_vrf)
            points_vrf['x'] = coords[0, :]
            points_vrf['y'] = coords[1, :]
            points_vrf['z'] = coords[2, :]
            points_vrf['intensity'] = points['intensity'] / 2 ** 16
            points_vrf['intensity'] = np.around(points_vrf['intensity'] * 100) / 100
            points_vrf['is_ground'] = points['is_ground']

            hd_pc = pc.VRFPointCloud(points_vrf, z_encoding=z_encoding)
            hd_pc.crop_to_extent(pc_crop_extent, crop_z=False)
            hd_pc.normalize_z_over_ground()
        else:
            # Unnorm for UTM coordinates
            hd_pc = pc.UTMPointCloud(points, z_encoding=z_encoding)     # converts to float64
            hd_pc.points['x'] = (hd_pc.points['x'] / scale) + offset_x
            hd_pc.points['y'] = (hd_pc.points['y'] / scale) + offset_y
            hd_pc.points['z'] = hd_pc.points['z'] / scale
            hd_pc.points['intensity'] = hd_pc.points['intensity'] / 2 ** 16
            hd_pc.points['z_over_ground'] = hd_pc.points['z_over_ground'] / scale
            hd_pc.crop_to_extent(pc_crop_extent, crop_z=False)
            hd_pc.normalize_z_over_ground()

        return hd_pc


def read_binary_hdpc_file(file_path, unnorm=True):
    r""" Reads a binary (hdpc) point cloud file using numpy.
    Args:
        file_path (str|Path): path of point cloud tile to load, e.g., "3DHD_CityScenes\HD_PointCloud_Tiles\HH_001.bin"
        unnorm (bool): true to unnorm integer coordinate system to UTM (float64)

    Returns:
        pc_dict (dict): point cloud as dictionary, keys: x, y, z, intensity, is_ground (flag)
    """

    pc_dict = {}
    key_list = ['x', 'y', 'z', 'intensity', 'is_ground']
    type_list = ['<i4', '<i4', '<i4', '<u2', 'u1']

    with open(file_path, "r", encoding='utf-8') as fid:
        num_points = np.fromfile(fid, count=1, dtype='<u4')[0]

        # Init
        for k, dtype in zip(key_list, type_list):
            pc_dict[k] = np.zeros([num_points], dtype=dtype)

        # Read all arrays
        for k, t in zip(key_list, type_list):
            pc_dict[k] = np.fromfile(fid, count=num_points, dtype=t)

        # Unnorm: float64 needed to store UTM x,y,z coordinates
        if unnorm:
            pc_dict['x'] = (pc_dict['x'] / 1000) + 500000
            pc_dict['y'] = (pc_dict['y'] / 1000) + 5000000
            pc_dict['z'] = pc_dict['z'] / 1000
            pc_dict['intensity'] = pc_dict['intensity'] / 2**16
        pc_dict['is_ground'] = pc_dict['is_ground'].astype(np.bool_)

        fid.close()

        return pc_dict


def read_sample_point_cloud(file_path, z_encoding='global'):
    """ Reads a generated point cloud crop and parses it into an PointCloud object.

    This function is used if point cloud crops have been generated (load_lvl_hdpc == 'generated' in config.ini)
    prior to training.

    Args:
        file_path (str|Path): path to generated hdpc crop (.npz)
        z_encoding (str): specifies the normalization algorithm of the z-dimension (only 'global' available)

    Returns (SamplePointCloud): PointCloud object

    """
    if str(file_path)[-3:] == 'npz':
        data = np.load(file_path)
        points = data[data.files[0]]
    else:
        raise ValueError(f"Unknown file type: {file_path} (valid options: npz)")

    # Unnorm
    dtype = np.dtype([
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("intensity", np.float32),
        ("is_ground", np.bool_)
    ])
    points_unnorm = np.zeros(len(points), dtype=dtype)
    points_unnorm['x'] = points['x'] / 100
    points_unnorm['y'] = points['y'] / 100
    points_unnorm['z'] = points['z'] / 100
    points_unnorm['intensity'] = points['intensity'] / 100
    points_unnorm['is_ground'] = points['is_ground']

    return pc.VRFPointCloud(points_unnorm, z_encoding)


# Run section ##########################################################################################################
########################################################################################################################


def main():
    pass


if __name__ == "__main__":
    main()
