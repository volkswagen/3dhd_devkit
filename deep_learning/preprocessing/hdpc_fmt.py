""" Feature map transformation (voxelization) for high-density point clouds.
"""
import numpy as np

from dataset import point_cloud as pc
from deep_learning.util import point_cloud_ops as pcops


# Class definitions ####################################################################################################
########################################################################################################################


class HdpcFMT:
    """ Feature map transformer (FMT) for high-density point clouds (HDPC).

    Voxelizes a point cloud into a tensor as input to the network's encoder.

    """

    def __init__(self, point_cloud=None):
        """
        Args:
            point_cloud (PointCloud): point cloud object (see dataset.point_cloud)
        """
        self.pc: pc.PointCloud = None
        if point_cloud is not None:
            self.set_point_cloud_data(point_cloud)

    def set_point_cloud_data(self, point_cloud):
        """ Set point cloud data for voxelization.
        """
        if not isinstance(point_cloud, pc.PointCloud):
            raise ValueError(f"Input must be of class PointCloud, but is class {type(point_cloud)}")
        self.pc = point_cloud

    def create_feature_map_voxels(self, fm_shape, fm_extent, voxel_preproc_settings):
        """ Pre-processing voxelizing a point cloud (list of points) into a tensor as network input.

        The generated hdpc_fm dict comprises:
        voxels (float tensor): [M, max_points, ndim] with [M voxels, max points per voxel (96), number dimensions (4)].
        coordinates (int tensor): [M, 3] with M voxels with x, y, z coordinates
        num_points_per_voxel (int tensor): [M] with M counters of points for each voxel

        Args:
            fm_shape (list[int]): number of voxels in x-, y-, and z-dimension
            fm_extent (list[list]): feature map extent as [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            voxel_preproc_settings (dict): pre-processing settings

        Returns:
            hdpc_fm (dict:tensor): output dict comprising voxels, coordinates, num_points

        """
        points = np.column_stack((self.pc.points['x'],
                                  self.pc.points['y'],
                                  self.pc.points['z_over_ground'],
                                  self.pc.points['intensity']))

        # Shuffle point list for quasi-random subsampling later
        np.random.seed(42)
        np.random.shuffle(points)

        # Voxelize point cloud
        # coors_range: [xmin, ymin, zmin, xmax, ymax, zmax]
        coors_range = np.array([fm_extent[0][0], fm_extent[1][0], fm_extent[2][0],
                                fm_extent[0][1], fm_extent[1][1], fm_extent[2][1]])
        max_voxels = int(fm_shape[0] * fm_shape[1] * fm_shape[2])
        voxels, coordinates, num_points = pcops.points_to_voxel(points=points,
                                                                voxel_size=voxel_preproc_settings['voxel_size'],
                                                                coors_range=coors_range,
                                                                max_points=voxel_preproc_settings['max_points_per_voxel'],
                                                                max_voxels=max_voxels,
                                                                reverse_index=True)

        # Filter voxels (keep only a subset for speed and lower memory requirements)
        max_keep = voxel_preproc_settings['max_num_voxels']
        if 0 < max_keep < voxels.shape[0]:
            voxels, coordinates, num_points = pcops.filter_voxels(voxels, coordinates, num_points,
                                                                  max_keep=max_keep,
                                                                  z_thres=.1)
        if voxel_preproc_settings['min_points_per_voxel'] > 1:
            keep_indices = np.argwhere(num_points >= voxel_preproc_settings['min_points_per_voxel'])[:, 0]
            voxels, coordinates, num_points = voxels[keep_indices], coordinates[keep_indices], num_points[keep_indices]

        hdpc_fm = {
            'voxels': voxels,
            'coordinates': coordinates,
            'num_points': num_points
        }
        return hdpc_fm


# Test section #########################################################################################################
########################################################################################################################


def main():
    pass


if __name__ == "__main__":
    main()
