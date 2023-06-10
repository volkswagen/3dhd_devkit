""" Visualization of voxelization.
"""

import numpy as np
from visualization.visualization_basics import plot_pc

# Module functions #####################################################################################################
########################################################################################################################


def vis_pc_from_voxels(voxels):
    """ Visualizes the voxelized point cloud.

    Note that voxels are zero padded with each voxel containing a fixed number of points (e.g., 96).

    Args:
        voxels (float tensor): [M, max_points, ndim] with [M voxels, max points per voxel (96), number dimensions (4)].
    """
    points = get_points_list_from_voxels(voxels)
    plot_pc(points[:, 0], points[:, 1], points[:, 2], points[:, 3])


def vis_removed_points(hdpc, voxels):
    """ Visualizes points lost due to discretization.

    Args:
        hdpc (point_cloud.PointCloud): Whole point cloud before voxelization
        voxels (float tensor): [M, max_points, ndim] voxelized point cloud
    """
    voxel_points = get_points_list_from_voxels(voxels)
    pc_points = np.column_stack((hdpc.points['x'],
                                 hdpc.points['y'],
                                 hdpc.points['z_over_ground'],
                                 hdpc.points['intensity']))

    # Get difference in points
    _, ncols = pc_points.shape  # nrows, ncols
    dtype = {'names': [f'f{i}' for i in range(ncols)], 'formats': ncols * [pc_points.dtype]}
    diff = np.setdiff1d(pc_points.copy().view(dtype), voxel_points.copy().view(dtype))
    diff = diff.view(pc_points.dtype).reshape(-1, 4)
    print(f"{pc_points.shape[0]} points --> {voxel_points.shape[0]} points ({diff.shape[0]} points deleted)")

    if diff.shape[0] > 0:  # Plot all removed points as purple
        plot_pc(diff[:, 0], diff[:, 1], diff[:, 2], [0.0] * diff.shape[0], colormap='spring')


def get_points_list_from_voxels(voxels):
    """ Removes zero-padding from voxels and returns list of points.
    """
    voxels = np.reshape(voxels, (voxels.shape[0] * voxels.shape[1], 4))
    indices = np.nonzero(voxels)  # for only one zero element
    valid_rows = indices[0]
    valid_rows = np.unique(valid_rows)
    voxels = voxels[valid_rows]
    return voxels
