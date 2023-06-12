""" Point cloud operations for voxelization and filtering.

Adopted from https://github.com/traveller59/second.pytorch
    PointPillars fork from SECOND.
    Original code written by Alex Lang and Oscar Beijbom, 2018.
    Licensed under MIT License [see LICENSE].
Modified by Christopher Plachetka and Benjamin Sertolli, 2022.
    Licensed under MIT License [see LICENSE].
"""

import numba
import numpy as np


@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=35,
                                    max_voxels=20000):
    """ Voxelization (reversing point order).
    """
    N = points.shape[0]
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num

@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=35,
                            max_voxels=20000):
    """ Voxelization.
    """
    N = points.shape[0]
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(points,
                    voxel_size,
                    coors_range,
                    max_points=35,
                    reverse_index=True,
                    max_voxels=20000):
    """ Converts a point cloud (N, >=3) to voxels.

    Args:
        points (np.array): [N, ndim] points[:, :3] provides xyz points and
            points[:, 3:] contain other information such as intensity.
            N: number of points, ndim: number of dimensions (x,y,z,intensity)
        voxel_size (list[float)]: [3] size of voxels in x,y,z [x_size, y_size, z_size], [meter]
        coors_range (np.array): [6] extent of point cloud [xmin, ymin, zmin, xmax, ymax, zmax]
        max_points (int): maximum number of points contained in a voxel
        reverse_index (bool): indicate whether return reversed coordinates.
            If points has xyz format and reverse_index is True, output
            coordinates will be zyx format, but points in features always xyz format.
        max_voxels (int): Maximum number of voxels to create

    Returns:
        voxels (float tensor): [M, max_points, ndim] with [M voxels, max points per voxel (96), number dimensions (4)].
        coordinates (int tensor): [M, 3] with M voxels with x, y, z coordinates
        num_points_per_voxel (int tensor): [M] with M point counters for each voxel
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    else:
        voxel_num = _points_to_voxel_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]

    return voxels, coors, num_points_per_voxel


def filter_voxels(voxels, coordinates, num_points, max_keep, z_thres=.1):
    """ Downsampling voxels based on their distribution.

    Fewer number of voxels increase speed and reduce GPU memory requirements.

    Args:
        voxels (float tensor): [M, max_points, ndim] with [M voxels, max points per voxel (96), number dimensions (4)].
        coordinates (int tensor): [M, 3] with M voxels with x, y, z coordinates
        num_points (int tensor): [M] with M point counters for each voxel
        max_keep (int): number of voxels to keep.
        z_thres (float): threshold value for z-distribution within a voxel

    Returns:
        voxels (float tensor): downsampled voxels
        coordinates (int tensor): downsampled coordinates
        num_points (int tensor): downsampled num_points
    """
    # Filter by z-distribution: remove voxels with too similar z-values (e.g., ground)
    z_vec = voxels[:, :, 2]
    min_z_vec = np.min(z_vec, axis=1)
    max_z_vec = np.max(z_vec, axis=1)
    delta_z_vec = max_z_vec - min_z_vec
    valid = delta_z_vec > z_thres

    # Keep valid ones
    voxels_valid = voxels[valid]
    coordinates_valid = coordinates[valid]
    num_points_valid = num_points[valid]
    n_valid_z = voxels_valid.shape[0]

    # Get counterparts
    invalid = np.invert(valid)
    voxels_invalid = voxels[invalid]
    coordinates_invalid = coordinates[invalid]
    num_points_invalid = num_points[invalid]
    n_invalid = voxels_invalid.shape[0]

    # Not enough voxels: Take voxels from "invalid" voxels if number allows
    if n_valid_z < max_keep:
        num_to_add = max_keep - n_valid_z
        keep_indices = np.array(list(range(0, n_invalid)))  # list of indices to randomply sample from
        keep_indices = np.random.choice(keep_indices, num_to_add)  # sample a number of voxels

        voxels_invalid = voxels_invalid[keep_indices]
        coordinates_invalid = coordinates_invalid[keep_indices]
        num_points_invalid = num_points_invalid[keep_indices]

        voxels = np.concatenate((voxels_valid, voxels_invalid), axis=0)
        coordinates = np.concatenate((coordinates_valid, coordinates_invalid), axis=0)
        num_points = np.concatenate((num_points_valid, num_points_invalid))

    elif n_valid_z > max_keep:
        keep_indices = np.array(list(range(0, voxels_valid.shape[0])))  # list from 1-M
        keep_indices = np.random.choice(keep_indices, max_keep)

        voxels = voxels_valid[keep_indices]
        coordinates = coordinates_valid[keep_indices]
        num_points = num_points_valid[keep_indices]

    return voxels, coordinates, num_points


def get_crop_2d(points, x, y, square_size):
    """ Creates a crop in the x-y-plane from a larger point cloud.

    Args:
        points (np.ndarray): array containing points, keys: x,y,z,intensity
        x (float): x-coordinate of crop
        y (float): x-coordinate of crop
        square_size (float): size of crop

    Returns:
        points (np.ndarray): created crop
    """
    x_min = x - square_size
    x_max = x + square_size

    y_min = y - square_size
    y_max = y + square_size

    valid_x = np.logical_and(points['x'] < x_max, points['x'] > x_min)
    points = points[valid_x]
    valid_y = np.logical_and(points['y'] < y_max, points['y'] > y_min)
    points = points[valid_y]

    return points


def get_crop_3d(points, x, y, z, cube_size):
    """ Creates cubic crop from a larger point cloud.

    Args:
        points (np.ndarray): array containing points, keys: x,y,z,intensity
        x (float): x-coordinate of crop
        y (float): x-coordinate of crop
        z (float): x-coordinate of crop
        cube_size (float): size of crop

    Returns:
        points (np.ndarray): created crop
    """
    points = get_crop_2d(points, x, y, cube_size)
    z_min = z - cube_size
    z_max = z + cube_size
    valid_z = np.logical_and(points['z'] < z_max, points['z'] > z_min)
    points = points[valid_z]

    return points


def remove_points_2d(points, x, y, square_size):
    """ Removes points from a point cloud.

    Args:
        points (np.ndarray): array containing points, keys: x,y,z,intensity
        x (float): x-coordinate of crop
        y (float): x-coordinate of crop
        square_size (float): size of crop

    Returns:
        points (np.ndarray): points without removed part
    """
    x_min = x - square_size
    x_max = x + square_size

    y_min = y - square_size
    y_max = y + square_size

    valid_x = np.logical_and(points['x'] < x_max, points['x'] > x_min)
    valid_y = np.logical_and(points['y'] < y_max, points['y'] > y_min)

    valid = np.logical_and(valid_x, valid_y)
    valid = np.invert(valid)
    points = points[valid]

    return points
