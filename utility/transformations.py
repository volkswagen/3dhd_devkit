""" Collection of methods for coordinate and point cloud transformations. """

import math as m
import numpy as np
import utm


def transform_wgs_to_utm(lat, lon):
    """
    This function converts wgs84 (lat, lon) values to utm (easting, northing) coordinates
    Args:
        lat (float): latitude value of a map element in WGS84
        lon (float): longitude value of a map element in WGS84

    Returns:
        ndarray containing utm coordinates (easting and northing) = (x, y)

    """
    utm_coords = utm.from_latlon(lat, lon)
    return [utm_coords[0], utm_coords[1]]


def transform_points_vrf_to_utm(points_vrf, position_utm, orientation_utm):
    """
    This function transforms laser points from vehicle coordinate system (VRF) to UTM coordinates using
    global position and heading information. For definition of coordinate systems see
    https://de.wikipedia.org/wiki/Fahrzeugkoordinatensystem#/media/Datei:RPY_angles_of_cars.png.

    Args:
        points_vrf (ndarray): points in vehicle reference frame (VRF) [3, N] = [[x], [y], [z]]
        position_utm (list | ndarray): vehicle position in global UTM coordinate system   [x, y, z=altitude]
        orientation_utm (list | ndarray): vehicle orientation in global UTM coordinate system [pitch, roll, yaw]

    Returns:
        points_utm: points transformed in UTM coordinates.
    """
    points_vrf = np.array(points_vrf)

    # Convert to radian
    roll = m.radians(orientation_utm[0])            # x
    pitch = m.radians(orientation_utm[1])           # y
    yaw = m.radians(orientation_utm[2])             # z

    # Rotation matrix
    R_roll = np.array([[1, 0, 0], [0, m.cos(roll), -m.sin(roll)], [0, m.sin(roll), m.cos(roll)]])
    R_pitch = np.array([[m.cos(pitch), 0, m.sin(pitch)], [0, 1, 0], [-m.sin(pitch), 0, m.cos(pitch)]])
    R_yaw = np.array([[m.cos(yaw), -m.sin(yaw), 0], [m.sin(yaw), m.cos(yaw), 0], [0, 0, 1]])

    R = np.matmul(R_yaw, R_pitch)
    R = np.matmul(R, R_roll)

    # Transform points
    points_utm = (np.dot(R, points_vrf).T + position_utm).T

    return points_utm


def transform_points_utm_to_vrf(points_utm, position_utm, orientation_utm):
    """
    This function transforms the measurements from utm into the vehicle coordinate system.

    Args:
        points_utm (ndarray): points in global UTM coordinate system [3, N] = [[x], [y], [z]]
        orientation_utm (list | ndarray): vehicle orientation in global UTM coordinate system [pitch, roll, yaw]
        position_utm (list | ndarray): vehicle position in global UTM coordinate system [x, y, z=altitude]

    Returns:
         points_vrf: points transformed into the vehicle reference frame (VRF).
    """

    # Convert to radian
    r = m.radians(orientation_utm[0])   # roll
    p = m.radians(orientation_utm[1])   # pitch
    y = m.radians(orientation_utm[2])   # yaw

    # Calculate rotation matrix (3x3), see Wikipedia
    rot = np.array([[m.cos(p) * m.cos(y), m.sin(r) * m.sin(p) * m.cos(y) - m.cos(r) * m.sin(y),
                     m.cos(r) * m.sin(p) * m.cos(y) + m.sin(r) * m.sin(y)],
                    [m.cos(p) * m.sin(y), m.sin(r) * m.sin(p) * m.sin(y) + m.cos(r) * m.cos(y),
                     m.cos(r) * m.sin(p) * m.sin(y) - m.sin(r) * m.cos(y)],
                    [(-1) * m.sin(p), m.sin(r) * m.cos(p), m.cos(r) * m.cos(p)]]).T

    # Transformation
    points_vrf = np.dot(rot, (points_utm.T - position_utm).T)

    return points_vrf


def rotate_points(points, yaw, centroid=None):
    """ Rotates points along the z-axis by <yaw> degrees. If a centroid is specified, it will be used as rotation base.

    Args:
        points (ndarray): points to rotate [3, N]
        yaw (float): yaw in degrees
        centroid (list | ndarray | None): [3] rotate around this centroid. If none, will rotate around [0, 0, 0].

    Returns:
        points: ndarray comprising the rotated points [3, N]
    """
    yaw = m.radians(yaw)
    # Calculate rotation matrix (3x3), see Wikipedia
    R = np.array([[m.cos(yaw), -m.sin(yaw), 0], [m.sin(yaw), m.cos(yaw), 0], [0, 0, 1]])

    # Transform points
    if centroid is None:
        points = np.dot(R, points)
    else:
        points = np.dot(R, (points.T - centroid).T).T + centroid
        points = points.T

    return points
