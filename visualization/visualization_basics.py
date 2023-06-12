""" Basic visualization functions.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import mayavi.mlab as mlab
import numpy as np
import pptk
from PIL import ImageColor
from matplotlib.colors import get_named_colors_mapping

from utility import transformations as trafo


# Class definitions ####################################################################################################
########################################################################################################################

class PlotC(ABC):
    @abstractmethod
    def plot(self):
        """ Creates a Mayavi plot.
        """
        return

    def show(self):
        """ Shows the created plot.
        """
        mlab.show()


class PointPlot(PlotC):
    """ Plotting class for point-like objects.
    """
    def __init__(self, x, y, z, s=None, **kwargs):
        """
        Args:
            x (list[float] | float): x-coordinates
            y (list[float] | float): x-coordinates
            z (list[float] | float): x-coordinates
            s (list[float] | float): additional attribute, e.g., intensity
            **kwargs: additional function parameters
        """
        self.x = x
        self.y = y
        self.z = z
        self.s = s
        self.kwargs = kwargs

    def plot(self):
        """ Creates a Mayavi plot.
        """
        if self.s is None:
            mlab.points3d(self.x, self.y, self.z, **self.kwargs)
        else:
            mlab.points3d(self.x, self.y, self.z, self.s, **self.kwargs)


class CuboidPlot(PlotC):
    """ Plotting class for cubes.
    """
    def __init__(self, x, y, z, size_x, size_y, size_z, yaw=0, **kwargs):
        """
        Args:
            x (float): x center coordinate
            y (float): y center coordinate
            z (float): z center coordinate
            size_x (float): size of cube in x-dim
            size_y (float): size of cube in y-dim
            size_z (float): size of cube in z-dim
            yaw (float): orientation of cube
            **kwargs: additional function parameters
        """
        self.x = x
        self.y = y
        self.z = z
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.yaw = yaw  # -yaw + 90
        self.kwargs = kwargs

    def plot(self):
        """ Creates a Mayavi plot.
        """
        # Define corner points of bounding cube
        points_list = []
        x_coords = [self.size_x / 2, -self.size_x / 2]
        y_coords = [self.size_y / 2, -self.size_y / 2]
        z_coords = [self.size_z / 2, -self.size_z / 2]
        for x in x_coords:
            points_list.append(np.array([x, y_coords[0], z_coords[0]]))
            points_list.append(np.array([x, y_coords[0], z_coords[1]]))
            points_list.append(np.array([x, y_coords[1], z_coords[1]]))
            points_list.append(np.array([x, y_coords[1], z_coords[0]]))

        # Rotate (in VRF), transpose and shift points
        points_norm = np.column_stack(points_list)
        points_norm = trafo.rotate_points(points_norm, self.yaw)
        points_norm = points_norm.T
        points = points_norm + np.array([self.x, self.y, self.z])

        # Plot two rectangles of cube (with points 0-3 and 4-7)
        for start in [0, 4]:
            first_point = np.reshape(points[start], (1, 3))
            rectangle = np.append(points[start:start + 4], first_point, axis=0)  # add first point to close rectangle
            mlab.plot3d(rectangle[:, 0], rectangle[:, 1], rectangle[:, 2], **self.kwargs)

        # Plot four connections between the two rectangles to finish cube
        for i in range(4):
            connection = [[points[i, dim], points[i + 4, dim]] for dim in range(3)]
            mlab.plot3d(connection[0], connection[1], connection[2], **self.kwargs)


# Module functions #####################################################################################################
########################################################################################################################


def vis_pc_pptk(point_cloud_dict, vis='intensity'):
    """ Visualizes a point cloud using PPTK framework.

    Args:
        point_cloud_dict (dict:float): point cloud as dictionary, keys: x, y, z, intensity
        vis (str): defines the point attribute to be visualized, e.g., 'intensity' or 'RGB'
    """

    # Use offset for proper visualization, otherwise points are not displayed correctly
    x_utm = point_cloud_dict['x'] - np.mean(point_cloud_dict['x'])
    y_utm = point_cloud_dict['y'] - np.mean(point_cloud_dict['y'])
    z_utm = point_cloud_dict['z']

    # Plot
    xyz = np.column_stack((x_utm, y_utm, z_utm))
    viewer = pptk.viewer(xyz)

    if vis == 'intensity':
        viewer.attributes(point_cloud_dict['intensity'])
        viewer.set(point_size=0.03)
        return

    if vis == 'rgb':
        rgb_list = list(zip(point_cloud_dict['R'], point_cloud_dict['G'], point_cloud_dict['B']))
        rgb = np.array(rgb_list) / 255
        viewer.attributes(rgb)

    # Display point cloud using pptk
    viewer.set(point_size=0.01)


def vis_pc(points, use_z_over_ground=True):
    """ Visualizes a point cloud using Mayavi (wrapper function simplifying function call).

    Args:
        points (dict:float | point_cloud.PointCloud): dict or point cloud objects, keys: x,y,z,z_over_ground, intensity
        use_z_over_ground (bool): whether to use 'z' or 'z_over_ground'
    """

    z = points['z']
    if use_z_over_ground:
        z = points['z_over_ground']

    plot_pc(points['x'], points['y'], z, points['intensity'])


def vis_pcs(pcs, colors, use_z_over_ground=True):
    """ Visualizes a list of point clouds using PPTK.

    Args:
        pcs (list[dict] | list[point_cloud.PointCloud]): list of point clouds (as dict or object)
        colors (list[list]): list of colors (one for each point cloud), e.g., [[R, G, B], [R, G, B]]
    """

    xyz_acc = np.zeros(shape=(0, 3))
    col_acc = np.zeros(shape=(0, 3))
    mean_x = np.mean(pcs[0]['x'])
    mean_y = np.mean(pcs[0]['y'])
    for pc, col in zip(pcs, colors):
        num_points = len(pc['x'])
        R = col[0]
        G = col[1]
        B = col[2]
        col = np.vstack((np.ones(num_points)*R, np.ones(num_points)*G, np.ones(num_points)*B))
        col = np.transpose(col)

        z = pc['z']
        if use_z_over_ground:
            z = pc['z_over_ground']

        xyz = np.column_stack((pc['x']-mean_x, pc['y']-mean_y, z))

        xyz_acc = np.concatenate((xyz_acc, xyz))
        col_acc = np.concatenate((col_acc, col))

    viewer = pptk.viewer(xyz_acc)
    viewer.attributes(col_acc)
    viewer.set(point_size=0.01)


def plot_pc(x, y, z, intensity, colormap='gray'):
    """ Visualizes a point cloud using Mayavi.

    Args:
        x (list[float]): x-coordinates
        y (list[float]): y-coordinates
        z (list[float]): z-coordinates
        intensity (list[float]): intensity values
        colormap (str): specifies the colormap for displaying intensity values
    """
    pc_plot = PointPlot(x, y, z, intensity,
                        colormap=colormap,
                        vmin=-0.1, vmax=.5,
                        mode='sphere',
                        scale_factor=.08,
                        scale_mode='none')
    pc_plot.plot()


def draw_cube(x, y, z, length, width, height, yaw, color=(1, 0, 0), opacity=.4):
    """ Draws a cube.

    The indices of points 1,2,3,4 in this function are counter clockwise.

    Args:
        x (float): middle_point x-coordinate
        y (float): middle_point y-coordinate
        z (float): middle_point z-coordinate
        length (float): length of cube (direction of yaw)
        width (float): width of cube (perpendicular to length vector)
        yaw (float): orientation mathematically positive
        color ((float, float, float)): RGB color
        opacity (float): opacity of cube, 1.0 -> no opacity

    Returns:

    """

    mid = np.array([x, y, z])
    yaw_l = np.radians(yaw)
    yaw_w = np.radians(yaw + 90)

    l_vec = np.array([length * np.cos(yaw_l), length * np.sin(yaw_l), 0])
    w_vec = np.array([width * np.cos(yaw_w), width * np.sin(yaw_w), 0])
    h_vec = np.array([0, 0, height])

    # Base face (vertices)
    b1 = mid - .5*w_vec - .5*l_vec - .5*h_vec
    b2 = mid - .5*w_vec + .5*l_vec - .5*h_vec
    b3 = mid + .5*w_vec + .5*l_vec - .5*h_vec
    b4 = mid + .5*w_vec - .5*l_vec - .5*h_vec

    # Top plate
    t1 = mid - .5 * w_vec - .5 * l_vec + .5 * h_vec
    t2 = mid - .5 * w_vec + .5 * l_vec + .5 * h_vec
    t3 = mid + .5 * w_vec + .5 * l_vec + .5 * h_vec
    t4 = mid + .5 * w_vec - .5 * l_vec + .5 * h_vec

    faces = [
        [b1, b2, b4, b3],   # base
        [t1, t2, t4, t3],   # top
        [b1, b2, t1, t2],   # length right
        [b3, b4, t3, t4],   # length left
        [b1, b4, t1, t4],   # width bottom
        [b2, b3, t2, t3]    # width top
    ]

    for face in faces:
        x, y, z = get_grid(face)
        mlab.mesh(x, y, z, opacity=opacity, color=color)


def get_grid(points_list):
    """ Helper function for cube generation.
    """
    f = np.array(points_list)
    x = np.reshape(f[:, 0], [2, 2])
    y = np.reshape(f[:, 1], [2, 2])
    z = np.reshape(f[:, 2], [2, 2])

    return x, y, z


def convert_color_strings_to_rgb(color_names: List[str]) -> List[Tuple[float]]:
    """ Converts color names to RGB tuples.

    Source for color names: https://htmlcolorcodes.com/color-names/

    Args:
        color_names (list[str]): list of strings

    Returns:
        list[(float, float, float)]: list of RGB tuples.
    """

    color_to_hex_map = get_named_colors_mapping()
    color_to_hex_map = {c: v for c, v in color_to_hex_map.items()
                        if isinstance(v, str)
                        and v.startswith('#')}
    colors_rgb = []
    for color_name in color_names:
        color_hex = color_to_hex_map[color_name.lower()] if not color_name.startswith('#') else color_name
        color_rgb_256 = ImageColor.getcolor(color_hex, "RGB")
        color_rgb = tuple(np.array(color_rgb_256) / 255.0)
        colors_rgb.append(color_rgb)
    return colors_rgb


def convert_color_string_to_rgb(color_name: str) -> Tuple[float]:
    """ Helper function for color name to RGB tuple conversion.
    """
    return convert_color_strings_to_rgb([color_name])[0]


# Script section #######################################################################################################
########################################################################################################################


def run_test_cube():
    """ Draws a cube.
    """
    draw_cube(x=1, y=2, z=3, length=8, width=3, height=1, yaw=325)
    mlab.show()


# Run section ##########################################################################################################
########################################################################################################################


def main():
    run_test_cube()


if __name__ == "__main__":
    main()
