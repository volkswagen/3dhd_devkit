""" Viewer for point clouds using PPTK or Mayavi (visualization frameworks).
"""
import os
import time
import numpy as np
import mayavi.mlab as mlab

from dataset_api.pc_reader import read_binary_hdpc_file
from visualization import visualization_basics as vis_basics

# Class definitions ####################################################################################################
########################################################################################################################


class PointCloudViewer:
    """ Viewer for displaying point cloud tiles using Mayavi or PPTK
    """
    def __init__(self, file_path, framework='mayavi'):
        """
        Args:
            file_path (str): file path of point cloud tile to visualize
            framework (str): visualization framework
        """
        self.file_path = file_path
        self.framework = framework
        self.hdpc = None

    def read_binary_file(self):
        """ Reads a binary encoded point cloud tile providing a point cloud dictionary (dict:float).
        """
        self.hdpc = read_binary_hdpc_file(self.file_path)

    def display_point_cloud(self, visualize_ground_separately=False):
        """ Displays a point cloud.

        Args:
            visualize_ground_separately (bool): whether to visualize ground classification
        """
        # Convert to dict (not dict type if read from binary file (tile))
        point_cloud_dict = self.hdpc

        # Normalize for visualization
        x_mean = np.mean(point_cloud_dict['x'])
        y_mean = np.mean(point_cloud_dict['y'])
        z_mean = np.mean(point_cloud_dict['z'])
        point_cloud_dict['x'] -= x_mean
        point_cloud_dict['y'] -= y_mean
        point_cloud_dict['z'] -= z_mean

        # Visualize ground points (only pptk)
        if visualize_ground_separately:
            gnd_pts, ele_pts = get_ground_and_elevated_points_from_dict(point_cloud_dict)
            vis_basics.vis_pcs([gnd_pts, ele_pts], colors=[(0, 1, 0), (1, 0, 0)], use_z_over_ground=False)
            return

        # Visualize whole point cloud using intensity
        if self.framework == 'pptk':
            vis_basics.vis_pc_pptk(point_cloud_dict, vis='intensity')
        else:
            vis_basics.vis_pc(point_cloud_dict, use_z_over_ground=False)


# Module functions #####################################################################################################
########################################################################################################################


def get_ground_and_elevated_points_from_dict(point_cloud_dict):
    """ Splits points in the point cloud into ground and elevated points for separate visualization.
    Args:
        point_cloud_dict (dict): point cloud dict, keys: x, y, z, intensity, is_ground

    Returns:
        gnd_points (dict): point cloud dict, keys: x, y, z, intensity, is_ground
        ele_points (dict): point cloud dict, keys: x, y, z, intensity, is_ground
    """
    # Covert to numpy
    dtype = np.dtype([
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("intensity", np.float32),
        ("is_ground", np.float32)
    ])
    num_points = len(point_cloud_dict['is_ground'])
    points = np.zeros(num_points, dtype=dtype)  # allocate object
    points['x'] = point_cloud_dict['x']
    points['y'] = point_cloud_dict['y']
    points['z'] = point_cloud_dict['z']
    points['intensity'] = point_cloud_dict['intensity']
    points['is_ground'] = point_cloud_dict['is_ground']

    # Get ground and elevated points
    gnd_points = points[point_cloud_dict['is_ground']]
    ele_points = points[~point_cloud_dict['is_ground']]

    return gnd_points, ele_points


# Script section #######################################################################################################
########################################################################################################################

def run_visualize_binary_point_cloud():
    # Settings
    source_path = r"E:\Datasets\3DHD_CityScenes\HD_PointCloud_Tiles"
    framework = 'pptk'
    vis_ground = False
    # framework = 'mayavi'

    # Visualize
    file_path = os.path.join(os.path.join(source_path, 'HH_005.bin'))
    pc_viewer = PointCloudViewer(file_path, framework=framework)
    print("Reading bin file...")
    pc_viewer.read_binary_file()
    print("Displaying...")

    # visualize_ground_separately=True ( only supported for pptk)
    pc_viewer.display_point_cloud(visualize_ground_separately=vis_ground)
    mlab.show()
    time.sleep(10)


# Run section ##########################################################################################################
########################################################################################################################


def main():
    run_visualize_binary_point_cloud()


if __name__ == "__main__":
    main()
