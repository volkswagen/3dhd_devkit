""" A viewer for 3DHD CityScenes (point cloud and map).
"""
import os
import mayavi.mlab as mlab
import numpy as np

from dataset_api import map_reader
from dataset_api import pc_reader

from visualization import visualization_basics as vis_basics
from visualization import visualization_elements as vis_elements

# Class definitions ####################################################################################################
########################################################################################################################


class DatasetViewer:
    """ Viewer for 3DHD CityScenes, displaying point clouds and HD map elements using Mayavi.
    """
    def __init__(self, pc_dir=None, map_dir=None, configured_element_types=None):
        """
        Args:
            pc_dir (str): directory of point cloud tiles
            map_dir (str): directory of map element json files
            configured_element_types (list[str]): list of map element types to visualize
        """
        self.pc_dir = pc_dir
        self.map_dir = map_dir
        self.configured_element_types = configured_element_types

        self.hdpc = None            # point cloud container
        self.map_elements = None    # list of map elements

        self._map_r = map_reader.MapReader(self.map_dir, self.configured_element_types)
        self._map_r.read_map_json_files()

        self._vis_norm_x = 500000   # normalization of points for visualization as Mayavi can't handle too large numbers
        self._vis_norm_y = 5000000
        self._vis_norm_z = 0

        self._vis_x_min = None
        self._vis_x_max = None
        self._vis_y_min = None
        self._vis_y_max = None

    def read_pc_binary_file(self, pc_file_name):
        """ Reads a binary point cloud tile.
        Args:
            pc_file_name (str): tile name to read
        """
        pc_binary_file_path = os.path.join(self.pc_dir, pc_file_name)
        self.hdpc = pc_reader.read_binary_hdpc_file(pc_binary_file_path)

    def read_map_elements(self):
        """ Reads configured element types.
        """
        map_r = map_reader.MapReader(map_base_dir=self.map_dir, configured_element_types=self.configured_element_types)
        map_r.read_map_json_files()
        self.map_elements = map_r.get_map_elements()

    def read_area(self, pc_file_name, radius=100):
        """ Reads and displays the point cloud and map elements in a specific area.
        Args:
            pc_file_name (str): Name of pc file to read
            radius (float): Radius around tile center within which to get map elements [meter]
        Returns: -
        """
        self.read_pc_binary_file(pc_file_name)
        x_mean_utm = np.mean(self.hdpc['x'])
        y_mean_utm = np.mean(self.hdpc['y'])
        self.map_elements = self._map_r.filter_map_elements_by_location(
            global_pose=(x_mean_utm, y_mean_utm), radius=radius, utm=True)

        print(f"Number of elements found: {len(self.map_elements)}")

    def display_point_cloud(self):
        """ Displays the point cloud.
        """
        # Normalize, required for UTM values as value ranges are too large for proper displaying by Mayavi.
        x_mean = np.mean(self.hdpc['x'])
        y_mean = np.mean(self.hdpc['y'])
        z_mean = np.mean(self.hdpc['z'])

        self.hdpc['x'] -= x_mean
        self.hdpc['y'] -= y_mean
        self.hdpc['z'] -= z_mean

        x_min = np.min(self.hdpc['x'])
        x_max = np.max(self.hdpc['x'])
        y_min = np.min(self.hdpc['y'])
        y_max = np.max(self.hdpc['y'])

        # Store norm values used for visualization (will be applied to map elements too)
        self._vis_x_min = x_min
        self._vis_x_max = x_max
        self._vis_y_min = y_min
        self._vis_y_max = y_max

        self._vis_norm_x = x_mean
        self._vis_norm_y = y_mean
        self._vis_norm_z = z_mean

        vis_basics.vis_pc(self.hdpc, use_z_over_ground=False)

    def display_map_elements(self):
        """ Displays map elements.

        Make sure to display the point cloud first to obtain normalization values for map rendering.
        """
        # Normalize map elements (otherwise not displayable as value ranges are too big)
        for element in self.map_elements:
            # Point elements: contain single coordinates. Line elements: multiple points
            if 'x_utm' in element.keys():
                element['x_utm'] -= self._vis_norm_x
                element['y_utm'] -= self._vis_norm_y
                element['z_utm'] -= self._vis_norm_z
            else:
                for i_point, point_utm in enumerate(element['points_utm']):
                    point_utm = list(point_utm)     # convert from tuple to list
                    point_utm[0] -= self._vis_norm_x
                    point_utm[1] -= self._vis_norm_y
                    point_utm[2] -= self._vis_norm_z

                    element['points_utm'][i_point] = point_utm

                # Compute a center of gravity for visualization limit
                element['points_utm'] = np.array(element['points_utm'])
                element['x_utm'] = np.mean(element['points_utm'][:, 0])
                element['y_utm'] = np.mean(element['points_utm'][:, 1])
                element['z_utm'] = np.mean(element['points_utm'][:, 2])

        # Limit map elements
        self.map_elements = [e for e in self.map_elements if e['x_utm'] > self._vis_x_min and e['x_utm'] < self._vis_x_max]
        self.map_elements = [e for e in self.map_elements if e['y_utm'] > self._vis_y_min and e['y_utm'] < self._vis_y_max]

        # Visualize
        if 'Lights' in self.configured_element_types:
            lights = [e for e in self.map_elements if e['type'] == 'TrafficLight']
            for e in lights:
                e['yaw_utm'] = e['yaw_utm'] - 90
            vis_elements.vis_map_elements(lights, vrf=False, display_class=False, color_by_dev_extension=True)

        if 'Poles' in self.configured_element_types:
            poles = [e for e in self.map_elements if e['type'] == 'Pole']
            vis_elements.vis_map_elements(poles, vrf=False, display_class=False, color_by_dev_extension=False)

        if 'Signs' in self.configured_element_types:
            signs = [e for e in self.map_elements if e['type'] == 'TrafficSign']
            vis_elements.vis_map_elements(signs, vrf=False, display_class=False, color_by_dev_extension=False)

        if 'CS_Obstacles_Point' in self.configured_element_types:
            cs_obstacles_point = [e for e in self.map_elements if e['type'] == 'CS_Obstacle_Point']
            vis_elements.vis_map_elements(
                cs_obstacles_point, vrf=False, display_class=False, color_by_dev_extension=True)

        if 'CS_Obstacles_Line' in self.configured_element_types:
            cs_obstacles_line = [e for e in self.map_elements if e['type'] == 'CS_Obstacle_Line']
            vis_elements.vis_road_furniture_objects(
                cs_obstacles_line, vrf=False, color_by_flag=False, display_class=False, color_by_dev_extension=True)

        if 'Lanes_Ordinary' in self.configured_element_types:
            lanes_ordinary = [e for e in self.map_elements if e['type'] == 'Lane_Ordinary']
            vis_elements.vis_lanes(
                lanes_ordinary, vrf=False, color_by_flag=False, display_class=False, color_by_dev_extension=True)

        if 'Lanes_Temporary' in self.configured_element_types:
            lanes_temporary = [e for e in self.map_elements if e['type'] == 'Lane_Temporary']
            vis_elements.vis_lanes(
                lanes_temporary, vrf=False, color_by_flag=False, display_class=False, color_by_dev_extension=True)

        if 'Markings_Line' in self.configured_element_types:
            markings_line = [e for e in self.map_elements if e['type'] == 'Marking_Line']
            vis_elements.vis_markings(markings_line, vrf=False, color_by_flag=False, color_by_color=False,
                                      display_class=False, color_by_dev_extension=False)

        if 'Markings_Polygon_Ordinary' in self.configured_element_types:
            markings_polygon_ordi = [e for e in self.map_elements if e['type'] == 'Marking_Polygon_Ordinary']
            vis_elements.vis_markings(markings_polygon_ordi, vrf=False, color_by_flag=False, color_by_color=False,
                                      display_class=False, color_by_dev_extension=True)

        if 'Markings_Polygon_Arrow' in self.configured_element_types:
            markings_polygon_arrow = [e for e in self.map_elements if e['type'] == 'Marking_Polygon_Arrow']
            vis_elements.vis_markings(markings_polygon_arrow, vrf=False, color_by_flag=False, color_by_color=False,
                                      display_class=False, color_by_dev_extension=True)

        if 'Markings_Polygon_Negation' in self.configured_element_types:
            markings_polygon_neg = [e for e in self.map_elements if e['type'] == 'Marking_Polygon_Negation']
            vis_elements.vis_markings(markings_polygon_neg, vrf=False, color_by_flag=False, color_by_color=False,
                                      display_class=False, color_by_dev_extension=True)

        if 'Markings_Polygon_Symbol' in self.configured_element_types:
            markings_polygon_sym = [e for e in self.map_elements if e['type'] == 'Marking_Polygon_Symbol']
            vis_elements.vis_markings(markings_polygon_sym, vrf=False, color_by_flag=False, color_by_color=False,
                                      display_class=False, color_by_dev_extension=True)

        if 'Markings_Polygon_Text' in self.configured_element_types:
            markings_polygon_txt = [e for e in self.map_elements if e['type'] == 'Marking_Polygon_Text']
            vis_elements.vis_markings(markings_polygon_txt, vrf=False, color_by_flag=False, color_by_color=False,
                                      display_class=False, color_by_dev_extension=True)

        if 'Relations' in self.configured_element_types:
            relations = [e for e in self.map_elements if e['type'] == 'Relation']
            vis_elements.vis_relations(relations, vrf=False, color_by_class=True, display_class=True)

        if 'Curbs' in self.configured_element_types:
            curbs = [e for e in self.map_elements if e['type'] == 'Curb']
            vis_elements.vis_road_furniture_objects(
                curbs, vrf=False, color_by_flag=False, display_class=False, color_by_dev_extension=False)


# Script section #######################################################################################################
########################################################################################################################

def run_view_test_area():
    # Settings
    pc_dir = r"C:\Workspace\datasets\3DHD_CityScenes\HD_PointCloud_Tiles"
    map_dir = r"C:\Workspace\datasets\3DHD_CityScenes\HD_Map"
    pc_file_name = "HH_005.bin"
    configured_element_types = [
        'Lights',
        'Poles',
        'Signs',
        'CS_Obstacles_Point',
        'CS_Obstacles_Line',
        'Lanes_Ordinary',
        'Lanes_Temporary',
        'Markings_Line',
        'Markings_Polygon_Ordinary',
        'Markings_Polygon_Negation',
        'Markings_Polygon_Arrow',
        'Markings_Polygon_Text',
        'Markings_Polygon_Symbol',
        'Relations',
        'Curbs'
    ]

    # View data
    dataset_viewer = DatasetViewer(pc_dir, map_dir, configured_element_types)
    print("Reading data...")
    dataset_viewer.read_area(pc_file_name, radius=50)

    print("Displaying...")
    mlab.figure(bgcolor=(1, 1, 1))
    dataset_viewer.display_point_cloud()
    dataset_viewer.display_map_elements()
    mlab.show()


# Run section ##########################################################################################################
########################################################################################################################


def main():
    run_view_test_area()


if __name__ == "__main__":
    main()
