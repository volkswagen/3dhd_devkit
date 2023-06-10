""" Reads map elements in the 3DHD CityScenes dataset. """

import os
import numpy as np

from dataset import helper
from utility import transformations as trafo

# Class definitions ####################################################################################################
########################################################################################################################


class MapReader:
    """ Reads map element JSON files.

    Additionally, this class provides methods for collecting map elements within a specified pose and radius.
    """

    def __init__(self, map_base_dir, configured_element_types):
        """
        Args:
            map_base_dir (str | Path): path to map json files, e.g., C:\3DHD_CityScenes\HD_Map
            configured_element_types (list[str]): configured element types
        """
        self.map_base_dir = map_base_dir
        self.configured_element_types = configured_element_types
        self._map_elements = []     # (list[dict]) of parsed map elements

    def filter_map_elements_by_location(self, global_pose, radius, utm=False):
        """ Filters map elements lying in the circle of specified radius around a global pose.
        Args:
            radius (float): defines region of interest (ROI) as radius around global pose for map elements
            global_pose (float, float): (lat, lon) latitude and longitude of ROI center
            utm (bool): true if global pose is already in UTM coordinate system

        Returns:
            filtered_elements (list[dict]): list of map elements lying inside the specified ROI
        """
        # Transform pose on map to UTM
        if not utm:
            global_pose_utm = trafo.transform_wgs_to_utm(global_pose[0], global_pose[1])
        else:
            global_pose_utm = global_pose

        # Find elements inside ROI
        filtered_elements = []
        for _, element in enumerate(self._map_elements):
            # Compute distance to global pose
            if 'points_utm' in element.keys():
                x_pos = element['points_utm'][0][0]
                y_pos = element['points_utm'][0][1]
            else:
                x_pos = element['x_utm']
                y_pos = element['y_utm']
            dist = np.sqrt((x_pos - global_pose_utm[0])**2 + (y_pos - global_pose_utm[1])**2)
            if dist <= radius:
                filtered_elements.append(element)

        return filtered_elements

    def read_map_json_files(self):
        """ Reads configured map json files.

        Returns: self._map_elements (list[dict]): all parsed map elements

        """
        for element_type in self.configured_element_types:
            filename = f"{element_type.capitalize()}.json"
            json_path = os.path.join(self.map_base_dir, filename)
            map_elements = helper.load_from_json(json_path)
            self._map_elements.extend(map_elements)

    def get_map_elements(self):
        return self._map_elements

# Run section ##########################################################################################################
########################################################################################################################


def main():
    pass


if __name__ == "__main__":
    main()
