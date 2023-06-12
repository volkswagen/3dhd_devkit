""" Interface defining map elements with their data fields.

For more info regarding data fields, see the 3DHD CityScenes documentation.
URL: https://zenodo.org/record/7085090#.Y_dpXWmZNaQ

"""

from abc import ABC, abstractmethod

# Class definitions  ###################################################################################################
########################################################################################################################


class MapElement(ABC):
    """ Base class defining the generic structure of map elements.
    """
    def __init__(self, e_type):
        """
        Args:
            e_type (str): type of the map element, e.g., TrafficSign, TrafficLight, Pole
        """
        self.e_type = e_type                    # (str) element type
        self.id = None                      # (int) identifier
        self.deviation_extension = None     # (str) temporary, experimental, inconsistent, negated

    @abstractmethod
    def _get_position(self):
        pass


class PointMapElement(MapElement):
    def __init__(self, e_type):
        super().__init__(e_type)

        self.lat = None     # (float) latitude (WGS84), degree
        self.lon = None     # (float) longitude (WGS84), degree
        self.x_utm = None   # (float) x coordinate in UTM32N [meter]
        self.y_utm = None   # (float) y coordinate in UTM32N [meter]
        self.z_utm = None   # (float) z coordinate in UTM32N, reference ellipsoid

    def _get_position(self):
        return [self.lat, self.lon]


class LineMapElement(MapElement):
    def __init__(self, e_type):
        super().__init__(e_type)

        self.points_wgs = None  # (list) points in WGS84 [[lat, lon, z], [lat, lon, z], ...]
        self.points_utm = None  # (list) of points in UTM32N [[x, y, z], [x, y, z], [x, y, z]]

    def _get_position(self):
        return self.points_wgs


class TrafficSign(PointMapElement):
    def __init__(self):
        e_type = 'TrafficSign'
        super().__init__(e_type)

        self.yaw_utm = None     # (float) orientation value in degree
        self.cls = None         # (str) classification according to german traffic sign catalog
        self.sub_cls = None     # (str) "
        self.shape = None       # (str) shape of traffic sign, e.g., rectangle, circle, etc.
        self.height = None      # (float) height of bounding rectangle
        self.width = None       # (float) width of bounding rectangle


class TrafficLight(PointMapElement):
    def __init__(self):
        e_type = 'TrafficLight'
        super().__init__(e_type)

        self.cls = None                 # (str) classification: vehicle, pedestrian, warning
        self.height = None              # (float) height of bounding box [meter]
        self.width = None               # (float) width of bounding box [meter], refers to width of light box face
        self.depth = None               # (float) depth of bounding box, 0.1 is a standard value [meter]
        self.yaw_utm = None             # (float) orientation of bounding box
        self.has_red = None             # (bool) true if traffic light has red light
        self.has_green = None           # (bool) true if traffic light has green light
        self.has_yellow = None          # (bool) true if traffic light has yellow light
        self.orientation_cls = None     # (str) vertical or horizontal (bounding box)
        self.signal_cls = None          # (str) bike, pedestrian, round, etc.


class Pole(PointMapElement):
    def __init__(self):
        e_type = 'Pole'
        super().__init__(e_type)

        self.cls = None         # (str) pole classification, also see deep_learning.util.lookups
        self.diameter = None    # (float) pole diameter [meter]


class RoadFurniture(LineMapElement):
    def __init__(self):
        e_type = 'RoadFurniture'
        super().__init__(e_type)

        self.height = None  # (float) standard value depending on class
        self.width = None   # (float) standard width depending on class
        self.cls = None     # (str) class of road furniture, class for CS_Obstacles_Line, e.g., fence


class Lane(LineMapElement):
    def __init__(self):
        e_type = 'Lane_Ordinary'  # (str) or Lane_Temporary
        super().__init__(e_type)

        self.connection_type = None     # (str) continuation, split, merge
        self.width = None               # (float) lane width [meter]
        self.successors = None          # (list[int]) succeeding lane ids


class Marking(LineMapElement):
    def __init__(self):
        e_type = 'Marking'
        super().__init__(e_type)

        self.cls = None         # (str) class, e.g., text, symbol, ...
        self.color = None       # (str) white, yellow


class Relation:
    def __init__(self):
        self.e_type = 'Relation'
        self.deviation_extension = None  # (str) temporary, experimental, inconsistent, negated

        self.points_wgs = None  # (list) points in WGS84 [(lat, lon), (lat, lon), ...]
        self.points_utm = None  # (list) of points in UTM32N [(x, y), (x, y), (x, y)]

        self.id = None              # (int) identifier
        self.cls = None             # (str) classification: is_traffic_sign_for, is_traffic_light_for, ...
        self.main_object = None     # (int) id of traffic sign or light, ....
        self.sec_object = None      # (int | list[int]) id (or ids) of secondary objects, e.g., lane or lanes


class ConstructionSite:
    def __init__(self):
        self.e_type = 'Construction_Site'
        self.id = None          # (int) identifier
        self.points_utm = None  # (list) of points in UTM32N [(x, y), (x, y), (x, y)]
