""" Visualization of map elements.
"""
import random
from typing import List, Union

import mayavi.mlab as mlab
import numpy as np

from deep_learning.util import lookups
from utility import transformations as trafo
import visualization.visualization_basics as vis_basics
from dataset.map_deviation import DeviationTypes, MapDeviation

# Class definitions ####################################################################################################
########################################################################################################################


class Pole:
    """ Visualization class for poles.
    """
    def __init__(self, x, y, z, radius, cls=None, height=2, **kwargs):
        """ All bounding shape parameters in meter.
        Args:
            x (float): center coordinate (base point)
            y (float): center coordinate (base point)
            z (float): center coordinate (base point)
            radius (float): radius of pole
            cls (str): classification (e.g., 'lamppost')
            height (float): height of pole (as per default value, see deep_learning.util.lookups)
            **kwargs: additional arguments
        """
        self.x = x
        self.y = y
        self.z = z
        self.cls = cls
        self.height = height
        self.radius = radius
        self.kwargs = kwargs

    def plot(self):
        coordinates = np.array([[self.x, self.y, self.z], [self.x, self.y, self.z + self.height]])
        mlab.plot3d(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], tube_radius=self.radius, **self.kwargs)

        if self.cls:
            mlab.text3d(self.x, self.y, self.z + self.height, self.cls, scale=.2)


class TrafficLight:
    """ Visualization class for traffic lights.
    """
    def __init__(self, x, y, z, width, height, yaw, cls=None, depth=.01, color=None, opacity=1.0, arrow_scale=3):
        """ All parameters define the light's faceplate (rectangle).

        Using a default depth, the faceplate is visualized as bounding box. All shape parameters in meter.

        Args:
            x (float): center coordinate
            y (float): center coordinate
            z (float): center coordinate
            width (float): width
            height (float): height
            yaw (float): orientation
            cls (str): classification (e.g., vehicle, people, warning)
            depth (float): default depth for displaying a box
            color ((float, float, float)): color of box
            opacity (float): 1.0 = no opacity
            arrow_scale (float): relative length of arrow
        """
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.depth = depth
        self.height = height
        self.yaw = yaw
        self.color = color if color is not None else (1, 0, 0)
        self.opacity = opacity
        self.arrow_scale = arrow_scale * 2  # scale 1 is only half the cube size, hence '* 2'
        self.cls = cls
        self.cube = vis_basics.CuboidPlot(x, y, z, depth, width, height, yaw, color=color, opacity=opacity)

    def plot(self):
        """ Plots the light's bounding shape as box.
        """
        self.cube.plot()
        # Plot arrow
        # Get orientation vector for arrow by rotating a point in front of cube
        ori_vec = np.array([[self.width / 2], [0], [0]])
        ori_vec = trafo.rotate_points(ori_vec, self.yaw)
        ori_vec = ori_vec.T.squeeze(axis=0)
        mlab.quiver3d(self.x, self.y, self.z, ori_vec[0], ori_vec[1], ori_vec[2], color=self.color,
                      scale_factor=self.arrow_scale, mode='arrow', opacity=self.opacity)

        if self.cls:
            mlab.text3d(self.x, self.y, self.z + self.height, self.cls, scale=.2)


class TrafficSign:
    """ Visualization class for traffic signs.
    """
    def __init__(self, x, y, z, cls, width, height, yaw, color=None, opacity=1.0, show_arrow=True):
        """ All parameters define the signs bounding shape (rectangle) in meter.

        Args:
            x (float): center coordinate
            y (float): center coordinate
            z (float): center coordinate
            width (float): width
            height (float): height
            yaw (float): orientation
            cls (str): classification (e.g., circle, rectangle, ..)
            color ((float, float, float)): RGB color of plot
            opacity (float): 1.0 = no opacity
            show_arrow (bool): draws an arrow indicating the sign's front
        """

        self.x = x
        self.y = y
        self.z = z
        self.cls = cls

        self.width = width
        self.height = height
        self.yaw = yaw
        self.color = color if color is not None else (1, 0, 0)
        self.opacity = opacity

        self.show_arrow = show_arrow
        self.arrow_length = 1
        self.arrow_scale = 1

    def plot(self):
        """ Plots the sign's bounding shape.
        """
        # Define corner points
        bot_1 = np.array([self.width / 2, 0, -self.height / 2])
        bot_2 = np.array([-self.width / 2, 0, -self.height / 2])
        top_1 = np.array([self.width / 2, 0, self.height / 2])
        top_2 = np.array([-self.width / 2, 0, self.height / 2])

        ori = np.array([0, 0 + self.arrow_length, 0])
        ori = np.expand_dims(ori, axis=1)

        # Rotate in VRF
        points = np.column_stack((bot_1, top_1, top_2, bot_2, bot_1))  # closed rectangle
        points = trafo.rotate_points(points, self.yaw)
        ori = trafo.rotate_points(ori, self.yaw)

        # Shift rotated points
        points = points.T + np.array([self.x, self.y, self.z])
        mlab.plot3d(points[:, 0], points[:, 1], points[:, 2], color=self.color, opacity=self.opacity)

        if self.show_arrow:
            mlab.quiver3d(self.x, self.y, self.z, ori[0], ori[1], ori[2],
                          color=self.color,
                          scale_factor=self.arrow_scale,
                          mode='arrow')
        if self.cls:
            mlab.text3d(self.x, self.y, self.z + self.height, self.cls, scale=.2)


class RoadFurniture:
    """ Visualization class for road furniture (CS_Obstacles_Line, e.g., fences).
    """
    def __init__(self, points, cls, width, height, color=None):
        """ All shape parameters in meter.
        Args:
            points (list[list]): list of coordinates (UTM or vehicle reference frame VRF)) [[x,y,z], ...]
            cls (str): classification, e.g., barrier
            width (float): width of object
            height (float): height of object
            color ((float, float, float)): RGB color of plot
        """
        self.points = points
        self.cls = cls  # classification, e.g., shape

        self.width = width
        self.height = height
        self.color = color if color is not None else (1, 0, 0)

    def plot(self):
        """ Plots the object's bounding shape.
        """
        for i_point in range(1, len(self.points)):
            point_1 = self.points[i_point - 1]
            point_2 = self.points[i_point]

            # Compute middle point
            point_1 = np.array(point_1)
            point_2 = np.array(point_2)
            vec_12 = point_2 - point_1
            mid = point_1 + .5 * vec_12
            length = np.linalg.norm(vec_12)
            yaw = np.degrees(np.arctan2(vec_12[1], vec_12[0]))

            vis_basics.draw_cube(mid[0], mid[1], mid[2], length, self.width, self.height, yaw, self.color, opacity=.4)

        if self.cls:
            mlab.text3d(self.points[0][0], self.points[0][1], self.points[0][2] + self.height, self.cls, scale=.2)


class Lane:
    """ Visualization class for lanes.
    """
    def __init__(self, points, cls, width=None, color=None):
        """ All shape parameters in meter.
        Args:
            points (list[list]): list of coordinates (UTM or vehicle reference frame VRF)) [[x,y,z], ...]
            cls (str): classification of lane (e.g., temporary)
            width (float): lane width
            color ((float, float, float)): RGB color of plot
        """
        self.points = points
        self.cls = cls
        self.width = width
        self.color = color if color is not None else (1, 0, 0)
        self.text_height = 1
        self.elevation = .3     # elevation of lane plot above ground points

    def plot(self):
        """ Plots the lane.
        """
        mlab.plot3d(self.points[:, 0], self.points[:, 1], self.points[:, 2] + self.elevation, color=self.color)

        if self.cls:
            if len(self.points) > 2:
                i_point = int(np.floor(len(self.points) / 2))
            else:
                i_point = 0

            mlab.text3d(self.points[i_point][0], self.points[i_point][1], self.points[i_point][2] + self.text_height,
                        self.cls, scale=.2)


class Marking:
    """ Visualization class for markings (lines or polygons).
    """
    def __init__(self, points, cls, color=None):
        """ All shape parameters in meter.
        Args:
            points (list[list]): list of coordinates (UTM or vehicle reference frame VRF)) [[x,y,z], ...]
            cls (str): classification of marking
            color ((float, float, float)): RGB color of plot
        """
        self.points = points
        self.cls = cls
        self.color = color if color is not None else (1, 0, 0)
        self.text_height = 1    # elevation of text
        self.elevation = 0.05

    def plot(self):
        mlab.plot3d(self.points[:, 0], self.points[:, 1], self.points[:, 2] + self.elevation, color=self.color)

        if self.cls:
            mlab.text3d(self.points[0][0], self.points[0][1], self.points[0][2] + self.text_height, self.cls, scale=.2)


class Relation:
    """ Visualization class for relations.
    """
    def __init__(self, points, cls, color=None):
        """ All shape parameters in meter.
        Args:
            points (list[list]): list of souce and target coordinate [[x,y,z], ...]
            cls (str): classification of relation, e.g., is_traffic_light_for
            color ((float, float, float)): RGB color of plot
        """
        self.points = points
        self.cls = cls
        self.color = color if color is not None else (1, 0, 0)
        self.text_height = 1
        self.elevation = .3

    def plot(self):
        mlab.plot3d(
            self.points[:, 0], self.points[:, 1], self.points[:, 2] + self.elevation, color=self.color)

        if self.cls:
            mlab.text3d(self.points[0][0], self.points[0][1], self.points[0][2] + self.text_height, self.cls, scale=.2)


# Module functions #####################################################################################################
########################################################################################################################


def vis_map_elements(map_elements: List[Union[dict, MapDeviation]], vrf=True, is_prediction=False, is_evaluation=False,
                     display_class=False, color_by_dev_extension=False, color_by_dev_state=True, color_by_eval_result=True):
    """ Wrapper function visualizing signs, lights, or poles (both for object and deviation detection).

    If deviations are provided (comprising both prior and current element), the most recent element
    (found in sensor data) is visualized, except for insertions (only comprising a prior element).

    Args:
        map_elements (list[dict] | list[MapDeviation]): list of elements or deviations to visualize
        vrf (bool): indicates if coordinates are in VRF
        is_prediction (bool): indicates if provides objects are predictions
        is_evaluation (bool): indicates if visualization by TP, FP, or FN status is desired
        display_class (bool): display the element's classification
        color_by_dev_extension (bool): color by the element's deviation state (temporary, experimental, ...)
    """
    # Check if object or deviation task
    deviation_detection = False
    if len(map_elements) > 0 and isinstance(map_elements[0], MapDeviation):
        deviation_detection = True

    # Visualize elements
    pole_height_lut = lookups.create_pole_height_lut()
    for elem in map_elements:
        deviation_type = None
        if deviation_detection:
            deviation_type = elem.deviation_type
            is_prediction = elem.is_prediction
            # occlusion_type = elem.occlusion_type
            elem = elem.get_most_recent() if is_evaluation else elem.get_least_recent()

        elem_type = infer_element_type(elem)
        if vrf:
            x = elem['x_vrf']
            y = elem['y_vrf']
            z = elem['z_vrf']
            yaw = elem['yaw_vrf'] if 'yaw_vrf' in elem else None
        else:
            x = elem['x_utm']
            y = elem['y_utm']
            z = elem['z_utm']
            yaw = -elem['yaw_utm'] if 'yaw_utm' in elem else None

        size = elem[lookups.get_main_size_features()[elem_type]]

        if elem_type == 'poles':
            pole_cls = elem['cls'] if 'cls' in elem else elem['class']
            if pole_cls:
                height = pole_height_lut[pole_cls]
            else:
                height = 3.0
        else:
            height = elem['height']

        color = get_element_color(elem, is_prediction, deviation_type, color_by_dev_state, color_by_eval_result)

        cls = None
        # if occlusion_type == 'bottom' and elem_type == 'poles':
        #     cls = f"{elem_type} | {occlusion_type}"
        if display_class:
            cls = f"{elem['id']}"
            cls += " | " + elem['cls']
            if elem_type == 'poles':
                cls += f" | {elem['diameter'] * 100:.2f} cm"
            elif elem_type == 'signs':
                # cls += f" | {elem['cls']} | {elem['sub_cls']}"
                cls += f" | {elem['yaw_utm']:.2f}"
                # cls += f" | {elem['shape']}"
            elif elem_type == 'lights':
                cls += f" | {elem['signal_cls']}"
                cls += f" | {elem['orientation_cls']}"
                cls += f" | {elem['yaw_utm']:.2f}"
                # cls += f" | red: {elem['has_red']}"
                # cls += f" | yellow: {elem['has_yellow']}"
                # cls += f" | green: {elem['has_green']}"

                if elem['cls'] == 'pedestrian':
                    color = (0, 0, 1)  # blue
                elif elem['cls'] == 'vehicle':
                    color = (0, 1, 1)  # cyan
                if not elem['has_green']:
                    color = (1, 0, 1)  # purple
                    elem['cls'] = 'warning'

        # deviation_extension is None or a string if extension is existing
        if color_by_dev_extension and elem['deviation_extension'] is not '':
            color = get_color_from_deviation_extension(elem)
            if cls:
                cls += ' | ' + elem['deviation_extension']
            else:
                cls = elem['deviation_extension']

        # Plot TP and FP (predicted map elements)
        opacity = 0.4 if elem_type == 'poles' else 1.0
        label_text = None
        label_color = None
        if deviation_detection and is_evaluation and is_prediction:
            if 'eval_class' not in elem.keys():
                raise KeyError("Error: Element without eval_class found.")
            if elem['eval_class'] == 'TP':
                label_text = f"{deviation_type.value} {elem_type[:-1]} [TP]"
                # label_text = None
                label_color = (0, 1, 0)  # green
            elif elem['eval_class'] == 'FP':
                label_color = (1, 0, 0)  # red
                # color = (.7, .7, .7)     # gray
                # opacity = .5
                label_text = f"{deviation_type.value} {elem_type[:-1]} [FP]"

        # Plot FN (GT elements) -> draw only elements without set "mapped_flag" (not associated with a prediction = FN)
        if deviation_detection and is_evaluation and not is_prediction:
            if "mapped_flag" in elem:
                # color = (0, 0, 1)
                continue    # skip associated GT elements
            else:
                # label_text = f"{deviation_type.value} {elem_type[:-1]} [FN]"
                label_color = (1, 0, 0)     # red
                color = (1, 1, 1)           # element color (white)

        # label_text = None
        if label_text is not None:
            if elem_type == 'poles':
                label_pos = [x, y, z+height]
            else:
                if elem['eval_class'] == 'FN':
                    label_pos = np.array([size/2+.05, 0, -height/2-.05])
                else:
                    label_pos = np.array([size/2+.05, 0, height/2+.05])
                label_pos = trafo.rotate_points(label_pos, yaw)
                label_pos = label_pos.T + np.array([x, y, z])

            label_pos[2] += random.uniform(-.2, .2)
            mlab.text3d(label_pos[0], label_pos[1], label_pos[2], label_text, scale=.2, color=label_color)

        if elem_type == 'poles':
            vis_elem = Pole(x, y, z, size / 2, cls, height, tube_sides=8, opacity=opacity, color=color)
        elif elem_type == 'signs':
            vis_elem = TrafficSign(x, y, z, cls, size, height, yaw, color=color, opacity=opacity, show_arrow=False)
        elif elem_type == 'lights':
            vis_elem = TrafficLight(x, y, z, size, height, yaw, cls=cls, depth=size, color=color, opacity=opacity)
        else:
            raise ValueError(f"Not defined for element_type '{elem_type}'")

        vis_elem.plot()


def infer_element_type(elem: dict) -> str:
    """ Infers the map element type according to the dict's keys.
    Args:
        elem (dict): element to check
    Returns:
        (str): group name (lights, signs, poles)
    """

    if 'type' in elem.keys():
        _, elem_type_lut = lookups.get_element_type_naming_lut()
        elem_type = elem_type_lut[elem['type']]
        return elem_type
    elif 'size' in elem.keys():
        return 'lights'
    elif 'diameter' in elem.keys():
        return 'poles'
    elif 'width' in elem.keys():
        return 'signs'
    else:
        raise ValueError(f"Cannot infer element_type from element: {elem}")


def get_element_color(
        elem: dict = None, is_prediction=False, deviation_type=None, color_by_dev_state=True, color_by_eval_result=True):
    """ Provides the color for visualization (e.g., according to the evaluation state)
    Args:
        elem (dict): element to color
        is_prediction (bool): indicates if element is predicted (or ground truth (GT))
        deviation_type (DeviationTypes): evaluation state (VER, DEL, INS, SUB)
        color_by_dev_state (bool): if True, color deviation states differently (otherwise black)

    Returns:
        color ((float), (float), (float)): RGB tuple
    """

    color = (1, 0, 0)  # red as default

    # Color according to evaluation state (deviation_type)
    if deviation_type is not None:
        if color_by_dev_state:
            color = {
                DeviationTypes.VERIFICATION.value: (0, 1, 0),   # green
                DeviationTypes.DELETION.value: (1, 0.5, 0),     # orange
                DeviationTypes.INSERTION.value: (1, 0, 0),      # red
                DeviationTypes.SUBSTITUTION.value: (0, 0, 1),   # blue
            }[deviation_type.value]
        else:
            color = (0, 0, 0)
        return color

    # Color according to association state (TP, FP, FN)
    if elem is not None and color_by_eval_result:
        if is_prediction:
            if 'eval_class' in elem.keys():
                color = {
                    'TP': (0, 1, 0),  # green
                    'FP': (0, 1, 1),  # cyan
                }[elem['eval_class']]
            else:
                base_val = 0.3
                intensity = min(1, base_val + elem['score'] * 2)
                color = (intensity, intensity, intensity)  # grey
        elif 'mapped_flag' in elem.keys():
            color = (1, 1, 1)  # white

    return color


def get_color_from_deviation_extension(element):
    """ Provides the visualization color according to the element's deviation extension.
    Args:
        element (dict): element as dictionary

    Returns:
        color ((float), (float), (float)): RGB tuple
    """
    color = (0, 0, 1)
    if element['deviation_extension'] == 'temporary':
        color = (1, .5, 0)
    if element['deviation_extension'] == 'experimental':
        color = (1, 0, 0)
    if element['deviation_extension'] == 'inconsistent':
        color = (1, 0, 0)
    if element['deviation_extension'] == 'negated':
        color = (1, 0, 0)

    return color


def vis_road_furniture_objects(road_furniture_objects, vrf=True, color_by_flag=False, color_by_dev_extension=False,
                               display_class=False):
    """ Visualizes line objects such as road furniture objects.

    Args:
        road_furniture_objects (list[dict]): list of road furniture objects (see map_elements_definition.py) as dicts
        vrf (bool): indicates if coordinates are in VRF
        color_by_flag (bool): color by association state (TP, FP, FN)
        display_class (bool): display the element's classification
        color_by_dev_extension (bool): color by the element's deviation state (temporary, experimental, ...)
    """

    visus_road_furniture = []

    for road_furniture in road_furniture_objects:
        if vrf:
            points = None
            raise ValueError("VRF visualization not supported for road furnitures!")
        else:
            points = road_furniture['points_utm']

        width = road_furniture['width']
        height = road_furniture['height']

        color = (0, 0, 1)
        if road_furniture['cls'] == 'CURB_TRAVERSABLE':
            color = (0, 1, 0)
        elif road_furniture['cls'] == 'CURB':
            color = (1, 0, 0)

        if color_by_flag:
            if 'mapped_flag' not in road_furniture.keys():
                color = (1, 0, 0)

        cls = None
        if display_class:
            cls = str(road_furniture['id'])
            cls += ' | ' + road_furniture['cls']

        if color_by_dev_extension and road_furniture['deviation_extension'] is not '':
            color = get_color_from_deviation_extension(road_furniture)
            if cls is None:
                cls = road_furniture['deviation_extension']
            else:
                cls += ' | ' + road_furniture['deviation_extension']

        vis_road_furniture = RoadFurniture(points, cls, width, height, color=color)
        vis_road_furniture.plot()
        visus_road_furniture.append(vis_road_furniture)

    return visus_road_furniture


def vis_lanes(lanes, vrf=True, color_by_flag=False, color_by_dev_extension=False, display_class=False):
    """ Visualizes lanes.

    Args:
        lanes (list[dict]): list of type lane (see map_elements_definition.py) as dicts
        vrf (bool): indicates if coordinates are in VRF
        color_by_flag (bool): color by association state (TP, FP, FN)
        display_class (bool): display the element's classification
        color_by_dev_extension (bool): color by the element's deviation state (temporary, experimental, ...)
    """

    visus_lanes = []
    for lane in lanes:
        if vrf:
            points = None
            raise ValueError("VRF visualization not supported for lanes!")
        else:
            points = lane['points_utm']

        color = (0, 1, 1)
        if color_by_flag:
            if 'mapped_flag' not in lane.keys():
                color = (1, 0, 0)

        cls = None
        if display_class:
            cls = str(lane['id']) + '\n'
            cls += ' | ' + lane['connection_type'] + '\n'
            cls += ' | ' + str(lane['width']) + '\n'
            cls += ' | ' + str(lane['successors']) + '\n'

        if color_by_dev_extension and lane['deviation_extension'] is not '':
            color = get_color_from_deviation_extension(lane)
            if cls is None:
                cls = lane['deviation_extension']
            else:
                cls += ' | ' + lane['deviation_extension']

        vis_lane = Lane(points, cls, width=lane['width'], color=color)
        vis_lane.plot()
        visus_lanes.append(vis_lane)

    return visus_lanes


def vis_markings(markings, vrf=True, color_by_flag=False, color_by_dev_extension=False, color_by_color=False,
                 display_class=False):
    """ Visualizes markings.

    Note: In Germany, temporary or experimental markings are colored yellow on the road.

    Args:
        markings (list[dict]): list of type lane (see map_elements_definition.py) as dicts
        vrf (bool): indicates if coordinates are in VRF
        color_by_flag (bool): color by association state (TP, FP, FN)
        display_class (bool): display the element's classification
        color_by_color (bool): display by the marking's color (white or yellow)
        color_by_dev_extension (bool): color by the element's deviation state (temporary, experimental, ...)
    """

    visus_markings = []
    for _, marking in enumerate(markings):
        if vrf:
            points = None
            raise ValueError("VRF visualization not supported for markings!")
        else:
            points = marking['points_utm']

        color = (0, 0, 1)
        if color_by_flag:
            if 'mapped_flag' not in marking.keys():
                color = (1, 0, 0)

        cls = None
        if display_class:
            cls = str(marking['id'])
            cls += ' | ' + marking['cls']
            cls += ' | ' + marking['color']

            if marking['deviation_extension'] is not '':
                cls += ' | ' + marking['deviation_extension']

        if color_by_color:
            if marking['color'] == 'yellow':
                color = (1, .8, 0)
            else:
                color = (0, 1, 1)

        if color_by_dev_extension and marking['deviation_extension'] is not '':
            color = get_color_from_deviation_extension(marking)

        vis_marking = Marking(points, cls, color=color)
        vis_marking.plot()
        visus_markings.append(vis_marking)

    return visus_markings


def vis_relations(relations, vrf=True, color_by_class=False, display_class=False):
    """ Visualizes relations.

    Args:
        relations (list[dict]): list of type lane (see map_elements_definition.py) as dicts
        vrf (bool): indicates if coordinates are in VRF
        color_by_class (bool): color by relation class
        display_class (bool): display the element's classification
    """

    visus_relations = []
    for relation in relations:
        if vrf:
            points = None
            raise ValueError("VRF visualization not supported for relations!")
        else:
            points = relation['points_utm']

        color = (0, 0, .5)
        if color_by_class:
            if relation['cls'] == 'is_traffic_sign_for':
                color = (0, 0, 1)  # blue
            if relation['cls'] == 'is_traffic_light_for':
                color = (.8, .2, 0)  # red
            if relation['cls'] == 'is_negation_for':
                color = (.1, .5, 0)  # orange

        cls = None
        if display_class:
            cls = str(relation['id'])
            cls += ' | ' + relation['cls']

        vis_relation = Relation(points, cls, color=color)
        vis_relation.plot()
        visus_relations.append(vis_relation)

    return visus_relations
