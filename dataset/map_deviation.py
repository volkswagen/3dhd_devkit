""" Module containing map deviation related specifications """

from enum import Enum
from typing import List, Union

from deep_learning.util.lookups import get_element_type_naming_lut


class DeviationTypes(Enum):
    """ Specified map deviation types """
    # Note: Do not change order.
    VERIFICATION = 'VER'
    INSERTION = 'INS'
    DELETION = 'DEL'
    SUBSTITUTION = 'SUB'


class MapDeviation:
    """ Class containing all relevant information regarding a map deviation. Most essentially, it comprises the type of
        deviation and the given map element in its prior and current state.

    Attributes:
          deviation_type (DeviationTypes):
          type_prior (str): map element type of __element_prior ('lights', 'poles', or 'signs')
          type_current (str): map element type of __element_current ('lights', 'poles', or 'signs')
          __element_prior (dict | None): corresponding map element in deviating map or None if no element exists
          __element_current (dict | None): corresponding map element in updated map or None if no element exists
          is_prediction (bool): true if the MapDeviation is a prediction, false if it is ground-truth
          occlusion_type (str | None): occlusion type if specified ('top', 'bottom', 'left', 'right'), otherwise None
    """
    def __init__(self,
                 deviation_type: DeviationTypes,
                 element_prior: Union[dict, None],
                 element_current: Union[dict, None],
                 is_prediction=False):
        """ Creates a MapDeviation object based on the given settings.

        Args:
            deviation_type: deviation type of the MapDeviation (VER, DEL, INS, SUB)
            element_prior: map element in deviating map (should be None for DEL)
            element_current: corresponding map element in updated map (should be None for INS)
            is_prediction: if true, denotes a predicted MapDeviation (false --> ground-truth MapDeviation)
        """

        self.deviation_type = deviation_type

        _, type_lut = get_element_type_naming_lut()  # maps to {'lights', 'poles', 'signs'}

        self.type_prior = element_prior['type'] if element_prior is not None else element_current['type']
        self.type_prior = type_lut[self.type_prior]
        self.type_current = element_current['type'] if element_current is not None else element_prior['type']
        self.type_current = type_lut[self.type_current]

        self.__element_prior = element_prior.copy() if element_prior is not None else None
        self.__element_current = element_current.copy() if element_current is not None else None

        self.is_prediction = is_prediction

        self.occlusion_type = None

    def get_prior(self) -> Union[dict, None]:  # don't return copy bc some keys are added during evaluation
        return self.__element_prior

    def get_current(self) -> Union[dict, None]:
        return self.__element_current

    def get_most_recent(self) -> dict:
        return self.__element_current if self.__element_current is not None else self.__element_prior

    def get_least_recent(self) -> dict:
        return self.__element_prior if self.__element_prior is not None else self.__element_current


def get_deviation_classes(include_aggregate_classes=True) -> List[str]:
    """ Get all available deviation classes as strings

    Args:
        include_aggregate_classes: if true, also adds "Deviating", "With-Prior", and "All" to list of classes

    Returns: list of deviation classes as strings
    """
    classes = []
    for dev_type in DeviationTypes:
        classes.append(dev_type.value)
    if include_aggregate_classes:
        classes += ['Deviating', 'With-Prior', 'All']
    return classes
