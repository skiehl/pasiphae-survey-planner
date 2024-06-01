#!/usr/bin/env python3
"""Pasiphae survey planner.
"""

from abc import ABCMeta, abstractmethod
from astropy.coordinates import SkyCoord
#from astropy.time import Time, TimeDelta
#from astropy import units as u
#from itertools import repeat
#from multiprocessing import Manager, Pool, Process
import numpy as np
#from pandas import DataFrame
#from statsmodels.api import add_constant, OLS
#from time import sleep
#from textwrap import dedent
#
#import constraints as c
from db import FieldManager
#from utilities import true_blocks

__author__ = "Sebastian Kiehlmann"
__credits__ = ["Sebastian Kiehlmann"]
__license__ = "BSD3"
__version__ = "0.1"
__maintainer__ = "Sebastian Kiehlmann"
__email__ = "skiehlmann@mail.de"
__status__ = "Production"


#==============================================================================
# CLASSES
#==============================================================================

class Prioritizer(object, metaclass=ABCMeta):
    """An class to assign priorities to a list of fields.
    """

    label = "Unique label for each type of prioratizer."

    #--------------------------------------------------------------------------
    @abstractmethod
    def __init__(self):
        """Create Prioratizer instance.

        Returns
        -------
        None

        Notes
        -----
        This is an abstract method. Each Prioratizer child class may have its
        own arguments to set it up.
        """

        pass

    #--------------------------------------------------------------------------
    @abstractmethod
    def prioratize(self, fields):
        """Assign priorities to fields.

        Arguments
        ---------
        fields : list of dict
            List of field dictionaries as returned by
            surveyplanner.Surveyplanner.get_fields().

        Returns
        -------
        priority : numpy.ndarray
            Each entry is a priority in the range [0, 1] corresponding to one
            field in the input field list.

        Notes
        -----
        This is an astract method. Each Prioratizer child class requiers a
        specific implementation of this method with the same argument and
        return variable.
        """

        pass

#==============================================================================
