#!/usr/bin/env python3
"""Pasiphae survey planner.
"""

from abc import ABCMeta, abstractmethod
import json

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

class Writer(object, metaclass=ABCMeta):
    """Abstract schedule writer class."""

    #--------------------------------------------------------------------------
    @abstractmethod
    def __init__(self):
        """Create Writer instance.

        Returns
        -------
        None

        Notes
        -----
        This is an abstract method. Each Writer child class may have its
        own arguments to set it up.
        """

        pass

    #--------------------------------------------------------------------------
    @abstractmethod
    def write(self, fields, filename):
        """Write schedule.

        Returns
        -------
        None

        Notes
        -----
        This is an abstract method. Each Writer child class will have its own
        implementation how the fields are written into a schedule.
        """

        pass

#==============================================================================

class WriterJSON(object, metaclass=ABCMeta):
    """Schedule writer that converts the fields list of dict directly into a
    JSON format."""

    pass

#==============================================================================

class WriterJSONPasiphae(object, metaclass=ABCMeta):
    """Schedule writer."""

    pass

#==============================================================================
