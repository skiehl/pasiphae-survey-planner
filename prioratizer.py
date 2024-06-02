#!/usr/bin/env python3
"""Pasiphae survey planner.
"""

from abc import ABCMeta, abstractmethod
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
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
        self.label : str
            The name of this prioritizer.

        Notes
        -----
        This is an astract method. Each Prioratizer child class requiers a
        specific implementation of this method with the same argument and
        return variable.
        """

        priority = np.zeros(len(fields)) # to be defined

        return priority, self.label

#==============================================================================

class PrioritizerSkyCoverage(Prioritizer):
    """Assign priorities to fields based on coverage around the fields.
    """

    label = 'SkyCoverage'

    #--------------------------------------------------------------------------
    def __init__(self, dbname, radius, full_sky=False, normalize=False):
        """Create PrioratizerSkyCoverage instance.

        Parameters
        ----------
        dbname : str
            Database name. Required to query field coordinates.
        radius : astropy.coordinates.Angle
            Defines the radius within which other fields are considered
            neighbors.
        full_sky : bool, optional
            If True, all fields regardless of the telescope association are
            considered for the count of the coverage. Otherwise, only fields
            associated with the same telescope as the fields to prioritize are
            considered. The default is False.
        normalize : bool, optional
            If True, the priorities are rescaled such that the maximum priority
            is 1. Otherwise, the maximum may be <=1. The default is False.

        Raises
        ------
        ValueError
            Raised, if `radius` is not astropy.coordinates.Angle.

        Returns
        -------
        None
        """

        # check input:
        if not isinstance(radius, Angle):
            raise ValueError("`radius` must be astropy.coordinates.Angle.")

        self.dbname = dbname
        self.radius = radius
        self.full_sky = full_sky
        self.normalize = normalize

    #--------------------------------------------------------------------------
    def _query_fields(self, telescope):
        """Get field coordinates and number of pending observations from
        database.

        Parameters
        ----------
        telescope : str
            Name of the telescope of current interest.

        Returns
        -------
        fields_coord : astropy.coordinates.SkyCoord
            Center coordinates of the queried fields.
        nobs_pending : numpy.ndarray (dtype: int)
            Number of pending observations of the queried fields.

        Notes
        -----
        If self.full_sky is True, the specified telescope is irrelevant. All
        active fields are queried. Otherwise, only active fields associated
        with the specified field are queried.
        """

        # query fields for specific telescope or all fields:
        if self.full_sky:
            telescope = None

        # database connection:
        manager = FieldManager(self.dbname)

        # get all active fields:
        fields = manager.get_fields(telescope=telescope)

        # extract coordinates and number of pending observations:
        fields_ra = []
        fields_dec = []
        nobs_pending = []

        for field in fields:
            fields_ra.append(field['center_ra'])
            fields_dec.append(field['center_dec'])
            nobs_pending.append(field['nobs_pending'])

        fields_coord = SkyCoord(fields_ra, fields_dec, unit='rad')
        nobs_pending = np.array(nobs_pending)

        return fields_coord, nobs_pending

    #--------------------------------------------------------------------------
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
        self.label : str
            The name of this prioritizer.

        Notes
        -----
        Assign a higher priority to fields in a neighborhood of fields that is
        closer to being finished. The size of the neighborhood depends on the
        `radius` set at class instanciation.
        """

        # get coordinates and number of pending observations for all active
        # fields:
        fields_coord, nobs_pending = self._query_fields(fields[0]['telescope'])

        n_tot = np.zeros(len(fields), dtype=int)
        n_done = np.zeros(len(fields), dtype=int)

        # iterate through fields:
        for i, field in enumerate(fields):
            # identify neighbors:
            field_coord = SkyCoord(
                    field['center_ra'], field['center_dec'], unit='rad')
            sel = fields_coord.separation(field_coord) <= self.radius

            # count neighbors and finished neighbors:
            n_tot[i] = np.sum(sel)
            n_done[i] = np.sum(nobs_pending[sel] == 0)

        # calculate priority:
        n_tot_max = np.max(n_tot)
        coverage_local = (n_done + 1) / (n_tot + 1)
        coverage_max = (n_done + 1) / (n_tot_max + 1)
        weight1 = n_done / n_tot
        weight2 = 1 - weight1
        priority = coverage_local * weight1 + coverage_max * weight2

        if self.normalize:
            priority /= priority.max()

        return priority, self.label

#==============================================================================

class PrioritizerFieldStatus(Prioritizer):
    """Assign priorities to fields based on the field observability status.
    """

    label = 'FieldStatus'

    #--------------------------------------------------------------------------
    def __init__(self, rising=False, plateauing=False, setting=False):
        """Create PrioratizerSkyCoverage instance.

        Parameters
        ----------
        rising : bool, optional
            If True, prioritize a field if it is rising. The default is False.
        plateauing : bool, optional
            If True, prioritize a field if it is plateauing. The default is
            False.
        setting : bool, optional
            If True, prioritize a field if it is setting. The default is False.

        Returns
        -------
        None
        """

        self.rising = bool(rising)
        self.plateauing = bool(plateauing)
        self.setting = bool(setting)

    #--------------------------------------------------------------------------
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
            Each entry is a priority of either 0 or 1 corresponding to one
            field in the input field list.
        self.label : str
            The name of this prioritizer.

        Notes
        -----
        Assign a priority of 1 to fields that are rising and/or plateauing
        and/or setting, depending on the parameters set at class instanciation.
        Otherwise, the priority is 0.
        """

        priority = np.zeros(len(fields))

        # iterate through fields:
        for i, field in enumerate(fields):
            if self.rising and field['status'] == 'rising':
                priority[i] = 1

            if self.plateauing and field['status'] == 'plateauing':
                priority[i] = 1

            if self.setting and field['status'] == 'setting':
                priority[i] = 1

        return priority, self.label

#==============================================================================

