#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime test of code snippet:
Module: constraints
Class: MoonDistance
Method: run
Question: Does saving the frame reduce the runtime?
Notes: The same applies to MoonPolarization and SunDistance
"""

from abc import ABCMeta, abstractmethod
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_moon
from astropy.time import Time
import astropy.units as u
import numpy as np
import timeit

#==============================================================================
# CLASSES
#==============================================================================

class Constraint(object, metaclass=ABCMeta):
    """Observational constraint. Defines whether a source is observable from a
    specified location at a specified time or not.
    """

    #--------------------------------------------------------------------------
    @abstractmethod
    def __init__(self):
        """Create Constraint instance.

        Notes
        -----
        Abstract method. Parameters depend on the specific constraint.
        """

        pass

    #--------------------------------------------------------------------------
    @abstractmethod
    def __str__(self):
        """Description of the class instance and its parameters.

        Notes
        -----
        Abstract method. String depends on the specific constraint.
        """

        pass

    #--------------------------------------------------------------------------
    def _same_frame(self, frame):
        """Check if given frame is the same as previously used frame.

        Parameters
        ----------
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.

        Returns
        -------
        bool
            True if the frame is the location and the time(s) in the frame are
            identical. False, otherwise.

        Notes
        -----
        This method is used in Constraint-classes that need to calculate e.g.
        the Moon or Sun position and avoid re-calculating it when the provided
        frame is the same as previously used to calculate those positions.
        """

        if self.last_frame is None:
            return False

        if self.last_frame.obstime.size != frame.obstime.size:
            return False

        if (self.last_frame.location == frame.location and  \
            np.all(self.last_frame.obstime == frame.obstime)):
            return True

        return False

    #--------------------------------------------------------------------------
    @abstractmethod
    def get(self, source_coord, frame, sel=None):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.
        sel : np.ndarray or None
            If not None, the array must be of boolean type. The constraint is
            evaluated only for times where this array is True. The default is
            None.

        Returns
        -------
        numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constraint, and False otherwise.

        Notes
        -----
        Abstract method. The implementation needs to be specified in the
        sub-classes.
        """

#==============================================================================

class MoonDistance(Constraint):
    """Only sources sufficiently separated from the Moon are observable.
    """

    #--------------------------------------------------------------------------
    def __init__(self, limit):
        """Create MoonDistance instance.

        Parameters
        ----------
        limit : float
            Minimum required angular separation between targets and the Moon in
            degrees.

        Returns
        -------
        None
        """

        self.limit = limit * u.deg
        self.last_frame = None
        self.moon_altaz = None

    #--------------------------------------------------------------------------
    def __str__(self):
        """Description of the class instance and its parameters.

        Returns
        -------
        str
            Description of the class instance and its parameters.
        """

        return 'Moon distance: {0:.2f}'.format(self.limit)

    #--------------------------------------------------------------------------
    def get(self, source_coord, frame, sel=None, frame_check=True):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.
        sel : np.ndarray or None
            If not None, the array must be of boolean type. The constraint is
            evaluated only for times where this array is True. The default is
            None.

        Returns
        -------
        out : numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constraint, and False otherwise.
        """

        if frame_check:
            # if frame is the same as before re-use Moon positions:
            if self._same_frame(frame):
                moon_altaz = self.moon_altaz
            # otherwise calculate new Moon positions:
            else:
                moon_altaz = get_moon(frame.obstime).transform_to(frame)
                self.moon_altaz = moon_altaz
                self.last_frame = frame

        else:
            moon_altaz = get_moon(frame.obstime).transform_to(frame)

        if sel is not None:
            frame = AltAz(location=frame.location, obstime=frame.obstime[sel])
            moon_altaz = moon_altaz[sel]

        source_altaz = source_coord.transform_to(frame)
        separation = source_altaz.separation(moon_altaz)
        observable = separation > self.limit

        return observable

#==============================================================================
# MAIN
#==============================================================================

if __name__ == '__main__':
    n_timeit = 1000

    constraint = MoonDistance(10)
    source_coord = SkyCoord(0*u.deg, 0*u.deg)
    time = Time('2024-01-01T00:00:00') + np.linspace(0, 1, 100) * u.d
    location = EarthLocation(
            lat='35d12m43s', lon='24d53m57s', height=1750)
    frame = AltAz(obstime=time, location=location)

    print('Average runtime:')

    # test with frame saving:
    results = timeit.timeit(
            lambda: constraint.get(
                source_coord, frame, sel=None, frame_check=True),
            number=n_timeit)
    results /= n_timeit
    print(f'With frame saving:    {results}')

    # test without frame saving:
    results = timeit.timeit(
            lambda: constraint.get(
                source_coord, frame, sel=None, frame_check=False),
            number=n_timeit)
    results /= n_timeit
    print(f'Without frame saving: {results}')
