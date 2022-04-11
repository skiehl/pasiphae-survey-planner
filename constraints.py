# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Classes for observational constraints.
"""

from abc import ABCMeta, abstractmethod
from astropy.coordinates import get_sun, get_moon
import astropy.units as u
import numpy as np

import utilities as ut

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

class Constraints(object):
    """List of constraints that are getting jointly evaluated.
    """

    #--------------------------------------------------------------------------
    def __init__(self):
        """List of constraints that are getting jointly evaluated.
        """

        self.constraints = []
        self.size = 0
        self.i = 0

    #--------------------------------------------------------------------------
    def __str__(self):

        text = 'Set of observational constraints:\n'
        for constraint in self.constraints:
            text = '{0:s}* {1:s}\n'.format(text, constraint.__str__())

        return text

    #--------------------------------------------------------------------------
    def __iter__(self):

        return self

    #--------------------------------------------------------------------------
    def __next__(self):

        if self.i >= self.size:
            raise StopIteration
        else:
            constraint = self.constraints[self.i]
            self.i += 1

            return constraint

    #--------------------------------------------------------------------------
    def add(self, constraint):
        """Add a new constraint.
        """

        # chack if Constraint instance:
        if not isinstance(constraint, Constraint):
            raise TypeError('Unsupported type: {0}'.format(type(constraint)))

        self.constraints.append(constraint)
        self.size += 1

    #--------------------------------------------------------------------------
    def get(self, source_coord, telescope):
        """Evaluate all constraints jointly.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target sources.
        telescope : Telescope
            Provides the telescope position and current date and time.

        Returns
        -------
        numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to all
            constraints, and False otherwise.
        """

        observable = np.logical_and.reduce(
                [constraint.get(source_coord, telescope) \
                 for constraint in self.constraints])

        return observable

#==============================================================================

class Constraint(object, metaclass=ABCMeta):
    """Observational constraint. Defines whether a source is observable from a
    specified location at a specified time or not.
    """

    #--------------------------------------------------------------------------
    @abstractmethod
    def __init__(self):
        """
        Notes
        -----
        Abstract method. Parameters depend on the specific constraint.
        """

        pass

    #--------------------------------------------------------------------------
    @abstractmethod
    def __str__(self):
        """
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

        Note
        -------
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
    def get(self, source_coord, telescope):
        """Evaluate the constraint for given targets, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target sources.
        telescope : Telescope
            Provides the telescope position and current date and time.

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

    #--------------------------------------------------------------------------
    @abstractmethod
    def get_params(self):
        """Get the constraint parameters.

        Returns
        -------
        dict
            The constraint parameters and values.

        Notes
        -----
        Abstract method. The implementation needs to be specified in the
        sub-classes. The dictionary keys need to match all arguments of the
        __init__() method.
        """


#==============================================================================

class ElevationLimit(Constraint):
    """Elevation limit: only sources above a specified elevation are
    observable.
    """

    #--------------------------------------------------------------------------
    def __init__(self, limit):
        """Create ElevationLimit instance.

        Parameters
        -----
        limit : float
            Lower elevation limit in degrees.
        """

        self.limit = limit * u.deg

    #--------------------------------------------------------------------------
    def __str__(self):
        """String representation.
        """

        return 'Elevation limit: {0:.2f}'.format(self.limit)

    #--------------------------------------------------------------------------
    def get(self, source_coord, frame):
        """Evaluate the constraint for given targets, a specific location and
        time.

        Parameters
        -----
        source_coord : astropy.SkyCoord
            Coordinates of the target sources.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.

        Returns
        -----
        out : numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constraint, and False otherwise.
        """

        altaz = source_coord.transform_to(frame)
        observable = altaz.alt >= self.limit

        return observable

    #--------------------------------------------------------------------------
    def get_params(self):
        """Returns the constraint parameters.

        Returns
        -----
        dict
            Constraint parameters.
        """

        params = {'limit': self.limit.value}

        return params

#==============================================================================

class AirmassLimit(Constraint):
    """Airmass limit: only sources below a specified airmass are observable.
    """

    #--------------------------------------------------------------------------
    def __init__(self, limit, conversion="secz"):
        """Create AirmassLimit instance.

        Parameters
        -----
        limit : float
            Upper airmass limit
        conversion : str, default="secz"
            Select the conversion method from altitude to airmass.
            Options: "secz" (default), "Rosenberg", "KastenYoung",
            "Young". See alt_to_airmass() docstring in utilities.py for
            details.
        """

        self.limit = limit
        self.conversion = conversion

    #--------------------------------------------------------------------------
    def __str__(self):
        """String representation.
        """

        return 'Airmass limit: {0:.2f}'.format(self.limit)

    #--------------------------------------------------------------------------
    def get(self, source_coord, frame):
        """Evaluate the constraint for given targets, a specific location and
        time.

        Parameters
        -----
        source_coord : astropy.SkyCoord
            Coordinates of the target sources.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.

        Returns
        -----
        out : numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constraint, and False otherwise.
        """

        altaz = source_coord.transform_to(frame)
        observable = ut.alt_to_airmass(altaz.alt, conversion=self.conversion) \
                <= self.limit

        return observable

    #--------------------------------------------------------------------------
    def get_params(self):
        """Returns the constraint parameters.

        Returns
        -----
        dict
            Constraint parameters.
        """

        params = {
                'limit': self.limit,
                'conversion': self.conversion}

        return params

#==============================================================================

class SunDistance(Constraint):
    """Only sources sufficiently separated from the Sun are observable.
    """

    #--------------------------------------------------------------------------
    def __init__(self, limit):
        """Create SunDistance instance.

        Parameters
        -----
        limit : float
            Minimum required angular separation between targets and the Sun in
            degrees.
        """

        self.limit = limit * u.deg
        self.last_frame = None
        self.sun_altaz = None

    #--------------------------------------------------------------------------
    def __str__(self):
        """String representation.
        """

        return 'Sun distance: {0:.2f}'.format(self.limit)

    #--------------------------------------------------------------------------
    def get(self, source_coord, frame):
        """Evaluate the constraint for given targets, a specific location and
        time.

        Parameters
        -----
        source_coord : astropy.SkyCoord
            Coordinates of the target sources.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.

        Returns
        -----
        out : numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constraint, and False otherwise.
        """

        # if frame is the same as before re-use Moon positions:
        if self._same_frame(frame):
            sun_altaz = self.sun_altaz
        # otherwise calculate new Moon positions:
        else:
            sun_altaz = get_sun(frame.obstime).transform_to(frame)
            self.sun_altaz = sun_altaz
            self.last_frame = frame

        source_altaz = source_coord.transform_to(frame)
        sun_altaz = get_sun(frame.obstime).transform_to(frame)
        separation = source_altaz.separation(sun_altaz)
        observable = separation > self.limit

        return observable

    #--------------------------------------------------------------------------
    def get_params(self):
        """Returns the constraint parameters.

        Returns
        -----
        dict
            Constraint parameters.
        """

        params = {'limit': self.limit.value}

        return params

#==============================================================================

class MoonDistance(Constraint):
    """Only sources sufficiently separated from the Moon are observable.
    """

    #--------------------------------------------------------------------------
    def __init__(self, limit):
        """Create MoonDistance instance.

        Parameters
        -----
        limit : float
            Minimum required angular separation between targets and the Moon in
            degrees.
        """

        self.limit = limit * u.deg
        self.last_frame = None
        self.moon_altaz = None

    #--------------------------------------------------------------------------
    def __str__(self):
        """String representation.
        """

        return 'Moon distance: {0:.2f}'.format(self.limit)

    #--------------------------------------------------------------------------
    def get(self, source_coord, frame):
        """Evaluate the constraint for given targets, a specific location and
        time.

        Parameters
        -----
        source_coord : astropy.SkyCoord
            Coordinates of the target sources.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.

        Returns
        -----
        out : numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constraint, and False otherwise.
        """

        # if frame is the same as before re-use Moon positions:
        if self._same_frame(frame):
            moon_altaz = self.moon_altaz
        # otherwise calculate new Moon positions:
        else:
            moon_altaz = get_moon(frame.obstime).transform_to(frame)
            self.moon_altaz = moon_altaz
            self.last_frame = frame

        source_altaz = source_coord.transform_to(frame)
        separation = source_altaz.separation(moon_altaz)
        observable = separation > self.limit

        return observable

    #--------------------------------------------------------------------------
    def get_params(self):
        """Returns the constraint parameters.

        Returns
        -----
        dict
            Constraint parameters.
        """

        params = {'limit': self.limit.value}

        return params

#==============================================================================

class MoonPolarization(Constraint):
    """Avoid polarized, scattered Moon light. Target sources in a specified
    angular range around 90 degrees separation from the Moon are not
    observable.
    """

    #--------------------------------------------------------------------------
    def __init__(self, limit):
        """Create MoonPolarization instance.

        Parameters
        -----
        limit : float
            Angular range to avoid polarized, scattered Moon light. Sources
            within the range (90-limit, 90+limit) degrees separation from the
            Moon are not observable.
        """

        self.limit = limit * u.deg
        self.limit_lo = (90. - limit) * u.deg
        self.limit_hi = (90. + limit) * u.deg
        self.last_frame = None
        self.moon_altaz = None

    #--------------------------------------------------------------------------
    def __str__(self):
        """String representation.
        """

        return 'Moon polarization: {0:.2f}'.format(self.limit)

    #--------------------------------------------------------------------------
    def get(self, source_coord, frame):
        """Evaluate the constraint for given targets, a specific location and
        time.

        Parameters
        -----
        source_coord : astropy.SkyCoord
            Coordinates of the target sources.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.

        Returns
        -----
        out : numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constrains, and False otherwise.
        """

        # if frame is the same as before re-use Moon positions:
        if self._same_frame(frame):
            moon_altaz = self.moon_altaz
        # otherwise calculate new Moon positions:
        else:
            moon_altaz = get_moon(frame.obstime).transform_to(frame)
            self.moon_altaz = moon_altaz
            self.last_frame = frame

        source_altaz = source_coord.transform_to(frame)
        separation = source_altaz.separation(moon_altaz)
        observable = np.logical_or(
                separation <= self.limit_lo, separation >= self.limit_hi)

        return observable

    #--------------------------------------------------------------------------
    def get_params(self):
        """Returns the constraint parameters.

        Returns
        -----
        dict
            Constraint parameters.
        """

        params = {'limit': self.limit.value}

        return params
