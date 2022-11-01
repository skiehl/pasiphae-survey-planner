# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Classes for observational constraints.
"""

from abc import ABCMeta, abstractmethod
from astropy.coordinates import get_sun, get_moon
import astropy.units as u
import numpy as np
from textwrap import dedent

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
        """Create Constraints instance.

        Returns
        -------
        None
        """

        self.constraints = []
        self.size = 0
        self.i = 0

    #--------------------------------------------------------------------------
    def __str__(self):
        """Returns string representation of all contained Constraint instances.

        Returns
        -------
        text : str
            Description of stored Constraint instances.
        """

        text = 'Set of observational constraints:\n'
        for constraint in self.constraints:
            text = '{0:s}* {1:s}\n'.format(text, constraint.__str__())

        return text

    #--------------------------------------------------------------------------
    def __iter__(self):
        """Make this class iterable.

        Returns
        -------
        Constraints
            Instance of this class.
        """

        return self

    #--------------------------------------------------------------------------
    def __next__(self):
        """Return the next stored Constraint instance.

        Returns
        -------
        Constraint
            Constraint instance stored in the Constraints instance.
        """

        if self.i >= self.size:
            raise StopIteration
        else:
            constraint = self.constraints[self.i]
            self.i += 1

            return constraint

    #--------------------------------------------------------------------------
    def add(self, constraint):
        """Add a new constraint.

        Parameters
        ----------
        constraint : constraint
            An instance of a Constraint as defined in this module.

        Raises
        ------
        TypeError
            Raised if the handed constraint is not of Constraint parent-class.

        Returns
        -------
        None
        """

        # chack if Constraint instance:
        if not isinstance(constraint, Constraint):
            raise TypeError('Unsupported type: {0}'.format(type(constraint)))

        self.constraints.append(constraint)
        self.size += 1

    #--------------------------------------------------------------------------
    def get(self, source_coord, frame):
        """Evaluate all constraints jointly.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.

        Returns
        -------
        numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to all
            constraints, and False otherwise.
        """

        observable = np.logical_and.reduce(
                [constraint.get(source_coord, frame) \
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
    def get(self, source_coord, frame):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.

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
        ----------
        limit : float
            Lower elevation limit in degrees.

        Returns
        -------
        None
        """

        self.limit = limit * u.deg

    #--------------------------------------------------------------------------
    def __str__(self):
        """Description of the class instance and its parameters.

        Returns
        -------
        str
            Description of the class instance and its parameters.
        """

        return 'Elevation limit: {0:.2f}'.format(self.limit)

    #--------------------------------------------------------------------------
    def get(self, source_coord, frame):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.

        Returns
        -------
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
        -------
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
        ----------
        limit : float
            Upper airmass limit
        conversion : str, default="secz"
            Select the conversion method from altitude to airmass.
            Options: "secz" (default), "Rosenberg", "KastenYoung",
            "Young". See alt_to_airmass() docstring in utilities.py for
            details.

        Returns
        -------
        None
        """

        self.limit = limit
        self.conversion = conversion

    #--------------------------------------------------------------------------
    def __str__(self):
        """Description of the class instance and its parameters.

        Returns
        -------
        str
            Description of the class instance and its parameters.
        """

        return 'Airmass limit: {0:.2f}'.format(self.limit)

    #--------------------------------------------------------------------------
    def get(self, source_coord, frame):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.

        Returns
        -------
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
        -------
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
        ----------
        limit : float
            Minimum required angular separation between targets and the Sun in
            degrees.

        Returns
        -------
        None
        """

        self.limit = limit * u.deg
        self.last_frame = None
        self.sun_altaz = None

    #--------------------------------------------------------------------------
    def __str__(self):
        """Description of the class instance and its parameters.

        Returns
        -------
        str
            Description of the class instance and its parameters.
        """

        return 'Sun distance: {0:.2f}'.format(self.limit)

    #--------------------------------------------------------------------------
    def get(self, source_coord, frame):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.

        Returns
        -------
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
        -------
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
    def get(self, source_coord, frame):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.

        Returns
        -------
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
        -------
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
        ----------
        limit : float
            Angular range to avoid polarized, scattered Moon light. Sources
            within the range (90-limit, 90+limit) degrees separation from the
            Moon are not observable.

        Returns
        -------
        None
        """

        self.limit = limit * u.deg
        self.limit_lo = (90. - limit) * u.deg
        self.limit_hi = (90. + limit) * u.deg
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

        return 'Moon polarization: {0:.2f}'.format(self.limit)

    #--------------------------------------------------------------------------
    def get(self, source_coord, frame):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.

        Returns
        -------
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
        observable = np.logical_or(
                separation <= self.limit_lo, separation >= self.limit_hi)

        return observable

    #--------------------------------------------------------------------------
    def get_params(self):
        """Returns the constraint parameters.

        Returns
        -------
        dict
            Constraint parameters.
        """

        params = {'limit': self.limit.value}

        return params

#==============================================================================

class PolyHADecLimit(Constraint):
    """Limits on the hourangle and declination defined by a polygon that
    encapsulates the allowed region.
    """

    #--------------------------------------------------------------------------
    def __init__(self, polygon):
        """Create PolyHADecLimit instance.

        Parameters
        ----------
        polygon : list of tuples of two floats
            Each list entry corresponds to one point that defines the
            Hourangle-Declination-limit outline. Each point is a tuple of
            two floats; the first one giving the hourangle in hours (-12, +12),
            the second giving the declination in degrees (-90, 90).

        Returns
        -------
        None
        """

        self.polygon = polygon

    #--------------------------------------------------------------------------
    def __str__(self):
        """Description of the class instance and its parameters.

        Returns
        -------
        str
            Description of the class instance and its parameters.
        """

        info = dedent(
                """\
                Polygon Hourangle-Declination limits:
                HA (h) Dec (deg)
                """)
        for ha, dec in self.polygon:
            info = f'{info}{ha:+6.2f} {dec:+6.2f}\n'

        return info

    #--------------------------------------------------------------------------
    def _orientation(self, points, q0, q1):
        """Orientation of a triangle spanned by points p, q1, q2; where p can
        be an array of many points.

        Parameters
        ----------
        points : numpy.ndarray
            Coordinates of the triangle's first point.
        q0 : tuple of floats
            Coordinate of the triangles second point.
        q1 : tuple of floats
            Coordinate of the triangles third point.

        Returns
        -------
        orientation : numpy.ndarray
            One dimensional array of int type. +1 for counter-clockwise
            triangles; -1 clockwise triangles; 0 otherwise.
        """

        orientation = np.sign(
                (q1[0] - q0[0]) * (points[1] - q0[1]) \
                - (points[0] - q0[0]) * (q1[1] - q0[1]))

        return orientation

    #--------------------------------------------------------------------------
    def _crossing(self, points, q0, q1):
        """Check which points, when extended toward the right, would cross the
        line segment spanned from q0 to q1, and whether the line segment
        crosses upward or downward.

        Parameters
        ----------
        points : numpy.ndarray
            Two dimensional array. The first column has to contain the
            hourangles, the second column the declinations.
        q0 : tuple of floats
            First coordinate of the line segment.
        q1 : tuple of floats
            Second coordinate of the line segment.

        Returns
        -------
        crossing : numpy.ndarray
            One dimensional array of int type. +1 for points whose extension
            to the right is crossed upward by the line segment; -1 for points
            whose extension to the right is crossed downward by the line
            segment; 0 otherwise.
        """

        p_heq_q0 = q0[1] <= points[1]
        p_heq_q1 = q1[1] <= points[1]
        p_left = self._orientation(points, q0, q1)

        crossing = np.zeros(points.shape[1], dtype=int)

        # count segments crossing upwards and right of point as +1:
        sel = np.logical_and.reduce([p_heq_q0, ~p_heq_q1, p_left > 0])
        crossing[sel] += 1

        # count segments crossing downwards and right of point as -1:
        sel = np.logical_and.reduce([~p_heq_q0, p_heq_q1, p_left < 0])
        crossing[sel] -= 1

        return crossing

    #--------------------------------------------------------------------------
    def _inside_polygon(self, points, polygon):
        """Check whether or not points are inside a given polygon.

        Parameters
        ----------
        points : numpy.ndarray
            Two dimensional array. The first column has to contain the
            hourangles, the second column the declinations.
        polygon : list of tuples of two floats
            Polygon points as stored in this class.

        Returns
        -------
        is_inside : numpy.ndarray
            One dimensional array of bool type. True, for points inside the
            polygon; False, otherwise.

        Notes
        -----
        The algorithm is a vectorized python port of that by Dan Sunday [1].

        References
        ----------
        [1] https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html
        """

        # close polygon:
        polygon = np.array(polygon + [polygon[0]])

        # iterate through polygon segments to get winding number:
        winding_number = np.zeros(points.shape[1], dtype=int)

        for q0, q1 in zip(polygon[0:-1], polygon[1:]):
            winding_number += self._crossing(points, q0, q1)

        is_inside = winding_number > 0

        return is_inside

    #--------------------------------------------------------------------------
    def get(self, source_coord, frame):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.

        Returns
        -------
        out : numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constraint, and False otherwise.
        """

        # create array of hourangle-declination points:
        lst = frame.obstime.sidereal_time('apparent')
        hourangle = (source_coord.ra - lst).hourangle
        hourangle = (12. + hourangle) % 24. - 12.
        dec = np.ones(hourangle.size) * source_coord.dec.deg
        points = np.array([hourangle, dec])

        # check which points are within the polygon:
        observable = self._inside_polygon(points, self.polygon)

        return observable

    #--------------------------------------------------------------------------
    def get_params(self):
        """Returns the constraint parameters.

        Returns
        -------
        dict
            Constraint parameters.
        """

        params = {'polygon': self.polygon}

        return params
