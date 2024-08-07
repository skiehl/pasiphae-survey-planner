#!/usr/bin/env python3
"""Classes for observational constraints.
"""

from abc import ABCMeta, abstractmethod
from astropy.coordinates import AltAz, get_body
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
# ABSTRACT CLASSES
#==============================================================================

class Constraint(object, metaclass=ABCMeta):
    """Observational constraint. Defines whether a source is observable from a
    specified location at a specified time or not.

    Notes
    -----
    Abstract class. This class should not be used directly as parent class.
    Custom Constraint-type classes should be built from ConstraintHard or
    ConstraintVariable classes.
    """

    constraint_type = None

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
    def get(self, source_coord, frame, sel=None, **kwargs):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.
        sel : np.ndarray or None, optional
            If not None, the array must be of boolean type. The constraint is
            evaluated only for times where this array is True. The default is
            None.
        kwargs
            Additional keyword arguments. They may be relevant only for some
            specific constraint classes. However, each class definition must be
            able to catch these.

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

class ConstraintHard(Constraint):
    """Hard observational constraint. Defines whether a source is observable
    from a specified location at a specified time or not.

    Notes
    -----
    Abstract class. This class should be used as parent class to any custom
    hard Constraint-type classes. Hard constraints are ones whose effect on
    fields does not change strongly from one day to another. Motion constraints
    are hard constraints.
    """

    constraint_type = 'hard'

#==============================================================================

class ConstraintVariable(Constraint):
    """Variable observational constraint. Defines whether a source is
    observable from a specified location at a specified time or not.

    Notes
    -----
    Abstract class. This class should be used as parent class to any custom
    variable Constraint-type classes. Variable constraints are ones whose
    effect on fields changes strongly from one day to another. E.g. a required
    minimum distance from the Moon depends on the position of the Moon on the
    sky and changes strongly from one day to another.
    """

    constraint_type = 'variable'

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

        self.constraints_hard = []
        self.constraints_var = []
        self.size = 0
        self.n_hard = 0
        self.n_var = 0
        self.i_hard = 0
        self.i_var = 0

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

        if self.i_hard + self.i_var >= self.size:
            raise StopIteration

        elif self.i_hard < self.n_hard:
            constraint = self.constraints_hard[self.i_hard]
            self.i_hard += 1

        else:
            constraint = self.constraints_var[self.i_var]
            self.i_var += 1

        return constraint

    #--------------------------------------------------------------------------
    def _evaluate_hard_constraints(self, source_coord, frame, **kwargs):
        """Evaluate the hard constaints.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.
        kwargs
            Additional keyword arguments forwarded to the constraints.

        Returns
        -------
        numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to all
            constraints, and False otherwise.
        """

        if self.n_hard == 0:
            observable = np.ones(frame.obstime.size, dtype=bool)

        elif self.n_hard == 1:
            observable = self.constraints_hard[0].get(
                    source_coord, frame, **kwargs)

        else:
            observable = self.constraints_hard[0].get(
                    source_coord, frame, **kwargs)

            for constraint in self.constraints_hard[1:]:

                # stop evaluating constraints, if source is not observable:
                if not np.any(observable):
                    break

                # update observability with additional constraint, only where
                # observable:
                observable[observable] = constraint.get(
                        source_coord, frame, observable, **kwargs)

        return observable

    #--------------------------------------------------------------------------
    def _evaluate_variable_constraints(self, source_coord, frame, **kwargs):
        """Evaluate the variable constaints.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.
        kwargs
            Additional keyword arguments forwarded to the constraints.

        Returns
        -------
        numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to all
            constraints, and False otherwise.
        """

        if self.n_var == 0:
            observable = np.ones(frame.obstime.size, dtype=bool)

        elif self.n_var == 1:
            observable = self.constraints_var[0].get(
                    source_coord, frame, **kwargs)

        else:
            observable = self.constraints_var[0].get(
                    source_coord, frame, **kwargs)

            for constraint in self.constraints_var[1:]:

                # stop evaluating constraints, if source is not observable:
                if not np.any(observable):
                    break

                # update observability with additional constraint, only where
                # observable:
                observable[observable] = constraint.get(
                        source_coord, frame, observable, **kwargs)

        return observable

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
            Raised if the handed `constraint` is not of Constraint-type.
            Raised if the handed `constraint` does not have `constraint_type`
            attribute value 'hard' or 'variable'.

        Returns
        -------
        None
        """

        # check if Constraint instance:
        if not isinstance(constraint, Constraint):
            raise TypeError('Unsupported type: {0}'.format(type(constraint)))

        # store constraint:
        if constraint.constraint_type == 'hard':
            self.constraints_hard.append(constraint)
            self.n_hard += 1

        elif constraint.constraint_type == 'variable':
            self.constraints_var.append(constraint)
            self.n_var += 1

        else:
            raise ValueError(
                "`constraint` has unsupported constraint_type value.")

        self.size += 1

    #--------------------------------------------------------------------------
    def get(self, source_coord, frame, **kwargs):
        """Evaluate all constraints jointly.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.
        kwargs
            Additional keyword arguments forwarded to the constraints.

        Returns
        -------
        numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to all
            constraints, and False otherwise.
        numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            hard constraints, and False otherwise.
        numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            variable constraints, and False otherwise.
        """

        observable_hard = self._evaluate_hard_constraints(
                source_coord, frame, **kwargs)
        observable_var = self._evaluate_variable_constraints(
                source_coord, frame, **kwargs)
        observable = np.logical_and(observable_hard, observable_var)

        return observable, observable_hard, observable_var

#==============================================================================

class AirmassLimit(ConstraintHard):
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
    def get(self, source_coord, frame, sel=None, **kwargs):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.
        sel : np.ndarray or None, optional
            If not None, the array must be of boolean type. The constraint is
            evaluated only for times where this array is True. The default is
            None.
        kwargs
            Used to catch keyword argmuments that are relevant to other
            constraint classes.

        Returns
        -------
        out : numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constraint, and False otherwise.
        """

        if sel is not None:
            frame = AltAz(location=frame.location, obstime=frame.obstime[sel])

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

class ElevationLimit(ConstraintHard):
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
    def get(self, source_coord, frame, sel=None, **kwargs):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.
        kwargs
            Used to catch keyword argmuments that are relevant to other
            constraint classes.

        Returns
        -------
        out : numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constraint, and False otherwise.
        """

        if sel is not None:
            frame = AltAz(location=frame.location, obstime=frame.obstime[sel])

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

class HourangleLimit(ConstraintHard):
    """Hourangle limit: only sources within a specific hourangle limit are
    observable.
    """

    #--------------------------------------------------------------------------
    def __init__(self, limit, limit_lo=None):
        """Create HourangleLimit instance.

        Parameters
        ----------
        limit : float
            Hourangle limit in hourangles.
        limit_lo : float, optional
            Lower hourangle limit in hourangles. Provide if the absolute value
            ofvthe lower limit differs from the upper limit. If not povided,
            the negative value of the limit provided in the first argument is
            automatically used as lower limit. The default is None.

        Returns
        -------
        None
        """

        self.limit_hi = limit * u.hourangle
        self.limit_lo = -limit if limit_lo is None else limit_lo
        self.limit_lo *= u.hourangle

    #--------------------------------------------------------------------------
    def __str__(self):
        """Description of the class instance and its parameters.

        Returns
        -------
        info : str
            Description of the class instance and its parameters.
        """

        if self.limit_hi == self.limit_lo:
            info = 'Hourangle limit: +/-{0:.2f}'.format(self.limit_hi)
        else:
            info = 'Hourangle limit:\n'
            info += 'Lower limit: {0:.2f}\n'.format(self.limit_lo)
            info += 'Upper limit: {0:.2f}'.format(self.limit_hi)

        return info

    #--------------------------------------------------------------------------
    def get(self, source_coord, frame, sel=None, **kwargs):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.
        sel : np.ndarray or None, optional
            If not None, the array must be of boolean type. The constraint is
            evaluated only for times where this array is True. The default is
            None.
        kwargs
            Used to catch keyword argmuments that are relevant to other
            constraint classes.

        Returns
        -------
        out : numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constraint, and False otherwise.
        """

        if sel is not None:
            frame = AltAz(location=frame.location, obstime=frame.obstime[sel])

        # get hourangles:
        lst = frame.obstime.sidereal_time(
                'apparent', longitude=frame.location.lon)
        hourangle = (source_coord.ra - lst).hourangle
        hourangle = (12. + hourangle) % 24. - 12.
        hourangle = hourangle * u.hourangle

        observable = np.logical_and(
                hourangle >= self.limit_lo, hourangle <= self.limit_hi)

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
                'limit': self.limit_hi.value, 'limit_lo': self.limit_lo.value}

        return params

#==============================================================================

class PolyHADecLimit(ConstraintHard):
    """Limits on the hourangle and declination defined by a polygon that
    encapsulates the allowed region.
    """

    #--------------------------------------------------------------------------
    def __init__(self, ha, dec):
        """Create PolyHADecLimit instance.

        Parameters
        ----------
        ha : list of floats
            Each list entry corresponds to one point that defines the
            Hourangle-Declination-limit outline. This list provides each
            point's hourangles in hours (-12, +12).
        dec : list of floats
            Each list entry corresponds to one point that defines the
            Hourangle-Declination-limit outline. This list provides each
            point's declination in degrees (-90, 90).

        Raises
        ------
        ValueError
            Raised, if the two input lists do not have the same length.

        Returns
        -------
        None
        """

        if len(ha) != len(dec):
            raise ValueError("'ha' and 'dec' must be of same length.")

        self.ha = list(ha)
        self.dec = list(dec)
        self.polygon = [(h, d) for (h, d) in zip(ha, dec)]

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

        return info[:-1]

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
    def get(self, source_coord, frame, sel=None, **kwargs):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.
        sel : np.ndarray or None, optional
            If not None, the array must be of boolean type. The constraint is
            evaluated only for times where this array is True. The default is
            None.
        kwargs
            Used to catch keyword argmuments that are relevant to other
            constraint classes.

        Returns
        -------
        out : numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constraint, and False otherwise.
        """

        if sel is not None:
            frame = AltAz(location=frame.location, obstime=frame.obstime[sel])

        # create array of hourangle-declination points:
        lst = frame.obstime.sidereal_time(
                'apparent', longitude=frame.location.lon)
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

        params = {'ha': self.ha, 'dec': self.dec}

        return params

#==============================================================================

class MoonDistance(ConstraintVariable):
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
    def get(self, source_coord, frame, sel=None, check_frame=True, **kwargs):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.
        sel : np.ndarray or None, optional
            If not None, the array must be of boolean type. The constraint is
            evaluated only for times where this array is True. The default is
            None.
        check_frame : bool, optional
            If True, the frame is compared to the previously saved frame and
            the saved Moon position is used, if the frames are identical;
            otherwise the Moon position is calculated and saved with the
            current frame. If False, the saved Moon position is not used and
            the saved frame is not overwritten.The default is True.
        kwargs
            Used to catch keyword argmuments that are relevant to other
            constraint classes.

        Returns
        -------
        out : numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constraint, and False otherwise.
        """

        # compare with saved frame:
        if check_frame:
            # if frame is the same as before re-use Moon positions:
            if self._same_frame(frame):
                moon_altaz = self.moon_altaz
            # otherwise calculate new Moon positions:
            else:
                moon_altaz = \
                        get_body('moon', frame.obstime).transform_to(frame)
                self.moon_altaz = moon_altaz
                self.last_frame = frame

        # ignore saved frame:
        else:
            moon_altaz = get_body('moon', frame.obstime).transform_to(frame)

        if sel is not None:
            frame = AltAz(location=frame.location, obstime=frame.obstime[sel])
            moon_altaz = moon_altaz[sel]

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

class MoonPolarization(ConstraintVariable):
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
    def get(self, source_coord, frame, sel=None, check_frame=True, **kwargs):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.
        sel : np.ndarray or None, optional
            If not None, the array must be of boolean type. The constraint is
            evaluated only for times where this array is True. The default is
            None.
        check_frame : bool
            If True, the frame is compared to the previously saved frame and
            the saved Moon position is used, if the frames are identical;
            otherwise the Moon position is calculated and saved with the
            current frame. If False, the saved Moon position is not used and
            the saved frame is not overwritten.The default is True.
        kwargs
            Used to catch keyword argmuments that are relevant to other
            constraint classes.

        Returns
        -------
        out : numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constraint, and False otherwise.
        """

        # compare with saved frame:
        if check_frame:
            # if frame is the same as before re-use Moon positions:
            if self._same_frame(frame):
                moon_altaz = self.moon_altaz
            # otherwise calculate new Moon positions:
            else:
                moon_altaz = \
                        get_body('moon', frame.obstime).transform_to(frame)
                self.moon_altaz = moon_altaz
                self.last_frame = frame

        # ignore saved frame:
        else:
            moon_altaz = get_body('moon', frame.obstime).transform_to(frame)

        if sel is not None:
            frame = AltAz(location=frame.location, obstime=frame.obstime[sel])
            moon_altaz = moon_altaz[sel]

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

class SunDistance(ConstraintVariable):
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
    def get(self, source_coord, frame, sel=None, check_frame=True, **kwargs):
        """Evaluate the constraint for a given target, a specific location and
        time.

        Parameters
        ----------
        source_coord : astropy.SkyCoord
            Coordinates of the target source.
        frame : astropy.coordinates.builtin_frames.altaz.AltAz
            A coordinate or frame in the Altitude-Azimuth system.
        sel : np.ndarray or None, optional
            If not None, the array must be of boolean type. The constraint is
            evaluated only for times where this array is True. The default is
            None.
        check_frame : bool
            If True, the frame is compared to the previously saved frame and
            the saved Moon position is used, if the frames are identical;
            otherwise the Moon position is calculated and saved with the
            current frame. If False, the saved Moon position is not used and
            the saved frame is not overwritten.The default is True.
        kwargs
            Used to catch keyword argmuments that are relevant to other
            constraint classes.

        Returns
        -------
        out : numpy.ndarray
            Array of boolean values. One entry for each input coordinate.
            The entry is True, if the source is observable according to the
            constraint, and False otherwise.
        """

        # compare with saved frame:
        if check_frame:
            # if frame is the same as before re-use Moon positions:
            if self._same_frame(frame):
                sun_altaz = self.sun_altaz
            # otherwise calculate new Moon positions:
            else:
                sun_altaz = get_body('sun', frame.obstime).transform_to(frame)
                self.sun_altaz = sun_altaz
                self.last_frame = frame

        # ignore saved frame:
        else:
            sun_altaz = get_body('sun', frame.obstime).transform_to(frame)

        if sel is not None:
            frame = AltAz(location=frame.location, obstime=frame.obstime[sel])
            sun_altaz = sun_altaz[sel]

        source_altaz = source_coord.transform_to(frame)
        sun_altaz = get_body('sun', frame.obstime).transform_to(frame)
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
