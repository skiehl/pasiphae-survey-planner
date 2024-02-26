# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Classes related to the motion of a telescope.
"""

from abc import ABCMeta, abstractmethod
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u
import numpy as np

# alternative download location since
# http://maia.usno.navy.mil/ser7/finals2000A.all is unavailable:
from astropy.utils import iers
iers.conf.iers_auto_url = \
    'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all'

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

class Position(object):
    """Position of a telescope.
    """

    #--------------------------------------------------------------------------
    def __init__(self, coord, loc, time):
        """Create a Position instance.

        Parameters
        -----
        coord : astropy.SkyCoord
            Coordinates of a source the telescope is currently pointed at.
        loc : astropy.EarthLocation
            Location of the telescope on Earth.
        time : astropy.Time
            Current date and time.

        Returns
        -----
        None
        """

        frame = AltAz(obstime=time, location=loc)
        altaz = coord.transform_to(frame)
        self.ra = coord.ra
        self.dec = coord.dec
        self.alt = altaz.alt
        self.az = altaz.az
        self. ha = self.ra - time.sidereal_time('apparent')
        self.za = self.dec - loc.lat

    #--------------------------------------------------------------------------
    def __str__(self):
        """Write out the position in various coordinate systems.

        Returns
        -----
        out : str
        """

        text = 'Right Ascension: {0:s}\n'.format(str(self.ra))
        text += 'Declination:     {0:s}\n'.format(str(self.dec))
        text += 'Altitude:       {0:6.2f}\n'.format(self.alt)
        text += 'Azimuth:        {0:6.2f}\n'.format(self.az)
        text += 'Hour angle:     {0:6.2f} h\n'.format(self.ha.hour)
        text += 'Zenith angle:   {0:6.2f}'.format(self.za)

        return text

#==============================================================================

class Telescope(object, metaclass=ABCMeta):
    """Telescope base class.

    Provides general and abstract methods for Telescope instances.
    Child classes for different telescope mounts need to specify the abstract
    methods.
    """

    class_name = "Telescope"

    #--------------------------------------------------------------------------
    @abstractmethod
    def __init__(
            self, lon, lat, height, name=''):
        """Create a Telescope instance.

        Parameters
        -----
        lon : str or astropy.Angle
            Longitude of telescope location. String input needs to be
            consistent with astropy.Angle definition.
        lat : str or astropy.Angle
            Latitude of telescope location. String input needs to be consistent
            with astropy.Angle definition.
        height : float
            Height of telescope location in meters.
        name : str, default=''
            Name of the telescope/observatory.

        Returns
        -----
        None

        Notes
        -----
        Abstract method. Mount specific attributes are defined in the child
        classes.
        """

        lat = Angle(lat)
        lon = Angle(lon)
        height = height * u.m

        self.loc = EarthLocation(
                lat=lat, lon=lon, height=height)
        self.name = name
        self.dome = False
        self.time = None
        self.frame = None
        self.pos = None

        self.slew_model_dome = None

        print('Telescope: {0:s} created.'.format(self.name))

    #--------------------------------------------------------------------------
    def __str__(self):
        """Write out location and current position of the telescope.

        Returns
        -----
        out : str
        """

        text = 'Telescope:  {0:s}\n'.format(self.name)
        text += 'Longitude:  {0:s}\n'.format(
                self.loc.lon.to_string(decimal=False))
        text += 'Longitude:  {0:s}\n'.format(
                self.loc.lat.to_string(decimal=False))
        text += 'Height:     {0:.2f}\n'.format(self.loc.height)
        text += 'Mount:      {0:s}\n'.format(self.mount)

        if self.time is not None:
            text += '\nTime:       {0:s}\n'.format(str(self.time))

        if self.pos is not None:
            text += '\n' + self.pos.__str__()

        return text

    #--------------------------------------------------------------------------
    def _print(self, verbose, note, pre=''):
        if verbose:
            print("{0:s}{1:s}: {2:s}".format(pre, self.class_name, note))

    #--------------------------------------------------------------------------
    @abstractmethod
    def set_slew_model(self, axis, slew_model):
        """Set a slew model for a specific axis.
        This slew model will define how to calculate the time needed to slew
        from one position to another.

        Parameters
        -----
        axis : str
            Specify axis: 'ra', 'dec', 'dome'
        slew_model : SlewModel
            Slew model used for the specified axis.

        Returns
        -----
        None

        Notes
        -----
        Abstract method. Axis is mount specific.
        """

        if axis == 'dome':
            self.dome = True
            self.slew_model_dome = slew_model
            print('Telescope: Dome slew model set.')

        elif axis == 'ra':
            # implement in child class
            pass

        elif axis == 'dec':
            # implement in child class
            pass

        else:
            raise ValueError(
                    "Unknown axis: '{0:s}'".format(axis))

        return None

    #--------------------------------------------------------------------------
    @abstractmethod
    def is_set(self):
        """Check if slew models have been set.

        Returns
        -----
        out : bool
            True, if slew model has been set for each axis. False, otherwise.

        Notes
        -----
        Abstract method. Mount dependent specifics need to be defined in child
        classes.
        """

        return False

    #--------------------------------------------------------------------------
    def set_time(self, time, verbose=True):
        """Set current time.

        Parameters
        -----
        time : astropy.Time
            Date and time in UTC. Accepts any input format that astropy.Time is
            accepting.
        verbose : bool, default=True
            If True, print out information.

        Returns
        -----
        None
        """

        if isinstance(time, str):
            self.time = Time(time, scale='utc', location=self.loc)
        else:
            self.time = time
        self.frame = AltAz(obstime=self.time, location=self.loc)

        self._print(verbose, 'Time set to: {0:s}'.format(
                self.time.strftime('%Y-%m-%d %H:%M:%S')))

    #--------------------------------------------------------------------------
    def set_pos(self, coord):
        """Set current position of telescope.

        Parameters
        -----
        coord : astropy.SkyCoord
            Coordinates of a source the telescope is currently pointed at.

        Returns
        -----
        None
        """

        if self.time is None:
            raise ValueError(
                    "Set time first before setting a position.")

        self.pos = Position(coord, self.loc, self.time)

        return None

    #--------------------------------------------------------------------------
    @abstractmethod
    def set_to_zenith(self):
        """Set telescope position to zenith.

        Returns
        -----
        None

        Notes
        -----
        Abstract method. Mount dependent specifics need to be defined in child
        classes.
        """

        # implement in child class
        pass

    #--------------------------------------------------------------------------
    @abstractmethod
    def get_slew_time(self, coord):
        """Calculate slew time from current position to given coordinates.

        Parameters
        -----
        coord : astropy.SkyCoord
            Source coordinates.

        Returns
        -----
        out : numpy.ndarray
            Slew time to each source from the current position.

        Notes
        -----
        Abstract method. Mount dependent specifics need to be defined in child
        classes.
        """

        # implement in child class
        pass

    #--------------------------------------------------------------------------
    def get_time(self):
        """Get the telescope's current date and time.

        Returns
        -----
        out : astropy.Time
            Current date and time of the telescope.
        """

        if self.time is None:
            raise ValueError('No date and time set yet.')

        return self.time

    #--------------------------------------------------------------------------
    def next_sun_set_rise(self, twilight, verbose=True):
        """Calculate time of the next Sun set and Sun rise.

        Parameters
        ----------
        twilight : str or float
            Select the Sun elevation at which observations should start/end.
            Choose from 'astronomical' (-18 deg), 'nautical' (-12 deg),
            'civil' (-6 deg), 'sunset' (0 deg). Or use float to set Sun
            elevation (in degrees).

        Raises
        ------
        ValueError
            If the input is not a float or from the list of allowed string
            inputs.

        Returns
        -------
        astropy.time.Time
            Time of the next Sun set.
        astropy.time.Time
            Time of the following Sun rise.

        """

        if isinstance(twilight, float):
            sunset = twilight * u.deg
        elif twilight == 'astronomical':
            sunset = -18. * u.deg
        elif twilight == 'nautical':
            sunset = -12. * u.deg
        elif twilight == 'civil':
            sunset = -6. * u.deg
        elif twilight == 'sunset':
            sunset = 0. * u.deg
        else:
            raise ValueError(
                "Either set a float or chose from 'astronomical', 'nautical'" \
                " 'civil', or 'sunset'.")

        self._print(verbose, 'Current time:    {0:s}'.format(
                self.time.strftime('%Y-%m-%d %H:%M:%S')))
        time = self.time + np.arange(0., 48., 0.01) * u.h
        frame = AltAz(obstime=time, location=self.loc)
        sun = get_sun(time).transform_to(frame)
        sun_alt = sun.alt
        night = sun_alt < sunset

        if np.all(~night):
            self._print(verbose, 'WARNING: Sun never sets!')
            return False

        if np.all(night):
            self._print(verbose, 'NOTE: Sun never rises!')
            return False

        # current time is at night, find next sun set and following sun rise:
        if night[0]:
            self._print(verbose, 'NOTE: current time is night time.')
            # index of next sun rise:
            i = np.argmax(~night)
            # index of next sun set:
            i += np.argmax(night[i:])
            # index of following sun rise:
            j = i + np.argmax(~night[i:]) - 1

        # current time is at day, find next sun set and sun rise:
        else:
            # index of next sun set:
            i = np.argmax(night)
            # index of following sun rise:
            j = i + np.argmax(~night[i:]) - 1


        # interpolate linearly:
        interp = (sunset.value - sun_alt[i-1].deg) \
                / (sun_alt[i].deg - sun_alt[i-1].deg)
        time_sunset = time[i-1] + (time[i] - time[i-1]) * interp
        interp = (sunset.value - sun_alt[j].deg) \
                / (sun_alt[j+1].deg - sun_alt[j].deg)
        time_sunrise = time[j] + (time[j+1] - time[j]) * interp

        self._print(verbose, 'Next night start: {0:s}'.format(
                time_sunset.strftime('%Y-%m-%d %H:%M:%S')))
        self._print(verbose, 'Next night stop:  {0:s}'.format(
                time_sunrise.strftime('%Y-%m-%d %H:%M:%S')))

        return time_sunset, time_sunrise

#==============================================================================

class TelescopeEq(Telescope):
    """Telescope with equatorial mount.
    """

    class_name = "TelescopeEq"
    mount = 'equatorial'

    #--------------------------------------------------------------------------
    def __init__(
            self, lat, lon, height, name=''):
        """Create a Telescope instance.

        Parameters
        -----
        lat : str or astropy.Angle
            Latitude of telescope location. String input needs to be consistent
            with astropy.Angle definition.
        lon : str or astropy.Angle
            Longitude of telescope location. String input needs to be
            consistent with astropy.Angle definition.
        height : float
            Height of telescope location in meters.
        name : str, default=''
            Name of the telescope/observatory.

        Returns
        -----
        None
        """

        super().__init__(lat, lon, height, name=name)
        self.slew_model_ra = None
        self.slew_model_dec = None

    #--------------------------------------------------------------------------
    def set_slew_model(self, axis, slew_model):
        """Set a slew model for a specific axis.
        This slew model will define how to calculate the time needed to slew
        from one position to another.

        Parameters
        -----
        axis : str
            Specify axis: 'ra', 'dec', 'dome'
        slew_model : SlewModel
            Slew model used for the specified axis.

        Returns
        -----
        None
        """

        if axis == 'dome':
            self.dome = True
            self.slew_model_dome = slew_model
            self._print(True, 'Dome slew model set.')

        elif axis == 'ra':
            self.slew_model_ra = slew_model
            self._print(True, 'RA slew model set.')

        elif axis == 'dec':
            self.slew_model_dec = slew_model
            self._print(True, 'Dec slew model set.')

        else:
            raise ValueError(
                    "Unsupported axis '{0:s}' for mount '{1:s}'.". format(
                            axis, self.mount))

        return None

    #--------------------------------------------------------------------------
    def is_set(self):
        """Check if slew models have been set.

        Returns
        -----
        out : bool
            True, if slew model has been set for each axis. False, otherwise.
        """

        ready = True

        if self.slew_model_ra is None:
            self._print(True, 'WARNING: No slew model for RA axis set.')
            ready = False

        if self.slew_model_dec is None:
            self._print(True, 'WARNING: No slew model for DEC axis set.')
            ready = False

        if self.slew_model_dome is None:
            self._print(True, 'WARNING: No slew model for dome set.')
            # only print(warning, dome is optional

        return ready

    #--------------------------------------------------------------------------
    def set_to_zenith(self):
        """Set telescope position to zenith.

        Returns
        -----
        None
        """

        if self.time is None:
            raise ValueError(
                    "Set time first before setting to zenith.")

        ra = self.time.sidereal_time('apparent')
        dec = self.loc.lat
        coord = SkyCoord(ra=ra, dec=dec)
        self.set_pos(coord)

    #--------------------------------------------------------------------------
    def get_slew_time(self, coord):
        """Calculate slew time from current position to given coordinates.

        Parameters
        -----
        coord : astropy.SkyCoord
            Source coordinates.

        Returns
        -----
        out : numpy.ndarray
            Slew time to each source from the current position.
        """

        # calculate slew times of individual axes:
        rot_ra = coord.ra - self.pos.ra
        rot_dec = self.pos.dec - coord.dec
        t_ra = self.slew_model_ra.get_slew_time(rot_ra)
        t_dec = self.slew_model_dec.get_slew_time(rot_dec)
        unit = t_ra._unit
        time = [t_ra.value, t_dec.value]

        if self.dome:
            t_dome = self.slew_model_dome.get_slew_time(rot_ra)
            time.append(t_dome.value)

        # take maximum of all axes:
        if coord.size == 1:
            time_max = np.max(time) * unit
        else:
            time_max = np.maximum.reduce(time) * unit

        return time_max

#==============================================================================

class SlewModel(object, metaclass=ABCMeta):
    """Slew model base class.
    """

    #--------------------------------------------------------------------------
    @abstractmethod
    def __init__(self):
        """Create a SlewModel instance.
        """

        return None

    #--------------------------------------------------------------------------
    @abstractmethod
    def __str__(self):
        """Write out information about the slew model.

        Returns
        -----
        out : str
        """

        return 'SlewModel instance.'

    #--------------------------------------------------------------------------
    @abstractmethod
    def get_slew_time(self, angle):
        """Calculate the slew time.

        Parameters
        -----
        angle : astropy.Angle
            Angle(s) by which to slew.

        Returns
        -----
        out : numpy.ndarray
            Time needed to slew by the given angle(s).
        """

        return None

#==============================================================================

class SlewModelLinear(SlewModel):
    """Linear slew model.
    """

    #--------------------------------------------------------------------------
    def __init__(self, t0_p, vel_p, t0_n=None, vel_n=None):
        """Create SlewModelLinear instance.

        Parameters
        -----
        t0_p : float
            Initial time in seconds needed to start slewing in positive
            direction.
        vel_p : float
            Slew velocity in degrees per second for slewing in positive
            direction.
        t0_n : float, default=None
            Initial time in seconds needed to start slewing in negaitve
            direction. Same as positive direction, if not set.
        vel_n : float, default=None
            Slew velocity in degrees per second for slewing in negative
            direction. Same as positive direction, if not set.
        """

        self.t0_p = abs(t0_p) * u.s
        self.vel_p = abs(vel_p) * u.deg / u.s
        if t0_n is None:
            self.t0_n = abs(t0_p)  * u.s
        else:
            self.t0_n = abs(t0_n)  * u.s
        if vel_n is None:
            self.vel_n = abs(vel_p) * u.deg / u.s
        else:
            self.vel_n = abs(vel_n) * u.deg / u.s

        return None

    #--------------------------------------------------------------------------
    def __str__(self):
        """Write out information about the slew model.

        Returns
        -----
        out : str
        """

        text = 'SlewModel: linear\n'
        if self.vel_n == self.vel_p and self.t0_n == self.t0_p:
            text += 'Velocity:        {0:10.4f}\n'.format(
                    self.vel_p)
            text += 'Additional time: {0:10.4f}\n'.format(self.t0_p)
        else:
            text += 'Positive direction:\n'
            text += 'Velocity:        {0:10.4f}\n'.format(
                    self.vel_p)
            text += 'Additional time: {0:10.4f}\n'.format(self.t0_p)
            text += 'Negative direction:\n'
            text += 'Velocity:        {0:10.4f}\n'.format(
                    self.vel_n)
            text += 'Additional time: {0:10.4f}\n'.format(self.t0_n)

        return text

    #--------------------------------------------------------------------------
    def get_slew_time(self, angle):
        """Calculate the slew time.

        Linear model:
        .. math::
            t = t_0 + a / v,
        where t_0 is an initial time offset, a is the angle to rotate by, and
        v is the rotation velocity. Parameters can be set individually for
        rotations in positive and negative directions.

        Parameters
        -----
        angle : astropy.Angle
            Angle(s) by which to slew.

        Returns
        -----
        out : numpy.ndarray
            Time needed to slew by the given angle(s).
        """

        time = np.ones(angle.size) * u.s

        if angle.size == 1:
            if angle > 0:
                time = angle  / self.vel_p + self.t0_p
            elif angle < 0:
                time = -angle  / self.vel_n + self.t0_n
            else:
                time = 0 * u.s

        elif angle.size > 1:
            sel = angle == 0
            time[sel] = 0

            sel = angle > 0
            time[sel] = angle[sel] / self.vel_p + self.t0_p

            sel = angle < 0
            time[sel] = -angle[sel] / self.vel_n + self.t0_n

        return time

#==============================================================================

class SlewModelConstAcc(object, metaclass=ABCMeta):
    """Slew model base class.
    """

    #--------------------------------------------------------------------------
    def __init__(
            self, acceleration, deceleration=None, velocity_max=None, t0=0.):
        """Create a SlewModel instance.

        Parameters
        ----------
        acceleration : float
            Angular acceleration in deg per seconds squared.
        deceleration : float, default=None
            Angular deceleration in deg per seconds squared. If None, the same
            value is used as for acceleration.
        velocity_max : float, default=None
            Maximum angular velocity in deg per second. If None, arbitrarily
            high velocities can be reached.
        t0 : float, default=0.
            Constant time offset added to the slew time.
        """

        self.acceleration = acceleration * u.deg / u.s**2
        if deceleration:
            self.deceleration = deceleration * u.deg / u.s**2
        else:
            self.deceleration = False
        if velocity_max:
            self.velocity_max = velocity_max * u.deg / u.s
        else:
            self.velocity_max = False
        self.t0 = t0 * u.s

        return None

    #--------------------------------------------------------------------------
    def __str__(self):
        """Write out information about the slew model.

        Returns
        -----
        out : str
        """

        text = 'SlewModel: constant acceleration\n'
        text += 'Acceleration:    {0:10.4f}\n'.format(
                    self.acceleration)
        if self.deceleration is not False:
            text += 'Deceleration:    {0:10.4f}\n'.format(
                        self.deceleration)
        if self.velocity_max is not False:
            text += 'Max. velocity:  {0:10.4f}\n'.format(
                    self.velocity_max)
        text += 'Additional time: {0:10.4f}\n'.format(self.t0)

        return text

    #--------------------------------------------------------------------------
    def _accelerate_decelerate(self, angle):
        """Calculate the slew time, when maximum velocity is not reached or not
        set.

        Parameters
        -----
        angle : astropy.Angle
            Angle(s) by which to slew.

        Returns
        -----
        out : numpy.ndarray
            Time needed to slew by the given angle(s).
        """

        # same acceleration and deceleration:
        if self.deceleration is False:
            time = 2. * np.sqrt(angle / self.acceleration)

        # different deceleration:
        else:
            norm_acc = self.acceleration + self.acceleration**2 \
                / self.deceleration
            norm_dec = self.deceleration + self.deceleration**2 \
                / self.acceleration
            time_acc = np.sqrt(2. * angle / norm_acc)
            time_dec = np.sqrt(2. * angle / norm_dec)
            time = time_acc + time_dec

        return time

    #--------------------------------------------------------------------------
    def get_slew_time(self, angle):
        """Calculate the slew time.

        Parameters
        -----
        angle : astropy.Angle
            Angle(s) by which to slew.

        Returns
        -----
        out : numpy.ndarray
            Time needed to slew by the given angle(s).
        """

        angle = np.absolute(angle)

        # expand dimension of scalar value to work as array in the following:
        if angle.size == 1:
            angle = np.expand_dims(angle, axis=0)

        # no maximum velocity:
        if self.velocity_max is False:
            time = self._accelerate_decelerate(angle)

        # with maximum velocity:
        else:
            angle_acc_max = self.velocity_max**2 / 2. / self.acceleration
            # same acceleration and deceleration:
            if self.deceleration is False:
                angle_dec_max = angle_acc_max
            # different deceleration:
            else:
                angle_dec_max = self.velocity_max**2 / 2. / self.deceleration
            angle_acc_dec_max = angle_acc_max + angle_dec_max

            time = np.zeros(angle.size) * u.s

            # where maximum velocity is not reached:
            sel = angle <= angle_acc_dec_max
            if np.any(sel):
                time[sel] = self._accelerate_decelerate(angle[sel])

            # where maximum velocity is reached:
            sel = ~sel
            if np.any(sel):
                # same acceleration and deceleration:
                if self.deceleration is False:
                    time[sel] = self.velocity_max / self.acceleration \
                        + angle[sel] / self.velocity_max
                # different deceleration:
                else:
                    t_acc_max = self.velocity_max / self.acceleration
                    t_dec_max = self.velocity_max / self.deceleration
                    #t_const = angle[sel] / self.velocity_max \
                    #    - (t_acc_max + t_dec_max) / 2.
                    #time[sel] = t_acc_max + t_dec_max + t_const
                    time[sel] = angle[sel] / self.velocity_max \
                        + (t_acc_max + t_dec_max) / 2.


        # add constant time offset:
        time += self.t0

        # reduce to scalar value, if initial input was scalar:
        if time.size == 1:
            time = time[0]

        return time
