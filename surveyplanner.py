# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Pasiphae survey planner.
"""

from astropy.coordinates import AltAz, Angle, EarthLocation, get_sun, SkyCoord
from astropy.time import Time, TimeDelta
from astropy import units as u
from itertools import repeat
from multiprocessing import Manager, Pool, Process
import numpy as np
from pandas import DataFrame
from scipy.stats import linregress
from statsmodels.api import add_constant, OLS
from time import sleep
from textwrap import dedent
from warnings import warn

import constraints as c
from db import DBConnectorSQLite
from utilities import true_blocks

import sys

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

class Field:
    """A field in the sky."""

    #--------------------------------------------------------------------------
    def __init__(
            self, fov, center_ra, center_dec, tilt=0., field_id=None,
            jd_first_obs_window=None, jd_next_obs_window=None, n_obs_tot=0,
            n_obs_done=0, n_obs_pending=0):
        """A field in the sky.

        Parameters
        ----------
        fov : float
            Field of view in radians. Diameter East to West and North to South.
        center_ra : float
            Right ascension of the field center in radians.
        center_dec : float
            Declination of the field center in radians.
        tilt : float, optional
            Tilt of the field such that the top and bottom borders are not
            parallel to the horizon. The default is 0..
        field_id : int, optional
            ID of the field. The default is None.
        jd_first_obs_window : float, optional
            The earliest Julian date for which an observing window was
            calculated for this field. The default is None.
        jd_next_obs_window : float, optional
            The latest Julian date for which an observing window was calculated
            for this field. The default is None.
        n_obs_tot : int, optional
            The number of observations associated with this field, irregardless
            of being pending or finished. The default is 0.
        n_obs_done : int, optional
            The number of observations finished for this field. The default is
            0.
        n_obs_pending : int, optional
            The number of observations pending for this field. The default is
            0.

        Returns
        -------
        None.
        """

        self.id = field_id
        self.fov = Angle(fov, unit='rad')
        self.center_coord = SkyCoord(center_ra, center_dec, unit='rad')
        self.center_ra = self.center_coord.ra
        self.center_dec = self.center_coord.dec
        self.tilt = Angle(tilt, unit='rad')
        self.jd_first_obs_window = jd_first_obs_window
        self.jd_next_obs_window = jd_next_obs_window
        self.obs_windows = []
        self.status = 0
        self.setting_in = -1
        self.n_obs_tot = n_obs_tot
        self.n_obs_done = n_obs_done
        self.n_obs_pending = n_obs_pending
        self.priority = -1

    #--------------------------------------------------------------------------
    def __str__(self):
        """Return information about the field instance.

        Returns
        -------
        info : str
            Description of main field properties.
        """

        info = dedent("""\
            Sky field {0:s}
            Field of view: {1:8.4f} arcmin
            Center RA:     {2:8.4f} deg
            Center Dec:    {3:+8.4f} deg
            Tilt:          {4:+8.4f} deg
            Status:        {5}
            Observations:  {6} total
                           {7} pending
                           {8} done
            """.format(
                f'{self.id}' if self.id is not None else '',
                self.fov.arcmin, self.center_ra.deg, self.center_dec.deg,
                self.tilt.deg, self._status_to_str(), self.n_obs_tot,
                self.n_obs_pending, self.n_obs_done))

        return info

    #--------------------------------------------------------------------------
    def _status_to_str(self):
        """Convert the field status ID to readable information.

        Returns
        -------
        status_str : str
            Description of the field status.
        """

        if self.status == 1:
            status_str = 'init'
        elif self.status == 2:
            status_str = 'not observable'
        elif self.status == 3:
            status_str = 'rising'
        elif self.status == 4:
            status_str = 'plateauing'
        elif self.status == 5:
            status_str = 'setting in {0:.2f}'.format(self.setting_in)

        return status_str

    #--------------------------------------------------------------------------
    def get_obs_window(
            self, telescope, frame, time_sunrise, refine):
        """Calculate time windows when the field is observable.

        Parameters
        ----------
        telescope : Telescope
            Telescope for which to calculate observability.
        frame : astropy.coordinates.AltAz
            Frame that provides the time steps at which observability is
            initially tested.
        time_sunrise : astropy.time.Time
            Date and time of sunrise.
        refine : astropy.time.TimeDelta
            Time accuracy at which the observing window is calculated. The
            precision of the observable time window is refined to this value.
            I.e. if the interval given in 'frame' is 10 minutes and refine
            is a TimeDelta corresponding to 1 minute, the window limits will
            be accurate to a minute.

        Returns
        -------
        obs_windows : list
            List of tuples. Each tuple contains two astropy.time.Time instances
            that mark the earliest time and latest time of a window during
            which the field is observable.

        Notes
        -----
        This method uses a frame as input instead of a start and stop time
        and interval, from which the frame could be created. The advantage is
        that the same initial frame can be used for all fields.
        """

        obs_windows = []
        temp_obs_windows = []
        observable = telescope.constraints.get(self.center_coord, frame)
        blocks = true_blocks(observable)

        for i, j in blocks:
            obs_window = (frame.obstime[i], frame.obstime[j])
            temp_obs_windows.append(obs_window)

        # increase precision for actual observing windows:
        if refine.value:
            time_interval = np.diff(frame.obstime.jd).max() * u.d

            # iterate through time windows:
            for t_start, t_stop in temp_obs_windows:
                # keep start time:
                if np.isclose(
                        t_start.jd, frame.obstime[0].jd,
                        atol = 1e-8, rtol = 1e-13):
                    pass
                # higher precision for start time:
                else:
                    t_start_new = t_start - time_interval
                    frame_temp = telescope.get_frame(
                            t_start_new, t_start, refine)
                    observable = telescope.constraints.get(
                            self.center_coord, frame_temp, check_frame=False)
                    k = np.argmax(observable)
                    t_start = frame_temp.obstime[k]

                # keep stop time:
                if np.isclose(
                        t_stop.jd, frame.obstime[-1].jd,
                        atol = 1e-8, rtol = 1e-13):
                    pass
                # higher precision for stop time:
                else:
                    t_stop_new = t_stop  + time_interval
                    frame_temp = telescope.get_frame(
                            t_stop, t_stop_new, refine)
                    observable = telescope.constraints.get(
                            self.center_coord, frame_temp, check_frame=False)
                    k = (frame_temp.obstime.value.size - 1
                            - np.argmax(observable[::-1]))
                    t_stop = frame_temp.obstime[k]

                if t_start != t_stop:
                    obs_windows.append((t_start, t_stop))

        # in case of no precision refinement:
        else:
            for t_start, t_stop in temp_obs_windows:
                if t_start != t_stop:
                    obs_windows.append((t_start, t_stop))

        return obs_windows

    #--------------------------------------------------------------------------
    def get_obs_duration(self):
        """Get the total duration of all observing windows.

        Returns
        -------
        duration : astropy.units.Quantity
            The duration in hours.
        """

        duration = 0 * u.day

        for obs_window in self.obs_windows:
            duration += obs_window.duration

        return duration

    #--------------------------------------------------------------------------
    def add_obs_window(self, obs_window):
        """Add observation window(s) to field.

        Parameters
        ----------
        obs_window : ObsWindow or list
            Observation window(s) that is/are added to the field. If multiple
            windows should be added provide a list of ObsWindow instances.

        Returns
        -------
        None.
        """

        if isinstance(obs_window, list):
            self.obs_windows += obs_window
        else:
            self.obs_windows.append(obs_window)

    #--------------------------------------------------------------------------
    def set_status(
            self, rising=None, plateauing=None, setting=None, setting_in=None,
            not_available=None):
        """Set the field status.

        Parameters
        ----------
        rising : bool, optional
            True, if the field is rising. The default is None.
        plateauing : bool, optional
            True, if the field is neither rising nor setting. The default is
            None.
        setting : bool, optional
            True, if the field is setting. The default is None.
        setting_in : astropy.unit.Quantity, optional
            The duration until the field is setting. The default is None.
        not_available : bool, optional
            True, if the status is unknown. The default is None.

        Returns
        -------
        None.
        """

        if not_available:
            self.status = -1
        elif rising:
            self.status = 1
        elif plateauing:
            self.status = 2
        elif setting:
            self.status = 3
            self.setting_in = setting_in

    #--------------------------------------------------------------------------
    def set_priority(self, priority):
        """Set priority.

        Parameters
        ----------
        priority : float
            Priority value assigned to this field.

        Returns
        -------
        None
        """

        self.priority = priority

#==============================================================================

class ObsWindow:
    """Time window of observability."""

    #--------------------------------------------------------------------------
    def __init__(self, start, stop, obs_window_id=None):
        """Time window of observability.

        Parameters
        ----------
        start : astropy.time.Time
            The start date and time of the observing window.
        stop : astropy.time.Time
            The stop date and time of the observing window.
        obs_window_id : int, optional
            Observing window ID as stored in the database. The default is None.

        Returns
        -------
        None
        """

        self.start = start
        self.stop = stop
        self.duration = (stop - start).value * u.day
        self.obs_window_id = obs_window_id

    #--------------------------------------------------------------------------
    def __str__(self):
        """Return information about the observing window.

        Returns
        -------
        info : str
            Description of the observing window.
        """

        return dedent("""\
            ObsWindow
            Start: {0}
            Stop:  {1}
            Duration: {2:.2f}""".format(
                self.start, self.stop, self.duration.to(u.hour)))

#==============================================================================

class Telescope:
    """A telescope.
    """

    #--------------------------------------------------------------------------
    def __init__(
            self, lat, lon, height, utc_offset, name='', telescope_id=None):
        """Create Telescope instance.

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
        utc_offset : float
            The local UTC offset in hours.
        name : str, default=''
            Name of the telescope/observatory.
        telescope_id : int
            ID in the database.

        Returns
        -----
        None
        """

        lat = Angle(lat)
        lon = Angle(lon)
        height = height * u.m

        self.loc = EarthLocation(
                lat=lat, lon=lon, height=height)
        self.name = name
        self.utc_offset = TimeDelta(utc_offset / 24. * u.day)
        self.telescope_id = telescope_id
        self.constraints = c.Constraints()

        print('Telescope {0:s} created.'.format(self.name))

    #--------------------------------------------------------------------------
    def __str__(self):
        """Return information about the telescope.

        Returns
        -------
        info : str
            Description of the telescope parameters.
        """

        return dedent("""\
            Telescope {0}
            Name: {1}
            Lat: {2:+12.4f}
            Lon: {3:12.4f}
            Height: {4:9.2f}
            UTC offset: {5:5.2f} hours
            """.format(
                '' if self.telescope_id is None else self.telescope_id,
                self.name, self.loc.lat, self.loc.lon, self.loc.height,
                self.utc_offset.value*24.))

    #--------------------------------------------------------------------------
    def add_constraint(self, constraint):
        """Add an observational constraint.

        Parameters
        -----
        constraint : Constraint instance
            Add a constraint that defines whether or not sources are observable
            at a given time.

        Returns
        -------
        None
        """

        self.constraints.add(constraint)
        print('Constraint added: {0:s}'.format(constraint.__str__()))

    #--------------------------------------------------------------------------
    def get_sun_set_rise(self, year, month, day, twilight):
        """Calculate time of the Sun set and Sun rise for a given date.

        Parameters
        ---------
        year : int
            Year for which to calculate the Sun set and rise time.
        month : int
            Month for which to calculate the Sun set and rise time.
        day : int
            Day for which to calculate the Sun set and rise time.
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
        NotImplemented
            If the Sun does not rise or set at all.

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

        # find next sun set and sun rise:
        time = Time(f'{year}-{month}-{day}T12:00:00') - self.utc_offset
        time += np.arange(0., 48., 0.1) * u.h
        frame = AltAz(obstime=time, location=self.loc)
        sun = get_sun(time).transform_to(frame)
        sun_alt = sun.alt
        night = sun_alt < sunset
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

        return time_sunset, time_sunrise

    #--------------------------------------------------------------------------
    def get_frame(self, time_start, time_stop, time_interval, round_up=False):
        """Create an AltAz frame for specified dates.

        Parameters
        ----------
        time_start : astropy.time.Time
            The start date for the frame.
        time_stop : astropy.time.Time
            The stop date for the frame.
        time_interval : astropy.time.TimeDelta
            The sampling time interval.
        round_up : bool, optional
            If True, the start time is rounded up according to the given time
            interval. Otherwise, the input start time is used. The default is
            True.

        Returns
        -------
        astropy.coordinates.AltAz
            The AltAz frame for specified dates.
        """

        # time offset of start time from next even time grid point at time
        # interval accuracy:
        dt0 = -np.mod(time_start.jd, time_interval.value) + time_interval.value

        # time offset of stop time from previous even time grid point at time
        # interval accuracy:
        dt1 = np.mod(time_stop.jd, time_interval.value)

        # create time steps:
        duration = time_stop.jd - time_start.jd - dt0
        dt = np.arange(0, duration, time_interval.value)

        if np.isclose(dt1, 0):
            dt = np.r_[0, dt+dt0]
        else:
            dt = np.r_[0, dt+dt0, dt[-1]+dt0+dt1]

        # create time and frame:
        time = time_start + dt * u.d
        frame = AltAz(obstime=time, location=self.loc)

        return frame

#==============================================================================

class SurveyPlanner:
    """Pasiphae survey planner.
    """

    #--------------------------------------------------------------------------
    def __init__(self, dbname):
        """Create SurveyPlanner instance.

        Parameters
        ----------
        dbname : str
            File name of the database.

        Returns
        -----
        None
        """

        self.dbname = dbname
        self.telescope = None
        self.twilight = None

    #--------------------------------------------------------------------------
    def _setup_observatory(self, observatory, no_constraints=False):
        """Load telescope parameters from database and add Telescope instance
        to SurveyPlanner.

        Parameters
        ----------
        observatory : str
            Observatory name as stored in the database.
        no_constraints : bool, optional
            If True, no constraints are loaded from the database and added to
            the telescope. The default is False.

        Returns
        -------
        parameter_set_id : int
            Parameter set ID corresponding to the used constraints.
        """

        # connect to database:
        db = DBConnectorSQLite(self.dbname)

        # create telescope:
        telescope = db.get_observatory(observatory)
        self.telescope = Telescope(
                telescope['lat'], telescope['lon'], telescope['height'],
                telescope['utc_offset'], name=telescope['name'])

        # load twilight, but skip loading other constraints:
        if no_constraints:
            parameter_set_id, constraints = db.get_constraints(observatory)
            self.twilight = constraints['Twilight']['twilight']
            return None

        # read constraints:
        parameter_set_id, constraints = db.get_constraints(observatory)
        self.twilight = constraints['Twilight']['twilight']
        del constraints['Twilight']

        # parse and add constraints:
        for constraint_name, params in constraints.items():

            # parse constraint code:
            text = f"c.{constraint_name}("

            for arg, val in params.items():
                if isinstance(val, str):
                    text += f"{arg}='{val}', "
                else:
                    text += f"{arg}={val}, "

            if text[-1] == "(":
                text += ")"
            else:
                text = f"{text[:-2]})"

            # evaluate code and add constraint to telescope:
            constraint = eval(text)
            self.telescope.add_constraint(constraint)

        return parameter_set_id

    #--------------------------------------------------------------------------
    def _tuple_to_field(self, field_tuple):
        """Convert a tuple that contains field information as queried from the
        database to a Field instance.

        Parameters
        ----------
        field_tuple : tuple
            Tuple as returned e.g. by db.get_fields().

        Returns
        -------
        field : Field
            A field instance created from the database entries.
        """

        field_id, fov, center_ra, center_dec, tilt, __, __, \
                jd_first_obs_window, jd_next_obs_window, n_obs_tot, \
                n_obs_done, n_obs_pending = field_tuple
        field = Field(
            fov, center_ra, center_dec, tilt, field_id=field_id,
            jd_first_obs_window=jd_first_obs_window,
            jd_next_obs_window=jd_next_obs_window, n_obs_tot=n_obs_tot,
            n_obs_done=n_obs_done, n_obs_pending=n_obs_pending)

        return field

    #--------------------------------------------------------------------------
    def _tuples_to_obs_windows(self, obs_windows_tuples):
        """Convert a tuple that contains observation window information as
        queried from the database to an ObsWindow instance.

        Parameters
        ----------
        obs_windows_tuples : tuple
            Tuple as returned e.g. by db.get_obs_windows_from_to().

        Returns
        -------
        obs_windows : ObsWindow
            Observation window.
        """

        obs_windows = []

        for obs_window_tuple in obs_windows_tuples:
            obs_window_id, __, date_start, date_stop, __, __ = \
                    obs_window_tuple
            date_start = Time(date_start)
            date_stop = Time(date_stop)
            obs_window = ObsWindow(
                    date_start, date_stop, obs_window_id=obs_window_id)
            obs_windows.append(obs_window)

        return obs_windows

    #--------------------------------------------------------------------------
    def _iter_fields(self, observatory=None, active=True):
        """Connect to database and iterate through fields.

        Parameters
        ----------
        observatory : str, optional
            Iterate only through fields associated with this observatory name.
            Otherwise, iterate through all fields. The default is None.
        active : bool, optional
            If True, only iterate through active fields. If False, only iterate
            through inactive fields. If None, iterate through all fields. The
            default is True.

        Yields
        ------
        field : Field
            Fields as stored in the database.
        """

        # read fields from database:
        db = DBConnectorSQLite(self.dbname)
        fields = db.iter_fields(observatory=observatory, active=active)

        for __, __, field in fields:
            field = self._tuple_to_field(field)

            yield field

    #--------------------------------------------------------------------------
    def _iter_observable_fields_by_night(
            self, observatory, night, observed=None, pending=None,
            active=True):
        """Iterate through fields observable during a given night, given
        specific selection criteria.

        Parameters
        ----------
        observatory : str
            Observatory name.
        night : astropy.time.Time
            Iterate through fields observable during the night that starts on
            the specified day. Time information is truncated.
        observed : bool or None, optional
            If True, iterate only through fields that have been observed at
            least once. If False, iterate only through fields that have never
            been observed. If None, iterate through fields irregardless of
            whether they have  been observed or not. The default is None.
        pending : bool or None, optional
            If True, iterate only through fields that have pending observations
            associated. If False, only iterate through fields that have no
            pending observations associated. If None, iterate through fields
            irregardless of whether they have pending observations associated
            or not. The default is None.
        active : bool or None, optional
            If True, only iterate through active fields. If False, only iterate
            through inactive fields. If None, iterate through fields active or
            not. The default is True.

        Yields
        ------
        field : Field
            Field(s) fulfilling the selected criteria.
        """

        # check that night input is date only:
        if night.iso[11:] != '00:00:00.000':
            print("WARNING: For argument 'night' provide date only. " \
                  "Time information is stripped. To get fields " \
                  "observable at specific time use 'time' argument.")
            night = Time(night.iso[:10])

        # connect to database:
        db = DBConnectorSQLite(self.dbname)

        # get observatory information:
        observatory_name = observatory
        observatory = db.get_observatory(observatory)
        utc_offset = observatory['utc_offset'] * u.h

        # get local noon of current and next day in UTC:
        noon_current = night + 12 * u.h - utc_offset
        noon_next = night + 36 * u.h - utc_offset
        # NOTE: ignoring daylight saving time

        # iterate through fields:
        for __, __, field in db.iter_fields(
                observatory=observatory_name, observed=observed,
                pending=pending, active=active):
            field_id = field[0]
            obs_windows = db.get_obs_windows_from_to(
                    field_id, noon_current, noon_next)

            if len(obs_windows) == 0:
                continue

            field = self._tuple_to_field(field)
            obs_windows = self._tuples_to_obs_windows(obs_windows)
            field.add_obs_window(obs_windows)
            date = night + 1. * u.d - utc_offset
            self._set_field_status(
                    field, date, days_before=3, days_after=7)

            yield field

    #--------------------------------------------------------------------------
    def _iter_observable_fields_by_datetime(
            self, observatory, datetime, observed=None, pending=None,
            active=True):
        """Iterate through fields observable during a given night, given
        specific selection criteria.

        Parameters
        ----------
        observatory : str
            Observatory name.
        datetime : astropy.time.Time
            Iterate through fields that are observable at the given time.
        observed : bool or None, optional
            If True, iterate only through fields that have been observed at
            least once. If False, iterate only through fields that have never
            been observed. If None, iterate through fields irregardless of
            whether they have  been observed or not. The default is None.
        pending : bool or None, optional
            If True, iterate only through fields that have pending observations
            associated. If False, only iterate through fields that have no
            pending observations associated. If None, iterate through fields
            irregardless of whether they have pending observations associated
            or not. The default is None.
        active : bool or None, optional
            If True, only iterate through active fields. If False, only iterate
            through inactive fields. If None, iterate through fields active or
            not. The default is True.

        Yields
        ------
        field : Field
            Field(s) fulfilling the selected criteria.
        """

        # connect to database:
        db = DBConnectorSQLite(self.dbname)
        observatory_name = observatory

        # iterate through fields:
        for __, __, field in db.iter_fields(
                observatory=observatory_name, observed=observed,
                pending=pending, active=active):
            field_id = field[0]
            obs_windows = db.get_obs_windows_by_datetime(
                    field_id, datetime)

            if len(obs_windows) == 0:
                continue

            field = self._tuple_to_field(field)
            obs_windows = self._tuples_to_obs_windows(obs_windows)
            field.add_obs_window(obs_windows)
            self._set_field_status(
                    field, datetime, days_before=3, days_after=7)

            yield field

    #--------------------------------------------------------------------------
    def _get_obs_window_durations(self, db, field_id, jd_start, jd_stop):
        """Get observability window durations of a specific field between two
        dates.

        Parameters
        ----------
        db : db.DBConnectorSQLite
            Active database connection.
        field_id : int
            Field ID.
        jd_start : float
            Query observing windows from this JD on.
        jd_stop : float
            Query observing windows up to this JD.

        Returns
        -------
        durations : pandas.DataFrame
            Containts the queried durations and columns for further processing.
        """

        durations = db.get_obs_window_durations(field_id, jd_start, jd_stop)
        durations = DataFrame(
                durations, columns=(
                    'observability_id', 'jd', 'status', 'duration')
                )
        durations = durations.groupby(
                ['observability_id', 'jd', 'status']).sum()
        durations.reset_index(inplace=True)
        durations['setting_in'] = \
                np.zeros(durations.shape[0]) * np.nan
        durations['update'] = np.zeros(durations.shape[0], dtype=bool)

        return durations

    #--------------------------------------------------------------------------
    def _update_unknown_status(self, durations):
        """Update the status where unknown, if possible.

        Parameters
        ----------
        durations : pandas.DataFrame
            Observing window durations and status information.

        Notes
        -------
        Changes are done inplace, therefore the dataframe does not have to be
        returned.
        """

        # identify blocks with unknown status:
        sel_unk = durations['status'] == 'unknown'

        # iterate through these blocks in reverse order:
        for i0, i1 in true_blocks(sel_unk)[::-1]:

            # skip blocks at the end:
            if i1 == durations.shape[0] - 1:
                continue

            # otherwise, update status based on next status that is not
            # 'not observable':
            for j in range(1, 5):
                # try to get next status, break if no more are available:
                try:
                    status = durations['status'][i1+j]
                except KeyError:
                    break

                # if status is 'not observable', go to next status:
                if status == 'not observable':
                    continue

                # update status:
                if status != 'unknown':
                    durations.loc[i0:i1, 'status'] = status
                    durations.loc[i0:i1, 'update'] = True

                # extrapolate setting duration, if needed:
                if status == 'setting':
                    setting_in = durations['setting_in'][i1+j] + j - 1
                    setting_in += np.arange(i1 - i0 + 1, 0, -1)
                    durations.loc[i0:i1, 'setting_in'] = setting_in

                break

            # all next statuses are 'not observable', set to 'setting':
            else:
                durations.loc[i0:i1, 'status'] = 'setting'
                durations.loc[i0:i1, 'update'] = True
                durations.loc[i0:i1, 'setting_in'] = np.arange(i1 - i0, -1, -1)

    #--------------------------------------------------------------------------
    def _obs_window_time_range_init(self, date_stop, date_start):
        """Determine the JD range for the first observing window calculation
        for new fields.

        Parameters
        ----------
        date_stop : astropy.time.Time
            The stop date and time of the observing window calculation.
        date_start : astropy.time.Time
            The stop date and time of the observing window calculation.

        Raises
        ------
        ValueError
            Raised if no start date is provided.
        ValueError
            Raised if the stop date is earlier than the start date.

        Returns
        -------
        jd_start : float
            The JD at which to start the observing window calculation.
        jd_stop : float
            The JD at which to stop the observing window calculation.
        n_days : int
            Number of days from start to stop date.
        agreed_to_gaps : bool
            True, if the user agreed to gaps in the observing window
            calculations by skipping over some days. False, otherwise.

        Notes
        -----
        - This method is called by `_add_obs_windows()`.
        """

        # check input:
        if date_start is None:
            raise ValueError(
                    "A start date is required either due to new fields or "
                    "changed constraints. Set `date_start` when calling "
                    "check_observability().")

        if date_start >= date_stop:
            raise ValueError("Start date must be earlier than stop date.")

        jd_stop = date_stop.jd
        jd_start = date_start.jd
        n_days = int(jd_stop - jd_start)

        return jd_start, jd_stop, n_days

    #--------------------------------------------------------------------------
    def _obs_window_time_range(
            self, db, date_stop, date_start, fields, agreed_to_gaps):
        """Determine the JD range for the observing window calculations.

        Parameters
        ----------
        db : db.DBConnectorSQLite
            Active database connection.
        date_stop : astropy.time.Time
            The stop date and time of the observing window calculation.
        date_start : astropy.time.Time
            The stop date and time of the observing window calculation.
        fields : list of Field instances
            The fields for which observing windows will be calculated.
        agreed_to_gaps : bool
            True, if the user agreed to gaps in the observing window
            calculations by skipping over some days. False, otherwise.

        Raises
        ------
        ValueError
            Raised if the stop date is earlier than the start date.

        Returns
        -------
        jd_start : float
            The JD at which to start the observing window calculation.
        jd_stop : float
            The JD at which to stop the observing window calculation.
        n_days : int
            Number of days from start to stop date.
        agreed_to_gaps : bool
            True, if the user agreed to gaps in the observing window
            calculations by skipping over some days. False, otherwise.

        Notes
        -----
        - This method is called by `_add_obs_windows()`.
        """

        jd_done = db.get_next_observability_jd()
        date_done = Time(jd_done, format='jd')
        jd_stop = date_stop.jd

        # no start date given, use earliest one required:
        if date_start is None:
            jd_start = jd_done

        # start date is later than earliest required date - ask user:
        elif date_start.jd > jd_done:
            date_done_str = date_done.iso.split(' ')[0]
            print('\n! ! WARNING ! !')
            print(f'Observabilities are stored until JD {jd_done:.1f}, '\
                  f'i.e. {date_done_str}')
            print(f'The start date is set to JD {date_start.jd:.1f}, i.e. '\
                  f'{date_start.iso.split(" ")[0]}.')
            print('Observabilities will not be stored for the days '\
                  'between. This may have severe impacts on the scheduling.')

            # ask user:
            if agreed_to_gaps is None:
                user_in = input(
                        'To continue with given start date despite the risks '\
                        'type "yes", type "all" to apply this choice to all '\
                        'observatories, otherwise press ENTER to calculate '\
                        'observing windows starting with the earliest '\
                        'required date", type "safe" to apply this choice to '\
                        'all observatories. Press Crtl+C to abort.\n')
                print('')

                if user_in.lower() == 'yes':
                    jd_start = date_start.jd
                elif user_in.lower() == 'all':
                    jd_start = date_start.jd
                    agreed_to_gaps = True
                elif user_in.lower() == 'safe':
                    jd_start = jd_done
                    agreed_to_gaps = False
                else:
                    jd_start = jd_done

            # user already agreed to continue with given date:
            elif agreed_to_gaps:
                print('You previously agreed to continue with the given '\
                      'start date.\n')
                jd_start = date_start.jd

            # user already agreed to continue with required date:
            else:
                print('You previously agreed to continue with the required '\
                      'start date.\n')
                jd_start = jd_done

        # start date is earlier than earliest required date - use required:
        else:
            jd_start = jd_done

        # check that stop JD is after start JD:
        if jd_stop < jd_start:
            raise ValueError('Start date is later than stop date.')

        n_days = int(jd_stop - jd_start)

        return jd_start, jd_stop, n_days, agreed_to_gaps

    #--------------------------------------------------------------------------
    def _read_obs_windows_from_queue(
            self, db, queue_obs_windows, n, duration_limit):
        """Read results from the queues to be written to the database.

        Parameters
        ----------
        db : db.DBConnectorSQLite
            Active database connection.
        queue_obs_windows : multiprocessing.managers.AutoProxy[Queue]
            Queue containing the observing windows.
        queue_field_ids : multiprocessing.managers.AutoProxy[Queue]
            Queue containing the IDs of fields whose `jd_next_obs_window` value
            needs to be updated.
        n : int
            Number of entries to read from the queue.
        duration_limit : astropy.time.TimeDelta
            Limit on the observing window duration above which a field is
            considered observable.

        Returns
        -------
        batch_field_ids : list of int
            IDs of the fields for Observability table entries and updating
            `jd_next_obs_window` value.
        batch_dates : list of float
            Current JD for Observability table entries.
        batch_status : list of str
            Status IDs for Observability table entries.
        batch_obs_windows_field_ids : list of int
            IDs of fields corresponding to the observing windows.
        batch_obs_windows_obs_ids: list of int
            IDs of the related entries in Observability table.
        batch_obs_windows_start : list of astropy.time.Time instances
            Observing windows start dates and times.
        batch_obs_windows_stop : list of astropy.time.Time instances
            Observing windows start dates and times.
        batch_obs_windows_duration : list of astropy.time.TimeDelta
            Observing windows durations in days.

        Notes
        -----
        - This method is called by `_add_observabilities_to_db()`.
        """

        # data storage for Observability table columns:
        batch_field_ids = []
        batch_status = []

        # data storage for ObsWindows
        batch_obs_windows_field_ids = []
        batch_obs_windows_obs_ids = []
        batch_obs_windows_start = []
        batch_obs_windows_stop = []
        batch_obs_windows_duration = []

        n_queued = queue_obs_windows.qsize()

        if n_queued < n:
            n = n_queued

        # get next observability_id:
        observability_id = db.get_next_observability_id()

        for __ in range(n):
            field_id, obs_windows = queue_obs_windows.get()
            batch_field_ids.append(field_id)
            observable = False

            for start, stop in obs_windows:
                duration = (stop - start)

                if duration >= duration_limit:
                    batch_obs_windows_field_ids.append(field_id)
                    batch_obs_windows_obs_ids.append(observability_id)
                    batch_obs_windows_start.append(start)
                    batch_obs_windows_stop.append(stop)
                    batch_obs_windows_duration.append(duration)
                    observable = True

            if observable:
                batch_status.append('init')
            else:
                batch_status.append('not observable')

            observability_id += 1

        return (batch_field_ids, batch_status, batch_obs_windows_field_ids,
                batch_obs_windows_obs_ids, batch_obs_windows_start,
                batch_obs_windows_stop, batch_obs_windows_duration)

    #--------------------------------------------------------------------------
    def _add_observabilities_to_db(
            self, db, counter, queue_obs_windows, jd, parameter_set_id,
            duration_limit, n_tbd, batch_write):
        """Write observability status and observation windows to database.

        Parameters
        ----------
        db : db.DBConnectorSQLite
            Active database connection.
        counter : multiprocessing.managers.ValueProxy
            Counter that stores how many fields have been processed.
        queue_obs_windows : multiprocessing.managers.AutoProxy[Queue]
            Queue containing the observing windows.
        jd : float
            JD of the current observing window calculation.
        parameter_set_id : int
            Parameter set ID of the used constraints.
        duration_limit : astropy.time.TimeDelta
            Limit on the observing window duration above which a field is
            considered observable.
        n_tbd : int
            Number of fields whose calculated observing windows need to be
            written to the database.
        batch_write : int
            Number of entries that should be batch written to the database.

        Returns
        -------
        None

        Notes
        -----
        - This method is running in a separate process started by
          `_add_obs_windows()`.
        """

        done = False
        jd_next = jd + 1

        while not done:
            sleep(1)
            n_done = counter.value
            n_queued = queue_obs_windows.qsize()

            if n_done >= n_tbd:
                done = True

            print(f'\rProgress: field {n_done} of {n_tbd} ' \
                  f'({n_done/n_tbd*100:.1f}%)..           ', end='')

            # extract batch of results from queue:
            if done or n_queued >= batch_write:
                print('\rProgress: reading from queue..                      ',
                      end='')
                batch_field_ids, batch_status, \
                batch_obs_windows_field_ids, batch_obs_windows_obs_ids, \
                batch_obs_windows_start, batch_obs_windows_stop, \
                batch_obs_windows_duration = self._read_obs_windows_from_queue(
                        db, queue_obs_windows, batch_write, duration_limit)
                n_queried = len(batch_field_ids)
                write = True

            else:
                write = False

            # write results to database:
            if write:
                print(f'\rProgress: writing {n_queried} entries to database..',
                      end='')

                # add observabilities to database:
                batch_dates = [jd]*len(batch_field_ids)
                batch_parameter_set_ids \
                        = [parameter_set_id]*len(batch_field_ids)
                db.add_observability(
                        batch_parameter_set_ids, batch_field_ids, batch_dates,
                        batch_status, active=True)

                # add observing windows to database:
                db.add_obs_windows(
                        batch_obs_windows_field_ids, batch_obs_windows_obs_ids,
                        batch_obs_windows_start, batch_obs_windows_stop,
                        batch_obs_windows_duration, active=True)

                # update Field information in database:
                db.update_time_ranges(
                        batch_field_ids, [jd_next]*len(batch_field_ids))

                n_done += n_queried

        print('\rProgress: done                                              ')

    #--------------------------------------------------------------------------
    def _find_obs_window_for_field(
            self, counter, counter_lock, queue_obs_windows, jd, frame,
            time_sunrise, field, init, refine):
        """Calculate observing window(s) for a specific field for a specific
        time frame.

        Parameters
        ----------
        counter : multiprocessing.managers.ValueProxy
            Counter that stores how many fields have been processed.
        counter_lock : multiprocessing.managers.AcquirerProxy
            A lock for the counter.
        queue_obs_windows : multiprocessing.managers.AutoProxy[Queue]
            Queue containing the observing windows.
        jd : float
            JD of the day for which the observing window is calculated.
        frame : astropy.coordinates.AltAz
            Frame that provides the time steps at which observability is
            initially tested.
        time_sunrise : astropy.time.Time
            Date and time of sunrise.
        field : Field instance
            The field for which the observing window is calculate.
        init : bool
            True, if the field does not have any observing windows calculated
            yet, which supresses the check that the calculation is not a
            repetition. Otherwise, this method makes sure that no days are
            repeated.
        refine : astropy.time.TimeDelta
            Time accuracy at which the observing window is calculated.

        Returns
        -------
        None

        Notes
        -----
        - This method is run by a pool of workers started by
          `_add_obs_windows()`.
        """

        # skip if this field was already covered for this JD:
        if not init and field.jd_next_obs_window > jd:
            pass

        # get observing windows and add them to queue:
        else:
            obs_windows = field.get_obs_window(
                    self.telescope, frame, time_sunrise, refine=refine)
            queue_obs_windows.put((field.id, obs_windows))

        with counter_lock:
            counter.value += 1

    #--------------------------------------------------------------------------
    def _add_obs_windows(
            self, db, fields, init, observatory, date_stop, date_start,
            duration_limit, batch_write, processes, time_interval_init,
            time_interval_refine, agreed_to_gaps):
        """

        Parameters
        ----------
        db : db.DBConnectorSQLite
            Active database connection.
        fields list of tuples
            List of the queried fields. Each tuple contains the field
            parameters.
        init : bool
            If True, observing windows are calculated for these fields for the
            first time, which requires some additional action. Otherwise, new
            observing windows are appended.
        observatory dict
            The telescope parameters as stored in the database.
        date_stop : astropy.time.Time
            The date until which observing windows should be calculated.
        date_start : astropy.time.Time
            The date from which on observing windows should be calculated. If
            this is later than the latest entries in the database, this will
            lead to gaps in the observing window calculation. The user is
            warned about this. It is safer to use this option only for the
            initial run and not afterwards.
        duration_limit : astropy.time.TimeDelta
            Limit on the observing window duration above which a field is
            considered observable.
        batch_write : int
            Observing window are temporarily saved and then written to the
            database in batches of this size.
        processes : int
            Number of processes that run the obsering window calcuations for
            different fields in parallel.
        time_interval_init : astropy.time.TimeDelta
            Time accuracy at which the observing windows are initially
            calculated. The accuracy can be refined with the next parameter.
        time_interval_refine : astropy.time.TimeDelta
            Time accuracy at which the observing windows are calculated after
            the initial coarse search.
        agreed_to_gaps : bool or None
            Whether the user agreed to time gaps in the observing window
            calculation or not.

        Returns
        -------
        agreed_to_gaps : bool
            Whether the user agreed to time gaps in the observing window
            calculation or not.

        Notes
        -----
        - This method is called by `check_observability()`.
        - This method starts `_add_observabilities_to_db()` in a separate
          process.
        - This method starts `_find_obs_window_for_field()` in a pool of
          processes.
        """

        n_fields = len(fields)

        # print out some information:
        if n_fields and init:
            print(f'Initial observing windows required for {n_fields} new '\
                  'fields.')
        elif n_fields:
            print(f'Observing windows required for {n_fields} fields.')
        elif not init:
            jd_done = db.get_next_observability_jd()
            date_done = Time(jd_done, format='jd')
            print('Observing windows for no further fields required. Stored '
                  f'already at least up to JD {jd_done:.0f}, i.e. '
                  f'{date_done.iso.split(" ")[0]}.')
            return None
        else:
            return None

        # setup observatory with constraints:
        parameter_set_id = self._setup_observatory(observatory)

        # get time range for observing window calculation:
        if init:
            jd_start, jd_stop, n_days = self._obs_window_time_range_init(
                    date_stop, date_start)
        else:
            jd_start, jd_stop, n_days, agreed_to_gaps \
            = self._obs_window_time_range(
                    db, date_stop, date_start, fields, agreed_to_gaps)

        # add start JD to database for new fields:
        if init:
            field_ids = [field.id for field in fields]
            db.init_observability_jd(field_ids, parameter_set_id, jd_start)

        print(f'Calculate observing windows for {n_days} days..')

        # iterate through days:
        for i, jd in enumerate(np.arange(jd_start, jd_stop, 1.), start=1):
            print(f'Day {i} of {n_days}, JD {jd:.0f}')

            # setup time frame:
            date = Time(jd, format='jd').datetime
            time_sunset, time_sunrise = \
                self.telescope.get_sun_set_rise(
                        date.year, date.month, date.day, self.twilight)
            frame = self.telescope.get_frame(
                    time_sunset, time_sunrise, time_interval_init,
                    round_up=False)

            # parallel process fields:
            manager = Manager()
            queue_obs_windows = manager.Queue()
            counter = manager.Value(int, 0)
            counter_lock = manager.Lock()
            writer = Process(
                    target=self._add_observabilities_to_db,
                    args=(db, counter, queue_obs_windows, jd, parameter_set_id,
                          duration_limit, n_fields, batch_write)
                    )
            writer.start()

            with Pool(processes=processes) as pool:
                pool.starmap(
                        self._find_obs_window_for_field,
                        zip(repeat(counter), repeat(counter_lock),
                            repeat(queue_obs_windows), repeat(jd),
                            repeat(frame), repeat(time_sunrise), fields,
                            repeat(init), repeat(time_interval_refine)))

            writer.join()

        return agreed_to_gaps

    #--------------------------------------------------------------------------
    def _get_fields_missing_status(self, db):
        """Get fields that have associated observing windows without
        observability status information.

        Parameters
        ----------
        db : db.DBConnectorSQLite
            Active database connection.

        Returns
        -------
        field_ids : numpy.ndarray
            Field IDs of fields with missing observing status.
        jd_min : numpy.ndarray
            JD of the first missing observing status corresponding to each
            field.
        jd_max : numpy.ndarray
            JD of the last missing observing status corresponding to each
            field.

        Notes
        -----
        - This method is called by `_add_status()`
        """

        fields = db.get_fields_missing_status()
        fields = DataFrame(fields, columns=('field_id', 'jd'))
        fields = fields.groupby('field_id').agg(
                jd_min=('jd', 'min'), jd_max=('jd', 'max'))
        field_ids = fields.index.to_numpy()
        jd_min = fields['jd_min'].to_numpy()
        jd_max = fields['jd_max'].to_numpy()

        return field_ids, jd_min, jd_max

    #--------------------------------------------------------------------------
    def _read_status_from_queue(self, db, queue_status, n, status_to_id):
        """Read results from the queues to be written to the database.

        Parameters
        ----------
        db : db.DBConnectorSQLite
            Active database connection.
        queue_status : multiprocessing.managers.AutoProxy[Queue]
            Queue containing the observability status.
        n : int
            Number of entries to read from the queue.
        status_to_id : dict
            Dictionary for conversion from status to corresponding ID.

        Returns
        -------
        batch_observability_ids : list of int
            IDs of the observability entries that need to be modified.
        batch_status : list of str
            IDs of the status of each observability entry.
        batch_setting_in : list of float
            Duration in days until the source is setting.

        Notes
        -----
        - This method is called by `_update_status_in_db()`.
        """

        # data storage for Observability table columns:
        batch_observability_ids = []
        batch_status = []
        batch_setting_in = []

        n_queued = queue_status.qsize()

        if n_queued < n:
            n = n_queued

        # get next observability_id:
        observability_id = db.get_next_observability_id()

        for __ in range(n):
            observability_id, status, setting_in = queue_status.get()
            batch_observability_ids.append(observability_id)
            batch_status.append(status_to_id[status])
            if np.isnan(setting_in):
                batch_setting_in.append(None)
            else:
                batch_setting_in.append(setting_in)

        return (batch_observability_ids, batch_status,
                batch_setting_in)

    #--------------------------------------------------------------------------
    def _update_status_in_db(
            self, db, counter, queue_status, n_tbd, batch_write):
        """Read observability status from queue and add it to database.

        Parameters
        ----------
        db : db.DBConnectorSQLite
            Active database connection.
        counter : multiprocessing.managers.ValueProxy
            Counter that stores how many fields have been processed.
        queue_status : multiprocessing.managers.AutoProxy[Queue]
            Queue containing the observability status.
        n_tbd : int
            Number of observabilities whose calculated status needs to be
            written to the database.
        batch_write : int
            Observing window are temporarily saved and then written to the
            database in batches of this size.

        Returns
        -------
        None

        Notes
        -----
        - This method is running in a separate process started by
          `_add_status()`.
        """

        # status string to status ID converter:
        status_to_id = {value: key for (key, value) in db.get_status()}

        done = False

        while not done:
            sleep(1)
            n_done = counter.value
            n_queued = queue_status.qsize()

            if n_done >= n_tbd:
                done = True

            print(f'\rProgress: field {n_done} of {n_tbd} ' \
                  f'({n_done/n_tbd*100:.1f}%)..           ', end='')

            # extract batch of results from queue:
            if done or n_queued >= batch_write:
                print('\rProgress: reading from queue..                      ',
                      end='')
                batch_observability_ids, batch_status, \
                batch_setting_in = self._read_status_from_queue(
                        db, queue_status, batch_write, status_to_id)
                n_queried = len(batch_observability_ids)
                write = True

            else:
                write = False

            # write results to database:
            if write:
                print(f'\rProgress: writing {n_queried} entries to database..',
                      end='')

                # update observabilities in database:
                db.update_observability_status(
                        batch_observability_ids, batch_status,
                        batch_setting_in)

                n_done += n_queried

        print('\rProgress: done                                              ')

    #--------------------------------------------------------------------------
    def _detect_outliers(self, jd, duration, outlier_threshold=0.6):
        """Outlier detection using Cook's distance.

        Parameters
        ----------
        jd : array-like
            JD.
        duration : array-like
            Durations of the observing windows for each JD.
        outlier_threshold : float, optional
            Threshold for outlier detection. Data points with a Cook's distance
            p-value lower than this threshold are considered outliers. The
            default is 0.6.

        Returns
        -------
        outliers : numpy.ndarray (dtype: bool)
            True, for detected outliers. False, otherwise.

        Notes
        -----
        - This method is called by `_update_status_for_field()`.
        """

        x = np.asarray(jd)
        y = np.asarray(duration)

        # linear regression:
        x = add_constant(x)
        model = OLS(y, x).fit()

        # check for outliers:
        influence = model.get_influence()
        __, cooks_pval = influence.cooks_distance
        outliers = cooks_pval < outlier_threshold

        return outliers

    #--------------------------------------------------------------------------
    def _get_status(
            self, jd, jds, durations, time_interval, status_threshold=6,
            mask_outl=None):
        """Status based on linear ordinary least square regression with
        exclusion of outliers.

        Parameters
        ----------
        jd : float
            JD of the date of interest.
        jds : array-like
            JDs of the observing windows.
        durations : array-like
            Durations of the observing windows for each JD.
        time_interval : astropy.time.TimeDelta
            Time accuracy at which the observing windows were calculated. The
            status is considered "plateauing" if the standard deviation
            of the durations is smaller than this time interval times the
            status_threshold.
        status_threshold : float, optional
            Scaling factor. See description of time_interval. The default is 6.
        mask_outl : numpy.ndarray (dtype: bool) or None, optional
            If given, Trues mark outliers that are removed from the OLS. The
            default is None.

        Returns
        -------
        status : str
            Either "rising", "plateauing", or "setting".
        setting_in : float
            If the status is "setting" this is the duration in days, when the
            field is setting. Otherwise a numpy.nan is returned.

        Notes
        -----
        - This method is called by `_update_status_for_field()`.
        """

        x = np.asarray(jds)
        y = np.asarray(durations)

        # remove outliers:
        if mask_outl is not None:
            x = x[~mask_outl]
            y = y[~mask_outl]

        # get status:
        if y.std() <= time_interval.value * status_threshold:
            status = 'plateauing'
            setting_in = np.nan

        elif np.median(np.diff(y)) > 0:
            status = 'rising'
            setting_in = np.nan

        else:
            status = 'setting'

            # linear regression for setting duration:
            x = add_constant(x)
            model = OLS(y, x).fit()
            slope = model.params[1]
            intercept = model.params[0]
            setting_in = -intercept / slope - jd

        return status, setting_in

    #--------------------------------------------------------------------------
    def _update_status_for_field(
            self, counter, counter_lock, queue_status, db, field_id, jd_before,
            jd_after, days_before, days_after, outlier_threshold,
            status_threshold, time_interval):
        """Determine the observability status for a specific field for each
        day, for which the status is not yet known, and store it in the
        database.

        Parameters
        ----------
        counter : multiprocessing.managers.ValueProxy
            Counter that stores how many fields have been processed.
        counter_lock : multiprocessing.managers.AcquirerProxy
            A lock for the counter.
        queue_status : multiprocessing.managers.AutoProxy[Queue]
            Queue containing the observability status.
        db : db.DBConnectorSQLite
            Active database connection.
        field_id : int
            Field ID.
        jd_before : float
            Field observabilities are queried from this JD.
        jd_after : float
            Field observabilities are queried up to this JD.
        days_before : int
            Determining the field's status requires the analysis of observation
            windows in a time range. This arguments sets how many days before
            are considered.
        days_after : int
            Determining the field's status requires the analysis of observation
            windows in a time range. This arguments sets how many days after
            are considered.
        outlier_threshold : float
            Threshold for outlier detection. Days with observing window
            durations outlying from the general trend are not considered in
            the status analysis.
        status_threshold : float
            Threshold for distinguishing plateauing from rising or setting. If
            the OLS p-value is larger than this threshold, the status is
            considered "plateauing"; otherwise "rising" or "setting" depending
            on the slope.
        time_interval : astropy.time.TimeDelta
            Time accuracy at which the observing windows were calculated.

        Returns
        -------
        None

        Notes
        -----
        - This method is run by a pool of workers started by `_add_status()`.
        """

        durations = self._get_obs_window_durations(
                db, field_id, jd_before, jd_after)
        sel_init = durations['status'] == 'init'

        # iterate through days:
        for i in np.nonzero(sel_init.to_numpy())[0]:
            jd = durations.iloc[i]['jd']
            jd_before = jd - days_before
            jd_after = jd + days_after
            sel_time = np.logical_and.reduce([
                    durations['jd'] >= jd_before,
                    durations['jd'] <= jd_after,
                    durations['status'] != 'not observable'])

            # not enough observing windows before or after:
            if (durations.loc[sel_time, 'jd'].iloc[0] != jd_before or
                    durations.loc[sel_time, 'jd'].iloc[-1] != jd_after):
                status = 'unknown'
                durations.at[i, 'status'] = status

            # determine status
            else:
                outliers = self._detect_outliers(
                        durations.loc[sel_time, 'jd'],
                        durations.loc[sel_time, 'duration'],
                        outlier_threshold=outlier_threshold)
                status, setting_in = self._get_status(
                        jd, durations.loc[sel_time, 'jd'],
                        durations.loc[sel_time, 'duration'], time_interval,
                        status_threshold=status_threshold, mask_outl=outliers)
                durations.at[i, 'status'] = status
                durations.at[i, 'update'] = True

                if status == 'setting':
                    durations.at[i, 'setting_in'] = setting_in

        self._update_unknown_status(durations)

        # add to queue:
        sel = durations['update']

        for __, status in durations.loc[sel].iterrows():
            queue_status.put((
                    status['observability_id'], status['status'],
                    status['setting_in']))

        with counter_lock:
            counter.value += 1

    #--------------------------------------------------------------------------
    def _add_status(
            self, db, days_before, days_after, outlier_threshold,
            status_threshold, time_interval, batch_write=10000, processes=1):
        """Determine the observability status for each field for each day and
        save it in the database.

        Parameters
        ----------
        db : db.DBConnectorSQLite
            Active database connection.
        days_before : int
            Determining the field's status requires the analysis of observation
            windows in a time range. This arguments sets how many days before
            are considered.
        days_after : int
            Determining the field's status requires the analysis of observation
            windows in a time range. This arguments sets how many days after
            are considered.
        outlier_threshold : float
            Threshold for outlier detection. Days with observing window
            durations outlying from the general trend are not considered in
            the status analysis.
        status_threshold : float
            Threshold for distinguishing plateauing from rising or setting. If
            the OLS p-value is larger than this threshold, the status is
            considered "plateauing"; otherwise "rising" or "setting" depending
            on the slope.
        time_interval : astropy.time.TimeDelta
            Time accuracy at which the observing windows were calculated.
        batch_write : int, optional
            Observing window are temporarily saved and then written to the
            database in batches of this size. The default is 10000.
        processes : int, optional
            Number of processes that run the obsering window calcuations for
            different fields in parallel. The default is 1.

        Returns
        -------
        None

        Notes
        -----
        - This method is called by `check_observability()`.
        - This method starts `_update_status_in_db()` in a separate process.
        - This method starts `_update_status_for_field()` in a pool of
          processes.
        """

        field_ids, jds_min, jds_max = self._get_fields_missing_status(db)
        jds_before = jds_min - days_before
        jds_after = jds_max + days_after
        n_fields = field_ids.shape[0]
        print('\nDetermine observability status of fields..')
        print(f'{n_fields} fields need status updates..')

        # parallel process fields:
        manager = Manager()
        queue_status = manager.Queue()
        counter = manager.Value(int, 0)
        counter_lock = manager.Lock()
        writer = Process(
                target=self._update_status_in_db,
                args=(db, counter, queue_status, n_fields, batch_write)
                )
        writer.start()

        with Pool(processes=processes) as pool:
            pool.starmap(
                    self._update_status_for_field,
                    zip(repeat(counter), repeat(counter_lock),
                        repeat(queue_status),
                        repeat(db), field_ids, jds_before, jds_after,
                        repeat(days_before), repeat(days_after),
                        repeat(outlier_threshold), repeat(status_threshold),
                        repeat(time_interval)))

        writer.join()

    #--------------------------------------------------------------------------
    def iter_observable_fields(
            self, observatory, night=None, datetime=None, observed=None,
            pending=None, active=True):
        """Iterate through fields observable during a given night or at a
        specific time, given specific selection criteria.

        Parameters
        ----------
        observatory : str
            Observatory name.
        night : astropy.time.Time, optional
            Iterate through fields observable during the night that starts on
            the specified day. Time information is truncated. Either set this
            argument or 'datetime'. If this argument is set, 'datetime' is not
            used.
        datetime : astropy.time.Time, optional
            Iterate through fields that are observable at the given time.
            Either set this argument or 'night'.
        observed : bool or None, optional
            If True, iterate only through fields that have been observed at
            least once. If False, iterate only through fields that have never
            been observed. If None, iterate through fields irregardless of
            whether they have  been observed or not. The default is None.
        pending : bool or None, optional
            If True, iterate only through fields that have pending observations
            associated. If False, only iterate through fields that have no
            pending observations associated. If None, iterate through fields
            irregardless of whether they have pending observations associated
            or not. The default is None.
        active : bool or None, optional
            If True, only iterate through active fields. If False, only iterate
            through inactive fields. If None, iterate through fields active or
            not. The default is True.

        Yields
        ------
        field : Field
            Field(s) fulfilling the selected criteria.
        """

        # check input:
        if night is not None:
            return self._iter_observable_fields_by_night(
                    observatory, night, observed=observed, pending=pending,
                    active=active)

        elif datetime is not None:
            return self._iter_observable_fields_by_datetime(
                    observatory, datetime, observed=observed, pending=pending,
                    active=active)

        else:
            raise ValueError(
                    "Either provide 'night' or 'datetime' argument.")

    #--------------------------------------------------------------------------
    def get_observable_fields(
            self, observatory, night=None, datetime=None, observed=None,
            pending=None, active=True):
        """Get a list of fields observable during a given night or at a
        specific time, given specific selection criteria.

        Parameters
        ----------
        observatory : str
            Observatory name.
        night : astropy.time.Time, optional
            Iterate through fields observable during the night that starts on
            the specified day. Time information is truncated. Either set this
            argument or 'datetime'. If this argument is set, 'datetime' is not
            used.
        datetime : astropy.time.Time, optional
            Iterate through fields that are observable at the given time.
            Either set this argument or 'night'.
        observed : bool or None, optional
            If True, iterate only through fields that have been observed at
            least once. If False, iterate only through fields that have never
            been observed. If None, iterate through fields irregardless of
            whether they have  been observed or not. The default is None.
        pending : bool or None, optional
            If True, iterate only through fields that have pending observations
            associated. If False, only iterate through fields that have no
            pending observations associated. If None, iterate through fields
            irregardless of whether they have pending observations associated
            or not. The default is None.
        active : bool or None, optional
            If True, only iterate through active fields. If False, only iterate
            through inactive fields. If None, iterate through fields active or
            not. The default is True.

        Returns
        ------
        observable_fields : list of Field
            Field(s) fulfilling the selected criteria.
        """

        observable_fields = [field for field in self.iter_observable_fields(
                observatory, night=night, datetime=datetime, observed=observed,
                pending=pending, active=active)]

        return observable_fields

    #--------------------------------------------------------------------------
    def get_night_start_end(self, observatory, datetime):
        """Get the start and stop time of a night for a given date.

        Parameters
        ----------
        observatory : str
            Observatory name as stored in the database.
        datetime : astropy.time.Time
            The night starting on that date is considered. Time information is
            truncated.

        Returns
        -------
        night_start : astropy.time.Time
            Start date and time of the night.
        night_stop : astropy.time.Time
            Stop date and time of the night.

        Notes
        -----
        The start and stop time depends on the definition of the twilight. This
        is set as fixed parameter in the database.
        """

        datetime = datetime.to_datetime()
        year = datetime.year
        month = datetime.month
        day = datetime.day

        self._setup_observatory(observatory, no_constraints=True)
        night_start, night_stop = self.telescope.get_sun_set_rise(
                year, month, day, self.twilight)

        return night_start, night_stop

    #--------------------------------------------------------------------------
    def iter_fields(
            self, observatory=None, observed=None, pending=None, active=True):
        """Iterate through fields, given specific selection criteria.

        Parameters
        ----------
        observatory : str, optional
            Iterate only through fields associated with this observatory.
            If None, iterate through all fields irregardless of the associated
            observatory.
        observed : bool or None, optional
            If True, iterate only through fields that have been observed at
            least once. If False, iterate only through fields that have never
            been observed. If None, iterate through fields irregardless of
            whether they have  been observed or not. The default is None.
        pending : bool or None, optional
            If True, iterate only through fields that have pending observations
            associated. If False, only iterate through fields that have no
            pending observations associated. If None, iterate through fields
            irregardless of whether they have pending observations associated
            or not. The default is None.
        active : bool or None, optional
            If True, only iterate through active fields. If False, only iterate
            through inactive fields. If None, iterate through fields active or
            not. The default is True.

        Yields
        ------
        field : Field
            Field(s) fulfilling the selected criteria.
        """

        # connect to database:
        db = DBConnectorSQLite(self.dbname)

        for field in db.get_fields(
                observatory=observatory, observed=observed, pending=pending,
                active=active):
            yield self._tuple_to_field(field)

    #--------------------------------------------------------------------------
    def get_fields(
            self, observatory=None, observed=None, pending=None, active=True):
        """Get a list of fields, given specific selection criteria.

        Parameters
        ----------
        observatory : str, optional
            Only get fields associated with this observatory. If None, get all
            fields irregardless of the associated observatory.
        observed : bool or None, optional
            If True, only get fields that have been observed at least once. If
            False, only get fields that have never been observed. If None, get
            fields irregardless of whether they have  been observed or not. The
            default is None.
        pending : bool or None, optional
            If True, only get fields that have pending observations associated.
            If False, only get fields that have no pending observations
            associated. If None, get fields irregardless of whether they have
            pending observations associated or not. The default is None.
        active : bool or None, optional
            If True, only get active fields. If False, only get inactive
            fields. If None, get fields active or not. The default is True.

        Yields
        ------
        field : list of Field
            Field(s) fulfilling the selected criteria.
        """

        fields = [field for field in self.iter_fields(
                observatory=observatory, observed=observed, pending=pending,
                active=active)]

        return fields

    #--------------------------------------------------------------------------
    def get_field_by_id(self, field_id, db=None):
        """Get a field by its database ID.

        Parameters
        ----------
        field_id : int
            ID of the field as stored in the database.
        db : db.DBConnectorSQLite, optional
            Active database connection. If none provided a new connection is
            established. The default is None.

        Raises
        ------
        ValueError
            Raise if no field with this ID exists.

        Returns
        -------
        field : Field
            Field as stored in the database under specified ID.
        """

        # connect to database:
        if db is None:
            db = DBConnectorSQLite(self.dbname)

        field = db.get_field_by_id(field_id)

        if not len(field):
            raise ValueError(f"Field with ID {field_id} does not exist.")

        field = self._tuple_to_field(field[0])

        return field

    #--------------------------------------------------------------------------
    def iter_fields_by_ids(self, field_ids):
        """Yield Field instances by their database ID.

        Parameters
        ----------
        field_ids : list
            IDs of the fields as stored in the database.

        Raises
        ------
        ValueError
            Raise if no field with this ID exists.

        Yields
        -------
        field : Field
            Field as stored in the database under specified ID.
        """

        # connect to database:
        db = DBConnectorSQLite(self.dbname)

        for field_id in field_ids:
            field = self.get_field_by_id(field_id, db=db)

            yield field

    #--------------------------------------------------------------------------
    def iter_field_ids_in_circles(
            self, circle_center, radius, observatory=None, observed=None,
            pending=None, active=True):
        """Iterate through different circle center locations and yield the IDs
        of fields located in those circles.

        Parameters
        ----------
        circle_center : astropy.coordinates.SkyCoord
            Center coordinates of the circle(s). A single coordinate or
            multiple coordinates can be provided in a SkyCoord instance.
        radius : astropy.units.Quantity
            The circle radius in deg or rad.
        observatory : str, optional
            Only count fields associated with this observatory. If None, count
            all fields irregardless of the associated observatory.
        observed : bool or None, optional
            If True, only count fields that have been observed at least once.
            If False, only count fields that have never been observed. If None,
            count fields irregardless of whether they have been observed or
            not. The default is None.
        pending : bool or None, optional
            If True, only count fields that have pending observations
            associated. If False, only count fields that have no pending
            observations associated. If None, count fields irregardless of
            whether they have pending observations associated or not. The
            default is None.
        active : bool or None, optional
            If True, only count active fields. If False, only count inactive
            fields. If None, count fields active or not. The default is True.

        Raises
        ------
        ValueError
            Raised if a data type other than SkyCoord is provided for
            'circle_center'..

        Yields
        ------
        numpy.ndarray
            Each int-dtype array lists the IDs of the fields located in the
            circle with the corresponding center coordinates, where the fields
            fulfill the selection criteria.
        """

        # check input:
        if not isinstance(circle_center, SkyCoord):
            raise ValueError("'circle_center' has to be SkyCoord instance.")

        # convert scalar to array:
        if not circle_center.shape:
            circle_center = SkyCoord([circle_center.ra], [circle_center.dec])

        # get coordinates of all fields meeting selection criteria:
        fields_id = []
        fields_ra = []
        fields_dec = []

        for field in self.iter_fields(
                observatory=observatory, observed=observed, pending=pending,
                active=active):

            fields_id.append(field.id)
            fields_ra.append(field.center_ra)
            fields_dec.append(field.center_dec)

        fields_id = np.array(fields_id)
        fields_coord = SkyCoord(fields_ra, fields_dec)
        del fields_ra, fields_dec

        # iterate through circle center coordinates:
        for coord in circle_center:
            in_circle = fields_coord.separation(coord) <= radius

            yield fields_id[in_circle]

    #--------------------------------------------------------------------------
    def check_observability(
            self, date_stop, date_start=None, duration_limit=60,
            batch_write=10000, processes=1, time_interval_init=300,
            time_interval_refine=5, days_before=7, days_after=7,
            outlier_threshold=0.7, status_threshold=6):
        """Calculate observing windows for all active fields and add them to
        the database.

        Parameters
        ----------
        date_stop : astropy.time.Time
            The date until which observing windows should be calculated.
        date_start : astropy.time.Time, optional
            The date from which on observing windows should be calculated. If
            this is later than the latest entries in the database, this will
            lead to gaps in the observing window calculation. The user is
            warned about this. It is safer to use this option only for the
            initial run and not afterwards. The default is None.
        duration_limit : float or astropy.units.quantity.Quantity, optional
            Limit on the observing window duration above which a field is
            considered observable. If a float is given, it is considered to be
            in seconds. Otherwise, provide an astropy.units.quantity.Quantity
            in any time unit, e.g. by multiplying with astropy.units.min. The
            default is 60.
        batch_write : int, optional
            Observing window are temporarily saved and then written to the
            database in batches of this size. The default is 10000.
        processes : int, optional
            Number of processes that run the obsering window calcuations for
            different fields in parallel. The default is 1.
        time_interval_init : float or Quantity, optional
            Time accuracy at which the observing windows are initially
            calculated. The accuracy can be refined with the next parameter.
            If a float is given, it is considered to be in seconds. Otherwise,
            provide an astropy.units.quantity.Quantity in any time unit, e.g.
            by multiplying with astropy.units.min. The default is 300.
        time_interval_refine : float or Quantity, optional
            Time accuracy in seconds at which the observing windows are
            calculated after the initial coarse search. If a float is given, it
            is considered to be in seconds. Otherwise, provide an
            astropy.units.quantity.Quantity in any time unit, e.g. by
            multiplying with astropy.units.min.The default is 5.
        days_before : int, optional
            Determining the field's status requires the analysis of observation
            windows in a time range. This arguments sets how many days before
            are considered. The default is 7.
        days_after : int, optional
            Determining the field's status requires the analysis of observation
            windows in a time range. This arguments sets how many days after
            are considered. The default is 7.
        outlier_threshold : float, optional
            Threshold for outlier detection. Days with observing window
            durations outlying from the general trend are not considered in
            the status analysis. The default is 0.7.
        status_threshold : float, optional
            Threshold for distinguishing plateauing from rising or setting. If
            the OLS p-value is larger than this threshold, the status is
            considered "plateauing"; otherwise "rising" or "setting" depending
            on the slope. The default is 6.

        Raises
        ------
        ValueError
            Raised if time_interval_init is smaller than time_interval_refine.

        Returns
        -------
        None

        Notes
        -----
        - This method performs two tasks: (1) It calculates the observing
          windows for each field for each day and saves them. (2) Then it
          determines the observability status (not observable, rising,
          plateauing, setting) for each field for each day.
        - This method starts at least two additional processes: (1) one or more
          worker processes - set by the `proccesses` parameter - that
          calculate(s) the observing windows and (2) one process that writes
          the observing windows to the database, similary for the second step
          of determining and saving the status of observability.
        """

        # converte inputs:
        batch_write = int(batch_write)

        if type(duration_limit) in [float, int]:
            duration_limit = TimeDelta(duration_limit * u.s)
        else:
            duration_limit = TimeDelta(duration_limit)

        if type(time_interval_init) in [float, int]:
            time_interval_init = TimeDelta(time_interval_init * u.s)
        else:
            time_interval_init = TimeDelta(time_interval_init)

        if type(time_interval_refine) in [float, int]:
            time_interval_refine = TimeDelta(time_interval_refine * u.s)
        else:
            time_interval_refine = TimeDelta(time_interval_refine)

        if time_interval_refine > time_interval_init:
            raise ValueError(
                    "`time_interval_refine` cannot be smaller than "
                    "`time_interval_init`.")

        print('Calculate observing windows until {0}..'.format(
                date_stop.iso[:10]))

        # connect to database:
        db = DBConnectorSQLite(self.dbname)
        jd_stop = date_stop.jd
        agreed_to_gaps = None

        # iterate through observatories:
        for i, m, observatory in db.iter_observatories():
            observatory_name = observatory['name']
            print(f'\nObservatory {i+1} of {m} selected: {observatory_name}')

            # get fields that need observing window calculations:
            print('Query fields..')
            fields_init = db.get_fields(
                    observatory=observatory_name, init_obs_windows=True)
            fields_tbd = db.get_fields(
                    observatory=observatory_name, needs_obs_windows=jd_stop)

            with Pool(processes=processes) as pool:
                    fields_init = pool.map(self._tuple_to_field, fields_init)
                    fields_tbd = pool.map(self._tuple_to_field, fields_tbd)

            # calculate observing windows for new fields:
            agreed_to_gaps = self._add_obs_windows(
                    db, fields_init, True, observatory_name, date_stop,
                    date_start, duration_limit, batch_write, processes,
                    time_interval_init, time_interval_refine, agreed_to_gaps)

            # calculate observing windows for fields:
            agreed_to_gaps = self._add_obs_windows(
                    db, fields_tbd, False, observatory_name, date_stop,
                    date_start, duration_limit, batch_write, processes,
                    time_interval_init, time_interval_refine, agreed_to_gaps)

        # determine observability status for each field for each day:
        self._add_status(
                db, days_before, days_after, outlier_threshold,
                status_threshold, time_interval_refine,
                batch_write=batch_write, processes=processes)

    #--------------------------------------------------------------------------
    def count_neighbors(
            self, radius, field_ids, observatory=None, observed=None,
            pending=None, active=True):
        """Count the neighbors of specificied fields given various criteria.

        Parameters
        ----------
        radius : astropy.units.Quantity
            Radius in which to count field neighbors. Needs to be a Quantity
            with unit 'rad' or 'deg'.
        field_ids : int or list
            ID of the field whose neighbors are searched for. Or list of
            multiple such IDs.
        observatory : str, optional
            Only count fields associated with this observatory. If None, count
            all fields irregardless of the associated observatory.
        observed : bool or None, optional
            If True, only count fields that have been observed at least once.
            If False, only count fields that have never been observed. If None,
            count fields irregardless of whether they have been observed or
            not. The default is None.
        pending : bool or None, optional
            If True, only count fields that have pending observations
            associated. If False, only count fields that have no pending
            observations associated. If None, count fields irregardless of
            whether they have pending observations associated or not. The
            default is None.
        active : bool or None, optional
            If True, only count active fields. If False, only count inactive
            fields. If None, count fields active or not. The default is True.

        Returns
        -------
        neighbor_count : numpy.ndarray or int
            Lists the number of field neighbors within the specified radius
            that fulfill the conditions for each of the input field IDs.
            If an integer was provided for 'field_ids', an integer is returned,
            otherwise a numpy.ndarray of integer-dtype.
        """

        if isinstance(field_ids, int):
            field_ids = [field_ids]
            return_int = True
        else:
            return_int = False

        # get coordinates of fields of interest:
        fields_ra = []
        fields_dec = []

        for field in self.iter_fields_by_ids(field_ids):
            fields_ra.append(field.center_ra)
            fields_dec.append(field.center_dec)

        fields_coord = SkyCoord(fields_ra, fields_dec)
        del fields_ra, fields_dec

        # iterate through field coordinates and count neighbors:
        n_neighbors = np.zeros(len(field_ids), dtype=int)

        for i, neighbor_ids in enumerate(self.iter_field_ids_in_circles(
                fields_coord, radius, observatory=observatory,
                observed=observed, pending=pending, active=active)):

            n_neighbors[i] = neighbor_ids.shape[0]

            # reduce count by 1 if field is part of the list:
            if field_ids[i] in neighbor_ids:
                n_neighbors[i] -= 1

        if return_int:
            n_neighbors = int(n_neighbors)

        return n_neighbors

#==============================================================================

class Prioritizer:
    """A class to assign priorities to a list of fields.
    """

    #--------------------------------------------------------------------------
    def __init__(self, surveyplanner):
        """Create Prioratizer instance.

        Parameters
        ----------
        surveyplanner : SurveyPlanner
            The SurveyPlanner instance this Prioratizer is used by.

        Returns
        -------
        None
        """

        self.surveyplanner = surveyplanner

    #--------------------------------------------------------------------------
    def _prioritize_by_sky_coverage(
            self, fields, radius, observatory=None, normalize=False):
        """Assign priority based on sky coverage.

        Parameters
        ----------
        fields : list of Field
            The fields that a priority is assigned to.
        radius : astropy.units.Quantity
            Radius in which to count field neighbors. Needs to be a Quantity
            with unit 'rad' or 'deg'.
        observatory : str, optional
            Only fields associated with this observatory are taken into
            consideration. If None, all fields are considered irregardless of
            the associated observatory.
        normalize : bool, optional
            If True, all priorities are scaled such that the highest priority
            has a value of 1.

        Returns
        -------
        priority : numpy.ndarray
            The priorities assigned to the input fields.

        Notes
        -----
        The main idea is that a field that is surrounded by many finished
        fields will get higher priority than a field with fewer finished,
        neighboring fields. This is implemented in the variable `coverage0`.
        However, a field with fewer neighbors (e.g. near the declination limit)
        should get the same priority as a field with more neighbors if both
        have no neighbors finished. This is implemented in the variable
        `coverage1`. A weighting between two approaches is applied, such that
        the first approach strongly applies to fields that have a strong impact
        on finishing the neighborhood and the second approach is more dominant
        when only we neighboring fields have been observed yet.
        """

        field_ids = [field.id for field in fields]

        # count all neighboring fields for each given field:
        count_all = self.surveyplanner.count_neighbors(
                radius, field_ids, observatory=observatory)
        count_all = np.array(count_all)

        # count finished neighboring fields for each given field:
        count_finished = self.surveyplanner.count_neighbors(
                radius, field_ids, pending=False, observatory=observatory)
        count_finished = np.array(count_finished)

        # calculate priority:
        count_all_max = count_all.max()
        coverage0 = (count_finished + 1) / (count_all + 1)
        coverage1 = (count_finished + 1) / (count_all_max + 1)
        weight0 = count_finished / count_all
        weight1 = 1. - weight0
        priority = coverage0 * weight0 + coverage1 * weight1

        # normalize:
        if normalize:
            priority = priority / priority.max()

        return priority

    #--------------------------------------------------------------------------
    def _prioritize_by_field_status(
            self, fields, rising=False, plateauing=False, setting=False):
        """Prioratize fields by status of observability.

        Parameters
        ----------
        fields : ist of Field
            The fields that a priority is assigned to.
        rising : bool, optional
            If True, prioritize a field if it is rising. The default is False.
        plateauing : bool, optional
            If True, prioritize a field if it is plateauing. The default is
            False.
        setting : bool, optional
            If True, prioritize a field if it is setting. The default is False.

        Returns
        -------
        priority : numpy.ndarray
            The priorities assigned to the input fields.

        Notes
        -----
        This prioritization just returns 0. or 1..

        The field status IDs are the following:
        rising: 1
        plateauing: 2
        setting: 3
        """

        priority = np.zeros(len(fields))

        for i, field in enumerate(fields):
            if rising and field.status == 1:
                priority[i] = 1.
            if plateauing and field.status == 2:
                priority[i] = 1.
            if setting and field.status == 3:
                priority[i] = 1.

        return priority

    #--------------------------------------------------------------------------
    def _add_priorities_to_fields(self, priorities, fields):
        """Add priority to each field.

        Parameters
        ----------
        priorities : numpy.ndarray
            The priority corresponding to each field.
        fields : list of Field
            The fields that the priorities correspond to and should be added
            to.

        Returns
        -------
        fields : list of Field
            The same list of Field instances, now with priorities stored.
        """

        for priority, field in zip(priorities, fields):
            field.set_priority(priority)

        return fields

    #--------------------------------------------------------------------------
    def prioritize(
            self, fields, weight_coverage=0., weight_rising=0.,
            weight_plateauing=0., weight_setting=0., normalize=False,
            coverage_radius=None, coverage_observatory=None,
            coverage_normalize=False, return_priorities=False):
        """Assign a priority to each field.

        Parameters
        ----------
        fields : ist of Field
            The fields that a priority is assigned to.
        weight_coverage : float or int, optional
            Weight assigned to the sky coverage priorities. The default is 0..
        weight_rising : float or int, optional
            Weight assigned to the rising fields. The default is 0..
        weight_plateauing : float or int, optional
            Weight assigned to the plateauing fields. The default is 0..
        weight_setting : float or int, optional
            Weight assigned to the setting fields. The default is 0..
        normalize : bool, optional
            If True, the final priorities after weighting are normalized to the
            interval [0, 1]. The default is False.
        coverage_radius : astropy.units.Quantity
            Radius in which to count field neighbors. Needs to be a Quantity
            with unit 'rad' or 'deg'. Required if `weight_coverage` is given.
        coverage_observatory : str, optional
            Only fields associated with this observatory are taken into
            consideration for calculation of the sky coverage. If None, all
            fields are considered irregardless of the associated observatory.
            Only has an effect, when `weight_coverage` is given.
        coverage_normalize : bool, optional
            If True, priorities are scaled such that the highest priority
            has a value of 1.
        return_priorities : bool, optional
            If True, the priorities based on the individual criteria and the
            final, joint priorities are returned as well as a dict. Otherwise,
            just fields with priorities added are returned. The default is
            False.

        Raises
        ------
        ValueError
            Raised if `weight_coverage` is given for a prioritization based on
            the sky coverage, but `coverage_radius` is not given, which is
            required to calculate the coverage.
        ValueError
            Raised if any of the weights `weight_*` is negative or not integer
            or float.

        Returns
        -------
        fields : list of Field
            The input fields now with priorities added.
        optional:
        priority : dict of numpy.ndarray
            The priorities based on the individual criteria and the final,
            joint priorities. Only returned if `return_all=True`.
        """

        # check input:
        if weight_coverage and coverage_radius is None:
            raise ValueError(
                    "When 'weight_coverage' is non-zero, 'coverage_radius' " \
                    "has to be set.")

        if type(weight_coverage) not in [float, int] or \
                weight_coverage < 0:
            raise ValueError("'weight_coverage' must be positive float.")

        if type(weight_rising) not in [float, int] or weight_rising < 0:
            raise ValueError("'weight_rising' must be positive float.")

        if type(weight_plateauing) not in [float, int] or \
                weight_plateauing < 0:
            raise ValueError("'weight_plateauing' must be positive float.")

        if type(weight_setting) not in [float, int] or \
                weight_setting < 0:
            raise ValueError("'weight_setting' must be positive float.")

        # get priorities by individual criteria:
        priorities = []
        weights = []
        priorities_dict = {}

        if weight_coverage:
            priority = self._prioritize_by_sky_coverage(
                    fields, coverage_radius,
                    observatory=coverage_observatory,
                    normalize=coverage_normalize)
            priorities.append(priority)
            weights.append(weight_coverage)
            priorities_dict['coverage'] = {
                    'weight': weight_coverage, 'priority': priority}

        if weight_rising:
            priority = self._prioritize_by_field_status(
                    fields, rising=True, plateauing=False, setting=False)
            priorities.append(priority)
            weights.append(weight_rising)
            priorities_dict['rising'] = {
                    'weight': weight_rising, 'priority': priority}

        if weight_plateauing:
            priority = self._prioritize_by_field_status(
                    fields, rising=False, plateauing=True, setting=False)
            priorities.append(priority)
            weights.append(weight_plateauing)
            priorities_dict['plateauing'] = {
                    'weight': weight_plateauing, 'priority': priority}

        if weight_setting:
            priority = self._prioritize_by_field_status(
                    fields, rising=False, plateauing=False, setting=True)
            priorities.append(priority)
            weights.append(weight_setting)
            priorities_dict['setting'] = {
                    'weight': weight_setting, 'priority': priority}

        # joint priority:
        priority = sum([w * p for w, p in zip(weights, priorities)])
        priority /= sum(weights)

        # normalize:
        if normalize:
            priority = priority / priority.max()


        fields = self._add_priorities_to_fields(priority, fields)
        priorities_dict['joint'] = priority

        if return_priorities:
            return fields, priorities_dict

        return priority


#==============================================================================
