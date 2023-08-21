# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Pasiphae survey planner.
"""

from astropy.coordinates import AltAz, Angle, EarthLocation, get_sun, SkyCoord
from astropy.time import Time, TimeDelta
from astropy import units as u
import numpy as np
from scipy.stats import linregress
from textwrap import dedent
from warnings import warn

import constraints as c
from db import DBConnectorSQLite

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
            latest_obs_window_jd=None, n_obs_tot=0, n_obs_done=0,
            n_obs_pending=0):
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
        latest_obs_window_jd : float, optional
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
        self.latest_obs_window_jd = latest_obs_window_jd
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

        if self.status == -1:
            status_str = 'not observable'
        elif self.status == 0:
            status_str = 'unknown/undefined'
        elif self.status == 1:
            status_str = 'rising'
        elif self.status == 2:
            status_str = 'plateauing'
        elif self.status == 3:
            status_str = 'setting in {0:.2f}'.format(self.setting_in)

        return status_str

    #--------------------------------------------------------------------------
    def _true_blocks(self, observable):
        """Find blocks of successive True's.

        Parameters
        ----------
        observable : nump.ndarray
            Boolean-type 1dim-array.

        Returns
        -------
        list
            Each element corresponds to one block of True's. The element is a
            list of two integers, the first marking the first index of the
            block, the second marking the last True entry of the block.
        """

        i = 0
        periods = []

        # iterate through array:
        while i < observable.size-1:
            if ~np.any(observable[i:]):
                break
            j = np.argmax(observable[i:]) + i
            k = np.argmax(~observable[j:]) + j
            if j == k and j != observable.size-1:
                k = observable.size
            periods.append((j,k-1))
            i = k

        return periods

    #--------------------------------------------------------------------------
    def get_obs_window(self, telescope, frame, refine=0*u.min):
        """Calculate time windows when the field is observable.

        Parameters
        ----------
        telescope : Telescope
            Telescope for which to calculate observability.
        frame : astropy.coordinates.AltAz
            Frame that provides the time steps at which observability is
            initially tested.
        refine : astropy.units.Quantity, optional
            Must be a time unit. If given, the precision of the observable time
            window is refined to this value. I.e. if the interval given in
            'frame' is 10 minutes and refine=1*u.min the window limits will be
            accurate to a minute. The default is 0*u.min.

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
        blocks = self._true_blocks(observable)

        for i, j in blocks:
            obs_window = (frame.obstime[i], frame.obstime[j])
            temp_obs_windows.append(obs_window)

        # increase precision for actual observing windows:
        if refine.value:
            time_interval = frame.obstime[1] - frame.obstime[0]

            # iterate through time windows:
            for t_start, t_stop in temp_obs_windows:
                # keep start time:
                if t_start == frame.obstime[0]:
                    pass
                # higher precision for start time:
                else:
                    t_start_new = t_start - time_interval
                    frame = telescope.get_frame(t_start_new, t_start, refine)
                    observable = telescope.constraints.get(
                            self.center_coord, frame)
                    k = np.argmax(observable)
                    t_start = frame.obstime[k]

                # keep stop time:
                if t_stop == frame.obstime[-1]:
                    pass
                # higher precision for stop time:
                else:
                    t_stop_new = t_stop + time_interval
                    frame = telescope.get_frame(t_stop, t_stop_new, refine)
                    observable = telescope.constraints.get(
                            self.center_coord, frame)
                    k = (frame.obstime.value.size - 1
                         - np.argmax(observable[::-1]))
                    t_stop = frame.obstime[k]

                #obs_windows.append(ObsWindow(t_start, t_stop)) # TODO
                obs_windows.append((t_start, t_stop))

        # in case of no precision refinement:
        else:
            for t_start, t_stop in temp_obs_windows:
                #obs_windows.append(ObsWindow(t_start, t_stop)) # TODO
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
            ID in the data base.

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

        print('Telescope: {0:s} created.'.format(self.name))

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
    def get_frame(self, time_start, time_stop, time_interval):
        """Create an AltAz frame for specified dates.

        Parameters
        ----------
        time_start : astropy.time.Time
            The start date for the frame.
        time_stop : astropy.time.Time
            The stop date for the frame.
        time_interval : astropy.units.Quantity
            The time interval in minutes.

        Returns
        -------
        astropy.coordinates.AltAz
            The AltAz frame for specified dates.
        """

        duration = (time_stop - time_start).to_value(time_interval.unit)
        time = time_start + np.arange(0, duration, time_interval.value) \
                * time_interval.unit
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
        None
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
            constraints = db.get_constraints(observatory)
            self.twilight = constraints['Twilight']['twilight']
            return None

        # read constraints:
        constraints = db.get_constraints(observatory)
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
                latest_obs_window_jd, n_obs_tot, n_obs_done, n_obs_pending \
                = field_tuple
        field = Field(
            fov, center_ra, center_dec, tilt, field_id=field_id,
            latest_obs_window_jd=latest_obs_window_jd, n_obs_tot=n_obs_tot,
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
    def _set_field_status(self, field, date, days_before=3, days_after=7):
        """Determine and save a field's status (rising, plateauing, setting).

        Parameters
        ----------
        field : Field
            A field.
        date : astropy.time.Time
            Date and time for which to determine the field's status.
        days_before : int, optional
            Determining the field's status requires the analysis of observation
            windows in a time range. This arguments sets how many days before
            'date' are included. The default is 3.
        days_after : TYPE, optional
            Determining the field's status requires the analysis of observation
            windows in a time range. This arguments sets how many days after
            'date' are included. The default is 7.

        Returns
        -------
        None
        """

        # connect to database:
        db = DBConnectorSQLite(self.dbname)

        # get durations of the next N observing windows
        date_start = date - days_before * u.d
        date_stop = date + days_after * u.d
        durations = db.get_obs_window_durations(
                field.id, date_start, date_stop)
        durations = np.array(durations).squeeze()

        # start and end date:
        date_start = date - days_before * u.d
        date_stop = date + days_after * u.d
        latest_obs_window_jd = field.latest_obs_window_jd

        # check that obs windows are available for time range
        if date_stop.jd > latest_obs_window_jd:
            warn(f"Calculate observing windows until {date_stop.iso[:10]}.")

        # get observing windows:
        db = DBConnectorSQLite(self.dbname)
        obs_windows = db.get_obs_windows_from_to(
                field.id, date_start, date_stop)
        obs_windows = self._tuples_to_obs_windows(obs_windows)
        durations = [obs_window.duration.value for obs_window in obs_windows]
        n_windows = len(durations)

        # check status - not observable:
        if n_windows == 0:
            field.set_status(not_available=True)

        # check status - only one observing windows:
        elif n_windows == 1:
            # Note: single appearance at last day could be rising or really
            # just a single appearance, defacto rising/plateauing/setting.
            # Single appearance at first day could be setting of just a single
            # appearance. Single appearance in middle would be indeed a single
            # appearance, unless no observing windows have been estimated the
            # before the current date and a setting source therefore appears as
            # single appearance.
            # We do not differenciate these cases. Classify as undetermined.
            # Rising/setting sources will be classified as such the
            # next/earlier day. Single-appearance sources should not be common.
            pass

        # check status - rising/plateauing/setting:
        else:
            x = np.arange(n_windows) # assuming daily calculated obs windows
            x -= days_before
            result = linregress(x, durations)

            # check status - plateauing:
            if result.pvalue >= 0.1:
                field.set_status(plateauing=True)

            # check status - rising:
            elif result.slope > 0:
                field.set_status(rising=True)

            # check status - setting:
            else:
                setting_in = -result.intercept / result.slope * u.d
                field.set_status(setting=True, setting_in=setting_in)

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
    def add_obs_windows(self, date_stop, date_start=None, batch_write=10000):
        """Calculate observation windows for all active fields and add them to
        the database.

        Parameters
        ----------
        date_stop : astropy.time.Time
            Calculate observation windows until this date. Time information is
            truncated.
        date_start : astropy.time.Time, optional
            Calculate observation windows starting with this date. Time
            information is truncated. If not set, observation windows will be
            calculated starting with the date at which the last calculation
            stopped. The default is None.
        batch_write : int, optional
            Observing windows will be gathered up to this number and then
            written as a batch to the database. The default is 10000.

        Returns
        -------
        None
        """

        print('Calculate observing windows until {0}..'.format(
                date_stop.iso[:10]))

        jd_stop = date_stop.jd
        user_agrees = False

        # temporary data storages:
        self.batch_field_ids  = []
        self.batch_obs_windows_start = []
        self.batch_obs_windows_stop = []
        self.batch_jd_stop = {}

        # connect to database:
        db = DBConnectorSQLite(self.dbname)

        # iterate through observatories:
        for i, m, observatory in db.iter_observatories():
            observatory_name = observatory['name']
            print(f'Observatory {i+1} of {m} selected: {observatory_name}')

            # get fields that need observing window calculations:
            fields_tbd = db.get_fields(
                    observatory=observatory_name, needs_obs_windows=jd_stop)
            n_fields_tbd = len(fields_tbd)

            # if all done, skip to next observatory:
            if n_fields_tbd == 0:
                print('Observing windows already stored for all fields up to '
                      f'JD {jd_stop}.')
                continue

            # setup observatory with constraints:
            self._setup_observatory(observatory_name)

            print(f'{n_fields_tbd} require calculation of observing windows..')

            # iterate through fields associated with observatory:
            for j, field in enumerate(fields_tbd):

                print('\rField {0} of {1} ({2:.1f}%)..'.format(
                        j+1, n_fields_tbd, j/n_fields_tbd*100),
                      end='')

                # create Field object from tuple data:
                field_id = field[0]
                field_fov = field[1]
                field_center_ra = field[2]
                field_center_dec = field[3]
                jd_next_obs_window = field[7]
                field = Field(field_fov, field_center_ra, field_center_dec)

                # get JD for next observing window calculation:
                if date_start is None:
                    now = Time.now().value
                    date_start = Time(
                        f'{now.year}-{now.month}-{now.day}T00:00:00')
                    jd_start = date_start.jd

                else:
                    jd_start = date_start.jd

                # no expected JD stored, use given one:
                if jd_next_obs_window is None:
                    pass

                # given JD is earlier than expected JD, change given JD:
                elif jd_start < jd_next_obs_window:
                    jd_start = jd_next_obs_window

                # given JD is later than expected JD, will result in gaps in
                # data base; user consent required; warning is skipped if user
                # already agreed; method stopped when user disagrees:
                elif (not user_agrees and jd_start > jd_next_obs_window):
                    date_next_obs_window = Time(
                        jd_next_obs_window, format='jd')
                    user_in = input(
                        'WARNING: The current start date for calculating '
                        'observing windows is later than the next date '
                        f'expected {date_next_obs_window} for some fields. '
                        'There will be gaps if continued. '
                        'Continue for all fields? (Y) ')

                    if user_in.lower() in ['y', 'yes', 'make it so!']:
                        user_agrees = True
                    else:
                        print('Aborted updating of observing windows!')

                        return None

                # check that stop JD is after start JD:
                if jd_stop < jd_start:
                    print('WARNING: start date is later than stop date.',
                          f'Field {field_id} skipped.')
                    continue

                # iterate through days:
                for jd in np.arange(jd_start, jd_stop, 1.):

                    # calculate observing windows:
                    date = Time(jd, format='jd').datetime
                    time_sunset, time_sunrise = \
                        self.telescope.get_sun_set_rise(
                                date.year, date.month, date.day, self.twilight)
                    time_interval_init = 10. * u.min
                    time_interval_refine = 1. * u.min
                    frame = self.telescope.get_frame(
                            time_sunset, time_sunrise, time_interval_init)
                    obs_windows = field.get_obs_window(
                            self.telescope, frame, refine=time_interval_refine)
                    self.batch_jd_stop[field_id] = jd + 1

                    # add observing windows to database:
                    for obs_window_start, obs_window_stop in obs_windows:
                        self.batch_field_ids.append(field_id)
                        self.batch_obs_windows_start.append(obs_window_start)
                        self.batch_obs_windows_stop.append(obs_window_stop)

                        # batch write to database:
                        if len(self.batch_field_ids) >= batch_write:
                            self._add_obs_windows_to_db(db)

            # batch write to database:
            self._add_obs_windows_to_db(db)

            print(f'\rField {j+1} of {n_fields_tbd} (100%)      ')

        # remove temporary storages:
        del self.batch_field_ids
        del self.batch_obs_windows_start
        del self.batch_obs_windows_stop
        del self.batch_jd_stop

        print('Calculating observing windows done.')

    #--------------------------------------------------------------------------
    def _add_obs_windows_to_db(self, db):
        """Add observing windows to the database.

        Parameters
        ----------
        db : db.DBConnectorSQLite
            Connection to the database.

        Returns
        -------
        None

        Notes
        -----
        This method adds observing windows to the database in a batch and
        updates the jd_next_obs_window entries in the Fields table.
        """

        if len(self.batch_field_ids):
            db.add_obs_windows(
                    self.batch_field_ids, self.batch_obs_windows_start,
                    self.batch_obs_windows_stop, active=True)

        # update Field information in data base:
        db.update_next_obs_window(
                list(self.batch_jd_stop.keys()),
                list(self.batch_jd_stop.values()))

        # reset data storages:
        self.batch_field_ids = []
        self.batch_obs_windows_start = []
        self.batch_obs_windows_stop = []
        self.batch_jd_stop = {}

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
