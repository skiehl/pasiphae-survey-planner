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
from skyfields import Field

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

        # skip loading constraints:
        if no_constraints:
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
        database to a skyfields.Field instance.

        Parameters
        ----------
        field_tuple : tuple
            Tuple as returned e.g. by db.get_fields().

        Returns
        -------
        field : skyfields.Field
            A field instance created from the database entries.
        """

        field_id, fov, center_ra, center_dec, tilt, __, __, \
            latest_obs_window_jd, n_obs_done, n_obs_pending = field_tuple
        field = Field(
            fov, center_ra, center_dec, tilt, field_id=field_id,
            latest_obs_window_jd=latest_obs_window_jd, n_obs_done=n_obs_done,
            n_obs_pending=n_obs_pending)

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
        field : skyfields.Field
            Fields as stored in the database.
        """

        # read fields from database:
        db = DBConnectorSQLite(self.dbname)
        fields = db.iter_fields(observatory=observatory, active=active)

        for __, __, field in fields:
            field = self._tuple_to_field(field)

            yield field

    #--------------------------------------------------------------------------
    def add_obs_windows(self, date_stop, date_start=None):
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

        Returns
        -------
        None
        """

        print('Calculate observing windows until {0}..'.format(
                date_stop.iso[:10]))

        jd_stop = date_stop.jd
        user_agrees = False

        # connect to database:
        db = DBConnectorSQLite(self.dbname)

        # iterate through observatories:
        for i, m, observatory in db.iter_observatories():
            observatory_name = observatory['name']
            print(f'Observatory {i+1} of {m} selected: {observatory_name}')

            # setup observatory with constraints:
            self._setup_observatory(observatory_name)

            # iterate through fields associated with observatory:
            for j, n, field in db.iter_fields(
                    observatory=observatory_name, active=True):

                print(f'\rField {j+1} of {n} ({j/n*100:.1f}%)..', end='')

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
                    time_interval = 10. * u.min
                    frame = self.telescope.get_frame(
                            time_sunset, time_sunrise, time_interval)
                    obs_windows = field.get_obs_window(
                            self.telescope, frame, refine=1*u.min)

                    # add observing windows to database:
                    for obs_window_start, obs_window_stop in obs_windows:
                        db.add_obs_window(
                                field_id, obs_window_start, obs_window_stop,
                                active=True)

                # update Field information in data base:
                db.update_next_obs_window(field_id, jd_stop)

            print(f'\rField {j+1} of {n} (100%)   ')
        print('Calculating observing windows done.')

    #--------------------------------------------------------------------------
    def _set_field_status(self, field, date, days_before=3, days_after=7):
        """Determine and save a field's status (rising, plateauing, setting).

        Parameters
        ----------
        field : skyfields.Field
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

        # check that obs windows are available for time range
        if date_stop.jd > field.latest_obs_window_jd:
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

        # check status - unclear:
        elif n_windows < 4:
            pass

        # check status - rising/plateauing/setting:
        else:
            x = np.arange(n_windows) # assuming daily calculated obs windows
            x -= days_before
            result = linregress(x, durations)

            # check status - plateauing:
            if result.pvalue >= 0.01:
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
        field : skyfields.Field
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
        field : skyfields.Field
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
        field : skyfields.Field
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
        observable_fields : list of skyfields.Field
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
        field : skyfields.Field
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
        field : list of skyfields.Field
            Field(s) fulfilling the selected criteria.
        """

        fields = [field for field in self.iter_fields(
                observatory=observatory, observed=observed, pending=pending,
                active=active)]

        return fields

    #--------------------------------------------------------------------------
    def get_field_by_id(self, field_id):
        """Get a field by its database ID.

        Parameters
        ----------
        field_id : int
            ID of the field as stored in the database.

        Raises
        ------
        ValueError
            Raise if no field with this ID exists.

        Returns
        -------
        field : skyfields.Field
            Field as stored in the database under specified ID.
        """

        # connect to database:
        db = DBConnectorSQLite(self.dbname)

        field = db.get_field_by_id(field_id)

        if not len(field):
            raise ValueError(f"Field with ID {field_id} does not exist.")

        field = self._tuple_to_field(field[0])

        return field

    #--------------------------------------------------------------------------
    def _count_neighbors(
            self, radius, field_id, observatory=None, observed=None,
            pending=None, active=True, return_neighbor_ids=False):
        """Count the neighbors of a specific field given various criteria.

        Parameters
        ----------
        radius : astropy.units.Quantity
            Radius in which to count field neighbors. Needs to be a Quantity
            with unit 'rad' or 'deg'.
        field_id : int
            ID of the field whose neighbors are searched for.
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
        return_neighbor_ids : bool, optional
            If True, the neighbor IDs are returnd as well. The default is
            False.

        Returns
        -------
        count : int
            Number of field neighbors within the specified radius that fulfill
            the conditions.
        neighbor_ids : np.ndarray
            IDs of the neighbors fulfilling the conditions. Only returned if
            'return_neighbor_ids' is True.
        """

        # get coordinates of field of interest:
        field = self.get_field_by_id(field_id)
        field_coord = SkyCoord(field.center_ra, field.center_dec)

        # get coordinates of all fields relevant for counting:
        fields_id = []
        fields_ra = []
        fields_dec = []

        for field in self.iter_fields(
                observatory=observatory, observed=observed, pending=pending,
                active=active):

            # skip the target field itself:
            if field_id == field.id:
                continue

            fields_id.append(field.id)
            fields_ra.append(field.center_ra)
            fields_dec.append(field.center_dec)

        fields_coord = SkyCoord(fields_ra, fields_dec)

        # count neighbors:
        sel = field_coord.separation(fields_coord) <= radius
        count = np.sum(sel)

        if return_neighbor_ids:
            neighbor_ids = np.array(fields_id)[sel]

            return count, neighbor_ids

        return count

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
        neighbor_count : list of int
            Lists the number of field neighbors within the specified radius
            that fulfill the conditions for each of the input field IDs.
        """

        if isinstance(field_ids, int):
            neighbor_count = self._count_neighbors(
                    radius, field_ids, observatory=observatory,
                    observed=observed, pending=pending, active=active)

        elif isinstance(field_ids, list):
            neighbor_count = []

            for field_id in field_ids:
                count = self._count_neighbors(
                        radius, field_id, observatory=observatory,
                        observed=observed, pending=pending, active=active)
                neighbor_count.append(count)

        else:
            raise ValueError("'field_ids' has to be int or list.")

        return neighbor_count

#==============================================================================
