#!/usr/bin/env python3
"""Pasiphae survey planner.
"""

from astropy.coordinates import AltAz, Angle, EarthLocation, get_sun, SkyCoord
from astropy.time import Time, TimeDelta
from astropy import units as u
from itertools import repeat
from multiprocessing import Manager, Pool, Process
import numpy as np
from pandas import DataFrame
from statsmodels.api import add_constant, OLS
from time import sleep
from textwrap import dedent

import constraints as c
from db import FieldManager, GuidestarManager, ObservabilityManager, \
        ObservationManager, TelescopeManager
from prioritizer import Prioritizer
from utilities import true_blocks

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
            jd_first_observability=None, jd_next_observability=None,
            n_obs_tot=0, n_obs_done=0, n_obs_pending=0):
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
        jd_first_observability : float, optional
            The earliest Julian date for which an observing window was
            calculated for this field. The default is None.
        jd_next_observability : float, optional
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
        self.jd_first_observability = jd_first_observability
        self.jd_next_observability = jd_next_observability
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
            Name of the telescope/telescope.
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

class ObservabilityPlanner:
    """Observability planner: Determine when fields are observable.
    """

    #--------------------------------------------------------------------------
    def __init__(self, dbname):
        """Create ObservabilityPlanner instance.

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
    def _check_db(self, all_fields):
        """Check if database is properly set up for observability calculations.

        Parameters
        ----------
        all_fields : bool
            If True, only warn about missing observations, but do not stop
            calculation of observabilities. Otherwise, stop calculation, if no
            observations are stored in the database.

        Returns
        -------
        ready : bool
            True, if the database is set up for the observability calculations.
            False, otherwise.
        """

        ready = True
        manager = ObservabilityManager(self.dbname)
        counts = manager.dbstatus(verbose=0)

        # check that telescopes exist:
        if not counts['telescopes']:
            print('\nWARNING: No telescopes stored in the database!')
            ready = False

        # check that constraints exist for each telescope:
        if np.any(np.array(counts['constraints per telescope']) == 0):
            print('\nWARNING: No constrains stored for some telescopes!')
            ready = False

        # check that fields exist:
        if not sum(counts['fields per telescope']):
            print('\nWARNING: No fields stored in database!')
            ready = False

        # warn if no fields exist for some telescope(s):
        if np.any(np.array(counts['fields per telescope']) == 0):
            print('\nWARNING: No fields stored for some telescopes!')
            # only warn

        # check that observations exist:
        if not counts['observations']:
            print('\nWARNING: No observations stored in database!')

            if all_fields:
                print('Calculate observabilities for all fields anyway.')
            else:
                ready = False

        if not ready:
            print('Use `DBCreator()` class from db.py to set up the ' \
                  'database.\nObservabilities cannot be calculated. Aborted!\n'
                  )

        return ready

    #--------------------------------------------------------------------------
    def _dict_to_field(self, field_dict):
        """Convert a dict that contains field information as queried from the
        database to a Field instance.

        Parameters
        ----------
        field_dict : dict
            Dict as returned by db.get_fields().

        Returns
        -------
        field : Field
            A field instance created from the database entries.
        """

        field_id = field_dict['field_id']
        fov = field_dict['fov']
        center_ra = field_dict['center_ra']
        center_dec = field_dict['center_dec']
        tilt = field_dict['tilt']
        jd_first = field_dict['jd_first']
        jd_next = field_dict['jd_next']
        n_obs_tot = field_dict['nobs_tot']
        n_obs_done = field_dict['nobs_done']
        n_obs_pending = field_dict['nobs_pending']
        field = Field(
            fov, center_ra, center_dec, tilt, field_id=field_id,
            jd_first_observability=jd_first, jd_next_observability=jd_next,
            n_obs_tot=n_obs_tot, n_obs_done=n_obs_done,
            n_obs_pending=n_obs_pending)

        return field

    #--------------------------------------------------------------------------
    def _setup_telescope(self, telescope):
        """Load telescope parameters from database and add Telescope instance
        to SurveyPlanner.

        Parameters
        ----------
        telescope : dict
            Telescope parameters as returned by `db.get_telescopes()`.

        Returns
        -------
        parameter_set_id : int
            Parameter set ID corresponding to the used constraints.
        """

        # create telescope:
        self.telescope = Telescope(
                telescope['lat']*u.rad, telescope['lon']*u.rad,
                telescope['height'], telescope['utc_offset'],
                name=telescope['name'])
        parameter_set_id = telescope['parameter_set_id']

        # read constraints:
        constraints = telescope['constraints']
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
        db : db.ObservabilityManager
            Observability manager database connection.
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
            Queue containing the IDs of fields whose `jd_next_observability`
            value needs to be updated.
        n : int
            Number of entries to read from the queue.
        duration_limit : astropy.time.TimeDelta
            Limit on the observing window duration above which a field is
            considered observable.

        Returns
        -------
        batch_field_ids : list of int
            IDs of the fields for Observability table entries and updating
            `jd_next_observability` value.
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
        db : db.ObservabilityManager
            Observability manager database connection.
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

        jd_next = jd + 1

        while True:
            sleep(1)
            n_done = counter.value
            n_queued = queue_obs_windows.qsize()

            if n_done >= n_tbd and n_queued == 0:
                break

            print(f'\rCalculations: {n_done}/{n_tbd} ' \
                  f'({n_done/n_tbd*100:.1f}%). Queued: {n_queued}. '\
                  'Processing..                                      ', end='')

            # extract batch of results from queue:
            if n_done >= n_tbd or n_queued >= batch_write:
                print(f'\rCalculations: {n_done}/{n_tbd} ' \
                      f'({n_done/n_tbd*100:.1f}%). Queued: {n_queued}. ' \
                      'Reading from queue..                          ', end='')
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
                print(f'\rCalculations: {n_done}/{n_tbd} ' \
                      f'({n_done/n_tbd*100:.1f}%). Queued: {n_queued}. ' \
                      f'Writing {n_queried} entries to database..', end='')

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

        print('\rProgress: done                                             ' \
              '                        ')

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
        if not init and field.jd_next_observability > jd:
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
            self, fields, init, telescope, date_stop, date_start,
            duration_limit, batch_write, processes, time_interval_init,
            time_interval_refine, agreed_to_gaps):
        """

        Parameters
        ----------
        fields list of tuples
            List of the queried fields. Each tuple contains the field
            parameters.
        init : bool
            If True, observing windows are calculated for these fields for the
            first time, which requires some additional action. Otherwise, new
            observing windows are appended.
        telescope : dict
            Telescope data as returned by `db.get_telescopes()`.
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
        db = ObservabilityManager(self.dbname)

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

        # setup telescope with constraints:
        parameter_set_id = self._setup_telescope(telescope)

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
    def _get_fields_missing_status(self):
        """Get fields that have associated observing windows without
        observability status information.

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

        db = FieldManager(self.dbname)
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
        status_to_id = db.status_to_id_converter()

        while True:
            sleep(1)
            n_done = counter.value
            n_queued = queue_status.qsize()

            if n_done >= n_tbd and n_queued == 0:
                break

            print(f'\rCalculations: {n_done}/{n_tbd} ' \
                  f'({n_done/n_tbd*100:.1f}%). Queued: {n_queued}. ' \
                  'Processing..                             ', end='')

            # extract batch of results from queue:
            if n_done >= n_tbd or n_queued >= batch_write:
                print(f'\rCalculations: {n_done}/{n_tbd} ' \
                      f'({n_done/n_tbd*100:.1f}%). Queued: {n_queued}. ' \
                      f'Reading from queue..                     ', end='')
                batch_observability_ids, batch_status, \
                batch_setting_in = self._read_status_from_queue(
                        db, queue_status, batch_write, status_to_id)
                n_queried = len(batch_observability_ids)
                write = True

            else:
                write = False

            # write results to database:
            if write:
                print(f'\rCalculations: {n_done}/{n_tbd} ' \
                      f'({n_done/n_tbd*100:.1f}%). Queued: {n_queued}. ' \
                      f'Writing {n_queried} entries to database..', end='')

                # update observabilities in database:
                db.update_observability_status(
                        batch_observability_ids, batch_status,
                        batch_setting_in)

                n_done += n_queried

        print('\rProgress: done                                             ' \
              '                        ')

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
            self, days_before, days_after, outlier_threshold,
            status_threshold, time_interval, batch_write=10000, processes=1):
        """Determine the observability status for each field for each day and
        save it in the database.

        Parameters
        ----------
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

        db = ObservabilityManager(self.dbname)
        field_ids, jds_min, jds_max = self._get_fields_missing_status()
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
    def observability(
            self, date_stop, date_start=None, duration_limit=60,
            batch_write=1000, processes=1, time_interval_init=600,
            time_interval_refine=60, days_before=7, days_after=7,
            outlier_threshold=0.7, status_threshold=6, all_fields=False):
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
        all_fields : bool, optional
            If False, observabilities are calculated only for fields with
            pending observations. If True, observabilities are calculated for
            all fields regardless of whether they have pending observations
            associated or not. The default is False.

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

        # check that database is properly set up for calculations:
        if not self._check_db(all_fields):
            return None

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

        # get telescopes from database:
        telescope_manager = TelescopeManager(self.dbname)
        telescopes = telescope_manager.get_telescopes(constraints=True)
        n_tel = len(telescopes)
        del telescope_manager

        jd_stop = date_stop.jd
        agreed_to_gaps = None

        if all_fields:
            pending = None
        else:
            pending = True

        # iterate through observatories:
        for i, telescope in enumerate(telescopes, start=1):
            telescope_name = telescope['name']
            print(f'\nTelescope {i} of {n_tel} selected: {telescope_name}')

            # get fields that need observing window calculations:
            print('Query fields..')
            field_manager = FieldManager(self.dbname)
            fields_init = field_manager.get_fields(
                    telescope=telescope_name, pending=pending,
                    init_obs_windows=True)
            fields_tbd = field_manager.get_fields(
                    telescope=telescope_name, pending=pending,
                    needs_obs_windows=jd_stop)
            del field_manager

            with Pool(processes=processes) as pool:
                    fields_init = pool.map(self._dict_to_field, fields_init)
                    fields_tbd = pool.map(self._dict_to_field, fields_tbd)

            # calculate observing windows for new fields:
            agreed_to_gaps = self._add_obs_windows(
                    fields_init, True, telescope, date_stop,
                    date_start, duration_limit, batch_write, processes,
                    time_interval_init, time_interval_refine, agreed_to_gaps)

            # calculate observing windows for fields:
            agreed_to_gaps = self._add_obs_windows(
                    fields_tbd, False, telescope, date_stop,
                    date_start, duration_limit, batch_write, processes,
                    time_interval_init, time_interval_refine, agreed_to_gaps)

        # determine observability status for each field for each day:
        self._add_status(
                days_before, days_after, outlier_threshold,
                status_threshold, time_interval_refine,
                batch_write=batch_write, processes=processes)

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
        self.prioritizers = {}
        self.weights = {}

    #--------------------------------------------------------------------------
    def _get_observable_fields_by_night(
            self, date, telescope=None, observed=None, pending=None,
            active=True):
        """Get fields observable during a given night, given specific selection
        criteria.

        Parameters
        ----------
        date : astropy.time.Time
            Date of the night start. Time information is truncated.
        telescope : str
            If specified, only fields associated with this telescope are
            returned. The default is None.
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

        Raises
        ------
        ValueError
            Raised, if `telscope` is neither string nor None.

        Returns
        ------
        fields : list of dict
            Each list item is a dict with the field parameters.
        """

        # database connections:
        telescope_manager= TelescopeManager(self.dbname)
        field_manager = FieldManager(self.dbname)

        # check input:
        if isinstance(telescope, str):
            telescopes = [telescope]
        elif telescope is None:
            telescopes = telescope_manager.get_telescope_names()
        else:
            raise ValueError('`telescope` must be string or None.')

        # warn user if time was set:
        if date.iso[11:] != '00:00:00.000':
            print("WARNING: For argument `observable_night` provide " \
                  "date only. Time information is truncated. To get " \
                  "fields observable at specific time use the " \
                  "`observable_time` argument.")

        # truncate time information from date:
        date = Time(date.iso[:10])

        fields = []

        # iterate through telescopes:
        for telescope_name in telescopes:
            # get local noon of current and next day in UTC:
            telescope = telescope_manager.get_telescopes(telescope_name)[0]
            utc_offset = telescope['utc_offset'] * u.h
            noon_current = date + 12 * u.h - utc_offset
            noon_next = date + 36 * u.h - utc_offset
            # NOTE: ignoring daylight saving time

            # query fields:
            fields += field_manager.get_fields(
                    telescope=telescope_name, observed=observed,
                    pending=pending,
                    observable_between=(noon_current.iso, noon_next.iso),
                    active=active)

        return fields

    #--------------------------------------------------------------------------
    def _get_observable_fields_by_datetime(
            self, date, telescope=None, observed=None, pending=None,
            active=True):
        """Get fields observable during a given night, given specific selection
        criteria.

        Parameters
        ----------
        date : astropy.time.Time
            Date of the night start. Time information is truncated.
        telescope : str, optional
            If specified, only fields associated with this telescope are
            returned. The default is None.
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
        fields : list of dict
            Each list item is a dict with the field parameters.
        """

        # database connections:
        field_manager = FieldManager(self.dbname)

        fields = field_manager.get_fields(
                telescope=telescope, observed=observed, pending=pending,
                observable_time=date.iso, active=active)

        return fields

    #--------------------------------------------------------------------------
    def _get_fields(
            self, telescope=None, observed=None, pending=None, active=True):
        """Get a list of fields, given specific selection criteria.

        Parameters
        ----------
        telescope : str, optional
            If specified, only fields associated with this telescope are
            returned. The default is None.
        observed : bool or None, optional
            If True, only fields that have been observed at least once are
            returned. If False, only fields that have never been observed are
            returned. If None, fields are returned regardless of whether they
            have been observed or not. The default is None.
        pending : bool or None, optional
            If True, only fields that have pending observations associated are
            returned. If False, only fields that have no pending observations
            associated are returned. If None, fields are returned regardless of
            whether they have pending observations associated or not. The
            default is None.
        active : bool or None, optional
            If True, only active fields are returned. If False, only inactive
            fields are returned. If None, active and deactivated fields are
            returned. The default is True.

        Returns
        -------
        fields : list of dict
            Each list item is a dict with the field parameters.
        """

        # connect to database:
        field_manager = FieldManager(self.dbname)

        fields = field_manager.get_fields(
                telescope=telescope, observed=observed, pending=pending,
                active=active)

        return fields

    #--------------------------------------------------------------------------
    def _get_priorities(self, fields):
        """Assign priorities to fiels.

        Parameters
        ----------
        fields : list of dict
            List of field dictionaries as returned by
            surveyplanner.Surveyplanner.get_fields().

        Returns
        -------
        None
        """

        priorities = {}

        # iterate through prioritizers:
        for label, prioritizer in self.prioritizers.items():
            print(f'  {label}.. ', end='')
            priorities[label], __ = prioritizer.prioratize(fields)
            print('done')

        priorities_joint = [
                priorities[label] * self.weights[label] \
                for label in self.prioritizers.keys()]
        priorities_joint = np.sum(priorities_joint, axis=0)
        priorities['Joint'] = priorities_joint

        # add priorities to fields:
        print('Add priorities to fields..')
        labels = priorities.keys()

        # iterate through fields:
        for i, field in enumerate(fields):
            for label in labels:
                field[f'priority{label}'] = priorities[label][i]

    #--------------------------------------------------------------------------
    def _get_guidestars(self, fields):
        """
        Add guide stars to the provided fields.

        Parameters
        ----------
        fields : list
            List of field dictionaries as returned by
            surveyplanner.Surveyplanner.get_fields().

        Returns
        -------
        None
        """

        print('Add guide stars to fields..')

        # query guidestars:
        manager = GuidestarManager(self.dbname)
        guidestars = manager.get_guidestars()
        field_ids = np.array(
                [guidestar['field_id'] for guidestar in guidestars])

        # add guidestars to fields:
        for field in fields:
            i_sel = np.nonzero(field['field_id'] == field_ids)[0]
            field_guidestars = []

            # remove irrelevant info from dict(s):
            for i in i_sel:
                guidestar = guidestars[i]
                del guidestar['field_id']
                del guidestar['active']
                field_guidestars.append(guidestar)

            field['guidestars'] = field_guidestars

    #--------------------------------------------------------------------------
    def _get_observations(self, fields):
        """
        Add pending observations to the provided fields.

        Parameters
        ----------
        fields : list
            List of field dictionaries as returned by
            surveyplanner.Surveyplanner.get_fields().

        Returns
        -------
        None
        """

        print('Add observations to fields..')

        # query observations:
        manager = ObservationManager(self.dbname)
        observations = manager.get_observations(done=False)
        field_ids = np.array(
                [observation['field_id'] for observation in observations])

        # add guidestars to fields:
        for field in fields:
            i_sel = np.nonzero(field['field_id'] == field_ids)[0]
            field_observations = []

            # remove irrelevant info from dict(s):
            for i in i_sel:
                observation = observations[i]
                del observation['field_id']
                del observation['active']
                del observation['done']
                del observation['scheduled']
                del observation['date_done']
                field_observations.append(observation)

            field['observations'] = field_observations

    #--------------------------------------------------------------------------
    def get_fields(
            self, observable_night=None, observable_time=None, telescope=None,
            observed=None, pending=None, active=True):
        """Get a list of fields, given specific selection criteria.

        Parameters
        ----------
        observable_night : astropy.time.Time or str, optional
            If a date is given, only fields are returned that are observable
            during the night that starts on the specified date. Time
            information is truncated. Either set this argument or
            `observable_time`. If this argument is set, `observable_time` is
            not used. The default is None.
        observable_time : astropy.time.Time or str, optional
            If a date and time is given, only fields are returned that are
            observable at that specific time. Either set this argument or
            `observable_night`. If `observable_night` is given, this argument
            is ignored. The default is None.
        telescope : str, optional
            If specified, only fields associated with this telescope are
            returned. The default is None.
        observed : bool or None, optional
            If True, only fields that have been observed at least once are
            returned. If False, only fields that have never been observed are
            returned. If None, fields are returned regardless of whether they
            have been observed or not. The default is None.
        pending : bool or None, optional
            If True, only fields that have pending observations associated are
            returned. If False, only fields that have no pending observations
            associated are returned. If None, fields are returned regardless of
            whether they have pending observations associated or not. The
            default is None.
        active : bool or None, optional
            If True, only active fields are returned. If False, only inactive
            fields are returned. If None, active and deactivated fields are
            returned. The default is True.

        Raises
        ------
        ValueError
            Raised, if `observable_night` is neither astropy.time.Time nor str.
            Raised, if `observable_time` is neither astropy.time.Time nor str.

        Returns
        -------
        fields : list of dict
            Each list item is a dict with the field parameters.
        """

        # get fields observable during a specific night:
        if observable_night is not None:
            # check input:
            if isinstance(observable_night, Time):
                pass
            elif isinstance(observable_night, str):
                observable_night = Time(observable_night)
            else:
                raise ValueError(
                        "`observable_night` must be astropy.time.Time or str.")

            # get fields:
            fields = self._get_observable_fields_by_night(
                    observable_night, telescope=telescope, observed=observed,
                    pending=pending, active=active)

        # get fields observable at a specific time:
        elif observable_time is not None:
            # check input:
            if isinstance(observable_time, Time):
                pass
            elif isinstance(observable_time, str):
                observable_time = Time(observable_time)
            else:
                raise ValueError(
                        "`observable_time` must be astropy.time.Time or str.")

            # get fields:
            fields = self._get_observable_fields_by_datetime(
                    observable_time, telescope=telescope, observed=observed,
                    pending=pending, active=active)

        # get fields regardless of their observability status:
        else:
            fields = self._get_fields(
                    telescope=telescope, observed=observed, pending=pending,
                    active=active)

        return fields

    #--------------------------------------------------------------------------
    def set_prioritizer(self, *prioritizers, weights=None):
        """Set prioritizers.

        Parameters
        ----------
        *prioritizers : Prioritizer-instance
            One or multiple prioritizers (defined in prioritizer.py) that will
            assign priorities to fields.
        weights : list of floats or None, optional
            Weights are only relevant if multiple prioritizers are set. If no
            weights are given, all priorities will be weighted equally. If
            weights are given, priorities are weighted accordingly and then
            summed. The same number of weights must be provided as number of
            prioritizers. The default is None.

        Raises
        ------
        TypeError
            Raised, if `weights` is neither list, tuple, nor None.
            Raised, if any values given to prioritizers are not
            Prioritizer-instances.
        ValueError
            Raised, if `weights` does not match the number of prioritizers.
            Raised, if the elements in `weights` are not floats larger than 0.


        Returns
        -------
        None
        """

        print("Set prioritizer(s)..")
        self.prioritizers = {}
        self.weights = {}

        # check weights:
        if weights is None:
            weights = np.ones(len(prioritizers))

        elif weights is not None:
            if type(weights) not in [list, tuple]:
                raise TypeError("`weights` must be list or tuple.")

            if len(prioritizers) != len(weights):
                raise ValueError(
                        "Length of `weights` does not match length of " \
                        "`prioratizers`.")

        else:
            raise ValueError("`weights` must be list, tuple, or None.")

        for weight in weights:
            if type(weight) not in [float, int] or weight <= 0:
                raise ValueError("All `weights` must be float (or int) >0.")

        # normalize weights:
        weights = np.array(weights, dtype=float)
        weights /= np.sum(weights)

        # iterate through prioritizers:
        for i, prioritizer in enumerate(prioritizers):
            # check input:
            if not isinstance(prioritizer, Prioritizer):
                raise TypeError(
                        "`prioritizer` must be a Prioritizer class instance.")

            # store prioritizer and weight:
            self.prioritizers[prioritizer.label] = prioritizer
            self.weights[prioritizer.label] = weights[i]
            print(f"Added prioritizer {prioritizer.label}.")

    #--------------------------------------------------------------------------
    def plan(self, night, telescope):
        """Identify and prioritize observable fields with pending observations
        for a specific night and telescope.

        Parameters
        ----------
        night : astropy.time.Time or str
            The date on which the night starts for which field observations
            should be planned.
        telescope : str
            Name of the telescope for which field observations should be
            planned.

        Raises
        ------
        TypeError
            Raise, if no prioritizer(s) have been set yet.

        Returns
        -------
        None
        """

        # check if prioritizers have been set:
        if not len(self.prioritizers):
            raise TypeError(
                    "No prioritizer(s) set yet. Use `set_prioritizer()` first."
                    )

        timeit_start = Time.now()
        # get active, observable, pending fields:
        print('Query observable, pending fields..')
        fields = self.get_fields(
                observable_night=night, telescope=telescope, pending=True,
                active=True)
        print('Finished after', Time.now() - timeit_start)

        # prioritize:
        print('Get priorities..')
        timeit_start = Time.now()
        self._get_priorities(fields)
        print('Finished after', Time.now() - timeit_start)

        timeit_start = Time.now()
        self._get_guidestars(fields)
        print('Finished after', Time.now() - timeit_start)

        timeit_start = Time.now()
        self._get_observations(fields)
        print('Finished after', Time.now() - timeit_start)

        self.fields = fields

#==============================================================================
