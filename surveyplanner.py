# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Sky fields for the pasiphae survey.
"""

from astropy.coordinates import AltAz, Angle, EarthLocation, get_sun
from astropy.time import Time, TimeDelta
from astropy import units as u
import numpy as np
from textwrap import dedent

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
    def __init__(self, start, stop):
        """Time window of observability.

        TODO"""

        self.start = start
        self.stop = stop
        self.duration = (stop - start).value * u.day

    #--------------------------------------------------------------------------
    def __str__(self):

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
        """A telescope.

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
        self.utc_offset = TimeDelta(utc_offset / 24.)
        self.telescope_id = telescope_id
        self.constraints = c.Constraints()

        print('Telescope: {0:s} created.'.format(self.name))

    #--------------------------------------------------------------------------
    def __str__(self):

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
        """Pasiphae survey planner.
        """

        self.dbname = dbname
        self.telescope = None
        self.twilight = None

    #--------------------------------------------------------------------------
    def _setup_observatory(self, observatory):
        """TBD
        """

        # connect to database:
        db = DBConnectorSQLite(self.dbname)

        # create telescope:
        telescope = db.get_observatory(observatory)
        self.telescope = Telescope(
                telescope['lat'], telescope['lon'], telescope['height'],
                telescope['utc_offset'], name=telescope['name'])

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
    def _iter_fields(self, observatory=None, active=None):
        """TBD
        """

        # read fields from database:
        db = DBConnectorSQLite(self.dbname)
        fields = db.get_fields(observatory=observatory, active=active)

        for (__, fov, center_ra, center_dec, tilt, __, field_id,
                latest_obs_window_jd) in fields:
            field = Field(
                fov, center_ra, center_dec, tilt, field_id=field_id,
                latest_obs_window_jd=latest_obs_window_jd)

            yield field

    #--------------------------------------------------------------------------
    def add_obs_windows(self, date_stop, date_start=None):
        """TBD
        """

        print(f'Calculate observing windows until {0}..'.format(
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

                        return False

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
        print(f'Calculating observing windows done.')

#==============================================================================
