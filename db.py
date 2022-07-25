# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Sky fields for the pasiphae survey.
"""

from astropy.time import Time
from astropy import units as u
import os
import sqlite3
import warnings

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

class NotInDatabase(Exception):
    """Custom exception raised when value does not exist in database."""

#==============================================================================

class SQLiteConnection:
    """Wrapper class for SQLite connection that allows the use of the
    python with-statement."""

    def __init__(self, db_file):
        """Wrapper class for SQLite connection that allows the use of the
        python with-statement.

        TODO
        """

        self.db_file = db_file

    def __enter__(self):

        self.connection = sqlite3.connect(self.db_file)

        return self.connection

    def __exit__(self, type, value, traceback):
        self.connection.close()

#==============================================================================

class DBConnectorSQLite:
    """SQLite database connector."""

    #--------------------------------------------------------------------------
    def __init__(self, db_file):
        """SQLite database connector."""

        self.db_file = db_file

    #--------------------------------------------------------------------------
    #def _connect(self):
    #    """TODO"""
    #
    #    self.connection

    #--------------------------------------------------------------------------
    def _query(self, connection, query, many=False, commit=False):
        """TODO"""

        cursor = connection.cursor()
        if many is False:
            result = cursor.execute(query)
        else:
            result = cursor.executemany(query, many)

        if commit:
            connection.commit()

        return result

    #--------------------------------------------------------------------------
    def _get_observatory_id(self, name):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT observatory_id FROM Observatories
                WHERE name = '{0}'
                """.format(name)
            result = self._query(connection, query).fetchone()

        if result is None:
            raise NotInDatabase(f"Observatory '{name}' does not exist.")

        observatory_id = result[0]

        return observatory_id

    #--------------------------------------------------------------------------
    def _get_parameter_set_id(self, observatory_id):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT parameter_set_id
                FROM ParameterSets
                WHERE (active = 1
                       AND observatory_id = {0})
                """.format(observatory_id)
            result = self._query(connection, query).fetchone()

        if result is None:
            parameter_set_id = -1
        else:
            parameter_set_id = result[0]

        return parameter_set_id

    #--------------------------------------------------------------------------
    def _inactivate_parameter_set(self, parameter_set_id):
        """TODO
        """

        with SQLiteConnection(self.db_file) as connection:

            query = """\
                UPDATE ParameterSets
                SET active = False
                WHERE parameter_set_id = {0}
                """.format(parameter_set_id)
            self._query(connection, query, commit=True)

    #--------------------------------------------------------------------------
    def _last_insert_id(self, connection):
        """TODO"""

        query = """SELECT last_insert_rowid()"""
        result = self._query(connection, query).fetchone()
        last_insert_id = result[0]

        return last_insert_id

    #--------------------------------------------------------------------------
    def _get_constraint_id(self, constraint_name):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT constraint_id FROM Constraints
                WHERE constraint_name = '{0}'
                """.format(constraint_name)
            result = self._query(connection, query).fetchone()

        if result is None:
            raise NotInDatabase(
                    f"Constraint '{constraint_name}' does not exist.")

        constraint_id = result[0]

        return constraint_id

    #--------------------------------------------------------------------------
    def _get_parameter_name_id(self, parameter_name):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:

            # check for parameter name:
            query = """\
                SELECT parameter_name_id FROM ParameterNames
                WHERE parameter_name = '{0}'
                """.format(parameter_name)
            result = self._query(connection, query).fetchone()

            # parameter name does not exist:
            if result is None:

                # add parameter name:
                query = """\
                    INSERT INTO ParameterNames (parameter_name)
                    VALUES ('{0}')
                    """.format(parameter_name)
                self._query(connection, query, commit=True)
                parameter_name_id = self._last_insert_id(connection)

            # parameter name exists:
            else:
                parameter_name_id = result[0]

        return parameter_name_id

    #--------------------------------------------------------------------------
    def _add_parameter(
            self, constraint_id, parameter_set_id, parameter_name, value=None,
            svalue=None):
        """TODO"""

        parameter_name_id = self._get_parameter_name_id(parameter_name)

        # add parameter:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                INSERT INTO Parameters (
                    constraint_id, parameter_set_id, parameter_name_id, value,
                    svalue)
                VALUES ({0}, {1}, {2}, {3}, {4})
                """.format(
                        constraint_id, parameter_set_id, parameter_name_id,
                        'NULL' if value is None else value,
                        'NULL' if svalue is None else f"'{svalue}'")

            self._query(connection, query, commit=True)

    #--------------------------------------------------------------------------
    def _add_parameter_set(self, observatory):
        """TODO"""

        # check if active parameter set exists:
        observatory_id = self._get_observatory_id(observatory)
        parameter_set_id = self._get_parameter_set_id(observatory_id)

        if parameter_set_id == -1:
            active = True
        else:
            response = input(
                f"An active parameter set for observator '{observatory} "
                "exists. Set former one inactive and new one active? (y/n)")

            if response.lower() in ['y', 'yes', 'make it so!']:
                active = True
                self._inactivate_parameter_set(parameter_set_id)
            else:
                active = False

        # add parameter set and parameters:
        with SQLiteConnection(self.db_file) as connection:

            # add parameter set:
            query = """\
                INSERT INTO ParameterSets (observatory_id, active, date)
                VALUES ({0}, {1}, CURRENT_TIMESTAMP)
                """.format(observatory_id, active)
            self._query(connection, query, commit=True)
            parameter_set_id = self._last_insert_id(connection)

        return parameter_set_id

    #--------------------------------------------------------------------------
    def _add_twilight(self, twilight, parameter_set_id):
        """TODO"""

        # parse input:
        if isinstance(twilight, float):
            pass
        elif twilight == 'astronomical':
            twilight = -18.
        elif twilight == 'nautical':
            twilight = -12.
        elif twilight == 'civil':
            twilight = -6.
        elif twilight == 'sunset':
            twilight = 0.
        else:
            raise ValueError(
                "Either set a float or chose from 'astronomical', "
                "'nautical', 'civil', or 'sunset'.")

        # add twilight:
        constraint_id = self._get_constraint_id('Twilight')
        self._add_parameter(
                constraint_id, parameter_set_id, 'twilight',
                value=twilight)

    #--------------------------------------------------------------------------
    def _add_constraint(self, constraint, parameter_set_id):
        """TODO"""

        # get constraint ID:
        constraint_name = constraint.__class__.__name__
        constraint_id = self._get_constraint_id(constraint_name)

        # iterate through parameters:
        for param, value in constraint.get_params().items():
            if isinstance(value, str):
                self._add_parameter(
                        constraint_id, parameter_set_id, param,
                        svalue=value)
            else:
                try:
                    value = float(value)
                    self._add_parameter(
                            constraint_id, parameter_set_id, param,
                            value=value)
                except:
                    raise ValueError(
                            "Value should be string, int, or float. Check the "
                            "constraint get_params() method.")

    #--------------------------------------------------------------------------
    def create_db(self):
        """TODO"""

        create = False

        # check if file exists:
        if os.path.exists(self.db_file):
            answer = input(
                'Database file exists. Overwrite (y) or cancel (enter)?')

            if answer.lower() in ['y', 'yes', 'make it so!']:
                os.system(f'rm {self.db_file}')
                create = True

        else:
            create = True

        # create file:
        if create:
            os.system(f'sqlite3 {self.db_file}')

        print(f"Database '{self.db_file}' created.")

        # create tables:
        with SQLiteConnection(self.db_file) as connection:

            # create Fields tables:
            query = """\
                CREATE TABLE Fields(
                    field_id integer PRIMARY KEY,
                    fov float,
                    center_ra float,
                    center_dec float,
                    tilt float,
                    observatory_id integer,
                    active boolean,
                    jd_next_obs_window float)
                """
            self._query(connection, query, commit=True)
            print("Table 'Fields' created.")

            # create Observatory table:
            query = """\
                CREATE TABLE Observatories(
                    observatory_id integer PRIMARY KEY,
                    name char(30),
                    lat float,
                    lon float,
                    height float,
                    utc_offset float)
                """
            self._query(connection, query, commit=True)
            print("Table 'Observatories' created.")

            # create ParameterSet table:
            query = """\
                CREATE TABLE ParameterSets(
                    parameter_set_id integer PRIMARY KEY,
                    observatory_id integer,
                    active bool,
                    date date)
                """
            self._query(connection, query, commit=True)
            print("Table 'ParameterSet' created.")

            # create Constraints table:
            query = """\
                CREATE TABLE Constraints(
                    constraint_id integer PRIMARY KEY,
                    constraint_name char(30))
                """
            self._query(connection, query, commit=True)
            print("Table 'Constraints' created.")

            # create Parameters table:
            query = """\
                CREATE TABLE Parameters(
                    parameter_id integer PRIMARY KEY,
                    constraint_id integer,
                    parameter_set_id integer,
                    parameter_name_id integer,
                    value float,
                    svalue char(30))
                """
            self._query(connection, query, commit=True)
            print("Table 'Parameters' created.")

            # create ParameterNames table:
            query = """\
                CREATE TABLE ParameterNames(
                    parameter_name_id integer PRIMARY KEY,
                    parameter_name char(30))
                """
            self._query(connection, query, commit=True)
            print("Table 'ParameterNames' created.")

            # create ObsWindows table:
            query = """\
                CREATE TABLE ObsWindows(
                    obswindow_id integer PRIMARY KEY,
                    field_id integer,
                    date_start date,
                    date_stop date,
                    duration float,
                    active bool)
                """
            self._query(connection, query, commit=True)
            print("Table 'ObsWindows' created.")

            # create Observations table:
            query = """\
                CREATE TABLE Observations(
                    observation_id integer PRIMARY KEY,
                    field_id integer,
                    exposure float,
                    repetitions int,
                    filter_id int,
                    scheduled bool,
                    done bool,
                    date date)
                """
            self._query(connection, query, commit=True)
            print("Table 'Observations' created.")

            # create Filters table:
            query = """\
                CREATE TABLE Filters(
                    filter_id integer PRIMARY KEY,
                    filter char(10))
                """
            self._query(connection, query, commit=True)
            print("Table 'Observations' created.")

            # define constraints:
            query = """\
                INSERT INTO Constraints (constraint_name)
                VALUES
                    ('Twilight'),
                    ('ElevationLimit'),
                    ('AirmassLimit'),
                    ('MoonDistance'),
                    ('MoonPolarization')
                """
            self._query(connection, query, commit=True)
            print("Constraints added to table 'Constraints'.")

    #--------------------------------------------------------------------------
    def add_observatory(self, name, lat, lon, height, utc_offset):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            # check if name exists:
            query = """\
                SELECT name FROM Observatories
                WHERE name = '{0}'
                """.format(name)
            result = self._query(connection, query).fetchall()

            if len(result) > 0:
                print(f"Observatory '{name}' already exists. Name needs " \
                      "to be unique.")

                return None

            # add observatory to database:
            query = """\
                INSERT INTO Observatories (name, lat, lon, height, utc_offset)
                VALUES ('{0}', {1}, {2}, {3}, {4});
                """.format(name, lat, lon, height, utc_offset)
            self._query(connection, query, commit=True)
            print(f"Observatory '{name}' added.")

    #--------------------------------------------------------------------------
    def get_observatory(self, name):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT * FROM Observatories
                WHERE name='{0}'
                """.format(name)
            result = self._query(connection, query).fetchone()

        telescope = {
            'telescope_id': result[0],
            'name': result[1],
            'lat': result[2] * u.rad,
            'lon': result[3] * u.rad,
            'height': result[4],
            'utc_offset': result[5]}

        return telescope

    #--------------------------------------------------------------------------
    def iter_observatories(self):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT * FROM Observatories
                """
            results = self._query(connection, query).fetchall()
            n = len(results)

        for i, result in enumerate(results):
            telescope = {
                'telescope_id': result[0],
                'name': result[1],
                'lat': result[2] * u.rad,
                'lon': result[3] * u.rad,
                'height': result[4],
                'utc_offset': result[5] * u.h}

            yield i, n, telescope

    #--------------------------------------------------------------------------
    def add_fields(self, fields, observatory, active=True):
        """TODO

        Note: Adding multiple fields at a time might be faster. However,
        finding the optimal number requires test and we do not add fields
        regularly. Therefore, a simple insertion loop is used.
        """

        observatory_id = self._get_observatory_id(observatory)
        active = bool(active)
        n_fields = len(fields)

        with SQLiteConnection(self.db_file) as connection:

            # iterate through fields:
            for i, field in enumerate(fields.fields):
                print(
                    f'\rAdding field {i} of {n_fields} ' \
                    f'({i*100./n_fields:.1f}%)..',
                    end='')

                fov = field.fov.rad
                center_ra = field.center_coord.ra.rad
                center_dec = field.center_coord.dec.rad
                tilt = field.tilt.rad

                query = """\
                    INSERT INTO Fields (
                        fov, center_ra, center_dec, tilt, observatory_id,
                        active)
                    VALUES ('{0}', {1}, {2}, {3}, {4}, {5});
                    """.format(
                        fov, center_ra, center_dec, tilt,
                        observatory_id, active)
                self._query(connection, query, commit=True)

            print(f'\r{n_fields} fields added to database.                   ')

    #--------------------------------------------------------------------------
    def get_fields(
            self, observatory=None, observed=None, pending=None, active=True):
        """TODO"""

        # set query condition for observed or not:
        if observed is None:
            condition_observed = ""
        elif observed:
            condition_observed =  " AND nobs_done > 0"
        else:
            condition_observed = " AND nobs_done = 0"

        # set query condition for pending observation or not:
        if pending is None:
            condition_pending = ""
        elif pending:
            condition_pending =  " AND nobs_pending > 0"
        else:
            condition_pending = " AND nobs_pending = 0"

        # set query condition for observatory:
        if observatory:
            condition_observatory = " AND name = '{0}'".format(observatory)
        else:
            condition_observatory = ""

        # query data base:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT f.field_id, f.fov, f.center_ra, f.center_dec,
                    f.tilt, o.name observatory, f.active,
                    f.jd_next_obs_window, p.nobs_done,
                    p.nobs_tot - p.nobs_done AS nobs_pending
                FROM Fields AS f
                LEFT JOIN Observatories AS o
                    ON f.observatory_id = o.observatory_id
                LEFT JOIN (
                	SELECT field_id, SUM(Done) nobs_done, COUNT(*) nobs_tot
                	FROM Observations
                	GROUP BY field_id
                	) AS p
                ON f.field_id = p.field_id
                WHERE (active = {0} {1} {2} {3});
                """.format(
                        active, condition_observatory, condition_observed,
                        condition_pending)
            result = self._query(connection, query).fetchall()

        return result

    #--------------------------------------------------------------------------
    def iter_fields(
            self, observatory=None, observed=None, pending=None, active=True):
        """TODO"""

        fields = self.get_fields(
                observatory=observatory, observed=observed, pending=pending,
                active=active)
        n = len(fields)

        for i, field in enumerate(fields):
            yield i, n, field

    #--------------------------------------------------------------------------
    def add_constraints(self, observatory, twilight, constraints=()):
        """TODO"""

        # add parameter set:
        parameter_set_id = self._add_parameter_set(observatory)

        # add twilight constraint:
        self._add_twilight(twilight, parameter_set_id)

        # add constraints:
        for constraint in constraints:
            self._add_constraint(constraint, parameter_set_id)

    #--------------------------------------------------------------------------
    def get_constraints(self, observatory):
        """TODO"""

        observatory_id = self._get_observatory_id(observatory)
        parameter_set_id = self._get_parameter_set_id(observatory_id)

        # no parameter set exists:
        if parameter_set_id == -1:
            raise NotInDatabase(
                "No active parameter set stored for observatory "
                f"'{observatory}'.")

        # query constraints and parameter values:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT c.constraint_name, pn.parameter_name, p.value, p.svalue
                FROM Parameters p
                LEFT JOIN ParameterNames pn
                    ON p.parameter_name_id = pn.parameter_name_id
                LEFT JOIN Constraints c
                    ON p.constraint_id = c.constraint_id
                WHERE p.parameter_set_id = {0}
                """.format(parameter_set_id)
            result = self._query(connection, query).fetchall()

        # parse to dictionary:
        constraints = {}

        for r in result:
            constraint_name = r[0]
            param_name = r[1]
            value = r[2]
            svalue = r[3]

            if constraint_name not in constraints.keys():
                constraints[constraint_name] = {}

            if value is None:
                constraints[constraint_name][param_name] = svalue
            else:
                constraints[constraint_name][param_name] = value

        return constraints

    #--------------------------------------------------------------------------
    def add_obs_window(self, field_id, date_start, date_stop, active=True):
        """TODO"""

        duration = (date_stop - date_start).value

        with SQLiteConnection(self.db_file) as connection:
            # add observatory to database:
            query = """\
                INSERT INTO ObsWindows (
                    field_id, date_start, date_stop, duration, active)
                VALUES ({0}, '{1}', '{2}', {3}, {4});
                """.format(
                    field_id, date_start.iso, date_stop.iso, duration, active)
            self._query(connection, query, commit=True)

    #--------------------------------------------------------------------------
    def update_next_obs_window(self, field_id, jd):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            # add observatory to database:
            query = """\
                UPDATE Fields
                SET jd_next_obs_window='{0}'
                WHERE field_id={1};
                """.format(jd, field_id)
            self._query(connection, query, commit=True)

    #--------------------------------------------------------------------------
    def get_obs_windows_from_to(self, field_id, date_start, date_stop):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            query = """
            SELECT * FROM ObsWindows
            WHERE (field_id={0} AND
                   date_start>'{1}' AND
                   date_stop<'{2}')
            """.format(field_id, date_start.iso, date_stop.iso)
            obs_windows = self._query(connection, query).fetchall()

        return obs_windows

    #--------------------------------------------------------------------------
    def get_obs_windows_by_datetime(self, field_id, datetime):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            query = """
            SELECT * FROM ObsWindows
            WHERE (field_id={0} AND
                   date_start<'{1}' AND
                   date_stop>'{1}')
            """.format(field_id, datetime.iso)
            obs_windows = self._query(connection, query).fetchall()

        return obs_windows

    #--------------------------------------------------------------------------
    def get_obs_window_durations(self, field_id, date_start, date_stop):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            query = """
            SELECT duration FROM ObsWindows
            WHERE (field_id={0} AND
                   date_start>'{1}' AND
                   date_stop<'{2}')
            """.format(field_id, date_start.iso, date_stop.iso)
            durations = self._query(connection, query).fetchall()

        return durations

    #--------------------------------------------------------------------------
    def _add_filter(self, filter_name):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                INSERT INTO Filters (filter)
                VALUES ('{0}')
                """.format(filter_name)
            self._query(connection, query, commit=True)
            last_insert_id = self._last_insert_id(connection)

        return last_insert_id

    #--------------------------------------------------------------------------
    def get_filter_id(self, filter_name):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT filter_id, filter
                FROM Filters
                WHERE filter='{0}';
                """.format(filter_name)
            results = self._query(connection, query).fetchall()

        if len(results) == 0:
            userin = input(
                    f"Filter '{filter_name} does not exist. Add it to data " \
                    "base? (y/n)")

            if userin.lower() in ['y', 'yes', 'make it so!']:
                filter_id = self._add_filter(filter_name)
            else:
                filter_id = False

        else:
            filter_id = results[0][0]

        return filter_id

    #--------------------------------------------------------------------------
    def _add_observation(self, field_id, exposure, repetitions, filter_id):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                INSERT INTO Observations (
                    field_id, exposure, repetitions, filter_id)
                VALUES ({0}, {1}, {2}, {3});
                """.format(field_id, exposure, repetitions, filter_id)
            self._query(connection, query, commit=True)

    #--------------------------------------------------------------------------
    def get_observations(self, field_id, exposure, repetitions, filter_id):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT *
                FROM Observations
                WHERE (
                    field_id={0} AND exposure={1} AND repetitions={2}
                    AND filter_id={3});
                """.format(field_id, exposure, repetitions, filter_id)
            results = self._query(connection, query).fetchall()

        return results

    #--------------------------------------------------------------------------
    def add_observations(self, field_id, exposure, repetitions, filter_name):
        """TODO"""

        # prepare field_ids:
        if isinstance(field_id, int):
            field_id = [field_id]
        elif isinstance(field_id, list):
            pass
        else:
            raise ValueError("'field_id' needs to be int or list of int.")

        n_fields = len(field_id)

        # prepare exposures:
        if type(exposure) in [float, int]:
            exposure = float(exposure)
            exposure = [exposure for __ in range(n_fields)]
        elif isinstance(exposure, list):
            if len(exposure) != n_fields:
                raise ValueError(
                        "'exposures' list need to be of the same length as " \
                        "'field_id'.")
        else:
            raise ValueError("'exposure' needs to be float or list of floats.")

        # prepare repetitions:
        if isinstance(repetitions, int):
            repetitions = [repetitions for __ in range(n_fields)]
        elif isinstance(repetitions, list):
            if len(repetitions) != n_fields:
                raise ValueError(
                        "'repetitions' list need to be of the same length " \
                        "'as field_id'.")
        else:
            raise ValueError("'repetitions' needs to be int or list of int.")

        # prepare filters:
        if isinstance(filter_name, str):
            filter_name = [filter_name for __ in range(n_fields)]
        elif isinstance(filter_name, list):
            if len(filter_name) != n_fields:
                raise ValueError(
                        "'filter_name' list need to be of the same length " \
                        "'as field_id'.")
        else:
            raise ValueError("'filter_name' needs to be str or list of str.")

        check_existence = True
        skip_existing = False
        filter_ids = {}
        data = []

        # prepare entries for adding:
        for field, exp, rep, filt in zip(
                field_id, exposure, repetitions, filter_name):
            # check if filter exists:
            if filt not in filter_ids.keys():
                filter_id = self.get_filter_id(filt)

                # stop, if the filter did not exist and was not added:
                if filter_id is False:
                    print("Filter was not added to database. No " \
                          "observations are added either.")
                    return False

                filter_ids[filt] = filter_id

            # check if observation entry exists:
            if check_existence:
                observations = self.get_observations(
                        field, exp, rep, filter_ids[filt])
                n_obs = len(observations)
                n_done = len([1 for obs in observations if obs[6]])

                if n_obs > n_done and skip_existing:
                    continue
                elif n_obs:
                    userin = input(
                            f"{n_obs} observation(s) with the same " \
                            "parameters already exist in data base. " \
                            f"{n_done} out of those are finished. Add new " \
                            "observation anyway? (y/n, 'ALL' to add all " \
                            "following without asking, or 'NONE' to skip " \
                            "all existing observations that have not been " \
                            "finished).")

                    if userin.lower in ['y', 'yes', 'make it so!']:
                        pass
                    elif userin == 'ALL':
                        check_existence = False
                    elif userin == 'NONE':
                        skip_existing = True
                        continue
                    else:
                        continue

            # add to data:
            data.append((field, exp, rep, filter_ids[filt], False, False))

        # add to data base:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                INSERT INTO Observations (
                    field_id, exposure, repetitions, filter_id, scheduled,
                    done)
                VALUES (?, ?, ?, ?, ?, ?);
                """
            self._query(connection, query, many=data, commit=True)

        n_obs = len(data)
        print(f"{n_obs} observations added to data base.")

    #--------------------------------------------------------------------------
    def _check_observation_status(self, observation_id):
        """TODO"""

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT scheduled, done
                FROM Observations
                WHERE observation_id = {0}
                """.format(observation_id)
            results = self._query(connection, query).fetchall()

        if results:
            exists = True
            scheduled = bool(results[0][0])
            done = bool(results[0][1])
        else:
            exists, scheduled, done = False, False, False

        return exists, scheduled, done

    #--------------------------------------------------------------------------
    def _set_observed_by_id(self, observation_id, date=None):
        """TODO"""

        # check input:
        if isinstance(observation_id, int):
            observation_ids = [observation_id]
        elif isinstance(observation_id, list):
            observation_ids = observation_id
        else:
            raise ValueError("'observation_id' must be int or list of int.")

        n_observations = len(observation_ids)

        if date is None:
            date = Time.now()
            dates = [date] * n_observations
        elif isinstance(date, Time):
            dates = [date] * n_observations
        elif isinstance(date, list):
            dates = date
        else:
            raise ValueError(
                    "'date' must be astropy.Time or list of astropy.Time.")

        count_set = 0

        # iterate through observation IDs and dates:
        for observation_id, date in zip(observation_ids, dates):
            # check if it exists and if it was observed already:
            exists, scheduled, done = self._check_observation_status(
                    observation_id)

            # does not exist - raise error:
            if not exists:
                warnings.warn(
                        f"Observation with ID {observation_id} does not exist "
                        "in database. Skipped.")

            # already marked as done - warn:
            elif done:
                warnings.warn(
                    f"Observation {observation_id} is already marked as done. "
                    "Skipped.")

            # set as observed:
            else:
                with SQLiteConnection(self.db_file) as connection:
                    query = """\
                        UPDATE Observations
                        SET scheduled = 0, done = 1, date = '{0}'
                        WHERE observation_id = {1};
                        """.format(date, observation_id)
                    self._query(connection, query, commit=True)
                    count_set += 1

        print(f"{count_set} (out of {n_observations}) observations set as "
              "done.")

    #--------------------------------------------------------------------------
    def _set_observed_by_params(
            self, field_id, exposure, repetitions, filter_name, date=None):
        """TODO"""

        # check input:
        if isinstance(field_id, int):
            field_ids = [field_id]
        elif isinstance(field_id, list):
            field_ids = field_id
        else:
            raise ValueError("'field_id' must be int or list of int.")

        n_fields = len(field_ids)

        if exposure is None or isinstance(exposure, float):
            exposure = [exposure] * n_fields
        elif isinstance(exposure, list):
            pass
        else:
            raise ValueError("'exposure' must be float or list of float.")

        if repetitions is None or isinstance(repetitions, int):
            repetitions = [repetitions] * n_fields
        elif isinstance(repetitions, list):
            pass
        else:
            raise ValueError("'repetitions' must be int or list of int.")

        if filter_name is None or isinstance(filter_name, str):
            filter_name = [filter_name] * n_fields
        elif isinstance(filter_name, list):
            pass
        else:
            raise ValueError("'filter_name' must be str or list of str.")

        observation_ids = []

        # iterate through fields and additional information:
        for field_id, exp, rep, filt in zip(
                field_ids, exposure, repetitions, filter_name):
            # build query:
            query = """\
                SELECT observation_id, field_id, exposure, repetitions, filter
                FROM Observations AS o
                LEFT JOIN Filters AS f
                ON o.filter_id = f.filter_id
                WHERE field_id = {0}
                """.format(field_id)

            if exp is not None:
                query = """\
                    {0} AND exposure = {1}
                    """.format(query, exp)

            if rep is not None:
                query = """\
                    {0} AND repetitions = {1}
                    """.format(query, rep)

            if filt is not None:
                query = """\
                    {0} AND filter = '{1}'
                    """.format(query, filt)

            # query observation ID:
            with SQLiteConnection(self.db_file) as connection:
                results = self._query(connection, query).fetchall()

            # multiple observation found - user input required:
            if len(results) > 1:
                info = "Multiple observations matching the criteria were " \
                    "found. Type 'A' to mark all as observed or select a " \
                    "specific observation ID:\n" \
                    "Obs ID field ID      exp      rep   filter\n" \
                    "------ -------- -------- -------- --------"

                for i, result in enumerate(results):
                    info = f"{info}\n{i:6d} {result[1]:8d} {result[2]:8.1f} " \
                        f"{result[3]:8d} {result[4]:>8}"

                userin = input(f"{info}\nSelection: ")

                if userin == 'A':
                    for result in results:
                        observation_ids.append(result[0])
                else:
                    try:
                        userin = int(userin)
                        observation_ids.append(results[userin][0])
                    except (ValueError, IndexError):
                        raise ValueError(
                                "Select either 'A' or one of the allowed IDs" \
                                " (int).")

            # one observation found - save observation ID:
            elif len(results) == 1:
                observation_ids.append(results[0][0])

            # no observation found - warn:
            else:
                warn_info = "No observation found with the following " \
                        f"specifications: field ID: {field_id}"
                if exp is not None:
                    warn_info = f"{warn_info}, exposure: {exp}"
                if rep is not None:
                    warn_info = f"{warn_info}, repetitions: {rep}"
                if filt is not None:
                    warn_info = f"{warn_info}, filter: {filt}"
                warn_info = f"{warn_info}. Skipped."
                warnings.warn(warn_info)

        # set observed via observation IDs:
        self._set_observed_by_id(observation_ids, date=date)

    #--------------------------------------------------------------------------
    def set_observed(
            self, observation_id=None, field_id=None, exposure=None,
            repetitions=None, filter_name=None, date=None):
        """TODO"""

        if observation_id is None:
            self._set_observed_by_params(
                    field_id, exposure, repetitions, filter_name, date=date)
        else:
            self._set_observed_by_id(observation_id, date=date)


#==============================================================================
