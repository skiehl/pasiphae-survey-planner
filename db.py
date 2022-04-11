# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Sky fields for the pasiphae survey.
"""

from astropy import units as u
import os
import sqlite3

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
    def _connect(self):
        """TODO"""

        self.connection

    #--------------------------------------------------------------------------
    def _query(self, connection, query, commit=False):
        """TODO"""

        cursor = connection.cursor()
        result = cursor.execute(query)

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
                    date date)
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
    def get_fields(self, observatory=None, active=True):
        """TODO"""

        # TODO: add argument 'observed' to query fields that have or have not
        # been observed

        with SQLiteConnection(self.db_file) as connection:

            # no specified observatory:
            if observatory is None:
                query = """\
                    SELECT field_id, fov, center_ra, center_dec, tilt, name,
                        active, jd_next_obs_window
                    FROM Fields AS f
                    LEFT JOIN Observatories AS o
                        ON f.observatory_id = o.observatory_id
                    WHERE active = {0}
                    """.format(active)
                result = self._query(connection, query).fetchall()

            # specified observatory:
            else:
                query = """\
                    SELECT field_id, fov, center_ra, center_dec, tilt, name,
                        active, jd_next_obs_window
                    FROM Fields AS f
                    LEFT JOIN Observatories AS o
                        ON f.observatory_id = o.observatory_id
                    WHERE (active = {0}
                           AND name = '{1}');
                    """.format(active, observatory)
                result = self._query(connection, query).fetchall()

        return result

    #--------------------------------------------------------------------------
    def iter_fields(self, observatory=None, active=True):
        """TODO"""

        fields = self.get_fields(observatory=observatory, active=active)
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

        duration = (date_stop - date_start).value / 24.

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

#==============================================================================
