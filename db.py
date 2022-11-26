# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Database interface for the Pasiphae survey planner.
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
        """Create SQLiteConnection instance.

        Parameters
        ----------
        db_file : str
            SQLite3 database file name.

        Returns
        -------
        None
        """

        self.db_file = db_file

    def __enter__(self):
        """Open database connection and return it.

        Returns
        -------
        sqlite3.Connection
            Open database connection.
        """

        self.connection = sqlite3.connect(self.db_file)

        return self.connection

    def __exit__(self, type, value, traceback):
        """Close database connection.

        Returns
        -------
        None
        """

        self.connection.close()

#==============================================================================

class DBConnectorSQLite:
    """SQLite database connector."""

    #--------------------------------------------------------------------------
    def __init__(self, db_file):
        """Create DBConnectorSQLite instance.

        Parameters
        ----------
        db_file : str
            SQLite3 database file name.

        Returns
        -------
        None
        """

        self.db_file = db_file

    #--------------------------------------------------------------------------
    def _query(self, connection, query, many=False, commit=False):
        """Query the database.

        Parameters
        ----------
        connection : sqlite3.Connection
            The database connection.
        query : str
            SQL query.
        many : list, optional
            List of data to add in a single commit. The default is False.
        commit : bool, optional
            Set True to commit to database. The default is False.

        Returns
        -------
        result : sqlite3.Cursor
            Query results.
        """

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
        """Get the observatory ID by name.

        Parameters
        ----------
        name : std
            Observatory name.

        Raises
        ------
        NotInDatabase
            Raised if observatory name does not exist in database.

        Returns
        -------
        observatory_id : int
            The ID of the observatory in the database.
        """

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
        """Get the parameter set ID associated with an observatory ID.

        Parameters
        ----------
        observatory_id : int
            Observatory ID.

        Returns
        -------
        parameter_set_id : int
            Parameter set ID.
        """

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
        """Set a parameter set to inactive.

        Parameters
        ----------
        parameter_set_id : int
            Parameter set ID.

        Returns
        -------
        None
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
        """Get the last inserted ID.

        Parameters
        ----------
        connection : sqlite3.Connection
            The database connection.

        Returns
        -------
        last_insert_id : int
            The last inserted ID.
        """

        query = """SELECT last_insert_rowid()"""
        result = self._query(connection, query).fetchone()
        last_insert_id = result[0]

        return last_insert_id

    #--------------------------------------------------------------------------
    def _get_constraint_id(self, constraint_name):
        """Get the constraint ID by constraint name.

        Parameters
        ----------
        constraint_name : str
            Constraint name.

        Raises
        ------
        NotInDatabase
            Raised if constraint name does not exist in database.

        Returns
        -------
        constraint_id : int
            Constraint ID.
        """

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
        """Get the parameter name ID by parameter name.

        Parameters
        ----------
        parameter_name : str
            Parameter name.

        Returns
        -------
        parameter_name_id : int
            Parameter name ID.
        """

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
        """Add a parameter to the database.

        Parameters
        ----------
        constraint_id : int
            ID of the constraint associated with the parameter.
        parameter_set_id : int
            ID of the associated parameter set.
        parameter_name : str
            Parameter name.
        value : float, optional
            The numerical value to be stored. The default is None.
        svalue : str, optional
            The string value to be stored. The default is None.

        Returns
        -------
        None
        """

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
        """Add a parameter set to the database.

        Parameters
        ----------
        observatory : str
            Name of the associated observatory.

        Returns
        -------
        parameter_set_id : int
            ID of the newly added parameter set.
        """

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
        """Add the twilight condition.

        Parameters
        ----------
        twilight : float or str
            If str, must be 'astronomical' (-18 deg), 'nautical' (-12 deg),
            'civil' (-6 deg), or 'sunset' (0 deg). Use float otherwise.
        parameter_set_id : int
            ID of the associated parameter set.

        Raises
        ------
        ValueError
            Raised if value of 'twilight' is neither float or one of the four
            allowed strings.

        Returns
        -------
        None
        """

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
        """Add constraint to database.

        Parameters
        ----------
        constraint : constraints.Constraint
            An observational constraint.
        parameter_set_id : TYPE
            ID of the associated parameter set.

        Raises
        ------
        ValueError
            Raised if the constraint returns anything but string, float, or int
            as parameters.

        Returns
        -------
        None
        """

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
    def _add_filter(self, filter_name):
        """Add a filter to the database.

        Parameters
        ----------
        filter_name : str
            Filter name. Must be a unique identifier in the database.

        Returns
        -------
        last_insert_id : int
            The last inserted ID.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                INSERT INTO Filters (filter)
                VALUES ('{0}')
                """.format(filter_name)
            self._query(connection, query, commit=True)
            last_insert_id = self._last_insert_id(connection)

        return last_insert_id

    #--------------------------------------------------------------------------
    def _add_observation(self, field_id, exposure, repetitions, filter_id):
        """Add observation to database.

        Parameters
        ----------
        field_id : int
            ID of the associated field.
        exposure : float
            Exposure time in seconds.
        repetitions : int
            Number of repetitions.
        filter_id : int
            Filter ID.

        Returns
        -------
        None
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                INSERT INTO Observations (
                    field_id, exposure, repetitions, filter_id)
                VALUES ({0}, {1}, {2}, {3});
                """.format(field_id, exposure, repetitions, filter_id)
            self._query(connection, query, commit=True)

    #--------------------------------------------------------------------------
    def create_db(self):
        """Create sqlite3 database.

        Returns
        -------
        None
        """

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
                    observatory_id integer
                        REFERENCES Observatories (observatory_id),
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
                    observatory_id integer
                        REFERENCES Observatories (observatory_id),
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
                    constraint_id integer
                        REFERENCES Constraints (constraint_id),
                    parameter_set_id integer
                        REFERENCES ParameterSets (parameter_set_id),
                    parameter_name_id integer
                        REFERENCES ParameterNames (parameter_name_id),
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
                    field_id integer
                        REFERENCES Fields (field_id),
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
                    field_id integer
                        REFERENCES Fields (field_id),
                    exposure float,
                    repetitions int,
                    filter_id int
                        REFERENCES Filters (filter_id),
                    scheduled bool,
                    done bool,
                    date date)
                """
            self._query(connection, query, commit=True)
            print("Table 'Observations' created.")

            # create Guidestars table:
            query = """\
                CREATE TABLE Guidestars(
                    guidestar_id integer PRIMARY KEY,
                    field_id integer
                        REFERENCES Fields (field_id),
                    ra float,
                    dec int,
                    active bool)
                """
            self._query(connection, query, commit=True)
            print("Table 'Guidestars' created.")

            # create Filters table:
            query = """\
                CREATE TABLE Filters(
                    filter_id integer PRIMARY KEY,
                    filter char(10))
                """
            self._query(connection, query, commit=True)
            print("Table 'Filters' created.")

            # define constraints:
            query = """\
                INSERT INTO Constraints (constraint_name)
                VALUES
                    ('Twilight'),
                    ('ElevationLimit'),
                    ('AirmassLimit'),
                    ('MoonDistance'),
                    ('MoonPolarization'),
                    ('PolyHADecLimit')
                """
            self._query(connection, query, commit=True)
            print("Constraints added to table 'Constraints'.")

    #--------------------------------------------------------------------------
    def add_observatory(self, name, lat, lon, height, utc_offset):
        """Add observatory to database.

        Parameters
        ----------
        name : str
            Observatory name. Must be a unique identifier in the database.
        lat : float
            Observatory latitude in radians.
        lon : float
            Observatory longitude in radians.
        height : float
            Observatory height in meters.
        utc_offset : int
            Observatory UTC offset (daylight saving time).

        Returns
        -------
        None
        """

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
        """Get observatory from database.

        Parameters
        ----------
        name : str
            Observatory name.

        Returns
        -------
        telescope : surveyplanner.Telescope
            The telescope with parameters as stored in the database.
        """

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
        """Iterate through observatories stored in database.

        Yields
        ------
        i : int
            Iteratively increasing counter.
        n : int
            Number of observatories stored in the database.
        telescope : surveyplanner.telescope
            The telescope with parameters as stored in the database.
        """

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
        """Add fields to the database.

        Parameters
        ----------
        fields : skyfields.Fields
            The fields to add.
        observatory : str
            Name of the observatory associated with the fields.
        active : bool, optional
            If True, fields are added as active, and as inactive otherwise. The
            default is True.

        Returns
        -------
        None

        Notes
        -----
        Adding multiple fields at a time might be faster. However, finding the
        optimal number requires test and we do not add fields regularly.
        Therefore, a simple insertion loop is used.
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
        """Get fields from the database, given various selection criteria.

        Parameters
        ----------
        observatory : str, optional
            Observatory name. If set, only query fields associated with this
            observatory. If None, query fields for all observatories. The
            default is None.
        observed : bool, optional
            If True, only query fields that have been observed at least once.
            If False, only query fields that have never been observed. In None,
            query fields independend of the observation status. The default is
            None.
        pending : bool, optional
            If True, only query fields that have pending observations
            associated. If False, only query fields that have no pending
            observations associated. If None, query fields independent of
            whether observations are pending or not. The default is None.
        active : bool, optional
            If True, only query active fields. If False, only query inactive
            fields. If None, query fields independent of whether they are
            active or not. The default is True.

        Returns
        -------
        result : list of tuples
            List of the queried fields. Each tuple contains the field
            parameters.
        """

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
        """Query fields from the database, given various selection criteria,
        and iterate through the results.

        Parameters
        ----------
        observatory : str, optional
            Observatory name. If set, only query fields associated with this
            observatory. If None, query fields for all observatories. The
            default is None.
        observed : bool, optional
            If True, only query fields that have been observed at least once.
            If False, only query fields that have never been observed. In None,
            query fields independend of the observation status. The default is
            None.
        pending : bool, optional
            If True, only query fields that have pending observations
            associated. If False, only query fields that have no pending
            observations associated. If None, query fields independent of
            whether observations are pending or not. The default is None.
        active : bool, optional
            If True, only query active fields. If False, only query inactive
            fields. If None, query fields independent of whether they are
            active or not. The default is True.

        Yields
        ------
        i : int
            Iteratively increasing counter.
        n : int
            Total number of queried fields.
        field : tuple
            The tuple contains the field parameters.

        Notes
        -----
        This method first uses the get_fields() method to get the total number
        of fields. There is no memory advantage in using this iterator over the
        get_fields() method.
        """

        fields = self.get_fields(
                observatory=observatory, observed=observed, pending=pending,
                active=active)
        n = len(fields)

        for i, field in enumerate(fields):
            yield i, n, field

    #--------------------------------------------------------------------------
    def get_field_by_id(self, field_id):
        """Query field from database by ID.

        Parameters
        ----------
        field_id : int
            Field ID.

        Returns
        -------
        result : list of tuple
            The list contains only one tuple. The tuple contains the field
            parameters.
        """

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
                    WHERE f.field_id = {0} AND p.field_id = {0};
                    """.format(field_id)
            result = self._query(connection, query).fetchall()

        return result

    #--------------------------------------------------------------------------
    def add_constraints(self, observatory, twilight, constraints=()):
        """Add constraints to database.

        Parameters
        ----------
        observatory : str
            Name of the observatory that the constraints are associated with.
        twilight : float or str
            If str, must be 'astronomical' (-18 deg), 'nautical' (-12 deg),
            'civil' (-6 deg), or 'sunset' (0 deg). Use float otherwise.
        constraints : list or tuple of constraints.Constraint, optional
            The constraints to be added to the database for the specified
            observatory. The default is ().

        Returns
        -------
        None
        """

        # add parameter set:
        parameter_set_id = self._add_parameter_set(observatory)

        # add twilight constraint:
        self._add_twilight(twilight, parameter_set_id)

        # add constraints:
        for constraint in constraints:
            self._add_constraint(constraint, parameter_set_id)

    #--------------------------------------------------------------------------
    def get_constraints(self, observatory):
        """Query constraints associated with a specified observatory from the
        database

        Parameters
        ----------
        observatory : str
            Name of the observatory.

        Raises
        ------
        NotInDatabase
            Raised if no parameter set is stored for the specified observatory.

        Returns
        -------
        constraints : dict of dict
            Dictionary of the constraints. The keys are the constraint names.
            The values are dictionaries that contain the constraint parameter
            names as keys and associated values.
        """

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
        """Add observation window for a specific field to database.

        Parameters
        ----------
        field_id : int
            ID of the field that the observation window is associated with.
        date_start : astropy.time.Time
            Start date and time of the observing window.
        date_stop : astropy.time.Time
            Stop date and time of the observing window.
        active : bool, optional
            If True, the observing windows is added as active, as inactive
            otherwise. The default is True.

        Returns
        -------
        None
        """

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
        """Update a field's database entry for the next observing window.

        Parameters
        ----------
        field_id : int
            ID of the associated field.
        jd : float
            Julian date of the next day for which the next observing window
            needs to be calculated for the specified field.

        Returns
        -------
        None
        """

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
        """Query observing window from the database for a specified field
        between a start and a stop date.

        Parameters
        ----------
        field_id : int
            ID of the field whose observing windows are queried.
        date_start : astropy.time.Time
            Query observing windows later than this time.
        date_stop : astropy.time.Time
            Query observing windows earlier than this time.

        Returns
        -------
        obs_windows : list of tuples
            List of the queried observing windows. Each tuple contains the
            observing window ID, the field ID, start and stop date, duration,
            and active status.
        """

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
        """Query observing window from the database for a specified field
        for a specific date and time.


        Parameters
        ----------
        field_id : int
            ID of the field whose observing windows are queried.
        datetime : astropy.time.Time
            Query the observing window that includes the specified datetime.

        Returns
        -------
        obs_window : list of tuples
            List of the queried observing windows. The list contains either no
            tuple, if no observing window includes the specified datetime, or
            one tuple for the resulting observing window. The tuple contains
            the observing window ID, the field ID, start and stop date,
            duration, and active status.
        """

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
        """Query the durations of observing window from the database for a
        specified field between a start and a stop date.

        Parameters
        ----------
        field_id : int
            ID of the field whose observing windows are queried.
        date_start : astropy.time.Time
            Query observing windows later than this time.
        date_stop : astropy.time.Time
            Query observing windows earlier than this time.

        Returns
        -------
        durations : list
            The durations of the queried observing windows in days.
        """

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
    def get_filter_id(self, filter_name):
        """Get the filter ID by the filter name.

        Parameters
        ----------
        filter_name : str
            Filter name.

        Returns
        -------
        filter_id : int
            Filter ID.

        Notes
        -----
        If the filter name does not exist in the database, the user is asked
        whether or not to add it.
        """

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
    def get_observations(self, field_id, exposure, repetitions, filter_id):
        """Query an observation from the database.

        Parameters
        ----------
        field_id : int
            ID of the associated field.
        exposure : float
            Exposure time in seconds.
        repetitions : int
            Number of repetitions.
        filter_id : int
            Filter ID.

        Returns
        -------
        results : list of tuples
            The list is empty if no observation was found. Otherwise, the list
            contains one tuple. The tuple contains the observation ID, field
            ID, exposure time in seconds, number of repetitions, filter_id,
            its scheduling status, its observation status, the datetime of
            the observation if it was finished.
        """

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
        """Add observation to database.

        Parameters
        ----------
        field_id : int
            ID of the associated field.
        exposure : float
            Exposure time in seconds.
        repetitions : int
            Number of repetitions.
        filter_name : str
            Filter name.

        Returns
        -------
        None
        """

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
        """Get the status of an observation.

        Parameters
        ----------
        observation_id : int
            ID of the observation.

        Returns
        -------
        exists : bool
            True, if observation with specified ID exists. False, otherwise.
        scheduled : bool
            True, if observation is marked as scheduled. False, otherwise.
        done : bool
            True, if observation is marked as finished. False, otherwise.
        """

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
        """Set an observation as observed.

        Parameters
        ----------
        observation_id : int or list of ints
            Observation ID(s) that should be marked as observed.
        date : astropy.time.Time or list therof, optional
            Date and time of the observation. If not provided, the current
            time when the observation is marked as observed is stored. The
            default is None.

        Raises
        ------
        ValueError
            Raised if 'observation_id' is not an int or list.
        ValueError
            Raised if 'date' is not an astropy.time.Time instance, list, or
            None.
        ValueError
            If the number of IDs and dates does not match.

        Returns
        -------
        None
        """

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

        if len(observation_ids) != len(dates):
            raise ValueError(
                    "The same number of IDs and dates must be provided or "
                    "a single ID and/or date.")

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
        """Set a field's observations identified by the observations's
        parameters as observed.

        Parameters
        ----------
        field_id : int or list of ints
            Field ID.
        exposure : float or list of floats
            Exposure time in seconds.
        repetitions : int or list of ints
            Number of repetitions.
        filter_name : str or list of str
            Filter name.
        date : astropy.time.Time or list therof, optional
            Date and time of the observation. If not provided, the current
            time when the observation is marked as observed is stored. The
            default is None.

        Raises
        ------
        ValueError
            Raised if 'field_id' is not int or list.
        ValueError
            Raised if 'exposure' is not float or list.
        ValueError
            Raised if 'exposure' is list and its length does not match the
            number of field IDs.
        ValueError
            Raised if 'repetitions' is not int or list.
        ValueError
            Raised if 'repetitions' is list and its length does not match the
            number of field IDs.
        ValueError
            Raised if 'filter_name' is not str or list.
        ValueError
            Raised if 'filter_names' is list and its length does not match the
            number of field IDs.

        Returns
        -------
        None
        """

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
            if len(exposure) != len(field_ids):
                raise ValueError(
                        "Number of field IDs and exposure entries does not "
                        "match.")
        else:
            raise ValueError("'exposure' must be float or list of float.")

        if repetitions is None or isinstance(repetitions, int):
            repetitions = [repetitions] * n_fields
        elif isinstance(repetitions, list):
            if len(exposure) != len(field_ids):
                raise ValueError(
                        "Number of field IDs and repetion entries does not "
                        "match.")
        else:
            raise ValueError("'repetitions' must be int or list of int.")

        if filter_name is None or isinstance(filter_name, str):
            filter_name = [filter_name] * n_fields
        elif isinstance(filter_name, list):
            if len(exposure) != len(field_ids):
                raise ValueError(
                        "Number of field IDs and filter entries does not "
                        "match.")
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
                        # TODO : BUG-fix: if list of dates is provided for all observations, the list of IDs will not match the list of dates and this will crash the next called method. I need to change the list of dates in this case as well.

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
        """Mark an observation as finished. Select observation either by its ID
        or by its parameters.

        Parameters
        ----------
        observation_id : int or list of ints or None
            Observation ID(s) that should be marked as observed. If None, the
            other arguments must be set to idenfy the observation(s). The
            default is None.
        field_id : int or list of ints
            Field ID. The default is None.
        exposure : float or list of floats
            Exposure time in seconds. The default is None.
        repetitions : int or list of ints
            Number of repetitions. The default is None.
        filter_name : str or list of str
            Filter name. The default is None.
        date : astropy.time.Time or list therof, optional
            Date and time of the observation. If not provided, the current
            time when the observation is marked as observed is stored. The
            default is None.

        Returns
        -------
        None
        """

        if observation_id is None:
            self._set_observed_by_params(
                    field_id, exposure, repetitions, filter_name, date=date)
        else:
            self._set_observed_by_id(observation_id, date=date)


#==============================================================================
