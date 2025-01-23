#!/usr/bin/env python3
"""Database interface for the Pasiphae survey planner.
"""

from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from astropy import units as u
from math import ceil
import numpy as np
import os
from pandas import DataFrame
import sqlite3
from textwrap import dedent
import warnings

from fieldgrid import FieldGrid

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

    #--------------------------------------------------------------------------
    def __init__(self, db_file, wal_mode=True):
        """Create SQLiteConnection instance.

        Parameters
        ----------
        db_file : str
            SQLite3 database file name.
        wal_mode : bool
            If True, open SQLite connnection with "Write-Ahead Logging" (WAL)
            mode. Otherwise, uses default rollback journal. WAL mode allows for
            parallel reading and writing. The default is True.

        Returns
        -------
        None

        Notes
        -----
        See [1] for details about "Write-Ahead Logging" (WAL).

        References
        ----------
        [1] https://www.sqlite.org/wal.html
        """

        self.db_file = db_file
        self.wal_mode = wal_mode

    #--------------------------------------------------------------------------
    def __enter__(self):
        """Open database connection and return it.

        Returns
        -------
        sqlite3.Connection
            Open database connection.
        """

        self.connection = sqlite3.connect(self.db_file)

        if self.wal_mode:
            self.connection.execute('pragma journal_mode=wal')

        self.connection.row_factory = self._dict_factory

        return self.connection

    #--------------------------------------------------------------------------
    def __exit__(self, type, value, traceback):
        """Close database connection.

        Returns
        -------
        None
        """

        self.connection.close()

    #--------------------------------------------------------------------------
    def _dict_factory(self, cursor, row):
        """Return query results as dict.

        Parameters
        ----------
        cursor : sqlite3.Cursor
            SQLite3 cursor.
        row : tuple
            Tuple containing the query results from one row.

        Returns
        -------
        dict
            Query results as dict, where keys are the column names.

        Notes
        -----
        This method is used by `__enter__()`.
        """

        fields = [column[0] for column in cursor.description]

        return {key: value for key, value in zip(fields, row)}

#==============================================================================

class DBManager:
    """SQLite3 database manager."""

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
    def _db_exists(self, verbose=1):
        """Check if the database file exists.

        Parameters
        ----------
        verbose : int, optional
            If 0, print no notification. If not zero and database does not
            exist, print notification. The default is 1.

        Returns
        -------
        bool
            True, if file exists. False, otherwise.
        """
        exists = os.path.exists(self.db_file)

        if not exists and verbose:
            print(f'SQLite3 database {self.db_file} does not exist. '
                  'Use `DBCreator()` to set up a new database.')

        return exists

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
        last_insert_id = result['last_insert_rowid()']

        return last_insert_id

    #--------------------------------------------------------------------------
    def dbstatus(self, verbose=1):
        """Return count various data entries in the database.

        Parameters
        ----------
        verbose : int, optional
            If 0, no info is printed. Otherwise, print out the results. The
            default is 1.

        Returns
        -------
        results : dict
            Number of telescopes, constraints, fields, observations, and
            guidestars stored in the database.
        """

        # query stats:
        with SQLiteConnection(self.db_file) as connection:

            # get number of telescopes:
            query = """\
                SELECT count(telescope_id) AS n
                FROM Telescopes
                """
            result = self._query(connection, query).fetchone()
            n_telescopes = result['n']

            # get number of constraints associated with each telescope:
            query = """\
                SELECT IFNULL(n, 0) AS n
                FROM Telescopes AS t
                LEFT JOIN (
                	SELECT ps.telescope_id, COUNT(DISTINCT p.constraint_id) AS n
                	FROM Parameters AS p
                	LEFT JOIN ParameterSets AS ps
                	ON p.parameter_set_id = ps.parameter_set_id
                	WHERE ps.active = 1
                	GROUP BY ps.telescope_id
                ) AS p
                ON t.telescope_id = p.telescope_id
                """
            results = self._query(connection, query).fetchall()
            n_constraints = [item['n'] for item in results]

            # get number of fields associated with each telescope:
            query = """\
                SELECT IFNULL(f.n, 0) AS n
                FROM Telescopes AS t
                LEFT JOIN (
                	SELECT telescope_id, COUNT(field_id) AS n
                	FROM Fields
                	GROUP BY telescope_id
                	) AS f
                ON t.telescope_id = f.telescope_id
                """
            results = self._query(connection, query).fetchall()
            n_fields = [item['n'] for item in results]

            # get number of active observations:
            query = """\
                SELECT COUNT(observation_id) AS n
                FROM Observations
                WHERE active = 1
                """
            result = self._query(connection, query).fetchone()
            n_obs_tot = result['n']

            # get number of active, pending observations:
            query = """\
                SELECT COUNT(observation_id) AS n
                FROM Observations
                WHERE (active = 1 AND done = 0)
                """
            result = self._query(connection, query).fetchone()
            n_obs_pending = result['n']

            # get number of fields without active observations:
            query = """\
                SELECT COUNT(f.field_id) AS n
                FROM Fields AS f
                LEFT JOIN  (
                	SELECT *
                	FROM Observations
                	WHERE active = 1) AS o
                ON f.field_id = o.field_id
                WHERE (f.active = 1 AND o.active IS NULL)
                """
            result = self._query(connection, query).fetchone()
            n_fields_wo_obs = result['n']

            # get number of active guidestars:
            query = """\
                SELECT COUNT(guidestar_id) AS n
                FROM Guidestars
                WHERE active = 1
                """
            result = self._query(connection, query).fetchone()
            n_guidestars = result['n']

            # get number of fields without active guidestars:
            query = """\
                SELECT COUNT(f.field_id) AS n
                FROM Fields AS f
                LEFT JOIN  (
                	SELECT *
                	FROM Guidestars
                	WHERE active = 1) AS g
                ON f.field_id = g.field_id
                WHERE (f.active = 1 AND g.active IS NULL)
                """
            result = self._query(connection, query).fetchone()
            n_fields_wo_guidestars = result['n']

        # print out results:
        if verbose:
            info_constraints = '\n'.join([
                    'Constraints telescope {0}: {1:10d}'.format(i, n) \
                    for i, n in enumerate(n_constraints, start=1)])
            info_fields = '\n'.join([
                    'Fields telescope {0}: {1:15d}'.format(i, n) \
                    for i, n in enumerate(n_fields, start=1)])


            info = dedent("""\
                ======= STORED IN DATABASE ========
                Telescopes:              {0:10d}
                {1}
                -----------------------------------
                Fields total:            {2:10d}
                {3}
                -----------------------------------
                Observations:            {4:10d}
                Pending observations:    {5:10d}
                Fields w/o observations: {6:10d}
                -----------------------------------
                Guidestars:              {7:10d}
                Fields w/o guidestars:   {8:10d}
                -----------------------------------
                """).format(
                        n_telescopes, info_constraints, sum(n_fields),
                        info_fields, n_obs_tot, n_obs_pending,
                        n_fields_wo_obs, n_guidestars, n_fields_wo_guidestars)
            print(info)

        # return results:
        results = {
            'telescopes': n_telescopes,
            'constraints per telescope': n_constraints,
            'fields per telescope': n_fields,
            'observations': n_obs_tot,
            'pending observations': n_obs_pending,
            'fields without observations': n_fields_wo_obs,
            'guidestarts': n_guidestars,
            'fields without guidestars': n_fields_wo_guidestars}

        return results

#==============================================================================

class TelescopeManager(DBManager):
    """Database manager for telescopes and constraints."""

    #--------------------------------------------------------------------------
    def _telescope_exists(self, name):
        """Check if a telescope is stored in the database under a given name.

        Parameters
        ----------
        name : str
            Telescope name.

        Returns
        -------
        exists : bool
            True, if telescope exists in the database. False, otherwise.

        Notes
        -----
        This method calls `_query()`.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT name
                FROM Telescopes
                WHERE name = '{0}'
                """.format(name)
            result = self._query(connection, query).fetchall()

        exists = len(result) > 0

        return exists

    #--------------------------------------------------------------------------
    def _get_parameter_set_id(self, telescope_id):
        """Get the parameter set ID associated with an telescope ID.

        Parameters
        ----------
        telescope_id : int
            Telescope ID.

        Returns
        -------
        parameter_set_id : int
            Parameter set ID.

        Notes
        -----
        This method calls `_query()`.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT parameter_set_id
                FROM ParameterSets
                WHERE (active = 1
                       AND telescope_id = {0})
                """.format(telescope_id)
            result = self._query(connection, query).fetchone()

        if result is None:
            parameter_set_id = -1
        else:
            parameter_set_id = result['parameter_set_id']

        return parameter_set_id

    #--------------------------------------------------------------------------
    def _check_parameter_sets(self, telescope):
        """Check if an active parameter set exists for specified telescope.

        Parameters
        ----------
        telescope : string
            Telescope name.

        Returns
        -------
        add_new : bool
            True, if a new parameter set should be added. False, otherwise.
        deactivate_old : bool
            True, if an existing parameter set should be deactivated. False,
            otherwise.
        parameter_set_id : int
            ID of the parameter set that should be deactivated.

        Notes
        -----
        This method calls `get_telescope_id()` and `_get_parameter_set_id()`.
        """

        # check if active parameter set exists:
        telescope_id = self.get_telescope_id(telescope)
        parameter_set_id = self._get_parameter_set_id(telescope_id)

        add_new = False
        deactivate_old = False

        if parameter_set_id == -1:
            add_new = True
        else:
            response = input(
                "WARNING: An active parameter set for telescope "
                f"'{telescope}' exists. If a new set is added the former "
                "one is marked as inactive. This will deactivate all stored "
                "observabilities and observing windows based on these "
                "parameters. They will remain in the database, but will also "
                "be marked as inactive. Add new parameter set? (y/n) ")

            if response.lower() in ['y', 'yes', 'make it so!']:
                add_new = True
                deactivate_old = True

        return add_new, deactivate_old, parameter_set_id

    #--------------------------------------------------------------------------
    def _deactivate_parameter_set(self, parameter_set_id):
        """Set a parameter set to inactive.

        Parameters
        ----------
        parameter_set_id : int
            Parameter set ID.

        Returns
        -------
        None

        Notes
        -----
        This method calls `_query()`.
        """

        with SQLiteConnection(self.db_file) as connection:

            query = """\
                UPDATE ParameterSets
                SET active = False, date_deactivated = CURRENT_TIMESTAMP
                WHERE parameter_set_id = {0}
                """.format(parameter_set_id)
            self._query(connection, query, commit=True)

        print(f'Parameter set with ID {parameter_set_id} deactivated.')

    #--------------------------------------------------------------------------
    def _deactivate_observabilities(self, parameter_set_id):
        """Set all observabilities to inactive that are associated with a
        specified parameter set ID.

        Parameters
        ----------
        parameter_set_id : int
            Parameter set ID.

        Returns
        -------
        None

        Notes
        -----
        This method calls `_query()`.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                UPDATE Observability
                SET active = False
                WHERE parameter_set_id = {0}
                """.format(parameter_set_id)
            self._query(connection, query, commit=True)
            result = self._query(connection, "SELECT CHANGES()").fetchone()
            n = result['CHANGES()']

        print(f'{n} corresponding observabilities deactivated.')

    #--------------------------------------------------------------------------
    def _deactivate_obs_windows(self, parameter_set_id):
        """Set all observing windows to inactive that are associated with a
        specified parameter set ID.

        Parameters
        ----------
        parameter_set_id : int
            Parameter set ID.

        Returns
        -------
        None

        Notes
        -----
        This method calls `_query()`.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT observability_id
                FROM Observability
                WHERE parameter_set_id = {0}
                """.format(parameter_set_id)
            observability_ids = self._query(connection, query).fetchall()
            observability_ids = [item[0] for item in observability_ids]

            n = 0

            for observability_id in observability_ids:
                query = """\
                    UPDATE ObsWindows
                    SET active = False
                    WHERE observability_id = {0}
                    """.format(observability_id)
                self._query(connection, query, commit=False)
                result = self._query(connection, "SELECT CHANGES()").fetchone()
                n += result['CHANGES()']

            connection.commit()
        print(f'{n} corresponding observing windows deactivated.')

    #--------------------------------------------------------------------------
    def _deactivate_time_ranges(self, parameter_set_id):
        """Set all time ranges to inactive that are associated with a specified
        parameter set ID.

        Parameters
        ----------
        parameter_set_id : int
            Parameter set ID.

        Returns
        -------
        None

        Notes
        -----
        This method calls `_query()`.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                UPDATE TimeRanges
                SET active = False
                WHERE parameter_set_id = {0}
                """.format(parameter_set_id)
            self._query(connection, query, commit=True)
            result = self._query(connection, "SELECT CHANGES()").fetchone()
            n = result['CHANGES()']

        print(f'{n} corresponding time ranges deactivated.')

    #--------------------------------------------------------------------------
    def _add_parameter_set(self, telescope):
        """Add a parameter set to the database.

        Parameters
        ----------
        telescope : str
            Name of the associated telescope.

        Returns
        -------
        parameter_set_id : int
            ID of the newly added parameter set.

        Notes
        -----
        This method calls `_query()`, `get_telescope_id()`, and
        `_last_insert_id()`.
        """

        telescope_id = self.get_telescope_id(telescope)

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                INSERT INTO ParameterSets (telescope_id, active, date_added)
                VALUES ({0}, {1}, CURRENT_TIMESTAMP)
                """.format(telescope_id, True)
            self._query(connection, query, commit=True)
            parameter_set_id = self._last_insert_id(connection)

        return parameter_set_id

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

        Notes
        -----
        This method calls `_query()`.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT constraint_id
                FROM Constraints
                WHERE constraint_name = '{0}'
                """.format(constraint_name)
            result = self._query(connection, query).fetchone()

        if result is None:
            raise NotInDatabase(
                    f"Constraint '{constraint_name}' does not exist.")

        constraint_id = result['constraint_id']

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

        This method calls `_query()` and `_last_insert_id()`.
        """

        with SQLiteConnection(self.db_file) as connection:

            # check for parameter name:
            query = """\
                SELECT parameter_name_id
                FROM ParameterNames
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
                parameter_name_id = result['parameter_name_id']

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

        Notes
        -----
        This method calls `_query()`.
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

        Notes
        -----
        This method calls `_get_constraint_id()` and `_add_parameter()`.
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
            Raised if the constraint returns anything but string, float, int,
            or list as parameters.

        Returns
        -------
        None
        """

        # get constraint ID:
        constraint_name = constraint.__class__.__name__
        constraint_id = self._get_constraint_id(constraint_name)

        # iterate through parameters:
        for param, value in constraint.get_params().items():
            if isinstance(value, list):
                for val in value:
                    self._add_parameter(
                            constraint_id, parameter_set_id, param,
                            value=val)

            elif isinstance(value, str):
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
    def _print_telescope(self, telescope):
        """Print info about telescope.

        Parameters
        ----------
        telescope : str
            Telescope name.

        Returns
        -------
        None
        """

        info = """\
            -----------------------------------
            Telescope ID: {0}
            Name: {1}
            Latitude : {2:10.2f} deg N
            Longitude: {3:10.2f} deg E
            Height:    {4:10.2f} m\
            """.format(
                    telescope['telescope_id'], telescope['name'],
                    Angle(telescope['lat']*u.rad).deg,
                    Angle(telescope['lon']*u.rad).deg, telescope['height'])

        print(dedent(info))

    #--------------------------------------------------------------------------
    def _print_parameter_set(self, parameter_set):
        """Print info about parameter set.

        Parameters
        ----------
        parameter_set : dict
            Parameter set dict as returned by _get_constraints().

        Returns
        -------
        None
        """

        info = """\
            -----------------------------------
            Parameter set ID: {0}
            Active: {1}\
            """.format(
                    parameter_set['parameter_set_id'], parameter_set['active'])

        print(dedent(info))

    #--------------------------------------------------------------------------
    def _print_constraints(self, constraints):
        """Print infos about constraints.

        Parameters
        ----------
        constraints : dict
            Constraints dict as returned by _get_constraints()..

        Returns
        -------
        None
        """

        print('Constraints:')

        for name, pars in constraints.items():
            print(f'* {name}')

            for name, par in pars.items():
                print(f'  - {name}:', par)

    #--------------------------------------------------------------------------
    def add_telescope(self, name, lat, lon, height, utc_offset):
        """Add telescope to database.

        Parameters
        ----------
        name : str
            Telescope name. Must be a unique identifier in the database.
        lat : float
            Telescope latitude in radians.
        lon : float
            Telescope longitude in radians.
        height : float
            Telescope height in meters.
        utc_offset : int
            Telescope UTC offset (daylight saving time).

        Raises
        ------
        ValueError
            Raised, if `name` is not string.
            Raised, if `lat` is neither float nor int.
            Raised, if `lon` is neither float nor int.
            Raised, if `height` is neither float nor int.
            Raised, if `utc_offset` is neither float nor int.

        Returns
        -------
        None
        """

        # check input:
        if not isinstance(name, str):
            raise ValueError("`name` must be string.")

        if type(lat) not in [float, int, np.float64]:
            raise ValueError("`lat` must be float.")

        if type(lon) not in [float, int, np.float64]:
            raise ValueError("`lon` must be float.")

        if type(height) not in [float, int, np.float64]:
            raise ValueError("`height` must be float.")

        if type(utc_offset) not in [float, int, np.float64]:
            raise ValueError("`utc_offset` must be int or float.")

        # open database connection:
        with SQLiteConnection(self.db_file) as connection:
            # check if telescope name exists:
            if self._telescope_exists(name):
                print(f"Telescope '{name}' already exists. Name needs " \
                      "to be unique.")

                return None

            # add telescope to database:
            query = """\
                INSERT INTO Telescopes (name, lat, lon, height, utc_offset)
                VALUES ('{0}', {1}, {2}, {3}, {4});
                """.format(name, lat, lon, height, utc_offset)
            self._query(connection, query, commit=True)
            print(f"Telescope '{name}' added.")

    #--------------------------------------------------------------------------
    def add_constraints(self, telescope, twilight, constraints=()):
        """Add constraints to database.

        Parameters
        ----------
        telescope : str
            Name of the telescope that the constraints are associated with.
        twilight : float or str
            If str, must be 'astronomical' (-18 deg), 'nautical' (-12 deg),
            'civil' (-6 deg), or 'sunset' (0 deg). Use float otherwise.
        constraints : list or tuple of constraints.Constraint, optional
            The constraints to be added to the database for the specified
            telescope. The default is ().

        Returns
        -------
        None

        Notes
        -----
        This method calls `_telescope_exists()`, `_check_paramter_sets()`,
        `_deactivate_parameter_set()`, `_deactivate_observabilities()`,
        `_deactivate_obs_windows()`, `_deactivate_time_ranges()`,
        `_add_parameter_set()`, `_add_twilight()`, and `_add_constraint()`.
        """

        # check if telescope exists:
        if not self._telescope_exists(telescope):
            print(f"WARNING: Telescope '{telescope}' does not exist in "
                  "database. Use TelescopeManager() to manage telescopes or "
                  "add new ones.")
            return None

        # check if parameter set exists:
        add_new, deactivate_old, deactivate_id = self._check_parameter_sets(
                telescope)

        # deactivate former parameter set and related stored items:
        if deactivate_old:
            self._deactivate_parameter_set(deactivate_id)
            self._deactivate_observabilities(deactivate_id)
            self._deactivate_obs_windows(deactivate_id)
            self._deactivate_time_ranges(deactivate_id)

        # add new parameter set:
        if add_new:

            # add parameter set:
            parameter_set_id = self._add_parameter_set(telescope)

            # add twilight constraint:
            self._add_twilight(twilight, parameter_set_id)

            # add constraints:
            for constraint in constraints:
                self._add_constraint(constraint, parameter_set_id)
                constraint_name = constraint.__class__.__name__
                print(f"Constraint '{constraint_name}' for telescope "
                      f"'{telescope}'.")

    #--------------------------------------------------------------------------
    def get_telescope_id(self, name):
        """Get the telescope ID by name.

        Parameters
        ----------
        name : std
            telescope name.

        Raises
        ------
        NotInDatabase
            Raised if telescope name does not exist in database.

        Returns
        -------
        telescope_id : int
            The ID of the telescope in the database.

        Notes
        -----
        This method calls `_query()`.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT telescope_id
                FROM Telescopes
                WHERE name = '{0}'
                """.format(name)
            result = self._query(connection, query).fetchone()

        if result is None:
            raise NotInDatabase(f"telescope '{name}' does not exist.")

        telescope_id = result['telescope_id']

        return telescope_id

    #--------------------------------------------------------------------------
    def get_telescope_names(self):
        """Get telescope names from database.

        Returns
        -------
        telescope_names : list of str
            Each list item is a telescope name stored in the database.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT name
                FROM Telescopes
                """
            results = self._query(connection, query).fetchall()

        telescope_names = [result['name'] for result in results]

        return telescope_names

    #--------------------------------------------------------------------------
    def get_telescopes(self, name=None, constraints=False):
        """Get telescope(s) from database.

        Parameters
        ----------
        name : str, optional
            telescope name. If none is given, all telescopes are returned
            as list of dict. The default is None.
        constraints : bool or str, optional
            If True, add active constraints to the telescope dict.
            If 'all', add all parameter sets associated with the telescope to
            the telescope dict.

        Returns
        -------
        telescopes : list of dict
            Each list item is a dictionary corresponding to one telescope.
            If a telescope name is given, the list contains only one item.
        """

        # define SQL WHERE conditions:
        if name is None:
            where_clause = ""
        elif isinstance(name, str):
            where_clause = f"WHERE name='{name}'"
        else:
            raise ValueError("`name` must be str or None.")

        # query telescope(s) from database:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT *
                FROM Telescopes
                {0}
                """.format(where_clause)
            results = self._query(connection, query).fetchall()

        # add constraints, if requested:
        telescopes = []

        for telescope in results:

            if isinstance(constraints, str) and constraints.lower() == 'all':
                parameter_sets = self.get_constraints(
                        telescope=telescope['name'], active=False)
                telescope['parameter_sets'] = parameter_sets
            elif constraints:
                parameter_set = self.get_constraints(
                        telescope=telescope['name'], active=True)
                telescope['parameter_set_id'] = \
                        parameter_set[0]['parameter_set_id']
                telescope['constraints'] = parameter_set[0]['constraints']

            telescopes.append(telescope)

        return telescopes

    #--------------------------------------------------------------------------
    def get_constraints(self, telescope=None, active=True):
        """Query constraints associated with a specified telescope from the
        database.

        Parameters
        ----------
        telescope : str, optional
            Name of the telescope. If None is given, constraints for all
            telescopes are returned. The default is None
        active : bool or None
            If True, only active constraints are returned. If False,
            constraints are returned regardless of whether they are active or
            not. The default is True.

        Raises
        ------
        NotInDatabase
            Raised if no parameter set is stored for the specified conditions.

        Returns
        -------
        parameter_sets : list of dict
            Each item is a dict that contains the parameter set ID,
            telescope name, whether the set is active or not, and a dict of
            constraints, which is structured as following. The keys are the
            constraint names. The values are dictionaries that contain the
            constraint parameter names as keys and associated values.

        Notes
        -----
        This method calls `_query()`.
        """

        # query constraints and parameter values:
        with SQLiteConnection(self.db_file) as connection:
            # define SQL WHERE conditions:
            if telescope and active:
                where_clause = \
                        f"WHERE(t.name = '{telescope}' AND ps.active = TRUE)"
            elif telescope:
                where_clause = f"WHERE(t.name = '{telescope}')"
            elif active:
                where_clause = "WHERE(ps.active = TRUE)"
            else:
                where_clause = ""

            query = """\
                SELECT ps.parameter_set_id, pn.parameter_name, p.value,
                    p.svalue, c.constraint_name, t.name AS telescope_name,
                    ps.active
                FROM Parameters p
                LEFT JOIN ParameterNames pn
                    ON p.parameter_name_id = pn.parameter_name_id
                LEFT JOIN Constraints c
                    ON p.constraint_id = c.constraint_id
                LEFT JOIN ParameterSets ps
                	ON p.parameter_set_id = ps.parameter_set_id
                LEFT JOIN Telescopes t
                	ON ps.telescope_id = t.telescope_id
                {0}
                """.format(where_clause)
            results = self._query(connection, query).fetchall()

        # no parameter set exists:
        if telescope and active and not results:
            raise NotInDatabase(
                "No active constraints stored for telescope "
                f"'{telescope}'.")
        elif telescope and not results:
            raise NotInDatabase(
                f"No constraints stored for telescope '{telescope}'.")
        elif active and not results:
            raise NotInDatabase("No active constraints stored.")
        elif not results:
            raise NotInDatabase("No constraints stored.")

        # parse to dictionaries:
        parameter_sets = []
        parameter_set = {}
        constraints = {}
        counter = 1

        for i, result in enumerate(results):
            parameter_set_id = result['parameter_set_id']
            parameter_name = result['parameter_name']
            value = result['value']
            svalue = result['svalue']
            constraint_name = result['constraint_name']
            telescope_name = result['telescope_name']
            active_set = bool(result['active'])

            if i == 0:
                counter = parameter_set_id

            # if parameter set ID increases, add set and start new one
            if parameter_set_id > counter:
                parameter_set['constraints'] = constraints
                parameter_sets.append(parameter_set)
                parameter_set = {}
                constraints = {}
                counter = parameter_set_id

            # add parameter set info:
            parameter_set['parameter_set_id'] = parameter_set_id
            parameter_set['telescope'] = telescope_name
            parameter_set['active'] = active_set

            # add constraint dictionary:
            if constraint_name not in constraints.keys():
                constraints[constraint_name] = {}

            # add parameter:
            if value is None:
                constraints[constraint_name][parameter_name] = svalue
            elif (parameter_name in constraints[constraint_name] \
                    and not isinstance(
                        constraints[constraint_name][parameter_name], list)):
                constraints[constraint_name][parameter_name] = [
                        constraints[constraint_name][parameter_name]]
                constraints[constraint_name][parameter_name].append(value)
            elif parameter_name in constraints[constraint_name]:
                constraints[constraint_name][parameter_name].append(value)
            else:
                constraints[constraint_name][parameter_name] = value

        # store final parameter set:
        parameter_set['constraints'] = constraints
        parameter_sets.append(parameter_set)

        return parameter_sets

    #--------------------------------------------------------------------------
    def info(self, constraints=False):
        """Print infos about telescopes and constraints.

        Parameters
        ----------
        constraints : bool or str, optional
            If 'all', print info about all parameter sets and constraints.
            If True, print info about active constraints. Otherwise, no info
            about constraints is printed. The default is False.

        Returns
        -------
        None
        """

        telescopes = self.get_telescopes(constraints=constraints)

        print('============ TELECOPES ============')
        print(f'{len(telescopes)} telescopes stored in database.')

        for telescope in telescopes:
            self._print_telescope(telescope)

            # print all parameter sets:
            if isinstance(constraints, str) and constraints.lower() == 'all':
                n_par_sets = len(telescope['parameter_sets'])
                print('-----------------------------------')
                print(f'{n_par_sets} associated parameter sets')

                for par_set in telescope['parameter_sets']:
                    self._print_parameter_set(par_set)
                    self._print_constraints(par_set['constraints'])

            # print active constraints:
            elif constraints:
                self._print_constraints(telescope['constraints'])

        print('-----------------------------------\n')

#==============================================================================

class FieldManager(DBManager):
    """Database manager for fields."""

    #--------------------------------------------------------------------------
    def add_fields(self, fields, telescope, active=True, n_batch=1000):
        """Add fields to the database.

        Parameters
        ----------
        fields : fieldgrid.FieldGrid
            The fields to add.
        telescope : str
            Name of the telescope associated with the fields.
        active : bool, optional
            If True, fields are added as active, and as inactive otherwise. The
            default is True.
        n_batch : int, optinal
            Add fields in batches of this size to the data base. The default is
            1000.

        Raises
        ------
        ValueError
            Raised, if 'fields' is not FieldGrid instance.
            Raised, if 'n_batch' is not an integer > 0.

        Returns
        -------
        None

        Notes
        -----
        This method uses `TelescopeManager()`. This method calls `_query()`.
        """

        # check input:
        if not isinstance(fields, FieldGrid):
            raise ValueError("'fields' must be FieldGrid instance.")

        if not isinstance(n_batch, int) or n_batch < 1:
            raise ValueError("'n_batch' must be integer > 0.")

        center_ras, center_decs = fields.get_center_coords()
        fov = fields.fov
        tilt = fields.tilt
        telescope_manager = TelescopeManager(self.db_file)
        telescope_id = telescope_manager.get_telescope_id(telescope)
        active = bool(active)
        n_fields = len(fields)
        n_iter = ceil(n_fields / n_batch)

        with SQLiteConnection(self.db_file) as connection:
            # iterate through batches:
            for i in range(n_iter):
                j = i * n_batch
                k = (i + 1) * n_batch
                print(
                    '\rAdding fields {0}-{1} of {2} ({3:.1f}%)..'.format(
                            j, k-1 if k <= n_fields else n_fields, n_fields,
                            i*100./n_iter),
                    end='')

                data = [(fov, center_ra, center_dec, tilt, telescope_id,
                         active) \
                        for center_ra, center_dec \
                        in zip(center_ras[j:k], center_decs[j:k])]

                query = """\
                    INSERT INTO Fields (
                        fov, center_ra, center_dec, tilt, telescope_id,
                        active)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """
                self._query(connection, query, many=data, commit=True)

            print(f'\r{n_fields} fields added to database.                   ')

    #--------------------------------------------------------------------------
    '''
    def get_fields(
            self, telescope=None, observed=None, pending=None,
            observable_between=None, observable_time=None, active=True,
            needs_obs_windows=None, init_obs_windows=False):
        """Get fields from the database, given various selection criteria.

        Parameters
        ----------
        telescope : str, optional
            Telescope name. If set, only query fields associated with this
            telescope. If None, query fields for all telescopes. The
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
        observable_between : tuple of str or None, optional
            Provide two strings that encode date+time. Then, only fields are
            returned that are observable (at least for some time) between the
            two specified times. Either set this argument or
            `observable_time`. If this argument is set, `observable_time` is
            not used. The default is None.
        observable_time : str or None, optional
            Provide a string that encodes a date+time. Then, only fields are
            returned that are observable at that specific time. Either set this
            argument or `observable_between`. If `observable_between` is given,
            this argument is ignored. The default is None.
        active : bool, optional
            If True, only query active fields. If False, only query inactive
            fields. If None, query fields independent of whether they are
            active or not. The default is True.
        needs_obs_window : float or None, optional
            If JD is given, only fields are returned that need additional
            observing window calculations up to this JD. The default is None.
        init_obs_windows : bool, optional
            If True, query field that do not have any observing windows stored
            yet. The default is False.

        Raises
        ------
        ValueError
            Raised, if `needs_obs_window` is set and `init_obs_windows=True`.
            Only one option can be selected at a time.

        Returns
        -------
        results : list of dict
            Each list item is a dict with the field parameters.
        """

        # check input:
        if needs_obs_windows is not None and init_obs_windows:
            raise ValueError(
                    'Either set `needs_obs_window` OR use '\
                    '`init_obs_windows=True`.')

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

        # set query condition for telescope:
        if telescope:
            condition_telescope = " AND telescope = '{0}'".format(telescope)
        else:
            condition_telescope = ""

        # set query condition for observable between two dates:
        if observable_between:
            join_sel = ", w.date_start, w.date_stop, w.duration, w.status, " \
                    "w.setting_duration"
            join_observable = """\
                INNER JOIN (
                	SELECT *
                	FROM Observable
                	WHERE (
                		date_start <= "{0}"
                		AND date_stop >= "{1}"
                		)
                	) AS w
                ON f.field_id = w.field_id
                """.format(observable_between[1], observable_between[0])

        # set query condition for observable at a specific time:
        elif observable_time:
            join_sel = ", w.date_start, w.date_stop, w.duration, w.status, " \
                    "w.setting_duration"
            join_observable = """\
                INNER JOIN (
                	SELECT *
                	FROM Observable
                	WHERE (
                		date_start <= "{0}"
                		AND date_stop >= "{0}")
                	) AS w
                ON f.field_id = w.field_id
                """.format(observable_time)

        else:
            join_sel = ""
            join_observable = ""

        # set query condition for observing window requirement:
        if needs_obs_windows:
            condition_obswindow = " AND jd_next < '{0}'".format(
                    needs_obs_windows)
        elif init_obs_windows:
            condition_obswindow = " AND jd_next IS NULL"
        else:
            condition_obswindow = ""

        # query data base:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT f.field_id, f.fov, f.center_ra, f.center_dec, f.tilt,
                    f.telescope, f.active, f.jd_first, f.jd_next, f.nobs_tot,
                    f.nobs_done, f.nobs_pending {0}
                FROM FieldsObs AS f
                {1}
                WHERE (active = {2} {3} {4} {5} {6});
                """.format(
                        join_sel, join_observable, active, condition_telescope,
                        condition_observed, condition_pending,
                        condition_obswindow)
            results = self._query(connection, query).fetchall()

        return results
    '''

    #--------------------------------------------------------------------------
    def get_fields(
            self, telescope=None, observed=None, pending=None,
            observable_between=None, observable_time=None, night=None,
            active=True, needs_obs_windows=None, init_obs_windows=False):
        """Get fields from the database, given various selection criteria.

        Parameters
        ----------
        telescope : str, optional
            Telescope name. If set, only query fields associated with this
            telescope. If None, query fields for all telescopes. The
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
        observable_between : tuple of str or None, optional
            Provide two strings that encode date+time. Then, only fields are
            returned that are observable (at least for some time) between the
            two specified times. The default is None.
        observable_time : str or None, optional
            Provide a string that encodes a date+time. Then, only fields are
            returned that are observable at that specific time. If
            `observable_between` is given, this argument is ignored. The
            default is None.
        night : str or None, optional
            Provide a string that encodes a date. All fields and their
            observability status (including non-observable) for the specified
            night are returned.If `observable_between` or `observable_time` is
            given, this argument is ignored. The default is None.
        active : bool, optional
            If True, only query active fields. If False, only query inactive
            fields. If None, query fields independent of whether they are
            active or not. The default is True.
        needs_obs_window : float or None, optional
            If JD is given, only fields are returned that need additional
            observing window calculations up to this JD. The default is None.
        init_obs_windows : bool, optional
            If True, query field that do not have any observing windows stored
            yet. The default is False.

        Raises
        ------
        ValueError
            Raised, if `needs_obs_window` is set and `init_obs_windows=True`.
            Only one option can be selected at a time.

        Returns
        -------
        results : list of dict
            Each list item is a dict with the field parameters.
        """

        # check input:
        if needs_obs_windows is not None and init_obs_windows:
            raise ValueError(
                    'Either set `needs_obs_window` OR use '\
                    '`init_obs_windows=True`.')

        selection = ""
        join = ""
        condition = ""

        # set query condition for observable between two dates:
        if observable_between:
            selection = ", o.date_start, o.date_stop, o.duration, o.status, " \
                    "o.setting_duration"
            join = """\
                LEFT JOIN Observable AS o
                ON f.field_id = o.field_id
                """
            condition = """\
                WHERE (
                    o.date_start < '{0}'
                    AND o.date_stop > '{1}')
                """.format(observable_between[1], observable_between[0])

        # set query condition for observable at a specific time:
        elif observable_time:
            selection = ", o.date_start, o.date_stop, o.duration, o.status, " \
                    "o.setting_duration"
            join = """\
                LEFT JOIN Observable AS o
                ON f.field_id = o.field_id
                """
            condition = """\
                WHERE (
                    o.date_start <= '{0}'
                    AND o.date_stop > '{0}')
                """.format(observable_time)

        # set query condition for a specified night:
        elif night:
            jd = Time(Time(night).iso[:10]).jd # truncate time
            selection = ", o.jd, o.date_start, o.date_stop, o.duration, " \
                    "o.status, o.setting_duration"
            join = """\
                LEFT JOIN Observable AS o
                ON f.field_id = o.field_id
                """
            condition = """\
                WHERE o.jd = '{0}'
                """.format(jd)

        # set query condition for observing window requirement:
        elif needs_obs_windows:
            condition = "WHERE jd_next < '{0}'".format(
                    needs_obs_windows)
        elif init_obs_windows:
            condition = "WHERE jd_next IS NULL"

        # query data base:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT f.field_id, f.fov, f.center_ra, f.center_dec, f.tilt,
                    f.telescope, f.active, f.jd_first, f.jd_next, f.nobs_tot,
                    f.nobs_done, f.nobs_pending {0}
                FROM FieldsObs AS f
                {1}
                {2};
                """.format(selection, join, condition)
            results_temp = self._query(connection, query).fetchall()

        # apply additional conditions:
        # note: post-filtering in python is done, because including these
        # conditions in the SQL query unproportionally blows up the runtime.
        results = []

        for __ in range(len(results_temp)):
            keep = True
            result = results_temp.pop(0)

            # condition: telescope:
            if telescope is None:
                pass
            elif telescope != result['telescope']:
                keep = False

            # condition: observed or not:
            if observed is None:
                pass
            elif observed and result['nobs_done'] == 0:
                keep = False
            elif not observed and result['nobs_done'] > 0:
                keep = False

            # condition: pending or not:
            if pending is None:
                pass
            elif pending and result['nobs_pending'] == 0:
                keep = False
            elif not pending and result['nobs_pending'] > 0:
                keep = False

            # condition: active or not:
            if active is None:
                pass
            elif active and not result['active']:
                keep = False
            elif not active and result['active']:
                keep = False

            if keep:
                results.append(result)

        return results

    #--------------------------------------------------------------------------
    def get_field_by_id(self, field_id):
        """Query field from database by ID.

        Parameters
        ----------
        field_id : int
            Field ID.

        Returns
        -------
        result : dict
            The field parameters.

        Notes
        -----
        This method calls `_query()`.
        """

        # query data base:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                    SELECT f.field_id, f.fov, f.center_ra, f.center_dec,
                        f.tilt, o.name telescope, f.active,
                        f.jd_first_obs_window, f.jd_next_obs_window,
                        p.nobs_done, p.nobs_tot,
                        p.nobs_tot - p.nobs_done AS nobs_pending
                    FROM Fields AS f
                    LEFT JOIN Observatories AS o
                        ON f.telescope_id = o.telescope_id
                    LEFT JOIN (
                        SELECT field_id, SUM(Done) nobs_done, COUNT(*) nobs_tot
                        FROM Observations
                        GROUP BY field_id
                        ) AS p
                    ON f.field_id = p.field_id
                    WHERE f.field_id = {0}
                    """.format(field_id)
            result = self._query(connection, query).fetchone()

        return result

    #--------------------------------------------------------------------------
    def get_fields_missing_setting_duration(self):
        # TODO: update docstring
        """Query fields from the database that have missing or unknown
        observability status in some entries.

        Returns
        -------
        results : list of dict
            Each dict contains the field ID and the corresponding MJD for
            which an observability is stored with unset or unknown status.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """
            WITH o AS (
                SELECT field_id, jd
                FROM Observability
                WHERE (
                	status_id = 4
                    AND setting_duration IS NULL
                	AND active = 1)
                )
            SELECT o.field_id, o.jd, t.jd_next
            FROM o
            LEFT JOIN TimeRanges AS t
            ON o.field_id = t.field_id
            """
            results = self._query(connection, query).fetchall()

        return results

    #--------------------------------------------------------------------------
    def annual_availability(self):
        """Query how many nights each active field is observable per year.

        Returns
        -------
        results : list of dict
            Each entry contains the field ID and the number of days the field
            is observable per year. The value is -1 if the observability has
            been checked for less than 365 consecutive days.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """
            SELECT tr1.field_id,
            	CASE WHEN tr1.jd_next-tr1.jd_first-1 >= 365
            	THEN a.availability
            	ELSE -1
            	END AS availability
            FROM TimeRanges tr1
            LEFT JOIN (
            	SELECT o.field_id, COUNT(o.jd) AS availability
            	FROM Observability o
            	LEFT JOIN ObservabilityStatus os
            	ON o.status_id = os.status_id
            	LEFT JOIN TimeRanges tr2
            	ON o.field_id = tr2.field_id
            	WHERE (
            		os.status != 'not observable'
            		AND o.active = 1
            		AND tr2.active = 1
            		AND o.jd >= tr2.jd_next-366
            		)
            	GROUP BY o.field_id
            	) AS a
            ON a.field_id = tr1.field_id
            """
            results = self._query(connection, query).fetchall()

            # replace Nones with 0:
            for result in results:
                if result['availability'] is None:
                    result['availability'] = 0

        return results

    #--------------------------------------------------------------------------
    def info(self, telescope=None, active=True):
        """Print info about fields.

        Parameters
        ----------
        telescope : bool or None, optional
            Telescope name. If given, print info only about fields associated
            with this telescope. The default is None.
        active : bool or None, optional
            If True, print info only about active fields. If False, print info
            only about inactive fields. If None, print info about fields,
            regardless of whether they are active or not. The default is True.

        Returns
        -------
        None
        """

        fields = self.get_fields(telescope=telescope, active=active)
        fields = DataFrame(fields)
        n_fields = fields.shape[0]

        print('============== FIELDS =============')

        if telescope is None and active is None:
            print('All fields in the database')
        elif telescope is None and active:
            print('All active fields in the database')
        elif telescope is None and not active:
            print('All inactive fields in the database')
        elif active:
            print(f"Active fields associated with telescope '{telescope}'")
        elif not active:
            print(f"Inactive fields associated with telescope '{telescope}'")
        else:
            print(f"Fields associated with telescope '{telescope}'")

        print('-----------------------------------')
        print(f'Total number of fields: {n_fields:11d}')

        if n_fields:
            n_pending_fields = np.sum(fields.iloc[:,9] > 0)
            n_pending_obs = np.sum(fields.iloc[:,9])
            n_finished_fields = np.sum(fields.iloc[:,9] == 0)
            n_finished_obs = np.sum(fields.iloc[:,8])

            print(f'Pending fields:         {n_pending_fields:11d}')
            print(f'Pending observations:   {n_pending_obs:11d}')
            print(f'Finished fields:        {n_finished_fields:11d}')
            print(f'Finished observations:  {n_finished_obs:11d}')

        print('-----------------------------------\n')

#==============================================================================

class GuidestarManager(DBManager):
    """Database manager for guidestars."""

    #--------------------------------------------------------------------------
    def _get_by_field_id(
            self, field_id, active=True, essential_columns_only=False):
        """Get guidestars for a specific field from database.

        Parameters
        ----------
        field_id : int
            Only guidestars associated with that field are returned.
        active : bool, optional
            If True, only active guidestars are returned. If False, only
            inactive guidestars are returned. If None, all guidestars are
            returned regardless of whether they are active or not. The default
            is True.
        essential_columns_only : bool, optional
            If True and field_id is not None, only the following columns are
            queried: guidestar_id, ra, dec. Otherwise, all columns are queried.

        Returns
        -------
        results : list of dict
            List of guidestars. Each dict contains the guidestar ID,
            associated field ID, guidestar right ascension in rad, and
            guidestar declination in rad.
        """

        # define queried columns:
        if essential_columns_only:
            columns = 'guidestar_id, ra, dec, mag'
        else:
            columns = '*'

        # define SQL WHERE clause:
        if active is None:
            where_clause = f"WHERE field_id = '{field_id}'"
        elif active:
            where_clause = f"WHERE (field_id = '{field_id}' AND active = TRUE)"
        else:
            where_clause = f"WHERE (field_id = '{field_id}' "\
                    "AND active = FALSE)"

        # query:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT {0}
                FROM Guidestars
                {1}
                """.format(columns, where_clause)
            results = self._query(connection, query).fetchall()

        return results

    #--------------------------------------------------------------------------
    def _get_all(self, active=True):
        """Get all guidestars from database.

        Parameters
        ----------
        active : bool, optional
            If True, only active guidestars are returned. If False, only
            inactive guidestars are returned. If None, all guidestars are
            returned regardless of whether they are active or not. The default
            is True.

        Returns
        -------
        results : list of dict
            List of guidestars. Each dict contains the guidestar ID,
            associated field ID, guidestar right ascension in rad, and
            guidestar declination in rad.
        """

        # define SQL WHERE clause:
        if active is None:
            where_clause = ""
        elif active:
            where_clause = "WHERE active = TRUE"
        else:
            where_clause = "WHERE active = FALSE"

        # query:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT *
                FROM Guidestars
                {0}
                """.format(where_clause)
            results = self._query(connection, query).fetchall()

        return results

    #--------------------------------------------------------------------------
    def _warn_repetition(self, field_ids, ras, decs, limit):
        """Warn if new guidestars for a field are close to guidestars already
        stored in the database for that field.

        Parameters
        ----------
        field_ids : numpy.ndarray
            IDs of the fields that the guidestar coordinates correspond to.
        ras : astropy.coord.Angle
            Guidestar right ascensions.
        decs : astropy.coord.Angle
            Guidestar declinations.
        limit : astropy.coord.Angle
            Separation limit. If a new guidestart is closer to an existing one
            than this limit, a warning is printed. The user is asked whether or
            not to keep this new guidestar.

        Returns
        -------
        field_ids : numpy.ndarray
            Field ID associations of the kept guidestars.
        ras : astropy.coord.Angle
            Right ascensions of the kept guidestars.
        decs : astropy.coord.Angle
            Declinations of the kept guidestars.
        """

        print('Checking for repetitions..')

        # get stored guidestar coordinates and associated field IDs:
        stored_gs_id = []
        stored_gs_field_ids = []
        stored_gs_ras = []
        stored_gs_decs =[]

        for guidestar in self._get_all():
            stored_gs_id.append(guidestar['guidestar_id'])
            stored_gs_field_ids.append(guidestar['field_id'])
            stored_gs_ras.append(guidestar['ra'])
            stored_gs_decs.append(guidestar['dec'])

        if not stored_gs_id:
            return field_ids, ras, decs

        stored_gs_coords = SkyCoord(stored_gs_ras, stored_gs_decs, unit='rad')
        del stored_gs_ras, stored_gs_decs

        stored_gs_field_ids = np.array(stored_gs_field_ids)
        new_gs_field_ids = field_ids
        new_gs_coords = SkyCoord(ras, decs)

        keep = np.ones(new_gs_field_ids.shape[0], dtype=bool)

        # iterate through new guidestars:
        for i, (field_id, coord) in enumerate(zip(
                new_gs_field_ids, new_gs_coords)):
            # calculate separations:
            sel = stored_gs_field_ids == field_id

            if not np.sum(sel):
                continue

            separation = stored_gs_coords[sel].separation(coord)
            close = separation <= limit

            # ask user about critical cases:
            if np.any(close):
                print(f'New guidestar\n  field ID: {field_id}\n'
                      f'  RA:   {coord.ra.to_string(pad=True)}\n'
                      f'  Dec: {coord.dec.to_string(alwayssign=1, pad=True)}\n'
                      'is close to the following stored guidestar(s):')

            for j, k in enumerate(np.nonzero(close)[0], start=1):
                print(f'{j}. Guidestar ID: {stored_gs_field_ids[sel][k]}\n'
                      '  RA:   {0}\n'.format(
                              stored_gs_coords[sel][k].ra.to_string(pad=True)),
                      '  Dec: {0}\n'.format(
                              stored_gs_coords[sel][k].dec.to_string(
                                  alwayssign=1, pad=True)),
                      f'  separation: {separation[k].deg:.4f} deg'
                      )

            if any(close):
                user_in = input('Add this new guidestar anyway? (y/n) ')
                if user_in.lower() not in ['y', 'yes', 'make it so!']:
                    keep[i] = False

        field_ids = field_ids[keep]
        ras = ras[keep]
        decs = decs[keep]

        return field_ids, ras, decs

    #--------------------------------------------------------------------------
    def _warn_separation(self, field_ids, ras, decs, limit):
        """Warn if new guidestars for a field are separated too much from the
        field center.

        Parameters
        ----------
        field_ids : numpy.ndarray
            IDs of the fields that the guidestar coordinates correspond to.
        ras : astropy.coord.Angle
            Guidestar right ascensions.
        decs : astropy.coord.Angle
            Guidestar declinations.
        limit : astropy.coord.Angle
            Separation limit. If a new guidestart is sparated from the
            corresponding field center by more than this limit, a warning is
            printed. The user is asked whether or not to keep this new
            guidestar.

        Returns
        -------
        field_ids : numpy.ndarray
            Field ID associations of the kept guidestars.
        ras : astropy.coord.Angle
            Right ascensions of the kept guidestars.
        decs : astropy.coord.Angle
            Declinations of the kept guidestars.
        """

        print('Checking for large separations..')

        keep = np.ones(len(field_ids), dtype=bool)

        with SQLiteConnection(self.db_file) as connection:
            for i, (field_id, ra, dec) in enumerate(zip(field_ids, ras, decs)):

                # query data base:
                query = """\
                        SELECT center_ra, center_dec
                        FROM Fields
                        WHERE field_id = {0};
                        """.format(field_id)
                result = self._query(connection, query).fetchone()
                field_ra = result['center_ra']
                field_dec = result['center_dec']

                # calculate separation:
                field_coord = SkyCoord(field_ra, field_dec, unit='rad')
                guidestar_coord = SkyCoord(ra, dec, unit='rad')
                separation = field_coord.separation(guidestar_coord)

                # ask user about critical cases:
                if separation > limit:
                    print(f'New guidestar {i} for field ID {field_id} is too '
                          'far from the field center with separation '
                          '{0}.'.format(separation.to_string(sep='dms')))
                    user_in = input('Add it to the database anyway? (y/n) ')

                    if user_in.lower() not in ['y', 'yes', 'make it so!']:
                        keep[i] = False

        field_ids = field_ids[keep]
        ras = ras[keep]
        decs = decs[keep]

        return field_ids, ras, decs

    #--------------------------------------------------------------------------
    def _warn_missing(self):
        """Check if any fields exist in the database without any associated
        guidestars.

        Returns
        -------
        field_ids : list
            Field IDs of fields without associated guidestars.

        Notes
        -----
        This method calls `get_fields_missing_guidestar()`.
        """

        print('Checking for fields missing guidestars..')

        # query data base:
        field_ids = self.get_fields_missing_guidestar()

        # inform user about fields without guidestars:
        if field_ids['none']:
            print('\nWARNING: Fields with the following IDs do not have '
                  'any guidestars associated:')
            text = ''

            for field_id in field_ids['none']:
                text = f'{text}{field_id}, '

            print(text[:-2])

        # inform user about fields without active guidestars:
        if field_ids['inactive']:
            print('\nWARNING: Fields with the following IDs do not have '
                  'active (but inactive) guidestars associated:')
            text = ''

            for field_id in field_ids['inactive']:
                text = f'{text}{field_id}, '

            print(text[:-2])

        if not field_ids['none'] and not field_ids['inactive']:
            print('All fields have at least one guidestar associated.')

        return field_ids

    #--------------------------------------------------------------------------
    def _add_guidestar(self, field_ids, ras, decs, mags, n_batch=1000):
        """Add new guidestars to the database.

        Parameters
        ----------
        field_ids : numpy.ndarray
            IDs of the fields that the guidestar coordinates correspond to.
        ras : astropy.coord.Angle
            Guidestar right ascensions.
        decs : astropy.coord.Angle
            Guidestar declinations.
        mags : list
            Guidestar magnitudes.
        n_batch : int, optinal
            Add guidestars in batches of this size to the data base. The
            default is 1000.

        Returns
        -------
        None
        """

        n_guidestars = len(field_ids)
        n_iter = ceil(n_guidestars / n_batch)

        with SQLiteConnection(self.db_file) as connection:
            # iterate through batches:
            for i in range(n_iter):
                j = i * n_batch
                k = (i + 1) * n_batch
                print(
                    '\rAdding guidestars {0}-{1} of {2} ({3:.1f}%)..'.format(
                            j, k-1 if k <= n_guidestars else n_guidestars,
                            n_guidestars, i*100./n_iter),
                    end='')

                data = [(int(field_id), float(ra), float(dec), float(mag),
                         True) for field_id, ra, dec, mag in zip(
                                field_ids[j:k], ras.rad[j:k], decs.rad[j:k],
                                mags[j:k])]

                query = """\
                    INSERT INTO Guidestars (
                        field_id, ra, dec, mag, active)
                    VALUES (?, ?, ?, ?, ?)
                    """
                self._query(connection, query, many=data, commit=True)

        print(f'\r{n_guidestars} new guidestars added to database.           ')

    #--------------------------------------------------------------------------
    def add_guidestars(
            self, field_ids, ra, dec, mag, warn_missing=True, warn_rep=0,
            warn_sep=0, n_batch=1000):
        """Add new guidestar(s) to the database.

        Parameters
        ----------
        field_ids : int or list of int
            IDs of the fields that the guidestar coordinates correspond to.
        ra : float or list of float
            Guidestar right ascensions in rad.
        dec : float or list of floats
            Guidestar declinations in rad.
        mag : float or list of floats
            Guidestar magnitude.
        warn_missing : bool, optional
            If True, warn about fields that do not have any associated
            guidestars stored in the database. The default is True.
        warn_rep : float or astopy.coord.Angle, optional
            If a float or Angle larger than 0 is given, the user is warned
            about new guidestars that may be duplicates of existing entries in
            the database. The value in `warn_rep` is the largest separation
            allowed not to be considered a duplicate. A float is interpreted as
            angle in rad. The default is 0.
        warn_sep : float or astopy.coord.Angle, optional
            If a float or Angle larger than 0 is given, the user is warned
            about new guidestars that may be too far off from the corresponding
            field center. The value in `warn_sep` is the largest separation
            allowed. A float is interpreted as angle in rad. The default is 0.
        n_batch : int, optinal
            Add guidestars in batches of this size to the data base. The
            default is 1000.

        Raises
        ------
        ValueError
            Raised, if `field_ids` is neither int nor list-like.
            Raised, if `ra` is neither float nor list-like.
            Raised, if `dec` is neither float nor list-like.
            Raised, if `mag` is neither float nor list-like.
            Raised, if `warn_missing` is not bool.
            Raised, if `warn_rep` is neither float nor astropy.coord.Angle.
            Raised, if `warn_sep` is neither float nor astropy.coord.Angle.
            Raised, if 'n_batch' is not an integer > 0.

        Returns
        -------
        None

        Notes
        -----
        This method calls `_add_guidestar()`, `_warn_repetition()`,
        `_warn_separation()`, and `_warn_missing()`.
        """

        # check input:
        if isinstance(field_ids, int):
            field_ids = np.array([field_ids])
        else:
            try:
                field_ids = np.array(field_ids)
            except:
                raise ValueError('`field_ids` must be int or list-like.')

        if type(ra) in [float, int, np.float64]:
            ras = [ra]
        else:
            try:
                ras = list(ra)
            except:
                raise ValueError('`ra` must be float or list-like.')

        ras = Angle(ras, unit='rad')

        if type(dec) in [float, int, np.float64]:
            decs = [dec]
        else:
            try:
                decs = list(dec)
            except:
                raise ValueError('`dec` must be float or list-like.')

        decs = Angle(decs, unit='rad')

        if type(mag) in [float, int, np.float64]:
            mag = [mag]
        else:
            try:
                mag = list(mag)
            except:
                raise ValueError('`mag` must be float or list-like.')

        if not isinstance(warn_missing, bool):
            raise ValueError('`warn_missing` must be bool.')

        if isinstance(warn_rep, Angle):
            separation_rep = warn_rep
            warn_rep = True
        elif type(warn_rep ) in [float, int, np.float64]:
            if warn_rep > 0:
                separation_rep = Angle(warn_rep, unit='rad')
                warn_rep = True
            else:
                warn_rep = False
        elif warn_rep:
            raise ValueError(
                    '`warn_rep` must be astropy.coordinates.Angle or float.')

        if isinstance(warn_sep, Angle):
            separation_sep = warn_sep
            warn_sep = True
        elif type(warn_sep ) in [float, int, np.float64]:
            if warn_sep > 0:
                separation_sep = Angle(warn_sep, unit='rad')
                warn_sep = True
            else:
                warn_sep = False
        elif warn_sep:
            raise ValueError(
                    '`warn_sep` must be astropy.coordinates.Angle or float.')

        if not isinstance(n_batch, int) or n_batch < 1:
            raise ValueError("'n_batch' must be integer > 0.")

        # warn about repetitions:
        if warn_rep:
            field_ids, ras, decs = self._warn_repetition(
                    field_ids, ras, decs, separation_rep)

        # warn about large separation from field center:
        if warn_sep:
            field_ids, ras, decs = self._warn_separation(
                    field_ids, ras, decs, separation_sep)

        # add to database:
        self._add_guidestar(field_ids, ras, decs, mag, n_batch=n_batch)

        # warn about fields without guidestars:
        if warn_missing:
            self._warn_missing()

    #--------------------------------------------------------------------------
    def get_guidestars(
            self, field_id=None, active=True, essential_columns_only=False):
        """Get guidestars from database.

        Parameters
        ----------
        field_id : int or None, optional
            If a field ID is given, only guidestars associated with that field
            are returned. If None, all guidestars are returned. The default is
            None.
        active : bool, optional
            If True, only active guidestars are returned. If False, only
            inactive guidestars are returned. If None, all guidestars are
            returned regardless of whether they are active or not. The default
            is True.
        essential_columns_only : bool, optional
            If True and field_id is not None, only the following columns are
            queried: guidestar_id, ra, dec. Otherwise, all columns are queried.

        Raises
        ------
        ValueError
            Raised, if `field_id` is neither int nor None.

        Returns
        -------
        results : list of tuples
            List of guidestars. Each tuple contains the guidestar ID,
            associated field ID, guidestar right ascension in rad, and
            guidestar declination in rad, and a flag whether the guidestar is
            active or not.

        Notes
        -----
        This method calls `_get_by_field_id()` or `_get_all()`
        """

        if isinstance(field_id, int):
            results = self._get_by_field_id(
                    field_id, active=active,
                    essential_columns_only=essential_columns_only)
        elif field_id is None:
           results = self._get_all(active=active)
        else:
            raise ValueError('`field_id` must be int or None.')

        return results

    #--------------------------------------------------------------------------
    def get_fields_missing_guidestar(self):
        """Get IDs of fields that do not have any associated guidestars.

        Returns
        -------
        field_ids : dict
            The key 'none' contains a list of field IDs that have no associated
            guidestars. The key 'inactive' contains a list of field IDs that
            only have inactive guidestars associated.

        Notes
        -----
        This method is called by `_warn_missing()`.
        """

        # query data base:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                    SELECT *
                    FROM (
                    	SELECT f.field_id, SUM(g.active) AS active
                    	FROM Fields AS f
                    	LEFT JOIN Guidestars AS g
                    	ON f.field_id = g.field_id
                    	GROUP BY f.field_id)
                    WHERE (
                    	active = 0
                    	OR active IS NULL)
                    """
            results = self._query(connection, query).fetchall()
            field_ids = {'none': [], 'inactive': []}

            for result in results:
                if result['active'] == 0:
                    field_ids['inactive'].append(result['field_id'])
                elif result['active'] is None:
                    field_ids['none'].append(result['field_id'])
                else:
                    raise ValueError("This should not happen!")

        return field_ids

    #--------------------------------------------------------------------------
    def deactivate(self, guidestar_ids):

        if isinstance(guidestar_ids, int):
            guidestar_ids = [guidestar_ids]
        elif type(guidestar_ids) in [list, tuple]:
            pass
        else:
            raise ValueError("`guidestar_ids` must be int or list.")

        with SQLiteConnection(self.db_file) as connection:
            for guidestar_id in guidestar_ids:
                query = """\
                    UPDATE Guidestars
                    SET active=FALSE
                    WHERE guidestar_id={0}
                    """.format(guidestar_id)
                self._query(connection, query, commit=False)

            connection.commit()

        print(f'Deactivated {len(guidestar_ids)} guidestars.')

    #--------------------------------------------------------------------------
    def info(self, field_id=None):
        """Print info about guidestars.

        Parameters
        ----------
        field_id : int or None, optional
            Field ID. If given, print info about guidestars associated with
            this field. Otherwise, print info about all guidestars. The default
            is None.

        Returns
        -------
        None

        Notes
        -----
        This method calls `get_guidestars()` and
        `get_fields_missing_guidestar()`.
        """

        guidestars = DataFrame(
                self.get_guidestars(field_id=field_id, active=None))

        if guidestars.shape[0]:
            n_tot = guidestars.shape[0]
            n_active = guidestars.loc[:, 'active'].sum()
            n_inactive = n_tot - n_active
            sel = guidestars.loc[:, 'active'] == 1
            n_fields = guidestars.loc[sel, 'field_id'].unique().shape[0]
        else:
            n_tot = 0
            n_active = 0
            n_inactive = 0
            n_fields = 0

        print('============ GUIDESTARS ===========')

        if field_id is not None:
            print(f'Associated with field ID {field_id}')

        print(f'Total:                       {n_tot:6d}')
        print(f'Active:                      {n_active:6d}')
        print(f'Inactive:                    {n_inactive:6d}')

        if field_id is None:
            field_ids = self.get_fields_missing_guidestar()
            n_fields_missing = len(field_ids['none'])
            n_fields_inactive = len(field_ids['inactive'])
            print(f'Fields w. guidestars:        {n_fields:6d}')
            print(f'Fields w/o guidestar:        {n_fields_missing:6d}')
            print(f'Fields w/o active guidestar: {n_fields_inactive:6d}')

        elif not n_active:
            print('WARNING: This field has no (active) guidestars!')

        print('-----------------------------------\n')

#==============================================================================

class ObservationManager(DBManager):
    """Database manager for fields."""

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

        This method calls `_query()` and `_last_insert_id()`.
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
    def _check_for_duplicates(
            self, field_id, exposure, repetitions, filter_name, reactivate_all,
            add_all, skip_active, skip_inactive):
        """Check if observations with the specified parameters exist in
        the database.

        Parameters
        ----------
        field_id : int
            Field ID.
        exposure : float
            Exposure time.
        repetitions : int
            Number of repetitions.
        filter_name : str
            Filter name.
        reactivate_all : bool
            If True, do not ask the user in the case that a matching inactive
            observation exists and re-activate it. Otherwise, ask the user.
        add_all : bool
            If True, do not ask the user in the case that a matching active
            observation exists and add the new one. Otherwise, ask the user.
        skip_active : bool
            If True, do not ask the user in the case that a matching inactive
            observation exists and skip adding the new one. Otherwise, ask the
            user.
        skip_inactive : bool
            If True, do not ask the user in the case that a matching active
            observation exists and skip adding the new one. Otherwise, ask the
            user.

        Returns
        -------
        add : bool
            Decision whether or the observation should be added.
        reactivate_all : bool
            In next calls of this method, re-activate inactive observations,
            instead of asking the user.
        add_all : bool
            In next calls of this method, add new observations even if active
            duplicates exists, instead of asking the user.
        skip_active : bool
            In next calls of this method, skip observations for which inactive
            duplicates exist, instead of asking the user.
        skip_inactive : bool
            In next calls of this method, skip observations for which active
            duplicates exist, instead of asking the user.

        Notes
        -----
        This method is called by `add_observations()`.
        """

        observations = self.get_observations(
                field_id=field_id, exposure=exposure, repetitions=repetitions,
                filter_name=filter_name, done=False, active=None)

        # no duplicates exist:
        if not len(observations):
            return True, reactivate_all, add_all, skip_active, skip_inactive

        # duplicates exist - prepare user interaction:
        observations = DataFrame(observations)
        n_active = observations['active'].sum()
        n_inactive = observations.shape[0] - n_active

        add = False
        reactivate = False
        activate_observation_id = None

        # active duplicates exists and shall be skipped:
        if n_active and skip_active:
            pass

        # active and inactive duplicates exists, ask whether to reactivate:
        elif n_active and n_inactive and not skip_inactive:
            userin = input(
                    f"{n_active} active and {n_inactive} unfinished " \
                    "observations with the same parameters already exist in " \
                    "data base. " \
                    "Reactivate one observation? (y/n, or type 'ALL' to " \
                    "reactivate in all following cases without asking, or " \
                    "'NONE' to skip all following cases).")

            if userin.lower() in ['y', 'yes', 'make it so!']:
                reactivate = True
            elif userin.lower() == 'all':
                reactivate = True
                reactivate_all = True
            elif userin.lower() == 'none':
                skip_inactive = True
            else:
                pass

        # active duplicates exist, ask whether to add or not:
        elif n_active and not add_all:
            userin = input(
                    f"{n_active} unfinished observation(s) with the same " \
                    "parameters already exist in data base. " \
                    "Add new observation anyway? (y/n, or type 'ALL' to add " \
                    "all following cases without asking, or 'NONE' to skip " \
                    "all following cases).")

            if userin.lower() in ['y', 'yes', 'make it so!']:
                add = True
            elif userin.lower() == 'all':
                add = True
                add_all = True
            elif userin.lower() == 'none':
                skip_active = True
            else:
                pass

        # inactive duplicates exist, ask whether to reactivate or not:
        elif n_inactive and not skip_inactive:
            userin = input(
                    f"{n_inactive} deactivated, unfinished observation(s) " \
                    "with the same parameters already exist in data base. " \
                    "Reactivate one observation? (y/n, or type 'ALL' to " \
                    "reactivate in all following cases without asking, or " \
                    "'NONE' to skip all following cases).")

            if userin.lower() in ['y', 'yes', 'make it so!']:
                add = True
            elif userin.lower() == 'all':
                add = True
                reactivate_all = True
                sel = observations['active'] == 0
                activate_observation_id = \
                        observations.loc[sel, 'observation_id'].values[0]
            elif userin.lower() == 'none':
                skip_inactive = True
            else:
                pass

        else:
            add = True

        # reactivate observation:
        if reactivate:
            sel = observations['active'] == 0
            activate_observation_id = \
                    int(observations.loc[sel, 'observation_id'].values[0])
            self.activate(activate_observation_id)

        return add, reactivate_all, add_all, skip_active, skip_inactive

    #--------------------------------------------------------------------------
    def _input_observations(
            self, observation_id, field_id, exposure, repetitions, filter_name,
            observed=None, scheduled=None):
        """Parse the argument inputs for the `set_observed()` and
        `set_scheduled()` methods.

        Parameters
        ----------
        observation_id : int or list or None, optional
            ID of an observation or multiple IDs of observations. If no IDs are
            given, observations need to be identified via the next four
            arguments. The default is None.
        field_id : int or None, optional
            Field ID. This and the next three arguments need to be provided to
            identify an observation uniquely. These four arguements only work
            in combination. Alternatively, use the `observationd_id`. The
            default is None.
        exposure : float or None, optional
            Exposure time in seconds. The default is None.
        repetitions : int or None, optional
            Number of repetitions. The default is None.
        filter_name : str or None, optional
            Filter name. The default is None.
        observed : bool or None, optional
            Needs to be True if observations are meant to be marked as
            observed and False if observations are meant to be marked as not
            observed. Otherwise, None. The default is None.
        scheduled : bool or None, optional
            Needs to be True if observations are meant to be marked as
            scheduled and False if observations are meant to be marked as not
            scheduled. Otherwise, None. The default is None.

        Raises
        ------
        ValueError
            Raised, if `observation_id` is neither int nor list.
            Raised, if `observation_id` is None and at least one of `field_id`,
            `exposure`, `repetitions`, and `filter_name` is also None.

        Returns
        -------
        observation_ids_checked : list
            List of observation IDs that will be set to (not)
            observed/scheduled.
        keep : list
            List of IDs to update the observation dates according to the
            observation selection made here.
        """

        if observed is True:
            not_observed = False
            not_scheduled = None
        elif observed is False:
            not_observed = True
            not_scheduled = None
        elif scheduled is True:
            not_observed = None
            not_scheduled = False
        elif scheduled is False:
            not_observed = None
            not_scheduled = True
        else:
            raise ValueError(
                'Either `oberved` or `scheduled` must be True or False.')

        observation_ids_checked = []
        keep = []

        # observation ID is provided:
        if observation_id is not None:
            # check input:
            if isinstance(observation_id, int):
                observation_ids = [observation_id]
            elif type(observation_id) in [list, tuple]:
                observation_ids = observation_id
            else:
                raise ValueError('`observation_id` must be int or list.')

            # check observations:
            for i, observation_id in enumerate(observation_ids):
                try:
                    observation = self.get_observations(
                            observation_id=observation_id)[0]
                except IndexError:
                    raise ValueError(
                            f'Observation ID {observation_id} does not '\
                            'exist in database.')

                # check if any observations are already marked as un/observed:
                if scheduled is None:
                    if observation['active'] == 0:
                        print(f'WARNING: Observation ID {observation_id} is ' \
                              'inactive. Cannot set inactive observation to ' \
                              'observed. Skipped!')

                    elif observation['done'] == observed:
                        if observed:
                            print('Note: observation with ID ' \
                                  f'{observation_id} is already marked as ' \
                                  'observed. Nothing will change.')
                        else:
                            print('Note: observation with ID ' \
                                  f'{observation_id} is already marked as ' \
                                  'not observed. Nothing will change.')

                    else:
                        observation_ids_checked.append(observation_id)
                        keep.append(i)

                # check if any observations are already marked as un/scheduled:
                else:
                    if observation['active'] == 0:
                        print(f'WARNING: Observation ID {observation_id} is ' \
                              'inactive. Cannot set inactive observation to ' \
                              'scheduled. Skipped!')

                    elif observation['scheduled'] == scheduled:
                        if scheduled:
                            print('Note: observation with ID ' \
                                  f'{observation_id} is already marked as ' \
                                  'scheduled. Nothing will change.')
                        else:
                            print('Note: observation with ID ' \
                                  f'{observation_id} is already marked as ' \
                                  'not scheduled. Nothing will change.')

                    else:
                        observation_ids_checked.append(observation_id)
                        keep.append(i)

        # observation ID is not provided:
        else:
            # check input:
            if (field_id is None or exposure is None or repetitions is None \
                or filter_name is None):
                raise ValueError(
                        'Either provide `observation_id` or provide all of ' \
                        'the following: `field_id`, `exposure`, ' \
                        '`repetitions`, and `filter_name`.')

            # check if observation is uniquely identified:
            observations = self.get_observations(
                field_id=field_id, exposure=exposure, repetitions=repetitions,
                filter_name=filter_name, done=not_observed,
                scheduled=not_scheduled, active=True)

            if len(observations) == 0:
                if observed is not None and observed:
                    print('WARNING: There are no unfinished observations ' \
                          'with the given parameters. Nothing will be ' \
                          'changed.')
                elif observed is not None:
                    print('WARNING: There are no finished observations ' \
                          'with the given parameters. Nothing will be ' \
                          'changed.')
                elif scheduled is not None and scheduled:
                    print('WARNING: There are no observations with the ' \
                          'given parameters that are not schedule. Nothing ' \
                          'will be changed.')
                else:
                    print('WARNING: There are no scheduled observations ' \
                          'with the given parameters. Nothing will be ' \
                          'changed.')

            elif len(observations) == 1:
                observation_ids_checked.append(
                        observations[0]['observation_id'])

            else:
                observation_ids_checked.append(
                        observations[0]['observation_id'])
                ids = [str(obs['observation_id']) for obs in observations]
                ids = ', '.join(ids)
                print('Note: There are multiple observations that match the ' \
                      f'given parameters. IDs: {ids}. The first one will be ' \
                      'changed.')

        return observation_ids_checked, keep

    #--------------------------------------------------------------------------
    def _parse_date(self, date):
        """

        Parameters
        ----------
        date : str or list
            Date and time of an observation in the format
            "YYYY-MM-DD hh:mm:ss.s".

        Raises
        ------
        ValueError
            Raised, if the string format cannot be parsed as astropy.time.Time.

        Returns
        -------
        date_str : str
            Date and time as str.
        """

        try:
            date_str = Time(date).iso

        except ValueError:
            raise ValueError(
                "String format format for `date` must be " \
                "'YYYY-MM-DD hh:mm:ss'.")

        return date_str

    #--------------------------------------------------------------------------
    def _input_date(self, date, n, keep):
        """Parse the date argument input for the `set_observed()` method.

        Parameters
        ----------
        date : str or list
            Date and time of an observation in the format
            "YYYY-MM-DD hh:mm:ss.s".
        n : int
            Number of corresponding observations.
        keep : list
            IDs of items that should be keep, as returned from
            `_input_observations()`. All items with no corresponding ID in this
            list will be removed.

        Raises
        ------
        ValueError
            Raised, if date is neither str nor list.

        Returns
        -------
        date_str : list
            List of date strings.
        """

        # multiple dates provided as list:
        if type(date) in [list, tuple]:

            # remove dates whose corresponding observations have been removed:
            date = [d for i, d in enumerate(date) if i in keep]

            # parse dates:
            date_str = [self._parse_date(d) for d in date]

        # single date provided as str:
        elif isinstance(date, str):
            # parse and duplicate date:
            date_str = [self._parse_date(date)]*n

        # no date provided:
        elif date is None:
            userin = input(
                    'No observation date and time specified. Use current ' \
                    'date and time? (y/n) ')

            if userin.lower() in ['y', 'yes', 'make it so!']:
                date_str = Time.now().iso
                date_str = [date_str]*n
            else:
                print('Aborted! Specify the observation date and time with '
                      'the next method call.')
                date_str = False

        # invalid input:
        else:
            raise ValueError('`date` must be string or list of strings.')

        return date_str

    #--------------------------------------------------------------------------
    def add_observations(
            self, exposure, repetitions, filter_name, field_id=None,
            telescope=None, check_for_duplicates=True, n_batch=1000):
        """Add observation to database.

        Parameters
        ----------
        exposure : float or list of float
            Exposure time in seconds.
        repetitions : int or list of int
            Number of repetitions.
        filter_name : str or list of str
            Filter name.
        field_id : int or list of int, optional
            ID(s) of the associated field(s). If None, the same observation is
            added to all active fields. The default is None.
        telescope : str, optional
            If field_id is None, this argument can be used to add observations
            only to those fields that are associated to the specified
            telescope. Otherwise, observations are added to all active
            fields. The default is None.
        check_for_duplicates, optional
            If True, before adding any new observation it is checked whether an
            active observation with the same parameters exists in the database.
            If that is the case, the user us asked whether or not to add the
            new observation anyway. This option may significantly increase the
            time to run this method. To skip the checks, this option should be
            set to False. The default is True.
        n_batch : int, optinal
            Add observations in batches of this size to the data base. The
            default is 1000.

        Returns
        -------
        None

        Notes
        -----
        This method uses `FieldManager()`. This method calls `get_filter_id()`,
        `_check_for_duplicates()`, and `get_observations()`.
        """

        # prepare field IDs:
        if field_id is None:
            field_manager = FieldManager(self.db_file)
            field_id = [field['field_id'] for field in
                        field_manager.get_fields(
                            telescope=telescope, active=True)]
        elif isinstance(field_id, int):
            field_id = [field_id]
        elif isinstance(field_id, list):
            pass
        else:
            raise ValueError(
                    "'field_id' needs to be int, list of int, or None.")

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

        reactivate_all = False
        add_all = False
        skip_active = False
        skip_inactive = False
        filter_ids = {}

        n_obs = len(field_id)
        n_iter = ceil(n_obs / n_batch)

        with SQLiteConnection(self.db_file) as connection:
            # iterate through batches:
            for i in range(n_iter):
                j = i * n_batch
                k = (i + 1) * n_batch
                print(
                    '\rAdding observations {0}-{1} of {2} ({3:.1f}%)..'.format(
                            j, k-1 if k <= n_obs else n_obs, n_obs,
                            i*100./n_iter),
                    end='')

                data = []

                # prepare entries for adding:
                for field, exp, rep, filt in zip(
                        field_id[j:k], exposure[j:k], repetitions[j:k],
                        filter_name[j:k]):

                    # check if filter exists:
                    if filt not in filter_ids.keys():
                        filter_id = self.get_filter_id(filt)

                        # stop, if the filter does not exist and is not added:
                        if filter_id is False:
                            print("Filter was not added to database. No " \
                                  "observations are added either.")
                            return False

                        filter_ids[filt] = filter_id

                    # check if observation entry exists:
                    if check_for_duplicates:
                        add, reactivate_all, add_all, skip_active, \
                        skip_inactive = self._check_for_duplicates(
                                    field, exp, rep, filt, reactivate_all,
                                    add_all, skip_active, skip_inactive)
                    else:
                        add = True

                    # add to data:
                    if add:
                        data.append((field, exp, rep, filter_ids[filt], False,
                                     False, True))

                # add to data base:
                query = """\
                    INSERT INTO Observations (
                        field_id, exposure, repetitions, filter_id, done,
                        scheduled, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """
                self._query(connection, query, many=data, commit=True)

        print(f"\r{n_obs} observation(s) added to data base.                 ")

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
        This method calls `_add_filter()`.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT filter_id, filter
                FROM Filters
                WHERE filter='{0}';
                """.format(filter_name)
            result = self._query(connection, query).fetchone()

        if not result:
            userin = input(
                    f"Filter '{filter_name} does not exist. Add it to data " \
                    "base? (y/n)")

            if userin.lower() in ['y', 'yes', 'make it so!']:
                filter_id = self._add_filter(filter_name)
            else:
                filter_id = False

        else:
            filter_id = result['filter_id']

        return filter_id

    #--------------------------------------------------------------------------
    def get_observations(
            self, observation_id=None, field_id=None, exposure=None,
            repetitions=None, filter_name=None, done=None, scheduled=None,
            active=True, essential_columns_only=False):
        """Query an observation from the database.

        Parameters
        ----------
        observation_id : int or None, optional
            If given, the observation with this ID is returned. All other
            arguments are irrelevant and ignored. If None, use the other
            arguments to specify which observations should be returned. The
            default is None.
        field_id : int or None, optional
            ID of the associated field. The default is None.
        exposure : float or None, optional
            Exposure time in seconds. The default is None.
        repetitions : int or None, optional
            Number of repetitions. The default is None.
        filter_name : str or None, optional
            Filter name.
        done : bool or None, optional
            If True, only search for finished observations. If False, only
            search for unfinished observations. If None, search for
            observations regardless of whether they are finished or not. The
            default is None.
        scheduled : bool or None, optional
            If True, only search for scheduled observations. If False, only
            search for non-scheduled observations. If None, search for
            observations regardless of whether they are scheduled or not. The
            default is None.
        done : bool or None, optional
            If True, only search for finished observations. If False, only
            search for observations not yet finished. If None, search for
            observations regardless of whether they are finished or not. The
            default is None.
        scheduled : bool or None, optional
            If True, only search for scheduled observations. If False, only
            search for observations not scheduled. If None, search for
            observations regardless of whether they are scheduled or not. The
            default is None.
        active : bool or None, optional
            If True, only search for active observations. If False, only search
            for inactive observations. If None, search for observations
            regardless of whether they are active or not. The default is True.
        essential_columns_only : bool, optional
            If True and field_id is not None, only the following columns are
            queried: observation_id, exposure, repetitions, filter. Otherwise,
            all columns are queried. The default is False.

        Returns
        -------
        results : list of dict
            Each list item is a dict with observation parameters. The list is
            empty if no observation was found matching the criteria.
        """

        # define which columns to query:
        if field_id is not None and essential_columns_only:
            columns = 'observation_id, exposure, repetitions, filter'
        else:
            columns = 'observation_id, field_id, exposure, repetitions, ' \
                    'filter, done, scheduled, active, date_done'

        # define SQL WHERE clause, when observation ID is provided:
        if observation_id is not None:
            where_clause = f'WHERE observation_id = {observation_id}'

        # define SQL WHERE conditions, when no observation ID is provided:
        else:
            where_clauses = []

            if field_id is None:
                pass
            elif isinstance(field_id, int):
                where_clause = f'field_id = {field_id}'
                where_clauses.append(where_clause)
            else:
                raise ValueError("`field_id` must be int or None.")

            if exposure is None:
                pass
            elif type(exposure) in [float, int]:
                where_clause = f'exposure = {exposure}'
                where_clauses.append(where_clause)
            else:
                raise ValueError("`exposure` must be float or None.")

            if repetitions is None:
                pass
            elif isinstance(repetitions, int):
                where_clause = f'repetitions = {repetitions}'
                where_clauses.append(where_clause)
            else:
                raise ValueError("`repetitions` must be int or None.")

            if filter_name is None:
                pass
            elif isinstance(filter_name, str):
                where_clause = f"filter = '{filter_name}'"
                where_clauses.append(where_clause)
            else:
                raise ValueError("`filter_name` must be str or None.")

            if active is None:
                pass
            elif active:
                where_clause = 'active = TRUE'
                where_clauses.append(where_clause)
            else:
                where_clause = 'active = FALSE'
                where_clauses.append(where_clause)

            if done is None:
                pass
            elif done:
                where_clause = 'done = TRUE'
                where_clauses.append(where_clause)
            else:
                where_clause = 'done = FALSE'
                where_clauses.append(where_clause)

            if scheduled is None:
                pass
            elif scheduled:
                where_clause = 'scheduled = TRUE'
                where_clauses.append(where_clause)
            else:
                where_clause = 'scheduled = FALSE'
                where_clauses.append(where_clause)

            if where_clauses:
                where_clause = ' AND '.join(where_clauses)
                where_clause = f'WHERE ({where_clause})'
            else:
                where_clause = ''

        # query:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT {0}
                FROM Observations AS o
                LEFT JOIN Filters AS f
                ON o.filter_id = f.filter_id
                {1}
                """.format(columns, where_clause)
            results = self._query(connection, query).fetchall()

        return results

    #--------------------------------------------------------------------------
    def activate(self, observation_ids):
        """Activate one or multiple observations.

        Parameters
        ----------
        observation_ids : int
            ID of the observation that should be set to active.

        Raises
        ------
        ValueError
            Raised, if `observation_id` is neither int nor list.

        Returns
        -------
        None

        Notes
        -----
        This method does not check if the observation(s) corresponding to the
        given ID(s) are actually inactive, before being set to active.
        """

        if isinstance(observation_ids, int):
            observation_ids = [observation_ids]
        elif type(observation_ids) in [list, tuple]:
            pass
        else:
            raise ValueError("`observation_ids` must be int or list.")

        with SQLiteConnection(self.db_file) as connection:
            for observation_id in observation_ids:
                query = """\
                    UPDATE Observations
                    SET active = True
                    WHERE observation_id = {0}
                    """.format(observation_id)
                self._query(connection, query, commit=False)

            connection.commit()

        print(f'Activated {len(observation_ids)} observation(s).')

    #--------------------------------------------------------------------------
    def deactivate(self, observation_ids):
        """Deactivate one or multiple observations.

        Parameters
        ----------
        observation_ids : int
            ID of the observation that should be set to inactive.

        Raises
        ------
        ValueError
            Raised, if `observation_id` is neither int nor list.

        Returns
        -------
        None

        Notes
        -----
        This method does not check if the observation(s) corresponding to the
        given ID(s) are actually active, before being set to inactive.
        """

        if isinstance(observation_ids, int):
            observation_ids = [observation_ids]
        elif type(observation_ids) in [list, tuple]:
            pass
        else:
            raise ValueError("`observation_ids` must be int or list.")

        with SQLiteConnection(self.db_file) as connection:
            for observation_id in observation_ids:
                query = """\
                    UPDATE Observations
                    SET active = FALSE
                    WHERE observation_id = {0}
                    """.format(observation_id)
                self._query(connection, query, commit=False)

            connection.commit()

        print(f'Deactivated {len(observation_ids)} observations.')

    #--------------------------------------------------------------------------
    def set_observed(
            self, observation_id=None, field_id=None, exposure=None,
            repetitions=None, filter_name=None, date=None, observed=True):
        """Mark observations as (not) observed.

        Parameters
        ----------
        observation_id : int or list or None, optional
            ID of an observation or multiple IDs of observations. If no IDs are
            given, observations need to be identified via the next four
            arguments. The default is None.
        field_id : int or None, optional
            Field ID. This and the next three arguments need to be provided to
            identify an observation uniquely. These four arguements only work
            in combination. Alternatively, use the `observationd_id`. The
            default is None.
        exposure : float or None, optional
            Exposure time in seconds. The default is None.
        repetitions : int or None, optional
            Number of repetitions. The default is None.
        filter_name : str or None, optional
            Filter name. The default is None.
        date : str or list or None, optional
            Date and time of the observation to be stored in the database.
            Date format: "YYYY-MM-DD hh:mm:ss.s". If only one date and time is
            given as str, it will be stored for all specified observations.
            Different dates and times can be provided as a list of the same
            length as observation IDs. If no date is given, the user may chose
            to use the current time. The date is only relevant when marking
            observations as observed.
        observed : bool, optional
            If True, mark the specified observation as observed. If False,
            mark the specified observation as not observed. In this case the
            stored observation date is automatically deleted. The latter works
            only with specifying an observation ID. It does not work with the
            other arguments. The default is True.

        Raises
        ------
        ValueError
            Raised, if `scheduled` is False and no observation ID is specified.

        Returns
        -------
        None

        Notes
        -----
        * Multiple observations can be marked only via the `observation_id`
          argument.
        * The method does not check whether the specified observations are
          already marked as (not) observed in the database.
        * Observations marked as observed are automatically marked as not
          scheduled.
        """

        # check input:
        if (type(observation_id) in [list, tuple]) \
                and (type(date) in [list, tuple]):
            if len(observation_id) != len(date):
                raise ValueError(
                        "Number of provided dates does not match number of " \
                        "given dates.")

        if not observed and observation_id is None:
            raise ValueError(
                "To mark an observation as NOT observed use the " \
                "`observation_id`.")

        observed = bool(observed)

        # get/check observation IDs:
        observation_ids, keep = self._input_observations(
                observation_id, field_id, exposure, repetitions, filter_name,
                observed=observed)
        n_obs = len(observation_ids)

        # check date input:
        if observed:
            dates = self._input_date(date, n_obs, keep)

            if dates is False:
                return None

        with SQLiteConnection(self.db_file) as connection:
            # set observations to not observed:
            if not observed:
                # iterate though observations:
                for observation_id in observation_ids:
                    query = """\
                        UPDATE Observations
                        SET done='{0}', scheduled=0, date_done=NULL
                        WHERE observation_id = {1}
                        """.format(int(observed), observation_id)
                    self._query(connection, query, commit=False)

            # set observations to observed:
            else:
                # iterate though observations:
                for observation_id, date in zip(observation_ids, dates):
                    query = """\
                        UPDATE Observations
                        SET done='{0}', scheduled=0, date_done='{1}'
                        WHERE observation_id = {2}
                        """.format(int(observed), date, observation_id)
                    self._query(connection, query, commit=False)

            connection.commit()

        if n_obs:
            status = 'finished' if observed else 'not finished'
            ids = ', '.join([str(obs_id) for obs_id in observation_ids])
            print(f'{n_obs} observations were marked as {status}. IDs: {ids}.')

    #--------------------------------------------------------------------------
    def set_scheduled(
            self, observation_id=None, field_id=None, exposure=None,
            repetitions=None, filter_name=None, scheduled=True):
        """Mark observations as (not) scheduled.

        Parameters
        ----------
        observation_id : int or list or None, optional
            ID of an observation or multiple IDs of observations. If no IDs are
            given, observations need to be identified via the next four
            arguments. The default is None.
        field_id : int or None, optional
            Field ID. This and the next three arguments need to be provided to
            identify an observation uniquely. These four arguements only work
            in combination. Alternatively, use the `observationd_id`. The
            default is None.
        exposure : float or None, optional
            Exposure time in seconds. The default is None.
        repetitions : int or None, optional
            Number of repetitions. The default is None.
        filter_name : str or None, optional
            Filter name. The default is None.
        scheduled : bool, optional
            If True, mark the specified observation as scheduled. If False,
            mark the specified observation as not scheduled. The latter works
            only with specifying an observation ID. It does not work with the
            other arguments. The default is True.

        Raises
        ------
        ValueError
            Raised, if `scheduled` is False and no observation ID is specified.

        Returns
        -------
        None

        Notes
        -----
        * Multiple observations can be marked only via the `observation_id`
          argument.
        * The method does not check whether the specified observations are
          already marked as (not) scheduled in the database.
        """

        if not scheduled and observation_id is None:
            raise ValueError(
                "To mark an observation as NOT observed use the " \
                "`observation_id`.")

        scheduled = bool(scheduled)

        # get/check observation IDs:
        observation_ids, keep = self._input_observations(
                observation_id, field_id, exposure, repetitions, filter_name,
                scheduled=scheduled)
        n_obs = len(observation_ids)

        with SQLiteConnection(self.db_file) as connection:
            # iterate though observations:
            for observation_id in observation_ids:
                query = """\
                    UPDATE Observations
                    SET scheduled='{0}'
                    WHERE observation_id = {1}
                    """.format(int(scheduled), observation_id)
                self._query(connection, query, commit=False)

            connection.commit()

        if n_obs:
            status = 'scheduled' if scheduled else 'not scheduled'
            ids = ', '.join([str(obs_id) for obs_id in observation_ids])
            print(f'{n_obs} observations were marked as {status}. IDs: {ids}.')

    #--------------------------------------------------------------------------
    def info(self):
        """Print infos about observations.

        Returns
        -------
        None
        """

        observations = self.get_observations(active=None)
        observations = DataFrame(observations)

        n_tot = observations.shape[0]
        n_active = observations['active'].sum()
        n_inactive = n_tot - n_active
        active = observations['active'] == 1
        n_done = observations.loc[active, 'done'].sum()
        n_pending = n_active - n_done

        print('=========== OBSERVATIONS ==========')
        print(f'Total:     {n_tot:24d}')
        print(f'Active:    {n_active:24d}')
        print(f'Inactive:  {n_inactive:24d}')
        print(f'Finished*: {n_done:24d}')
        print(f'Pending*:  {n_pending:24d}')
        print('* of those that are active.')
        print('-----------------------------------\n')

#==============================================================================

class ObservabilityManager(DBManager):
    """Database manager for observabilities and observing windows."""

    #--------------------------------------------------------------------------
    def init_observability_jd(self, field_ids, parameter_set_id, jd):
        """Set JD of first observing window calculation for new fields.

        Parameters
        ----------
        field_ids : list of int
            List of field IDs.
        parameter_set_id : int
            Parameter set ID to store with the time range entries.
        jd : float
            JD of the first observability calculation for the fields with
            IDs given by the first argument.

        Returns
        -------
        None
        """

        # prepare data to write:
        data = []

        for field_id in field_ids:
            data.append((field_id, parameter_set_id, jd, True))

        # write to database:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                INSERT INTO TimeRanges (
                    field_id, parameter_set_id, jd_first, active
                )
                VALUES (?, ?, ?, ?)
                """
            self._query(connection, query, many=data, commit=True)

    #--------------------------------------------------------------------------
    def add_observability(
            self, parameter_set_ids, field_ids, dates, status, active=True):
        """Add observability for a specific field and date to database.

        Parameters
        ----------
        parameter_set_ids : list of int
            Parameter set ID that the observability calculation is based on.
        field_ids : list of int
            ID of the field that the observability was calculated for.
        dates : list of float
            JD of the observability calculation.
        status : list of str
            Observability status. Each item can be 'init' or 'not observable'.
        active : bool, optional
            If True, the observing windows is added as active, as inactive
            otherwise. The default is True.

        Returns
        -------
        None
        """

        # status to status ID conversion:
        status_to_id = self.status_to_id_converter()

        # prepare data:
        data = []

        for field_id, par_id, date, stat in zip(
                field_ids, parameter_set_ids, dates, status):
            data.append((
                    field_id, par_id, date, status_to_id[stat], active))

        # add observation windows to database:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                INSERT INTO Observability (
                    field_id, parameter_set_id, jd, status_id, active)
                VALUES (?, ?, ?, ?, ?);
                """
            self._query(connection, query, many=data, commit=True)

    #--------------------------------------------------------------------------
    def add_obs_windows(
            self, field_ids, observability_ids, dates_start, dates_stop,
            duration, active=True):
        """Add observation window for a specific field to database.

        Parameters
        ----------
        field_ids : int or list of int
            ID of the field that the observation window is associated with.
        observability_ids : list of int
            IDs of the related entries in Observability table.
        dates_start : astropy.time.Time or list of Time-instances
            Start date and time of the observing window.
        dates_stop : astropy.time.Time or list of Time-instances
            Stop date and time of the observing window.
        duration : astropy.time.TimeDelta
            Duration of the observing window in days.
        active : bool, optional
            If True, the observing windows is added as active, as inactive
            otherwise. The default is True.

        Returns
        -------
        None
        """

        # check input:
        if isinstance(field_ids, int):
            field_ids = [field_ids]
            dates_start = [dates_start]
            dates_stop = [dates_stop]

        # prepare data:
        data = []

        for field_id, observability_id, date_start, date_stop, dur in zip(
                field_ids, observability_ids, dates_start, dates_stop,
                duration):
            data.append((
                    field_id, observability_id, date_start.iso, date_stop.iso,
                    dur.value, active))

        # add observation windows to database:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                INSERT INTO ObsWindows (
                    field_id, observability_id, date_start, date_stop,
                    duration, active)
                VALUES (?, ?, ?, ?, ?, ?);
                """
            self._query(connection, query, many=data, commit=True)

    #--------------------------------------------------------------------------
    def update_time_ranges(self, field_ids, jds):
        """Update TimeRanges table with new JDs.

        Parameters
        ----------
        field_id : int or list of int
            ID of the associated field.
        jd : float or list of floar
            Julian date of the next day for which the next observability
            needs to be calculated for the specified fields.

        Returns
        -------
        None
        """

        # check input:
        if isinstance(field_ids, int):
            field_ids = [field_ids]

        if type(jds) in [float, int]:
            jds = [jds]*len(field_ids)

        with SQLiteConnection(self.db_file) as connection:
            # iterate though fields:
            for field_id, jd in zip(field_ids, jds):
                query = """\
                    UPDATE TimeRanges
                    SET jd_next='{0}'
                    WHERE (
                        field_id={1}
                        AND active=1)
                    """.format(jd, field_id)
                self._query(connection, query, commit=False)

            connection.commit()

    #--------------------------------------------------------------------------
    def update_setting_durations(
            self, observability_ids, setting_durations):
        """Update observability status.

        Parameters
        ----------
        observability_ids : list of int
            Observability IDs where database needs to be updated.
        setting_durations : list of float
            Durations in days until which a field will set.

        Returns
        -------
        None
        """

        # check input:
        if isinstance(observability_ids, int):
            observability_ids = [observability_ids]
            setting_durations = [setting_durations]

        with SQLiteConnection(self.db_file) as connection:
            # iterate though entries:
            for observability_id, setting_duration in zip(
                    observability_ids, setting_durations):
                query = """\
                    UPDATE Observability
                    SET setting_duration={0}
                    WHERE observability_id={1};
                    """.format(setting_duration, observability_id)
                self._query(connection, query, commit=False)

            connection.commit()

    #--------------------------------------------------------------------------
    def get_next_observability_jd(self):

        with SQLiteConnection(self.db_file) as connection:
            query = """
            SELECT MIN(jd_next) AS jd
            FROM TimeRanges
            WHERE active=1
            """
            jd = self._query(connection, query).fetchone()['jd']

        return jd

    #--------------------------------------------------------------------------
    def get_next_observability_id(self):

        with SQLiteConnection(self.db_file) as connection:
            query = """
            SELECT MAX(observability_id) AS id
            FROM Observability
            """
            highest_id = self._query(connection, query).fetchone()['id']
            next_id = 1 if highest_id is None else highest_id + 1

        return next_id

    #--------------------------------------------------------------------------
    def status_to_id_converter(self):
        """Get dict that converts status to status ID.

        Returns
        -------
        status_to_id : dict
            Dict with statuses as keys and corresponnding IDs as values.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT status_id, status
                FROM ObservabilityStatus;
                """
            results = self._query(connection, query).fetchall()

            status_to_id = {}

            for result in results:
                status_to_id[result['status']] = result['status_id']

        return status_to_id

    #--------------------------------------------------------------------------
    def get_obs_window_durations(self, field_id, jd_start, jd_stop):
        """Query the durations of observing window from the database for a
        specified field between a start and a stop date.

        Parameters
        ----------
        field_id : int
            ID of the field whose observing windows are queried.
        jd_start : float
            Query observabilities equal to or later than this time.
        jd_stop : float
            Query observabilities equal to or earlier than this time.

        Returns
        -------
        results : list of tuples
            Each tuple consists of the ID of the observability, the
            corresponding MJD, and the duration of the corresponding observing
            window.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """
            SELECT a.observability_id, a.jd, b.status, a.duration,
                   a.setting_duration
            FROM (
                SELECT o.observability_id, o.jd, o.status_id, ow.duration,
                       o.setting_duration
                FROM Observability o
                LEFT JOIN ObsWindows ow
                ON o.observability_id = ow.observability_id
                WHERE (
                	o.field_id = {0}
                	AND o.jd >= {1}
                	AND o.jd <= {2}
                	AND o.active = 1)
                ) AS a
            LEFT JOIN ObservabilityStatus b
            ON a.status_id = b.status_id
            """.format(field_id, jd_start, jd_stop)
            results = self._query(connection, query).fetchall()

        return results

#==============================================================================

class DBCreator(DBManager):
    """Database manager for creating a new survey planner database."""

    #--------------------------------------------------------------------------
    def _create(self):
        """Create sqlite3 database.

        Returns
        -------
        None
        """

        # create file:
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
                    telescope_id integer
                        REFERENCES Telescopes (telescope_id),
                    active boolean);
                """
            self._query(connection, query, commit=True)
            print("Table 'Fields' created.")

            # create Telescopes table:
            query = """\
                CREATE TABLE Telescopes(
                    telescope_id integer PRIMARY KEY,
                    name char(30),
                    lat float,
                    lon float,
                    height float,
                    utc_offset float);
                """
            self._query(connection, query, commit=True)
            print("Table 'Telescopes' created.")

            # create ParameterSet table:
            query = """\
                CREATE TABLE ParameterSets(
                    parameter_set_id integer PRIMARY KEY,
                    telescope_id integer
                        REFERENCES Telescopes (telescope_id),
                    active bool,
                    date_added date,
                    date_deactivated date);
                """
            self._query(connection, query, commit=True)
            print("Table 'ParameterSets' created.")

            # create Constraints table:
            query = """\
                CREATE TABLE Constraints(
                    constraint_id integer PRIMARY KEY,
                    constraint_name char(30));
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
                    svalue char(30));
                """
            self._query(connection, query, commit=True)
            print("Table 'Parameters' created.")

            # create ParameterNames table:
            query = """\
                CREATE TABLE ParameterNames(
                    parameter_name_id integer PRIMARY KEY,
                    parameter_name char(30));
                """
            self._query(connection, query, commit=True)
            print("Table 'ParameterNames' created.")

            # create Observability table:
            query = """\
                CREATE TABLE Observability(
                    observability_id integer PRIMARY KEY,
                    field_id integer
                        REFERENCES Fields (field_id),
                    parameter_set_id integer
                        REFERENCES ParameterSets (parameter_set_id),
                    jd float,
                    status_id int
                        REFERENCES ObservabilityStatus (status_id),
                    setting_duration float,
                    active bool);
                """

            self._query(connection, query, commit=True)
            print("Table 'Observability' created.")

            # create ObservabilityStatus table:
            query = """\
                CREATE TABLE ObservabilityStatus(
                    status_id integer PRIMARY KEY,
                    status char(14));
                """

            self._query(connection, query, commit=True)
            print("Table 'ObservabilityStatus' created.")

            # create ObsWindows table:
            query = """\
                CREATE TABLE ObsWindows(
                    obswindow_id integer PRIMARY KEY,
                    field_id integer
                        REFERENCES Fields (field_id),
                    observability_id int
                        REFERENCES Observability (observability_id),
                    date_start date,
                    date_stop date,
                    duration float,
                    active bool);
                """
            self._query(connection, query, commit=True)
            print("Table 'ObsWindows' created.")

            # create TimeRanges table:
            query = """\
                CREATE TABLE TimeRanges(
                    time_range_id integer PRIMARY KEY,
                    field_id integer
                        REFERENCES Fields (field_id),
                    parameter_set_id int
                        REFERENCES ParameterSets (parameter_set_id),
                    jd_first float,
                    jd_next float,
                    active bool);
                """
            self._query(connection, query, commit=True)
            print("Table 'TimeRanges' created.")

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
                    done bool,
                    scheduled bool,
                    active bool,
                    date_done date);
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
                    mag float,
                    active bool);
                """
            self._query(connection, query, commit=True)
            print("Table 'Guidestars' created.")

            # create Filters table:
            query = """\
                CREATE TABLE Filters(
                    filter_id integer PRIMARY KEY,
                    filter char(10));
                """
            self._query(connection, query, commit=True)
            print("Table 'Filters' created.")

            # create FieldsObs view:
            query = """\
                CREATE VIEW FieldsObs AS
                SELECT f.field_id, f.fov, f.center_ra, f.center_dec,
                    f.tilt, t.name AS telescope, f.active, r.jd_first,
                    r.jd_next, p.nobs_tot, p.nobs_done,
                    p.nobs_tot - p.nobs_done AS nobs_pending
                FROM Fields AS f
                LEFT JOIN (
                	SELECT field_id, jd_first, jd_next
                	FROM TimeRanges
                	WHERE active = 1) AS r
                ON f.field_id = r.field_id
                LEFT JOIN Telescopes AS t
                ON f.telescope_id = t.telescope_id
                LEFT JOIN (
                	SELECT field_id, SUM(Done) nobs_done, COUNT(*) nobs_tot
                	FROM Observations
                    WHERE active = 1
                	GROUP BY field_id
                	) AS p
                ON f.field_id = p.field_id;
                """
            self._query(connection, query, commit=True)
            print("View 'FieldsObs' created.")

            # create Observable view:
            query = """\
                CREATE VIEW Observable AS
                WITH o AS (
                	SELECT o.observability_id, o.field_id, o.jd, os.status, o.setting_duration
                	FROM Observability AS o
                	LEFT JOIN ObservabilityStatus AS os
                	ON o.status_id = os.status_id
                	WHERE o.active = 1
                	),
                	ow AS (
                	SELECT observability_id, date_start, date_stop, duration
                	FROM ObsWindows
                	WHERE active = 1
                	)
                SELECT *
                FROM o
                LEFT JOIN ow
                ON o.observability_id = ow.observability_id;
                """
            self._query(connection, query, commit=True)
            print("View 'Observable' created.")

            # define constraints:
            query = """\
                INSERT INTO Constraints (constraint_name)
                VALUES
                    ('Twilight'),
                    ('AirmassLimit'),
                    ('ElevationLimit'),
                    ('HourangleLimit'),
                    ('MoonDistance'),
                    ('MoonPolarization'),
                    ('PolyHADecLimit'),
                    ('SunDistance');
                """
            self._query(connection, query, commit=True)
            print("Constraints added to table 'Constraints'.")

            # define status:
            query = """\
                INSERT INTO ObservabilityStatus (status)
                VALUES
                    ('not observable'),
                    ('rising'),
                    ('plateauing'),
                    ('setting');
                """
            self._query(connection, query, commit=True)
            print("Statuses added to table 'ObservabilityStatus'.")

        return None

    #--------------------------------------------------------------------------
    def create(self):
        """Create sqlite3 database.

        Returns
        -------
        create : bool
            True, if a new database file was created. False, otherwise.
        """

        create = False

        # check if file exists:
        if self._db_exists(verbose=0):
            answer = input(
                'Database file exists. Overwrite (y) or cancel (enter)?')

            if answer.lower() in ['y', 'yes', 'make it so!']:
                os.system(f'rm {self.db_file}')
                create = True

        else:
            create = True

        # create file:
        if create:
            self._create()
            print('Database creation finished.')
            print('\nNote: Next you need to add observatories, constraints, '
                  'fields, guidestars, and observations.')

        else:
            print(f"Existing database '{self.db_file}' kept.")

        return create

    #--------------------------------------------------------------------------
    def add_telescope(self, name, lat, lon, height, utc_offset):
        """Add telescope to database.

        Parameters
        ----------
        name : str
            Telescope name. Must be a unique identifier in the database.
        lat : float
            Telescope latitude in radians.
        lon : float
            Telescope longitude in radians.
        height : float
            Telescope height in meters.
        utc_offset : int
            Telescope UTC offset (daylight saving time).

        Returns
        -------
        None
        """

        manager = TelescopeManager(self.db_file)
        manager.add_telescope(name, lat, lon, height, utc_offset)

    #--------------------------------------------------------------------------
    def add_constraints(self, telescope, twilight, constraints=()):
        """Add constraints to database.

        Parameters
        ----------
        telescope : str
            Name of the telescope that the constraints are associated with.
        twilight : float or str
            If str, must be 'astronomical' (-18 deg), 'nautical' (-12 deg),
            'civil' (-6 deg), or 'sunset' (0 deg). Use float otherwise.
        constraints : list or tuple of constraints.Constraint, optional
            The constraints to be added to the database for the specified
            telescope. The default is ().

        Returns
        -------
        None
        """

        manager = TelescopeManager(self.db_file)
        manager.add_constraints(telescope, twilight, constraints=constraints)

    #--------------------------------------------------------------------------
    def add_fields(self, fields, telescope, active=True, n_batch=1000):
        """Add fields to the database.

        Parameters
        ----------
        fields : fieldgrid.FieldGrid
            The fields to add.
        telescope : str
            Name of the telescope associated with the fields.
        active : bool, optional
            If True, fields are added as active, and as inactive otherwise. The
            default is True.
        n_batch : int, optinal
            Add fields in batches of this size to the data base. The default is
            1000.

        Returns
        -------
        None
        """

        manager = FieldManager(self.db_file)
        manager.add_fields(fields, telescope, active=active, n_batch=n_batch)

    #--------------------------------------------------------------------------
    def add_guidestars(
            self, field_ids, ra, dec, mag, warn_missing=True, warn_rep=0,
            warn_sep=0, n_batch=1000):
        """Add new guidestar(s) to the database.

        Parameters
        ----------
        field_ids : int or list of int
            IDs of the fields that the guidestar coordinates correspond to.
        ras : float or list of float
            Guidestar right ascensions in rad.
        decs : float or list of floats
            Guidestar declinations in rad.
        mag : float or list of floats
            Guidestar magnitudes.
        warn_missing : bool, optional
            If True, warn about fields that do not have any associated
            guidestars stored in the database. The default is True.
        warn_rep : float or astopy.coord.Angle, optional
            If a float or Angle larger than 0 is given, the user is warned
            about new guidestars that may be duplicates of existing entries in
            the database. The value in `warn_rep` is the largest separation
            allowed not to be considered a duplicate. A float is interpreted as
            angle in rad. The default is 0.
        warn_sep : float or astopy.coord.Angle, optional
            If a float or Angle larger than 0 is given, the user is warned
            about new guidestars that may be too far off from the corresponding
            field center. The value in `warn_sep` is the largest separation
            allowed. A float is interpreted as angle in rad. The default is 0.
        n_batch : int, optinal
            Add guidestars in batches of this size to the data base. The
            default is 1000.

        Returns
        -------
        None
        """

        manager = GuidestarManager(self.db_file)
        manager.add_guidestars(
                field_ids, ra, dec, mag, warn_missing=warn_missing,
                warn_rep=warn_rep, warn_sep=warn_sep, n_batch=n_batch)

    #--------------------------------------------------------------------------
    def add_observations(
            self, exposure, repetitions, filter_name, field_id=None,
            telescope=None, check_for_duplicates=True, n_batch=1000):
        """Add observation to database.

        Parameters
        ----------
        exposure : float or list of float
            Exposure time in seconds.
        repetitions : int or list of int
            Number of repetitions.
        filter_name : str or list of str
            Filter name.
        field_id : int or list of int, optional
            ID(s) of the associated field(s). If None, the same observation is
            added to all active fields. The default is None.
        telescope : str, optional
            If field_id is None, this argument can be used to add observations
            only to those fields that are associated to the specified
            telescope. Otherwise, observations are added to all active
            fields. The default is None.
        check_for_duplicates, optional
            If True, before adding any new observation it is checked whether an
            active observation with the same parameters exists in the database.
            If that is the case, the user us asked whether or not to add the
            new observation anyway. This option may significantly increase the
            time to run this method. To skip the checks, this option should be
            set to False. The default is True.
        n_batch : int, optinal
            Add observations in batches of this size to the data base. The
            default is 1000.

        Returns
        -------
        None
        """

        manager = ObservationManager(self.db_file)
        manager.add_observations(
                exposure, repetitions, filter_name, field_id=field_id,
                telescope=telescope, check_for_duplicates=check_for_duplicates,
                n_batch=n_batch)

#==============================================================================
