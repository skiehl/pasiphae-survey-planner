#!/usr/bin/env python3
"""Database interface for the Pasiphae survey planner.
"""

from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from astropy import units as u
from math import ceil
import numpy as np
import os
import sqlite3
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
        last_insert_id = result[0]

        return last_insert_id

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
            parameter_set_id = result[0]

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
            n = self._query(connection, "SELECT CHANGES()").fetchone()[0]

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
                n += self._query(connection, "SELECT CHANGES()").fetchone()[0]
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
            n = self._query(connection, "SELECT CHANGES()").fetchone()[0]

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
            print(f"Telescope '{telescope}' does not exist in database. Use "
                  "TelescopeManager() to manage telescopes or add new ones.")
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

        telescope_id = result[0]

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

        telescope_names = [result[0] for result in results]

        return telescope_names

    #--------------------------------------------------------------------------
    def get_telescope(self, name=None):
        """Get telescope(s) from database.

        Parameters
        ----------
        name : str, optional
            telescope name. If none is given, all telescopes are returned
            as list of dict. The default is None.

        Returns
        -------
        telescopes : dict or list of dict
            If a name is given, a dictionary is returned with the telescope
            data. If no name is given a list is returned where each item is a
            dictionary corresponding to one telescope.
        """
        # TODO: add option to include constraints in returned dict

        if name is None:
            where_clause = ""
        elif isinstance(name, str):
            where_clause = f"WHERE name='{name}'"
        else:
            raise ValueError("`name` must be str or None.")

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT *
                FROM Telescopes
                {0}
                """.format(where_clause)
            results = self._query(connection, query).fetchall()

        telescopes = []

        for result in results:
            telescope = {
                'telescope_id': result[0],
                'name': result[1],
                'lat': result[2] * u.rad,
                'lon': result[3] * u.rad,
                'height': result[4],
                'utc_offset': result[5]}
            telescopes.append(telescope)

        if name is not None:
            telescopes = telescopes[0]

        return telescopes

    #--------------------------------------------------------------------------
    def get_constraints(self, telescope):
        """Query constraints associated with a specified telescope from the
        database

        Parameters
        ----------
        telescope : str
            Name of the telescope.

        Raises
        ------
        NotInDatabase
            Raised if no parameter set is stored for the specified telescope.

        Returns
        -------
        parameter_set_id : int
            Parameter set ID corresponding to the constraints.
        constraints : dict of dict
            Dictionary of the constraints. The keys are the constraint names.
            The values are dictionaries that contain the constraint parameter
            names as keys and associated values.

        Notes
        -----
        This method calls `get_telescope_id()`, `_get_parameter_set_id()`, and
        `_query()`.
        """

        telescope_id = self.get_telescope_id(telescope)
        parameter_set_id = self._get_parameter_set_id(telescope_id)

        # no parameter set exists:
        if parameter_set_id == -1:
            raise NotInDatabase(
                "No active parameter set stored for telescope "
                f"'{telescope}'.")

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
            elif (param_name in constraints[constraint_name] and not
                  isinstance(constraints[constraint_name][param_name], list)):
                constraints[constraint_name][param_name] = [
                        constraints[constraint_name][param_name]]
                constraints[constraint_name][param_name].append(value)
            elif param_name in constraints[constraint_name]:
                constraints[constraint_name][param_name].append(value)
            else:
                constraints[constraint_name][param_name] = value

        return parameter_set_id, constraints

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
            Raised, if 'n_batch' is not integer > 0.

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
    def get_fields(
            self, telescope=None, observed=None, pending=None, active=True,
            needs_obs_windows=None, init_obs_windows=False):
        """Get fields from the database, given various selection criteria.

        Parameters
        ----------
        telescope : str, optional
            Observatory name. If set, only query fields associated with this
            telescope. If None, query fields for all observatories. The
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
        needs_obs_window : float, optional
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
        result : list of tuples
            List of the queried fields. Each tuple contains the field
            parameters.
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
            condition_telescope = " AND name = '{0}'".format(telescope)
        else:
            condition_telescope = ""

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
                	GROUP BY field_id
                	) AS p
                ON f.field_id = p.field_id
                WHERE (active = {0} {1} {2} {3} {4});
                """.format(
                        active, condition_telescope, condition_observed,
                        condition_pending, condition_obswindow)
            result = self._query(connection, query).fetchall()

        return result

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
                    WHERE f.field_id = {0} AND p.field_id = {0};
                    """.format(field_id)
            result = self._query(connection, query).fetchall()

        return result

    #--------------------------------------------------------------------------
    def get_fields_missing_status(self):
        """Query fields from the database that have missing or unknown
        observability status in some entries.

        Returns
        -------
        result : list of tuples
            Each tuple contains the field ID and the corresponding MJD for
            which an observability is stored with unset or unknown status.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """
            SELECT field_id, jd
            FROM Observability
            WHERE (
            	status_id = 1
            	AND active = 1)
            """
            result = self._query(connection, query).fetchall()

        return result

#==============================================================================

class GuidestarManager(DBManager):
    """Database manager for guide stars."""

    #--------------------------------------------------------------------------
    def _get_by_field_id(self, field_id):
        """Get guide stars for a specific field from database.

        Parameters
        ----------
        field_id : int
            Only guidestars associated with that field are returned.

        Returns
        -------
        results : list of tuples
            List of guidestars. Each tuple contains the guidestar ID,
            associated field ID, guidestar right ascension in rad, and
            guidestar declination in rad.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT *
                FROM Guidestars
                WHERE field_id='{0}'
                """.format(field_id)
            results = self._query(connection, query).fetchall()

        return results

    #--------------------------------------------------------------------------
    def _get_all(self):
        """Get all guide stars from database.

        Returns
        -------
        results : list of tuples
            List of guidestars. Each tuple contains the guidestar ID,
            associated field ID, guidestar right ascension in rad, and
            guidestar declination in rad.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT *
                FROM Guidestars
                WHERE active = 1
                """
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

        # get stored guidestar coordinates and associated field IDs:
        stored_gs_id = []
        stored_gs_field_ids = []
        stored_gs_ras = []
        stored_gs_decs =[]

        for guidestar_id, field_id, ra, dec, __ in self._get_all():
            stored_gs_id.append(guidestar_id)
            stored_gs_field_ids.append(field_id)
            stored_gs_ras.append(ra)
            stored_gs_decs.append(dec)

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

        keep = np.ones(len(field_ids), dtype=bool)

        with SQLiteConnection(self.db_file) as connection:
            for i, (field_id, ra, dec) in enumerate(zip(field_ids, ras, decs)):

                # query data base:
                query = """\
                        SELECT center_ra, center_dec
                        FROM Fields
                        WHERE field_id = {0};
                        """.format(field_id)
                field_ra, field_dec \
                        = self._query(connection, query).fetchone()

                # calculate separation:
                field_coord = SkyCoord(field_ra, field_dec, unit='rad')
                guidestar_coord = SkyCoord(ra, dec, unit='rad')
                separation = field_coord.separation(guidestar_coord)

                # ask user about critical cases:
                if separation > limit:
                    print(f'New guide star {i} for field ID {field_id} is too '
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
        results : list
            Field IDs of fields without associated guidestars.
        """

        # query data base:
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                    SELECT f.field_id
                    FROM Fields AS f
                    LEFT JOIN Guidestars AS g
                    ON f.field_id = g.field_id
                    WHERE g.guidestar_id IS NULL;
                    """
            results = [x[0] for x in self._query(connection, query).fetchall()]

            # inform user:
            if results:
                print('\nWARNING: Fields with the following IDs do not have '
                      'any guidestars associated:')
                text = ''

                for field_id in results:
                    text = f'{text}{field_id}, '

                print(text[:-2])

        return results

    #--------------------------------------------------------------------------
    def _add_guidestar(self, field_ids, ras, decs):
        """Add new guidestars to the database.

        Parameters
        ----------
        field_ids : numpy.ndarray
            IDs of the fields that the guidestar coordinates correspond to.
        ras : astropy.coord.Angle
            Guidestar right ascensions.
        decs : astropy.coord.Angle
            Guidestar declinations.

        Returns
        -------
        None
        """

        data = [(int(field_id), float(ra), float(dec), True) \
                for field_id, ra, dec in zip(field_ids, ras.rad, decs.rad)]

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                INSERT INTO Guidestars (
                    field_id, ra, dec, active)
                VALUES (?, ?, ?, ?)
                """
            self._query(connection, query, many=data, commit=True)

        print(f'{len(data)} new guidestars added to database.')

    #--------------------------------------------------------------------------
    def add_guidestar(self, field_ids, ra, dec, warn_missing=True, warn_rep=0,
            warn_sep=0):
        """Add new guidestars to the database.

        Parameters
        ----------
        field_ids : int or list of int
            IDs of the fields that the guidestar coordinates correspond to.
        ras : float or list of float
            Guidestar right ascensions in rad.
        decs : float or list of floats
            Guidestar declinations in rad.
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

        Raises
        ------
        ValueError
            Raised if, `field_ids` is neither int nor list-like.
            Raised if, `ra` is neither float nor list-like.
            Raised if, `dec` is neither float nor list-like.
            Raised if, `warn_missing` is not bool.
            Raised if, `warn_rep` is neither float nor astropy.coord.Angle.
            Raised if, `warn_sep` is neither float nor astropy.coord.Angle.

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

        if not isinstance(warn_missing, bool):
            raise ValueError('`warn_missing` must be bool.')

        if isinstance(warn_rep, Angle):
            separation_rep = warn_rep
            warn_rep = True
        elif type(warn_rep ) in [float, int, np.float64]:
            separation_rep = Angle(warn_rep, unit='rad')
            warn_rep = True
        elif warn_rep:
            raise ValueError(
                    '`warn_rep` must be astropy.coordinates.Angle or float.')

        if isinstance(warn_sep, Angle):
            separation_sep = warn_sep
            warn_sep = True
        elif type(warn_sep ) in [float, int, np.float64]:
            separation_sep = Angle(warn_sep, unit='rad')
            warn_sep = True
        elif warn_sep:
            raise ValueError(
                    '`warn_sep` must be astropy.coordinates.Angle or float.')

        # warn about repetitions:
        if warn_rep:
            field_ids, ras, decs = self._warn_repetition(
                    field_ids, ras, decs, separation_rep)

        # warn about large separation from field center:
        if warn_sep:
            field_ids, ras, decs = self._warn_separation(
                    field_ids, ras, decs, separation_sep)

        # add to database:
        self._add_guidestar(field_ids, ras, decs)

        # warn about fields without guidestars:
        if warn_missing:
            self._warn_missing()

    #--------------------------------------------------------------------------
    def get_guidestar(self, field_id=None):
        """Get guide stars from database.

        Parameters
        ----------
        field_id : int or None, optional
            If a field ID is given, only guidestars associated with that field
            are returned. If None, all guidestars are returned. The default is
            None.

        Raises
        ------
        ValueError
            Raised, if `field_id` is neither int nor None.

        Returns
        -------
        results : list of tuples
            List of guidestars. Each tuple contains the guidestar ID,
            associated field ID, guidestar right ascension in rad, and
            guidestar declination in rad.

        Notes
        -----
        This method calls `_get_by_field_id()` or `_get_all()`
        """

        if isinstance(field_id, int):
            results = self._get_by_field_id(field_id)
        elif field_id is None:
           results = self._get_all()
        else:
            raise ValueError('`field_id` must be int or None.')

        return results

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
    def add_observations(
            self, exposure, repetitions, filter_name, field_id=None,
            telescope=None):
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

        Returns
        -------
        None

        Notes
        -----
        This method uses `FieldManager()`. This method calls `get_filter_id()`,
        `get_observations()`.
        """

        # prepare field IDs:
        if field_id is None:
            field_manager = FieldManager(self.db_file)
            field_id = [field[0] for field in
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

                    if userin.lower() in ['y', 'yes', 'make it so!']:
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
    def get_observations(self, field_id, exposure, repetitions, filter_id):
        # TODO: make all these parameters optional
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
        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT *
                FROM ObservabilityStatus;
                """
            results = self._query(connection, query).fetchall()

        status_to_id = {k: i for i, k in results}

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
    def update_observability_status(
            self, observability_ids, status_ids, setting_in):
        """Update observability status.

        Parameters
        ----------
        observability_ids : list of int
            Observability IDs where database needs to be updated.
        status_ids : list of int
            Status IDs to set in the database.
        setting_in : list of float
            Durations in days until which a field will set.

        Returns
        -------
        None
        """

        # check input:
        if isinstance(observability_ids, int):
            observability_ids = [observability_ids]
            status_ids = [status_ids]
            setting_in = [setting_in]

        with SQLiteConnection(self.db_file) as connection:
            # iterate though entries:
            for observability_id, status_id, setting in zip(
                    observability_ids, status_ids, setting_in):
                if setting is None:
                    query = """\
                        UPDATE Observability
                        SET status_id={0}
                        WHERE observability_id={1};
                        """.format(status_id, observability_id)
                else:
                    query = """\
                        UPDATE Observability
                        SET status_id={0}, setting_duration={1}
                        WHERE observability_id={2};
                        """.format(status_id, setting, observability_id)
                self._query(connection, query, commit=False)

            connection.commit()

    #--------------------------------------------------------------------------
    def get_next_observability_jd(self):

        with SQLiteConnection(self.db_file) as connection:
            query = """
            SELECT MIN(jd_next)
            FROM TimeRanges
            WHERE active=1
            """
            jd = self._query(connection, query).fetchall()[0][0]

        return jd

    #--------------------------------------------------------------------------
    def get_next_observability_id(self):

        with SQLiteConnection(self.db_file) as connection:
            query = """
            SELECT MAX(observability_id)
            FROM Observability
            """
            highest_id = self._query(connection, query).fetchall()[0][0]
            next_id = 1 if highest_id is None else highest_id + 1

        return next_id

    #--------------------------------------------------------------------------
    def get_status(self):
        """Get status ID and corresponding status.

        Returns
        -------
        results : list of tuples
            Each entry contains an ID and its corresponding status string.
        """

        with SQLiteConnection(self.db_file) as connection:
            query = """\
                SELECT status_id, status
                FROM ObservabilityStatus;
                """
            results = self._query(connection, query).fetchall()

        return results

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
            SELECT a.observability_id, a.jd, b.status, a.duration
            FROM (
                SELECT o.observability_id, o.jd, o.status_id, ow.duration
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
                    scheduled bool,
                    done bool,
                    date date);
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
                    ('init'),
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
            self, field_ids, ra, dec, warn_missing=True, warn_rep=0,
            warn_sep=0):
        """Add new guidestars to the database.

        Parameters
        ----------
        field_ids : int or list of int
            IDs of the fields that the guidestar coordinates correspond to.
        ras : float or list of float
            Guidestar right ascensions in rad.
        decs : float or list of floats
            Guidestar declinations in rad.
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

        Returns
        -------
        None
        """

        manager = GuidestarManager(self.db_file)
        manager.add_guidestar(
                field_ids, ra, dec, warn_missing=warn_missing,
                warn_rep=warn_rep, warn_sep=warn_sep)

    #--------------------------------------------------------------------------
    def add_observations(
            self, exposure, repetitions, filter_name, field_id=None,
            telescope=None):
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

        Returns
        -------
        None
        """

        manager = ObservationManager(self.db_file)
        manager.add_observations(
                exposure, repetitions, filter_name, field_id=field_id,
                telescope=telescope)

#==============================================================================
