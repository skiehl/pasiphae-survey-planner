#!/usr/bin/env python3
"""Run tests for adding a new parameter set."""

from astropy.time import Time
import astropy.units as u
import os
import platform

import constraints as c
from db import DBConnectorSQLite
from surveyplanner import SurveyPlanner

#==============================================================================
# CONFIG
#==============================================================================

time_interval_init = 600 * u.s
time_interval_refine = 60 * u.s
duration_limit = 5 * u.min
processes = 2
batch_write = 200

date_stop = Time('2024-01-03')
date_start = Time('2024-01-01')

reset_db = False
db_init = 'test_planner_init.sqlite3'
db_name = 'test_planner_temp.sqlite3'

#==============================================================================
# MAIN
#==============================================================================

if __name__ == '__main__':

    if reset_db:
        if platform.system() == 'Linux':
            os.system(f'cp {db_init} {db_name}')
        elif platform.system() == 'Windows':
            os.system(f'xcopy {db_init} {db_name} /y')
        else:
            raise ValueError('Unknown operating system.')

    # add new parameter set:
    db = DBConnectorSQLite(db_name)
    twilight = 'nautical'
    airmass_limit = c.AirmassLimit(2.)
    moon_distance = c.MoonDistance(5.)
    hourangle_limit = c.HourangleLimit(5.33)
    db.add_constraints(
            'Skinakas', twilight,
            constraints=(airmass_limit, hourangle_limit, moon_distance))

    # calculate observabilities:
    planner = SurveyPlanner(db_name)
    planner.check_observability(
            date_stop, date_start=date_start, duration_limit=duration_limit,
            processes=processes, batch_write=batch_write,
            time_interval_init=time_interval_init,
            time_interval_refine=time_interval_refine)
