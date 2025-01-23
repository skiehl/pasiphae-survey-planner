#!/usr/bin/env python3
"""Run tests for the observability calculation."""

from astropy.time import Time
import astropy.units as u
import os
import platform

from surveyplanner import ObservabilityPlanner

#==============================================================================
# CONFIG
#==============================================================================

time_interval_init = 600 * u.s
time_interval_refine = 60 * u.s
duration_limit = 5 * u.min
processes = 10
batch_write = 2000
all_fields = True

date_start = Time('2024-10-01')
date_stop = Time('2025-08-01')

reset_db = False
#db_init = 'test_planner_init.sqlite3'
#db_name = 'test_planner.sqlite3'
#db_init = 'test_planner_min_constr_init.sqlite3'
#db_name = 'test_planner_min_constr.sqlite3'
db_init = 'test_strategy_init.sqlite3'
db_name = 'test_strategy.sqlite3'

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

    timeit_start = Time.now()
    planner = ObservabilityPlanner(db_name)
    planner.observability(
            date_stop, date_start=date_start, duration_limit=duration_limit,
            processes=processes, batch_write=batch_write,
            time_interval_init=time_interval_init,
            time_interval_refine=time_interval_refine, all_fields=all_fields)
    timeit_total = Time.now() - timeit_start
    print('Runtime:', timeit_total.value * 24, 'hours')
