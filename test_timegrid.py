#!/usr/bin/env python3
"""Run tests with different time grids."""

from astropy.time import Time
import astropy.units as u
import numpy as np
import os
from ObservabilityPlanner import ObservabilityPlanner

__author__ = "Sebastian Kiehlmann"
__credits__ = ["Sebastian Kiehlmann"]
__license__ = "BSD3"
__version__ = "0.1"
__maintainer__ = "Sebastian Kiehlmann"
__email__ = "skiehlmann@mail.de"
__status__ = "Production"

#==============================================================================
# CONFIG
#==============================================================================

date_start = Time('2024-01-01')
date_stop = Time('2024-02-01')
duration_limit = 5 * u.min
processes = 10
batch_write = 10000

time_interval_init = 600
time_interval_refine = 10

db_init = 'test_planner_init.sqlite3'
db_file = f'test_timegrid_{time_interval_init}-{time_interval_refine}.sqlite3'
reset_db = True

log = 'timegrid_tests.dat'

#==============================================================================
# MAIN
#==============================================================================

if reset_db:
    os.system(f'cp {db_init} {db_file}')

time_start = Time.now()
print(f'Start observing window calculation at {time_start}..')

ObservabilityPlanner = ObservabilityPlanner(db_file)
ObservabilityPlanner.check_observability(
        date_stop, date_start=date_start, duration_limit=duration_limit,
        batch_write=batch_write, processes=processes,
        time_interval_init=time_interval_init,
        time_interval_refine=time_interval_refine, days_before=7, days_after=7)

time_done = Time.now()
runtime = time_done - time_start
print(f'Finshed at {time_done} after {runtime}.')

if log:
    with open(log, mode='a') as f:
        n_days = int(np.round((date_stop - date_start).value, 0))
        f.write(f'kallisto,{date_start},{date_stop},{n_days},' \
                f'{time_interval_init},{time_interval_refine},{processes},' \
                f'{batch_write},0,{runtime.value}\n')
