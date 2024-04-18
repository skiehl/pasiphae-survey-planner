#!/usr/bin/env python3
"""Run runtime tests."""

from astropy.time import Time
import astropy.units as u
from itertools import product
import os
import platform
from surveyplanner import ObservabilityPlanner

#==============================================================================
# CONFIG
#==============================================================================

n_tests = 1
processes = 5
batch_write = (100, 200, 500, 1000, 2000)
time_interval_init = 600 # sec
time_interval_refine = 60 # sec

date_stop = Time('2024-01-02')
date_start = Time('2024-01-01')
n_days = int(round((date_stop - date_start).value, 0))

computer = 'kallisto'

reset_db = True
#db_init = 'test_planner_init.sqlite3'
db_init = 'test_strategy_init.sqlite3 '
db_name = 'test_runtime.sqlite3'

save_runtime = 'runtime_tests_5_fullgrid_kallisto_add.dat'
#save_runtime = False

#==============================================================================
# MAIN
#==============================================================================

if __name__ == '__main__':
    # create results file:
    if not os.path.exists(save_runtime):
        with open(save_runtime, mode='w') as f:
            f.write('#computer,date_start,date_stop,n_days,time_interval_init,'
                    'time_interval_refine,processes,batch_write,test_id,'
                    'runtime_total (days)\n')

    # change variable parameters into lists:
    if isinstance(processes, int):
        iter_processes = [processes]
    else:
        iter_processes = list(processes)

    if isinstance(batch_write, int):
        iter_batch_write = [batch_write]
    else:
        iter_batch_write = list(batch_write)

    if type(time_interval_init) not in [list, tuple]:
        iter_time_interval_init = [time_interval_init]
    else:
        iter_time_interval_init = list(time_interval_init)

    if type(time_interval_refine) not in [list, tuple]:
        iter_time_interval_refine = [time_interval_refine]
    else:
        iter_time_interval_refine = list(time_interval_refine)

    # iterate through all combinations of variable parameters:
    for (time_interval_init, time_interval_refine, processes, batch_write) in \
            product(iter_time_interval_init, iter_time_interval_refine,
                    iter_processes, iter_batch_write):

        # iterate through test repetitions:
        for i in range(n_tests):

            if reset_db:
                if platform.system() == 'Linux':
                    os.system(f'cp {db_init} {db_name}')
                elif platform.system() == 'Windows':
                    os.system(f'xcopy {db_init} {db_name} /y')
                else:
                    raise ValueError('Unknown operating system.')

            timeit_start = Time.now()
            planner = ObservabilityPlanner(db_name)
            planner.check_observability(
                    date_stop, date_start=date_start, duration_limit=5*u.min,
                    processes=processes, batch_write=batch_write,
                    time_interval_init=time_interval_init,
                    time_interval_refine=time_interval_refine, all_fields=True)
            timeit_total = Time.now() - timeit_start
            print('Runtime:', timeit_total.value * 24, 'hours')

            if save_runtime:
                with open(save_runtime, mode='a') as f:
                    f.write(f'{computer},{date_start.iso.split(" ")[0]},'
                            f'{date_stop.iso.split(" ")[0]},{n_days},'
                            f'{time_interval_init},{time_interval_refine},'
                            f'{processes},{batch_write},{i},{timeit_total}\n')
