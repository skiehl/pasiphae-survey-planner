#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.time import Time
import os
import platform
from surveyplanner import SurveyPlanner

#==============================================================================
# CONFIG
#==============================================================================

n_tests = 4
processes = 1
version_tag = f't4-{processes}'

# version tags:
# t0: before performance boosts
# t1: batch writing
# t2: constraint calculation improvement
# t3-1: parallelization with redundance reduction (1 core)
# t3-2: " (2 cores)
# t3-4: " (4 cores)
# t4: constraint, frame saving

#==============================================================================
# MAIN
#==============================================================================

if __name__ == '__main__':
    for i in range(n_tests):
        if platform.system() == 'Linux':
            os.system('cp test_planner.sqlite3 test_planner_temp.sqlite3')
        elif platform.system() == 'Windows':
            os.system('xcopy test_planner.sqlite3 test_planner_temp.sqlite3 /y')
        else:
            raise ValueError('Unknown operating system.')

        date_stop = Time('2024-04-22')

        timeit_start = Time.now()
        planner = SurveyPlanner('test_planner_temp.sqlite3')
        planner.add_obs_windows(date_stop, processes=processes, batch_write=1000)
        timeit_total = Time.now() - timeit_start

        with open('runtime_tests.dat', mode='a') as f:
            f.write(f'{version_tag},{i},{timeit_total}\n')
