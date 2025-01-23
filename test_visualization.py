#!/usr/bin/env python3
"""Run tests of visualizations."""

from astropy.coordinates import Angle
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import os
import platform

from db import ObservationManager
from prioritizer import PrioritizerSkyCoverage, PrioritizerFieldStatus, \
        PrioritizerAnnualAvailability
from surveyplanner import SurveyPlanner
from visualizations import SurveyVisualizer, ObservabilityVisualizer, \
        AnnualObservabilityVisualizer, PriorityVisualizer

#==============================================================================
# CONFIG
#==============================================================================

db_init = 'test_planner.sqlite3'
#db_name = 'test_strategy.sqlite3'
db_name = 'test_visualization.sqlite3'
reset_db = False

telescope = None # 'Skinakas'
date = "2024-06-04"
radius = Angle(10 * u.deg)
full_sky = True
availability_scale = 2
normalize = True


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

        manager = ObservationManager(db_name)
        manager.set_observed(observation_id=list(range(1, 6)))

    timeit_start = Time.now()

    # create survey planner:
    planner = SurveyPlanner(db_name)
    """
    # show survey status:
    visualizer = SurveyVisualizer(surveyplanner=planner)
    visualizer.set_fields(planner.query_fields())
    visualizer.plot(s=6)
    plt.show()
    visualizer.plot(galactic=True, projection='mollweide', s=6)
    plt.show()
    visualizer.plot(galactic=True, projection='aitoff', s=6)
    plt.show()
    """

    """
    # show observability:
    visualizer = ObservabilityVisualizer(surveyplanner=planner)
    visualizer.set_fields(planner.query_fields(night=date))
    visualizer.plot(
            'start', night=None, galactic=False, projection='mollweide',
            plot_kws={'s': 6})
    plt.show()
    visualizer.plot(
            'stop', night=date, galactic=True, projection='mollweide',
            plot_kws={'s': 6})
    plt.show()
    visualizer.plot(
            'duration', night=date, galactic=False, projection='aitoff',
            plot_kws={'s': 6})
    plt.show()
    visualizer.plot(
            'setting_duration', night=date, galactic=True, projection='hammer',
            plot_kws={'s': 6})
    plt.show()
    visualizer.plot(
            'status', night=date, galactic=False, projection='lambert',
            plot_kws={'s': 6})
    plt.show()
    """

    """
    # show annual observability:
    visualizer = AnnualObservabilityVisualizer(surveyplanner=planner)
    visualizer.plot(
        rate=True, galactic=True, projection='aitoff', plot_kws={'s': 6})
    plt.show()
    visualizer.plot(
        rate=True, galactic=False, projection='lambert', plot_kws={'s': 6})
    plt.show()
    """


    # add prioritizers:
    prioritizer_coverage = PrioritizerSkyCoverage(
            db_name, radius=radius, full_sky=full_sky, normalize=normalize)
    prioratizer_status = PrioritizerFieldStatus(
            rising=True, plateauing=True, setting=False)
    prioritizer_availability = PrioritizerAnnualAvailability(
            db_name, scale=availability_scale, normalize=normalize)
    planner.set_prioritizer(
            prioritizer_coverage, prioratizer_status, prioritizer_availability,
            weights=[2, 1, 2])
    visualizer = PriorityVisualizer(surveyplanner=planner)
    visualizer.plot('SkyCoverage', night=date, telescope='Skinakas')
    visualizer.plot('FieldStatus', night=date, telescope='Skinakas')
    visualizer.plot('AnnualAvailability', night=date, telescope='Skinakas')
    visualizer.plot('Joint', night=date, telescope='Skinakas')
    plt.show()

    timeit_total = Time.now() - timeit_start
    print('Runtime:', timeit_total.value * 24 * 60, 'min')
