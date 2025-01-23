#!/usr/bin/env python3
"""Run tests of visualizations."""

from astropy.coordinates import Angle
import astropy.units as u
from matplotlib import colors
import matplotlib.pyplot as plt
import os
import platform

from db import ObservationManager
from prioritizer import PrioritizerSkyCoverage, PrioritizerFieldStatus, \
        PrioritizerAnnualAvailability
from surveyplanner import SurveyPlanner
from visualizations import SurveyVisualizer, PriorityVisualizer

#==============================================================================
# CONFIG
#==============================================================================

db_init = 'test_planner.sqlite3'
#db_name = 'test_strategy.sqlite3'
db_name = 'test_visualization.sqlite3'
reset_db = False

telescope = 'Skinakas'
#telescope = 'SAAO'

date = "2024-02-01"
#date = "2024-07-01"

root_figs = f'plots/priorities/{telescope}_{date}/'

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

    # create figure directory:
    if not os.path.isdir(root_figs):
        os.makedirs(root_figs)

    # create survey planner:
    planner = SurveyPlanner(db_name)

    i_test = 0

    # test sky coverage prioritizer (1/3):
    i_test += 1; print(f'\nTest {i_test}')
    prioritizer = PrioritizerSkyCoverage(
            db_name, radius=Angle(10*u.deg), full_sky=False,
            normalize=True)
    planner.set_prioritizer(prioritizer)
    visualizer = SurveyVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            plot_kws={'zorder': 0, 's': 3,
                      'cmap': colors.ListedColormap(
                              [(0, 0, 0), (0.7, 0.7, 0.7)], name='greys')})
    visualizer = PriorityVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            'SkyCoverage', night=date, telescope=telescope,
            ax=ax, cax=cax, plot_kws={'zorder': 1})
    ax.set_title(
            'Prioritizer: SkyCoverage, radius: 10 deg, full sky: False' \
            f'\nDate: {date}',
            y=1.1)
    fig.savefig(f'{root_figs}/SkyCoverage_1_radius10.png')

    # test sky coverage prioritizer (2/3):
    i_test += 1; print(f'\nTest {i_test}')
    prioritizer = PrioritizerSkyCoverage(
            db_name, radius=Angle(20*u.deg), full_sky=False,
            normalize=True)
    planner.set_prioritizer(prioritizer)
    visualizer = SurveyVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            plot_kws={'zorder': 0, 's': 3,
                      'cmap': colors.ListedColormap(
                              [(0, 0, 0), (0.7, 0.7, 0.7)], name='greys')})
    visualizer = PriorityVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            'SkyCoverage', night=date, telescope=telescope,
            ax=ax, cax=cax, plot_kws={'zorder': 1})
    ax.set_title(
            'Prioritizer: SkyCoverage, radius: 20 deg, full sky: False' \
            f'\nDate: {date}',
            y=1.1)
    fig.savefig(f'{root_figs}/SkyCoverage_2_radius20.png')

    # test sky coverage prioritizer (3/3):
    i_test += 1; print(f'\nTest {i_test}')
    prioritizer = PrioritizerSkyCoverage(
            db_name, radius=Angle(20*u.deg), full_sky=True,
            normalize=True)
    planner.set_prioritizer(prioritizer)
    visualizer = SurveyVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            plot_kws={'zorder': 0, 's': 3,
                      'cmap': colors.ListedColormap(
                              [(0, 0, 0), (0.7, 0.7, 0.7)], name='greys')})
    visualizer = PriorityVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            'SkyCoverage', night=date, telescope=telescope,
            ax=ax, cax=cax, plot_kws={'zorder': 1})
    ax.set_title(
            'Prioritizer: SkyCoverage, radius: 20 deg, full sky: True' \
            f'\nDate: {date}',
            y=1.1)
    fig.savefig(f'{root_figs}/SkyCoverage_3_radius20-fullsky.png')

    # test annual availability prioritizer (1/4):
    i_test += 1; print(f'\nTest {i_test}')
    prioritizer = PrioritizerAnnualAvailability(
            db_name, scale=1, normalize=False)
    planner.set_prioritizer(prioritizer)
    visualizer = PriorityVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            'AnnualAvailability', night=date, telescope=telescope)
    ax.set_title(
            'Prioritizer: AnnualAvailability, scaling: 1, normalize: False' \
            f'\nDate: {date}',
            y=1.1)
    fig.savefig(f'{root_figs}/AnnualAvailability_1_scale1.png')

    # test annual availability prioritizer (2/4):
    i_test += 1; print(f'\nTest {i_test}')
    prioritizer = PrioritizerAnnualAvailability(
            db_name, scale=2, normalize=False)
    planner.set_prioritizer(prioritizer)
    visualizer = PriorityVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            'AnnualAvailability', night=date, telescope=telescope)
    ax.set_title(
            'Prioritizer: AnnualAvailability, scaling: 2, normalize: False' \
            f'\nDate: {date}',
            y=1.1)
    fig.savefig(f'{root_figs}/AnnualAvailability_2_scale2.png')

    # test annual availability prioritizer (3/4):
    i_test += 1; print(f'\nTest {i_test}')
    prioritizer = PrioritizerAnnualAvailability(
            db_name, scale=3, normalize=False)
    planner.set_prioritizer(prioritizer)
    visualizer = PriorityVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            'AnnualAvailability', night=date, telescope=telescope)
    ax.set_title(
            'Prioritizer: AnnualAvailability, scaling: 3, normalize: False' \
            f'\nDate: {date}',
            y=1.1)
    fig.savefig(f'{root_figs}/AnnualAvailability_3_scale3.png')

    # test annual availability prioritizer (4/4):
    i_test += 1; print(f'\nTest {i_test}')
    prioritizer = PrioritizerAnnualAvailability(
            db_name, scale=3, normalize=True)
    planner.set_prioritizer(prioritizer)
    visualizer = PriorityVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            'AnnualAvailability', night=date, telescope=telescope)
    ax.set_title(
            'Prioritizer: AnnualAvailability, scaling: 3, normalize: True' \
            f'\nDate: {date}',
            y=1.1)
    fig.savefig(f'{root_figs}/AnnualAvailability_4_scale3_normalized.png')

    # test field status prioritizer (1/3):
    i_test += 1; print(f'\nTest {i_test}')
    prioritizer = PrioritizerFieldStatus(rising=True)
    planner.set_prioritizer(prioritizer)
    visualizer = PriorityVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            'FieldStatus', night=date, telescope=telescope)
    ax.set_title(
            'Prioritizer: FieldStatus, status: rising' \
            f'\nDate: {date}',
            y=1.1)
    fig.savefig(f'{root_figs}/FieldStatus_1_rising.png')

    # test field status prioritizer (2/3):
    i_test += 1; print(f'\nTest {i_test}')
    prioritizer = PrioritizerFieldStatus(plateauing=True)
    planner.set_prioritizer(prioritizer)
    visualizer = PriorityVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            'FieldStatus', night=date, telescope=telescope)
    ax.set_title(
            'Prioritizer: FieldStatus, status: plateauing' \
            f'\nDate: {date}',
            y=1.1)
    fig.savefig(f'{root_figs}/FieldStatus_2_plateauing.png')

    # test field status prioritizer (3/3):
    i_test += 1; print(f'\nTest {i_test}')
    prioritizer = PrioritizerFieldStatus(rising=True, plateauing=True)
    planner.set_prioritizer(prioritizer)
    visualizer = PriorityVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            'FieldStatus', night=date, telescope=telescope)
    ax.set_title(
            'Prioritizer: FieldStatus, status: rising or plateauing' \
            f'\nDate: {date}',
            y=1.1)
    fig.savefig(f'{root_figs}/FieldStatus_3_rising_or_plateauing.png')

    # test field status prioritizer (3/3):
    i_test += 1; print(f'\nTest {i_test}')
    prioritizer = PrioritizerFieldStatus(setting=True)
    planner.set_prioritizer(prioritizer)
    visualizer = PriorityVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            'FieldStatus', night=date, telescope=telescope)
    ax.set_title(
            'Prioritizer: FieldStatus, status: setting' \
            f'\nDate: {date}',
            y=1.1)
    fig.savefig(f'{root_figs}/FieldStatus_3_setting.png')

    # test joint prioritization (1/3):
    i_test += 1; print(f'\nTest {i_test}')
    prioritizer_skycoverage = PrioritizerSkyCoverage(
            db_name, radius=Angle(20*u.deg), full_sky=True,
            normalize=True)
    prioritizer_annual = PrioritizerAnnualAvailability(
            db_name, scale=1, normalize=False)
    prioritizer_status = PrioritizerFieldStatus(rising=True, plateauing=True)
    planner.set_prioritizer(
            prioritizer_skycoverage, prioritizer_annual, prioritizer_status,
            weights=[1, 1, 1], normalize=False)
    visualizer = SurveyVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            plot_kws={'zorder': 0, 's': 3,
                      'cmap': colors.ListedColormap(
                              [(0, 0, 0), (0.7, 0.7, 0.7)], name='greys')})
    visualizer = PriorityVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            'Joint', night=date, telescope=telescope,
            ax=ax, cax=cax, plot_kws={'zorder': 1})
    ax.set_title(
            'Prioritizer: Joint - SkyCoverage (1), Annual availability (1), ' \
            'Field status (1)' \
            f'\nDate: {date}',
            y=1.1)
    fig.savefig(f'{root_figs}/Joint_1.png')

    # test joint prioritization (1/3):
    i_test += 1; print(f'\nTest {i_test}')
    prioritizer_skycoverage = PrioritizerSkyCoverage(
            db_name, radius=Angle(20*u.deg), full_sky=True,
            normalize=True)
    prioritizer_annual = PrioritizerAnnualAvailability(
            db_name, scale=1, normalize=False)
    prioritizer_status = PrioritizerFieldStatus(rising=True, plateauing=True)
    planner.set_prioritizer(
            prioritizer_skycoverage, prioritizer_annual, prioritizer_status,
            weights=[3, 1, 1], normalize=False)
    visualizer = SurveyVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            plot_kws={'zorder': 0, 's': 3,
                      'cmap': colors.ListedColormap(
                              [(0, 0, 0), (0.7, 0.7, 0.7)], name='greys')})
    visualizer = PriorityVisualizer(surveyplanner=planner)
    fig, ax, cax = visualizer.plot(
            'Joint', night=date, telescope=telescope,
            ax=ax, cax=cax, plot_kws={'zorder': 1})
    ax.set_title(
            'Prioritizer: Joint - SkyCoverage (3), Annual availability (1), ' \
            'Field status (1)' \
            f'\nDate: {date}',
            y=1.1)
    fig.savefig(f'{root_figs}/Joint_2.png')
