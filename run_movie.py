#!/usr/bin/env python3
"""
Visualize field availability over a year.
"""

from astropy.time import Time
import astropy.units as u
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import os

from surveyplanner import SurveyPlanner
from visualizations import ObservabilityVisualizer

#==============================================================================
# CONFIG
#==============================================================================

db_name = 'test_planner.sqlite3'

dates = Time('2024-01-01') + np.arange(365) * u.d
properties = ['status', 'start', 'stop', 'duration', 'setting_duration']
coord = 'equatorial' # options: equatorial, galactic
projection = 'lambert' # options: mollweide, aitoff, lambert

make_movie = True

dir_figs = 'movie/'
dir_figs = os.path.join(dir_figs, f'{coord}_{projection}/')

#==============================================================================
# MAIN
#==============================================================================

titles = {
        'status': 'Observability status',
        'start': 'Observability window start UTC',
        'stop': 'Observability window stop UTC',
        'duration': 'Observability window duration (hours)',
        'setting_duration': 'Field setting duration (days)'}

planner = SurveyPlanner(db_name)
visualizer = ObservabilityVisualizer(surveyplanner=planner)

# iterate through dates:
for date in dates:
    print(f'Plot date {date.iso[:10]}..')

    fig = plt.figure(figsize=(9.6, 5.4))
    gs = GridSpec(1, 2, width_ratios=[30, 1])
    ax = plt.subplot(gs[0], projection=projection)
    cax = plt.subplot(gs[1])
    plt.subplots_adjust(
            top=0.95, bottom=0.05, right=0.9, left=0.05, wspace=0.05)
    plt.margins(0,0)

    # iterate through properties:
    for prop in properties:
        # clear figure:
        ax.cla()
        cax.cla()

        if prop == 'status':
            cmap = plt.cm.Set1
        else:
            cmap = None

        if coord.lower() == 'equatorial':
            galactic = False
        elif coord.lower() == 'galactic':
            galactic = True
        else:
            raise ValueError("Unsupported coordinate system.")

        # plot:
        visualizer.plot(
                prop, night=date, galactic=galactic, ax=ax, cax=cax,
                plot_kws={'s': 6, 'cmap': cmap})


        if prop == 'status':
            labels = cax.get_yticklabels()
            cax.set_yticklabels(
                labels, rotation=70, ha='left', rotation_mode='anchor')

        # add figure title:
        fig.suptitle(f'{titles[prop]}\n{date.iso[:10]}')
        cax.set_ylabel(titles[prop], labelpad=8)

        # save:
        dir_fig = os.path.join(dir_figs, f'{prop}/')

        if not os.path.exists(dir_fig):
            os.makedirs(dir_fig)

        file_fig = os.path.join(dir_fig, f'{date.iso[:10]}.png')
        plt.savefig(file_fig, dpi=200)

    plt.close(fig)

# make movie:
if make_movie:
    for prop in properties:
        file_movie = f'{prop}.mp4'
        command = "ffmpeg -framerate 5 -pattern_type glob -i '{0}*.png' " \
               "-c:v libx264  -profile:v high -crf 20 -pix_fmt yuv420p " \
               "{0}{1}".format(
                      os.path.join(dir_figs, f'{prop}/'), file_movie)
        os.system(command)

#==============================================================================
