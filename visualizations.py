#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualizations for the survey planner.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np

from skyfields import SkyFields

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

class FieldGridVisualizer():
    """Visualization of the field grid.
    """

    #--------------------------------------------------------------------------
    def __init__(self, fields_n=None, fields_s=None):
        # TODO

        self.fields_n = None
        self.fields_n = None
        self._add_fields(fields_n, fields_s)

    #--------------------------------------------------------------------------
    def _add_fields(self, fields_n, fields_s):
        # TODO

        if fields_n is None:
           pass
        elif isinstance(fields_n, SkyFields):
            self.fields_n = fields_n.get_fields()
        elif isinstance(fields_n, list):
            self.fields_n = fields_n
        else:
            raise ValueError(
                    "Either provide SkyFields instance or list of Fields.")

        if fields_s is None:
           pass
        elif isinstance(fields_s, SkyFields):
            self.fields_s = fields_s.get_fields()
        elif isinstance(fields_s, list):
            self.fields_s = fields_s
        else:
            raise ValueError(
                    "Either provide SkyFields instance or list of Fields.")

    #--------------------------------------------------------------------------
    def _create_orthographic_figure(
            self, ax, central_longitude, central_latitude):
        # TODO

        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            projection = ccrs.Orthographic(central_longitude, central_latitude)
        else:
            fig = plt.gcf()

        ax = fig.add_subplot(1, 1, 1, projection=projection)

        ax.set_global()
        gl = ax.gridlines(
                crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1,
                color='magenta', linestyle=':')
        gl.xlocator = MultipleLocator(30)
        gl.ylocator = MultipleLocator(15)

        return fig, ax

    #--------------------------------------------------------------------------
    def _create_mollweide_figure(self, ax):
        # TODO

        if ax is None:
            fig = plt.figure(figsize=(16, 10))
        else:
            fig = plt.gcf()


        ax = fig.add_subplot(111, projection='mollweide')
        ax.grid(True, color='magenta', linestyle=':')

        return fig, ax

    #--------------------------------------------------------------------------
    def _plot_field_orthogonal(self, field, ax, color):
        # TODO

        # extract corner coordinates:
        corners_ra = field.corners_coord.ra.deg
        corners_ra = np.r_[corners_ra, corners_ra[0]]
        corners_dec = field.corners_coord.dec.deg
        corners_dec = np.r_[corners_dec, corners_dec[0]]

        # plot outline:
        ax.plot(corners_ra, corners_dec, color=color, linestyle='-',
                linewidth=0.5, transform=ccrs.PlateCarree())

    #--------------------------------------------------------------------------
    def _plot_field_mollweide(self, field, ax, color):
        # TODO

        # extract corner coordinates:
        corners_ra = field.corners_coord.ra.rad
        corners_ra = np.r_[corners_ra, corners_ra[0]]
        corners_ra = np.where(
                corners_ra > np.pi, corners_ra - 2*np.pi, corners_ra)
        corners_dec = field.corners_coord.dec.rad
        corners_dec = np.r_[corners_dec, corners_dec[0]]

        # plot wrapping fields:
        if corners_ra[0] > 0 and corners_ra[1] < 0:
            # plot right part:
            ax.plot([np.pi, corners_ra[0], corners_ra[0], np.pi],
                    [corners_dec[2], corners_dec[2], corners_dec[0],
                     corners_dec[0]],
                    color=color, linestyle='-', linewidth=0.5)

            # plot left part:
            ax.plot([-np.pi, corners_ra[1], corners_ra[1], -np.pi],
                    [corners_dec[2], corners_dec[2], corners_dec[0],
                     corners_dec[0]],
                    color=color, linestyle='-', linewidth=0.5)


        # plot non-wrapping fields:
        else:
            ax.plot(corners_ra, corners_dec, color=color, linestyle='-',
                    linewidth=0.5)

    #--------------------------------------------------------------------------
    def orthographic(
            self, central_longitude=0., central_latitude=0., fields_n=None,
            fields_s=None, ax=None):
        # TODO

        # add fields:
        self._add_fields(fields_n, fields_s)

        # create figure:
        fig, ax = self._create_orthographic_figure(
                ax, central_longitude, central_latitude)

        # raise error, when no (northern) fields are provided:
        if self.fields_n is None:
            raise ValueError(
                    "No fields provided. Either provide during instance " \
                    "creation or through plotting method.")

        # plot (northern) fields:
        for field in self.fields_n:
            self._plot_field_orthogonal(field, ax, 'k')

        # plot southern fields:
        if self.fields_s is not None:
            for field in self.fields_s:
                self._plot_field_orthogonal(field, ax, 'b')

        return fig, ax

    #--------------------------------------------------------------------------
    def mollweide(self, fields_n=None, fields_s=None, ax=None):
        # TODO

        # add fields:
        self._add_fields(fields_n, fields_s)

        # create figure:
        fig, ax = self._create_mollweide_figure(ax)

        # plot (northern) fields:
        for field in self.fields_n:
            self._plot_field_mollweide(field, ax, 'k')

        # plot southern fields:
        if self.fields_s is not None:
            for field in self.fields_s:
                self._plot_field_mollweide(field, ax, 'b')

        return fig, ax

#==============================================================================

class FieldAvailabilityVisualizer():
    """Visualizations of the field availability.
    """

    #--------------------------------------------------------------------------
    def __init__(self, fields=None):
        # TODO

        self.fields = fields

    #--------------------------------------------------------------------------
    def _add_fields(self, fields):
        # TODO

        if fields is not None:
            self.fields = fields

        self._extract_field_data()

    #--------------------------------------------------------------------------
    def _extract_field_data(self):
        # TODO

        if self.fields is None:
            raise ValueError(
                    "No fields provided. Either provide during instance " \
                    "creation or through plotting method.")

        self.n_fields = len(self.fields)
        self.fields_ra = np.zeros(self.n_fields)
        self.fields_dec = np.zeros(self.n_fields)
        self.fields_dur = np.zeros(self.n_fields)
        self.fields_stat = np.zeros(self.n_fields, dtype=int)
        self.fields_set_dur = []

        for i, field in enumerate(self.fields):
            self.fields_ra[i] = field.center_ra.rad
            self.fields_dec[i] = field.center_dec.rad
            self.fields_dur[i] = field.get_obs_duration().value
            self.fields_stat[i] = field.status
            if field.status == 3:
                self.fields_set_dur.append(field.setting_in.value)

        self.fields_ra = np.where(
                self.fields_ra > np.pi,
                self.fields_ra - 2*np.pi,
                self.fields_ra)

    #--------------------------------------------------------------------------
    def _create_figure(self, ax, cax):
        # TODO

        if ax is None:
            fig = plt.figure(figsize=(16, 10))
            ax = fig.add_subplot(111, projection='mollweide')
        elif cax is None:
            raise ValueError(
                    "When 'ax' is provided also provide 'cax' for placing " \
                    "the colorbar.")
        else:
            fig = plt.gcf()

        ax.grid(True, color='m', linestyle=':')

        return fig, ax, cax

    #--------------------------------------------------------------------------
    def _add_colorbar_to_scatterplot(self, cax, sc):
        # TODO

        if cax is not None:
            cbar = plt.colorbar(sc, cax=cax)
        else:
            cbar = plt.colorbar(sc)
            cax = cbar.ax

        return cax, cbar

    #--------------------------------------------------------------------------
    def field_duration(
            self, fields=None, night_duration=None, ax=None, cax=None,
            **kwargs):
        # TODO

        # extract coordinates and durations from fields:
        self._add_fields(fields)

        # create figure:
        fig, ax, cax = self._create_figure(ax, cax)

        # devide duration by night duration, if applicable:
        if night_duration is None:
            data = self.fields_dur * 24.
            label = 'Duration of availability (hours)'
        else:
            data = self.fields_dur / night_duration.value
            label = 'Availability fraction of night'

        # plot data:
        sc = ax.scatter(
                self.fields_ra, self.fields_dec, c=data, **kwargs)

        # add colorbar:
        cax, cbar = self._add_colorbar_to_scatterplot(cax, sc)
        cbar.ax.set_ylabel(label)

        return fig, ax, cbar

    #--------------------------------------------------------------------------
    def field_set_duration(self, fields=None, ax=None, cax=None, **kwargs):
        # TODO

        # extract coordinates and setting durations from fields:
        self._add_fields(fields)

        # create figure:
        fig, ax, cax = self._create_figure(ax, cax)

        # plot data:
        sel = self.fields_stat == 3
        sc = ax.scatter(
                self.fields_ra[sel], self.fields_dec[sel],
                c=self.fields_set_dur, **kwargs)

        # add colorbar:
        cax, cbar = self._add_colorbar_to_scatterplot(cax, sc)
        cbar.ax.set_ylabel('Duration until setting (days)')

        return fig, ax, cbar

    #--------------------------------------------------------------------------
    def field_status(
            self, fields=None, ax=None, cax=None, cmap=None, **kwargs):
        # TODO

        # extract coordinates and status from fields:
        self._add_fields(fields)

        # create figure:
        fig, ax, cax = self._create_figure(ax, cax)

        # create 5-color colormap:
        if cmap is None:
            cmap = plt.cm.rainbow

        norm = colors.BoundaryNorm(np.arange(-1.5, 4, 1), cmap.N)

        # plot data:
        sc = ax.scatter(
                self.fields_ra, self.fields_dec, c=self.fields_stat, norm=norm,
                cmap=cmap, **kwargs)
        cbar = plt.colorbar(sc, ticks=np.arange(-1, 4))
        cbar.ax.set_yticklabels([
                'undefined', 'not observable', 'rising', 'plateauing',
                'setting'])

        return fig, ax, cbar
