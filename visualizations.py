#!/usr/bin/env python3
"""Visualizations for the Pasiphae survey planner.
"""

from abc import ABCMeta, abstractmethod
from astropy.coordinates import SkyCoord
from astropy.time import Time
import cartopy.crs as ccrs
from copy import deepcopy
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from pandas import DataFrame, Series
import warnings

from fieldgrid import FieldGrid
from prioritizer import PrioritizerAnnualAvailability
from surveyplanner import SurveyPlanner

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
    def __init__(self, *grids):
        """Create FieldGridVisualizer instance.

        Parameters
        ----------
        grids : skyfields.FieldGrid or list of skyfields.Field
            The fields to be drawn. If a list is provided it must contain
            skyfields.Field instances. Multiple grids can be provided that will
            be plotted in different colors. If no fields are provided at class
            instanciation, they must be provided when calling the plotting
            method.

        Returns
        -------
        None
        """

        self.grids = []
        self._add_grids(*grids)

    #--------------------------------------------------------------------------
    def _extract_coords_from_list(self, fields):
        # TODO: docstring

        raise NotImplementedError()

    #--------------------------------------------------------------------------
    def _add_grids(self, *grids):
        """Add field grid(s) to class instance. Fields formerly stored will be
        overwritten.

        Parameters
        ----------
        grids : skyfields.FieldGrid or list of skyfields.Field
            The fields to be drawn. If a list is provided it must contain
            skyfields.Field instances.

        Raises
        ------
        ValueError
            Raised when anything but skyfields.FieldGrid, list, or None is
            provided to either of the arguments.

        Returns
        -------
        None
        """

        self.grids = []

        for grid in grids:
            self.grids.append({})

            if isinstance(grid, FieldGrid):
                self.grids[-1]['center_ras'], self.grids[-1]['center_decs'] \
                        = grid.get_center_coords()
                self.grids[-1]['corner_ras'], self.grids[-1]['corner_decs'] \
                        = grid.get_corner_coords()

            elif isinstance(grid, list):
                self.grids[-1]['center_ras'], self.grids[-1]['center_decs'], \
                self.grids[-1]['corner_ras'], self.grids[-1]['corner_decs'] \
                        = self._extract_coords_from_list(grid)

            else:
                raise ValueError(
                        "Either provide FieldGrid instance or list of Fields.")

    #--------------------------------------------------------------------------
    def _create_orthographic_figure(
            self, ax, central_longitude, central_latitude):
        """Create figure for orthographic plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            The subplot to draw into. If None, a new figure and Axes instance
            are created.
        central_longitude : float
            Longitude in deg that is in the center of the projection.
        central_latitude : float
            Latitude in deg that is in the center of the projection.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure instance drawn to.
        ax : matplotlib.axes.Axes
            The Axes instance drawn to.
        """

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
        """Create figure for Mollweide projection plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            The subplot to draw into. If None, a new figure and Axes instance
            are created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure instance drawn to.
        ax : matplotlib.axes.Axes
            The Axes instance drawn to.
        """

        if ax is None:
            fig = plt.figure(figsize=(16, 10))
        else:
            fig = plt.gcf()


        ax = fig.add_subplot(111, projection='mollweide')
        ax.grid(True, color='magenta', linestyle=':')

        return fig, ax

    #--------------------------------------------------------------------------
    def _plot_field_orthographic(self, corner_ras, corner_decs, ax, color):
        """Draw a single field to the orthogonal projection plot.

        Parameters
        ----------
        corner_ras : np.ndarray
            Right ascensions of the four field corner points in radians
        corner_decs : np.ndarray
            Declinations of the four field corner points in radians..
        ax : matplotlib.axes.Axes
            The Axes instance to draw to.
        color : color
            Any color specification acceptable to matplotlib.

        Returns
        -------
        None
        """

        # extract corner coordinates:
        corner_ras = corner_ras / np.pi * 180. # deg
        corner_ras = np.r_[corner_ras, corner_ras[0]]
        corner_decs = corner_decs / np.pi * 180. # deg
        corner_decs = np.r_[corner_decs, corner_decs[0]]


        if np.any(np.diff(corner_ras) > 180.):
            corner_ras = np.where(
                    corner_ras > 180., corner_ras - 360., corner_ras)

        # plot outline:
        ax.plot(corner_ras, corner_decs, color=color, linestyle='-',
                linewidth=0.5, transform=ccrs.PlateCarree())

    #--------------------------------------------------------------------------
    def _plot_field_mollweide(self, corner_ras, corner_decs, ax, color):
        """Draw a single field to the Mollweide projection plot.

        Parameters
        ----------
        corner_ras : np.ndarray
            Right ascensions of the four field corner points in radians
        corner_decs : np.ndarray
            Declinations of the four field corner points in radians..
        ax : matplotlib.axes.Axes
            The Axes instance to draw to.
        color : color
            Any color specification acceptable to matplotlib.

        Returns
        -------
        None
        """

        # extract corner coordinates:
        corner_ras = np.r_[corner_ras, corner_ras[0]]
        corner_ras = np.where(
                corner_ras > np.pi, corner_ras - 2 * np.pi, corner_ras)
        corner_decs = np.r_[corner_decs, corner_decs[0]]

        # plot wrapping fields:
        if corner_ras[0] > 0 and corner_ras[1] < 0:
            # plot right part:
            ax.plot([np.pi, corner_ras[0], corner_ras[0], np.pi],
                    [corner_decs[2], corner_decs[2], corner_decs[0],
                     corner_decs[0]],
                    color=color, linestyle='-', linewidth=0.5)

            # plot left part:
            ax.plot([-np.pi, corner_ras[1], corner_ras[1], -np.pi],
                    [corner_decs[2], corner_decs[2], corner_decs[0],
                     corner_decs[0]],
                    color=color, linestyle='-', linewidth=0.5)

        # plot non-wrapping fields:
        else:
            ax.plot(corner_ras, corner_decs, color=color, linestyle='-',
                    linewidth=0.5)

    #--------------------------------------------------------------------------
    def _plot_galactic_plane_orthographic(self, ax, gal_lat_lim, n=100):
        """Plot the galactic plane latitude limits.

        Parameters
        ----------
        matplotlib.axes.Axes
           The Axes instance to draw to.
        gal_lat_lim : float
            Galactic latitude limit in radians. Will only be plotted if
            different from zero.
        n : int, optional
            Number of points used for the plotted lines. The default is 100.

        Returns
        -------
        None
        """

        if not gal_lat_lim:
            return None

        gal_lat_lim *= 180. / np.pi
        l = np.linspace(0, 360., n)

        for sign in [-1, 1]:
            b = gal_lat_lim * sign * np.ones(l.shape[0])
            gcoord = SkyCoord(l=l, b=b, unit='deg', frame='galactic')
            coord = gcoord.fk5
            ax.plot(coord.ra.deg, coord.dec.deg, marker='None', color='orange',
                    linestyle='-', transform=ccrs.PlateCarree())

    #--------------------------------------------------------------------------
    def _plot_galactic_plane_mollweide(self, ax, gal_lat_lim, n=100):
        """Plot the galactic plane latitude limits.

        Parameters
        ----------
        matplotlib.axes.Axes
           The Axes instance to draw to.
        gal_lat_lim : float
            Galactic latitude limit in radians. Will only be plotted if
            different from zero.
        n : int, optional
            Number of points used for the plotted lines. The default is 100.

        Returns
        -------
        None
        """

        if not gal_lat_lim:
            return None

        gal_lat_lim *= 180. / np.pi
        l = np.linspace(0, 360., n+1)
        l = np.r_[l, l[1]]

        for sign in [-1, 1]:
            b = sign * gal_lat_lim * np.ones(l.shape[0])
            gcoord = SkyCoord(l=l, b=b, unit='deg', frame='galactic')
            coord = gcoord.fk5
            ra = np.where(
                    coord.ra.rad > np.pi, coord.ra.rad - 2. * np.pi,
                    coord.ra.rad)
            dec = coord.dec.rad

            i_wrap = np.nonzero(np.absolute(np.diff(ra)) > np.pi)[0]
            i_wrap = [0] + list(i_wrap + 1) + [-1]

            for i, j in zip(i_wrap[:-1], i_wrap[1:]):
                ax.plot(ra[i:j], dec[i:j], marker='None', color='orange',
                        linestyle='-')

    #--------------------------------------------------------------------------
    def orthographic(
            self, *grids, gal_lat_lim=0, central_longitude=25.,
            central_latitude=45., outlines=False, ax=None, **kwargs):
        # TODO: docstring
        """Orthographic plot of fields.

        Parameters
        ----------
        grids : skyfields.FieldGrid or list of skyfields.Field
            The fields to be drawn. If a list is provided it must contain
            skyfields.Field instances. Multiple grids can be provided that will
            be plotted in different colors. If fields were provided at class
            instanciation, they do not have to be provided here. If provided
            here, these fields will overwrite any fields added before.
        gal_lat_lim : float, optional
            Galactic latitude limit in radians. Will only be plotted if
            different from zero. The default is 0.
        central_longitude : float
            Longitude in deg that is in the center of the projection. The
            default is 25..
        central_latitude : float
            Latitude in deg that is in the center of the projection. The
            default is 45..
        outlines : bool, optional
            If True, plot field outlines. Otherwise only field centers are
            plotted. The default is False.
        ax : matplotlib.axes.Axes, optional
            The Axes instance to draw to. If no Axes is provided a new figure
            and axes instance are created. The default is None.

        Raises
        ------
        ValueError
            Raised if no fields are stored in this instance.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure instance drawn to.
        ax : matplotlib.axes.Axes
            The Axes instance drawn to.
        """

        # add grids:
        if len(grids):
            self._add_grids(*grids)

        # create figure:
        fig, ax = self._create_orthographic_figure(
                ax, central_longitude, central_latitude)

        # plot Galactic plane:
        self._plot_galactic_plane_orthographic(ax, gal_lat_lim)

        # warn when no grids are provided:
        if not len(self.grids):
            warnings.warn(
                    "No field grids provided. Either provide during instance" \
                    " creation or through plotting method.")

            return fig, ax

        # iterate through grids:
        for grid in self.grids:
            # plot field outlines:
            if outlines:
                for corner_ras, corner_decs in zip(
                        grid['corner_ras'], grid['corner_decs']):
                    self._plot_field_orthographic(
                            corner_ras, corner_decs, ax, 'k')

            # or plot field centers:
            else:
                ax.plot(np.degrees(grid['center_ras']),
                        np.degrees(grid['center_decs']), linestyle='None',
                        marker='o', transform=ccrs.PlateCarree(), **kwargs)

        return fig, ax

    #--------------------------------------------------------------------------
    def mollweide(self, *grids, gal_lat_lim=0, outlines=False, ax=None, **kwargs):
        """Mollweide plot of fields.

        Parameters
        ----------
        grids : skyfields.FieldGrid or list of skyfields.Field
            The fields to be drawn. If a list is provided it must contain
            skyfields.Field instances. Multiple grids can be provided that will
            be plotted in different colors. If fields were provided at class
            instanciation, they do not have to be provided here. If provided
            here, these fields will overwrite any fields added before.
        gal_lat_lim : float, optional
            Galactic latitude limit in radians. Will only be plotted if
            different from zero. The default is 0.
        outlines : bool, optional
            If True, plot field outlines. Otherwise only field centers are
            plotted. The default is False.
        ax : matplotlib.axes.Axes, optional
            The Axes instance to draw to. If no Axes is provided a new figure
            and axes instance are created. The default is None.

        Raises
        ------
        ValueError
            Raised if no fields are stored in this instance.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure instance drawn to.
        ax : matplotlib.axes.Axes
            The Axes instance drawn to.
        """

        # add grids:
        if len(grids):
            self._add_grids(*grids)

        # create figure:
        fig, ax = self._create_mollweide_figure(ax)

        # plot Galactic plane:
        self._plot_galactic_plane_mollweide(ax, gal_lat_lim)

        # warn when no grids are provided:
        if not len(self.grids):
            warnings.warn(
                    "No field grids provided. Either provide during instance" \
                    " creation or through plotting method.")

            return fig, ax

        # iterate through grids:
        for grid in self.grids:
            # plot field outlines:
            if outlines:
                for corner_ras, corner_decs in zip(
                        grid['corner_ras'], grid['corner_decs']):
                    self._plot_field_mollweide(
                            corner_ras, corner_decs, ax, 'k')

            # or plot field centers:
            else:
                center_ras = np.where(
                        grid['center_ras'] > np.pi,
                        grid['center_ras'] - 2. * np.pi,
                        grid['center_ras'])
                ax.plot(center_ras, grid['center_decs'],
                        linestyle='None', marker='o', **kwargs)

        return fig, ax

#==============================================================================
