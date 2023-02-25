#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualizations for the Pasiphae survey planner.
"""

import cartopy.crs as ccrs
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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
        """Create FieldGridVisualizer instance.

        Parameters
        ----------
        fields_n : skyfields.SkyFields or list of skyfields.Field, optional
            The fields to be drawn. If a list is provided it must contain
            skyfields.Field instances. If no fields are provided at class
            instantiation, they must be provided when calling the plotting
            method. The default is None.
        fields_s : skyfields.SkyFields or list of skyfields.Field, optional
            Fields to be drawn in a differnet color. If a list is provided it
            must contain skyfields.Field instances. The default is None.

        Returns
        -------
        None
        """

        self.fields_n = []
        self.fields_n = []
        self._add_fields(fields_n, fields_s)

    #--------------------------------------------------------------------------
    def _add_fields(self, fields_n, fields_s):
        """Add fields to class instance. Fields formerly stored will be
        overwritten.

        Parameters
        ----------
        fields_n : skyfields.SkyFields or list of skyfields.Field,
            The fields to be drawn. If a list is provided it must contain
            skyfields.Field instances.
        fields_s : skyfields.SkyFields or list of skyfields.Field or None
            Fields to be drawn in a differnet color. If a list is provided it
            must contain skyfields.Field instances. The default is None.

        Raises
        ------
        ValueError
            Raised when anything but skyfields.SkyFields, list, or None is
            provided to either of the arguments.

        Returns
        -------
        None
        """

        if fields_n is None:
           pass
        elif isinstance(fields_n, SkyFields):
            self.fields_n = fields_n.get_fields()
        elif isinstance(fields_n, list):
            self.fields_n = fields_n
        else:
            raise ValueError(
                    "Either provide SkyFields instance or list of Fields.")

        if fields_n is not None and fields_s is None:
            self.fields_s = []
        elif fields_s is None:
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
    def _plot_field_orthogonal(self, field, ax, color):
        """Draw a single field to the orthogonal projection plot.

        Parameters
        ----------
        field : skyfields.Field
            The Field instance to draw.
        ax : matplotlib.axes.Axes
            The Axes instance to draw to.
        color : color
            Any color specification acceptable to matplotlib.

        Returns
        -------
        None
        """

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
        """Draw a single field to the Mollweide projection plot.

        Parameters
        ----------
        field : skyfields.Field
            The Field instance to draw.
        ax : matplotlib.axes.Axes
            The Axes instance to draw to.
        color : color
            Any color specification acceptable to matplotlib.

        Returns
        -------
        None
        """

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
        """Orthographic plot of fields.

        Parameters
        ----------
        central_longitude : float
            Longitude in deg that is in the center of the projection. The
            default is 0..
        central_latitude : float
            Latitude in deg that is in the center of the projection. The
            default is 0..
        fields_n : skyfields.SkyFields or list of skyfields.Field, optional
            The fields to be drawn. If a list is provided it must contain
            skyfields.Field instances. If no fields are provided the ones added
            at instantiation are drawn. Fields provided here will overwrite
            any fields added previously. The default is None.
        fields_s : skyfields.SkyFields or list of skyfields.Field, optional
            Fields to be drawn in a differnet color. If a list is provided it
            must contain skyfields.Field instances. If no fields are provided
            the ones added at instantiation are drawn. Fields provided here
            will overwrite any fields added previously. The default is None.
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

        # add fields:
        self._add_fields(fields_n, fields_s)

        # raise error, when no (northern) fields are provided:
        if not len(self.fields_n):
            raise ValueError(
                    "No fields provided. Either provide during instance " \
                    "creation or through plotting method.")

        # create figure:
        fig, ax = self._create_orthographic_figure(
                ax, central_longitude, central_latitude)

        # plot (northern) fields:
        for field in self.fields_n:
            self._plot_field_orthogonal(field, ax, 'k')

        # plot southern fields:
        if len(self.fields_s):
            for field in self.fields_s:
                self._plot_field_orthogonal(field, ax, 'b')

        return fig, ax

    #--------------------------------------------------------------------------
    def mollweide(self, fields_n=None, fields_s=None, ax=None):
        """Mollweide plot of fields.

        Parameters
        ----------
        fields_n : skyfields.SkyFields or list of skyfields.Field, optional
            The fields to be drawn. If a list is provided it must contain
            skyfields.Field instances. If no fields are provided the ones added
            at instantiation are drawn. Fields provided here will overwrite
            any fields added previously. The default is None.
        fields_s : skyfields.SkyFields or list of skyfields.Field, optional
            Fields to be drawn in a differnet color. If a list is provided it
            must contain skyfields.Field instances. If no fields are provided
            the ones added at instantiation are drawn. Fields provided here
            will overwrite any fields added previously. The default is None.
        ax : matplotlib.axes.Axes, optional
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

        # add fields:
        self._add_fields(fields_n, fields_s)

        # raise error, when no (northern) fields are provided:
        if not len(self.fields_n):
            raise ValueError(
                    "No fields provided. Either provide during instance " \
                    "creation or through plotting method.")

        # create figure:
        fig, ax = self._create_mollweide_figure(ax)

        # plot (northern) fields:
        for field in self.fields_n:
            self._plot_field_mollweide(field, ax, 'k')

        # plot southern fields:
        if len(self.fields_s):
            for field in self.fields_s:
                self._plot_field_mollweide(field, ax, 'b')

        return fig, ax

#==============================================================================

class FieldVisualizer():
    """Parent class for field visualizations.
    """
    # TODO: docstring

    #--------------------------------------------------------------------------
    def __init__(self, fields_all=None, fields_obs=None):
    # TODO: docstring

        self.fields = {'all': None, 'obs': None}
        self.add_fields(fields_all, fields_obs)

    #--------------------------------------------------------------------------
    def _shift_ra(self, key):
        """Shift RA to the correct quadrants for plotting.

        Parameters
        ----------
        key : str
            Key for the field RAs. Either 'all' or 'obs'.

        Returns
        -------
        None
        """

        self.fields[key]['ra'] = np.where(
                self.fields[key]['ra'] > np.pi,
                self.fields[key]['ra'] - 2*np.pi,
                self.fields[key]['ra'])

    #--------------------------------------------------------------------------
    def _extract_data_fields_all(self, fields):
        """Extract data from list of Fields with information about
        corresponding observations.

        Parameters
        ----------
        fields : list of skyfields.Field instances
            A list of fields.

        Returns
        -------
        None
        """

        key = 'all'
        n_fields = len(fields)

        self.fields[key] = {}
        self.fields[key]['n_fields'] = n_fields
        self.fields[key]['ra'] = np.zeros(n_fields)
        self.fields[key]['dec'] = np.zeros(n_fields)
        self.fields[key]['n_obs_tot'] = np.zeros(n_fields, dtype=int)
        self.fields[key]['n_obs_done'] = np.zeros(n_fields, dtype=int)
        self.fields[key]['n_obs_pending'] = np.zeros(n_fields, dtype=int)

        for i, field in enumerate(fields):
            self.fields[key]['ra'][i] = field.center_ra.rad
            self.fields[key]['dec'][i] = field.center_dec.rad
            self.fields[key]['n_obs_tot'][i] = field.n_obs_tot
            self.fields[key]['n_obs_done'][i] = field.n_obs_done
            self.fields[key]['n_obs_pending'][i] = field.n_obs_pending

        self._shift_ra('all')
        self.fields[key]['obs_done'] = \
                self.fields[key]['n_obs_tot'] == self.fields[key]['n_obs_done']
        self.fields[key]['obs_pending'] = \
                self.fields[key]['n_obs_pending'] > 0

    #--------------------------------------------------------------------------
    def _extract_data_fields_obs(self, fields):
        """Extract data from list of Fields with information about the field
        observability.

        Parameters
        ----------
        fields : list of skyfields.Field instances
            A list of fields.

        Returns
        -------
        None
        """

        key = 'obs'
        n_fields = len(fields)

        self.fields[key] = {}
        self.fields[key]['n_fields'] = n_fields
        self.fields[key]['ra'] = np.zeros(n_fields)
        self.fields[key]['dec'] = np.zeros(n_fields)
        self.fields[key]['dur_obs'] = np.zeros(n_fields)
        self.fields[key]['status'] = np.zeros(n_fields, dtype=int)
        self.fields[key]['dur_set'] = np.zeros(n_fields)
        self.fields[key]['priority'] = np.zeros(n_fields)

        for i, field in enumerate(fields):
            self.fields[key]['ra'][i] = field.center_ra.rad
            self.fields[key]['dec'][i] = field.center_dec.rad
            self.fields[key]['dur_obs'][i] = field.get_obs_duration().value
            self.fields[key]['status'][i] = field.status
            self.fields[key]['dur_set'][i] = \
                    field.setting_in.value if field.status == 3 else -1
            self.fields[key]['priority'][i] = field.priority

        self._shift_ra('obs')

    #--------------------------------------------------------------------------
    def _create_figure(self, ax, cax):
        """Create figure for plotting.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            The subplot to plot to. If None, a new figure is created.
        cax : matplotlib.axes.Axes or None
            The subplot to draw the colorbar in. Must not be None when 'ax' is
            not None.

        Raises
        ------
        ValueError
            Raised if 'ax' is procided, but 'cax' is not.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure instance drawn to.
        ax : matplotlib.axes.Axes
            The Axes instance drawn to.
        cax : matplotlib.axes.Axes
            The Axes instance with the colorbar.
        """

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
    def _add_colorbar_to_scatterplot(self, cax, sc, label):
        """Add colorbar to a scatter plot.

        Parameters
        ----------
        cax : matplotlib.axes.Axes or None
            The subplot to draw the colorbar in.
        sc : matplotlib.collections.PathCollection
            The scatter plot.
        label : str
            Colorbar label.

        Returns
        -------
        cax : matplotlib.axes.Axes
            The Axes instance with the colorbar.
        cbar : matplotlib.colorbar.Colorbar
            The colorbar.
        """

        if cax is not None:
            cbar = plt.colorbar(sc, cax=cax)
        else:
            cbar = plt.colorbar(sc)
            cax = cbar.ax

        cbar.ax.set_ylabel(label)

        return cax, cbar

    #--------------------------------------------------------------------------
    def add_fields(self, fields_all=None, fields_obs=None):
        """Add fields to class instance. Fields formerly stored will be
        overwritten.

        Raises
        ------
        ValueError
            Raised when anything but a list is provided.

        Parameters
        ----------
        fields : list of skyfields.Field,
            The fields to be drawn.

        Returns
        -------
        None
        """

        if fields_all is None:
            pass
        elif isinstance(fields_all, list):
            self._extract_data_fields_all(fields_all)
        else:
            raise ValueError(
                    "'fields_all' must be a list of skyfields.Field instances."
                    )

        if fields_obs is None:
            pass
        elif isinstance(fields_obs, list):
            self._extract_data_fields_obs(fields_obs)
        else:
            raise ValueError(
                    "'fields_obs' must be a list of skyfields.Field instances."
                    )

    #--------------------------------------------------------------------------
    def show(
            self, show_all=True, show_finished=False, show_pending=False,
            show_status=False, show_duration=False, show_set_duration=False,
            show_priority=False,
            night_duration=None, priority=None,
            fields_all=None, fields_obs=None, cmap=None, ax=None, cax=None):
        # TODO: docstring

        # TODO: I want to separate all individual plotting parts into separate
        # methods that will also have **kwargs for individual ploting that
        # allows more control. This method will be a single-interface, quick-
        # look option.
        # TODO: Each individual plotting method should throw a warning when
        # the required fields are not available.

        ms_fields = 12
        ms_obs = 36

        # add fields if needed:
        self.add_fields(fields_all, fields_obs)

        # create figure:
        fig, ax, cax = self._create_figure(ax, None)

        # show all fields:
        if show_all and self.fields['all']:
            ax.scatter(
                    self.fields['all']['ra'], self.fields['all']['dec'],
                    marker='o', s=ms_fields, facecolor='0.7', zorder=0)

        # show finished fields:
        if show_finished and self.fields['all']:
            sel = self.fields['all']['obs_done']
            ax.scatter(
                    self.fields['all']['ra'][sel],
                    self.fields['all']['dec'][sel],
                    marker='o', s=ms_fields, facecolor='k', zorder=2)

        # show pending fields:
        if show_pending and self.fields['all']:
            sel = self.fields['all']['obs_pending']
            ax.scatter(
                    self.fields['all']['ra'][sel],
                    self.fields['all']['dec'][sel],
                    marker='o', s=ms_fields, edgecolor='k', facecolor='w',
                    zorder=2)

        # show observable field status:
        if show_status and self.fields['obs']:
            if cmap is None:
                cmap = plt.cm.rainbow

            norm = colors.BoundaryNorm(np.arange(-0.5, 4, 1), cmap.N)
            sc = ax.scatter(
                    self.fields['obs']['ra'], self.fields['obs']['dec'],
                    c=self.fields['obs']['status'], norm=norm, cmap=cmap,
                    s=ms_obs, marker='o', zorder=1, alpha=0.8)
            cbar = plt.colorbar(sc, ticks=np.arange(0, 4))
            cbar.ax.set_yticklabels([
                    'undefined', 'rising', 'plateauing', 'setting'])

        # show duration of observability:
        elif show_duration and self.fields['obs']:
            # devide duration by night duration, if applicable:
            if night_duration is None:
                data = self.fields['obs']['dur_obs'] * 24.
                label = 'Duration of availability (hours)'
            else:
                data = self.fields['obs']['dur_obs'] / night_duration.value
                label = 'Availability fraction of night'

            # plot data:
            sc = ax.scatter(
                    self.fields['obs']['ra'], self.fields['obs']['dec'],
                    c=data, s=ms_obs)
            cax, cbar = self._add_colorbar_to_scatterplot(cax, sc, label)

        # show duration until setting:
        elif show_set_duration and self.fields['obs']:
           sel = self.fields['obs']['status'] == 3
           sc = ax.scatter(
                   self.fields['obs']['ra'][sel],
                   self.fields['obs']['dec'][sel],
                   c=self.fields['obs']['dur_set'][sel], s=ms_obs)
           cax, cbar = self._add_colorbar_to_scatterplot(
                   cax, sc, 'Duration until setting (days)')

        # show priority:
        elif show_priority and self.fields['obs']:
            if priority is None:
                data = self.fields['obs']['priority']
            else:
                data = priority

            sc = ax.scatter(
                    self.fields['obs']['ra'], self.fields['obs']['dec'],
                    c=data, s=ms_obs)
            cax, cbar = self._add_colorbar_to_scatterplot(
                cax, sc, 'Priority')

        return fig, ax, cax



#==============================================================================

