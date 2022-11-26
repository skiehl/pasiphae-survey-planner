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

    #--------------------------------------------------------------------------
    def __init__(self, fields=None):
        """Create FieldVisualizer instance.

        Parameters
        ----------
        fields : list of skyfields.Field, optional
            The fields to be drawn. If none are provided, fields need to be
            added when calling the plotting methods. The default is None.

        Returns
        -------
        None
        """

        self.n_fields = 0
        self.fields = []
        self._add_fields(fields)

    #--------------------------------------------------------------------------
    def _add_fields(self, fields):
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

        if fields is None:
            pass

        elif isinstance(fields, list):
            self.fields = fields
            self._extract_field_data()

        else:
            raise ValueError(
                    "'fields' must be a list of skyfields.Field instances.")

    #--------------------------------------------------------------------------
    def _extract_field_data(self):
        """Extract data from list of Fields.

        Returns
        -------
        None
        """

        self.n_fields = len(self.fields)
        self.fields_ra = np.zeros(self.n_fields)
        self.fields_dec = np.zeros(self.n_fields)
        self.fields_dur = np.zeros(self.n_fields)
        self.fields_stat = np.zeros(self.n_fields, dtype=int)
        self.fields_set_dur = []
        self.fields_obs_done = np.zeros(self.n_fields)
        self.fields_obs_pending = np.zeros(self.n_fields)

        for i, field in enumerate(self.fields):
            self.fields_ra[i] = field.center_ra.rad
            self.fields_dec[i] = field.center_dec.rad
            self.fields_dur[i] = field.get_obs_duration().value
            self.fields_stat[i] = field.status
            if field.status == 3:
                self.fields_set_dur.append(field.setting_in.value)
            self.fields_obs_done[i] = field.n_obs_done
            self.fields_obs_pending[i] = field.n_obs_pending

        self.fields_ra = np.where(
                self.fields_ra > np.pi,
                self.fields_ra - 2*np.pi,
                self.fields_ra)

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
    def _add_colorbar_to_scatterplot(self, cax, sc):
        """Add colorbar to a scatter plot.

        Parameters
        ----------
        cax : matplotlib.axes.Axes or None
            The subplot to draw the colorbar in.
        sc : matplotlib.collections.PathCollection
            The scatter plot.

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

        return cax, cbar

#==============================================================================

class FieldAvailabilityVisualizer(FieldVisualizer):
    """Visualizations of the field availability.
    """

    #--------------------------------------------------------------------------
    def field_duration(
            self, fields=None, night_duration=None, ax=None, cax=None,
            **kwargs):
        """Plot the field positions. Color code the duration of the field
        availability. If night_duration is given, the availability is shown as
        the fraction of the night.

        Parameters
        ----------
        fields : list of skyfields.Field, optional
            The fields to be drawn. If none are provided, fields must have been
            set during instantiation. If new fields are provided, they
            overwrite any fields stored previously. The default is None.
        night_duration : astropy.units.Quantity, optional
            The duration of the night in days. The default is None.
        ax : matplotlib.axes.Axes, optional
            The subplot to plot to. If None, a new figure is created.
        cax : matplotlib.axes.Axes, optional
            The subplot to draw the colorbar in. Must not be None when 'ax' is
            not None.
        **kwargs
            The keyword arguments are passed to `matplotlib.pyplot.scatter()`

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure instance drawn to.
        ax : matplotlib.axes.Axes
            The Axes instance drawn to.
        cax : matplotlib.axes.Axes
            The Axes instance with the colorbar.
        cbar : matplotlib.colorbar.Colorbar
            The colorbar.
        """

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

        return fig, ax, cax, cbar

    #--------------------------------------------------------------------------
    def field_set_duration(self, fields=None, ax=None, cax=None, **kwargs):
        """Plot the positions of setting fields. Color code the duration of the
        until when each field is setting.

        Parameters
        ----------
        fields : list of skyfields.Field, optional
            The fields to be drawn. If none are provided, fields must have been
            set during instantiation. If new fields are provided, they
            overwrite any fields stored previously. The default is None.
        ax : matplotlib.axes.Axes, optional
            The subplot to plot to. If None, a new figure is created.
        cax : matplotlib.axes.Axes, optional
            The subplot to draw the colorbar in. Must not be None when 'ax' is
            not None.
        **kwargs
            The keyword arguments are passed to `matplotlib.pyplot.scatter()`

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure instance drawn to.
        ax : matplotlib.axes.Axes
            The Axes instance drawn to.
        cax : matplotlib.axes.Axes
            The Axes instance with the colorbar.
        cbar : matplotlib.colorbar.Colorbar
            The colorbar.
        """

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

        return fig, ax, cax, cbar

    #--------------------------------------------------------------------------
    def field_status(
            self, fields=None, ax=None, cax=None, cmap=None, **kwargs):
        """Plot the field positions. Color code the status of each field.

        Parameters
        ----------
        fields : list of skyfields.Field, optional
            The fields to be drawn. If none are provided, fields must have been
            set during instantiation. If new fields are provided, they
            overwrite any fields stored previously. The default is None.
        ax : matplotlib.axes.Axes, optional
            The subplot to plot to. If None, a new figure is created.
        cax : matplotlib.axes.Axes, optional
            The subplot to draw the colorbar in. Must not be None when 'ax' is
            not None.
        cmap : matplotlib.colors.Colormap, optional
            Colormap used for the color coding.
        **kwargs
            The keyword arguments are passed to `matplotlib.pyplot.scatter()`

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure instance drawn to.
        ax : matplotlib.axes.Axes
            The Axes instance drawn to.
        cax : matplotlib.axes.Axes
            The Axes instance with the colorbar.
        cbar : matplotlib.colorbar.Colorbar
            The colorbar.
        """

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

        return fig, ax, cax, cbar

#==============================================================================

class FieldObservationVisualizer(FieldVisualizer):
    """Visualizations of the field observation status.
    """

    #--------------------------------------------------------------------------
    def field_status(
            self, fields=None, ax=None, **kwargs):
        """Plot the field positions. Color code the observation status.

        Parameters
        ----------
        fields : list of skyfields.Field, optional
            The fields to be drawn. If none are provided, fields must have been
            set during instantiation. If new fields are provided, they
            overwrite any fields stored previously. The default is None.
        ax : matplotlib.axes.Axes, optional
            The subplot to plot to. If None, a new figure is created.
        **kwargs
            The keyword arguments are passed to `matplotlib.pyplot.scatter()`

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure instance drawn to.
        ax : matplotlib.axes.Axes
            The Axes instance drawn to.
        """

        # extract coordinates and status from fields:
        self._add_fields(fields)

        # create figure:
        fig, ax, cax = self._create_figure(ax, None)

        # plot pending fields:
        sel = self.fields_obs_pending > 0
        ax.scatter(
                self.fields_ra[sel], self.fields_dec[sel],
                edgecolor='0.4', facecolor='w', zorder=2, **kwargs)

        # plot observed fields:
        sel = self.fields_obs_done > 0
        ax.scatter(
                self.fields_ra[sel], self.fields_dec[sel], color='k',
                marker='s', zorder=1, **kwargs)

        return fig, ax

#==============================================================================

