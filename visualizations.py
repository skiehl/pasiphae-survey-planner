#!/usr/bin/env python3
"""Visualizations for the Pasiphae survey planner.
"""

from astropy.coordinates import SkyCoord
import cartopy.crs as ccrs
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import warnings

from fieldgrid import FieldGrid

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

