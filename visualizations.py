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

class FieldVisualizer(object, metaclass=ABCMeta):
    """Parent class for field visualizations.
    """

    #--------------------------------------------------------------------------
    def __init__(self, surveyplanner=None, fields=None):
        """Parent class for field visualizations.

        Parameters
        ----------
        surveyplanner : SurveyPlanner-type, optional
            A SurveyPlanner instance that provides an access point to a
            database to query the fields that should be plotted. If not
            provided, fields have to be given either through the `fields`
            argument or through the `set_fields()` method. The default is None.
        fields : list of dict, optional
            A list of dictionaries as returned e.g. by the
            `SurveyPlanner().query_fields()` method. The default is None.

        Raises
        ------
        ValueError
            Raised, if `surveyplanner` is not a SurveyPlanner-instance.

        Returns
        -------
        None
        """

        # check surveyplanner input:
        if not (surveyplanner is None
                or isinstance(surveyplanner, SurveyPlanner)):
            raise ValueError(
                "`surveyplanner` must be SurveyPlanner-instance or None.")

        self.surveyplanner = surveyplanner

        # check fields input:
        self._parse_fields(fields)
        self.fields = fields

        if surveyplanner is None and fields is None:
            print("WARNING: No surveyplanner and no fields provided. Use " \
                  "`set_fields() to provide fields before plotting.")

        self.fig = None
        self.ax = None
        self.cax = None
        self.projection = None

    #--------------------------------------------------------------------------
    @abstractmethod
    def _get_fields(self):
        """Get fields from the stored SurveyPlanner instance with the
        information that is required for plotting.

        Raises
        ------
        ValueError
            Raised, if no SurveyPlanner instance was provided at class
            instanciation.

        Returns
        -------
        None

        Notes
        -----
        This is an abstract method. Different child classes may require
        different field information for plotting. This method has to be
        implemented to provide this specific information. This method must
        assign a pandas.DataFrame() to the class attribute `self.fields`.
        """

        # check if surveyplanner is available:
        if self.surveyplanner is None:
            raise ValueError(
                    "No survey planner was provided at instanciation. Cannot "
                    "query fields. Either use `set_fields()` to provide fields"
                    " for plotting or create new visualizer instance and "
                    "provide `surveyplanner`.")

        print('Querying fields..')

        # custom code here

        self._galactic_coords()
        self._shift_ra()

    #--------------------------------------------------------------------------
    @abstractmethod
    def _parse_fields(self, fields, keys, error_message):
        """Check if the provided fields contain the information that is
        required for plotting.

        Parameters
        ----------
        fields : any type
            The "fields" that are provided as `fields` at class instanciation
            or through the `set_fields()` method. Expected is a list of dict.
        keys : list of str
            The keys that each list item dictionary must contain.
        error_message : TYPE
            The error message to print if the `fields` are not providing the
            required information.

        Raises
        ------
        ValueError
            Raised, if `fields` is None.
            Raised, if `fields` is not list.
            Raised, if any item in `fields` list is not dict.
            Raised, if any dict in `fields` list does not contain all of the
            keys listed in `keys`.

        Returns
        -------
        bool
            True, if the `fields` input fulfills the criteria. Otherwise,
            a ValueError is raised.

        Notes
        -----
        This is an abstract method. Each class may require different entries
        in the dictionaries listed in `fields`. The child class should specify
        the `keys` and `error_message` and then call this parent method as:
        super()._parse_fields(fields, keys, error_message).
        """

        if fields is None:
            return True

        if not isinstance(fields, list):
            raise ValueError(error_message)

        for field in fields:
            if not isinstance(field, dict):
                raise ValueError(error_message)

            for key in keys:
                if key not in field.keys():
                    raise ValueError(error_message)

        return True

    #--------------------------------------------------------------------------
    def _galactic_coords(self):
        """Add Galactic coordinates to fields dataframe.

        Returns
        -------
        None
        """
        coord = SkyCoord(
                self.fields['center_ra'], self.fields['center_dec'],
                unit='rad')
        coord = coord.galactic

        self.fields.insert(4, 'center_l', coord.l.rad)
        self.fields.insert(5, 'center_b', coord.b.rad)
        self.fields['center_l'] = np.where(
                self.fields['center_l'] > np.pi,
                self.fields['center_l'] - 2 * np.pi,
                self.fields['center_l'])

    #--------------------------------------------------------------------------
    def _shift_ra(self):
        """Shift RA to the correct quadrants for plotting.

        Returns
        -------
        None
        """

        self.fields['center_ra'] = np.where(
                self.fields['center_ra'] > np.pi,
                self.fields['center_ra'] - 2 * np.pi,
                self.fields['center_ra'])

    #--------------------------------------------------------------------------
    def _check_fields(self, **kwargs):
        """Check if fields have been stored. Otherwise, get fields from the
        stored SurveyPlanner instance.

        Parameters
        ----------
        **kwargs
            Key word arguments forwarded to the `_get_fields()` method.

        Returns
        -------
        None
        """

        if self.fields is None:
            self._get_fields(**kwargs)

    #--------------------------------------------------------------------------
    def _create_figure(self, projection='mollweide', ax=None, cax=None):
        """Create a figure.

        Parameters
        ----------
        projection : str, optional
            Projection that should be used for plotting the field coordinates.
            Must be 'mollweide', 'aitoff', 'hammer', or 'lambert'. The default
            is 'mollweide'.
        ax : matplotlib.Axes or None, optional
            If None, a new axis is created. Otherwise, the provided Axes is
            kept. The default is None.
        cax : matplotlib.Axes or None, optional
            If None, a new axis is created. Otherwise, the provided Axes is
            kept. The default is None.

        Raises
        ------
        ValueError
            Raise, if `ax` is neither matplotlib.Axes nor None.
            Raise, if `cax` is neither matplotlib.Axes nor None.

        Returns
        -------
        None
        """

        if isinstance(ax, plt.Axes):
            self.fig = plt.gcf()
            self.ax = ax
            self.projection = repr(ax)[1:-7].lower()
        elif ax is None or projection != self.projection:
            self.fig = plt.figure(figsize=(16, 10))
            self.ax = self.fig.add_subplot(111, projection=projection)
            self.projection = projection
        else:
            raise ValueError("`ax` must be matplotlib.Axes instance.")

        if cax is None:
            pass
        elif isinstance(cax, plt.Axes):
            self.cax = cax
        else:
            raise ValueError("`cax` must be matplotlib.Axes instance.")

        self.ax.grid(True, color='m', linestyle=':')

    #--------------------------------------------------------------------------
    def _get_coord(self, galactic):
        """Get Equatorial or Galactic coordinates.

        Parameters
        ----------
        galactic : bool
            If True, return Galactic coordinates. Otherwise, Equatorial
            coordinates.

        Returns
        -------
        x : pandas.Series
            Either Equatorial right ascension or Galactic longitude in radians.
        y : pandas.Series
            Either Equatorial declination or Galactic latitude in radians.
        """

        if galactic:
            x = self.fields['center_l'] * -1 # for plotting from +180 to -180
            y = self.fields['center_b']

        else:
            x = self.fields['center_ra']
            y = self.fields['center_dec']

        return x, y

    #--------------------------------------------------------------------------
    def _plot_other_fields(self, x, y, sel, other_kws):
        # TODO: docstring

        # default settings for plotting non-setting fields:
        if 'c' not in other_kws.keys():
            other_kws['c'] = '0.8'
        if 'marker' not in other_kws.keys():
            other_kws['marker'] = '.'
        if 's' not in other_kws.keys():
            other_kws['s'] = 2

        # plot non-setting fields:
        self.ax.scatter(
                x=x.loc[~sel], y=y.loc[~sel], **other_kws)

    #--------------------------------------------------------------------------
    def _reverse_xticklabels(self):
        """Reverse the x-axis tick labels in Galactic coordinates plots.

        Returns
        -------
        None
        """

        # get tick labels:
        labels = self.ax.get_xticklabels()
        labels_x = []
        labels_y = []
        labels_text = []

        for label in labels:
            labels_x.append(label._x)
            labels_y.append(label._y)
            labels_text.append(label._text)

        # reverse tick labels:
        labels_new = [
                plt.Text(x, y, text) for x, y, text in \
                zip(labels_x, labels_y, labels_text[::-1])]
        self.ax.set_xticklabels(labels_new)

    #--------------------------------------------------------------------------
    def set_fields(self, fields):
        """Set fields for plotting.

        Parameters
        ----------
        fields : list of dict
            List of dictionaries, where each list item provides on field's
            information. The `fields` should be queried from a SurveyPlanner
            instance throught the `SurveyPlanner().query_fields()` or
            SurveyPlanner().get_fields()` methods.

        Returns
        -------
        None
        """

        self._parse_fields(fields)
        self.fields = DataFrame(fields)
        self._galactic_coords()
        self._shift_ra()

    #--------------------------------------------------------------------------
    @abstractmethod
    def plot(
            self, galactic=False, projection='mollweide', ax=None, cax=None,
            **kwargs):
        """Plot the fields.

        Parameters
        ----------
        galactic : bool, optional
            If True, the plot will show Galactic coordinates; otherwise
            Equatorial coordinates. The default is False.
        projection : str, optional
            Projection that should be used for plotting the field coordinates.
            Must be 'mollweide', 'aitoff', 'hammer', or 'lambert'. If `ax` is
            provided, `projection` is ignored. The default is 'mollweide'.
        ax : matplotlib.Axes or None
            If None, a new Axes is created. Otherwise, the fields are plotted
            to the provided Axes. The default is None.
        cax : matplotlib.Axes or None
            If None, a new Axes is created for the colorbar. Otherwise, the
            colorbar is plotted in the provided Axes. The default is None.
        **kwargs
            Key word arguments forwarded to the plotting function.

        Returns
        -------
        self.fig : matplotlib.Figure
            The figure.
        self.ax : matplotlib.Axes
            The axes containing the plot.
        self.cax : matplotlib.Axes
            The axes containing the colorbar, if created. Otherwise, None.

        Notes
        -----
        This is an abstract method. Each child class will require its specific
        plotting method.
        """

        # check if fields exist:
        self._check_fields()

        # create figure:
        self._create_figure(projection, ax, cax)

        # select appropriate coordinates:
        x, y = self._get_coord(galactic)

        # custom code here

        # reverse x tick labels for Galactic coordinate plots:
        if galactic:
            self._reverse_xticklabels()

        return self.fig, self.ax, self.cax

#==============================================================================

class SurveyVisualizer(FieldVisualizer):
    """Visualize the survey status.
    """

    #--------------------------------------------------------------------------
    def _get_fields(self):
        """Get fields from the stored SurveyPlanner instance with the
        information that is required for plotting.

        Raises
        ------
        ValueError
            Raised, if no SurveyPlanner instance was provided at class
            instanciation.

        Returns
        -------
        None
        """

        # check if surveyplanner is available:
        if self.surveyplanner is None:
            raise ValueError(
                    "No survey planner was provided at instanciation. Cannot "
                    "query fields. Either use `set_fields()` to provide fields"
                    " for plotting or create new visualizer instance and "
                    "provide `surveyplanner`.")

        print('Querying fields..')

        self.fields = DataFrame(self.surveyplanner.query_fields())
        self._galactic_coords()
        self._shift_ra()

    #--------------------------------------------------------------------------
    def _parse_fields(self, fields):
        """Check if the provided fields contain the information that is
        required for plotting.

        Parameters
        ----------
        fields : any type
            The "fields" that are provided as `fields` at class instanciation
            or through the `set_fields()` method. Expected is a list of dict.

        Raises
        ------
        ValueError
            Raised, if `fields` is None.
            Raised, if `fields` is not list.
            Raised, if any item in `fields` list is not dict.
            Raised, if any dict in `fields` list does not contain all of the
            keys listed in `keys`.

        Returns
        -------
        bool
            True, if the `fields` input fulfills the criteria. Otherwise,
            a ValueError is raised.
        """

        keys = ['center_ra', 'center_dec', 'nobs_pending']
        error_message = "List of fields does not have the correct format. " \
                "Use SurveyPlanner().query_fields() to get a list of fields."

        return super()._parse_fields(fields, keys, error_message)

    #--------------------------------------------------------------------------
    def plot(
            self, galactic=False, projection='mollweide', ax=None, cax=None,
            plot_kws={}):
        """Plot the fields.

        Parameters
        ----------
        galactic : bool, optional
            If True, the plot will show Galactic coordinates; otherwise
            Equatorial coordinates. The default is False.
        projection : str, optional
            Projection that should be used for plotting the field coordinates.
            Must be 'mollweide', 'aitoff', 'hammer', 'lambert', or 'geo'. If
            `ax` is provided, `projection` is ignored. The default is
            'mollweide'.
        ax : matplotlib.Axes or None, optional
            If None, a new Axes is created. Otherwise, the fields are plotted
            to the provided Axes. The default is None.
        cax : matplotlib.Axes or None, optional
            If None, a new Axes is created for the colorbar. Otherwise, the
            colorbar is plotted in the provided Axes. The default is None.
        plot_kws : dict, optional
            Key word arguments forwarded to the plotting function
            `matplotlib.pyplot.scatter()`. The default is {}.

        Returns
        -------
        self.fig : matplotlib.Figure
            The figure.
        self.ax : matplotlib.Axes
            The axes containing the plot.
        self.cax : matplotlib.Axes
            The axes containing the colorbar.
        """

        # check if fields exist:
        self._check_fields()
        self.fields['status'] = np.where(
                self.fields['nobs_pending'], 'pending', 'finished')

        # create figure:
        self._create_figure(projection, ax, cax)

        # select appropriate coordinates:
        x, y = self._get_coord(galactic)

        # define colors:
        if 'cmap' in plot_kws.keys():
            cmap = plot_kws['cmap']
        else:
            cmap = colors.ListedColormap([
                    (0.32, 0.9, 0.29), (1, 0.95, 0.42)], name='green_grey')

        norm = colors.BoundaryNorm(np.arange(-0.5, 2), cmap.N)
        plot_kws.pop('cmap', None)
        plot_kws.pop('norm', None)

        # plot:
        sc = self.ax.scatter(
                x=x, y=y, c=self.fields['nobs_pending'], cmap=cmap, norm=norm,
                **plot_kws)
        cbar = plt.colorbar(sc, ticks=[0, 1], cax=self.cax)
        cbar.ax.set_yticklabels(['finished', 'pending'])

        # reverse x tick labels for Galactic coordinate plots:
        if galactic:
            self._reverse_xticklabels()

        return self.fig, self.ax, self.cax

#==============================================================================

class ObservabilityVisualizer(FieldVisualizer):
    """Visualize the observability of fields at a given date.
    """

    #--------------------------------------------------------------------------
    def __init__(self, surveyplanner=None, fields=None):
        """Visualize the observability of fields at a given date.

        Parameters
        ----------
        surveyplanner : SurveyPlanner-type, optional
            A SurveyPlanner instance that provides an access point to a
            database to query the fields that should be plotted. If not
            provided, fields have to be given either through the `fields`
            argument or through the `set_fields()` method. The default is None.
        fields : list of dict, optional
            A list of dictionaries as returned e.g. by the
            `SurveyPlanner().query_fields()` method. The default is None.

        Raises
        ------
        ValueError
            Raised, if `surveyplanner` is not a SurveyPlanner-instance.

        Returns
        -------
        None
        """

        super().__init__(surveyplanner=surveyplanner, fields=fields)
        self.night = None

    #--------------------------------------------------------------------------
    def _check_fields(self, night):
        """Check if fields have been stored for the requested night. Otherwise,
        get fields from the stored SurveyPlanner instance.

        Parameters
        ----------
        night : astropy.time.Time or str
            Check if this provided night is identical with the one stored in
            this class instance. If yes, then use the stored fields.
            Otherwise, query fields for this given night from the stored
            SurveyPlanner instance.

        Returns
        -------
        None
        """

        if self.fields is None or self.night != night:
            self._get_fields(night=night)

    #--------------------------------------------------------------------------
    def _get_fields(self, night):
        """Get fields from the stored SurveyPlanner instance for the specified
        night with the information that is required for plotting.

        Parameters
        ----------
        night : astropy.time.Time or str
            The night for which fields should be queried.

        Raises
        ------
        ValueError
            Raised, if no SurveyPlanner instance was provided at class
            instanciation.

        Returns
        -------
        None
        """

        # check if surveyplanner is available:
        if self.surveyplanner is None:
            raise ValueError(
                    "No survey planner was provided at instanciation. Cannot "
                    "query fields. Either use `set_fields()` to provide fields"
                    " for plotting or create new visualizer instance and "
                    "provide `surveyplanner`.")

        # check that night is defined:
        if type(night) not in [Time, str]:
            raise ValueError(
                    "`night` must be astropy.time.Time or str.")

        print('Querying fields..')

        self.fields = DataFrame(self.surveyplanner.query_fields(night=night))
        self._galactic_coords()
        self._shift_ra()
        self.night = night

    #--------------------------------------------------------------------------
    def _parse_fields(self, fields):
        """Check if the provided fields contain the information that is
        required for plotting.

        Parameters
        ----------
        fields : any type
            The "fields" that are provided as `fields` at class instanciation
            or through the `set_fields()` method. Expected is a list of dict.

        Raises
        ------
        ValueError
            Raised, if `fields` is None.
            Raised, if `fields` is not list.
            Raised, if any item in `fields` list is not dict.
            Raised, if any dict in `fields` list does not contain all of the
            keys listed in `keys`.

        Returns
        -------
        bool
            True, if the `fields` input fulfills the criteria. Otherwise,
            a ValueError is raised.
        """

        keys = ['center_ra', 'center_dec', 'date_start', 'date_stop',
                'duration', 'status', 'setting_duration']
        error_message = "List of fields does not have the correct format. " \
                "Use e.g. SurveyPlanner().query_fields(date='2024-01-01') " \
                "to get a list of fields."

        return super()._parse_fields(fields, keys, error_message)

    #--------------------------------------------------------------------------
    def _plot_time(self, x, y, key, ax, cax, plot_kws, other_kws):
        """Plot the field observability window start or stop time.

        Parameters
        ----------
        x : pandas.Series
            Either Equatorial right ascension or Galactic longitude in radians.
        y : pandas.Series
            Either Equatorial declination or Galactic latitude in radians.
        ax : matplotlib.Axes or None, optional
            If None, a new Axes is created. Otherwise, the fields are plotted
            to the provided Axes. The default is None.
        cax : matplotlib.Axes or None, optional
            If None, a new Axes is created for the colorbar. Otherwise, the
            colorbar is plotted in the provided Axes. The default is None.
        plot_kws : dict, optional
            Key word arguments forwarded to the plotting function
            `matplotlib.pyplot.scatter()` that plots the observable fields. The
            default is {}.
        other_kws : dict, optional
            Key word arguments forwarded to the plotting function
            `matplotlib.pyplot.scatter()` that plots the non-observable fields.
            The default is {}.

        Returns
        -------
        None
        """

        # extract observability start or stop time for color coding:
        sel = self.fields['status'] != 'not observable'
        time = self.fields.loc[sel, f'date_{key}'].values.astype(str)
        time = Time(time).mjd
        time -= np.floor(time.max())
        time *= 24

        # plot start/stop time:
        sc = self.ax.scatter(
                x=x.loc[sel], y=y.loc[sel], c=time, **plot_kws)
        cbar = plt.colorbar(sc, cax=self.cax)

        # plot non-observable fields:
        self._plot_other_fields(x, y, sel, other_kws)

        # edit colorbar:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            cbar.ax.set_ylabel(f'Observability window {key} UTC')
            cbar.ax.yaxis.set_major_locator(MultipleLocator(1))
            labels = [text.get_text().replace('−', '-') \
                      for text in cbar.ax.get_yticklabels()]
            labels = np.array(labels, dtype=float)
            labels = np.mod(labels, 24).astype(int)
            cbar.ax.set_yticklabels(labels)

    #--------------------------------------------------------------------------
    def _plot_duration(self, x, y, ax, cax, plot_kws, other_kws):
        """Plot the field observability window duration.

        Parameters
        ----------
        x : pandas.Series
            Either Equatorial right ascension or Galactic longitude in radians.
        y : pandas.Series
            Either Equatorial declination or Galactic latitude in radians.
        ax : matplotlib.Axes or None, optional
            If None, a new Axes is created. Otherwise, the fields are plotted
            to the provided Axes. The default is None.
        cax : matplotlib.Axes or None, optional
            If None, a new Axes is created for the colorbar. Otherwise, the
            colorbar is plotted in the provided Axes. The default is None.
        plot_kws : dict, optional
            Key word arguments forwarded to the plotting function
            `matplotlib.pyplot.scatter()` that plots the observable fields. The
            default is {}.
        other_kws : dict, optional
            Key word arguments forwarded to the plotting function
            `matplotlib.pyplot.scatter()` that plots the non-observable fields.
            The default is {}.

        Returns
        -------
        None
        """

        # extract duration for color coding:
        sel = self.fields['status'] != 'not observable'
        duration = self.fields.loc[sel, 'duration'].values * 24

        # plot start/stop time:
        sc = self.ax.scatter(
                x=x.loc[sel], y=y.loc[sel], c=duration, **plot_kws)
        cbar = plt.colorbar(sc, cax=self.cax)
        cbar.ax.set_ylabel('Observability window duration (hours)')

        # plot non-observable fields:
        self._plot_other_fields(x, y, sel, other_kws)

    #--------------------------------------------------------------------------
    def _plot_setting_duration(self, x, y, ax, cax, plot_kws, other_kws):
        """Plot the field setting duration.

        Parameters
        ----------
        x : pandas.Series
            Either Equatorial right ascension or Galactic longitude in radians.
        y : pandas.Series
            Either Equatorial declination or Galactic latitude in radians.
        ax : matplotlib.Axes or None, optional
            If None, a new Axes is created. Otherwise, the fields are plotted
            to the provided Axes. The default is None.
        cax : matplotlib.Axes or None, optional
            If None, a new Axes is created for the colorbar. Otherwise, the
            colorbar is plotted in the provided Axes. The default is None.
        plot_kws : dict, optional
            Key word arguments forwarded to the plotting function
            `matplotlib.pyplot.scatter()` that plots the setting fields. The
            default is {}.
        other_kws : dict, optional
            Key word arguments forwarded to the plotting function
            `matplotlib.pyplot.scatter()` that plots the non-setting fields.
            The default is {}.

        Returns
        -------
        None
        """

        # extract duration for color coding:
        sel = self.fields['status'] == 'setting'
        duration = self.fields.loc[sel, 'setting_duration'].values

        # plot start/stop time:
        sc = self.ax.scatter(
                x=x.loc[sel], y=y.loc[sel],
                c=duration, vmin=0, vmax=365, **plot_kws)
        cbar = plt.colorbar(sc, cax=self.cax)
        cbar.ax.set_ylabel('Field setting duration (days)')

        # plot non-setting fields:
        self._plot_other_fields(x, y, sel, other_kws)

    #--------------------------------------------------------------------------
    def _plot_status(self, x, y, ax, cax, plot_kws):
        """Plot the field status.

        Parameters
        ----------
        x : pandas.Series
            Either Equatorial right ascension or Galactic longitude in radians.
        y : pandas.Series
            Either Equatorial declination or Galactic latitude in radians.
        ax : matplotlib.Axes or None, optional
            If None, a new Axes is created. Otherwise, the fields are plotted
            to the provided Axes. The default is None.
        cax : matplotlib.Axes or None, optional
            If None, a new Axes is created for the colorbar. Otherwise, the
            colorbar is plotted in the provided Axes. The default is None.
        plot_kws : dict, optional
            Key word arguments forwarded to the plotting function
            `matplotlib.pyplot.scatter()`. The default is {}.

        Returns
        -------
        None
        """

        # extract observability status for color coding:
        status = {'not observable': 0, 'rising': 1, 'plateauing': 2,
                  'setting': 3}
        c = list(map(lambda s: status[s], self.fields['status']))

        # create 3-color colormap:
        if 'cmap' not in plot_kws.keys():
            plot_kws['cmap'] = plt.cm.rainbow_r

        norm = colors.BoundaryNorm(np.arange(-0.5, 4.5), plot_kws['cmap'].N)

        # plot fields:
        sc = self.ax.scatter(
                x, y, c=c, norm=norm, marker='o', **plot_kws)
        cbar = plt.colorbar(sc, ticks=range(4), cax=self.cax)
        cbar.ax.set_yticklabels([
                'not observable', 'rising', 'plateauing', 'setting'])

    #--------------------------------------------------------------------------
    def plot(
            self, obs, night=None, galactic=False, projection='mollweide',
            ax=None, cax=None, plot_kws={}, other_kws={}):
        """Plot the fields.

        Parameters
        ----------
        obs : str
            Observing window property that should be displayed as color.
            Choose 'status', 'start', 'stop', 'duration', or
            'setting_duration'.
        night : astropy.time.Time, str, or None, optional
            The night for which fields should be plotted. If fields have been
            provided through the `set_fields()` method, this argument can be
            left as None. If a date is provided and it does not match the date
            associated with the already stored fields, fields are queried. The
            default is None.
        galactic : bool, optional
            If True, the plot will show Galactic coordinates; otherwise
            Equatorial coordinates. The default is False.
        projection : str, optional
            Projection that should be used for plotting the field coordinates.
            Must be 'mollweide', 'aitoff', 'hammer', 'lambert', or 'geo'. If
            `ax` is provided, `projection` is ignored. The default is
            'mollweide'.
        ax : matplotlib.Axes or None, optional
            If None, a new Axes is created. Otherwise, the fields are plotted
            to the provided Axes. The default is None.
        cax : matplotlib.Axes or None, optional
            If None, a new Axes is created for the colorbar. Otherwise, the
            colorbar is plotted in the provided Axes. The default is None.
        plot_kws : dict, optional
            Key word arguments forwarded to the plotting function
            `matplotlib.pyplot.scatter()` that plots the fields of interest.
            The default is {}.
        other_kws : dict, optional
            Key word arguments forwarded to the plotting function
            `matplotlib.pyplot.scatter()` that plots the remaining fields. E.g.
            the non-observable or non setting fields. The default is {}.

        Returns
        -------
        self.fig : matplotlib.Figure
            The figure.
        self.ax : matplotlib.Axes
            The axes containing the plot.
        self.cax : matplotlib.Axes
            The axes containing the colorbar.
        """

        # check if fields exist:
        self._check_fields(night=night)

        # create figure:
        self._create_figure(projection, ax, cax)

        # select appropriate coordinates:
        x, y = self._get_coord(galactic)

        if obs in ['start', 'stop']:
            cax = self._plot_time(x, y, obs, ax, cax, plot_kws, other_kws)
        elif obs == 'duration':
            cax = self._plot_duration(x, y, ax, cax, plot_kws, other_kws)
        elif obs == 'setting_duration':
            cax = self._plot_setting_duration(
                    x, y, ax, cax, plot_kws, other_kws)
        elif obs == 'status':
            cax = self._plot_status(x, y, ax, cax, plot_kws)
        else:
            raise ValueError(
                    "`obs` must be 'start', 'stop', 'duration', " \
                    "'setting_duration', or 'status'.")

        # reverse x tick labels for Galactic coordinate plots:
        if galactic:
            self._reverse_xticklabels()

        return self.fig, self.ax, self.cax

#==============================================================================

class AnnualObservabilityVisualizer(FieldVisualizer):
    """Visualize the annual observability of fields.
    """

    #--------------------------------------------------------------------------
    def _get_fields(self):
        """Get fields from the stored SurveyPlanner instance with the
        information that is required for plotting.

        Raises
        ------
        ValueError
            Raised, if no SurveyPlanner instance was provided at class
            instanciation.

        Returns
        -------
        None
        """
        # check if surveyplanner is available:
        if self.surveyplanner is None:
            raise ValueError(
                    "No survey planner was provided at instanciation. Cannot "
                    "query fields. Either use `set_fields()` to provide fields"
                    " for plotting or create new visualizer instance and "
                    "provide `surveyplanner`.")

        print('Querying fields..')

        surveyplanner = deepcopy(self.surveyplanner)
        prioritizer = PrioritizerAnnualAvailability(surveyplanner.dbname)
        surveyplanner.set_prioritizer(prioritizer)
        fields = surveyplanner.query_fields()
        surveyplanner._get_annual_observability(fields)
        self.fields = DataFrame(fields)
        self._galactic_coords()
        self._shift_ra()

    #--------------------------------------------------------------------------
    def _parse_fields(self, fields):
        """Check if the provided fields contain the information that is
        required for plotting.

        Parameters
        ----------
        fields : any type
            The "fields" that are provided as `fields` at class instanciation
            or through the `set_fields()` method. Expected is a list of dict.

        Raises
        ------
        ValueError
            Raised, if `fields` is None.
            Raised, if `fields` is not list.
            Raised, if any item in `fields` list is not dict.
            Raised, if any dict in `fields` list does not contain all of the
            keys listed in `keys`.

        Returns
        -------
        bool
            True, if the `fields` input fulfills the criteria. Otherwise,
            a ValueError is raised.
        """

        keys = ['center_ra', 'center_dec', 'annual_availability']
        error_message = "List of fields does not have the correct format. " \
                "Use e.g. SurveyPlanner().plan('2024-01-01') with the " \
                "PrioritizerAnnualAvailability added as prioritizer and " \
                "then SurveyPlanner().get_fields() to get a list of fields."

        return super()._parse_fields(fields, keys, error_message)

    #--------------------------------------------------------------------------
    def plot(
            self, rate=False, galactic=False, projection='mollweide',
            ax=None, cax=None, plot_kws={}):
        """Plot the fields.

        Parameters
        ----------
        rate : bool, optional
            If True, plot annual availability rate. Otherwise, plot count of
            available days per year. The default is False.
        galactic : bool, optional
            If True, the plot will show Galactic coordinates; otherwise
            Equatorial coordinates. The default is False.
        projection : str, optional
            Projection that should be used for plotting the field coordinates.
            Must be 'mollweide', 'aitoff', 'hammer', 'lambert', or 'geo'. If
            `ax` is provided, `projection` is ignored. The default is
            'mollweide'.
        ax : matplotlib.Axes or None, optional
            If None, a new Axes is created. Otherwise, the fields are plotted
            to the provided Axes. The default is None.
        cax : matplotlib.Axes or None, optional
            If None, a new Axes is created for the colorbar. Otherwise, the
            colorbar is plotted in the provided Axes. The default is None.
        plot_kws : dict, optional
            Key word arguments forwarded to the plotting function
            `matplotlib.pyplot.scatter()`. The default is {}.

        Returns
        -------
        self.fig : matplotlib.Figure
            The figure.
        self.ax : matplotlib.Axes
            The axes containing the plot.
        self.cax : matplotlib.Axes
            The axes containing the colorbar.
        """

        # check if fields exist:
        self._check_fields()

        # create figure:
        self._create_figure(projection, ax, cax)

        # select appropriate coordinates:
        x, y = self._get_coord(galactic)

        # extract annual availability for color coding:
        availability = self.fields.annual_availability.apply(Series)

        if rate:
            availability = availability['available_rate']
            label = 'Annual availability rate'
        else:
            availability = availability['available_days']
            label = 'Annual availability (days)'

        # plot start/stop time:
        sc = self.ax.scatter(x=x, y=y, c=availability, **plot_kws)
        cbar = plt.colorbar(sc, cax=self.cax)
        cbar.ax.set_ylabel(label)

        # reverse x tick labels for Galactic coordinate plots:
        if galactic:
            self._reverse_xticklabels()

        return self.fig, self.ax, self.cax

#==============================================================================
