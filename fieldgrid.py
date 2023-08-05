# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Sky fields for the Pasiphae survey.
"""

from abc import  ABCMeta, abstractmethod
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
import numpy as np
from textwrap import dedent

__author__ = "Sebastian Kiehlmann"
__credits__ = ["Sebastian Kiehlmann"]
__license__ = "BSD3"
__version__ = "0.1"
__maintainer__ = "Sebastian Kiehlmann"
__email__ = "skiehlmann@mail.de"
__status__ = "Production"

#==============================================================================
# FUNCTIONS
#==============================================================================

def cart_to_sphere(x, y, z):
    """Transform cartesian to spherical coordinates.

    Parameters
    ----------
    x : np.ndarray or float
        x-coordinates to transform.
    y : np.ndarray or float
        y-coordinates to transform.
    z : np.ndarray or float
        z-coordinates to transform.

    Returns
    -------
    ra : np.ndarray or float
        Right ascension in radians.
    dec : np.ndarray or float
        Declination in radians.
    """

    r = np.sqrt(x**2 + y**2 + z**2)
    za = np.arccos(z / r)
    dec = np.pi / 2. - za
    ra = np.arctan2(y, x)
    ra = np.mod(ra, 2*np.pi)

    return ra, dec

#--------------------------------------------------------------------------
def sphere_to_cart(ra, dec):
    """Transform spherical to cartesian coordinates.

    Parameters
    ----------
    ra : np.ndarray or float
        Right ascension(s) in radians.
    dec : np.ndarray or float
        Declination(s) in radians.

    Returns
    -------
    x : np.ndarray or float
        x-coordinate(s).
    y : np.ndarray or float
        y-coordinate(s).
    z : np.ndarray or float
        z-coordinate(s).
    """

    za = np.pi / 2. - dec
    x = np.sin(za) * np.cos(ra)
    y = np.sin(za) * np.sin(ra)
    z = np.cos(za)

    return x, y, z

#--------------------------------------------------------------------------
def rot_tilt(x, y, z, tilt):
    """Rotate around x-axis by tilt angle.

    Parameters
    ----------
    x : np.ndarray or float
        x-coordinates to rotate.
    y : np.ndarray or float
        y-coordinates to rotate.
    z : np.ndarray or float
        z-coordinates to rotate.
    tilt : float
        Angle in radians by which the coordinates are rotated.

    Returns
    -------
    x_rot : np.ndarray or float
        Rotated x-coordinates.
    y_rot : np.ndarray or float
        Rotated y-coordinates.
    z_rot : np.ndarray or float
        Rotated z-coordinates.
    """

    x_rot = x
    y_rot = y * np.cos(tilt) - z * np.sin(tilt)
    z_rot = y * np.sin(tilt) + z * np.cos(tilt)

    return x_rot, y_rot, z_rot

#--------------------------------------------------------------------------
def rot_dec(x, y, z, dec):
    """Rotate around y-axis by declination angle.

    Parameters
    ----------
    x : np.ndarray or float
        x-coordinates to rotate.
    y : np.ndarray or float
        y-coordinates to rotate.
    z : np.ndarray or float
        z-coordinates to rotate.
    dec : float
        Angle in radians by which the coordinates are rotated.

    Returns
    -------
    x_rot : np.ndarray or float
        Rotated x-coordinates.
    y_rot : np.ndarray or float
        Rotated y-coordinates.
    z_rot : np.ndarray or float
        Rotated z-coordinates.
    """

    dec = -dec
    x_rot = x * np.cos(dec) + z * np.sin(dec)
    y_rot = y
    z_rot = -x * np.sin(dec) + z * np.cos(dec)

    return x_rot, y_rot, z_rot

#--------------------------------------------------------------------------
def rot_ra(x, y, z, ra):
    """Rotate around z-axis by right ascension angle.

    Parameters
    ----------
    x : np.ndarray or float
        x-coordinates to rotate.
    y : np.ndarray or float
        y-coordinates to rotate.
    z : np.ndarray or float
        z-coordinates to rotate.
    ra : float
        Angle in radians by which the coordinates are rotated.

    Returns
    -------
    x_rot : np.ndarray or float
        Rotated x-coordinates.
    y_rot : np.ndarray or float
        Rotated y-coordinates.
    z_rot : np.ndarray or float
        Rotated z-coordinates.
    """

    x_rot = x * np.cos(ra) - y * np.sin(ra)
    y_rot = x * np.sin(ra) + y * np.cos(ra)
    z_rot = z

    return x_rot, y_rot, z_rot

#==============================================================================
# CLASSES
#==============================================================================

class FieldGrid(metaclass=ABCMeta):
    """Separation of the sky into fields."""

    #--------------------------------------------------------------------------
    def __init__(
            self, fov, overlap_ns=0, overlap_ew=0, tilt=0,
            dec_lim_north=np.pi/2, dec_lim_south=-np.pi/2, gal_lat_lim=0,
            gal_lat_lim_strict=False, verbose=0):
        """Create FieldGrid instance.

        Parameters
        ----------
        fov : float
            Field of view side length in radians.
        overlap_ns : float, optional
            Overlap between neighboring fields in North-South direction in
            radian. The default is 0.
        overlap_ew : float, optional
            Overlap between neighboring fields in East-West direction in
            radian. The default is 0.
        tilt : float, optional
            Tilt of the field of view in radians. The default is 0.
        dec_lim_north : float, optional
            Northern declination limit in radians. Fields North of this limit
            are excluded. The default is np.pi/2.
        dec_lim_south : float, optional
            Southern declination limit in radians. Fields South of this limit
            are excluded. The default is -np.pi/2.
        gal_lat_lim : float, optional
            Galactic latitude limit in radians. If the limit is X, fields with
            Galactic latitude in [-X, X] are excluded. The default is 0.
        gal_lat_lim_strict : bool, optional
            If False, fields will be excluded when their field center is
            within the Galactic latitude limits. If True, field will be
            excluded only if all of the field corners are within the Galactic
            latitude limits. The default is False.
        verbose : int, optional
            Set level of verbosity. 0 only gives most essential information.
            1 provides more details; 2 even more. -1 turns of any messages. The
            default is 0.

        Raises
        ------
        ValueError
            Raise with the North-South or East-West overlap exceeds the field
            of view size.
            Raised when the Northern or Southern declination limits exceed
            pi/2.
            Raised when Southern declination limit is higher than Southern
            declination limit.

        Returns
        -------
        None
        """

        # check input:
        if overlap_ns >= fov:
            raise ValueError("Overlap must be smaller than field of view.")
        if overlap_ew >= fov:
            raise ValueError("Overlap must be smaller than field of view.")
        if abs(dec_lim_north) > np.pi / 2.:
            raise ValueError("Northern declination limit cannot exceed pi/2.")
        if abs(dec_lim_south) > np.pi / 2.:
            raise ValueError("Southern declination limit cannot exceed -pi/2.")
        if dec_lim_south > dec_lim_north:
            raise ValueError(
                    "Northern declination must be higher than Southern "
                    "declination limit.")

        self.fov = fov
        self.overlap_ns = overlap_ns
        self.overlap_ew = overlap_ew
        self.tilt = tilt
        self.dec_lim_north = dec_lim_north
        self.dec_lim_south = dec_lim_south
        self.gal_lat_lim = gal_lat_lim
        self.gal_lat_lim_strict = gal_lat_lim_strict
        self.center_ras = None
        self.center_decs = None
        self.corner_ras = None
        self.corner_decs = None
        self.verbose = verbose

        self._create_fields()

    #--------------------------------------------------------------------------
    def __len__(self):
        """Returns length of field grid. I.e. number of fields."""

        return self.center_ras.shape[0] if self.center_ras is not None else 0

    #--------------------------------------------------------------------------
    @abstractmethod
    def __str__(self):
        """Return information about the FieldGrid instance.

        Returns
        -------
        info : str
            Description of main properties.
        """

        info = dedent("""\
            FieldGrid : Field grid
            Field of view:    {0:7.4f} deg
            Overlap N-S       {1:7.4f} deg
            Overlap E-W       {2:7.4f} deg
            Tilt:             {3:+7.4f} deg
            Gal. lat. lim:    {4:s}
            Dec. lim. N:      {5:s}
            Dec. lim. S:      {6:s}
            Number of fields: {7:d}""".format(
                np.degrees(self.fov), np.degrees(self.overlap_ns),
                np.degrees(self.overlap_ew), np.degrees(self.tilt),
                f'{np.degrees(self.gal_lat_lim):7.4f} deg' \
                    if self.gal_lat_lim else 'None',
                f'{np.degrees(self.dec_lim_north):7.4f} deg' \
                    if self.dec_lim_north else 'None',
                f'{np.degrees(self.dec_lim_south):7.4f} deg' \
                    if self.dec_lim_south else 'None',
                self.center_ras.shape[0]))

        return info

    #--------------------------------------------------------------------------
    def _field_corners_init(self, fov):
        """Create field corner points in cartesian coordinates.

        Parameters
        ----------
        fov : float
            Field of view in radians.

        Returns
        -------
        x : numpy.ndarray
            Cartesian x-coordinates of the field corner points.
        y : numpy.ndarray
            Cartesian y-coordinates of the field corner points.
        z : numpy.ndarray
            Cartesian z-coordinates of the field corner points.
        """

        diff = np.tan(fov / 2.)
        x = np.ones(4)
        y = np.array([-diff, diff, diff, -diff])
        z = np.array([-diff, -diff, diff, diff])

        return x, y, z

    #--------------------------------------------------------------------------
    def _field_corners_rot(self, fov, tilt=0, center_ra=0, center_dec=0):
        """Calculate field corner points at specified field center coordinates.

        Parameters
        ----------
        fov : float
            Field of view side length in radians.
        tilt : float, optional
            Tilt of the field of view in radians. The default is 0.
        center_ra : float, optional
            Field center right ascension. The default is 0.
        center_dec : float, optional
            Field center declination. The default is 0.

        Returns
        -------
        corner_ras : np.ndarray or float
            Corner right ascension in radians.
        corner_decs : np.ndarray or float
            Corner declination in radians.
        """

        x, y, z = self._field_corners_init(fov)
        x, y, z = rot_tilt(x, y, z, tilt)
        x, y, z = rot_dec(x, y, z, center_dec)
        x, y, z = rot_ra(x, y, z, center_ra)
        corner_ras, corner_decs = cart_to_sphere(x, y, z)

        return corner_ras, corner_decs

    #--------------------------------------------------------------------------
    def _calc_field_corners(self):
        """Calculate field corners.

        Returns
        -------
        None
        """

        if self.verbose > 0:
            print('  Calculate field corners..')

        corner_ras = []
        corner_decs = []
        n_fields = len(self.center_ras)

        # iterate through field centers:
        for i, (ra, dec) in enumerate(zip(self.center_ras, self.center_decs)):
            print(f'\r    Field {i+1} of {n_fields} ' \
                  f'({i/n_fields*100.:.1f} %)..',
                  end='')

            ras, decs = self._field_corners_rot(
                    self.fov, tilt=self.tilt, center_ra=ra, center_dec=dec)
            corner_ras.append(ras)
            corner_decs.append(decs)

        self.corner_ras = np.array(corner_ras)
        self.corner_decs = np.array(corner_decs)

        print('\r    Done                                                    ')

    #--------------------------------------------------------------------------
    @abstractmethod
    def _calc_field_centers(self):
        """Calculate field centers.

        Returns
        -------
        None

        Notes
        -----
        This is an abstract method that needs to be specified in the
        sub-classes. The field centers depend on the specific grid layout that
        is specified in sub-classes of this generic parent class.
        """

        pass

    #--------------------------------------------------------------------------
    def in_galactic_plane(
            self, gal_lat_lim, center_ras=None, center_decs=None,
            corner_ras=None, corner_decs=None, verbose=0):
        """Check which fields are located within the Galactic latitude limits.

        Parameters
        ----------
        gal_lat_lim : float, optional
            Galactic latitude limit in radians. If the limit is X, fields with
            Galactic latitude in [-X, X] are flagged.
        center_ras : np.ndarray
            Field center right ascensions in radians. The default is None
        center_decs : np.ndarray
            Field center declinations in radians. The default is None
        corner_ras : np.ndarray
            Field corner right ascensions in radians. The default is None
        corner_decs : np.ndarray
            Field corner declinations in radians. The default is None
        verbose : TYPE, optional
            Set level of verbosity. Information is printed by this method only
            if verbose > 0. The default is 0.

        Returns
        -------
        np.ndarray
            Boolean array. Fields within the Galactic plane limits are marked
            as True, otherwise as False.

        Notes
        -----
        How the fields are identified as within the limits depends on the
        `gal_lat_lim_strict` argument set with the class initialization.

        If simple, i.e. `gal_lat_lim_strict=False`, fields will be flagged
        when their field center is within the Galactic latitude limits. If
        strict, i.e. `gal_lat_lim_strict=True`, field will be flagged only if
        all of the field corners are within the Galactic latitude limits.
        """

        # check input:
        if center_ras is not None and center_decs is not None:
            gal_lat_lim_strict = False
            n_fields = center_ras.shape[0]
        elif corner_ras is not None and corner_decs is not None:
            gal_lat_lim_strict = True
            n_fields = corner_ras.shape[0]

        # stop if Galactic latitude limit is 0:
        if not gal_lat_lim:
            return np.zeros(n_fields)

        # otherwise, start checking:
        if verbose > 0:
            print('  Identify fields in Galactic plane..')

        # consider field centers:
        if not gal_lat_lim_strict:
            coord = SkyCoord(
                    center_ras, center_decs, unit='rad', frame='icrs')
            coord = coord.transform_to('galactic')
            sel = np.logical_and(
                    coord.b.rad < gal_lat_lim,
                    coord.b.rad > -gal_lat_lim)

        # consider field corners:
        else:
            coord = SkyCoord(
                    self.corner_ras, self.corner_decs, unit='rad',
                    frame='icrs')
            coord = coord.transform_to('galactic')
            sel = np.logical_and(
                    np.all(coord.b.rad < self.gal_lat_lim, axis=1),
                    np.all(coord.b.rad > -self.gal_lat_lim, axis=1))

        return sel

    #--------------------------------------------------------------------------
    def _avoid_galactic_plane(self):
        """Remove fiels that are located in the Galactic plane limits.

        Returns
        -------
        None

        Notes
        -----
        The Galactic plane limits are given at the initialization of the class.
        How the fields are identified as within the limits depends on the
        `gal_lat_lim_strict` argument set with the class initialization.
        Verbosity: This method only prints out information if the verbosity set
        with the class initialization is > 1.
        """

        if not self.gal_lat_lim:
            return None

        # consider field centers:
        if not self.gal_lat_lim_strict:
            sel = self.in_galactic_plane(
                    self.gal_lat_lim, center_ras=self.center_ras,
                    center_decs=self.center_decs, verbose=self.verbose)

        # consider field corners:
        else:
            sel = self.in_galactic_plane(
                    self.gal_lat_lim, corner_ras=self.corner_ras,
                    corner_decs=self.corner_decs, verbose=self.verbose)


        if self.verbose > 1:
            print('    Galactic latitude limit: +/-{0:.1f} deg'.format(
                    np.degrees(self.gal_lat_lim)))

            if self.gal_lat_lim_strict:
                print('    Application: field corners')
            else:
                print('    Application: field centers')

            print(f'    Fields removed:   {np.sum(sel)}')
            print(f'    Fields remaining: {np.sum(~sel)}')

        self.center_ras = self.center_ras[~sel]
        self.center_decs = self.center_decs[~sel]
        self.corner_ras = self.corner_ras[~sel]
        self.corner_decs = self.corner_decs[~sel]

    #--------------------------------------------------------------------------
    def _create_fields(self):
        """Create fields.

        Returns
        -------
        None

        Notes
        -----
        Verbosity: This method prints out information if the verbosity set
        with the class initialization is >= 0. This is the most basic level of
        verbosity.
        """

        if self.verbose > -1:
            print('Create fields..')

        self._calc_field_centers()
        self._calc_field_corners()
        self._avoid_galactic_plane()

        if self.verbose > -1:
            print(f'Final number of fields: {self.center_ras.size}')

    #--------------------------------------------------------------------------
    def get_center_coords(self):
        """Return the field center coordinates.

        Returns
        -------
        self.center_ras : np.ndarray
            Field center right ascensions.
        self.center_decs : np.ndarray
            Field center declinations.
        """

        return self.center_ras, self.center_decs

    #--------------------------------------------------------------------------
    def get_corner_coords(self):
        """Return the field corner coordinates.

        Returns
        -------
        self.center_ras : np.ndarray
            Field corner right ascensions. Nx4 dimensional array. First axis
            related to the N fields. Second axis to the four field corners.
        self.center_decs : np.ndarray
            Field corner declinations. Nx4 dimensional array. First axis
            related to the N fields. Second axis to the four field corners.
        """

        return self.corner_ras, self.corner_decs

#==============================================================================

class FieldGridIsoLat(FieldGrid):
    """Separation of the sky into fields, placing fields onto isolatitudinal
    rings.
    """

    grid_type = 'isolatitudinal grid'

    #--------------------------------------------------------------------------
    def __init__(
            self, fov, overlap_ns=0, overlap_ew=0, tilt=0,
            dec_lim_north=np.pi/2, dec_lim_south=-np.pi/2, gal_lat_lim=0,
            gal_lat_lim_strict=False, verbose=0):
        """Create FieldGridIsoLat instance.

        Parameters
        ----------
        fov : float
            Field of view side length in radians.
        overlap_ns : float, optional
            Overlap between neighboring fields in North-South direction in
            radian. The default is 0.
        overlap_ew : float, optional
            Overlap between neighboring fields in East-West direction in
            radian. The default is 0.
        tilt : float, optional
            Tilt of the field of view in radians. The default is 0.
        dec_lim_north : float, optional
            Northern declination limit in radians. Fields North of this limit
            are excluded. The default is np.pi/2.
        dec_lim_south : float, optional
            Southern declination limit in radians. Fields South of this limit
            are excluded. The default is -np.pi/2.
        gal_lat_lim : float, optional
            Galactic latitude limit in radians. If the limit is X, fields with
            Galactic latitude in [-X, X] are excluded. The default is 0.
        gal_lat_lim_strict : bool, optional
            If False, fields will be excluded when their field center is
            within the Galactic latitude limits. If True, field will be
            excluded only if all of the field corners are within the Galactic
            latitude limits. The default is False.
        verbose : int, optional
            Set level of verbosity. 0 only gives most essential information.
            1 provides more details; 2 even more. -1 turns of any messages. The
            default is 0.

        Raises
        ------
        ValueError
            Raise with the North-South or East-West overlap exceeds the field
            of view size.
            Raised when the Northern or Southern declination limits exceed
            pi/2.
            Raised when Southern declination limit is higher than Southern
            declination limit.

        Returns
        -------
        None
        """

        super(FieldGridIsoLat, self).__init__(
                fov, overlap_ns=overlap_ns, overlap_ew=overlap_ew, tilt=tilt,
                dec_lim_north=dec_lim_north, dec_lim_south=dec_lim_south,
                gal_lat_lim=gal_lat_lim, gal_lat_lim_strict=gal_lat_lim_strict,
                verbose=verbose)

    #--------------------------------------------------------------------------
    def __str__(self):
        """Return information about the FieldGrid instance.

        Returns
        -------
        info : str
            Description of main properties.
        """

        info = dedent("""\
            FieldGridIsoLat : Iso-latitudinal field grid
            Field of view:    {0:7.4f} deg
            Overlap N-S       {1:7.4f} deg
            Overlap E-W       {2:7.4f} deg
            Tilt:             {3:+7.4f} deg
            Gal. lat. lim:    {4:s}
            Dec. lim. N:      {5:s}
            Dec. lim. S:      {6:s}
            Number of fields: {7:d}""".format(
                np.degrees(self.fov), np.degrees(self.overlap_ns),
                np.degrees(self.overlap_ew), np.degrees(self.tilt),
                f'{np.degrees(self.gal_lat_lim):7.4f} deg' \
                    if self.gal_lat_lim else 'None',
                f'{np.degrees(self.dec_lim_north):7.4f} deg' \
                    if self.dec_lim_north else 'None',
                f'{np.degrees(self.dec_lim_south):7.4f} deg' \
                    if self.dec_lim_south else 'None',
                self.center_ras.shape[0]))

        return info

    #--------------------------------------------------------------------------
    def _split_declination(self):
        """Split declination into rings.

        Returns
        -------
        None

        Notes
        -----
        Verbosity: This method prints out information if the verbosity set
        with the class initialization is > 1. This is the most detailed level
        of verbosity.
        """

        dec_range = self.dec_lim_north - self.dec_lim_south
        field_range = self.fov - self.overlap_ns
        n = (dec_range - self.overlap_ns) / field_range

        # round when n (almost) is an interger number:
        if np.isclose(np.mod(n, 1), 0):
            n = int(np.round(n))

        # otherwise ceil:
        else:
            n = int(np.ceil(n))

        if self.verbose > 1:
            print(f'    Number of declination circles: {n}')

        # calculate declinations of isolatitudinal rings:
        dec_range_real = n * field_range + self.overlap_ns
        offset = (dec_range_real - dec_range) / 2.
        dec0 = self.dec_lim_south + self.fov / 2. - offset
        dec1 = dec0 + field_range * (n - 1)
        self.declinations = np.linspace(dec0, dec1, n)

    #--------------------------------------------------------------------------
    def _close_gaps(self, ras, decs, dec, n):
        """Close gaps in declination rings.

        Parameters
        ----------
        ras : np.ndarray
            Field center right ascensions in radians.
        decs : np.ndarray
            Field center declinations in radians.
        dec : float
            Declination of the declination ring in radians. Must equal all
            entries in `decs`.
        n : int
            Number of fields in declination ring. Must equal the shape of `ras`
            and `decs`.

        Returns
        -------
        ras : np.ndarray
            Field center right ascensions in radians.
        decs : np.ndarray
            Field center declinations in radians.
        n : int
            New number of fields in declination ring.

        Notes
        -----
        At higher declinations, where neighboring fields in a declination ring
        are tilted relative to another, gaps can occur. This method takes an
        initial set of fields and adds fields (if necessary) until all gaps are
        closed.
        Verbosity: This method prints out information if the verbosity set
        with the class initialization is > 1. This is the most detailed level
        of verbosity.
        """

        # get first two field's corners:
        field0_corner_ras, field0_corner_decs = self._field_corners_rot(
                self.fov, tilt=self.tilt, center_ra=ras[0], center_dec=decs[0])
        field1_corner_ras, field1_corner_decs = self._field_corners_rot(
                self.fov, tilt=self.tilt, center_ra=ras[1], center_dec=decs[1])

        # select two field corners - fields in the South:
        if field0_corner_decs[0] < 0:
            i = 2
            j = 3

        # select two field corners - fields in the North:
        else:
            i = 1
            j = 0

        ra0 = field0_corner_ras[i]
        ra1 = field1_corner_ras[j]

        if ra0 < ra1 and self.verbose > 1:
            print('Closing gaps. ', end='')

        # increase number of fields until gaps are removed:
        while ra0 < ra1:
            n += 1
            ras = np.linspace(0, 2.*np.pi, n+1)[:-1]
            decs = np.ones(n) * dec

            ra0 = self._field_corners_rot(
                    self.fov, tilt=self.tilt, center_ra=ras[0],
                    center_dec=decs[0])[0][i]
            ra1 = self._field_corners_rot(
                    self.fov, tilt=self.tilt, center_ra=ras[1],
                    center_dec=decs[1])[0][j]

        return ras, decs, n

    #--------------------------------------------------------------------------
    def _field_centers_along_dec(self, dec, n_min=3):
        """Create fields along a declination ring.

        Parameters
        ----------
        dec : float
            Declination of the declination ring in radians.
        n_min : int, optional
            Minimum number of fields required. The default is 3.

        Returns
        -------
        ras : np.ndarray
            Field center right ascensions.
        decs : np.ndarray
            Field center declinations.

        Notes
        -----
        Verbosity: This method prints out information if the verbosity set
        with the class initialization is > 1. This is the most detailed level
        of verbosity.
        """

        if self.verbose > 1:
            print(f'    Dec: {np.degrees(dec):+6.2f} deg. ', end='')

        # create one field at the pole:
        if np.isclose(np.absolute(dec), np.pi/2.):
            n = 1
            ras = np.array([0])
            decs = np.array([dec])

        # otherwise split isolatitudinal ring into n fields:
        else:
            n = int(np.ceil(
                    2 * np.pi / (self.fov - self.overlap_ew)) * np.cos(dec))

            if n < n_min:
                n = n_min

            ras = np.linspace(0, 2.*np.pi, n+1)[:-1]
            decs = np.ones(n) * dec
            ras, decs, n = self._close_gaps(ras, decs, dec, n)

        if self.verbose > 1:
            print(f'Number of fields: {n:6d}')

        return ras, decs

    #--------------------------------------------------------------------------
    def _calc_field_centers(self):
        """Calculate field centers.

        Returns
        -------
        None

        Notes
        -----
        This overwrites the generic method from the parent class and implements
        the creation of this specific field grid.
        Verbosity: This method prints out information if the verbosity set
        with the class initialization is > 0. This is the more detailed level
        of verbosity.
        """

        if self.verbose > 0:
            print('  Calculate field centers..')

        self._split_declination()
        center_ra = []
        center_dec = []

        for dec in self.declinations:
            ra, dec = self._field_centers_along_dec(dec)
            center_ra.append(ra)
            center_dec.append(dec)

        self.center_ras = np.concatenate(center_ra)
        self.center_decs = np.concatenate(center_dec)

#==============================================================================

class FieldGridGrtCirc(FieldGrid):
    """Separation of the sky into fields, placing fields on great circles.
    """

    grid_type = 'tilted great circle grid'

    #--------------------------------------------------------------------------
    def __init__(
            self, fov, overlap_ns=0, overlap_ew=0, tilt=0,
            dec_lim_north=np.pi/2, dec_lim_south=-np.pi/2,
            dec_lim_strict=False, gal_lat_lim=0, gal_lat_lim_strict=False,
            frame_rot_ra=0, frame_rot_dec=0, verbose=0):
        """Create FieldGridGrtCirc instance.

        Parameters
        ----------
        fov : float
            Field of view side length in radians.
        overlap_ns : float, optional
            Overlap between neighboring fields in North-South direction in
            radian. The default is 0.
        overlap_ew : float, optional
            Overlap between neighboring fields in East-West direction in
            radian. The default is 0.
        tilt : float, optional
            Tilt of the field of view in radians. The default is 0.
        dec_lim_north : float, optional
            Northern declination limit in radians. Fields North of this limit
            are excluded. The default is np.pi/2.
        dec_lim_south : float, optional
            Southern declination limit in radians. Fields South of this limit
            are excluded. The default is -np.pi/2.
        gal_lat_lim : float, optional
            Galactic latitude limit in radians. If the limit is X, fields with
            Galactic latitude in [-X, X] are excluded. The default is 0.
        gal_lat_lim_strict : bool, optional
            If False, fields will be excluded when their field center is
            within the Galactic latitude limits. If True, field will be
            excluded only if all of the field corners are within the Galactic
            latitude limits. The default is False.
        frame_rot_ra = float, optional
            Angle in radians, by which the frame grid is rotated in right
            ascension. The right ascension rotation is executed before the
            declination rotation. The default is 0.
        frame_rot_dec = float, optional
            Angle in radians, by which the frame grid is rotated in
            declination. The right ascension rotation is executed before the
            declination rotation. The default is 0.
        verbose : int, optional
            Set level of verbosity. 0 only gives most essential information.
            1 provides more details; 2 even more. -1 turns of any messages. The
            default is 0.

        Raises
        ------
        ValueError
            Raise with the North-South or East-West overlap exceeds the field
            of view size.
            Raised when the Northern or Southern declination limits exceed
            pi/2.
            Raised when Southern declination limit is higher than Southern
            declination limit.

        Returns
        -------
        None
        """

        self.dec_lim_strict = dec_lim_strict
        self.frame_rot_ra = np.mod(frame_rot_ra, np.pi*2)
        self.frame_rot_dec = np.mod(frame_rot_dec, np.pi)

        super(FieldGridGrtCirc, self).__init__(
                fov, overlap_ns=overlap_ns, overlap_ew=overlap_ew, tilt=tilt,
                dec_lim_north=dec_lim_north, dec_lim_south=dec_lim_south,
                gal_lat_lim=gal_lat_lim, gal_lat_lim_strict=gal_lat_lim_strict,
                verbose=verbose)

    #--------------------------------------------------------------------------
    def __str__(self):
        """Return information about the FieldGrid instance.

        Returns
        -------
        info : str
            Description of main properties.
        """

        info = dedent("""\
            FieldGridGrtCirc : Tilted great-circle field grid
            Field of view:    {0:7.4f} deg
            Overlap N-S       {1:7.4f} deg
            Overlap E-W       {2:7.4f} deg
            Tilt:             {3:+7.4f} deg
            Gal. lat. lim:    {4:s}
            Dec. lim. N:      {5:s}
            Dec. lim. S:      {6:s}
            Grid rot. RA:     {7:s}
            Grid rot. dec:    {8:s}
            Number of fields: {9:d}""".format(
                np.degrees(self.fov), np.degrees(self.overlap_ns),
                np.degrees(self.overlap_ew), np.degrees(self.tilt),
                f'{np.degrees(self.gal_lat_lim):7.4f} deg' \
                    if self.gal_lat_lim else 'None',
                f'{np.degrees(self.dec_lim_north):7.4f} deg' \
                    if self.dec_lim_north else 'None',
                    f'{np.degrees(self.dec_lim_south):7.4f} deg' \
                        if self.dec_lim_south else 'None',
                f'{np.degrees(self.frame_rot_ra):7.4f} deg' \
                    if self.frame_rot_ra else 'None',
                f'{np.degrees(self.frame_rot_dec):7.4f} deg' \
                    if self.frame_rot_dec else 'None',
                self.center_ras.shape[0]))

        return info

    #--------------------------------------------------------------------------
    def _split_declination(self):
        """Split full sky declination range into equdistant steps.

        Returns
        -------
        decs : np.ndarray
            Declinations.

        Notes
        -----
        Verbosity: This method prints out information if the verbosity set
        with the class initialization is > 1. This is the most detailed level
        of verbosity.
        """

        field_range = self.fov - self.overlap_ns
        n = int(np.ceil((np.pi - self.overlap_ns) / field_range))
        offset = (n * field_range + self.overlap_ns - np.pi) / 2.
        dec0 = -np.pi / 2. + self.fov / 2. - offset
        dec1 = +np.pi / 2. - self.fov / 2. + offset
        decs = np.linspace(dec0, dec1, n)

        if self.verbose > 1:
            print(f'    Number of declinations: {n}')

        return decs

    #--------------------------------------------------------------------------
    def _split_ra(self):
        """Split full sky right ascension range into equdistant steps.

        Returns
        -------
        ras : np.ndarray
            Right ascensions.

        Notes
        -----
        Verbosity: This method prints out information if the verbosity set
        with the class initialization is > 1. This is the most detailed level
        of verbosity.
        """

        n = int(np.ceil(2 * np.pi / (self.fov - self.overlap_ew)))
        ras = np.linspace(0, 2.*np.pi, n+1)[:-1]

        if self.verbose > 1:
            print(f'    Number of RA half circles: {n}')

        return ras

    #--------------------------------------------------------------------------
    def _rotate_grid(self, ras, decs):
        """Rotate the field grid in right ascension and declination.

        Parameters
        ----------
        ras : np.ndarray
            Field center right ascensions in radians.
        decs : np.ndarray
            Field center declinations in radians.

        Returns
        -------
        ras_rot : np.ndarray
            Field center right ascensions in radians after the rotation.
        decs_rot : np.ndarray
            Field center declinations in radians after the rotation.

        Notes
        -----
        The rotation angles are given at the class intitialization through
        arguments `frame_rot_ra` and `frame_rot_dec`.
        The right ascension rotation is executed before the
        declination rotation.
        Verbosity: This method prints out information if the verbosity set
        with the class initialization is > 1. This is the most detailed level
        of verbosity.
        """

        if not self.frame_rot_ra and not self.frame_rot_dec:
            return ras, decs

        x, y, z = sphere_to_cart(ras, decs)

        if self.frame_rot_dec:
            if self.verbose > 1:
                print('    Rotate frame by {0} deg in declination'.format(
                        np.degrees(self.frame_rot_dec)))

            x, y, z = rot_dec(x, y, z, self.frame_rot_dec)


        if self.frame_rot_ra:
            if self.verbose > 1:
                print('    Rotate frame by {0} deg in RA'.format(
                        np.degrees(self.frame_rot_ra)))

                x, y, z = rot_ra(x, y, z, self.frame_rot_ra)

        ras_rot, decs_rot = cart_to_sphere(x, y, z)

        return ras_rot, decs_rot

    #--------------------------------------------------------------------------
    def _declination_limits(self):
        """Apply declination limits.

        Returns
        -------
        None

        Notes
        -----
        The declination limits are given at the class intitialization through
        arguments `dec_lim_north` and `dec_lim_south`.
        Declination limits are applied after rotating the grid.
        Verbosity: This method prints out information if the verbosity set
        with the class initialization is > 0.  Details are only given at
        verbosity > 1.
        """

        apply_north = ~np.isclose(self.dec_lim_north, np.pi/2.)
        apply_south = ~np.isclose(self.dec_lim_south, -np.pi/2.)

        if self.verbose > 0:
            print('  Apply declination limits..')

        if self.verbose > 1:
            if apply_north:
                print('    Dec. lim. North: {}'.format(
                        np.degrees(self.dec_lim_north)))
            else:
                print('    Dec. lim. North: none')

            if apply_south:
                print('    Dec. lim. South: {}'.format(
                        np.degrees(self.dec_lim_south)))
            else:
                print('    Dec. lim. South: none')

        sel = np.ones(self.center_ras.shape[0], dtype=bool)

        # strict Norther limit:
        if apply_north  and self.dec_lim_strict:
            sel = np.logical_and(
                sel, np.any(self.corner_decs <= self.dec_lim_north, axis=1))

        # loose Northern limit:
        elif apply_north:
            sel = np.logical_and(sel, self.center_decs <= self.dec_lim_north)

        # strict Souther limit:
        if apply_south and self.dec_lim_strict:
            sel = np.logical_and(
                sel, np.any(self.corner_decs >= self.dec_lim_south, axis=1))

        # loose Southern limit:
        elif apply_south:
            sel = np.logical_and(sel, self.center_decs >= self.dec_lim_south)

        self.center_ras = self.center_ras[sel]
        self.center_decs = self.center_decs[sel]
        self.corner_ras = self.corner_ras[sel]
        self.corner_decs = self.corner_decs[sel]

        if self.verbose > 1:
            print(f'    Fields removed:   {np.sum(~sel):6d}')
            print(f'    Fields remaining: {np.sum(sel):6d}')

    #--------------------------------------------------------------------------
    def _calc_field_centers(self):
        """Calculate field centers.

        Returns
        -------
        None

        Notes
        -----
        This overwrites the generic method from the parent class and implements
        the creation of this specific field grid.
        Verbosity: This method prints out information if the verbosity set
        with the class initialization is > 0. This is the more detailed level
        of verbosity.
        """

        if self.verbose > 0:
            print('  Calculate field centers..')

        ras = self._split_ra()
        decs = self._split_declination()
        ras, decs = np.meshgrid(ras, decs)
        ras = ras.flatten()
        decs = decs.flatten()
        ras, decs = self._rotate_grid(ras, decs)
        self.center_ras = ras
        self.center_decs = decs

    #--------------------------------------------------------------------------
    def _create_fields(self):
        """Create fields.

        Returns
        -------
        None

        Notes
        -----
        This method overwrites the generic method from the parent class, as
        the order of steps differs.
        Verbosity: This method prints out information if the verbosity set
        with the class initialization is >= 0. This is the most basic level of
        verbosity.
        """

        if self.verbose > -1:
            print('Create fields..')

        self._calc_field_centers()
        self._calc_field_corners()
        self._avoid_galactic_plane()
        self._declination_limits()

        if self.verbose > -1:
            print(f'Final number of fields: {self.center_ras.size}')

#==============================================================================

class FieldGridTester:
    """A class to test field grid setups.
    """

    #--------------------------------------------------------------------------
    def __init__(self, grid, sampler='spherical'):
        """Crate FieldGridTester instance.

        Parameters
        ----------
        grid : FieldGrid-type
            A field grid that needs to be tested.
        sampler : str, optional
            Select a sampler. 'sherical' samples point uniformly over the sky.
            'radec', samples points uniformly in the RA-dec plane, i.e. more
            points at higher declinations. The default is 'spherical'.

        Raises
        ------
        ValueError
            Raised if `grid` is not of FieldGrid-type.
            Raised if `sample` is not 'spherical' or 'radec'.

        Returns
        -------
        None

        Notes
        -----
        The choice of the sampler affects the type of test. The sperical
        sampler is used to estimate the fraction of sky coverage with no fields
        (gaps), 1 field, 2 fields, etc over the sky. The radec-sampler is used
        to identify gaps, which are mote likely to occure near the poles.
        """

        # check input:
        if not isinstance(grid, FieldGrid):
            raise ValueError("'grid' must be a FieldGrid instance.")
        if sampler.lower() in ['spherical', 'radec']:
            self.sampler = sampler.lower()
        else:
            raise ValueError(
                    "'sampler' must be 'spherical' or 'radec'.")

        self.grid = grid

        self.test_points = {
                'ra': [], 'dec': [], 'n_fields': [], 'field_ids': []}

    #--------------------------------------------------------------------------
    def __repr__(self):
        """Return information about the FieldGridTester instance.

        Returns
        -------
        info : str
            Description of tester properties.
        """

        info = dedent(
                """\
                FieldGridTester
                Grid type: {0:s}
                Fields: {1:d}
                Test points: {2:d}""".format(
                    self.grid.grid_type, self.grid.center_ras.shape[0],
                    len(self)))

        return info

    #--------------------------------------------------------------------------
    def __len__(self):
        """Returns length of field grid tester. I.e. number of test points."""

        return len(self.test_points['ra'])

    #--------------------------------------------------------------------------
    def _sample_spherical(
            self, n_points, dec_lim_north=np.pi/2, dec_lim_south=-np.pi/2,
            gal_lat_lim=0):
        """Draw random samples uniformly over the sky.

        Parameters
        ----------
        n_points : int
            Number of random samples.
        dec_lim_north : float, optional
            Northern declination limit in radian. The default is np.pi/2.
        dec_lim_south : float, optional
            Southern declination limit in radian. The default is -np.pi/2.
        gal_lat_lim : float, optional
            Galactic latitude limit in radians. If the limit is X, points with
            Galactic latitude in [-X, X] are excluded. The default is 0.

        Returns
        -------
        ra : np.ndarray
            Right ascensions of the randomly samples points in radians.
        dec : np.ndarray
            Declinations of the randomly samples points in radians.
        """

        ra = []
        dec = []
        n_needed = n_points

        while True:
            vec = np.random.randn(3, n_points)
            vec /= np.linalg.norm(vec, axis=0)
            more_ras, more_decs = cart_to_sphere(
                    vec[0], vec[1], vec[2])
            sel = np.logical_and(
                    more_decs >= dec_lim_south, more_decs <= dec_lim_north)
            more_ras = more_ras[sel]
            more_decs = more_decs[sel]
            sel = self.grid.in_galactic_plane(
                    gal_lat_lim, center_ras=more_ras, center_decs=more_decs)
            more_ras = more_ras[~sel][:n_needed]
            more_decs = more_decs[~sel][:n_needed]
            ra.append(more_ras)
            dec.append(more_decs)
            n_needed -= more_ras.shape[0]

            if n_needed < 1:
                break

        ra = np.concatenate(ra)
        dec = np.concatenate(dec)

        return ra, dec

    #--------------------------------------------------------------------------
    def _sample_radec(
                self, n_points, dec_lim_north=np.pi/2, dec_lim_south=-np.pi/2,
                gal_lat_lim=0):
        """Draw random samples uniformly in the RA-Dec plane

        Parameters
        ----------
        n_points : int
            Number of random samples.
        dec_lim_north : float, optional
            Northern declination limit in radian. The default is np.pi/2.
        dec_lim_south : float, optional
            Southern declination limit in radian. The default is -np.pi/2.
        gal_lat_lim : float, optional
            Galactic latitude limit in radians. If the limit is X, points with
            Galactic latitude in [-X, X] are excluded. The default is 0.

        Returns
        -------
        ra : np.ndarray
            Right ascensions of the randomly samples points in radians.
        dec : np.ndarray
            Declinations of the randomly samples points in radians.
        """

        ra = []
        dec = []
        n_needed = n_points

        while True:
            more_ras = np.random.uniform(0, 2.*np.pi, n_points)
            more_decs = np.random.uniform(
                    dec_lim_south, dec_lim_north, n_points)
            sel = self.grid.in_galactic_plane(
                    gal_lat_lim, center_ras=more_ras, center_decs=more_decs)
            more_ras = more_ras[~sel][:n_needed]
            more_decs = more_decs[~sel][:n_needed]
            ra.append(more_ras)
            dec.append(more_decs)
            n_needed -= more_ras.shape[0]

            if n_needed < 1:
                break

        ra = np.concatenate(ra)
        dec = np.concatenate(dec)

        return ra, dec

    #--------------------------------------------------------------------------
    def _sample_test_points(self, n_points):
        """Draw random test points.

        Parameters
        ----------
        n_points : int
            Number of random test points.

        Returns
        -------
        n_needed : int
            Number of new test points.
        ra : np.ndarray
            Right ascensions of the randomly samples points in radians.
        dec : np.ndarray
            Declinations of the randomly samples points in radians.

        Notes
        -----
        This method checks how many test points are already stored in this
        instance. `n_points` refers to the total number of test points
        requested. Only if it exceeds the number of test points already stored,
        new test points are created.
        """

        n_done = len(self.test_points['ra'])
        n_needed = n_points - n_done
        print(f'Test points requested: {n_points:6d}')
        print(f'Test points stored:    {n_done:6d}')
        print(f'Test points needed:    {n_needed:6d}')

        # stop if enough test points exist already:
        if n_needed <= 0:
            print('Done.')
            return 0, [], []

        # create new test points:
        print('Sample test points..')

        if self.sampler == 'spherical':
            points_ra, points_dec = self._sample_spherical(
                    n_needed, dec_lim_north=self.grid.dec_lim_north,
                    dec_lim_south=self.grid.dec_lim_south,
                    gal_lat_lim=self.grid.gal_lat_lim)

        elif self.sampler == 'radec':
            points_ra, points_dec = self._sample_radec(
                    n_needed, dec_lim_north=self.grid.dec_lim_north,
                    dec_lim_south=self.grid.dec_lim_south,
                    gal_lat_lim=self.grid.gal_lat_lim)

        else:
            raise NotImplementedError(
                    f"Sampler '{self.sampler}' not implemented.")

        return n_needed, points_ra, points_dec

    #--------------------------------------------------------------------------
    def _orientation(self, p, q0, q1):
        """Calculate the orientiation of a point relative to a line.

        Parameters
        ----------
        p : np.ndarray
            2D-coordinates of the point.
        q0 : TYPE
            2D-coordinates of the first point describing the line.
        q1 : TYPE
            2D-coordinates of the second point describing the line.

        Returns
        -------
        sign : int
            +1 or -1 depending on the orientation.
        """

        sign = np.sign(
                (q1[0] - q0[0]) * (p[1] - q0[1]) - (p[0] - q0[0]) \
                * (q1[1] - q0[1]))

        return sign

    #--------------------------------------------------------------------------
    def _crossing(self, p, q0, q1):
        """Check if  a horizontal line originating from point p towards the
        right would cross the line spanned by points q0 and q1.

        Parameters
        ----------
        p : np.ndarray
            2D-coordinates of the point.
        q0 : TYPE
            2D-coordinates of the first point describing the line.
        q1 : TYPE
            2D-coordinates of the second point describing the line.

        Returns
        -------
        cross : int
            0 if it does cross. +1 or -1 if if crosses, depending on the
            orientation.
        """

        p_heq_q0 = q0[1] <= p[1]
        p_heq_q1 = q1[1] <= p[1]
        p_left = self._orientation(p, q0, q1)

        if p_heq_q0 and ~p_heq_q1 and p_left > 0:
            cross = +1
        elif ~p_heq_q0 and p_heq_q1 and p_left < 0:
            cross = -1
        else:
            cross = 0

        return cross

    #--------------------------------------------------------------------------
    def _inside_polygon(self, point, polygon):
        """Test if a point is located within a polygon.

        Parameters
        ----------
        point : np.ndarray
            2D-coordinates of the point.
        polygon : list of np.ndarray
            List of the 2D-coordinates of the points that span the polygon.

        Returns
        -------
        is_inside : bool
            True, if the point is located within the polygon. False, otherwise.
        """

        polygon = np.array(polygon + [polygon[0]])

        winding_number = 0

        for q0, q1 in zip(polygon[0:-1], polygon[1:]):
            winding_number += self._crossing(point, q0, q1)

        is_inside = winding_number > 0

        return is_inside

    #--------------------------------------------------------------------------
    def _summary_gaps(self, get=False):
        """Get results from the test for gaps, using the radec sampler.

        Parameters
        ----------
        get : bool, optional
            If True, return coordinates of the found gaps. Otherwise, return
            None. In either case the method prints out the most basic
            information. The default is False.

        Returns
        -------
        gaps : dict or None
            The dict contains the coordinates of the identified gaps, if
            `get=True`. Otherwise, None is returned.
        """

        i_gaps = np.nonzero(np.array(self.test_points['n_fields']) == 0)[0]
        n_gaps = i_gaps.shape[0]

        print(f'Gaps found: {n_gaps}')

        if get:
            gaps = {'ra': [self.test_points['ra'][i] for i in i_gaps],
                    'dec': [self.test_points['dec'][i] for i in i_gaps]}
        else:
            gaps = None

        return gaps

    #--------------------------------------------------------------------------
    def _summary_fractions(self, get=False):
        """Return results from the test with the spherical sampler.

        Parameters
        ----------
        get : bool, optional
            If True, returns the number counts of test points within gaps, a
            single field, two fields, etc. Otherwise, returns None. In either
            case the method prints out the most basic information. The default
            is False.

        Returns
        -------
        fractions : dict
            The number counts of test points within gaps, a single field, two
            fields, etc., if  `get=True`. Otherwise, None is returned.
        """

        n_fields = np.array(self.test_points['n_fields'])

        fraction = np.sum(n_fields == 0) / n_fields.shape[0]
        print(f'Sky fraction with gaps:                 {fraction:.1e}')
        fraction = np.sum(n_fields == 1) / n_fields.shape[0]
        print(f'Sky fraction with single field:       {fraction*100:5.1f} %')
        fraction = np.sum(n_fields > 1) / n_fields.shape[0]
        print(f'Sky fraction with overlapping fields: {fraction*100:5.1f} %')


        if get:
            fractions = {'region': [], 'n_points': [], 'fraction': []}

            count = np.sum(n_fields == 0)
            fractions['region'].append('gaps')
            fractions['n_points'].append(count)
            fractions['fraction'].append(count/n_fields.shape[0])

            count = np.sum(n_fields == 1)
            fractions['region'].append('single field')
            fractions['n_points'].append(count)
            fractions['fraction'].append(count/n_fields.shape[0])

            for i in range(2, n_fields.max()):
                count = np.sum(n_fields == i)
                fractions['region'].append(f'{i} fields overlapping')
                fractions['n_points'].append(count)
                fractions['fraction'].append(count/n_fields.shape[0])

        else:
            fractions = None

        return fractions

    #--------------------------------------------------------------------------
    def test(self, n_points=0, points_ra=None, points_dec=None):
        """Run the grid test.

        Parameters
        ----------
        n_points : int
            Number of random samples. The default is 0.
        points_ra : array-type, optional
            Test point right ascensions. The default is None.
        points_dec : array-type, optional
            Test point declinations. The default is None.

        Raises
        ------
        ValueError
            If `points_ra` and `points_dec` are not of same length.
            Raised if neither test points are provided through `points_ra` and
            `points_dec` nor a number of test points is set through `n_points`.

        Returns
        -------
        None
        """

        # check input:
        if points_ra is not None and points_dec is not None:
            if len(points_ra) != len(points_dec):
                raise ValueError(
                        "`points_ra` and `points_dec` need to be of same "
                        "length.")
            self.test_points = {
                    'ra': [], 'dec': [], 'n_fields': [], 'field_ids': []}
            n_needed = len(points_ra)
        elif n_points > 0:
            n_needed, points_ra, points_dec = self._sample_test_points(
                    n_points)
            if not n_needed:
                return None
        else:
            raise ValueError(
                    "Either set number of points to sample in 'n_points' or "
                    "provide test points RA and declination to 'points_ra' "
                    "and 'points_dec'.")


        # storage for results:
        n_assoc_fields = np.zeros(n_needed, dtype=int)
        assoc_field_ids = [[] for i in range(n_needed)]

        # prepare field association identification:
        print('Identify test point field associations..')
        n_fields = len(self.grid)
        coord_points = SkyCoord(points_ra, points_dec, unit='rad')
        coord_field_centers = SkyCoord(
                self.grid.center_ras, self.grid.center_decs, unit='rad')

        # create initial field corners:
        corner_ras, corner_decs = self.grid._field_corners_rot(
                self.grid.fov, tilt=self.grid.tilt, center_ra=0,
                center_dec=0)
        corner_ras[0] -= 2. * np.pi
        corner_ras[3] -= 2. * np.pi
        polygon = [[ra, dec] for ra, dec in zip(corner_ras, corner_decs)]
        del corner_ras, corner_decs

        # iterate through fields:
        for i, coord in enumerate(coord_field_centers):
            print(f'\r  Field {i+1} of {n_fields} '
                  f'({i/n_fields*100:.1f}%)..', end='')

            # identify close points:
            separation = coord.separation(coord_points).rad
            close = separation < np.sqrt(2) * self.grid.fov
            i_close = np.nonzero(close)[0]

            if not np.any(close):
                continue

            # rotate frame for close points:
            x, y, z = sphere_to_cart(
                    points_ra[close], points_dec[close])
            x, y, z = rot_ra(x, y, z, -coord.ra.rad)
            x, y, z = rot_dec(x, y, z, -coord.dec.rad)
            points_ra_rot, points_dec_rot = cart_to_sphere(x, y, z)
            points_ra_rot = np.where(
                    points_ra_rot>np.pi, points_ra_rot-2.*np.pi, points_ra_rot)

            # iterate through close, rotated points:
            for ra, dec, j in zip(points_ra_rot, points_dec_rot, i_close):
                # check if point is in field:
                inside = self._inside_polygon((ra, dec), polygon)

                # store results:
                if inside:
                    n_assoc_fields[j] += 1
                    assoc_field_ids[j].append(i)

        print('\r  Done.                                          ')

        # store results:
        self.test_points['ra'] += list(points_ra)
        self.test_points['dec'] += list(points_dec)
        self.test_points['n_fields'] += list(n_assoc_fields)
        self.test_points['field_ids'] += assoc_field_ids

    #--------------------------------------------------------------------------
    def get_results(self):
        """Get test points and their field associations.

        Returns
        -------
        self.test_points : dict
            Test point coordinates, number of field associations and IDs of
            associated fields.
        """

        return self.test_points

    #--------------------------------------------------------------------------
    def summary(self, get=False):
        """Get summary of the test.

        Parameters
        ----------
        get : bool, optional
            If True, return the full test results. The dictionary structure
            depends on the test, i.e. on the sampler selected at the class
            initialization. If False, None is returned. In either case the
            main results are printed out. The default is False.

        Returns
        -------
        dict or None
            Test results as dictionary, if `get=True`.
            If the sampler was 'sperical', the dict contains the number counts
            of test points within gaps, a single field, two fields, etc.
            If the sampler was 'radec', ` the dict contains the coordinates of
            the identified gaps.
        """

        if self.sampler == 'spherical':
            return self._summary_fractions(get=get)
        elif self.sampler == 'radec':
            return self._summary_gaps(get=get)

#==============================================================================
