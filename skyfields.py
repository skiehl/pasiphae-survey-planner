# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Sky fields for the pasiphae survey.
"""

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
# CLASSES
#==============================================================================

class Field:
    """A field in the sky."""

    #--------------------------------------------------------------------------
    def __init__(
            self, fov, ra, dec, tilt=0., field_id=None,
            latest_obs_window_jd=None, n_obs_done=0, n_obs_pending=0):
        """A field in the sky."""

        self.id = field_id
        self.fov = Angle(fov, unit='rad')
        self.center_coord = SkyCoord(ra, dec, unit='rad')
        self.center_ra = self.center_coord.ra
        self.center_dec = self.center_coord.dec
        self.tilt = Angle(tilt, unit='rad')
        self.corners_coord = self._calc_field_corners()
        self.corners_ra = self.corners_coord.ra
        self.corners_dec = self.corners_coord.dec
        self.latest_obs_window_jd = latest_obs_window_jd
        self.obs_windows = []
        self.status = -1
        self.setting_in = None
        self.n_obs_done = n_obs_done
        self.n_obs_pending = n_obs_pending

    #--------------------------------------------------------------------------
    def __str__(self):

        info = dedent("""\
            Sky field {0:s}
            Field of view: {1:8.4f} arcmin
            Center RA:     {2:8.4f} deg
            Center Dec:    {3:+8.4f} deg
            Tilt:          {4:+8.4f} deg
            Status:        {5}
            Observations:  {6} pending
                           {7} done
            """.format(
                f'{self.id}' if self.id is not None else '',
                self.fov.arcmin, self.center_ra.deg, self.center_dec.deg,
                self.tilt.deg, self._status_to_str(), self.n_obs_pending,
                self.n_obs_done))

        return info

    #--------------------------------------------------------------------------
    def _status_to_str(self):
        """TBD
        """

        if self.status == -1:
            status_str = 'unknown/undefined'
        elif self.status == 0:
            status_str = 'not observable'
        elif self.status == 1:
            status_str = 'rising'
        elif self.status == 2:
            status_str = 'plateauing'
        elif self.status == 3:
            status_str = 'setting in {0}'.format(self.setting_in)

        return status_str

    #--------------------------------------------------------------------------
    def _field_corners_init(self, fov):
        """Create field corner points in cartesian coordinates.

        TODO
        """

        diff = np.tan(fov / 2.)
        x = np.ones(4)
        y = np.array([-diff, diff, diff, -diff])
        z = np.array([-diff, -diff, diff, diff])

        return x, y, z

    #--------------------------------------------------------------------------
    def _rot_tilt(self, x, y, z, tilt):
        """Rotate around x-axis by tilt angle.

        TODO
        """

        x_rot = x
        y_rot = y * np.cos(tilt) - z * np.sin(tilt)
        z_rot = y * np.sin(tilt) + z * np.cos(tilt)

        return x_rot, y_rot, z_rot

    #--------------------------------------------------------------------------
    def _rot_dec(self, x, y, z, dec):
        """Rotate around y-axis by declination angle.

        TODO"""

        dec = -dec
        x_rot = x * np.cos(dec) + z * np.sin(dec)
        y_rot = y
        z_rot = -x * np.sin(dec) + z * np.cos(dec)

        return x_rot, y_rot, z_rot

    #--------------------------------------------------------------------------
    def _rot_ra(self, x, y, z, ra):
        """Rotate around z-axis by right ascension angle.

        TODO"""

        x_rot = x * np.cos(ra) - y * np.sin(ra)
        y_rot = x * np.sin(ra) + y * np.cos(ra)
        z_rot = z

        return x_rot, y_rot, z_rot

    #--------------------------------------------------------------------------
    def _cart_to_sphere(self, x, y, z):
        """Transform cartesian to spherical coordinates.

        TODO"""

        r = np.sqrt(x**2 + y**2 + z**2)
        za = np.arccos(z / r)
        dec = np.pi / 2. - za
        ra = np.arctan2(y, x)

        return ra, dec

    #--------------------------------------------------------------------------
    def _calc_field_corners(self):
        """Calculate field corner points at specified field center coordinates.

        TODO
        """

        x, y, z = self._field_corners_init(self.fov.rad)
        x, y, z = self._rot_tilt(x, y, z, self.tilt.rad)
        x, y, z = self._rot_dec(x, y, z, self.center_dec.rad)
        x, y, z = self._rot_ra(x, y, z, self.center_ra.rad)
        ra, dec = self._cart_to_sphere(x, y, z)
        corners_coord = SkyCoord(ra, dec, unit='rad')

        return corners_coord

    #--------------------------------------------------------------------------
    def _true_blocks(self, observable):
        """Find blocks of successive True's.

        Parameters
        ----------
        observable : nump.ndarray
            Boolean-type 1dim-array.

        Returns
        -------
        list
            Each element corresponds to one block of True's. The element is a
            list of two integers, the first marking the first index of the
            block, the second marking the last True entry of the block.
        """

        i = 0
        periods = []

        # iterate through array:
        while i < observable.size-1:
            if ~np.any(observable[i:]):
                break
            j = np.argmax(observable[i:]) + i
            k = np.argmax(~observable[j:]) + j
            if j == k and j != observable.size-1:
                k = observable.size
            periods.append((j,k-1))
            i = k

        return periods

    #--------------------------------------------------------------------------
    def get_obs_window(self, telescope, frame, refine=0*u.min):
        """Calculate time windows when the field is observable.

        Parameters
        ----------
        telescope : Telescope
            Telescope for which to calculate observability.
        frame : astropy.coordinates.AltAz
            Frame that provides the time steps at which observability is
            initially tested.
        refine : astropy.units.Quantity, optional
            Must be a time unit. If given, the precision of the observable time
            window is refined to this value. I.e. if the interval given in
            'frame' is 10 minutes and refine=1*u.min the window limits will be
            accurate to a minute. The default is 0*u.min.

        Returns
        -------
        obs_windows : list
            List of tuples. Each tuple contains two astropy.time.Time instances
            that mark the earliest time and latest time of a window during
            which the field is observable.

        Notes
        -----
        This method uses a frame as input instead of a start and stop time
        and interval, from which the frame could be created. The advantage is
        that the same initial frame can be used for all fields.
        """

        obs_windows = []
        temp_obs_windows = []
        observable = telescope.constraints.get(self.center_coord, frame)
        blocks = self._true_blocks(observable)

        for i, j in blocks:
            obs_window = (frame.obstime[i], frame.obstime[j])
            temp_obs_windows.append(obs_window)

        # increase precision for actual observing windows:
        if refine.value:
            time_interval = frame.obstime[1] - frame.obstime[0]

            # iterate through time windows:
            for t_start, t_stop in temp_obs_windows:
                # keep start time:
                if t_start == frame.obstime[0]:
                    pass
                # higher precision for start time:
                else:
                    t_start_new = t_start - time_interval
                    frame = telescope.get_frame(t_start_new, t_start, refine)
                    observable = telescope.constraints.get(
                            self.center_coord, frame)
                    k = np.argmax(observable)
                    t_start = frame.obstime[k]

                # keep stop time:
                if t_stop == frame.obstime[-1]:
                    pass
                # higher precision for stop time:
                else:
                    t_stop_new = t_stop + time_interval
                    frame = telescope.get_frame(t_stop, t_stop_new, refine)
                    observable = telescope.constraints.get(
                            self.center_coord, frame)
                    k = (frame.obstime.value.size - 1
                         - np.argmax(observable[::-1]))
                    t_stop = frame.obstime[k]

                #obs_windows.append(ObsWindow(t_start, t_stop)) # TODO
                obs_windows.append((t_start, t_stop))

        # in case of no precision refinement:
        else:
            for t_start, t_stop in temp_obs_windows:
                #obs_windows.append(ObsWindow(t_start, t_stop)) # TODO
                obs_windows.append((t_start, t_stop))

        return obs_windows

    #--------------------------------------------------------------------------
    def get_obs_duration(self):

        duration = 0 * u.day

        for obs_window in self.obs_windows:
            duration += obs_window.duration

        return duration

    #--------------------------------------------------------------------------
    def add_obs_window(self, obs_window):
        """Add observation window(s) to field.

        Parameters
        ----------
        obs_window : ObsWindow or list
            Observation window(s) that is/are added to the field. If multiple
            windows should be added provide a list of ObsWindow instances.

        Returns
        -------
        None.
        """

        if isinstance(obs_window, list):
            self.obs_windows += obs_window
        else:
            self.obs_windows.append(obs_window)

    #--------------------------------------------------------------------------
    def set_status(
            self, rising=None, plateauing=None, setting=None, setting_in=None,
            not_available=None):

        if not_available:
            self.status = 0
        elif rising:
            self.status = 1
        elif plateauing:
            self.status = 2
        elif setting:
            self.status = 3
            self.setting_in = setting_in

#==============================================================================

class SkyFields:
    """Separation of the sky into fields."""

    #--------------------------------------------------------------------------
    def __init__(
            self, fov, overlap, tilt=0., b_lim=0., dec_lim_north=None,
            dec_lim_south=None):
        """Separation of the sky into fields.

        TODO"""

        if overlap >= fov:
            raise ValueError("Overlap must be smaller than field of view.")

        self.fov = fov
        self.overlap = overlap
        self.tilt = tilt
        self.b_lim = b_lim
        self.dec_lim_north = dec_lim_north
        self.dec_lim_south = dec_lim_south
        self.fields = []
        self.dec_field_ids = {}
        self.decs = None

        self._create_fields()

    #--------------------------------------------------------------------------
    def __str__(self):

        return dedent("""\
            SkyFields
            Field of view:    {0:7.4f} deg
            Overlap           {1:7.4f} deg
            Tilt:             {2:+7.4f} deg
            Gal. lat. lim:    {3:s}
            Dec. lim. N:      {4:s}
            Dec. lim. S:      {5:s}
            Number of fields: {6:d}""".format(
                np.degrees(self.fov), np.degrees(self.overlap),
                np.degrees(self.tilt),
                f'{np.degrees(self.b_lim):7.4f} deg' if self.b_lim else 'no',
                f'{np.degrees(self.dec_lim_north):7.4f} deg' \
                    if self.dec_lim_north else 'None',
                    f'{np.degrees(self.dec_lim_south):7.4f} deg' \
                        if self.dec_lim_south else 'None',
                len(self.fields)))

    #--------------------------------------------------------------------------
    def __len__(self):

        return len(self.fields)

    #--------------------------------------------------------------------------
    def _split_lat(self, dec):
        """Split a circle at a given declination into equidistant fields.

        TODO"""

        # one field at the North/South pole:
        if np.isclose(np.absolute(dec), np.pi/2.):
            n = 1
            ras = np.zeros(1)

        # split along circle:
        else:
            n = int(np.ceil(2 * np.pi / (self.fov - self.overlap))
                    * np.cos(dec))
            ras = np.linspace(0., 2.*np.pi, n+1)[:-1]

        return ras

    #--------------------------------------------------------------------------
    def _split_lon(self):
        """Split latitude into equidistant declinations.

        TODO"""

        n = int(np.ceil(np.pi / 2. / (self.fov - self.overlap))) * 2 - 1
        decs = np.linspace(-np.pi/2., np.pi/2., n)

        # apply northern declination limit:
        if self.dec_lim_north is not None:
            decs = decs[decs <= self.dec_lim_north]

        # apply southern declination limit:
        if self.dec_lim_south is not None:
            decs = decs[decs >= self.dec_lim_south]

        return decs

    #--------------------------------------------------------------------------
    def _create_fields(self):
        """Split sky into fields.

        TODO"""

        print('Creating fields..')

        n = 0
        self.decs = self._split_lon()
        n_decs = self.decs.size

        for i, dec in enumerate(self.decs):
            print(f'\rDec {i+1} of {n_decs}. {i*100./n_decs:.1f}%..', end='')

            dec_field_ids = []

            for ra in self._split_lat(dec):

                # create field:
                field = Field(self.fov, ra, dec, tilt=self.tilt, field_id=n)

                # check if in Galactic plane:
                if self.b_lim:
                    coord = field.center_coord.transform_to('galactic')

                    # skip if all corners are within Galactic latitute limit:
                    if np.all(np.logical_and(
                            coord.b.rad < self.b_lim,
                            coord.b.rad > -self.b_lim)):
                        continue

                # store field:
                self.fields.append(field)
                dec_field_ids.append(n)
                n += 1

            # store field ids for current declination:
            self.dec_field_ids[dec] = dec_field_ids

        print('\rDone.                              ')

    #--------------------------------------------------------------------------
    def get_fields(self, dec=None):
        """Return fields.

        TODO"""

        # no declination specified:
        if dec is None:
            fields = self.fields

        # declination not available:
        elif np.sum(np.isclose(dec, self.decs)) == 0:
            raise ValueError('Declination not available.')

        else:
            fields = [self.fields[i] for i in self.dec_field_ids[dec]]

        return fields

    #--------------------------------------------------------------------------
    def get_field_centers(self, dec=None):
        """Return field center coordinates.

        TODO"""

        ras = []
        decs = []

        for field in self.get_fields(dec=dec):
            ra, dec = field.get_center_coord()
            ras.append(ra)
            decs.append(dec)

        ras = np.array(ras)
        decs = np.array(decs)

        return ras, decs

#==============================================================================
