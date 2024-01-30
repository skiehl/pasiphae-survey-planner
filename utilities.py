# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Utility functions.
"""

import numpy as np
import astropy.units as u

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

#==============================================================================

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

#==============================================================================

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

#==============================================================================

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

#==============================================================================

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

def za_to_airmass(za, conversion="secz"):
    """Convert zenith angle to airmass.

    Parameters
    ----
    za : astropy.Angle
        Zenith angle value(s).
    conversion : str, default="secz"
        Sets the conversion method. Implememented are:
            * "secz": the standard conversion.
            * "Rosenberg": [1]
            * "KastenYoung": [2]
            * "Young": [3]

    Returns
    -----
    out : np.array
        Airmass corresponding to input zenith angle(s).

    References
    -----
    [1] Rozenberg, G. V. 1966. "Twilight: A Study in Atmospheric Optics".
    New York: Plenum Press, 160.
    [2] Kasten, F.; Young, A. T. 1989. "Revised optical air mass tables and
    approximation formula". Applied Optics. 28 (22): 4735–4738.
    [3] Young, A. T. 1994. "Air mass and refraction". Applied Optics.
    33:1108–1110.
    """

    if za.size > 1:
        sel = za < 90. * u.deg
        cosza = np.cos(za[sel])
        za = za[sel]
    else:
        if za > 90. *u.deg:
            return np.inf
        cosza = np.cos(za).value
        sel = None

    if conversion == "secz":
        airmass = 1. / cosza

    elif conversion == "Rosenberg":
        airmass = 1. / (cosza + 0.025 * np.exp(-11. * cosza))

    elif conversion == "KastenYoung":
        airmass = 1. / (
                cosza + 0.50572 * (96.07995 - za.value)**(-1.6364))

    elif conversion == "Young":
        term1 = 1.002432 * cosza**2
        term2 = 0.148386 * cosza
        term3 = 0.0096467
        term4 = cosza**3
        term5 = 0.149864 * cosza**2
        term6 = 0.0102963 * cosza
        term7 = 0.000303978

        airmass = (term1 + term2 +term3) / (term4 + term5 + term6 + term7)

    else:
        raise ValueError(
                "ZA to airmass conversion '{0:s}' not implemented.".format(
                        str(conversion)))

    if sel is not None:
        airmass_temp = np.ones(sel.size) * np.inf
        airmass_temp[sel] = airmass
        airmass = airmass_temp

    return airmass

#==============================================================================

def alt_to_airmass(alt, conversion="secz"):
    """Convert altitude to airmass.

    Parameters
    ----
    alt : astropy.Angle
        Altitude angle value(s).
    conversion : str, default="secz"
        Sets the conversion method. Implememented are:
            * "secz": the standard conversion.
            * "Rosenberg": [1]
            * "KastenYoung": [2]
            * "Young": [3]

    Returns
    -----
    out : np.array
        Airmass corresponding to input altitude angle(s).

    References
    -----
    [1] Rozenberg, G. V. 1966. "Twilight: A Study in Atmospheric Optics".
    New York: Plenum Press, 160.
    [2] Kasten, F.; Young, A. T. 1989. "Revised optical air mass tables and
    approximation formula". Applied Optics. 28 (22): 4735–4738.
    [3] Young, A. T. 1994. "Air mass and refraction". Applied Optics.
    33:1108–1110.
    """

    za = 90. * u.deg - alt
    airmass = za_to_airmass(za, conversion=conversion)

    return airmass

#==============================================================================

def true_blocks(values):
    """Find blocks of successive True's.

    Parameters
    ----------
    values : nump.ndarray
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
    while i < values.size-1:
        if ~np.any(values[i:]):
            break
        j = np.argmax(values[i:]) + i
        k = np.argmax(~values[j:]) + j
        if j == k and j != values.size-1:
            k = values.size
        periods.append((j,k-1))
        i = k

    return periods

#==============================================================================
