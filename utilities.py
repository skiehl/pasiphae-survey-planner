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

#------------------------------------------------------------------------------

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
