#!/usr/bin/env python3
"""Common functions used by different classes.
"""

import numpy as np

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

def _orientation(p, q0, q1):
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

    Notes
    -----
    Used by `_crossing()` function.
    """

    sign = np.sign(
            (q1[0] - q0[0]) * (p[1] - q0[1]) - (p[0] - q0[0]) \
            * (q1[1] - q0[1]))

    return sign

#------------------------------------------------------------------------------
def _crossing(p, q0, q1):
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

    Notes
    -----
    Used by `inside_polygon()` function.
    """

    p_heq_q0 = q0[1] <= p[1]
    p_heq_q1 = q1[1] <= p[1]
    p_left = _orientation(p, q0, q1)

    if p_heq_q0 and ~p_heq_q1 and p_left > 0:
        cross = +1
    elif ~p_heq_q0 and p_heq_q1 and p_left < 0:
        cross = -1
    else:
        cross = 0

    return cross

#------------------------------------------------------------------------------
def inside_polygon(point, polygon):
    """Test if a point is located within a polygon.

    Parameters
    ----------
    point : np.ndarray
        2D-coordinates of the point.
    polygon : array-like
        List of the 2D-coordinates of the points that span the polygon.

    Returns
    -------
    is_inside : bool
        True, if the point is located within the polygon. False, otherwise.
    """

    polygon = np.asarray(polygon)
    polygon = np.r_[polygon, np.expand_dims(polygon[0], 0)]

    winding_number = 0

    for q0, q1 in zip(polygon[0:-1], polygon[1:]):
        winding_number += _crossing(point, q0, q1)

    is_inside = winding_number > 0

    return is_inside

#------------------------------------------------------------------------------
def _xdist(point, lp0, lp1):
    """Calculate distance between a point and a line along x-axis.

    Parameters
    ----------
    point : array-like
        Point's x- and y-coordinates.
    lp0 : array-like
        x- and y-coordinates of the line segments first point.
    lp1 : TYPE
        x- and y-coordinates of the line segments second point.

    Returns
    -------
    dist : float
        Distance between point and line segment along x-axis.

    Notes
    -----
    Used by `close_to_edge()` function.
    """

    lx = (point[1] - lp0[1]) * (lp1[0] - lp0[0]) / ( lp1[1] - lp0[1]) + lp0[0]
    dist = abs(point[0] - lx)

    return dist

#------------------------------------------------------------------------------
def _ydist(point, lp0, lp1):
    """Calculate distance between a point and a line along y-axis.

    Parameters
    ----------
    point : array-like
        Point's x- and y-coordinates.
    lp0 : array-like
        x- and y-coordinates of the line segments first point.
    lp1 : TYPE
        x- and y-coordinates of the line segments second point.

    Returns
    -------
    dist : float
        Distance between point and line segment along y-axis.

    Notes
    -----
    Used by `close_to_edge()` function.
    """

    ly = (point[0] - lp0[0]) * (lp1[1] - lp0[1]) / ( lp1[0] - lp0[0]) + lp0[1]
    dist = abs(point[1] - ly)

    return dist

#------------------------------------------------------------------------------
def close_to_edge(point, polygon, limit):
    """Determine if a point inside a polygon is close to the edge within a
    given limit.

    Parameters
    ----------
    point : array-like
        Point's x- and y-coordinates.
    polygon : array-like
        List of the 2D-coordinates of the points that span the polygon.
    limit : float
        Distance limit. If the distance along the x- or y-axis is smaller than
        this value, the function returns True. Otherwise, False.

    Returns
    -------
    bool
        True, if distance along x- or y-axis is smaller than given limit.
        False, otherwise.
    """

    if not limit:
        return False

    polygon = np.asarray(polygon)
    polygon = np.r_[polygon, np.expand_dims(polygon[0], 0)]

    # iterate through polygon sides:
    for lp0, lp1 in zip(polygon[:-1], polygon[1:]):
        # calculate y-distance if point falls on x-range of line segment:
        if point[0] > min(lp0[0], lp1[0]) and point[0] < max(lp0[0], lp1[0]):
            dist = _ydist(point, lp0, lp1)

            if dist <= limit:
                return True

        # calculate x-distance if point falls on y-range of line segment:
        if point[1] > min(lp0[1], lp1[1]) and point[1] < max(lp0[1], lp1[1]):
            dist = _xdist(point, lp0, lp1)

            if dist <= limit:
                return True

    return False

#------------------------------------------------------------------------------
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

def rotate_frame(ra, dec, instrument_center, tilt=0):
    """Rotate coordinate frame such that field center becomes (0, 0).

    Parameters
    ----------
    ra : np.ndarray or float
        Right ascension(s) in radians.
    dec : np.ndarray or float
        Declination(s) in radians.
    instrument_center : astropy.coordinates.SkyCoord
        Coordinates of the instrument center.
    circle_offset : astropy.coordinates.Angle
        Offset of the circle center from the science field center.
    tilt : float, optional
        Rotation angle in radians to account for instrument orientation. The
        default is 0.

    Returns
    -------
    ra_rot : numpy.array
        Right ascensions in the rotated frame.
    dec_rot : numpy.array
        Declinations in the rotated frame.
    """

    x, y, z = sphere_to_cart(ra, dec)
    x, y, z = rot_ra(x, y, z, -instrument_center.ra.rad)
    x, y, z = rot_dec(x, y, z, -instrument_center.dec.rad)

    if tilt:
        x, y, z = rot_tilt(x, y, z, tilt)

    ra_rot, dec_rot = cart_to_sphere(x, y, z)
    ra_rot = np.where(ra_rot > np.pi, ra_rot-2*np.pi, ra_rot)

    return ra_rot, dec_rot

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
