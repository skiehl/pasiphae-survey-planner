#!/usr/bin/env python3
"""Run tests of Telescope instance."""

from astropy.coordinates import Angle
from astropy.time import Time

from surveyplanner import Telescope

#==============================================================================
# CONFIG
#==============================================================================

date = '2024-10-01'
twilight = 'nautical'

telescopes = {
    'Skinakas': {
        'lat': Angle('35:12:43 deg'),
        'lon': Angle('24:53:57 deg'),
        'height': 1750,
        'utc_offset': 2
        },
    'SAAO': {
        'lat': Angle('-32:22:46 deg'),
        'lon': Angle('20:48:38.5 deg'),
        'height': 1798,
        'utc_offset': 2
        }
    }

#==============================================================================
# MAIN
#==============================================================================


date = Time(date)
year = date.to_datetime().year
month = date.to_datetime().month
day = date.to_datetime().day

print('Date:', date, '\n')

for name, params in telescopes.items():
    telescope = Telescope(
            params['lat'], params['lon'], params['height'],
            params['utc_offset'], name=name)
    sun_set_utc, sun_rise_utc = telescope.get_sun_set_rise(
            year, month, day, twilight)

    print('Sun set UTC: ', sun_set_utc)
    print('Sun rise UTC:', sun_rise_utc)
    print('')
