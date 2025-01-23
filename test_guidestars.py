#!/usr/bin/env python3
"""Run tests for adding guidestars."""

from astropy.time import Time
import astropy.units as u
import os
import platform

from astropy.coordinates import Angle
import astropy.units as u

from db import DBConnectorSQLite

#==============================================================================
# CONFIG
#==============================================================================

reset_db = True
db_init = 'test_planner_init.sqlite3'
db_name = 'test_planner_temp.sqlite3'

#==============================================================================
# MAIN
#==============================================================================

if __name__ == '__main__':

    if reset_db:
        if platform.system() == 'Linux':
            os.system(f'cp {db_init} {db_name}')
        elif platform.system() == 'Windows':
            os.system(f'xcopy {db_init} {db_name} /y')
        else:
            raise ValueError('Unknown operating system.')

    # get fields:
    db = DBConnectorSQLite(db_name)
    field_ids = []
    field_center_ras = []
    field_center_decs = []

    for field in db.get_fields():
        field_ids.append(field[0])
        field_center_ras.append(field[2])
        field_center_decs.append(field[3])

    # add pseudo-guidestars for each field but the last, simply using the
    # field center coordinate:
    print('\nTEST: Add guidestars..')
    db.add_guidestars(
            field_ids[:-2], field_center_ras[:-2], field_center_decs[:-2],
            warn_missing=True, warn_rep=Angle(2*u.arcmin),
            warn_sep=Angle(30*u.arcmin))


    # add duplicate:
    print('\nTEST: Add a duplicate guidestar..')
    db.add_guidestars(
            field_ids[0], field_center_ras[0]+Angle(10*u.arcsec).rad,
            field_center_decs[0], warn_missing=False,
            warn_rep=Angle(2*u.arcmin), warn_sep=Angle(30*u.arcmin))

    # add guidestar far off:
    print('\nTEST: Add too far guidestar..')
    db.add_guidestars(
            field_ids[0], field_center_ras[0]+Angle(40*u.arcmin).rad,
            field_center_decs[0], warn_missing=False,
            warn_rep=Angle(2*u.arcmin), warn_sep=Angle(30*u.arcmin))
