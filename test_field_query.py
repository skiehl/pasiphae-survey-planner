#!/usr/bin/env python3
"""Run tests for the observability calculation."""

from pandas import DataFrame

from db import FieldManager

#==============================================================================
# CONFIG
#==============================================================================

db_name = 'test_planner.sqlite3'

#==============================================================================
# MAIN
#==============================================================================

if __name__ == '__main__':

    manager = FieldManager(db_name)
    fields = manager.get_fields(observable_between=(
            "2024-01-01 12:00:00", "2024-01-02 12:00:00"))
    fields = DataFrame(fields)
    print(fields.shape)

    fields = manager.get_fields(telescope='Skinakas', observable_between=(
            "2024-01-01 12:00:00", "2024-01-02 12:00:00"))
    fields = DataFrame(fields)
    print(fields.shape)

    fields = manager.get_fields(telescope='Skinakas', observable_between=(
            "2024-01-01 12:00:00", "2024-01-02 12:00:00"), pending=True)
    fields = DataFrame(fields)
    print(fields.shape)
