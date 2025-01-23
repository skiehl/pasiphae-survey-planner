# pasiphae-survey-planner

## About

The [Pasiphae project](http://pasiphae.science/) aims to map, with
unprecedented accuracy, the polarization of millions of stars at areas of the
sky away from the Galactic plane, in both the Northern and the Southern
hemispheres. New instruments, calibration and analysis methods, and dedicated
software are currently under development.

The pasiphae-survey-planner package is aimed at planning, optimizing, and
tracking the observations. For the survey the sky will be divided into almost
150,000 fields, which are expected to take about 5 years to finish
observing. The observations are subject to various constraints.

This package is not aimed at optimizing the observations each night. Its
purpose is the survey optimization. It consists of three main parts:

- A database that is used to keep track of all Pasiphae targets (i.e. fields)
  and pending or performed observations of those. This package provides the
  python interface to create and operate with this database.
- Determine when each target (with pending observations) is observable.
  Observations are subject to various constraints. In a first step this
  python package allows us to calculate if and during which time interval each
  target is observable for any given night. These "observing windows" can be
  calculated in advance and are stored in the database for quick access.
- For a given night, obtain a list of all targets with pending observations
  that can be obsered during this night. Assign them with priorities, depending
  on the survey strategy.

Such a list of targets with priorities and observing windows, may then be
handed to an observation planner that optimizes the nightly observations. This
is not part of this package.

## Status

This software is under development.

## Modules

A brief overview over the package's modules:

- **constraints.py**: Classes that define observational constraints that limit
  when fields are observable.
- **db.py**: Classes that provide interfaces to individual aspects of the
  underlying SQLite3 database.
- **io.py**: Classes to write a night's selected and prioritized observations
  into e.g. a JSON format.
- **fieldgrid.py**: Classes used to divide the sky into fields. This module was
  developed in
  [pasiphae-field-grid](https://github.com/skiehl/pasiphae-field-grid).
- **prioritizer.py**: Classes that implement different strategies to prioritize
  nightly targets.
- **surveyplanner.py**: The main classes that allow the user to calculate when
  fields are observable and to select and prioritize fields for nightly
  observations.
- **utilities.py**: General purpose functions.
- **visualizations.py** Classes to visualize e.g. the field grid, the field
  observability, the survey progression.

## Notebooks

- **CreateDB.ipynb**: Create a database based on the Pasiphae survey grid and a
  test database based on a coarser field grid for faster testing.
- **Doc_DBManagers.ipynb**: User guide for all classes in `db.py` that are used
  for interacting with the database.
- **Develop_\*.ipynb**: Used to develop the stategies and implementations of
  various aspects of the survey planning software. Note that these may not
  reflect the latest status of the code implementation, but rather the
  development process.
- **Test\*.ipynb**: Used to test various aspects of the survey planning
  software. Note that these test may be based on database that are not included
  oin the repository and thus may crash if the database with the required
  entries is not created first.
- **Visualize_Observability.ipynb**: Used to visualize the observability of
  the full sky over the course of a year, based on a coarse field grid. Note
  that the results are using a database that is not included in the repository.
  Therefore, this notebook cannot be executed successfully.

## Scripts

- **run_observability.py**: For an existing database determine the
  observabilities for its stored targets and store them in the database. This
  script can be used to prepare test databases or run the actual observability
  calculations for the Pasiphae survey.
- **run_movie**: Used to create movies that show different aspects of the
  Pasiphae field observabilities over the course of a year. This script
  requires a database that has the observabilities for a full year stored. This
  database is not included in the repository.
- **test_\*.py**`: Used to test various aspects of the survey planning
  software. Note that these test may be based on database that are not included
  oin the repository and thus may crash if the database with the required
  entries is not created first.

## Auxilliary files

- `info/flowcharts/`: Figures visualizing the connections between the modules
  and visualizing the database schema.
- `info/schedule_json_format/`: The prototype JSON formats for the schedules
  that need to be produced for the control software. Relevant for the
  development of the `io.py` module.
- `info/walops_constraints/`: Info about the WALOP-S motion constraints.
- `info/MethodCalls.txt`: Progression of method calls when
  `SurveyPlanner().check_observability()` from the `surveyplanner.py` module is
  called.
- `test_data/`: Different test results used in some of the notebooks.
