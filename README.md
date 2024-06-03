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

## Status

This software is currently under development.

## Modules

A brief overview over the package's modules:

- **constraints.py**: Classes that define observational constraints that limit
  when fields are observable.
- **db.py**: Classes that provide interfaces to individual aspects of the
  underlying SQLite3 database.
- **fieldgrid.py**: Classes used to divide the sky into fields.
- **prioritizer.py**: Classes that implement different strategies to prioritize
  nightly targets.
- **surveyplanner.py**: The main classes that allow the user to calculate when
  fields are observable and to select and prioritize fields for nightly
  observations.
- **utilities.py**: General purpose functions.
- **visualizations.py** Classes to visualize e.g. the field grid, the field
  observability, the survey progression.

## Notebooks

- **Doc_\*.ipynb**: Software demonstration.
- **Develop_\*.ipynb** (and other notebooks): Used for the development and
  testing of various aspects of the final package. They may not reflect the
  latest state of the software and may be incompatible with the latest state of
  the modules. The will eventually be removed from the repository.
