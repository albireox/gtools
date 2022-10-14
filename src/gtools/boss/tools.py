#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-06-27
# @Filename: tools.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import numpy
from astropy.io import fits
from astropy.table import Table


__all__ = ["nslice", "list_exposures"]


def nslice(
    i0: int | numpy.ndarray | list | tuple,
    i1: int | None = None,
    j0: int | None = None,
    j1: int | None = None,
):
    """Returns a Numpy slice."""

    try:
        if not isinstance(i0, int) and iter(i0):
            i0, i1, j0, j1 = i0
    except TypeError:
        pass

    if i0 is None or i1 is None or j0 is None or j1 is None:
        raise ValueError("Invalid inputs in nslice.")

    return numpy.s_[i0:i1, j0:j1]


def list_exposures(directory: str | pathlib.Path):
    """Returns a table with all the exposures in a directory."""

    directory = pathlib.Path(directory)

    files = sorted(directory.glob("*.fit*"))

    data = []
    for file in list(files):
        header = fits.getheader(str(file))
        data.append(
            (
                str(file.name),
                header.get("FLAVOR", "?"),
                header.get("EXPTIME", "?"),
                header.get("HARTMANN", "?"),
                header.get("COLLA", "?"),
            )
        )

    return Table(
        rows=data,
        names=["File", "Flavour", "ExpTime", "Hartmann", "Collimator"],
    )
