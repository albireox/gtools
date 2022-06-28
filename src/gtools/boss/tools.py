#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-06-27
# @Filename: tools.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import numpy


__all__ = ["nslice"]


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
