#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-03-11
# @Filename: defaults.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations


TRIMSEC1 = (1, 2043, 2041, 4080)  # Data section for quadrant 1
TRIMSEC2 = (2078, 4120, 2041, 4080)  # Data section for quadrant 2
TRIMSEC3 = (1, 2043, 1, 2040)  # Data section for quadrant 3
TRIMSEC4 = (2078, 4120, 1, 2040)  # Data section for quadrant 4

BIASSEC1 = (2044, 2060, 2041, 4080)  # Overscan section for quadrant 1
BIASSEC2 = (2061, 2077, 2041, 4080)  # Overscan section for quadrant 2
BIASSEC3 = (2044, 2060, 1, 2040)  # Overscan section for quadrant 3
BIASSEC4 = (2061, 2077, 1, 2040)  # Overscan section for quadrant 4
