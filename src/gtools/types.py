#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-11-26
# @Filename: types.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import Literal

import astropy.visualization


INTERVAL_T = Literal["zscale", "minmax"] | tuple[float, float]
STRETCH_T = Literal["linear", "log"] | astropy.visualization.BaseStretch
