#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-11-26
# @Filename: types.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import Any, Literal

import astropy.visualization
import nptyping as npt


ARRAY_2D = npt.NDArray[npt.Shape["*, *"], Any]
ARRAY_2D_f32 = npt.NDArray[npt.Shape["*, *"], npt.Float32]
ARRAY_2D_u16 = npt.NDArray[npt.Shape["*, *"], npt.UInt16]

INTERVAL_T = Literal["zscale", "minmax"] | tuple[float, float]
STRETCH_T = Literal["linear", "log"] | astropy.visualization.BaseStretch
