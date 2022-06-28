#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-06-27
# @Filename: constants.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations


__all__ = ["GAIN", "OVERSCAN_PIXELS", "OVERSCAN_LINES", "DATA_REGION"]

GAIN = {
    "b1": [1.048, 1.048, 1.018, 1.006],
    "b2": [1.040, 0.994, 1.002, 1.010],
    "r1": [1.966, 1.566, 1.542, 1.546],
    "r2": [1.598, 1.656, 1.582, 1.594],
}

OVERSCAN_PIXELS = {"r": 112, "b": 78}
OVERSCAN_LINES = {"r": 56, "b": 48}

DATA_REGION = {"r": [48, -48, 119, -119], "b": [56, -56, 128, -128]}
