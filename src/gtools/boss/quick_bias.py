#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-06-26
# @Filename: quick_bias.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import numpy
from astropy.io import fits
from astropy.stats import sigma_clip

from .parameters import DATA_REGION, GAIN, OVERSCAN_LINES, OVERSCAN_PIXELS
from .tools import nslice


__all__ = ["quick_bias"]


def quick_bias(filename: str | pathlib.Path, apply_gain: bool = True):
    """Performs a quick bias subtraction and applies the gain factors."""

    filename = pathlib.Path(filename)

    if not filename.exists():
        raise FileExistsError(f"File {filename!s} does not exist.")

    data, header = fits.getdata(str(filename), header=True)

    camera = header["CAMERAS"]
    red = "r" in camera

    image = data[nslice(DATA_REGION["r" if red else "b"])].astype(numpy.float32)

    isize, jsize = image.shape
    isize //= 2
    jsize //= 2

    olines = OVERSCAN_LINES["r" if red else "b"]
    opixels = OVERSCAN_PIXELS["r" if red else "b"]

    # Iterate over quadrants starting from lower-left CCW.
    for ii, jj in [[0, 0], [0, 1], [1, 1], [1, 0]]:
        qslice = nslice(ii * isize, (ii + 1) * isize, jj * jsize, (jj + 1) * jsize)
        qdata = image[qslice]

        if jj == 0:
            bias_j0 = 10
            bias_j1 = opixels - 10
        else:
            bias_j0 = data.shape[1] - opixels + 10
            bias_j1 = data.shape[1] - 10

        if ii == 0:
            bias_i0 = 10
            bias_i1 = olines - 10
        else:
            bias_i0 = data.shape[0] - olines + 10
            bias_i1 = data.shape[0] - 10

        bslice = nslice(bias_i0, bias_i1, bias_j0, bias_j1)
        bias_image = data[bslice]

        bias = numpy.nanmedian(sigma_clip(bias_image, masked=False))

        gain = GAIN[camera][ii * 2 + jj] if apply_gain else 1.0

        qdata = gain * (qdata - bias)

        image[qslice] = qdata

    return image
