#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-03-11
# @Filename: detrend.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import numpy
import numpy.typing as npt
from astropy.io import fits

from . import defaults, tools


__all__ = ["apply_overscan"]


def apply_overscan(
    data: str | pathlib.Path | npt.NDArray[numpy.uint16],
    header: fits.Header | dict | None = None,
    extension: str | int = 0,
) -> npt.NDArray[numpy.float32]:
    """Applies overscan correction to a data array.

    Parameters
    ----------
    data
        The data array or the path to a FITS file.
    header
        The header of the FITS file. This can be ignored if ``data`` is a FITS
        file. If not provided and ``data`` is an array, assumes that the data and
        trim sections are the usual ones.
    extension
        The extension to read if ``data`` is a FITS file.

    Returns
    -------
    overscan_corrected
        The data array with the overscan corrected.

    """

    if isinstance(data, (str, pathlib.Path)):
        data, header = fits.getdata(data, ext=extension, header=True)

    assert isinstance(data, numpy.ndarray)

    quads: list[npt.NDArray[numpy.float32]] = []

    for i in range(1, 5):
        if header is not None:
            trimsec = header.get(f"TRIMSEC{i}", None)
            biassec = header.get(f"BIASSEC{i}", None)

        if header is None or trimsec is None or biassec is None:
            trimsec = getattr(defaults, f"TRIMSEC{i}")
            biassec = getattr(defaults, f"BIASSEC{i}")

        trim_slice = tools.slice_to_2d_slice(trimsec, order="C", index=1)
        bias_slice = tools.slice_to_2d_slice(biassec, order="C", index=1)

        bias = data[bias_slice]
        data_q = data[trim_slice].astype(numpy.float32)

        overscan_q = numpy.median(bias.astype(numpy.float32), axis=1)
        overscan_q = overscan_q[:, numpy.newaxis]

        data_q -= overscan_q
        quads.append(data_q)

    n_cols = quads[0].shape[1] + quads[1].shape[1]
    n_rows = quads[0].shape[0] + quads[2].shape[0]

    stitched = numpy.empty((n_rows, n_cols), dtype=numpy.float32)
    stitched[quads[0].shape[0] :, : quads[0].shape[1]] = quads[0]
    stitched[quads[0].shape[0] :, quads[0].shape[1] :] = quads[1]
    stitched[: quads[0].shape[0], : quads[0].shape[1]] = quads[2]
    stitched[: quads[0].shape[0], quads[0].shape[1] :] = quads[3]

    return stitched
