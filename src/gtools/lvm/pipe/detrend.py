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


__all__ = ["apply_overscan", "apply_pixelmask", "apply_bias"]


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


def apply_pixelmask(
    data: str | pathlib.Path | npt.NDArray[numpy.float32],
    pixelmask: str | pathlib.Path | npt.NDArray[numpy.bool],
) -> npt.NDArray[numpy.float32]:
    """Applies a pixel mask to a data array.

    Parameters
    ----------
    data
        The data array or the path to a FITS file. In either case, the data must
        have been overscan corrected and trimmed to the usual ``4080x4086``
        dimensions.
    pixelmask
        The pixel mask to apply or path to a FITS file. In the latter case, the
        ``BADPIX`` extension is used.

    Returns
    -------
    masked_data
        The data array with the pixel mask applied.

    """

    if isinstance(data, (str, pathlib.Path)):
        data = fits.getdata(data, ext=0)

    if isinstance(pixelmask, (str, pathlib.Path)):
        pixelmask = fits.open(pixelmask)["BADPIX"].data

    assert isinstance(data, numpy.ndarray)
    assert isinstance(pixelmask, numpy.ndarray)

    pixelmask = pixelmask.astype(numpy.bool)

    assert data.shape == pixelmask.shape

    data[pixelmask] = numpy.nan

    return data


def apply_bias(
    data: str | pathlib.Path | npt.NDArray[numpy.float32],
    bias: str | pathlib.Path | npt.NDArray[numpy.float32],
) -> npt.NDArray[numpy.float32]:
    """Applies a pixel mask to a data array.

    Parameters
    ----------
    data
        The data array or the path to a FITS file. In either case, the data must
        have been overscan corrected and trimmed to the usual ``4080x4086``
        dimensions.
    bias
        The bias levels to apply or path to a FITS file. In the latter case, the
        first extension is used. The bias array must have been overscan corrected.

    Returns
    -------
    bias_data
        The data array with the bias levels have been corrected.

    """

    if isinstance(data, (str, pathlib.Path)):
        data = fits.getdata(data, ext=0)

    if isinstance(bias, (str, pathlib.Path)):
        bias = fits.open(bias)["BADPIX"].data

    assert isinstance(data, numpy.ndarray)
    assert isinstance(bias, numpy.ndarray)

    assert data.shape == bias.shape

    data -= bias

    return data
