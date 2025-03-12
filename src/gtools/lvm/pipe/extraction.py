#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-03-11
# @Filename: extraction.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import os.path
import pathlib
from functools import partial

from typing import Literal, Sequence

import numpy
import numpy.typing as npt
import polars
from astropy.io import fits

from .detrend import apply_overscan


__all__ = ["quick_extraction", "extract_and_stitch"]


DATA = pathlib.Path(__file__).parent / "data"


def quick_extraction(
    file_: pathlib.Path | str,
    detrend: bool = False,
    mode: Literal["sum", "mean", "cent"] = "sum",
):
    """Runs a quick extraction using pre-computed traces.

    Parameters
    ----------
    file_
        The path to the raw file to extract.
    detrend
        Whether to apply detrending before extraction.
    mode
        The extraction mode. Can be ``'sum'`` (sums the flux in the trace),
        ``'mean'`` (returns the mean flux in the trace), or ``'cent'`` (returns
        the flux at the centre of the trace).

    Returns
    -------
    data
        The extracted data, as a row-stacked array.

    """

    # Load data and header.
    raw: npt.NDArray[numpy.uint16] = fits.getdata(file_, ext=0)
    header = fits.getheader(file_, ext=0)

    # Apply detrending if requested.
    if detrend:
        data = apply_overscan(raw, header)
    else:
        data = raw.astype(numpy.float32)

    # Load the traces.
    cent_trace = polars.read_parquet(DATA / "cent_trace.parquet")
    width_trace = polars.read_parquet(DATA / "width_trace.parquet")

    ccd = header["CCD"]
    cent_trace = cent_trace.filter(polars.col.ccd == ccd)
    width_trace = width_trace.filter(polars.col.ccd == ccd)

    xrange = numpy.arange(data.shape[1])
    extracted: npt.NDArray[numpy.float32] = numpy.empty(
        (len(cent_trace), len(xrange)),
        dtype=numpy.float32,
    )

    for row in cent_trace.rows(named=True):
        nfibre = row["nfibre"]

        cent_ncoeffs = len([col for col in row if col.startswith("coeff")])
        cent_coeffs = [row[f"coeff{i}"] for i in range(cent_ncoeffs)]
        ycent = numpy.polyval(cent_coeffs[::-1], xrange)

        width_row = width_trace.filter(polars.col.nfibre == nfibre).to_dicts()[0]
        width_ncoeffs = len([col for col in width_row if col.startswith("coeff")])
        width_coeffs = [width_row[f"coeff{i}"] for i in range(width_ncoeffs)]
        ywidth = numpy.polyval(width_coeffs[::-1], xrange)

        y0 = numpy.floor(ycent - ywidth / 2).astype(int)
        y1 = numpy.ceil(ycent + ywidth / 2).astype(int)

        if mode == "cent":
            fibre_data = [data[int(ycent[xx]), xx] for xx in xrange]
        elif mode == "sum":
            fibre_data = [data[y0[xx] : y1[xx] + 1, xx].sum() for xx in xrange]
        elif mode == "mean":
            fibre_data = [data[y0[xx] : y1[xx] + 1, xx].mean() for xx in xrange]

        extracted[nfibre - 1, :] = numpy.array(fibre_data)

    return extracted


def _extract_one(
    file_: pathlib.Path,
    detrend: bool = True,
    mode: Literal["sum", "mean", "cent"] = "sum",
):
    """Extracts the data from a single file."""

    ccd = fits.getheader(file_, ext=0)["CCD"]

    return {ccd: quick_extraction(file_, detrend=detrend, mode=mode)}


def extract_and_stitch(
    files_: Sequence[str | pathlib.Path] | str,
    detrend: bool = True,
    mode: Literal["sum", "mean", "cent"] = "sum",
):
    """Extracts fibre data and returns a single array.

    Parameters
    ----------
    files_
        The list of files to extract, which must belong to the same exposure for
        all the different cameras. If a string is passed, it will be used as a
        glob pattern to find the files.
    detrend
        Whether to apply detrending before extraction.
    mode
        The extraction mode. See :obj:`.quick_extraction`.

    Returns
    -------
    extracted
        The extracted data, as a row-stacked array, ordered by fibre number and
        with all brz cameras stitched together.

    """

    if isinstance(files_, str):
        dirname = os.path.dirname(os.path.abspath(files_))
        basename = os.path.basename(files_)
        files_ = list(pathlib.Path(dirname).glob(basename))

    if len(files_) != 9:
        raise ValueError("Invalid number of files passed.")

    files_ = [pathlib.Path(file_) for file_ in files_]

    with multiprocessing.Pool(processes=3) as pool:
        results = pool.map(partial(_extract_one, detrend=detrend, mode=mode), files_)

    extracted: dict[str, npt.NDArray[numpy.float32]] = {}
    for result in results:
        for key, value in result.items():
            extracted[key] = value

    fw_data: list[npt.NDArray[numpy.float32]] = []

    for spec in [1, 2, 3]:
        fw_cam: npt.NDArray[numpy.float32] | None = None
        for cam in ["b", "r", "z"]:
            cam_key = f"{cam}{spec}"
            if cam_key not in extracted:
                raise ValueError(f"Missing data for {cam_key}")

            cam_data = extracted[cam_key]
            if fw_cam is None:
                fw_cam = cam_data
            else:
                fw_cam = numpy.hstack([fw_cam, cam_data])

        assert fw_cam is not None
        fw_data.append(fw_cam)

    return numpy.vstack(fw_data).astype(numpy.float32)
