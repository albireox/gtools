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

from gtools.lvm.pipe import log
from gtools.lvm.pipe.detrend import apply_bias, apply_overscan, apply_pixelmask


__all__ = ["quick_extraction", "gaussian_extract", "extract_and_stitch"]


DATA = pathlib.Path(__file__).parent / "data"

RAW_SHAPE: tuple[int, int] = (4080, 4120)
OVERSCAN_SHAPE: tuple[int, int] = (4080, 4086)


def quick_extraction(
    data: npt.NDArray[numpy.float32],
    camera: str,
    func: Literal["sum", "cent"] = "sum",
    subpixel: bool = True,
):
    """Runs a quick extraction using pre-computed traces.

    Parameters
    ----------
    data
        The data array from which to extract the data. Must be a 2D,
        overscan-corrected array, optionally with the bias and pixelmask
        corrections applied.
    camera
        The camera with which the data was taken (``b1``, ``b2``, ...)
    func
        The extraction function. Can be ``'sum'`` (sums the flux in the trace)
        or ``'cent'`` (returns the flux at the centre of the trace). In this
        quick extraction mode all pixels in the trace are fully included.
    subpixel
        If ``True``, the extraction will take into account the subpixel nature of
        the traces width.

    Returns
    -------
    data
        The extracted data, as a row-stacked array.

    """

    data = data.astype(numpy.float32)

    # Load the traces.
    cent_trace = polars.read_parquet(DATA / "cent_trace.parquet")
    width_trace = polars.read_parquet(DATA / "width_trace.parquet")

    cent_trace = cent_trace.filter(polars.col.ccd == camera)
    width_trace = width_trace.filter(polars.col.ccd == camera)

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

        fibre_data: npt.NDArray[numpy.float32]
        if func == "cent":
            fibre_data = numpy.array([data[int(ycent[xx]), xx] for xx in xrange])
        elif func == "sum":
            fibre_data = numpy.zeros(len(xrange), dtype=numpy.float32)
            for xx in xrange:
                fextr = data[y0[xx] : y1[xx] + 1, xx]

                if subpixel:
                    # Take into account the fraction of the first and last pixel.
                    frac = (ywidth[xx] - int(ywidth[xx])) / 2
                    fextr[0] *= frac
                    fextr[-1] *= frac
                    fibre_data[xx] = numpy.nansum(fextr)
        else:
            raise ValueError("Invalid func {func!r}.")

        extracted[nfibre - 1, :] = fibre_data.astype(numpy.float32)

    return extracted


def gaussian_extract(
    data: npt.NDArray[numpy.float32],
    camera: str,
    func: Literal["sum", "cent"] = "sum",
    threshold: float = 0.01,
):
    """Extraction using pre-computed traces using a Gaussian profile.

    For each wavelength element, the flux for a given fibre is extracted integrating
    over a Gaussian profile centred at the trace position and with a sigma equal to
    the width of the trace at that wavelength.

    Parameters
    ----------
    data
        The data array from which to extract the data. Must be a 2D,
        overscan-corrected array, optionally with the bias and pixelmask
        corrections applied.
    camera
        The camera with which the data was taken (``b1``, ``b2``, ...)
    func
        The extraction function to use. Can be ``'sum'`` (sums the flux in the trace)
        or ``'cent'`` (returns the flux at the centre of the trace).
    threshold
        The threshold below which the Gaussian profile is set to zero.

    Returns
    -------
    data
        The extracted data, as a row-stacked array.

    """

    data = data.astype(numpy.float32)

    if func not in ["sum", "cent"]:
        raise ValueError(f"Invalid func {func!r}.")

    # Load the traces.
    cent_trace = polars.read_parquet(DATA / "cent_trace.parquet")
    width_trace = polars.read_parquet(DATA / "width_trace.parquet")

    cent_trace = cent_trace.filter(polars.col.ccd == camera)
    width_trace = width_trace.filter(polars.col.ccd == camera)

    xrange = numpy.arange(data.shape[1])
    yrange = numpy.arange(data.shape[0])

    extracted: npt.NDArray[numpy.float32] = numpy.empty(
        (len(cent_trace), len(xrange)),
        dtype=numpy.float32,
    )

    for ii, row in enumerate(cent_trace.rows(named=True)):
        nfibre = row["nfibre"]

        cent_ncoeffs = len([col for col in row if col.startswith("coeff")])
        cent_coeffs = [row[f"coeff{i}"] for i in range(cent_ncoeffs)]
        fcent = numpy.polyval(cent_coeffs[::-1], xrange)

        width_row = width_trace.filter(polars.col.nfibre == nfibre).to_dicts()[0]
        width_ncoeffs = len([col for col in width_row if col.startswith("coeff")])
        width_coeffs = [width_row[f"coeff{i}"] for i in range(width_ncoeffs)]
        fwidth = numpy.polyval(width_coeffs[::-1], xrange)

        fibre_data = numpy.zeros(len(xrange), dtype=numpy.float32)

        for xx in xrange:
            aa = numpy.interp(fcent[xx], yrange, data[:, xx])

            if func == "cent":
                fibre_data[xx] = aa
                continue

            # The area below a Gaussian is a*sqrt(2*pi)*sigma, so we select the
            # value of a to match the value of the central pixel. We interpolate
            # the value of the central pixel.
            egauss = aa * numpy.abs(fwidth[xx]) * numpy.sqrt(2 * numpy.pi)
            fibre_data[xx] = egauss

        extracted[nfibre - 1, :] = fibre_data

    return extracted


def _extract_one(
    file_: pathlib.Path,
    calibs_directory: str | pathlib.Path | None = None,
    apply_calibrations: bool = True,
    extraction_mode: Literal["quick", "gaussian"] = "quick",
    extraction_func: Literal["sum", "cent"] = "sum",
):
    """Extracts the data from a single file."""

    ccd = fits.getheader(file_, ext=0)["CCD"]

    data = fits.getdata(file_, ext=0)
    assert isinstance(data, numpy.ndarray)

    if data.shape == RAW_SHAPE:
        log.debug(f"{file_!s}: applying overscan correction.")
        data = apply_overscan(file_, extension=0)

    calibs_directory = pathlib.Path(calibs_directory) if calibs_directory else None

    if calibs_directory and apply_calibrations:
        bias_file = calibs_directory / f"lvm-mbias-{ccd}.fits"
        pixelmask_file = calibs_directory / "pixelmasks" / f"lvm-mpixmask-{ccd}.fits"

        if bias_file.exists():
            data = apply_bias(data, bias_file)
        else:
            log.warning(f"{file_!s}: cannot find bias file {bias_file!s}")

        if pixelmask_file.exists():
            data = apply_pixelmask(data, pixelmask_file)
        else:
            log.warning(f"{file_!s}: cannot find pixelmask file {pixelmask_file!s}")

    if extraction_mode == "quick":
        return {ccd: quick_extraction(data, ccd, func=extraction_func)}
    elif extraction_mode == "gaussian":
        return {ccd: gaussian_extract(data, ccd, func=extraction_func)}


def extract_and_stitch(
    files_: Sequence[str | pathlib.Path] | str,
    calibs_directory: str | pathlib.Path | None = None,
    apply_calibrations: bool = True,
    extraction_mode: Literal["quick", "gaussian"] = "quick",
    extraction_func: Literal["sum", "cent"] = "sum",
    n_cpus: int = 4,
):
    """Extracts fibre data and returns a single array.

    Parameters
    ----------
    files_
        The list of files to extract, which must belong to the same exposure for
        all the different cameras. If a string is passed, it will be used as a
        glob pattern to find the files. If the format of the file data is not
        compatible with an overscan-corrected array, the `.apply_overscan` function
        will be called with the file.
    calibs_directory
        The directory containing the calibration files. If not provided no
        calibrations other than the overscan correction will be applied.
    apply_calibrations
        Applies the pixelmask and bias corrections. Set to ``False`` if the
        input data is already corrected. An overscan calibration is always applied
        if the data does not match the expected shape. If
        the ``lvm-mfiberflat_twilight-*.fits`` files are found in ``calibs_directory``,
        a relative flux correction is applied to the extracted spectra.
        This argument is ignored if ``calibs_directory`` is not provided.
    extraction_mode
        The extraction mode to use. Can be ``'quick'`` or ``'gaussian'``.
    extraction_func
        The function to use to evaluate the extracted data. See
        :obj:`.quick_extraction`.
    n_cpus
        Number of CPUs to use to extract the data using multiprocessing. Set to zero
        to disable multiprocessing.

    Returns
    -------
    extracted
        The extracted data, as a row-stacked array, ordered by fibre number and
        with all brz cameras stitched together. If cameras are missing in the
        list of input files, the corresponding section of the array will be
        filled with NaNs.

    """

    if isinstance(files_, str):
        dirname = os.path.dirname(os.path.abspath(files_))
        basename = os.path.basename(files_)
        files_ = list(pathlib.Path(dirname).glob(basename))

    files_ = [pathlib.Path(file_) for file_ in files_]

    nf: int = 648
    nw: int = 4086

    extract_partial = partial(
        _extract_one,
        calibs_directory=calibs_directory,
        apply_calibrations=apply_calibrations,
        extraction_mode=extraction_mode,
        extraction_func=extraction_func,
    )

    if n_cpus == 0 or n_cpus is False or n_cpus is None:
        results = map(extract_partial, files_)
    else:
        with multiprocessing.Pool(processes=n_cpus) as pool:
            results = pool.map(extract_partial, files_)

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
                cam_data = numpy.full((nf, nw), numpy.nan, dtype=numpy.float32)
            else:
                cam_data = extracted[cam_key]

            if fw_cam is None:
                fw_cam = cam_data
            else:
                fw_cam = numpy.hstack([fw_cam, cam_data])

        assert fw_cam is not None
        fw_data.append(fw_cam)

    return numpy.vstack(fw_data).astype(numpy.float32)
