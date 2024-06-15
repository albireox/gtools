#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-06-03
# @Filename: sframe_to_parquet.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import json
import multiprocessing
import os
import pathlib
import random
from functools import partial

import healpy
import numpy
import polars
from astropy.io import fits
from rich.progress import Progress


__all__ = ["sframe_to_parquet"]


def sframe_to_parquet(
    file: os.PathLike | pathlib.Path,
    outfile: os.PathLike | pathlib.Path | None = None,
    write_headers: bool = True,
):
    """Converts an ``lvmSFrame`` file to a Parquet file."""

    file = pathlib.Path(file)

    hdul = fits.open(file)

    header = hdul[0].header

    flux = hdul["FLUX"].data.astype("float32")

    n_fibres = flux.shape[0]
    n_wavelengths = flux.shape[1]
    frange = numpy.arange(n_fibres)  # Range of fibre IDs
    wrange = numpy.arange(n_wavelengths)  # Range of wavelength elements

    meshgrid = numpy.meshgrid(frange, wrange, indexing="ij")

    f_flatten = meshgrid[0].flatten() + 1  # Fibre IDs are 1-indexed
    w_flatten = meshgrid[1].flatten()

    flux_flatten = flux.flatten()

    ivar_flatten = hdul["IVAR"].data.astype("float32").flatten()
    mask_flatten = hdul["MASK"].data.astype("uint8").flatten()
    wave_flatten = numpy.tile(hdul["WAVE"].data.astype("float32").flatten(), n_fibres)
    lsf_flatten = hdul["LSF"].data.astype("float32").flatten()
    sky_flatten = hdul["SKY"].data.astype("float32").flatten()
    sky_ivar_flatten = hdul["SKY_IVAR"].data.astype("float32").flatten()

    slitmap = hdul["SLITMAP"].data
    ra = numpy.repeat(slitmap["ra"].astype("float32"), n_wavelengths)
    dec = numpy.repeat(slitmap["dec"].astype("float32"), n_wavelengths)
    targettype = numpy.repeat(slitmap["targettype"].astype("S8"), n_wavelengths)

    healpix = healpy.ang2pix(2**14, ra, dec, lonlat=True, nest=True)

    df = polars.DataFrame(
        [
            polars.Series("fiberid", f_flatten, dtype=polars.UInt16),
            polars.Series("wavelength_index", w_flatten, dtype=polars.UInt16),
            polars.Series("wavelength", wave_flatten, dtype=polars.Float32),
            polars.Series("flux", flux_flatten, dtype=polars.Float32),
            polars.Series("ivar", ivar_flatten, dtype=polars.Float32),
            polars.Series("mask", mask_flatten, dtype=polars.UInt8),
            polars.Series("lsf", lsf_flatten, dtype=polars.Float32),
            polars.Series("sky", sky_flatten, dtype=polars.Float32),
            polars.Series("sky_ivar", sky_ivar_flatten, dtype=polars.Float32),
            polars.Series("ra", ra, dtype=polars.Float32),
            polars.Series("dec", dec, dtype=polars.Float32),
            polars.Series("targettype", targettype, dtype=polars.String),
            polars.Series("healpix_norder14_nest", healpix, dtype=polars.UInt32),
        ]
    )

    df = df.with_columns(targettype=polars.col.targettype.str.to_lowercase())

    df = df.with_columns(
        tile_id=polars.lit(header["TILE_ID"], dtype=polars.Int32),
        sframe_filename=polars.lit(file.name, dtype=polars.String),
    )

    if outfile is None:
        return df

    outfile = pathlib.Path(outfile)
    df.write_parquet(
        outfile,
        use_pyarrow=True,
        # compression="lz4",
        pyarrow_options={"partition_cols": ["tile_id"]},
    )

    if write_headers:
        headers = headers_to_dict(hdul)
        outfile_json = outfile.with_suffix(".json")
        with open(outfile_json, "w") as f:
            json.dump(headers, f, indent=4)

        return df, headers

    return df


def headers_to_dict(file: os.PathLike | pathlib.Path | fits.HDUList):
    """Converts the headers of an ``lvmSFrame`` file to a dictionary."""

    hdul: fits.HDUList
    if isinstance(file, fits.HDUList):
        hdul = file
    else:
        file = pathlib.Path(file)
        hdul = fits.open(file)

    headers = {}

    for hdu in hdul:
        hdu_header = dict(hdu.header)
        hdu_header_clean = {
            kk: vv
            for kk, vv in hdu_header.items()
            if not isinstance(vv, fits.header._HeaderCommentaryCards)
        }
        headers[hdu.name] = hdu_header_clean

    return headers


def _batch_convert_helper(outdir: pathlib.Path, file: pathlib.Path):
    """Helper for `.batch_convert`."""

    # outfile = outdir / file.with_suffix(".parquet").name
    sframe_to_parquet(file, outfile=outdir)


def batch_convert(
    inpath: os.PathLike | pathlib.Path,
    outpath: os.PathLike | pathlib.Path,
):
    """Batch converts all ``lvmSFrame`` files in a directory to Parquet."""

    inpath = pathlib.Path(inpath)
    outpath = pathlib.Path(outpath)

    outpath.mkdir(exist_ok=True, parents=True)

    files = list(inpath.glob("**/lvmSFrame*.fits"))
    random.shuffle(files)

    batch_convert_helper_partial = partial(_batch_convert_helper, outpath)

    bar = Progress()
    task_id = bar.add_task("Converting", total=len(list(files)))
    bar.start()

    with multiprocessing.Pool(32) as pool:
        for __ in pool.imap_unordered(batch_convert_helper_partial, files):
            bar.update(task_id=task_id, advance=1)
