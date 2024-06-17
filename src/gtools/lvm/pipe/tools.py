#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-06-15
# @Filename: tools.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import nptyping as npt
import numpy
import pandas
import peewee
import polars
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import LinearNDInterpolator


__all__ = [
    "get_gaiaxp_cone",
    "XP_WAVE_SAMPLE",
    "slitmap_to_polars",
    "get_wavelength_array_from_header",
]

ARRAY_1D = npt.NDArray[npt.Shape["*"], npt.Float]
ARRAY_2D = npt.NDArray[npt.Shape["*,*"], npt.Float]


# Default wavelength sampling for the XP spectra in Angstroms.
XP_WAVE_SAMPLE = numpy.arange(3360.0, 10200.0 + 20.0, 20.0)


def get_gaiaxp_cone(
    connection: peewee.PostgresqlDatabase,
    ra: float,
    dec: float,
    radius: float,
    gmag_limit: float | None = None,
    limit: int | None = None,
    gaia_dr3_source_table: str = "catalogdb.gaia_dr3_source",
    gaia_dr3_xp_table="catalogdb.gaia_dr3_xp_sampled_mean_spectrum",
) -> polars.DataFrame:
    """Returns XP spectra in a region.

    Parameters
    ----------
    connection
        The database connection. Usually an ``sdssdb`` connection to the
        ``sdss5db`` database in ``operations``.
    ra, dec
        The coordinates of the centre of the cone.
    radius
        The radius of the cone in degrees.
    gmag_limit
        The limit in G magnitude. If not provided, no limit is applied.
    limit
        The maximum number of rows to return.
    gaia_dr3_source_table
        The name of the table with the Gaia DR3 sources. Can be a schema-qualified
        table name.
    gaia_dr3_xp_table
        The name of the table with the Gaia DR3 XP spectra. Can be a schema-qualified
        table name.

    Returns
    -------
    data
        A Polars data frame with the results of the query.

    """

    query = f"""
        SELECT xp.source_id, xp.ra, xp.dec, xp.flux AS flux_xp,
            xp.flux_error AS flux_error_xp, g3.phot_rp_mean_mag,
            g3.phot_bp_mean_mag, g3.phot_g_mean_mag
        FROM {gaia_dr3_xp_table} AS xp
        JOIN {gaia_dr3_source_table} AS g3 ON xp.source_id = g3.source_id
        WHERE q3c_radial_query(xp.ra, xp.dec, {ra}, {dec}, {radius})
    """

    if gmag_limit is not None:
        query += f" AND g3.phot_g_mean_mag < {gmag_limit}"

    if limit is not None:
        query += f" ORDER BY g3.phot_g_mean_mag LIMIT {limit}"

    df = polars.read_database(query, connection)

    # Cast flux and error to array of floats.
    df = df.with_columns(
        polars.col(["flux_xp", "flux_error_xp"])
        .str.strip_chars("[]")
        .str.split(",")
        .cast(polars.Array(polars.Float32, len(XP_WAVE_SAMPLE)))
    )

    # Add the wavelength column.
    df = df.with_columns(
        wave_xp=polars.lit(
            XP_WAVE_SAMPLE.tolist(),
            dtype=polars.Array(polars.Float32, len(XP_WAVE_SAMPLE)),
        )
    )

    return df


def slitmap_to_polars(
    slitmap: fits.fitsrec.FITS_rec
    | numpy.ndarray
    | Table
    | pandas.DataFrame
    | polars.DataFrame,
):
    """Converts a slitmap object to a Polars DataFrame."""

    if isinstance(slitmap, polars.DataFrame):
        return slitmap
    elif isinstance(slitmap, fits.fitsrec.FITS_rec):
        return polars.DataFrame(slitmap.tolist(), schema=slitmap.columns.names)  # type: ignore
    elif isinstance(slitmap, numpy.ndarray):
        return polars.from_numpy(slitmap)
    elif isinstance(slitmap, Table):
        return polars.DataFrame(slitmap.to_pandas())

    raise ValueError("slitmap type not recognised.")


def get_wavelength_array_from_header(header: fits.Header) -> numpy.ndarray:
    """Returns the wavelength array from a header WCS.

    This assumes the header corresponds to an RSS file in which the wavelength
    direction is the first FITS axis (along columns).

    """

    naxis = header["NAXIS1"]
    wave_start = header["CRVAL1"]
    wave_step = header["CDELT1"]
    ref_pixel = header["CRPIX1"] - 1  # 0-index based. WCS is 1-indexed.

    # Make reference pixel the first pixel.
    wave_start -= wave_step * ref_pixel

    return wave_start + wave_step * numpy.arange(naxis)


def slitmap_radec_to_xy(
    slitmap: polars.DataFrame,
    ra: float | ARRAY_1D,
    dec: float | ARRAY_1D,
    targettype: str | None = None,
) -> ARRAY_2D:
    """Converts RA, Dec to XY coordinates in the slitmap using piecewise interpolation.

    Parameters
    ----------
    slitmap
        The slitmap table.
    ra, dec
        The RA and Dec to convert.
    targettype
        The target type to select from the slitmap. If ``None``, all entries
        are used.

    Returns
    -------
    x, y
        The x and y coordinates in the slitmap.

    """

    slitmap = slitmap.clone()
    if targettype is not None:
        slitmap = slitmap.with_columns(polars.col.targettype.str.to_lowercase())
        slitmap = slitmap.filter(polars.col.targettype == targettype)

    interp = LinearNDInterpolator(slitmap[["ra", "dec"]], slitmap[["xpmm", "ypmm"]])

    return numpy.atleast_2d(interp(ra, dec))
