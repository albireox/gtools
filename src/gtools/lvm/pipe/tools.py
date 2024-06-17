#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-06-15
# @Filename: tools.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib

from typing import Sequence

import nptyping as npt
import numpy
import pandas
import peewee
import polars
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from scipy import signal
from scipy.integrate import simps
from scipy.interpolate import LinearNDInterpolator


__all__ = [
    "get_gaiaxp_cone",
    "XP_WAVE_SAMPLE",
    "slitmap_to_polars",
    "get_wavelength_array_from_header",
    "get_extinction_correction",
    "slitmap_radec_to_xy",
    "calculate_secz",
    "mean_absolute_deviation",
    "butter_lowpass_filter",
    "filter_channel",
]

ARRAY_1D = npt.NDArray[npt.Shape["*"], npt.Float]
ARRAY_2D = npt.NDArray[npt.Shape["*,*"], npt.Float]


# Default wavelength sampling for the XP spectra in Angstroms.
XP_WAVE_SAMPLE = numpy.arange(3360.0, 10200.0 + 20.0, 20.0, dtype=numpy.float32)


def get_gaiaxp(
    connection: peewee.PostgresqlDatabase,
    source_id: int,
    extra_info: bool = True,
    gaia_dr3_source_table: str = "catalogdb.gaia_dr3_source",
    gaia_dr3_xp_table="catalogdb.gaia_dr3_xp_sampled_mean_spectrum",
) -> polars.DataFrame | None:
    """Returns the XP spectrum for a given Gaia DR3 source ID.

    Parameters
    ----------
    connection
        The database connection. Usually an ``sdssdb`` connection to the
        ``sdss5db`` database in ``operations``.
    source_id
        The Gaia DR3 source ID.
    extra_info
        If ``True``, returns magnitude information about the source from the
        Gaia DR3 source table (this will make the query slightly slower).
    gaia_dr3_source_table
        The name of the table with the Gaia DR3 sources. Can be a schema-qualified
        table name.
    gaia_dr3_xp_table
        The name of the table with the Gaia DR3 XP spectra. Can be a schema-qualified
        table name.

    Returns
    -------
    data
        A Polars data frame with the results of the query. Returns ``None``
        if there is not an XP spectrum associated with the source ID.
        XP fluxes are converted to erg/s/cm^2/A.

    """

    if extra_info:
        query = f"""
            SELECT xp.source_id, xp.ra, xp.dec, 0 AS wave_xp, xp.flux AS flux_xp,
                xp.flux_error AS flux_error_xp, g3.phot_rp_mean_mag,
                g3.phot_bp_mean_mag, g3.phot_g_mean_mag
            FROM {gaia_dr3_xp_table} AS xp
            JOIN {gaia_dr3_source_table} AS g3 ON xp.source_id = g3.source_id
            WHERE xp.source_id = {source_id}
        """
    else:
        query = f"""
            SELECT source_id, ra, dec, 0 AS wave_xp, flux AS flux_xp,
                flux_error AS flux_error_xp
            FROM {gaia_dr3_xp_table}
            WHERE source_id = {source_id}
        """

    df = polars.read_database(query, connection)

    if len(df) == 0:
        return None

    # Cast flux and error to array of floats.
    df = df.with_columns(
        polars.col(["flux_xp", "flux_error_xp"])
        .str.strip_chars("[]")
        .str.split(",")
        .cast(polars.Array(polars.Float32, len(XP_WAVE_SAMPLE)))
    )

    # Convert to erg/s/cm^2/A.
    df = df.with_columns(
        polars.col("flux_xp", "flux_error_xp")
        .map_elements(_W_to_erg, return_dtype=polars.List(polars.Float64))
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


def get_gaiaxp_cone(
    connection: peewee.PostgresqlDatabase,
    ra: float,
    dec: float,
    radius: float,
    extra_info: bool = True,
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
    extra_info
        If ``True``, returns magnitude information about the source from the
        Gaia DR3 source table (this will make the query slightly slower).
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
        XP fluxes are converted to erg/s/cm^2/A.

    """

    if extra_info:
        query = f"""
            SELECT xp.source_id, xp.ra, xp.dec, 0 AS wave_xp, xp.flux AS flux_xp,
                xp.flux_error AS flux_error_xp, g3.phot_rp_mean_mag,
                g3.phot_bp_mean_mag, g3.phot_g_mean_mag
            FROM {gaia_dr3_xp_table} AS xp
            JOIN {gaia_dr3_source_table} AS g3 ON xp.source_id = g3.source_id
            WHERE q3c_radial_query(xp.ra, xp.dec, {ra}, {dec}, {radius})
        """
    else:
        query = f"""
            SELECT source_id, ra, dec, 0 AS wave_xp, flux AS flux_xp,
                flux_error AS flux_error_xp
            FROM {gaia_dr3_xp_table}
            WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {radius})
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

    # Convert to erg/s/cm^2/A.
    df = df.with_columns(
        polars.col("flux_xp", "flux_error_xp")
        .map_elements(_W_to_erg, return_dtype=polars.List(polars.Float64))
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


def _W_to_erg(
    data: float | Sequence[float],
) -> float | Sequence[float]:
    """Converts W/s/micron to erg/s/cm^2/A."""

    factor = 1e7 * 1e-1 * 1e-4

    if isinstance(data, (int, float)):
        return factor * data

    flux = factor * numpy.array(data, dtype=numpy.float32)
    return flux.tolist()


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


def get_extinction_correction(wave: ARRAY_1D) -> ARRAY_2D:
    """Returns the extinction correction at LCO for the given wavelengths."""

    LVMCORE_DIR = os.getenv("LVMCORE_DIR")
    if LVMCORE_DIR is None:
        raise ValueError("LVMCORE_DIR is not set.")

    ext_curve_file = pathlib.Path(LVMCORE_DIR + "/etc/lco_extinction.txt")
    if not ext_curve_file.exists():
        raise FileNotFoundError(f"Extinction curve file {ext_curve_file} not found.")

    txt = numpy.genfromtxt(str(ext_curve_file))
    lext, ext = txt[:, 0], txt[:, 1]
    ext_wave = numpy.interp(wave, lext, ext)

    return ext_wave


def calculate_secz(ra: float, dec: float, isot: str) -> float:
    """Calculates the airmass for a given observation at LCO."""

    lco = EarthLocation.from_geodetic(
        lon=-70.70166667,
        lat=-29.00333333,
        height=2282.0,
    )

    skyc = SkyCoord(ra=ra, dec=dec, unit="deg")
    time = Time(isot, scale="utc", format="isot")
    altaz = skyc.transform_to(AltAz(obstime=time, location=lco))

    secz = 1.0 / numpy.cos(numpy.radians(90.0 - float(altaz.alt.deg)))

    return secz


def mean_absolute_deviation(vals):
    """Robust estimate of RMS.

    See https://en.wikipedia.org/wiki/Median_absolute_deviation

    """

    mval = numpy.nanmedian(vals)
    rms = 1.4826 * numpy.nanmedian(numpy.abs(vals - mval))

    return mval, rms


def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    """Applies a Butterworth low-pass filter to the data."""

    normal_cutoff = float(cutoff_freq) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype="lowpass")
    y = signal.filtfilt(b, a, data)

    return y


def filter_channel(w, f, k=3):
    """Filters a channel using a low-pass Butterworth filter."""

    c = numpy.where(numpy.isfinite(f))
    s = butter_lowpass_filter(f[c], 0.01, 2)
    res = s - f[c]

    mres, rms = mean_absolute_deviation(res)
    good = numpy.where(numpy.abs(res - mres) < k * rms)

    return w[c][good], f[c][good]


def spec_to_LVM_mAB(channel: str, w: ARRAY_1D, f: ARRAY_1D):
    """Calculates the LVM AB magnitude from a spectrum.

    LVM photometric system: Gaussian filter with sigma 250A centred in channels
    at 4500, 6500, and 8500A.

    """

    if channel == "b":
        return spec_to_mAB(w, f, w, numpy.exp(-0.5 * ((w - 4500) / 250) ** 2))
    elif channel == "r":
        return spec_to_mAB(w, f, w, numpy.exp(-0.5 * ((w - 6500) / 250) ** 2))
    else:
        return spec_to_mAB(w, f, w, numpy.exp(-0.5 * ((w - 8500) / 250) ** 2))


def spec_to_mAB(lam, spec, lamf, filt):
    """Calculate AB magnitude in filter.

    Determines the AB magnitude ``(lamf, filt)`` given a spectrum
    ``(lam, spec)`` in ergs/s/cm^2/A.

    """

    c_AAs = 2.99792458e18  # Speed of light in Angstrom/s

    filt_int = numpy.interp(lam, lamf, filt)  # Interpolate to common wavelength axis

    I1 = simps(spec * filt_int * lam, lam)
    I2 = simps(filt_int / lam, lam)

    fnu = I1 / I2 / c_AAs  # Average flux density
    mab = -2.5 * numpy.log10(fnu) - 48.6  # AB magnitude

    if numpy.isnan(mab):
        mab = -9999.9

    return mab
