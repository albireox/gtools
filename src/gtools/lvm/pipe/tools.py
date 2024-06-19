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

from typing import Literal, Sequence

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
from scipy.interpolate import LinearNDInterpolator, interp1d


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
    "fframe_to_hobject",
    "get_sky_mask_uves",
    "get_z_continuum_mask",
    "interpolate_mask",
]

PathType = os.PathLike | str | pathlib.Path

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
                g3.phot_bp_mean_mag, g3.phot_g_mean_mag, g3.teff_gspphot
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


def _W_to_erg(data: float | Sequence[float]) -> float | Sequence[float]:
    """Converts W/s/micron to erg/s/cm^2/A."""

    factor = 1e7 * 1e-1 * 1e-4

    if isinstance(data, (int, float)):
        return factor * data

    flux = factor * numpy.array(data, dtype=numpy.float32)
    return flux.tolist()


def get_standard_info(file: PathType):
    """Returns a data frame with the standard information from the header.

    Parameters
    ----------
    file
        The path to the FITS file. Assumes that the first extension is the
        primary header with all the pointing and observational informational.

    Returns
    -------
    data
        A Polars data frame with the standards.

    """

    header = fits.getheader(file, 0)
    ccd = header.get("CCD", None).upper()

    data: list[dict] = []

    for istd in range(1, 16):
        data.append(
            {
                "std_id": istd,
                "source_id": header.get(f"STD{istd}ID", None),
                "ra": header.get(f"STD{istd}RA", None),
                "dec": header.get(f"STD{istd}DE", None),
                "acquired": header.get(f"STD{istd}ACQ", None),
                "t0": header.get(f"STD{istd}T0", None),
                "t1": header.get(f"STD{istd}T1", None),
                "exp_time": header.get(f"STD{istd}EXP", None),
                "fibre": header.get(f"STD{istd}FIB", None),
                "m_ab": header.get(f"STD{istd}{ccd}AB", None),
                "m_inst": header.get(f"STD{istd}{ccd}IN", None),
            }
        )

    df = polars.DataFrame(
        data,
        schema={
            "std_id": polars.Int16,
            "source_id": polars.Int64,
            "ra": polars.Float64,
            "dec": polars.Float64,
            "secz": polars.Float32,
            "acquired": polars.Boolean,
            "t0": polars.String,
            "t1": polars.String,
            "exp_time": polars.Float32,
            "fibre": polars.String,
            "m_ab": polars.Float32,
            "m_inst": polars.Float32,
        },
    )

    secz_data = []
    for row in df.iter_rows(named=True):
        if row["ra"] is None or row["dec"] is None:
            secz_data.append(None)
            continue
        secz = calculate_secz(row["ra"], row["dec"], row["t0"])
        secz_data.append(secz)

    df = df.with_columns(secz=polars.Series(secz_data, dtype=polars.Float32))

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


def get_extinction_correction(wave: ARRAY_1D) -> ARRAY_1D:
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


def get_sky_mask_uves(wave: ARRAY_1D, width: float = 3, threshold: float = 2):
    """Generate a mask for the bright sky lines.

    Mask every line at +/- width, where width in same units as wave (Angstroms).
    Only lines with a flux larger than threshold (in 10E-16 ergs/cm^2/s/A) are masked.
    The line list is from https://www.eso.org/observing/dfo/quality/UVES/pipeline/sky_spectrum.html

    Returns a bool Numpy array the same size as wave with sky line wavelengths
    marked as ``True``.

    """

    LVMCORE_DIR = os.getenv("LVMCORE_DIR")
    assert LVMCORE_DIR is not None, "LVMCORE_DIR is not set."

    p = os.path.join(LVMCORE_DIR, "etc", "UVES_sky_lines.txt")
    txt = numpy.genfromtxt(p)
    skyw, skyf = txt[:, 1], txt[:, 4]

    select = skyf > threshold
    lines = skyw[select]

    # Do NOT mask Ha if it is present in the sky table
    ha = (lines > 6562) & (lines < 6564)
    lines = lines[~ha]

    # Create the mask.
    mask = numpy.zeros_like(wave, dtype=bool)
    if width > 0.0:
        for line in lines:
            if (line <= wave[0]) or (line >= wave[-1]):
                continue
            ii = numpy.where((wave >= line - width) & (wave <= line + width))[0]
            mask[ii] = True

    return mask


def get_z_continuum_mask(wave: numpy.ndarray):
    """Some clean regions at the red edge of the NIR channel (hand picked).

    This is a positive mask, i.e., the regions to be masked are marked as ``True``
    are sky-free regions.

    """

    good = [
        [9230, 9280],
        [9408, 9415],
        [9464, 9472],
        [9608, 9512],
        [9575, 9590],
        [9593, 9603],
        [9640, 9650],
        [9760, 9775],
    ]
    mask = numpy.zeros_like(wave, dtype=bool)
    for r in good:
        if (r[0] <= wave[0]) or (r[1] >= wave[-1]):
            continue
        ii = numpy.where((wave >= r[0]) & (wave <= r[1]))[0]
        mask[ii] = True

    # Do not mask before first region
    mask[numpy.where(wave <= good[0][0])] = True

    return mask


def interpolate_mask(
    x: numpy.ndarray,
    y: numpy.ndarray,
    mask: numpy.ndarray,
    kind: str = "linear",
    fill_value: float | numpy.ndarray | Literal["extrapolate"] = "extrapolate",
):
    """Interpolates missing values in an array.

    Parameters
    ----------
    x, y
        Numpy arrays, samples and values.
    mask
        Boolean mask, ``True`` for masked values.
    kind
        Interpolation method, one of ``'linear'``, ``'nearest'``,
        ``'nearest-up'``, ``'zero'``, ``'slinear'``, ``'quadratic'``,
        ``'cubic'``, ``'previous'``, or ``'next'``.
    fill_value
        Which value to use for filling up data outside the convex hull of
        known pixel values. Default is 0, Has no effect for ``'nearest'``.

    Returns
    -------
    data
        The input array with missing values interpolated

    """

    if not numpy.any(mask):
        return y

    known_x, known_v = x[~mask], y[~mask]
    missing_x = x[mask]
    missing_idx = numpy.where(mask)

    f = interp1d(known_x, known_v, kind=kind, fill_value=fill_value)  # type: ignore
    yy = y.copy()
    yy[missing_idx] = f(missing_x)

    return yy


def fframe_to_hobject(fframe: PathType, outfile: PathType):
    """Converts an ``lvmFFrame`` file to ``lvm-object`` by undoing flux calibration."""

    fframe = pathlib.Path(fframe)
    outfile = pathlib.Path(outfile)

    if not fframe.exists():
        raise FileNotFoundError(f"FFrame file {fframe} not found.")

    hdul = fits.open(fframe)

    flux = hdul["FLUX"].data
    header = hdul[0].header

    header["FLUXCAL"] = False

    wave = get_wavelength_array_from_header(hdul[1].header)

    secz = header["TESCIAM"]

    slitmap = hdul["SLITMAP"].data

    fluxcal = hdul["FLUXCAL"].data
    t_fluxcal = Table(fluxcal)

    exptimes = numpy.ones(len(slitmap)) * header["EXPTIME"]
    for std_hd in t_fluxcal.colnames:
        if std_hd in ["mean", "rms"]:
            continue

        exptime = header[f"{std_hd[:-3]}EXP"]
        fiberid = header[f"{std_hd[:-3]}FIB"]

        if fiberid is None:
            continue

        exptimes[slitmap["orig_ifulabel"] == fiberid] = exptime

    exptimes = numpy.atleast_2d(exptimes).T

    ext = get_extinction_correction(wave)

    fluxcorr_factor = exptimes / 10 ** (0.4 * ext * secz) / fluxcal["mean"]

    flux *= fluxcorr_factor
    hdul["SKY_EAST"].data *= fluxcorr_factor
    hdul["SKY_WEST"].data *= fluxcorr_factor

    flux[hdul["MASK"].data > 0] = numpy.nan
    flux[~numpy.isfinite(flux)] = numpy.nan

    t_fluxcal.remove_columns(["mean", "rms"])

    header["CDELT1"] = hdul[1].header["CDELT1"]
    header["CRVAL1"] = hdul[1].header["CRVAL1"]
    header["CUNIT1"] = hdul[1].header["CUNIT1"]
    header["CTYPE1"] = hdul[1].header["CTYPE1"]
    header["CRPIX1"] = hdul[1].header["CRPIX1"]

    new_hdul = fits.HDUList(
        [
            fits.PrimaryHDU(data=flux, header=header),
            hdul["MASK"],
            hdul["SKY_EAST"],
            hdul["SKY_WEST"],
            fits.BinTableHDU(t_fluxcal, name="FLUXCAL"),
            hdul["SLITMAP"],
        ]
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    new_hdul.writeto(outfile, overwrite=True)
