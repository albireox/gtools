#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-06-15
# @Filename: self_calibration.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib

import nptyping
import numpy
import peewee
import polars
import seaborn
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate

from sdsstools.logger import get_logger

from gtools.lvm.pipe.tools import (
    calculate_secz,
    filter_channel,
    get_extinction_correction,
    get_gaiaxp_cone,
    get_wavelength_array_from_header,
    slitmap_radec_to_xy,
    slitmap_to_polars,
)
from gtools.lvm.plotting import plot_rss


ARRAY_2D_F32 = nptyping.NDArray[nptyping.Shape["*,*"], nptyping.Float32]
PathType = os.PathLike | str | pathlib.Path


log = get_logger("gtools.lvm.pipe.self_calibration", use_rich_handler=True)


def lvm_fluxcal_self(
    hobject: PathType,
    connection: peewee.PostgresqlDatabase | None = None,
    silent: bool = False,
    plot: bool = False,
    plot_dir: PathType | None = None,
    gmag_limit: float | None = None,
    nstar_limit: int | None = None,
    max_sep: float = 7.0,
):
    """Performs flux calibration of a science frame using stars in the field.

    Parameters
    ----------
    hobject
        The path to the ``hobject`` file. This must be final stage of the ``hobject``
        file, after flux calibration has been performed and the sensitivity functions
        have been added to the ``FLUXCAL`` extension, but before the flux calibration
        has been applied to the data (the ``lvmFFrame`` file).
    connection
        The database connection to ``sdss5db`` used to retrieve the Gaia sources.
        If ``None``, uses the default connection from ``sdssdb``.
    silent
        If ``True``, does not print any output.
    plot
        If ``True``, generate plots.
    plot_dir
        The directory where to save the plots. If ``None``, defaults to the
        directory where the ``hobject`` file is located.
    gmag_limit
        The limit in G magnitude for the Gaia sources. If ``None``, no limit is applied.
    nstar_limit
        The maximum number of stars to use for the calibration. This limit is only
        applied after all other filters have been applied.
    max_sep
        The maximum separation to a fibre centre, in arcsec, to consider a star
        as a calibration source.

    """

    if silent is False:
        log.sh.setLevel(5)
    else:
        log.sh.setLevel(100)

    hobject = pathlib.Path(hobject).absolute()

    if not hobject.exists():
        raise FileNotFoundError(f"{hobject} does not exist.")

    if plot_dir is None:
        plot_dir = hobject.parent
    else:
        plot_dir = pathlib.Path(plot_dir).absolute()

    log.info(f"Processing file {hobject!s}")

    hdul = fits.open(hobject)

    data = hdul["PRIMARY"].data.astype(numpy.float32)
    wave = get_wavelength_array_from_header(hdul[0].header)

    slitmap = slitmap_to_polars(hdul["SLITMAP"].data)

    bore_ra = hdul[0].header["POSCIRA"]
    bore_dec = hdul[0].header["POSCIDE"]
    secz = calculate_secz(bore_ra, bore_dec, hdul[0].header["OBSTIME"])

    exp_time = hdul[0].header["EXPTIME"]

    log.info(f"Retrieving Gaia sources around ({bore_ra:.3f}, {bore_dec:.3f}) deg.")

    if connection is None:
        log.warning("No connection provided. Using default connection from sdssdb.")
        from sdssdb.peewee.sdss5db import database as connection

    gaia_sources = get_gaiaxp_cone(
        connection,
        bore_ra,
        bore_dec,
        0.25,
        gmag_limit=gmag_limit,
        limit=None,
    )
    log.debug(f"{len(gaia_sources)} Gaia DR3 XP sources retrieved.")

    log.debug(
        "Rejecting stars with with separation to "
        f"fibre centre > {max_sep:.1f} arcsec."
    )

    # Get neighrest fibre in the slitmap for each Gaia star.
    slitmap_sci = slitmap.filter(polars.col.targettype == "science")
    gaia_coords = SkyCoord(gaia_sources["ra"], gaia_sources["dec"], unit="deg")
    slitmap_coords = SkyCoord(slitmap_sci["ra"], slitmap_sci["dec"], unit="deg")

    idx, sep2d, _ = gaia_coords.match_to_catalog_sky(slitmap_coords)
    mask = sep2d.arcsec < max_sep
    fiberid = slitmap_sci["fiberid"].to_numpy()[idx]

    gaia_sources = gaia_sources.with_columns(valid=mask, fiberid=polars.lit(fiberid))

    # Add index column and mark cases where multiple stars
    # fall on the same fibre as invalid.
    gaia_sources = gaia_sources.with_row_index("index", 1)

    gaia_sources_dup = gaia_sources["fiberid"].is_duplicated()
    gaia_sources[gaia_sources_dup.arg_true(), "valid"] = False

    log.info(f"Number of Gaia sources within max_sep: {gaia_sources['valid'].sum()}")

    gaia_valid = gaia_sources.filter(polars.col.valid)
    if nstar_limit is not None and gaia_valid.height > nstar_limit:
        log.info(f"Selecting top {nstar_limit} sources.")
        gaia_valid = gaia_valid.sort("phot_g_mean_mag").head(nstar_limit)
        gaia_sources = gaia_sources.with_columns(
            valid=polars.col.index.is_in(gaia_valid["index"])
        )

    # Calculate the xy position of the Gaia sources in the slitmap. Mostly for plotting.
    gaia_slitmap_xy = slitmap_radec_to_xy(
        slitmap,
        gaia_sources["ra"].to_numpy(),
        gaia_sources["dec"].to_numpy(),
        targettype="science",
    )

    gaia_sources = gaia_sources.with_columns(
        xpmm=polars.lit(gaia_slitmap_xy[:, 0], dtype=polars.Float32),
        ypmm=polars.lit(gaia_slitmap_xy[:, 1], dtype=polars.Float32),
    )

    if plot:
        # Plot the IFU map with the Gaia sources, colouring by validity.
        _plot_ifu_map(data, slitmap, gaia_sources, plot_dir, hobject)

    # Create a dataframe of fibre fluxes.
    df = slitmap_sci.clone()
    df = df.insert_at_idx(1, polars.Series("fiber_idx", df["fiberid"] - 1))

    df = df.with_columns(
        flux=polars.Series(
            data[df["fiber_idx"], :],
            dtype=polars.Array(polars.Float32, data.shape[1]),
        )
    )

    # Combine with Gaia data.
    valid_gaia = gaia_sources.filter(polars.col.valid)
    valid_gaia = valid_gaia.rename({"ra": "ra_gaia", "dec": "dec_gaia"})

    df = df.join(
        valid_gaia[
            [
                "fiberid",
                "source_id",
                "ra_gaia",
                "dec_gaia",
                "wave_xp",
                "flux_xp",
                "phot_g_mean_mag",
            ]
        ],
        on="fiberid",
        how="left",
    )

    # Resample Gaia XP spectrum to the same wavelength as the science frame.
    log.debug("Resampling Gaia XP spectra to the same wavelength as the science frame.")

    flux_xp_resampled = (
        df["wave_xp", "flux_xp"]
        .map_rows(lambda row: _interp_gaia_xp_udf(wave, *row))
        .cast(polars.Array(polars.Float32, wave.size))
    )
    df = df.with_columns(flux_xp_resampled=flux_xp_resampled.to_series())

    # Retrieve sky corrections.
    log.debug("Retrieving sky correction.")
    sky_corr = get_weighted_sky_corr(hobject)

    df = df.with_columns(
        sky_corr=polars.Series(
            sky_corr[df["fiber_idx"]],
            dtype=polars.Array(polars.Float32, sky_corr.shape[1]),
        )
    )

    # Get extinction correction.
    log.debug("Getting extinction correction for LCO.")
    ext = get_extinction_correction(wave)

    # Calculate the sensitivity function.
    log.info("Calculating sensitivity function for each source.")
    sens = map(
        lambda row: _calc_sensitivity_udf(row, wave, ext, secz, exp_time), df.to_dicts()
    )

    array_type = polars.Array(polars.Float32, wave.size)
    sens_df = polars.DataFrame(sens).cast(array_type)

    df = df.with_columns(sens_df)

    _plot_sensitivity_function(df, wave, plot_dir, hobject)


def _plot_ifu_map(
    data: ARRAY_2D_F32,
    slitmap: polars.DataFrame,
    gaia_sources: polars.DataFrame,
    plot_dir: pathlib.Path,
    hobject: pathlib.Path,
):
    """Plot the IFU map with the Gaia sources."""

    fig_rss, ax_rss = plot_rss(
        data,
        slitmap,
        fibre_type="science",
        mode="xy",
        patch="hex",
        cmap="mako_r",
        interactive=False,
    )
    fig_rss.savefig(plot_dir / f"{hobject.stem}_map.pdf")

    seaborn.set_palette("deep")

    valid_xy = gaia_sources.filter(
        polars.col.xpmm.is_not_nan(),
        polars.col.ypmm.is_not_nan(),
    )

    selected_gaia = valid_xy.filter(polars.col.valid)
    ax_rss.scatter(
        selected_gaia["xpmm"],
        selected_gaia["ypmm"],
        color="r",
        marker="*",
        s=100,
    )

    not_selected_gaia = valid_xy.filter(polars.col.valid.not_())
    ax_rss.scatter(
        not_selected_gaia["xpmm"],
        not_selected_gaia["ypmm"],
        color="0.7",
        marker="*",
        s=100,
    )

    fig_rss.savefig(plot_dir / f"{hobject.stem}_map_gaia.pdf")

    seaborn.reset_defaults()

    plt.close("all")


def _plot_sensitivity_function(
    df: polars.DataFrame,
    wave: numpy.ndarray,
    plot_dir: pathlib.Path,
    hobject: pathlib.Path,
):
    """Plot the sensitivity function for each Gaia source."""

    stem = str(plot_dir / f"{hobject.stem}")

    seaborn.set_theme(context="paper", style="ticks", font_scale=0.9)

    gaia = df.filter(polars.col.source_id.is_not_null())

    with plt.ioff():
        with PdfPages(f"{stem}_sensitivity.pdf") as pdf:
            for row in gaia.to_dicts():
                source_id = row["source_id"]
                fiberid = row["fiberid"]
                ra = row["ra_gaia"]
                dec = row["dec_gaia"]

                flux_corrected = numpy.array(row["flux_corrected"], dtype=numpy.float32)
                flux_xp = numpy.array(row["flux_xp_resampled"], dtype=numpy.float32)

                sens_smooth = row["sens_smooth"]

                flux_corrected /= numpy.nanmedian(flux_corrected[500:-500])
                flux_xp /= numpy.nanmedian(flux_xp)

                fig, ax = plt.subplots()
                ax2 = ax.twinx()

                ax2.plot(wave, sens_smooth, "k-", zorder=10, label="Sensitivity")

                ax.plot(
                    wave,
                    flux_corrected,
                    color="0.8",
                    linestyle="-",
                    zorder=0,
                    label="Corrected flux",
                )
                ax.plot(
                    wave,
                    flux_xp,
                    color="r",
                    linestyle="-",
                    zorder=20,
                    label="XP flux",
                )

                ax2.text(
                    0.5,
                    0.98,
                    f"source_id={source_id}\nfiberid={fiberid} ({ra:.3f}, {dec:.3f})",
                    horizontalalignment="center",
                    verticalalignment="top",
                    transform=ax2.transAxes,
                )

                # Labels and legeneds
                ax.set_xlabel("Wavelength [A]")

                ax.set_ylabel("Normalised flux")
                ax2.set_ylabel("Sensitivity response (XP / corrected flux)")

                ax.set_ylim(0, 2)

                ax.legend(loc="upper left")
                ax2.legend(loc="upper right")

                pdf.savefig(fig)
                plt.close(fig)


def _interp_gaia_xp_udf(
    wave: numpy.ndarray,
    xp_wave: numpy.ndarray | None,
    xp_flux: numpy.ndarray | None,
) -> polars.Series | None:
    """Interpolates a Gaia XP spectrum to the science frame wavelength."""

    if xp_wave is None or xp_flux is None:
        return None

    return polars.Series(numpy.interp(wave, xp_wave, xp_flux))


def _calc_sensitivity_udf(
    row: dict, wave: numpy.ndarray, ext: numpy.ndarray, secz: float, exp_time: float
) -> dict:
    """Calculates the sensitivity function for a given source."""

    flux = numpy.array(row["flux"], dtype=numpy.float32)
    sky_corr = numpy.array(row["sky_corr"], dtype=numpy.float32)

    flux_corrected = ((flux - sky_corr) / exp_time) * 10 ** (0.4 * ext * secz)
    flux_corrected = flux_corrected.tolist()

    if row["flux_xp_resampled"] is None:
        return {"flux_corrected": flux_corrected}

    flux_xp_resampled = numpy.array(row["flux_xp_resampled"], dtype=numpy.float32)

    sens = flux_xp_resampled / flux_corrected

    wgood, sgood = filter_channel(wave, sens, 2)
    sens_smooth = interpolate.make_smoothing_spline(wgood, sgood, lam=1e4)

    return {
        "flux_corrected": flux_corrected,
        "sens": sens.tolist(),
        "sens_smooth": sens_smooth(wave).tolist(),
    }


def get_weighted_sky_corr(
    file: PathType,
    weights: tuple[float, float] | None = None,
) -> ARRAY_2D_F32:
    """Returns the weighted sky correction array.

    Parameters
    ----------
    file
        The file containing the measured sky values. It must include extensions
        ``SKY_EAST`` and ``SKY_WEST``.
    weights
        The weights to apply to the sky values (first element for east, second
        for west). If ``None``, tries to use the ``SKYEW`` and ``SKYWW`` header
        values. Otherwise defaults to equal weights.

    Returns
    -------
    sky_corr
        RSS array with the sky correction for each fibre.

    """

    hdul = fits.open(file)

    assert "SKY_EAST" in hdul, "SKY_EAST extension not found."
    assert "SKY_WEST" in hdul, "SKY_WEST extension not found."

    if weights is None:
        prim_header = hdul[0].header
        if "SKYEW" in prim_header and "SKYWW" in prim_header:
            weights = (prim_header["SKYEW"], prim_header["SKYWW"])
        else:
            weights = (0.5, 0.5)

    sky_corr = hdul["SKY_EAST"].data * weights[0] + hdul["SKY_WEST"].data * weights[1]

    return sky_corr.astype(numpy.float64)
