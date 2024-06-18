#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-06-15
# @Filename: flux_calibration.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib

from typing import Literal

import nptyping
import numpy
import peewee
import polars
import seaborn
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import biweight_location, biweight_scale
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate

from sdsstools.logger import get_logger

from gtools.lvm.pipe.tools import (
    calculate_secz,
    filter_channel,
    get_extinction_correction,
    get_gaiaxp,
    get_gaiaxp_cone,
    get_sky_mask_uves,
    get_standard_info,
    get_wavelength_array_from_header,
    get_z_continuum_mask,
    interpolate_mask,
    slitmap_radec_to_xy,
    slitmap_to_polars,
)
from gtools.lvm.plotting import plot_rss


__all__ = ["flux_calibration_self", "flux_calibration"]


ARRAY_2D_F32 = nptyping.NDArray[nptyping.Shape["*,*"], nptyping.Float32]
PathType = os.PathLike | str | pathlib.Path


log = get_logger("gtools.lvm.pipe.flux_calibration", use_rich_handler=True)


def flux_calibration(
    hobject: PathType,
    connection: peewee.PostgresqlDatabase | None = None,
    silent: bool = False,
    plot: bool = False,
    plot_dir: PathType | None = None,
):
    """Performs flux calibration using the observed standards.

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

    Returns
    -------
    df
        A DataFrame with the sensitivity function for each standard star.

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

    plot_stem = str(plot_dir / f"{hobject.stem}")

    if connection is None:
        log.warning("No connection provided. Using default connection from sdssdb.")
        from sdssdb.peewee.sdss5db import database as connection

    log.info(f"Processing file {hobject!s}")

    hdul = fits.open(hobject)

    data = hdul["PRIMARY"].data.astype(numpy.float32)

    wave = get_wavelength_array_from_header(hdul[0].header)
    ARRAY_TYPE = polars.Array(polars.Float32, wave.size)

    channel = hdul[0].header["CCD"].lower()

    slitmap = slitmap_to_polars(hdul["SLITMAP"].data)
    std_map = slitmap.filter(polars.col.targettype == "standard")

    standards = get_standard_info(hobject)
    standards = standards.rename({"ra": "ra_std", "dec": "dec_std"})

    df = std_map.join(standards, left_on="orig_ifulabel", right_on="fibre")
    df = df.insert_at_idx(1, polars.Series("fiber_idx", df["fiberid"] - 1))

    flux = data[df["fiber_idx"], :]

    df = df.with_columns(flux=polars.Series(flux, dtype=ARRAY_TYPE))

    log.debug("Retrieving Gaia sources.")

    gaia_data = [
        get_gaiaxp(connection, row["source_id"])
        for row in df.iter_rows(named=True)
        if row["source_id"]
    ]
    gaia_data = polars.concat([row for row in gaia_data if row is not None])
    gaia_data = gaia_data.rename({"ra": "ra_gaia", "dec": "dec_gaia"})

    df = df.join(gaia_data, on="source_id")

    df = _post_process_flux_calibration_df(
        df,
        channel,
        wave,
        hobject,
        plot=plot,
        plot_stem=plot_stem,
    )

    return df


def flux_calibration_self(
    hobject: PathType,
    connection: peewee.PostgresqlDatabase | None = None,
    gmag_limit: float | None = None,
    nstar_limit: int | None = None,
    max_sep: float = 7.0,
    reject_multiple_per_fibre: bool | Literal["keep_brightest"] = True,
    silent: bool = False,
    plot: bool = False,
    plot_dir: PathType | None = None,
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
    gmag_limit
        The limit in G magnitude for the Gaia sources. If ``None``, no limit is applied.
    nstar_limit
        The maximum number of stars to use for the calibration. This limit is only
        applied after all other filters have been applied.
    max_sep
        The maximum separation to a fibre centre, in arcsec, to consider a star
        as a calibration source.
    reject_multiple_per_fibre
        If ``True``, rejects all stars that fall on the same fibre as another star.
        ``False`` will consider all the stars valid. If ``keep_brightest``, fibres with
        more than one star will keep the brightest one in the G band.
    silent
        If ``True``, does not print any output.
    plot
        If ``True``, generate plots.
    plot_dir
        The directory where to save the plots. If ``None``, defaults to the
        directory where the ``hobject`` file is located.

    Returns
    -------
    df
        A DataFrame with the corrected fluxes for each science fibre and the
        sensitivity function for each matched Gaia star.

    """

    if silent is False:
        log.sh.setLevel(5)
    else:
        log.sh.setLevel(100)

    if reject_multiple_per_fibre not in [True, False, "keep_brightest"]:
        raise ValueError("Invalid value for reject_multiple_per_fibre.")

    hobject = pathlib.Path(hobject).absolute()

    if not hobject.exists():
        raise FileNotFoundError(f"{hobject} does not exist.")

    if plot_dir is None:
        plot_dir = hobject.parent
    else:
        plot_dir = pathlib.Path(plot_dir).absolute()

    plot_stem = str(plot_dir / f"{hobject.stem}")

    log.info(f"Processing file {hobject!s}")

    hdul = fits.open(hobject)

    data = hdul["PRIMARY"].data.astype(numpy.float32)
    wave = get_wavelength_array_from_header(hdul[0].header)

    channel = hdul[0].header["CCD"].lower()

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
        f"fibre centre < {max_sep:.1f} arcsec."
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

    # Deal with fibres that have more than one star associated with them.
    if reject_multiple_per_fibre is True:
        # Reject all stars that have a duplicate fibre.
        gaia_sources_dup = gaia_sources["fiberid"].is_duplicated()
        gaia_sources[gaia_sources_dup.arg_true(), "valid"] = False
    elif reject_multiple_per_fibre == "keep_brightest":
        # Keep the brightest star in the G band.
        valid_gaia = (
            gaia_sources.filter(polars.col.valid)
            .group_by("fiberid")
            .agg(polars.col.source_id.top_k_by("phot_g_mean_mag", k=1))
            .explode("source_id")
        )
        gaia_sources = gaia_sources.with_columns(
            valid=polars.col.source_id.is_in(valid_gaia["source_id"])
        )
    else:
        # Keep all.
        pass

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
        _plot_ifu_map(data, slitmap, gaia_sources, plot_stem)

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

    # Add secz and exp_time columns. _post_process_flux_calibration_df will
    # use them to apply the extinction correction.
    df = df.with_columns(secz=polars.lit(secz), exp_time=polars.lit(exp_time))

    df = _post_process_flux_calibration_df(
        df,
        channel,
        wave,
        hobject,
        plot=plot,
        plot_stem=plot_stem,
    )

    return df


def _post_process_flux_calibration_df(
    df: polars.DataFrame,
    channel: str,
    wave: numpy.ndarray,
    hobject: pathlib.Path,
    plot: bool = False,
    plot_stem: str | None = None,
):
    """Helper function common to ``flux_calibration`` and ``flux_calibration_self``.

    This function receives a data frame that includes the fluxes for each relevant
    fibre and associated Gaia XP fluxes. It performs the following tasks:

    - Resamples the Gaia XP spectra to the same wavelength as the LVM frame.
    - Retrieves and applies the sky correction for the LVM frame.
    - Retrieves the extinction correction for LCO and applies it.
    - Calculates the sensitivity function for each source.
    - Optionally, plots the sensitivity functions.

    Returns the data frame with the sensitivity functions added.

    """

    if plot:
        assert plot_stem is not None, "plot_stem must be provided if plot is True."

    # Resample Gaia XP spectrum to the same wavelength as the science frame.
    log.debug("Resampling Gaia XP spectra to the same wavelength as the science frame.")
    df = _resample_gaia_xp(df, wave)

    # Retrieve sky corrections.
    log.debug("Retrieving sky correction.")
    sky_corr = get_weighted_sky_corr(hobject)

    df = df.with_columns(
        sky_corr=polars.Series(
            sky_corr[df["fiber_idx"]],
            dtype=polars.Array(polars.Float32, sky_corr.shape[1]),
        )
    )

    # Get the sky masks.
    sky_mask = get_sky_mask_uves(wave, width=3)
    sky_mask_z: numpy.ndarray | None = None
    if channel == "z":
        sky_mask_z = get_z_continuum_mask(wave)

    # Get extinction correction.
    log.debug("Getting extinction correction for LCO.")
    ext = get_extinction_correction(wave)

    # Calculate the sensitivity function.
    log.info("Calculating sensitivity function for each source.")
    sens = map(
        lambda row: _calc_sensitivity_udf(
            row,
            wave,
            ext,
            channel,
            sky_mask=sky_mask,
            sky_mask_z=sky_mask_z,
            secz=row["secz"],
            exp_time=row["exp_time"],
        ),
        df.to_dicts(),
    )

    ARRAY_TYPE = polars.Array(polars.Float32, wave.size)
    sens_df = polars.DataFrame(
        sens,
        schema={
            "flux_corrected": ARRAY_TYPE,
            "sens": ARRAY_TYPE,
            "sens_smooth": ARRAY_TYPE,
        },
    )

    df = df.with_columns(sens_df)

    if plot:
        log.info("Plotting sensitivity functions.")

        # Plot the sensitivity function for each Gaia source.
        _plot_sensitivity_functions(
            df,
            wave,
            f"{plot_stem}_sensitivity.pdf",
            title=hobject.name,
        )

        # Plot mean sensitivity function and residuals.
        _plot_sensitivity_mean(
            df,
            wave,
            f"{plot_stem}_sensitivity_mean.pdf",
            title=hobject.name,
        )

    return df


def _plot_ifu_map(
    data: ARRAY_2D_F32,
    slitmap: polars.DataFrame,
    gaia_sources: polars.DataFrame,
    plot_file_stem: str,
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
    fig_rss.savefig(f"{plot_file_stem}_map.pdf")

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

    fig_rss.savefig(f"{plot_file_stem}_map_gaia.pdf")

    seaborn.reset_defaults()

    plt.close("all")


def _plot_sensitivity_functions(
    df: polars.DataFrame,
    wave: numpy.ndarray,
    file_path: PathType,
    title: str | None = None,
):
    """Plot the sensitivity function for each Gaia source."""

    seaborn.set_theme(context="paper", style="ticks", font_scale=0.9)

    gaia = df.filter(polars.col.source_id.is_not_null())

    with plt.ioff():
        with PdfPages(file_path) as pdf:
            for nn, row in enumerate(gaia.to_dicts()):
                source_id = row["source_id"]
                fiberid = row["fiberid"]
                ra = row["ra_gaia"]
                dec = row["dec_gaia"]

                flux_corrected = numpy.array(row["flux_corrected"], dtype=numpy.float32)
                flux_xp = numpy.array(row["flux_xp_resampled"], dtype=numpy.float32)

                sens_smooth = numpy.array(row["sens_smooth"])

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

                if nn == 0:
                    ax.set_title(title)

                ax.set_ylim(min([-0.2, flux_xp.min() * 0.7]), flux_xp.max() * 1.2)

                ax.set_ylabel("Normalised flux")
                ax2.set_ylabel("Sensitivity response (XP / corrected flux)")

                ax.legend(loc="upper left")
                ax2.legend(loc="upper right")

                pdf.savefig(fig)
                plt.close(fig)

    plt.close("all")
    seaborn.reset_defaults()


def _plot_sensitivity_mean(
    df: polars.DataFrame,
    wave: numpy.ndarray,
    plot_file_path: str,
    title: str | None = None,
):
    """Plot the mean sensitivity function and residuals."""

    stds = df.filter(polars.col.sens.is_not_null())
    sens_mean, sens_rms = get_mean_sensitivity_response(df, use_smooth=True)

    seaborn.set_theme(context="paper", style="ticks", font_scale=0.8)

    with plt.ioff():
        fig, axes = plt.subplots(2, height_ratios=[4, 1], sharex=True)
        fig.subplots_adjust(hspace=0)

        ax = axes[0]

        for row in stds.iter_rows(named=True):
            ax.plot(wave, row["sens_smooth"], "0.8", lw=0.9, alpha=0.8)

        ax.plot(wave, sens_mean, "r-", linewidth=2, label="Mean sensitivity")

        ax.legend(loc="upper right")

        # Residuals
        ax_res = axes[1]

        # Get the current limits for the reference lines.
        xlim0, xlim1 = ax_res.get_xlim()

        ax_res.fill_between(
            wave,
            sens_rms / sens_mean,
            -sens_rms / sens_mean,
            color="b",
            lw=0,
            alpha=0.5,
        )

        ax_res.hlines(
            [0.05, -0.05],
            xlim0,
            xlim1,
            color="k",
            linestyle="dotted",
            lw=0.5,
        )
        ax_res.hlines(
            [0.1, -0.1],
            xlim0,
            xlim1,
            color="k",
            linestyle="dashed",
            lw=0.5,
        )

        # Labels and aesthetics
        ax.set_ylabel("Sensitivity response")

        ax_res.set_xlabel("Wavelength [A]")
        ax_res.set_ylabel("Rel. residuals")

        ax.set_title(title)

        # Remove the first y-tick label to avoid overlap with the second axis.
        y_ticks = ax.yaxis.get_major_ticks()
        y_ticks[0].label1.set_visible(False)

        # Force xlims which may have changed.
        ax_res.set_xlim(xlim0, xlim1)

        ax.set_ylim(sens_mean.min() * 0.8, sens_mean.max() * 1.1)

        fig.savefig(plot_file_path)

    plt.close("all")
    seaborn.reset_defaults()


def _resample_gaia_xp(df: polars.DataFrame, wave: numpy.ndarray) -> polars.DataFrame:
    """Resamples the Gaia XP spectra to the science frame wavelength."""

    flux_xp_resampled = (
        df["wave_xp", "flux_xp"]
        .map_rows(lambda row: _interp_gaia_xp_udf(wave, *row))
        .cast(polars.Array(polars.Float32, wave.size))
    )
    df = df.with_columns(flux_xp_resampled=flux_xp_resampled.to_series())

    return df


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
    row: dict,
    wave: numpy.ndarray,
    ext: numpy.ndarray,
    channel: str,
    sky_mask: numpy.ndarray | None = None,
    sky_mask_z: numpy.ndarray | None = None,
    secz: float = 1.0,
    exp_time: float = 900.0,
) -> dict:
    """Calculates the sensitivity function for a given source."""

    flux = numpy.array(row["flux"], dtype=numpy.float32)
    sky_corr = numpy.array(row["sky_corr"], dtype=numpy.float32)

    flux_skycorr = (flux - sky_corr) / exp_time

    if sky_mask is not None:
        flux_skycorr = interpolate_mask(
            wave,
            flux_skycorr,
            sky_mask,
            fill_value="extrapolate",
        )

    if channel == "z" and sky_mask_z is not None:
        # The z mask is a positive one, so we negate it to get the regions
        # to interpolate over.
        flux_skycorr = interpolate_mask(
            wave,
            flux_skycorr,
            ~sky_mask_z,
            fill_value="extrapolate",
        )

    flux_corrected = (flux_skycorr) * 10 ** (0.4 * ext * secz)

    if row["flux_xp_resampled"] is None:
        return {"flux_corrected": flux_corrected.tolist()}

    flux_xp_resampled = numpy.array(row["flux_xp_resampled"], dtype=numpy.float32)

    sens = flux_xp_resampled / flux_corrected

    wgood, sgood = filter_channel(wave, sens, 2)
    sens_smooth = interpolate.make_smoothing_spline(wgood, sgood, lam=1e4)

    return {
        "flux_corrected": flux_corrected.tolist(),
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


def get_mean_sensitivity_response(df: polars.DataFrame, use_smooth: bool = True):
    """Calculates the mean sensitivity response and RMS using biweight statistics."""

    col = "sens_smooth" if use_smooth else "sens"
    sens_array = df.filter(polars.col.sens.is_not_null())[col].to_numpy()

    sens_rms: numpy.ndarray = biweight_scale(sens_array, axis=0, ignore_nan=True)
    sens_mean: numpy.ndarray = biweight_location(sens_array, axis=0, ignore_nan=True)

    return sens_mean.astype(numpy.float32), sens_rms.astype(numpy.float32)
