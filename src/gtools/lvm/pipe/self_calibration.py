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

import peewee
import polars
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib import pyplot as plt

from sdsstools.logger import get_logger

from gtools.lvm.pipe.tools import (
    get_gaiaxp_cone,
    slitmap_radec_to_xy,
    slitmap_to_polars,
)
from gtools.lvm.plotting import plot_rss


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

    slitmap = slitmap_to_polars(hdul["SLITMAP"].data)

    bore_ra = hdul[0].header["POSCIRA"]
    bore_dec = hdul[0].header["POSCIDE"]

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

    if plot:
        fig_rss, ax_rss = plot_rss(
            hdul["PRIMARY"].data,
            slitmap,
            fibre_type="science",
            mode="xy",
            patch="hex",
            cmap="mako_r",
            interactive=False,
        )
        fig_rss.savefig(plot_dir / f"{hobject.stem}_map.pdf")

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

    # Create a dataframe of fibre fluxes.
    df = slitmap_sci.clone()
    flux_ordered = hdul["PRIMARY"].data[df["fiberid"] - 1, :]

    df = df.with_columns(
        flux=polars.Series(
            flux_ordered.tolist(),
            dtype=polars.Array(polars.Float32, flux_ordered.shape[1]),
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

    print(df)

    plt.close("all")
