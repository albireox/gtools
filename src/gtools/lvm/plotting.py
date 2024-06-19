#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-06-15
# @Filename: plotting.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import nptyping as npt
import numpy
import polars
import seaborn
from astropy.visualization import ImageNormalize, LogStretch, MinMaxInterval
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Patch, RegularPolygon


DATA_ARRAY = npt.NDArray[npt.Shape["*,*"], npt.Float]
ARRAY_1D = npt.NDArray[npt.Shape["*"], npt.Float]


def plot_rss(
    data: DATA_ARRAY,
    slitmap: polars.DataFrame,
    func: Literal["sum", "mean"] = "sum",
    fibre_type: Literal["science", "sky", "standard"] = "science",
    mode: Literal["xy", "radec"] = "xy",
    patch: Literal["circle", "hex"] = "hex",
    interactive: bool = False,
    cmap: str = "mako_r",
    position_angle: float = 0.0,
    wavelength_array: ARRAY_1D | None = None,
    wavelength_range: tuple[float, float] | None = None,
):
    """Plots an RSS file as a two dimensional image.

    TODO: this code needs changes to plot the data correctly when dec != 0 in
          radec mode.

    Parameters
    ----------
    data
        The RSS data array to plot.
    slitmap
        The slitmap table.
    func
        The function to apply to the data. Either ``'sum'`` or ``'mean'``.
    fibre_type
        The fibre type to plot.
    mode
        Plot in ``'xy'`` or ``'radec'`` coordinates.
    patch
        The type of patch to use. Either ``'circle'`` or ``'hex'``.
    interactive
        Whether to plot interactively.
    cmap
        The colormap to use.
    position_angle
        The position angle of the IFU.
    wavelength_array
        The array of wavelengths for the data. Must match the length of the second
        dimension of ``data``.
    wavelength_range
        The range of wavelengths to plot. If not provided, the full range is used.

    Returns
    -------
    fig, ax
        The matplotlib figure and axis.

    """

    slitmap = slitmap.with_columns(polars.col.targettype.str.to_lowercase())
    slitmap_type = slitmap.filter(polars.col.targettype == fibre_type)

    if patch == "circle":
        patch_radius = 0.165 / 0.99  # 0.99 is a fudge factor
    elif patch == "hex":
        patch_radius = 0.286 / numpy.sqrt(2) * 0.965  # 0.965 is a fudge factor
    else:
        raise ValueError("Invalid patch type")

    if mode == "radec":
        # From the AGcam, 1 arcsec = 9 microns. Here we convert to deg/mm
        patch_radius *= 1.0 / 9.0 * 1000 / 3600
    elif mode != "xy":
        raise ValueError("Invalid mode")

    data_type = data[slitmap_type["fiberid"] - 1]

    if wavelength_range is not None:
        if wavelength_array is None:
            raise ValueError("wavelength_array is required with wavelength_range.")
        data_type = data_type[
            :,
            (wavelength_array >= wavelength_range[0])
            & (wavelength_array <= wavelength_range[1]),
        ]

    data_flat = (
        numpy.nansum(data_type, axis=1)
        if func == "sum"
        else numpy.nanmean(data_type, axis=1)
    )

    slitmap_type = slitmap_type.with_columns(data_flat=data_flat)

    seaborn.set_theme(context="notebook", style="white")

    interactive_mode = plt.ion if interactive else plt.ioff

    with interactive_mode():
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect("equal")

        patches: list[Patch] = []

        for row in slitmap_type.sort("ypmm").to_dicts():
            xval = row["xpmm"] if mode == "xy" else row["ra"]
            yval = row["ypmm"] if mode == "xy" else row["dec"]

            if patch == "circle":
                patches.append(
                    Circle(
                        (xval, yval),
                        patch_radius,
                        edgecolor="none",
                        linewidth=0,
                    )
                )
            elif patch == "hex":
                patches.append(
                    RegularPolygon(
                        (xval, yval),
                        numVertices=6,
                        radius=patch_radius,
                        orientation=(90 - position_angle) / 180 * numpy.pi,
                        edgecolor="none",
                        linewidth=0,
                    )
                )

        data_array = slitmap_type["data_flat"]

        norm = ImageNormalize(interval=MinMaxInterval(), stretch=LogStretch())  # type: ignore

        collection = PatchCollection(
            patches,
            cmap=seaborn.color_palette(cmap, as_cmap=True),
            norm=norm,
            edgecolor="none",
            linewidth=0,
        )

        collection.set_array(data_array)
        ax.add_collection(collection)

        cbar = fig.colorbar(collection, ax=ax)

        ax.autoscale_view()

        if mode == "radec":
            ax.set_xlabel("RA [deg]")
            ax.set_ylabel("Dec [deg]")
        else:
            ax.set_xlabel("X [mm]")
            ax.set_ylabel("Y [mm]")

        cbar.set_label("Flux [e-]")

        fig.tight_layout()
        seaborn.despine()

    seaborn.reset_defaults()

    return fig, ax
