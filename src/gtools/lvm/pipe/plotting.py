#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-03-12
# @Filename: plotting.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import Any, Literal

import numpy
import numpy.typing as npt
import polars
from astropy.visualization import (
    ImageNormalize,
    LinearStretch,
    LogStretch,
    MinMaxInterval,
    SqrtStretch,
    ZScaleInterval,
)
from matplotlib import pyplot as plt

from gtools.lvm.pipe.tools import read_fibermap


def plot_ifu_data(
    data: npt.NDArray[numpy.float32],
    wrange: tuple[int, int] | None = None,
    stretch: Literal["linear", "sqrt", "log"] = "linear",
    interval: Literal["zscale", "minmax"] = "minmax",
    cmap: str = "cividis",
    filename: str | pathlib.Path | None = None,
):
    """Plots an IFU data array.

    Parameters
    ----------
    data
        The data array to plot. Usually the output of a quick extraction.
    wrange
        The wavelength range to plot as a tuple of wavelength indices ``[wmin, wmax]``
        (the second dimension in the ``data`` array). If not provided, the full
        wavelength range is plotted.
    stretch
        The stretch to use for the normalization.
    interval
        The interval to use for the normalization.
    cmap
        The colormap to use.
    filename
        The filename to save the plot to. If not provided, the plot is shown.

    """

    if filename is not None:
        plt.ioff()
    else:
        plt.ion()

    fibermap = read_fibermap().sort("spectrographid", "fiberid")

    assert data.shape[0] == len(fibermap), "Data shape does not match fibermap length."

    if wrange is not None:
        data = data[:, wrange[0] : wrange[1]].sum(axis=1)
    else:
        data = data.sum(axis=1)

    interval_obj: Any
    if interval == "zscale":
        interval_obj = ZScaleInterval()
    elif interval == "minmax":
        interval_obj = MinMaxInterval()
    else:
        raise ValueError("invalid interval")

    stretch_obj: Any
    if stretch == "linear":
        stretch_obj = LinearStretch()
    elif stretch == "sqrt":
        stretch_obj = SqrtStretch()
    elif stretch == "log":
        stretch_obj = LogStretch()
    else:
        raise ValueError("invalid stretch")

    norm: Any = ImageNormalize(data, interval=interval_obj, stretch=stretch_obj)

    fig, ax = plt.subplots(figsize=(12, 12))

    for ifib in range(data.shape[0]):
        fibermap_row = fibermap.filter(polars.col.fiberid == ifib + 1)

        xpmm = fibermap_row[0, "xpmm"]
        ypmm = fibermap_row[0, "ypmm"]

        targettype = fibermap_row[0, "targettype"]
        if targettype != "science":
            continue

        ax.scatter(
            xpmm,
            ypmm,
            s=320,
            marker=(6, 0, 90),  # hexagon rotated 90 degrees   # type: ignore
            c=data[ifib],
            edgecolors="None",
            norm=norm,
            cmap=cmap,
        )

    ax.set_aspect("equal")

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    fig.tight_layout()

    if filename:
        fig.savefig(filename)
    else:
        plt.show(block=True)

    plt.close("all")

    return fig, ax
