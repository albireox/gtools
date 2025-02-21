#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-10-26
# @Filename: plotting.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.visualization
import numpy.typing as npt

from gtools.types import INTERVAL_T, STRETCH_T


if TYPE_CHECKING:
    from matplotlib.axes import Axes


__all__ = ["plot_with_stretch"]


def plot_with_stretch(
    data: npt.ArrayLike,
    interval: INTERVAL_T = "zscale",
    stretch: STRETCH_T = "linear",
    ax: Axes | None = None,
):
    """Plots normalised data using a wrapper around astropy's ``imshow_norm``.

    Parameters
    ----------
    data
        The 2D data array to plot.
    interval
        The interval to plot. Can be a tuple with zmin and zmax values,
        ``'zscale'``, or ``'minmax'``.
    stretch
        The stretch to use, either an Astropy ``BaseStretch`` instance,
        ``'linear'``, or ``'log'``.
    ax
        The matplotlib axes on which to display the image. If not provided,
        creates new axes.

    Returns
    -------
    imshow_norm_return
        The output of ``imshow_norm``, a tuple with an ``AxesImage`` as the
        first element and ``ImageNormalize`` as the second.

    """

    if isinstance(stretch, str):
        if stretch == "linear":
            stretch = astropy.visualization.LinearStretch()
        elif stretch == "log":
            stretch = astropy.visualization.LogStretch()
        else:
            raise ValueError(f"Invalid stretch {stretch!r}.")

    if isinstance(interval, str):
        if interval == "minmax":
            interval = astropy.visualization.MinMaxInterval().get_limits(data)
        elif interval == "zscale":
            interval = astropy.visualization.ZScaleInterval().get_limits(data)
        else:
            raise ValueError(f"Invalid interval {interval!r}.")

    return astropy.visualization.imshow_norm(
        data,
        ax=ax,
        vmin=interval[0],
        vmax=interval[1],
        stretch=stretch,
    )
