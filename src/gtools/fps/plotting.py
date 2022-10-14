#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-06-30
# @Filename: plotting.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import matplotlib
import pandas
import seaborn
from coordio import calibration
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Wedge


seaborn.set_palette("deep")


HOLE_DIAMETER = 18.0  # mm


def plot_wok(
    observatory: str,
    ax: Axes | None = None,
    highlight: list[str] = [],
    show_type: bool = False,
    show_legend: bool | None = None,
    show_coordinates: bool = False,
) -> Axes:
    """Plots a representation of the wok."""

    wok_coords = calibration.wokCoords
    wok_coords = wok_coords.loc[observatory]

    assert isinstance(wok_coords, pandas.DataFrame)
    assert len(wok_coords) > 0, "No holes founds."

    if not ax:
        # with seaborn.axes_style("white"):
        _, ax = plt.subplots()

    for _, data in wok_coords.reset_index().iterrows():
        hole_id = data.holeID
        hole_type = data.holeType
        xwok = data.xWok
        ywok = data.yWok

        if show_type is False or hole_type == "Fiducial":
            patch = Circle(
                (xwok, ywok),
                HOLE_DIAMETER / 2,
                fill=False,
                edgecolor="k",
                linewidth=1,
            )
            ax.add_patch(patch)
        else:
            if hole_type == "ApogeeBoss":
                wedge_colour1 = "r"
                wedge_colour2 = "b"
            else:
                wedge_colour1 = wedge_colour2 = "b"

            wedge1 = Wedge(
                (xwok, ywok),
                HOLE_DIAMETER / 2,
                90,
                270,
                width=0,
                fill=False,
                edgecolor=wedge_colour1,
                linewidth=1,
            )
            wedge2 = Wedge(
                (xwok, ywok),
                HOLE_DIAMETER / 2,
                270,
                90,
                width=0,
                fill=False,
                edgecolor=wedge_colour2,
                linewidth=1,
            )

            ax.add_patch(wedge1)
            ax.add_patch(wedge2)

        if show_coordinates:
            with matplotlib.rc_context(rc={"text.usetex": False}):
                ax.text(xwok, ywok - 0.75, str(hole_id), fontsize=2.5, ha="center")

        if highlight and hole_id.upper() in highlight:
            patch = Circle(
                (xwok, ywok),
                HOLE_DIAMETER / 2 + 1.5,
                fill=False,
                edgecolor="y",
                linewidth=1.2,
            )
            ax.add_patch(patch)

    seaborn.despine(ax=ax)

    buffer = HOLE_DIAMETER * 2
    ax.set_xlim(wok_coords.xWok.min() - buffer, wok_coords.xWok.max() + buffer)
    ax.set_ylim(wok_coords.yWok.min() - buffer, wok_coords.yWok.max() + buffer)

    ax.set_aspect(1)

    ax.set_xlabel(r"$x_{\rm wok}\ {\rm [mm]}$")
    ax.set_ylabel(r"$y_{\rm wok}\ {\rm [mm]}$")

    return ax
