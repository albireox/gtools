#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-03-12
# @Filename: plotting.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import Any

import numpy
import numpy.typing as npt
import polars
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt

from gtools.lvm.pipe.tools import read_fibermap


def plot_ifu_data(data: npt.NDArray[numpy.float32]):
    """Plots an IFU data array."""

    fibermap = read_fibermap().sort("spectrographid", "fiberid")

    assert data.shape[0] == len(fibermap), "Data shape does not match fibermap length."

    data = data[:, 5000:6000].sum(axis=1)
    data /= data.max()

    norm: Any = simple_norm(data, "sqrt")

    fig, ax = plt.subplots()

    for ifib in range(data.shape[0]):
        fibermap_row = fibermap.filter(polars.col.fiberid == ifib + 1)
        xpmm = fibermap_row[0, "xpmm"]
        ypmm = fibermap_row[0, "ypmm"]
        targettype = fibermap_row[0, "targettype"]
        if targettype != "science":
            continue

        fib_data = data[ifib]

        # fib_patch = RegularPolygon(
        #     (xpmm - 0.5, ypmm - 0.5),
        #     numVertices=6,
        #     radius=0.5,
        #     c=fib_data / max_sum,
        # )

        # ax.add_patch(fib_patch)
        ax.scatter(
            xpmm,
            ypmm,
            s=85,
            c=fib_data,
            edgecolors="None",
            norm=norm,
        )

    ax.set_aspect("equal")
    fig.show()
