#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-03-11
# @Filename: tools.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib

from typing import Literal

import numpy
import polars
import yaml
from astropy.io import fits


__all__ = [
    "slice_to_2d_slice",
    "mtrace_to_dataframe",
    "mwidth_to_dataframe",
    "read_fibermap",
]


def slice_to_2d_slice(
    data: str | tuple[int, int, int, int],
    order: Literal["C", "R"] = "C",
    index: int = 0,
) -> tuple[slice[int, int, int | None], slice[int, int, int | None]]:
    """Returns a Numpy 2D slice."""

    if isinstance(data, str):
        parts = data.strip("[]").split(",")
        s00, s01 = parts[0].split(":")
        s10, s11 = parts[1].split(":")
        slice_ = (
            slice(int(s00) - index, int(s01) - index + 1),
            slice(int(s10) - index, int(s11) - index + 1),
        )
    else:
        slice_ = (
            slice(data[0] - index, data[1] - index + 1),
            slice(data[2] - index, data[3] - index + 1),
        )

    if order == "C":
        slice_ = (slice_[1], slice_[0])

    return slice_


def mtrace_to_dataframe(file_: pathlib.Path | str, ccd: str):
    """Converts an ``lvm-mtrace`` file to a Polars data frame."""

    coeffs = fits.getdata(file_, ext=3)

    nfibre = coeffs.shape[0]

    df = polars.DataFrame(
        {
            "nfibre": polars.Series("nfibre", range(1, nfibre + 1), polars.Int32),
            "ccd": ccd,
            **{
                f"coeff{i}": polars.Series(
                    f"coeff{i}",
                    coeffs[:, i].tolist(),
                    polars.Float32,
                )
                for i in range(coeffs.shape[1])
            },
        }
    )

    return df


def mwidth_to_dataframe(file_: pathlib.Path | str, ccd: str):
    """Converts an ``lvm-mwidth`` file to a Polars data frame."""

    return mtrace_to_dataframe(file_, ccd)


def read_fibermap(path: str | pathlib.Path | None = None) -> polars.DataFrame:
    """Reads the fibermap file.

    Parameters
    ----------
    path
        Path to the fibermap file. If :obj:`None` uses the path from the
        configuration file.

    Returns
    -------
    fibermap
        A Polars DataFrame with the fibermap data.

    """

    if path is None:
        path = (
            pathlib.Path(os.environ["LVMCORE_DIR"])
            / "metrology"
            / "lvm_fiducial_fibermap.yaml"
        )

    else:
        path = pathlib.Path(path).absolute()

    assert isinstance(path, pathlib.Path)

    if not path.exists():
        raise FileNotFoundError(f"Fibermap file {str(path)} does not exist.")

    fibermap_y = yaml.load(open(str(path)), Loader=yaml.CFullLoader)

    schema = fibermap_y["schema"]
    cols = [it["name"] for it in schema]
    dtypes = [it["dtype"] if it["dtype"] != "str" else "<U8" for it in schema]

    fibers = polars.from_numpy(
        numpy.array(
            [tuple(fibs) for fibs in fibermap_y["fibers"]],
            dtype=list(zip(cols, dtypes)),
        ),
    )

    # Lower-case some columns.
    fibers = fibers.with_columns(
        polars.col("targettype", "telescope").str.to_lowercase(),
        fibername=polars.col.orig_ifulabel,
    )

    return fibers
