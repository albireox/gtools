#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-02-05
# @Filename: configuration.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib

from typing import Literal, Sequence, overload

import polars
from rich.progress import track

from gtools import log


REQUIRED_COLS = [
    "observatory",
    "MJD",
    "configuration_id",
    "design_id",
    "positionerId",
    "fiberType",
]
BOOLEAN_COLS = [
    "assigned",
    "on_target",
    "disabled",
    "valid",
    "decollided",
    "too",
    "is_dithered",
]
DISABLED_REQUIRED_COLS = REQUIRED_COLS + [
    "alpha",
    "beta",
    "valid",
    "on_target",
    "assigned",
]


def read_all_configurations(
    observatories: Sequence[str] = ["APO", "LCO"],
    sdsscore: str | pathlib.Path | None = None,
    best_effort: bool = True,
    columns: list[str] | None = None,
    flavour: Literal["S", "FS"] = "FS",
) -> polars.DataFrame:
    """Reads all the configurations and returns a concatenated dataframe.

    Parameters
    ----------
    observatories
        The list of observatories to read.
    sdsscore
        The path to the root of the ``SDSSCORE`` checkout. If :obj:`None`, uses the
        ``$SDSSCORE_DIR`` environment variable.
    best_effort
        If :obj:`True`, will attempt to cast older datamodels and will ignore any error
        if that is not possible.
    columns
        The columns to read. If :obj:`None`, reads all columns.
    flavour
        The flavour of the configuration to read. Currently only ``S`` and ``FS``
        are supported.

    """

    console = log.rich_console

    if sdsscore is None:
        assert "SDSSCORE_DIR" in os.environ, "SDSSCORE_DIR not set."
        sdsscore_path = pathlib.Path(os.environ["SDSSCORE_DIR"])
    else:
        sdsscore_path = pathlib.Path(sdsscore)

    assert sdsscore_path.exists(), f"{sdsscore_path} does not exist."

    obs_data: list[polars.DataFrame] = []

    if columns is not None:
        columns = list(set(REQUIRED_COLS).union(set(columns)))

    for obs in observatories:
        summary_path = sdsscore_path / obs.lower() / "summary_files/"

        files = sorted(summary_path.glob(f"**/confSummary{flavour}*.parquet"))
        summaries: list[polars.DataFrame] = []
        for file in track(files, description=obs, transient=True, console=console):
            try:
                try:
                    summary = polars.read_parquet(file, columns=columns)
                except Exception:
                    summary = polars.read_parquet(file)
                    if columns is not None:
                        valid_cols = [col for col in summary.columns if col in columns]
                        summary = summary.select(*valid_cols)

                # Some early APO configurations have invalid design_id. Those can
                # be safely ignored.
                if not summary["design_id"].dtype.is_integer():
                    continue

                summaries.append(summary)
            except Exception:
                if not best_effort:
                    raise

        if len(summaries) == 0:
            continue

        obs_data.append(polars.concat(summaries, how="diagonal_relaxed"))

    data = polars.concat(obs_data, how="diagonal_relaxed")
    del obs_data

    data = data.select(
        *REQUIRED_COLS,
        *[col for col in data.columns if col not in REQUIRED_COLS],
    )

    # Cast boolean columns from integer.
    data = data.with_columns(
        polars.col(set(BOOLEAN_COLS).intersection(data.columns)).cast(polars.Boolean)
    )

    # Add disabled column.
    if all(col in data.columns for col in DISABLED_REQUIRED_COLS):
        disabled = (
            disabled_robots(data)
            .select(
                "observatory",
                "MJD",
                "positionerId",
            )
            .with_columns(disabled=True)
        )
        data = (
            data.drop("disabled", strict=False)
            .join(
                disabled,
                on=["observatory", "MJD", "positionerId"],
                how="left",
            )
            .with_columns(polars.col.disabled.fill_null(False))
        )

    return data.sort(REQUIRED_COLS)


@overload
def disabled_robots(
    data: pathlib.Path | str | polars.DataFrame,
    return_count: Literal[False] = False,
) -> polars.DataFrame: ...


@overload
def disabled_robots(
    data: pathlib.Path | str | polars.DataFrame,
    return_count: Literal[True] = True,
) -> tuple[polars.DataFrame, polars.DataFrame]: ...


def disabled_robots(
    data: pathlib.Path | str | polars.DataFrame,
    return_count: bool = False,
) -> polars.DataFrame | tuple[polars.DataFrame, polars.DataFrame]:
    """Returns a data frame of disabled robots per MJD and observatory.

    Parameters
    ----------
    data
        A concatenated data frame with all the confSummary file information or
        the path to the file.
    return_count
        If :obj:`True`, returns a tuple with the data frame of disabled robots
        per MJD and observatory, and a data frame with the number of MJDs a robot
        was disabled.

    """

    if not isinstance(data, polars.DataFrame):
        data = polars.read_parquet(data)

    # Check that the required columns are present.
    if not all(col in data.columns for col in DISABLED_REQUIRED_COLS):
        raise ValueError(
            "The input data frame is missing required columns. "
            f"Required columns are: {', '.join(DISABLED_REQUIRED_COLS)}."
        )

    # Cast to proper boolean type.
    d1 = data.cast({"assigned": polars.Boolean, "on_target": polars.Boolean})

    pcol = polars.col
    over_c = ["MJD", "positionerId", "observatory"]
    hypot = (pcol.alpha.std().pow(2) + pcol.beta.std().pow(2)).sqrt()

    # Check if the input data frame has a disabled column. In that case get the values
    # for which the column is not null (that column was only added on 60717). For
    # those we just use the information in the column.
    disabled_from_col: polars.DataFrame | None = None
    if "disabled" in d1.columns:
        disabled_from_col = d1.filter(polars.col.disabled.is_not_null())
        disabled_from_col = disabled_from_col.with_columns(
            disabled=polars.col.disabled.all().over(over_c),
            alpha_median=pcol.alpha.median().over(over_c),
            beta_median=pcol.beta.median().over(over_c),
        )

        d1 = d1.filter(polars.col.disabled.is_null())

    # Determine whether a robot was disabled for a given MJD and observatory.
    # We use two metrics: either the robot was never on target for that MJD (even if
    # it had assigned targets) or the standard deviation of the positioner coordinates
    # is below 0.5 for the entire night, which means that it didn't move.
    d2 = (
        d1.select(
            "MJD",
            "positionerId",
            "observatory",
            "valid",
            "on_target",
            "assigned",
            "alpha",
            "beta",
        )
        .filter(pcol.assigned)
        .with_columns(
            alpha_median=pcol.alpha.median().over(over_c),
            beta_median=pcol.beta.median().over(over_c),
            none_on_target=(pcol.on_target.not_().all()).over(over_c),
            positioner_std=(hypot).over(over_c),
        )
        .with_columns(disabled=(pcol.none_on_target | (pcol.positioner_std < 5)))
    )

    cols = [
        "MJD",
        "positionerId",
        "observatory",
        "valid",
        "on_target",
        "assigned",
        "alpha",
        "beta",
        "disabled",
        "alpha_median",
        "beta_median",
    ]
    d2 = d2.select(*cols)

    # Join back with the data frame that has the disabled column.
    if disabled_from_col is not None and disabled_from_col.height > 0:
        d2 = polars.concat([d2, disabled_from_col.select(*cols)], how="vertical")

    # Create a list of disabled robots per MJD and observatory.
    d3 = d2.filter(pcol.disabled).select("observatory", "MJD", "positionerId").unique()

    # Include the median alpha and beta values for each disabled robot, each MJD.
    d4 = d3.join(
        d2.select(
            [
                "MJD",
                "observatory",
                "positionerId",
                "alpha_median",
                "beta_median",
            ]
        ).unique(["MJD", "observatory", "positionerId"]),
        on=["observatory", "MJD", "positionerId"],
        how="inner",
    )

    d4 = d4.sort("observatory", "MJD", "positionerId")

    # Include the median alpha and beta values for each disabled robot, each MJD.

    if return_count:
        count = d3.group_by("positionerId").count().sort("positionerId")
        return d4, count

    return d4
