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

from typing import Literal, Sequence

import polars


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

    if sdsscore is None:
        assert "SDSSCORE_DIR" in os.environ, "SDSSCORE_DIR not set."
        sdsscore_path = pathlib.Path(os.environ["SDSSCORE_DIR"])
    else:
        sdsscore_path = pathlib.Path(sdsscore)

    assert sdsscore_path.exists(), f"{sdsscore_path} does not exist."

    obs_data: list[polars.DataFrame] = []

    required_cols = ["configuration_id", "positionerId", "observatory"]
    if columns is not None:
        columns = list(set(required_cols).union(set(columns)))

    for obs in observatories:
        summary_path = sdsscore_path / obs.lower() / "summary_files/"

        files = sorted(summary_path.glob(f"**/confSummary{flavour}*.parquet"))[::-1]
        summaries: list[polars.DataFrame] = []
        for file in files:
            try:
                summary = polars.read_parquet(file, columns=columns)
                summaries.append(summary)
            except Exception:
                if not best_effort:
                    raise

        last_schema = summaries[0].schema
        cast_summaries: list[polars.DataFrame] = []
        for summary in summaries:
            try:
                cast_summary = summary.cast(last_schema)  # type: ignore
                cast_summaries.append(cast_summary)
            except Exception:
                if not best_effort:
                    raise

        if len(cast_summaries) == 0:
            continue

        obs_data.append(polars.concat(cast_summaries))

    data = polars.concat(obs_data)
    data = data.select(
        *required_cols,
        *[col for col in data.columns if col not in required_cols],
    )

    return data.sort(["observatory", "configuration_id", "positionerId"])
