#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-09-04
# @Filename: datamodel.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib

from typing import Sequence

import pandas
from astropy.io import fits


AnyPath = os.PathLike | str


def create_csv_from_file(
    file: AnyPath,
    outpath: AnyPath | None = None,
    extra_columns: Sequence[str] = ["Notes"],
):
    """Creates a CSV stub for a data model from a file."""

    file = pathlib.Path(file)

    outpath = pathlib.Path(outpath) if outpath else file.with_suffix(".csv")
    root = outpath.parent / outpath.stem

    if str(file).endswith(".fits") or str(file).endswith(".fits.gz"):
        for iext, extension in enumerate(hdul := fits.open(str(file))):
            name = extension.name if extension.name else iext

            header_kws = []
            for card in extension.header.cards:
                value = "" if isinstance(card.value, fits.Undefined) else card.value
                header_kws.append((card.keyword, value, card.comment))

            df = pandas.DataFrame(
                header_kws,
                columns=["Keyword", "Sample value", "Comment"],
            )
            for col in extra_columns:
                df[col] = ""

            df.to_csv(str(root) + f"_{name}.csv", index=False)

    elif str(file).endswith(".parquet"):
        df = pandas.read_parquet(file)

        df_cols = pandas.DataFrame(
            [df.columns, df.dtypes],
            index=["Column", "Data type"],
        ).T

        for col in extra_columns:
            df_cols[col] = ""

        df_cols.to_csv(outpath, index=False)
