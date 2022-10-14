#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-06-26
# @Filename: __main__.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import click


@click.group()
def gtools():
    """gallegoj tools."""

    pass


@gtools.command(name="boss-exposures")
@click.argument("PATH", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def boss_exposures(path: str):
    """Lists BOSS exposures."""

    from gtools.boss.tools import list_exposures

    list_exposures(str(path)).pprint_all()


if __name__ == "__main__":
    gtools()
