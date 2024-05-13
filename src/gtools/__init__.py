#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-06-26
# @Filename: __init__.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from sdsstools import get_logger, get_package_version


NAME = "gtools"

log = get_logger(NAME, use_rich_handler=True)

__version__ = get_package_version(path=__file__, package_name=NAME)
