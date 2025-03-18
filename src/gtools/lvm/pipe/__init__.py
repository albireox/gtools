#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-03-11
# @Filename: __init__.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from sdsstools.logger import get_logger


log = get_logger("gtools.lvm.pipe", use_rich_handler=True)


from .detrend import *
from .extraction import *
from .plotting import *
from .tools import *
