# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

"""SAMFire modules


The :mod:`~hyperspy.api.samfire` module contains the following submodules:

fit_tests
    Tests to check fit convergence when running SAMFire

global_strategies
    Available global strategies to use in SAMFire

local_strategies
    Available global strategies to use in SAMFire

SamfirePool
    The parallel pool, customized to run SAMFire.

"""

from hyperspy.samfire_utils import (
    fit_tests,
    global_strategies,
    local_strategies
    )


__all__ = [
    'fit_tests',
    'global_strategies',
    'local_strategies',
    ]


def __dir__():
    return sorted(__all__)
