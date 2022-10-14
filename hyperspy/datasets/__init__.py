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

"""
The :mod:`hyperspy.datasets` module includes access to local and remote
datasets.

Functions:

    eelsdb
        Download spectra from the EELS data base https://eelsdb.eu

Submodules:

The :mod:`hyperspy.datasets` module contains the following submodules:

    :mod:`hyperspy.datasets.artificial_data`
        Artificial datasets generated with HyperSpy.

    :mod:`hyperspy.datasets.example_signals`
        Example datasets distributed with HyperSpy.

"""

from hyperspy.misc.eels.eelsdb import eelsdb
from hyperspy.datasets import artificial_data, example_signals


__all__ = [
    'artificial_data',
    'eelsdb',
    'example_signals',
    ]


def __dir__():
    return sorted(__all__)
