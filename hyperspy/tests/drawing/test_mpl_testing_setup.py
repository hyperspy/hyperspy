# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib
import pytest
from distutils.version import LooseVersion


def test_mlp_agg_for_testing():
    assert matplotlib.get_backend() == 'agg'


def test_mpl_version():
    # for simplicity, only matplotlib 2.x is supported for testing
    assert LooseVersion(matplotlib.__version__) >= LooseVersion('2.0.0')
