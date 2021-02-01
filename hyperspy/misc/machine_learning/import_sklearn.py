# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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

"""
Import sklearn.* and randomized_svd from scikit-learn
"""

import warnings
import importlib
from lazyasd import lazyobject

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        @lazyobject
        def sklearn():
            return importlib.import_module('sklearn')

        @lazyobject
        def decomposition():
            return importlib.import_module('sklearn.decomposition')

        @lazyobject
        def cluster():
            return importlib.import_module('sklearn.cluster')

        @lazyobject
        def preprocessing():
            return importlib.import_module('sklearn.preprocessing')

        @lazyobject
        def metrics():
            return importlib.import_module('sklearn.metrics')

        @lazyobject
        def extmath():
            return importlib.import_module('sklearn.utils.extmath')

        sklearn_installed = True

except ImportError:  # pragma: no cover
    randomized_svd = None
    sklearn_installed = False
