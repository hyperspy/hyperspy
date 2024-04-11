# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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
Import sklearn.* and randomized_svd from scikit-learn
"""

import importlib
import warnings

sklearn_spec = importlib.util.find_spec("sklearn")

if sklearn_spec is None:  # pragma: no cover
    randomized_svd = None
    sklearn_installed = False
else:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import sklearn  # noqa: F401
        import sklearn.cluster  # noqa: F401
        import sklearn.decomposition  # noqa: F401
        import sklearn.metrics  # noqa: F401
        import sklearn.preprocessing  # noqa: F401
        from sklearn.utils.extmath import randomized_svd  # noqa: F401

        sklearn_installed = True
