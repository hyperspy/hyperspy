# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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

import os
import nose.tools as nt
import numpy as np

def remove_file(filename):
    try:
        if os.path.exists(filename):
            os.remove(filename)
    except WindowsError:
        pass    # If we don't do this, we might mask real exceptions


def assert_deep_almost_equal(actual, expected, *args, **kwargs):
    """ Assert that two complex structures have almost equal contents.

    Compares lists, dicts and tuples recursively. Checks numeric values
    using :py:meth:`nose.tools.assert_almost_equal` and
    checks all other values with :py:meth:`nose.tools.assert_equal`.
    Accepts additional positional and keyword arguments and pass those
    intact to assert_almost_equal() (that's how you specify comparison
    precision).

    Parameters
    ----------
    actual: lists, dicts or tuples
    
    expected: lists, dicts or tuples
    """
    is_root = not '__trace' in kwargs
    trace = kwargs.pop('__trace', 'ROOT')
    try:
        if isinstance(expected, (int, float, complex)):
            nt.assert_almost_equal(expected, actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, np.ndarray)):
            nt.assert_equal(len(expected), len(actual))
            for index in range(len(expected)):
                v1, v2 = expected[index], actual[index]
                assert_deep_almost_equal(v1, v2,
                                         __trace=repr(index), *args, **kwargs)
        elif isinstance(expected, dict):
            nt.assert_equal(set(expected), set(actual))
            for key in expected:
                assert_deep_almost_equal(expected[key], actual[key],
                                         __trace=repr(key), *args, **kwargs)
        else:
            nt.assert_equal(expected, actual)
    except AssertionError as exc:
        exc.__dict__.setdefault('traces', []).append(trace)
        if is_root:
            trace = ' -> '.join(reversed(exc.traces))
            exc = AssertionError("%s\nTRACE: %s" % (exc.message, trace))
        raise exc
        