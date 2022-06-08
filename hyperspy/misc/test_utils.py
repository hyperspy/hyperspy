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

import os
from contextlib import contextmanager
import warnings

import numpy as np


@contextmanager
def ignore_warning(message="", category=None):
    with warnings.catch_warnings():
        if category:
            warnings.filterwarnings("ignore", message, category=category)
        else:
            warnings.filterwarnings("ignore", message)
        yield


def check_closing_plot(s, check_data_changed_close=True):
    # When using the interactive function with the pointer, some events can't
    # be closed. Fix it once the ROI has been implemented for the pointer.
    assert s._plot.signal_plot is None
    assert s._plot.navigator_plot is None
    # Ideally we should check all events
    assert len(s.axes_manager.events.indices_changed.connected) == 0
    if check_data_changed_close:
        assert len(s.events.data_changed.connected) == 0


def update_close_figure(check_data_changed_close=True):
    def decorator2(function):
        def wrapper():
            signal = function()
            p = signal._plot
            p.close()
            check_closing_plot(signal, check_data_changed_close)

        return wrapper

    return decorator2


# Adapted from:
# https://github.com/gem/oq-engine/blob/master/openquake/server/tests/helpers.py
def assert_deep_almost_equal(actual, expected, *args, **kwargs):
    """Assert that two complex structures have almost equal contents.
    Compares lists, dicts and tuples recursively. Checks numeric values
    using :py:func:`numpy.testing.assert_allclose` and
    checks all other values with :py:func:`numpy.testing.assert_equal`.
    Accepts additional positional and keyword arguments and pass those
    intact to assert_allclose() (that's how you specify comparison
    precision).

    Parameters
    ----------
    actual: list, dict or tuple
        Actual values to compare.
    expected: list, dict or tuple
        Expected values.
    *args :
        Arguments are passed to :py:func:`numpy.testing.assert_allclose` or
        :py:func:`assert_deep_almost_equal`.
    **kwargs :
        Keyword arguments are passed to
        :py:func:`numpy.testing.assert_allclose` or
        :py:func:`assert_deep_almost_equal`.
    """
    is_root = not "__trace" in kwargs
    trace = kwargs.pop("__trace", "ROOT")
    try:
        if isinstance(expected, (int, float, complex)):
            np.testing.assert_allclose(expected, actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, np.ndarray)):
            assert len(expected) == len(actual)
            for index in range(len(expected)):
                v1, v2 = expected[index], actual[index]
                assert_deep_almost_equal(v1, v2, __trace=repr(index), *args, **kwargs)
        elif isinstance(expected, dict):
            assert set(expected) == set(actual)
            for key in expected:
                assert_deep_almost_equal(
                    expected[key], actual[key], __trace=repr(key), *args, **kwargs
                )
        else:
            assert expected == actual
    except AssertionError as exc:
        exc.__dict__.setdefault("traces", []).append(trace)
        if is_root:
            trace = " -> ".join(reversed(exc.traces))
            exc = AssertionError("%s\nTRACE: %s" % (exc, trace))
        raise exc


def sanitize_dict(dictionary):
    new_dictionary = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            new_dictionary[key] = sanitize_dict(value)
        elif value is not None:
            new_dictionary[key] = value
    return new_dictionary


def check_running_tests_in_CI():
    if "CI" in os.environ:
        return os.environ.get("CI")
