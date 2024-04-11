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

import dask.array as da
import numpy as np
import pytest

from hyperspy import signals
from hyperspy.misc.utils import (
    closest_power_of_two,
    fsdict,
    get_array_module,
    is_cupy_array,
    is_hyperspy_signal,
    parse_quantity,
    shorten_name,
    slugify,
    str2num,
    strlist2enumeration,
    swapelem,
    to_numpy,
)

try:
    import cupy as cp

    CUPY_INSTALLED = True
except ImportError:
    CUPY_INSTALLED = False

skip_cupy = pytest.mark.skipif(not CUPY_INSTALLED, reason="cupy is required")


def test_slugify():
    assert slugify("a") == "a"
    assert slugify("1a") == "1a"
    assert slugify("1") == "1"
    assert slugify("a a") == "a_a"
    assert slugify(42) == "42"
    assert slugify(3.14159) == "314159"
    assert slugify("├── Node1") == "Node1"

    assert slugify("a", valid_variable_name=True) == "a"
    assert slugify("1a", valid_variable_name=True) == "Number_1a"
    assert slugify("1", valid_variable_name=True) == "Number_1"

    assert slugify("a", valid_variable_name=False) == "a"
    assert slugify("1a", valid_variable_name=False) == "1a"
    assert slugify("1", valid_variable_name=False) == "1"


def test_parse_quantity():
    # From the metadata specification, the quantity is defined as
    # "name (units)" without backets in the name of the quantity
    assert parse_quantity("a (b)") == ("a", "b")
    assert parse_quantity("a (b/(c))") == ("a", "b/(c)")
    assert parse_quantity("a (c) (b/(c))") == ("a (c)", "b/(c)")
    assert parse_quantity("a [b]") == ("a [b]", "")
    assert parse_quantity("a [b]", opening="[", closing="]") == ("a", "b")


def test_is_hyperspy_signal():
    s = signals.Signal1D(np.zeros((5, 5, 5)))
    p = object()
    assert is_hyperspy_signal(s) is True
    assert is_hyperspy_signal(p) is False


def test_strlist2enumeration():
    assert strlist2enumeration([]) == ""
    assert strlist2enumeration("a") == "a"
    assert strlist2enumeration(["a"]) == "a"
    assert strlist2enumeration(["a", "b"]) == "a and b"
    assert strlist2enumeration(["a", "b", "c"]) == "a, b and c"


def test_str2num():
    assert (
        str2num("2.17\t 3.14\t 42\n 1\t 2\t 3")
        == np.array([[2.17, 3.14, 42.0], [1.0, 2.0, 3.0]])
    ).all()


def test_swapelem():
    L = ["a", "b", "c"]
    swapelem(L, 1, 2)
    assert L == ["a", "c", "b"]


def test_fsdict():
    parrot = {}
    fsdict(
        ["This", "is", "a", "dead", "parrot"], "It has gone to meet its maker", parrot
    )
    fsdict(["This", "parrot", "is", "no", "more"], "It is an ex parrot", parrot)
    fsdict(
        ["This", "parrot", "has", "seized", "to", "be"],
        "It is pushing up the daisies",
        parrot,
    )
    fsdict([""], "I recognize a dead parrot when I see one", parrot)
    assert (
        parrot["This"]["is"]["a"]["dead"]["parrot"] == "It has gone to meet its maker"
    )
    assert parrot["This"]["parrot"]["is"]["no"]["more"] == "It is an ex parrot"
    assert (
        parrot["This"]["parrot"]["has"]["seized"]["to"]["be"]
        == "It is pushing up the daisies"
    )
    assert parrot[""] == "I recognize a dead parrot when I see one"


def test_closest_power_of_two():
    assert closest_power_of_two(5) == 8
    assert closest_power_of_two(13) == 16
    assert closest_power_of_two(120) == 128
    assert closest_power_of_two(973) == 1024


def test_shorten_name():
    assert (
        shorten_name("And now for soemthing completely different.", 16)
        == "And now for so.."
    )


@skip_cupy
def test_is_cupy_array():
    cp_array = cp.array([0, 1, 2])
    np_array = np.array([0, 1, 2])
    assert is_cupy_array(cp_array)
    assert not is_cupy_array(np_array)


@skip_cupy
def test_to_numpy():
    cp_array = cp.array([0, 1, 2])
    np_array = np.array([0, 1, 2])
    np.testing.assert_allclose(to_numpy(cp_array), np_array)
    np.testing.assert_allclose(to_numpy(np_array), np_array)


def test_to_numpy_error():
    da_array = da.array([0, 1, 2])
    with pytest.raises(TypeError):
        to_numpy(da_array)

    list_array = [[0, 1, 2]]
    with pytest.raises(TypeError):
        to_numpy(list_array)


def test_get_array_module():
    np_array = np.array([0, 1, 2])
    assert get_array_module(np_array) == np


@skip_cupy
def test_get_array_module_cupy():
    cp_array = cp.array([0, 1, 2])
    assert get_array_module(cp_array) == cp
