# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

import importlib
import warnings

from hyperspy.exceptions import VisibleDeprecationWarning

# ruff: noqa: F822

__all__ = [
    "is_dask_array",
    "attrsetter",
    "stash_active_state",
    "dummy_context_manager",
    "str2num",
    "parse_quantity",
    "slugify",
    "DictionaryTreeBrowser",
    "strlist2enumeration",
    "ensure_unicode",
    "check_long_string",
    "replace_html_symbols",
    "add_key_value",
    "swapelem",
    "rollelem",
    "fsdict",
    "find_subclasses",
    "isiterable",
    "ordinal",
    "underline",
    "closest_power_of_two",
    "stack",
    "shorten_name",
    "transpose",
    "multiply",
    "iterable_not_string",
    "add_scalar_axis",
    "get_object_package_info",
    "is_hyperspy_signal",
    "nested_dictionary_merge",
    "is_cupy_array",
    "to_numpy",
    "get_array_module",
    "display",
    "TupleSA",
    "_get_block_pattern",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name == "_get_block_pattern":
        warnings.warn(
            "`_get_block_pattern` has moved to `hyperspy.misc.dask_utils`. "
            "It is for internal use only and may be removed in the future.",
            VisibleDeprecationWarning,
        )
        return getattr(importlib.import_module("hyperspy.misc.dask_utils"), name)
    if name in __all__:
        return getattr(importlib.import_module("hyperspy.misc._utils"), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
