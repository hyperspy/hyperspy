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


import importlib
import logging

import hyperspy.api_nogui
from hyperspy.api_nogui import __doc__, get_configuration_directory_path
from hyperspy.defaults_parser import preferences
from hyperspy.logger import set_log_level
from hyperspy.Release import version as __version__


_logger = logging.getLogger(__name__)


__all__ = hyperspy.api_nogui.__all__
_import_mapping = hyperspy.api_nogui._import_mapping


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        if name in _import_mapping.keys():
            import_path = 'hyperspy' + _import_mapping.get(name)
            return getattr(importlib.import_module(import_path), name)
        else:
            return importlib.import_module("." + name, 'hyperspy')
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
