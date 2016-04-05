# -*- coding: utf-8 -*-
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

import sys

import logging
import warnings
from hyperspy.exceptions import VisibleDeprecationWarning

_logger = logging.getLogger(__name__)


def warning_exit(text):
    _logger.critical(text)
    warnings.warn(
        "The function `warning_exit()` has been deprecated, and "
        "will be removed in HyperSpy 0.10. Please raise an appropriate "
        "`exception instead.",
        VisibleDeprecationWarning)
    sys.exit(1)


def warning(text):
    _logger.warning(text)
    warnings.warn(
        "The function `warning()` has been deprecated in favour of python "
        "logging. It will be removed in HyperSpy 0.10. Please use "
        "`logging.getLogger(__name__).warning()` instead.",
        VisibleDeprecationWarning)


def information(text):
    _logger.info(text)
    warnings.warn(
        "The function `information()` has been deprecated in favour of python "
        "logging. It will be removed in HyperSpy 0.10. Please use "
        "`logging.getLogger(__name__).info()` instead.",
        VisibleDeprecationWarning)


def alert(text):
    _logger.error(text)
    warnings.warn(
        "The function `alert()` has been deprecated in favour of python "
        "logging. It will be removed in HyperSpy 0.10. Please use "
        "`logging.getLogger(__name__).error()` instead.",
        VisibleDeprecationWarning)
