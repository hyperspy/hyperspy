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

"""Common docstring snippets for utils.

"""

STACK_METADATA_ARG = \
    """stack_metadata : {bool, int}
        If integer, this value defines the index of the signal in the signal
        list, from which the ``metadata`` and ``original_metadata`` are taken.
        If ``True``, the ``original_metadata`` and ``metadata`` of each signals
        are stacked and saved in ``original_metadata.stack_elements`` of the
        returned signal. In this case, the ``metadata`` are copied from the
        first signal in the list.
        If False, the ``metadata`` and ``original_metadata`` are not copied."""
