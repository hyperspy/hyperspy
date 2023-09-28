# -*- coding: utf-8 -*-
# Copyright 2007-2023 The exspy developers
#
# This file is part of exspy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# exspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with exspy. If not, see <https://www.gnu.org/licenses/#GPL>.

import pytest

from exspy.misc.eds.utils import _get_element_and_line


def test_get_element_and_line():
    assert _get_element_and_line('Mn_Ka') == ('Mn', 'Ka')

    with pytest.raises(ValueError):
        _get_element_and_line('MnKa') == -1
