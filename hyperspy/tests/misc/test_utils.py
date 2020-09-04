# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

from hyperspy.misc.utils import slugify, parse_quantity, is_hyperspy_signal
from hyperspy import signals
import numpy as np


def test_slugify():
    assert slugify('a') == 'a'
    assert slugify('1a') == '1a'
    assert slugify('1') == '1'
    assert slugify('a a') == 'a_a'

    assert slugify('a', valid_variable_name=True) == 'a'
    assert slugify('1a', valid_variable_name=True) == 'Number_1a'
    assert slugify('1', valid_variable_name=True) == 'Number_1'

    assert slugify('a', valid_variable_name=False) == 'a'
    assert slugify('1a', valid_variable_name=False) == '1a'
    assert slugify('1', valid_variable_name=False) == '1'


def test_parse_quantity():
    # From the metadata specification, the quantity is defined as
    # "name (units)" without backets in the name of the quantity
    assert parse_quantity('a (b)') == ('a', 'b')
    assert parse_quantity('a (b/(c))') == ('a', 'b/(c)')
    assert parse_quantity('a (c) (b/(c))') == ('a (c)', 'b/(c)')
    assert parse_quantity('a [b]') == ('a [b]', '')
    assert parse_quantity('a [b]', opening = '[', closing = ']') == ('a', 'b')


def test_is_hyperspy_signal():
    s = signals.Signal1D(np.zeros((5, 5, 5)))
    p = object()
    assert is_hyperspy_signal(s) is True
    assert is_hyperspy_signal(p) is False
