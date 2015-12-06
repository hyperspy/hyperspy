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
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>

import nose.tools as nt
import numpy as np

from hyperspy.misc.array_tools import dict2sarray


dt = [('x', np.uint8), ('y', np.uint16), ('text', (bytes, 6))]

@nt.raises(ValueError)
def test_d2s_fail():
    d = dict(x=5, y=10, text='abcdef')
    dict2sarray(d)

def test_d2s_dtype():
    d = dict(x=5, y=10, text='abcdef')
    ref = np.zeros((1,), dtype=dt)
    ref['x'] = 5
    ref['y'] = 10
    ref['text'] = 'abcdef'
    
    nt.assert_equal(ref, dict2sarray(d, dtype=dt))

def test_d2s_extra_dict_ok():
    d = dict(x=5, y=10, text='abcdef', other=55)
    ref = np.zeros((1,), dtype=dt)
    ref['x'] = 5
    ref['y'] = 10
    ref['text'] = 'abcdef'
    
    nt.assert_equal(ref, dict2sarray(d, dtype=dt))

def test_d2s_sarray():
    d = dict(x=5, y=10, text='abcdef')
    
    base = np.zeros((1,), dtype=dt)
    base['x'] = 65
    base['text'] = 'gg'

    ref = np.zeros((1,), dtype=dt)
    ref['x'] = 5
    ref['y'] = 10
    ref['text'] = 'abcdef'
    
    nt.assert_equal(ref, dict2sarray(d, sarray=base))

def test_d2s_partial_sarray():
    d = dict(text='abcdef')
    
    base = np.zeros((1,), dtype=dt)
    base['x'] = 65
    base['text'] = 'gg'

    ref = np.zeros((1,), dtype=dt)
    ref['x'] = 65
    ref['y'] = 0
    ref['text'] = 'abcdef'
    
    nt.assert_equal(ref, dict2sarray(d, sarray=base))

def test_d2s_type_cast_ok():
    d = dict(x='34', text=55)

    ref = np.zeros((1,), dtype=dt)
    ref['x'] = 34
    ref['y'] = 0
    ref['text'] = '55'
    
    nt.assert_equal(ref, dict2sarray(d, dtype=dt))

@nt.raises(ValueError)
def test_d2s_type_cast_invalid():
    d = dict(x='Test')
    dict2sarray(d, dtype=dt)

def test_d2s_string_cut():
    d = dict(text='Testerstring')
    sa = dict2sarray(d, dtype=dt)
    nt.assert_equal(sa['text'][0], 'Tester')