# Copyright 2007-2015 The HyperSpy developers
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
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.


import numpy as np

from os import remove
import nose.tools as nt
from hyperspy._signals.spectrum import Spectrum
from hyperspy.io import load
from hyperspy.components import Gaussian, Lorentzian


class TestModelStash:

    def setUp(self):
        s = Spectrum(range(100))
        m = s.create_model()
        m.append(Gaussian())
        m.fit()
        self.m = m

    def test_stash_methods(self):
        m = self.m
        m.stash.save()
        d = m.as_dictionary(True)
        nt.assert_dict_equal(
            d,
            m.spectrum.metadata.Analysis.models.a._dict.as_dictionary())
        m.append(Lorentzian())
        m.stash.apply('a')
        nt.assert_equal(m[0].name, 'Gaussian')
        nt.assert_equal(len(m), 1)
        m.append(Lorentzian())
        m.append(Lorentzian())
        m.stash.pop('a')
        nt.assert_equal(m[0].name, 'Gaussian')
        nt.assert_equal(len(m), 1)
        m.stash.save()
        m.stash.remove('a')

    @nt.raises(KeyError)
    def test_stash_name_error1(self):
        m = self.m
        m.stash.pop()

    @nt.raises(KeyError)
    def test_stash_name_error1(self):
        m = self.m
        m.stash.save()
        m.stash.pop('b')

    def test_stash_history(self):
        m = self.m
        m.stash.save('a')
        m.append(Lorentzian())
        m.stash.save('b')
        m.stash.apply('a')
        nt.assert_equal(len(m), 1)
        m.stash.apply()
        nt.assert_equal(len(m), 2)
        m.append(Lorentzian())
        m.stash.save('a')
        m.stash.apply()
        nt.assert_equal(len(m), 3)


class TestModelSaving:

    def setUp(self):
        s = Spectrum(range(100))
        m = s.create_model()
        m.append(Gaussian())
        m.components.Gaussian.A.value = 13
        m.components.Gaussian.name = 'something'
        self.m = m

    def test_save_and_load_model(self):
        m = self.m
        s = m.spectrum
        m.stash.save()
        s.save('tmp.hdf5')
        l = load('tmp.hdf5')
        nt.assert_true(l.metadata.has_item('Analysis.models.a'))
        n = l.create_model()
        nt.assert_equal(len(n.stash), 1)
        n.stash.apply()
        nt.assert_equal(n.components.something.A.value, 13)

    def tearDown(self):
        remove('tmp.hdf5')
