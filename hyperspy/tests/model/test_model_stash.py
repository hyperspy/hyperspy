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
        np.testing.assert_equal(
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
    def test_stash_name_error2(self):
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


class TestStashFetching:

    def setUp(self):
        s = Spectrum([range(100), ] * 2)
        m = s.create_model()
        m.append(Gaussian())
        m.components.Gaussian.A.value = 3.
        m.components.Gaussian.A.map['values'].fill(-1)
        self.m = m

    def test_stash_fetch_noargs(self):
        m = self.m
        m.stash.save()
        m.components.Gaussian.A.value = 5.
        m.components.Gaussian.A.map['values'].fill(3)
        m.stash.fetch_values()
        nt.assert_equal(m.components.Gaussian.A.value, 3.)
        nt.assert_true(np.all(m.components.Gaussian.A.map['values'] == -1.))

    def test_stash_component_list(self):
        m = self.m
        m.append(Lorentzian())
        m.components.Lorentzian.A.value = 14.
        m.components.Lorentzian.A.map['values'].fill(-31)

        m.stash.save('a')

        m.components.Gaussian.A.value = 5.
        m.components.Gaussian.A.map['values'].fill(3)
        m.components.Lorentzian.A.value = 5.
        m.components.Lorentzian.A.map['values'].fill(3)

        m.stash.save('b')

        m.stash.fetch_values('a', component_list=['Lorentzian', ])

        nt.assert_equal(m['Gaussian'].A.value, 5.)
        nt.assert_true(np.all(m['Gaussian'].A.map['values'] == 3))
        nt.assert_equal(m['Lorentzian'].A.value, 14.)
        nt.assert_true(np.all(m['Lorentzian'].A.map['values'] == -31))

        m.stash.apply('b', backup=False)

        m.stash.fetch_values('a', component_list=[m[1], ])

        nt.assert_equal(m['Gaussian'].A.value, 5.)
        nt.assert_true(np.all(m['Gaussian'].A.map['values'] == 3))
        nt.assert_equal(m['Lorentzian'].A.value, 14.)
        nt.assert_true(np.all(m['Lorentzian'].A.map['values'] == -31))

    def test_stash_parameter_list(self):
        m = self.m
        m.components.Gaussian.centre.value = 14.
        m.stash.save('a')
        m.components.Gaussian.centre.value = 0.3
        m.components.Gaussian.A.value = 5.
        m.stash.fetch_values('a', parameter_list=['centre', ])
        nt.assert_equal(m.components.Gaussian.A.value, 5.)
        nt.assert_equal(m.components.Gaussian.centre.value, 14.)

    def test_stash_mask(self):
        m = self.m
        m.stash.save('a')
        m.components.Gaussian.A.map['values'].fill(3)
        m.stash.fetch_values(mask=[True, False])
        nt.assert_equal(m.components.Gaussian.A.map['values'][0], -1)
        nt.assert_equal(m.components.Gaussian.A.map['values'][1], 3)


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
