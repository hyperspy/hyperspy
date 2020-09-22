# Copyright 2007-2016 The HyperSpy developers
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


from os import remove
from unittest import mock
import gc

import numpy as np

import pytest

from hyperspy._signals.signal1d import Signal1D
from hyperspy.io import load
from hyperspy.components1d import Gaussian


def clean_model_dictionary(d):
    for c in d['components']:
        for p in c['parameters']:
            del p['self']
    return d


class TestModelStoring:

    def setup_method(self, method):
        s = Signal1D(range(100))
        m = s.create_model()
        m.append(Gaussian())
        m.fit()
        self.m = m

    def test_models_getattr(self):
        m = self.m
        s = m.signal
        m.store()
        assert s.models.a is s.models['a']

    def test_models_stub_methods(self):
        m = self.m
        s = m.signal
        m.store()
        s.models.pop = mock.MagicMock()
        s.models.remove = mock.MagicMock()
        s.models.restore = mock.MagicMock()
        s.models.a.restore()
        s.models.a.remove()
        s.models.a.pop()

        assert s.models.pop.call_count == 1
        assert s.models.remove.call_count == 1
        assert s.models.restore.call_count == 1

        assert s.models.pop.call_args[0] == ('a',)
        assert s.models.remove.call_args[0] == ('a',)
        assert s.models.restore.call_args[0] == ('a',)

    def test_models_pop(self):
        m = self.m
        s = m.signal
        m.store()
        s.models.remove = mock.MagicMock()
        s.models.restore = mock.MagicMock()
        s.models.pop('a')
        assert s.models.remove.call_count == 1
        assert s.models.restore.call_count == 1
        assert s.models.remove.call_args[0] == ('a',)
        assert s.models.restore.call_args[0] == ('a',)

    def test_model_store(self):
        m = self.m
        m.store()
        d = m.as_dictionary(True)
        np.testing.assert_equal(
            d,
            m.signal.models._models.a._dict.as_dictionary())

    def test_actually_stored(self):
        m = self.m
        m.store()
        m[0].A.map['values'][0] += 13.33
        m1 = m.signal.models.a.restore()
        assert m[0].A.map['values'] != m1[0].A.map['values']

    def test_models_restore_remove(self):
        m = self.m
        s = m.signal
        m.store('a')
        m1 = s.models.restore('a')
        m2 = s.models.a.restore()
        d_o = clean_model_dictionary(m.as_dictionary())
        d_1 = clean_model_dictionary(m1.as_dictionary())
        d_2 = clean_model_dictionary(m2.as_dictionary())
        np.testing.assert_equal(d_o, d_1)
        np.testing.assert_equal(d_o, d_2)
        assert 1 == len(s.models)
        s.models.a.remove()
        assert 0 == len(s.models)

    def test_store_name_error1(self):
        s = self.m.signal
        with pytest.raises(KeyError):
            s.models.restore('a')

    def test_store_name_error2(self):
        s = self.m.signal
        with pytest.raises(KeyError):
            s.models.restore(3)

    def test_store_name_error3(self):
        s = self.m.signal
        with pytest.raises(KeyError):
            s.models.restore('_a')

    def test_store_name_error4(self):
        s = self.m.signal
        with pytest.raises(KeyError):
            s.models.restore('a._dict')

    def test_store_name_error5(self):
        s = self.m.signal
        self.m.store('b')
        with pytest.raises(KeyError):
            s.models.restore('a')


class TestModelSaving:

    def setup_method(self, method):
        s = Signal1D(range(100))
        m = s.create_model()
        m.append(Gaussian())
        m.components.Gaussian.A.value = 13
        m.components.Gaussian.name = 'something'
        self.m = m

    def test_save_and_load_model(self):
        m = self.m
        m.save('tmp.hdf5', overwrite=True)
        l = load('tmp.hdf5')
        assert hasattr(l.models, 'a')
        n = l.models.restore('a')
        assert n.components.something.A.value == 13

    def teardown_method(self, method):
        gc.collect()        # Make sure any memmaps are closed first!
        remove('tmp.hdf5')
