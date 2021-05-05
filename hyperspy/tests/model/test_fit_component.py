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


import numpy as np
import pytest

from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D
from hyperspy.components1d import Gaussian
from hyperspy.models.model1d import ComponentFit
from hyperspy.exceptions import SignalDimensionError


class TestFitOneComponent:

    def setup_method(self, method):
        g = Gaussian()
        g.A.value = 10000.0
        g.centre.value = 5000.0
        g.sigma.value = 500.0
        axis = np.arange(10000)
        s = Signal1D(g.function(axis))
        m = s.create_model()
        self.model = m
        self.g = g
        self.axis = axis

    @pytest.mark.parametrize("signal_range", [(4000, 6000), 'interactive'])
    def test_fit_component(self, signal_range):
        m = self.model
        axis = self.axis
        g = self.g

        g1 = Gaussian()
        m.append(g1)
        cf = ComponentFit(m, g1, signal_range=signal_range)
        if signal_range == 'interactive':
            cf.ss_left_value, cf.ss_right_value = (4000, 6000)
        cf._fit_fired()
        np.testing.assert_allclose(g.function(axis),
                                   g1.function(axis),
                                   rtol=0.0,
                                   atol=10e-3)

    def test_component_not_in_model(self):
        with pytest.raises(ValueError):
            self.model.fit_component(self.g)


def test_Component_fit_wrong_signal():
    s = Signal2D(np.arange(2*3*4).reshape(2, 3, 4))
    m = s.create_model()
    with pytest.raises(SignalDimensionError):
        ComponentFit(m, Gaussian())


class TestFitSeveralComponent:

    def setup_method(self, method):
        gs1 = Gaussian()
        gs1.A.value = 10000.0
        gs1.centre.value = 5000.0
        gs1.sigma.value = 500.0

        gs2 = Gaussian()
        gs2.A.value = 60000.0
        gs2.centre.value = 2000.0
        gs2.sigma.value = 300.0

        gs3 = Gaussian()
        gs3.A.value = 20000.0
        gs3.centre.value = 6000.0
        gs3.sigma.value = 100.0

        axis = np.arange(10000)
        total_signal = (gs1.function(axis) +
                        gs2.function(axis) +
                        gs3.function(axis))

        s = Signal1D(total_signal)
        m = s.create_model()

        g1 = Gaussian()
        g2 = Gaussian()
        g3 = Gaussian()
        m.append(g1)
        m.append(g2)
        m.append(g3)

        self.model = m
        self.gs1 = gs1
        self.gs2 = gs2
        self.gs3 = gs3
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.axis = axis
        self.rtol = 0.01

    def test_fit_component_active_state(self):
        m = self.model
        axis = self.axis
        g1 = self.g1
        g2 = self.g2
        g3 = self.g3
        g2.active = True
        g3.active = False
        m.fit_component(g1, signal_range=(4500, 5200), fit_independent=True)
        np.testing.assert_allclose(self.gs1.function(axis),
                                   g1.function(axis),
                                   rtol=self.rtol,
                                   atol=10e-3)
        assert g1.active
        assert g2.active
        assert not g3.active

    def test_fit_component_free_state(self):
        m = self.model
        axis = self.axis
        g1 = self.g1
        g2 = self.g2
        g3 = self.g3
        g2.A.free = False
        g2.sigma.free = False
        m.fit_component(g1, signal_range=(4500, 5200))
        np.testing.assert_allclose(self.gs1.function(axis),
                                   g1.function(axis),
                                   rtol=self.rtol,
                                   atol=10e-3)

        assert g1.A.free
        assert g1.sigma.free
        assert g1.centre.free

        assert not g2.A.free
        assert not g2.sigma.free
        assert g2.centre.free

        assert g3.A.free
        assert g3.sigma.free
        assert g3.centre.free

    def test_fit_multiple_component(self):
        m = self.model
        g1 = self.g1
        g2 = self.g2
        g3 = self.g3
        m.fit_component(g1, signal_range=(4500, 5200))
        m.fit_component(g2, signal_range=(1500, 2200))
        m.fit_component(g3, signal_range=(5800, 6150))
        np.testing.assert_allclose(self.model.signal.data,
                                   m(),
                                   rtol=self.rtol,
                                   atol=10e-3)


class TestFitSI:

    def setup_method(self, method):
        s = Signal1D(np.random.random((2, 2, 8)))
        m = s.create_model()
        G = Gaussian()
        m.append(G)

        self.model = m
        self.G = G

    def test_fit_spectrum_image(self):
        m = self.model
        G = self.G
        # HyperSpy 2.0: remove setting iterpath='serpentine'
        m.fit_component(G, signal_range=(2, 7), only_current=False,
                        iterpath='serpentine')
        m.axes_manager.indices = (0, 0)
        A = G.A.value
        m.axes_manager.indices = (1, 1)
        B = G.A.value
        assert not A == B
