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


import numpy as np

from hyperspy._signals.signal1d import Signal1D
from hyperspy.components1d import Gaussian


class TestSetParameterInModel:

    def setup_method(self, method):
        g1 = Gaussian()
        g2 = Gaussian()
        g3 = Gaussian()
        s = Signal1D(np.arange(10))
        m = s.create_model()
        m.append(g1)
        m.append(g2)
        m.append(g3)
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.model = m

    def test_set_parameter_in_model_not_free(self):
        m = self.model
        g1 = self.g1
        g2 = self.g2
        g3 = self.g3
        m.set_parameters_not_free()
        assert len(g1.free_parameters) == 0
        assert len(g2.free_parameters) == 0
        assert len(g3.free_parameters) == 0

    def test_set_parameter_in_model_free(self):
        m = self.model
        g1 = self.g1
        g2 = self.g2
        g3 = self.g3
        g1.A.free = False
        g2.sigma.free = False
        g3.centre.free = False
        m.set_parameters_free()
        assert len(g1.free_parameters) == len(g1.parameters)
        assert len(g2.free_parameters) == len(g2.parameters)
        assert len(g3.free_parameters) == len(g3.parameters)

    def test_set_parameter_in_model1(self):
        m = self.model
        g1 = self.g1
        g2 = self.g2
        g3 = self.g3
        m.set_parameters_not_free([g1, g2])
        assert len(g1.free_parameters) == 0
        assert len(g2.free_parameters) == 0
        assert len(g3.free_parameters) == len(g3.parameters)

    def test_set_parameter_in_model2(self):
        m = self.model
        g1 = self.g1
        g2 = self.g2
        g3 = self.g3
        m.set_parameters_not_free()
        m.set_parameters_free([g3])
        assert len(g1.free_parameters) == 0
        assert len(g2.free_parameters) == 0
        assert len(g3.free_parameters) == len(g3.parameters)

    def test_set_parameter_in_model3(self):
        m = self.model
        g1 = self.g1
        g2 = self.g2
        g3 = self.g3
        m.set_parameters_not_free(parameter_name_list=['A'])
        assert not g1.A.free
        assert g1.sigma.free
        assert g1.centre.free
        assert not g2.A.free
        assert g2.sigma.free
        assert g2.centre.free
        assert not g3.A.free
        assert g3.sigma.free
        assert g3.centre.free

    def test_set_parameter_in_model4(self):
        m = self.model
        g1 = self.g1
        g2 = self.g2
        g3 = self.g3
        m.set_parameters_not_free([g2], parameter_name_list=['A'])
        assert g1.A.free
        assert g1.sigma.free
        assert g1.centre.free
        assert not g2.A.free
        assert g2.sigma.free
        assert g2.centre.free
        assert g3.A.free
        assert g3.sigma.free
        assert g3.centre.free

    def test_set_parameter_in_model5(self):
        m = self.model
        g1 = self.g1
        g2 = self.g2
        g3 = self.g3
        m.set_parameters_not_free()
        m.set_parameters_free([g1], parameter_name_list=['centre'])
        assert not g1.A.free
        assert not g1.sigma.free
        assert g1.centre.free
        assert not g2.A.free
        assert not g2.sigma.free
        assert not g2.centre.free
        assert not g3.A.free
        assert not g3.sigma.free
        assert not g3.centre.free
