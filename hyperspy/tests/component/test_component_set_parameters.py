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


from hyperspy.components1d import Gaussian


class TestSetParameters:

    def setup_method(self, method):
        self.gaussian = Gaussian()

    def test_set_parameters_not_free1(self):
        g = self.gaussian
        g.set_parameters_not_free()
        free_parameters = len(g.free_parameters)
        assert free_parameters == 0

    def test_set_parameters_not_free2(self):
        g = self.gaussian
        g.set_parameters_not_free(parameter_name_list=['A'])
        free_parameters = len(g.free_parameters)
        parameters = len(g.parameters) - 1
        assert free_parameters == parameters

    def test_set_parameters_free1(self):
        g = self.gaussian
        g.A.free = False
        g.set_parameters_free()
        free_parameters = len(g.free_parameters)
        parameters = len(g.parameters)
        assert free_parameters == parameters

    def test_set_parameters_free2(self):
        g = self.gaussian
        g.A.free = False
        g.centre.free = False
        g.sigma.free = False
        g.set_parameters_free(parameter_name_list=['A'])
        free_parameters = len(g.free_parameters)
        parameters = len(g.parameters) - 2
        assert free_parameters == parameters
