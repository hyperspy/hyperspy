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

import pytest

from hyperspy.datasets.example_signals import EDS_SEM_Spectrum


class TestSetParameters:

    def setup_method(self):
        self.model = EDS_SEM_Spectrum().create_model()

    @pytest.mark.parametrize("fancy", (True, False))
    def test_component_print_current_values(self, fancy):
        self.model[0].print_current_values(fancy=fancy)

    @pytest.mark.parametrize("fancy", (True, False))
    def test_model_print_current_values(self, fancy):
        self.model.print_current_values(fancy=fancy)
