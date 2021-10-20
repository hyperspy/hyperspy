# Copyright 2007-2021 The HyperSpy developers
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
from hyperspy.misc.model_tools import (_is_iter, _iter_join, _non_iter,
                                       current_component_values,
                                       current_model_values)


class TestSetParameters:

    def setup_method(self):
        self.model = EDS_SEM_Spectrum().create_model()
        self.component = self.model[1]
        # We use bmin instead of A because it's a bit more exotic
        self.component.A.bmin = 1.23456789012
        self.component_not_free = self.model[3]
        self.component_not_free.set_parameters_not_free()
        self.component_not_free.A.bmin = 9.87654321098
        self.component_inactive = self.model[4]
        self.component_inactive.active = False
        self.component_inactive.A.bmin = 5.67890123456

    @pytest.mark.parametrize("only_free, only_active", [(True, False), (True, False)])
    def test_component_current_component_values(self, only_free, only_active):
        "Many decimals aren't printed, few decimals are"
        string_representation = str(current_component_values(self.component, only_free, only_active).__repr__())
        html_representation = str(current_component_values(self.component, only_free, only_active)._repr_html_())
        assert "1.234" in string_representation
        assert "1.23456789012" not in string_representation
        assert "1.234" in html_representation
        assert "1.23456789012" not in html_representation

    def test_component_current_component_values_only_free(self, only_free=True, only_active=False):
        "Parameters with free=False values should not be present in repr"
        string_representation = str(current_component_values(self.component_not_free, only_free, only_active).__repr__())
        html_representation = str(current_component_values(self.component_not_free, only_free, only_active)._repr_html_())
        assert "9.87" not in string_representation
        assert "9.87" not in html_representation

    @pytest.mark.parametrize("only_free, only_active", [(True, False), (True, False)])
    def test_component_current_model_values(self, only_free, only_active):
        "Many decimals aren't printed, few decimals are"
        string_representation = str(current_model_values(self.model, only_free, only_active).__repr__())
        html_representation = str(current_model_values(self.model, only_free, only_active)._repr_html_())
        assert "1.234" in string_representation
        assert "1.23456789012" not in string_representation
        assert "1.234" in html_representation
        assert "1.23456789012" not in html_representation

        if only_free:
            # Parameters with free=False values should not be present in repr
            assert "9.87" not in string_representation
            assert "9.87" not in html_representation
        if only_active:
            # components with active=False values should not be present in repr"
            assert "5.67" not in string_representation
            assert "5.67" not in html_representation

    @pytest.mark.parametrize("only_free, only_active", [(True, False), (True, False)])
    def test_component_current_model_values_comp_list(self, only_free, only_active):
        comp_list = [self.component, self.component_not_free, self.component_inactive]
        string_representation = str(current_model_values(self.model, only_free, only_active, comp_list).__repr__())
        html_representation = str(current_model_values(self.model, only_free, only_active, comp_list)._repr_html_())
        assert "1.234" in string_representation
        assert "1.23456789012" not in string_representation
        assert "1.234" in html_representation
        assert "1.23456789012" not in html_representation

        if only_free:
            assert "9.87" not in string_representation
            assert "9.87" not in html_representation
        if only_active:
            assert "5.67" not in string_representation
            assert "5.67" not in html_representation

    @pytest.mark.parametrize("fancy", (True, False))
    def test_model_current_model_values(self, fancy):
        self.model.print_current_values(fancy=fancy)
        
    @pytest.mark.parametrize("fancy", (True, False))
    def test_component_print_current_values(self, fancy):
        self.model[0].print_current_values(fancy=fancy)

    @pytest.mark.parametrize("fancy", (True, False))
    def test_model_print_current_values(self, fancy):
        self.model.print_current_values(fancy=fancy)

    def test_zero_in_fancy_print(self):
        "Ensure parameters with value=0 are printed too"
        assert "<td>a1</td><td>True</td><td>     0</td>" in current_component_values(self.model[0])._repr_html_()

    def test_zero_in_normal_print(self):
        "Ensure parameters with value=0 are printed too"
        assert "            a0 |  True |          0 |" in str(current_component_values(self.model[0]).__repr__)

    def test_related_tools(self):
        assert _is_iter([1,2,3])
        assert _is_iter((1,2,3))
        assert not _is_iter(1)

        assert _iter_join([1.2345678, 5.67890]) == '(1.23457, 5.6789)'
        assert _iter_join([1.2345678, 5.67890]) == '(1.23457, 5.6789)'
        assert _iter_join([1, 5]) == '(     1,      5)'

        assert _non_iter(None) == ""
        assert _non_iter(5) == '     5'
        assert _non_iter(5.123456789) == '5.12346'
