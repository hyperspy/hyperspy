# Copyright 2007-2026 The HyperSpy developers
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
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.misc.model_tools import (
    CurrentComponentValues,
    CurrentModelValues,
    _format_string,
)


class TestSetParameters:
    def setup_method(self):
        model = hs.signals.Signal1D(np.arange(100)).create_model()
        p0 = hs.model.components1D.Polynomial(order=6)
        g1 = hs.model.components1D.Gaussian()
        g2 = hs.model.components1D.Gaussian()
        g3 = hs.model.components1D.Gaussian()
        g4 = hs.model.components1D.Gaussian()
        model.extend([p0, g1, g2, g3, g4])
        component = model[1]
        # We use bmin instead of A because it's a bit more exotic
        component.A.bmin = 1.23456789012
        component_twinned = model[2]
        component_twinned.A.twin = g1.A
        component_not_free = model[3]
        component_not_free.set_parameters_not_free()
        component_not_free.A.bmin = 9.87654321098
        component_inactive = model[4]
        component_inactive.active = False
        component_inactive.A.bmin = 5.67890123456
        self.model = model
        self.component = component
        self.component_not_free = component_not_free
        self.component_inactive = component_inactive

    @pytest.mark.parametrize("only_free, only_active", [(True, False), (True, False)])
    def test_component_current_component_values(self, only_free, only_active):
        """Many decimals aren't printed, few decimals are"""
        string_representation = str(
            CurrentComponentValues(self.component, only_free, only_active).__repr__()
        )
        html_representation = str(
            CurrentComponentValues(self.component, only_free, only_active)._repr_html_()
        )
        assert "1.234" in string_representation
        assert "1.23456789012" not in string_representation
        assert "1.234" in html_representation
        assert "1.23456789012" not in html_representation

    def test_component_current_component_values_only_free(
        self, only_free=True, only_active=False
    ):
        """Parameters with free=False values should not be present in repr"""
        string_representation = str(
            CurrentComponentValues(
                self.component_not_free, only_free, only_active
            ).__repr__()
        )
        html_representation = str(
            CurrentComponentValues(
                self.component_not_free, only_free, only_active
            )._repr_html_()
        )
        assert "9.87" not in string_representation
        assert "9.87" not in html_representation

    @pytest.mark.parametrize("only_free, only_active", [(True, False), (True, False)])
    def test_component_current_model_values(self, only_free, only_active):
        """Many decimals aren't printed, few decimals are"""
        string_representation = str(
            CurrentModelValues(self.model, only_free, only_active).__repr__()
        )
        html_representation = str(
            CurrentModelValues(self.model, only_free, only_active)._repr_html_()
        )
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
        string_representation = str(
            CurrentModelValues(self.model, only_free, only_active, comp_list).__repr__()
        )
        html_representation = str(
            CurrentModelValues(
                self.model, only_free, only_active, comp_list
            )._repr_html_()
        )
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

    def test_model_current_model_values(self):
        self.model.print_current_values()

    def test_component_print_current_values(self):
        self.model[0].print_current_values()

    def test_model_print_current_values(self):
        self.model.print_current_values()

    def test_zero_in_html_print(self):
        """Ensure parameters with value=0 are printed too"""
        html = CurrentComponentValues(self.model[0])._repr_html_()
        assert "<table" in html
        assert "a1" in html
        assert "True" in html
        assert "0" in html

    def test_zero_in_normal_print(self):
        """Ensure parameters with value=0 are printed too"""
        table = str(CurrentComponentValues(self.model[0]))
        assert "a0" in table
        assert "True" in table
        assert "0" in table

    def test_twinned_in_print(self):
        table = str(CurrentComponentValues(self.model[2]))
        assert "A" in table
        assert "Twinned" in table


class TestHtmlEscaping:
    def setup_method(self):
        s = hs.signals.Signal1D(np.zeros((12, 25, 48)))
        s.metadata.General.title = "<b>UNTRUSTED_TITLE</b>"
        model = s.create_model()
        component = hs.model.components1D.Gaussian()
        component.name = "<b>UNTRUSTED_COMPONENT</b>"
        model.append(component)
        self.model = model
        self.component = component

    def test_current_model_values_html_escapes_title_and_component(self):
        """Markup in the signal title and component name is rendered as text."""
        html = CurrentModelValues(self.model)._repr_html_()
        assert "&lt;b&gt;UNTRUSTED_TITLE&lt;/b&gt;" in html
        assert "&lt;b&gt;UNTRUSTED_COMPONENT&lt;/b&gt;" in html
        assert "<b>UNTRUSTED_" not in html

    @pytest.mark.parametrize("only_active", [True, False])
    def test_current_component_values_html_escapes_name(self, only_active):
        """Markup in the component name is rendered as text in both headers."""
        html = CurrentComponentValues(
            self.component, only_active=only_active
        )._repr_html_()
        assert "&lt;b&gt;UNTRUSTED_COMPONENT&lt;/b&gt;" in html
        assert "<b>UNTRUSTED_COMPONENT" not in html

    def test_current_component_values_html_escapes_parameter_cell(self):
        """The parameter table cell escapes markup (PrettyTable control)."""
        self.component.A.name = "<i>UNTRUSTED_PARAM</i>"
        html = CurrentComponentValues(self.component)._repr_html_()
        assert "&lt;i&gt;UNTRUSTED_PARAM&lt;/i&gt;" in html
        assert "<i>UNTRUSTED_PARAM" not in html


def test_format_string():
    assert _format_string(None) == ""
    assert _format_string(5) == "5"
    assert _format_string(5.123456789) == "5.1235"
    assert _format_string(5.123456789, format_string=".6g") == "5.12346"
    assert _format_string((0, 1, 2)) == "(0, 1, 2)"
    assert (
        _format_string((0.123, 1.234, 2.345), format_string=".2g") == "(0.12, 1.2, 2.3)"
    )
    assert _format_string("Hello, World!") == "Hello, World!"
    assert _format_string("Hello, World!", max_length=5) == "He..."
    assert _format_string("Hello, World!", max_length=5, add_ellipsis=False) == "Hello"
