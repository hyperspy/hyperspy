# -*- coding: utf-8 -*-
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

import inspect
import itertools

import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy import components1d
from hyperspy.component import Component

TRUE_FALSE_2_TUPLE = [p for p in itertools.product((True, False), repeat=2)]


def get_components1d_name_list():
    components1d_name_list = []
    for c_name in dir(components1d):
        obj = getattr(components1d, c_name)
        if inspect.isclass(obj) and issubclass(obj, Component):
            components1d_name_list.append(c_name)
    return components1d_name_list


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in true_divide:RuntimeWarning"
)
@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in true_divide:RuntimeWarning"
)
@pytest.mark.filterwarnings("ignore:invalid value encountered in cos:RuntimeWarning")
@pytest.mark.parametrize("component_name", get_components1d_name_list())
def test_creation_components1d(component_name):
    s = hs.signals.Signal1D(np.zeros(1024))
    s.axes_manager[0].offset = 100
    s.axes_manager[0].scale = 0.01

    kwargs = {}
    if component_name == "ScalableFixedPattern":
        kwargs["signal1D"] = s
    elif component_name == "Expression":
        kwargs.update({"expression": "a*x+b", "name": "linear"})
    elif component_name == "Bleasdale":
        # This component only works with numexpr.
        pytest.importorskip("numexpr")

    component = getattr(components1d, component_name)(**kwargs)
    component.function(np.arange(1, 100))

    # Do a export/import cycle to check all the components can be re-created.
    m = s.create_model()
    m.append(component)
    model_dict = m.as_dictionary()

    m2 = s.create_model()
    m2._load_dictionary(model_dict)

    # For Expression based component which uses sympy to compute gradient
    # automatically, check that the gradient are working
    for parameter in component.parameters:
        grad = getattr(component, f"grad_{parameter.name}", None)
        if grad is not None:
            grad(np.arange(1, 100))
