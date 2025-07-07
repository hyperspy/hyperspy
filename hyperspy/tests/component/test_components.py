# -*- coding: utf-8 -*-
# Copyright 2007-2025 The HyperSpy developers
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


@pytest.mark.parametrize("component_name", get_components1d_name_list())
def test_component_name_parameter(component_name):
    """Test that all components accept the name parameter and set it correctly."""
    # Skip components that need special arguments
    kwargs = {}
    if component_name == "ScalableFixedPattern":
        s = hs.signals.Signal1D(np.zeros(100))
        kwargs["signal1D"] = s
    elif component_name == "Expression":
        kwargs.update({"expression": "a*x+b", "name": "TestExpression"})
    elif component_name == "Bleasdale":
        # This component only works with numexpr.
        pytest.importorskip("numexpr")

    # Create a custom name for the component
    custom_name = f"My{component_name}"

    # For Expression component, we already set the name above
    if component_name != "Expression":
        kwargs["name"] = custom_name
    else:
        custom_name = "TestExpression"

    # Create the component with custom name
    component_class = getattr(components1d, component_name)
    component = component_class(**kwargs)

    # Test that the component was created successfully
    assert component is not None

    # Test that the official name (_id_name) is set correctly (should be class name)
    assert component._id_name == component_name

    # Test that the nickname (.name) is set to the custom name
    assert component.name == custom_name

    # Test that we can change the name after creation
    new_name = f"Updated{component_name}"
    component.name = new_name
    assert component.name == new_name

    # Test that the official name remains unchanged
    assert component._id_name == component_name


def test_expression_component_name_parameter_specific():
    """Test Expression component name parameter in more detail."""
    # Test basic Expression with name
    expr1 = hs.model.components1D.Expression("a*x + b", name="LinearFunction", a=1, b=0)
    assert expr1.name == "LinearFunction"
    assert expr1._id_name == "Expression"

    # Test Expression with complex formula
    expr2 = hs.model.components1D.Expression(
        "A * exp(-((x - centre)/sigma)**2)",
        name="CustomGaussian",
        A=1000,
        centre=100,
        sigma=10,
    )
    assert expr2.name == "CustomGaussian"
    assert expr2._id_name == "Expression"

    # Test default name behavior if not provided
    expr3 = hs.model.components1D.Expression("a*x + b", name="Default", a=1, b=0)
    assert expr3.name == "Default"


def test_expression_based_components_name_parameter():
    """Test that Expression-based components properly handle the name parameter."""
    # Test Gaussian component
    gaussian = hs.model.components1D.Gaussian(name="MyGaussian")
    assert gaussian.name == "MyGaussian"
    assert gaussian._id_name == "Gaussian"

    # Test with default name
    gaussian_default = hs.model.components1D.Gaussian()
    assert gaussian_default.name == "Gaussian"
    assert gaussian_default._id_name == "Gaussian"

    # Test PowerLaw component
    powerlaw = hs.model.components1D.PowerLaw(name="MyPowerLaw")
    assert powerlaw.name == "MyPowerLaw"
    assert powerlaw._id_name == "PowerLaw"

    # Test Lorentzian component
    lorentzian = hs.model.components1D.Lorentzian(name="MyLorentzian")
    assert lorentzian.name == "MyLorentzian"
    assert lorentzian._id_name == "Lorentzian"


def test_direct_component_subclasses_name_parameter():
    """Test that components inheriting directly from Component handle the name parameter."""
    # Test Offset component
    offset = hs.model.components1D.Offset(name="MyOffset")
    assert offset.name == "MyOffset"
    assert offset._id_name == "Offset"

    # Test with default name
    offset_default = hs.model.components1D.Offset()
    assert offset_default.name == "Offset"
    assert offset_default._id_name == "Offset"


def test_name_parameter_in_model():
    """Test that named components work correctly in models."""
    s = hs.signals.Signal1D(np.zeros(100))
    m = s.create_model()

    # Add components with custom names
    gaussian = hs.model.components1D.Gaussian(name="Peak1")
    background = hs.model.components1D.Offset(name="Background")

    m.append(gaussian)
    m.append(background)

    # Test that components can be accessed by their custom names
    assert hasattr(m.components, "Peak1")
    assert hasattr(m.components, "Background")
    assert m.components.Peak1 is gaussian
    assert m.components.Background is background

    # Test that changing name after adding to model updates access
    gaussian.name = "MainPeak"
    assert hasattr(m.components, "MainPeak")
    assert m.components.MainPeak is gaussian
