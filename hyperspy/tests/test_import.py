# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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


def test_import_version():
    from hyperspy import __version__  # noqa: F401


def test_import():
    import hyperspy

    for obj_name in hyperspy.__all__:
        getattr(hyperspy, obj_name)


def test_import_api():
    import hyperspy.api

    for obj_name in hyperspy.api.__all__:
        getattr(hyperspy.api, obj_name)


def test_import_data():
    import hyperspy.data

    for obj_name in hyperspy.data.__all__:
        getattr(hyperspy.data, obj_name)


def test_import_utils():
    import hyperspy.utils

    for obj_name in hyperspy.utils.__all__:
        getattr(hyperspy.utils, obj_name)


def test_import_components1D():
    import hyperspy.api as hs

    for obj_name in hs.model.components1D.__all__:
        getattr(hs.model.components1D, obj_name)


def test_import_components2D():
    import hyperspy.api as hs

    for obj_name in hs.model.components2D.__all__:
        getattr(hs.model.components2D, obj_name)


def test_import_signals():
    import hyperspy.api as hs

    for obj_name in hs.signals.__all__:
        getattr(hs.signals, obj_name)


def test_import_attribute_error():
    import hyperspy

    try:
        hyperspy.inexisting_module
    except AttributeError:
        pass


def test_import_api_attribute_error():
    import hyperspy.api

    try:
        hyperspy.api.inexisting_module
    except AttributeError:
        pass


def test_dir():
    import hyperspy

    d = dir(hyperspy)
    assert d == ["__version__", "api"]


def test_dir_api():
    import hyperspy.api

    d = dir(hyperspy.api)
    assert d == [
        "__version__",
        "data",
        "get_configuration_directory_path",
        "interactive",
        "load",
        "model",
        "plot",
        "preferences",
        "print_known_signal_types",
        "roi",
        "samfire",
        "set_log_level",
        "signals",
        "stack",
        "transpose",
    ]


def test_dir_data():
    import hyperspy.data

    d = dir(hyperspy.data)
    assert d == [
        "atomic_resolution_image",
        "luminescence_signal",
        "two_gaussians",
        "wave_image",
    ]


def test_dir_utils():
    import hyperspy.utils

    d = dir(hyperspy.utils)
    assert d == [
        "interactive",
        "markers",
        "model",
        "plot",
        "print_known_signal_types",
        "roi",
        "samfire",
        "stack",
        "transpose",
    ]


def test_dir_utils_markers():
    import hyperspy.utils.markers

    d = dir(hyperspy.utils.markers)
    assert d == [
        "Arrows",
        "Circles",
        "Ellipses",
        "HorizontalLines",
        "Lines",
        "Markers",
        "Points",
        "Polygons",
        "Rectangles",
        "Squares",
        "Texts",
        "VerticalLines",
    ]


def test_dir_utils_model():
    import hyperspy.utils.model

    d = dir(hyperspy.utils.model)
    assert d == [
        "components1D",
        "components2D",
    ]


def test_dir_utils_plot():
    import hyperspy.utils.plot

    d = dir(hyperspy.utils.plot)
    assert d == [
        "markers",
        "plot_histograms",
        "plot_images",
        "plot_roi_map",
        "plot_signals",
        "plot_spectra",
    ]


def test_dir_utils_roi():
    import hyperspy.utils.roi

    d = dir(hyperspy.utils.roi)
    assert d == [
        "CircleROI",
        "Line2DROI",
        "Point1DROI",
        "Point2DROI",
        "RectangularROI",
        "SpanROI",
    ]


def test_dir_utils_samfire():
    import hyperspy.utils.samfire

    d = dir(hyperspy.utils.samfire)
    assert d == [
        "SamfirePool",
        "fit_tests",
        "global_strategies",
        "local_strategies",
    ]


def test_dir_utils_samfire2():
    import hyperspy.utils.samfire

    d = dir(hyperspy.utils.samfire.fit_tests)
    assert d == [
        "AIC_test",
        "AICc_test",
        "BIC_test",
        "red_chisq_test",
    ]


def test_dir_utils_samfire3():
    import hyperspy.utils.samfire

    d = dir(hyperspy.utils.samfire.global_strategies)
    assert d == [
        "GlobalStrategy",
        "HistogramStrategy",
    ]


def test_dir_utils_samfire4():
    import hyperspy.utils.samfire

    d = dir(hyperspy.utils.samfire.local_strategies)
    assert d == [
        "LocalStrategy",
        "ReducedChiSquaredStrategy",
    ]


def test_pint_default_unit_registry():
    import pint

    import hyperspy.api as hs

    # the pint unit registry used by hyperspy must be the
    # same as pint default for interoperability reason
    # See https://github.com/hgrecco/pint/issues/108
    # and https://github.com/hgrecco/pint/issues/623
    assert id(hs._ureg) == id(pint.get_application_registry())
