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

"""Direct, independently checked files for expression deserialization."""

import hashlib
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy import __version__ as hs_version

SENTINEL = "hs3699_sentinel.txt"
EXPERIMENT = "Experiments/__unnamed__"
FA_GROUP = EXPERIMENT + "/axis-2"
COMPONENTS = "Analysis/models/audit/_dict/_list_2_components"
EXPRESSION_GROUP = COMPONENTS + "/1"
PARAMETERS = EXPRESSION_GROUP + "/_list_2_parameters"
FA_ATTRS = {
    "_type": "FunctionalDataAxis",
    "a": np.int64(100),
    "b": np.int64(10),
    "expression": "a / (x + 1) + b",
    "is_binned": np.False_,
    "name": "_None_",
    "navigate": np.False_,
    "size": np.int64(11),
    "units": "_None_",
}
EXPRESSION_ATTRS = {
    "_active_array": "_None_",
    "_id_name": "Expression",
    "_list_empty_linear_parameter_list": "_None_",
    "active": np.True_,
    "active_is_multidimensional": np.False_,
    "compute_gradients": np.True_,
    "expression": "a*x + b; b = 2*c",
    "module": "numpy",
    "name": "lin",
    "package": "hyperspy",
    "package_version": hs_version,
    "position": "_None_",
}
A_ATTRS = {
    "_id_name": "a",
    "_linear": np.True_,
    "_list_empty__twins": "_None_",
    "ext_bounded": np.False_,
    "ext_force_positive": np.False_,
    "free": np.False_,
    "name": "a",
    "std": "_None_",
    "twin_function_expr": "2*x + 1",
    "twin_inverse_function_expr": "",
    "units": "",
    "value": np.float64(3.0),
}
C_ATTRS = {
    "_id_name": "c",
    "_linear": np.True_,
    "_list_empty__twins": "_None_",
    "ext_bounded": np.False_,
    "ext_force_positive": np.False_,
    "free": np.True_,
    "name": "c",
    "std": "_None_",
    "twin_function_expr": "",
    "twin_inverse_function_expr": "",
    "units": "",
    "value": np.float64(0.5),
}
EXPRESSION_CHILDREN = {"_list_2_parameters", "_whitelist", "rename_pars"}
PARAMETER_CHILDREN = {"_tuple_2__bounds", "_whitelist", "map"}
PARAMETER_OBJECTS = ("", "/_tuple_2__bounds", "/_whitelist", "/map")
EXPERIMENT_OBJECTS = {"Experiments"} | {
    EXPERIMENT + name
    for name in (
        "",
        "/attributes",
        "/axis-0",
        "/axis-1",
        "/axis-2",
        "/data",
        "/learning_results",
        "/metadata",
        "/metadata/General",
        "/metadata/General/FileIO",
        "/metadata/General/FileIO/0",
        "/metadata/Signal",
        "/metadata/_HyperSpy",
        "/metadata/_HyperSpy/Folding",
        "/original_metadata",
    )
}
FA_OBJECTS = EXPERIMENT_OBJECTS | {FA_GROUP + "/x"}
MODEL_OBJECTS = (
    EXPERIMENT_OBJECTS
    | {"Analysis", "Analysis/models"}
    | {
        "Analysis/models/audit" + name
        for name in (
            "",
            "/_dict",
            "/_dict/_channel_switches",
            "/_dict/_list_2_components",
            "/_dict/_whitelist",
            "/_dict/chisq.data",
            "/_dict/dof.data",
            "/components",
            "/components/Gaussian",
            "/components/lin",
        )
    }
    | {COMPONENTS + "/0" + name for name in ("", "/_list_3_parameters", "/_whitelist")}
    | {
        COMPONENTS + "/0/_list_3_parameters/" + index + name
        for index in "012"
        for name in PARAMETER_OBJECTS
    }
    | {COMPONENTS + "/0/_list_3_parameters/0/_list__twins"}
    | {
        EXPRESSION_GROUP + name
        for name in ("", "/_list_2_parameters", "/_whitelist", "/rename_pars")
    }
    | {PARAMETERS + "/" + index + name for index in "01" for name in PARAMETER_OBJECTS}
)
# The writer stores id() of each live Parameter as its ``self`` attribute to
# reconnect twins on restore. The builder records those ids from its own
# objects, keyed by the saved path, so the expected value never comes from a file.
STORED_PARAMETER_IDS = {}
MODEL_CASES = (
    (
        "B1-expression-base",
        EXPRESSION_GROUP,
        "expression",
        "a*x + b; b = 2*c",
        "a*x + b + ",
        "; b = 2*c",
    ),
    (
        "B2-expression-substitution-rhs",
        EXPRESSION_GROUP,
        "expression",
        "a*x + b; b = 2*c",
        "a*x + b; b = 2*c + ",
        "",
    ),
    (
        "B3-twin-function",
        PARAMETERS + "/0",
        "twin_function_expr",
        "2*x + 1",
        "2*x + 1 + ",
        "",
    ),
    (
        "B4-twin-inverse",
        PARAMETERS + "/0",
        "twin_inverse_function_expr",
        "",
        "(x - 1)/2 + ",
        "",
    ),
)


def payload(sentinel_path) -> str:
    return (
        "0*(__import__('pathlib').Path("
        f"{str(sentinel_path)!r}).write_text('hs3699') or 0)"
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_attrs(path, group) -> dict:
    with h5py.File(path, "r") as f:
        return dict(f[group].attrs)


def _children(path, group) -> set[str]:
    with h5py.File(path, "r") as f:
        return set(f[group])


def _object_names(path):
    with h5py.File(path, "r") as f:
        names = set()
        f.visit(names.add)
        return names


def _assert_attrs(actual, expected):
    assert actual.keys() == expected.keys()
    for name, literal in expected.items():
        assert type(actual[name]) is type(literal), name
        if isinstance(literal, np.floating):
            np.testing.assert_allclose(actual[name], literal, rtol=0, atol=0)
        else:
            assert actual[name] == literal, name


def _tamper_attr(src, dst, group, name, expected_old, new) -> None:
    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        assert group in f, group
        attrs = f[group].attrs
        assert name in attrs, name
        assert type(attrs[name]) is type(expected_old)
        assert attrs[name] == expected_old
        assert type(new) is type(expected_old)
        f[group].attrs[name] = new


def _tamper_vlen_element(
    src, dst, dataset, index, expected_old_bytes, new_bytes
) -> None:
    assert dataset.rsplit("/", 1)[-1] == "_tuple_position"
    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        assert dataset in f
        values = f[dataset]
        assert values.shape == (2,)
        assert values.dtype == np.dtype("O")
        assert dict(values.attrs) == {}
        assert 0 <= index < 2
        assert type(expected_old_bytes) is bytes
        assert type(new_bytes) is bytes
        assert type(values[index]) is bytes
        assert values[index] == expected_old_bytes
        values[index] = new_bytes


def build_functional_axis_file(tmp_path) -> Path:
    data = np.arange(5 * 7 * 11, dtype=float).reshape(5, 7, 11)
    s = hs.signals.Signal1D(data)
    s.axes_manager.signal_axes[0].convert_to_functional_data_axis(
        "a / (x + 1) + b", a=100, b=10
    )
    path = tmp_path / "fa_control.hspy"
    s.save(path)
    return path


def build_stored_model_file(tmp_path) -> Path:
    data = np.arange(5 * 7 * 11, dtype=float).reshape(5, 7, 11) / 100 + 1
    s = hs.signals.Signal1D(data)
    m = s.create_model()
    g = hs.model.components1D.Gaussian()
    e = hs.model.components1D.Expression("a*x + b; b = 2*c", name="lin", a=2.0, c=0.5)
    m.append(g)
    m.append(e)
    e.a.twin = g.A
    e.a.twin_function_expr = "2*x + 1"
    m.store("audit")
    path = tmp_path / "model_control.hspy"
    s.save(path)
    STORED_PARAMETER_IDS[path] = {"0": id(e.a), "1": id(e.c)}
    return path


def assert_functional_axis_truth(path):
    _assert_attrs(_read_attrs(path, FA_GROUP), FA_ATTRS)
    assert _children(path, FA_GROUP) == {"x"}
    for group, size, navigate in (
        (FA_GROUP + "/x", 11, np.False_),
        (EXPERIMENT + "/axis-0", 5, np.True_),
        (EXPERIMENT + "/axis-1", 7, np.True_),
    ):
        _assert_attrs(
            _read_attrs(path, group),
            {
                "_type": "UniformDataAxis",
                "is_binned": np.False_,
                "name": "_None_",
                "navigate": navigate,
                "offset": np.float64(0),
                "scale": np.float64(1),
                "size": np.int64(size),
                "units": "_None_",
            },
        )
        assert _children(path, group) == set()
    assert _object_names(path) == FA_OBJECTS
    with h5py.File(path, "r") as f:
        data = f[EXPERIMENT + "/data"]
        assert data.shape == (5, 7, 11)
        assert data.dtype == np.dtype("float64")
        np.testing.assert_array_equal(
            data[:], np.arange(5 * 7 * 11, dtype=float).reshape(5, 7, 11)
        )


def assert_stored_model_truth(path):
    _assert_attrs(_read_attrs(path, EXPRESSION_GROUP), EXPRESSION_ATTRS)
    assert _children(path, EXPRESSION_GROUP) == EXPRESSION_CHILDREN
    assert _children(path, PARAMETERS) == {"0", "1"}
    assert _children(path, EXPRESSION_GROUP + "/rename_pars") == set()
    assert _read_attrs(path, EXPRESSION_GROUP + "/rename_pars") == {}
    gaussian = _read_attrs(path, COMPONENTS + "/0")
    assert type(gaussian["_id_name"]) is str
    assert gaussian["_id_name"] == "Gaussian"
    identities = STORED_PARAMETER_IDS[path]
    for index, expected in (("0", A_ATTRS), ("1", C_ATTRS)):
        _assert_attrs(
            _read_attrs(path, PARAMETERS + "/" + index),
            {**expected, "self": np.int64(identities[index])},
        )
        assert _children(path, PARAMETERS + "/" + index) == PARAMETER_CHILDREN
    assert _object_names(path) == MODEL_OBJECTS
    _assert_attrs(
        _read_attrs(path, EXPRESSION_GROUP + "/_whitelist"),
        {
            "_active_array": "",
            "_id_name": "",
            "active": "",
            "active_is_multidimensional": "",
            "compute_gradients": "init",
            "expression": "init",
            "linear_parameter_list": "init",
            "module": "init",
            "name": "init",
            "position": "init",
            "rename_pars": "init",
        },
    )
    with h5py.File(path, "r") as f:
        data = f[EXPERIMENT + "/data"]
        assert data.shape == (5, 7, 11)
        assert data.dtype == np.dtype("float64")
        np.testing.assert_allclose(
            data[:], np.arange(5 * 7 * 11, dtype=float).reshape(5, 7, 11) / 100 + 1
        )


def test_functional_axis_truth(tmp_path):
    assert_functional_axis_truth(build_functional_axis_file(tmp_path))


def test_stored_model_truth(tmp_path):
    assert_stored_model_truth(build_stored_model_file(tmp_path))


@pytest.mark.parametrize("lazy", (False, True))
def test_functional_axis_control(tmp_path, lazy):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    path = build_functional_axis_file(tmp_path)
    assert_functional_axis_truth(path)
    digest = _sha256(path)
    loaded = hs.load(path, lazy=lazy)
    try:
        assert loaded._lazy is lazy
        np.testing.assert_allclose(
            loaded.axes_manager.signal_axes[0].axis,
            100 / (np.arange(11) + 1) + 10,
        )
        assert not sentinel.exists()
    finally:
        if lazy:
            loaded.close_file()
        assert _sha256(path) == digest
        assert_functional_axis_truth(path)


@pytest.mark.parametrize("lazy", (False, True))
def test_stored_model_control(tmp_path, lazy):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    path = build_stored_model_file(tmp_path)
    assert_stored_model_truth(path)
    digest = _sha256(path)
    loaded = hs.load(path, lazy=lazy)
    try:
        assert loaded._lazy is lazy
        assert not sentinel.exists()
        model = loaded.models.restore("audit")
        assert model[1].a.twin is model[0].A
        assert model[1].a.twin_function_expr == "2*x + 1"
        np.testing.assert_allclose(
            model[1].function(np.arange(11.0)), 3 * np.arange(11.0) + 1
        )
        assert not sentinel.exists()
    finally:
        if lazy:
            loaded.close_file()
        assert _sha256(path) == digest
        assert_stored_model_truth(path)


def test_payload_is_effective_when_evaluated(tmp_path):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    # Only this code-owned benign string is evaluated, never file-derived text.
    result = eval(payload(sentinel), {"__builtins__": __builtins__}, {})
    assert result == 0
    assert sentinel.read_text() == "hs3699"


@pytest.mark.parametrize("fault", ("old", "type", "group"))
def test_negative_attribute(tmp_path, fault):
    path = build_functional_axis_file(tmp_path)
    assert_functional_axis_truth(path)
    digest = _sha256(path)
    group, name, old, new = FA_GROUP, "expression", "wrong", "new"
    if fault == "type":
        name, old, new = "a", np.float64(100), np.float64(101)
    elif fault == "group":
        group, old = "missing-group", "a / (x + 1) + b"
    with pytest.raises(AssertionError):
        _tamper_attr(path, tmp_path / "wrong.hspy", group, name, old, new)
    assert _sha256(path) == digest
    assert_functional_axis_truth(path)


@pytest.mark.parametrize("fault", ("old", "index", "none"))
def test_vlen_control_negative(tmp_path, fault):
    path = tmp_path / "position_control.hspy"
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "_tuple_position", data=["x0", "y0"], dtype=h5py.string_dtype()
        )
    with h5py.File(path, "r") as f:
        values = f["_tuple_position"]
        assert values.shape == (2,)
        assert values.dtype == np.dtype("O")
        assert dict(values.attrs) == {}
        assert list(values[:]) == [b"x0", b"y0"]
        assert all(type(value) is bytes for value in values[:])
    digest = _sha256(path)
    dst = tmp_path / "position_copy.hspy"
    if fault == "none":
        _tamper_vlen_element(path, dst, "_tuple_position", 0, b"x0", b"x1")
        with h5py.File(dst, "r") as f:
            assert list(f["_tuple_position"][:]) == [b"x1", b"y0"]
    else:
        with pytest.raises(AssertionError):
            _tamper_vlen_element(
                path,
                dst,
                "_tuple_position",
                2 if fault == "index" else 0,
                b"wrong" if fault == "old" else b"x0",
                b"x1",
            )
    assert _sha256(path) == digest


@pytest.mark.parametrize("lazy", (False, True))
def test_functional_axis_attack(tmp_path, lazy):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = build_functional_axis_file(tmp_path)
    assert_functional_axis_truth(control)
    digest = _sha256(control)
    dst = tmp_path / "fa_tampered.hspy"
    new = "a / (x + 1) + b + " + payload(sentinel)
    _tamper_attr(control, dst, FA_GROUP, "expression", "a / (x + 1) + b", new)
    loaded = None
    try:
        _assert_attrs(_read_attrs(dst, FA_GROUP), {**FA_ATTRS, "expression": new})
        assert _children(dst, FA_GROUP) == {"x"}
        assert _object_names(dst) == FA_OBJECTS
        with pytest.raises(ValueError):
            loaded = hs.load(dst, lazy=lazy)
        assert not sentinel.exists()
    finally:
        print(
            f"case=A1 lazy={lazy} sentinel_exists={sentinel.exists()} sentinel_text={sentinel.read_text() if sentinel.exists() else None!r}"
        )
        if loaded is not None and lazy:
            loaded.close_file()
        assert _sha256(control) == digest, "control hash changed"
        assert_functional_axis_truth(control)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("case", MODEL_CASES, ids=[case[0] for case in MODEL_CASES])
def test_stored_model_attack(tmp_path, lazy, case):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = build_stored_model_file(tmp_path)
    assert_stored_model_truth(control)
    digest = _sha256(control)
    case_id, group, name, old, prefix, suffix = case
    new = prefix + payload(sentinel) + suffix
    dst = tmp_path / "model_tampered.hspy"
    _tamper_attr(control, dst, group, name, old, new)
    loaded = None
    try:
        if group == EXPRESSION_GROUP:
            expected = {**EXPRESSION_ATTRS, name: new}
            children = EXPRESSION_CHILDREN
        else:
            expected = {
                **A_ATTRS,
                "self": np.int64(STORED_PARAMETER_IDS[control]["0"]),
                name: new,
            }
            children = PARAMETER_CHILDREN
        _assert_attrs(_read_attrs(dst, group), expected)
        assert _children(dst, group) == children
        assert _object_names(dst) == MODEL_OBJECTS
        loaded = hs.load(dst, lazy=lazy)
        assert not sentinel.exists()
        with pytest.raises(ValueError):
            loaded.models.restore("audit")
        assert not sentinel.exists()
    finally:
        print(
            f"case={case_id} lazy={lazy} sentinel_exists={sentinel.exists()} sentinel_text={sentinel.read_text() if sentinel.exists() else None!r}"
        )
        if loaded is not None and lazy:
            loaded.close_file()
        assert _sha256(control) == digest, "control hash changed"
        assert_stored_model_truth(control)


# --- Task 3: file-derived expressions at every sink -------------------------

ROUTE_AXIS_ATTRS = {
    "_type": "FunctionalDataAxis",
    "a": np.int64(3),
    "b": np.int64(10),
    "expression": "a * x + b",
    "is_binned": np.False_,
    "name": "_None_",
    "navigate": np.False_,
    "size": np.int64(7),
    "units": "_None_",
}
ROUTE_X_ATTRS = {
    "_type": "UniformDataAxis",
    "is_binned": np.False_,
    "name": "_None_",
    "navigate": np.False_,
    "offset": np.float64(0),
    "scale": np.float64(1),
    "size": np.int64(7),
    "units": "_None_",
}
MAIN_AXIS_ATTRS = {
    "_type": "FunctionalDataAxis",
    "a": np.int64(100),
    "b": np.int64(10),
    "expression": "a / (x + 1) + b",
    "is_binned": np.False_,
    "name": "_None_",
    "navigate": np.False_,
    "size": np.int64(11),
    "units": "_None_",
}
MAIN_X_ATTRS = {**ROUTE_X_ATTRS, "size": np.int64(11)}
ROUTE_CASES = (
    "main-axis",
    "metadata-axis",
    "metadata-axes-manager",
    "metadata-signal",
    "original-metadata-signal",
    "component-signal",
)
ROUTE_TARGETS = {
    "main-axis": (EXPERIMENT + "/axis-2", "expression", "a / (x + 1) + b"),
    "metadata-axis": (
        EXPERIMENT + "/metadata/Fixture/_hspy_Axis_fax",
        "expression",
        "a * x + b",
    ),
    "metadata-axes-manager": (
        EXPERIMENT + "/metadata/Fixture/_list_1__hspy_AxesManager_am/0",
        "expression",
        "a * x + b",
    ),
    "metadata-signal": (
        EXPERIMENT + "/metadata/Fixture/_sig_nested/_list_1_axes/0",
        "expression",
        "a * x + b",
    ),
    "original-metadata-signal": (
        EXPERIMENT + "/original_metadata/Fixture/_sig_nested/_list_1_axes/0",
        "expression",
        "a * x + b",
    ),
    "component-signal": (
        "Analysis/models/sfp/_dict/_list_1_components/0/signal1D/_list_1_axes/0",
        "expression",
        "a * x + b",
    ),
}


def _functional_axis(size=7):
    from hyperspy.axes import FunctionalDataAxis

    return FunctionalDataAxis("a * x + b", a=3, b=10, size=size)


def _route_host():
    return hs.signals.Signal1D(
        np.arange(5 * 7 * 11, dtype=float).reshape(5, 7, 11) / 100 + 1
    )


def build_main_axis_route(tmp_path, suffix=".hspy"):
    s = hs.signals.Signal1D(np.arange(5 * 7 * 11, dtype=float).reshape(5, 7, 11))
    s.axes_manager.signal_axes[0].convert_to_functional_data_axis(
        "a / (x + 1) + b", a=100, b=10
    )
    path = tmp_path / f"route_main_axis{suffix}"
    s.save(path)
    return path


def build_metadata_axis_route(tmp_path, suffix=".hspy"):
    s = _route_host()
    s.metadata.add_node("Fixture")
    s.metadata.Fixture.fax = _functional_axis()
    path = tmp_path / f"route_metadata_axis{suffix}"
    s.save(path)
    return path


def build_metadata_axes_manager_route(tmp_path, suffix=".hspy"):
    from hyperspy.axes import AxesManager

    s = _route_host()
    s.metadata.add_node("Fixture")
    s.metadata.Fixture.am = AxesManager([_functional_axis()])
    path = tmp_path / f"route_metadata_axes_manager{suffix}"
    s.save(path)
    return path


def build_metadata_signal_route(tmp_path, suffix=".hspy"):
    s = _route_host()
    nested = hs.signals.Signal1D(np.arange(7.0))
    nested.axes_manager.signal_axes[0].convert_to_functional_data_axis(
        "a * x + b", a=3, b=10
    )
    s.metadata.add_node("Fixture")
    s.metadata.Fixture.nested = nested
    path = tmp_path / f"route_metadata_signal{suffix}"
    s.save(path)
    return path


def build_original_metadata_signal_route(tmp_path, suffix=".hspy"):
    s = _route_host()
    nested = hs.signals.Signal1D(np.arange(7.0))
    nested.axes_manager.signal_axes[0].convert_to_functional_data_axis(
        "a * x + b", a=3, b=10
    )
    s.original_metadata.add_node("Fixture")
    s.original_metadata.Fixture.nested = nested
    path = tmp_path / f"route_original_metadata_signal{suffix}"
    s.save(path)
    return path


def build_component_signal_route(tmp_path, suffix=".hspy"):
    s = _route_host()
    reference = hs.signals.Signal1D(np.arange(7.0) + 1)
    reference.axes_manager.signal_axes[0].convert_to_functional_data_axis(
        "a * x + b", a=3, b=10
    )
    model = s.create_model()
    model.append(hs.model.components1D.ScalableFixedPattern(reference))
    model.store("sfp")
    path = tmp_path / f"route_component_signal{suffix}"
    s.save(path)
    return path


_route_builders = {
    "main-axis": build_main_axis_route,
    "metadata-axis": build_metadata_axis_route,
    "metadata-axes-manager": build_metadata_axes_manager_route,
    "metadata-signal": build_metadata_signal_route,
    "original-metadata-signal": build_original_metadata_signal_route,
    "component-signal": build_component_signal_route,
}


def _route_expected_attrs(route):
    return MAIN_AXIS_ATTRS if route == "main-axis" else ROUTE_AXIS_ATTRS


def _route_expected_axis(route):
    if route == "main-axis":
        return 100 / (np.arange(11) + 1) + 10
    return 3 * np.arange(7) + 10


def assert_route_file_truth(path, route):
    group = ROUTE_TARGETS[route][0]
    _assert_attrs(_read_attrs(path, group), _route_expected_attrs(route))
    assert _children(path, group) == {"x"}
    _assert_attrs(
        _read_attrs(path, group + "/x"),
        MAIN_X_ATTRS if route == "main-axis" else ROUTE_X_ATTRS,
    )


def assert_route_loaded_axis(route, loaded):
    if route == "main-axis":
        axis = loaded.axes_manager.signal_axes[0].axis
    elif route == "metadata-axis":
        axis = loaded.metadata.Fixture.fax.axis
    elif route == "metadata-axes-manager":
        axis = loaded.metadata.Fixture.am[0].axis
    elif route == "metadata-signal":
        axis = loaded.metadata.Fixture.nested.axes_manager.signal_axes[0].axis
    elif route == "original-metadata-signal":
        axis = loaded.original_metadata.Fixture.nested.axes_manager.signal_axes[0].axis
    else:
        model = loaded.models.restore("sfp")
        assert type(model[0]).__name__ == "ScalableFixedPattern"
        axis = model[0].signal.axes_manager.signal_axes[0].axis
    np.testing.assert_allclose(axis, _route_expected_axis(route))


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("route", ROUTE_CASES)
def test_route_control(tmp_path, lazy, route):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = _route_builders[route](tmp_path)
    assert_route_file_truth(control, route)
    digest = _sha256(control)
    loaded = hs.load(control, lazy=lazy)
    try:
        assert loaded._lazy is lazy
        assert_route_loaded_axis(route, loaded)
        assert not sentinel.exists()
    finally:
        print(f"case={route}-control lazy={lazy} sentinel_exists={sentinel.exists()}")
        if lazy:
            loaded.close_file()
        assert _sha256(control) == digest, "control hash changed"
        assert_route_file_truth(control, route)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("route", ROUTE_CASES)
def test_route_attack(tmp_path, lazy, route):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = _route_builders[route](tmp_path)
    assert_route_file_truth(control, route)
    digest = _sha256(control)
    group, name, old = ROUTE_TARGETS[route]
    new = old + " + " + payload(sentinel)
    dst = tmp_path / f"route_{route}_tampered.hspy"
    _tamper_attr(control, dst, group, name, old, new)
    loaded = None
    try:
        _assert_attrs(
            _read_attrs(dst, group), {**_route_expected_attrs(route), name: new}
        )
        assert _children(dst, group) == {"x"}
        if route == "component-signal":
            loaded = hs.load(dst, lazy=lazy)
            assert not sentinel.exists()
            with pytest.raises(ValueError):
                loaded.models.restore("sfp")
        else:
            with pytest.raises(ValueError):
                loaded = hs.load(dst, lazy=lazy)
        assert not sentinel.exists()
    finally:
        print(
            f"case={route}-attack lazy={lazy} sentinel_exists={sentinel.exists()} "
            f"sentinel_text={sentinel.read_text() if sentinel.exists() else None!r}"
        )
        if loaded is not None and lazy:
            loaded.close_file()
        assert _sha256(control) == digest, "control hash changed"
        assert_route_file_truth(control, route)


POSITION_COMPONENT = "Analysis/models/m2/_dict/_list_1_components/0"
POSITION_ARRAY = POSITION_COMPONENT + "/_tuple_position"
POSITION_EXPRESSION = "A*exp(-((x-x0)**2+(y-y0)**2)/(2*s**2)); s = _s"


def _position_component():
    return hs.model.components2D.Expression(
        POSITION_EXPRESSION,
        name="g2",
        add_rotation=True,
        position=("x0", "y0"),
        rename_pars={"_s": "sigma"},
    )


def build_position_model_file(tmp_path, suffix=".hspy"):
    s = hs.signals.Signal2D(np.random.default_rng(3699).random((6, 5, 13)))
    model = s.create_model()
    model.append(_position_component())
    model.store("m2")
    path = tmp_path / f"position_model{suffix}"
    s.save(path)
    return path


def assert_position_file_truth(path):
    attrs = _read_attrs(path, POSITION_COMPONENT)
    assert attrs["_id_name"] == "Expression"
    assert attrs["name"] == "g2"
    assert attrs["add_rotation"] == np.True_
    assert attrs["expression"] == POSITION_EXPRESSION
    assert type(attrs["add_rotation"]) is np.bool_
    assert _read_attrs(path, POSITION_COMPONENT + "/rename_pars") == {"_s": "sigma"}
    with h5py.File(path, "r") as f:
        values = f[POSITION_ARRAY]
        assert values.shape == (2,)
        assert values.dtype == np.dtype("O")
        assert dict(values.attrs) == {}
        assert list(values[:]) == [b"x0", b"y0"]
        assert all(type(value) is bytes for value in values[:])


@pytest.mark.parametrize("lazy", (False, True))
def test_position_stored_control(tmp_path, lazy):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = build_position_model_file(tmp_path)
    assert_position_file_truth(control)
    digest = _sha256(control)
    loaded = hs.load(control, lazy=lazy)
    try:
        assert loaded._lazy is lazy
        model = loaded.models.restore("m2")
        component = model[0]
        assert component._position_x is component.x0
        assert component._position_y is component.y0
        x, y = np.meshgrid(
            np.linspace(-2.0, 3.0, 11),
            np.linspace(-1.5, 2.5, 7),
            indexing="ij",
        )
        np.testing.assert_allclose(
            component.function(x, y), _position_component().function(x, y)
        )
        assert not sentinel.exists()
    finally:
        if lazy:
            loaded.close_file()
        assert _sha256(control) == digest, "control hash changed"
        assert_position_file_truth(control)


@pytest.mark.parametrize("lazy", (False, True))
def test_position_stored_attack(tmp_path, lazy):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = build_position_model_file(tmp_path)
    assert_position_file_truth(control)
    digest = _sha256(control)
    new = payload(sentinel).encode()
    dst = tmp_path / "position_tampered.hspy"
    _tamper_vlen_element(control, dst, POSITION_ARRAY, 1, b"y0", new)
    with h5py.File(dst, "r") as f:
        values = f[POSITION_ARRAY]
        assert values.shape == (2,)
        assert values.dtype == np.dtype("O")
        assert dict(values.attrs) == {}
        assert list(values[:]) == [b"x0", new]
    loaded = None
    try:
        loaded = hs.load(dst, lazy=lazy)
        assert not sentinel.exists(), "restore must be the trigger"
        with pytest.raises(ValueError):
            loaded.models.restore("m2")
        assert not sentinel.exists()
    finally:
        print(
            f"case=C3-position lazy={lazy} sentinel_exists={sentinel.exists()} "
            f"sentinel_text={sentinel.read_text() if sentinel.exists() else None!r}"
        )
        if loaded is not None and lazy:
            loaded.close_file()
        assert _sha256(control) == digest, "control hash changed"
        assert_position_file_truth(control)


def test_rotation_center_constructor_attack(tmp_path):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    with pytest.raises(ValueError) as info:
        hs.model.components2D.Expression(
            POSITION_EXPRESSION,
            name="g2",
            add_rotation=True,
            rotation_center=("1", payload(sentinel)),
        )
    assert info.value.rule == "R-IDENT-UNICODE"
    assert not sentinel.exists()
    print(f"case=rotation_center-attack sentinel_exists={sentinel.exists()}")


def test_rotation_center_constructor_control():
    rotated = hs.model.components2D.Expression(
        POSITION_EXPRESSION,
        name="g2",
        add_rotation=True,
        rotation_center=(1, 2),
    )
    plain = hs.model.components2D.Expression(POSITION_EXPRESSION, name="g2")
    assert [p.name for p in rotated.parameters] == [
        "A",
        "_s",
        "rotation_angle",
        "x0",
        "y0",
    ]
    x, y = np.meshgrid(
        np.linspace(-2.0, 3.0, 11), np.linspace(-1.5, 2.5, 7), indexing="ij"
    )
    np.testing.assert_allclose(rotated.function(x, y), plain.function(x, y))


def test_rename_pars_payload_constructor_attack(tmp_path):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    with pytest.raises(ValueError) as info:
        hs.model.components1D.Expression(
            "a*x + b", name="e", rename_pars={"b": payload(sentinel)}
        )
    assert info.value.rule == "R-IDENT-UNICODE"
    assert not sentinel.exists()


def test_rename_pars_dunder_constructor_attack(tmp_path):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    with pytest.raises(ValueError) as info:
        hs.model.components1D.Expression(
            "a*x + b", name="e", rename_pars={"b": "__x__"}
        )
    assert info.value.rule == "R-IDENT-DUNDER"
    assert not sentinel.exists()


def test_position_payload_constructor_attack(tmp_path):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    with pytest.raises(ValueError) as info:
        hs.model.components1D.Expression("a*x", name="e", position=payload(sentinel))
    assert info.value.rule == "R-IDENT-UNICODE"
    assert not sentinel.exists()


def test_default_none_constructor_control():
    component = hs.model.components1D.Expression(
        "a*x",
        name="e",
        position=None,
        rotation_center=None,
        rename_pars=None,
    )
    assert [p.name for p in component.parameters] == ["a"]
    assert component.a.value == 0.0
    x = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(component.function(x), np.zeros_like(x))


def test_shipped_exponential_default_control():
    component = hs.model.components1D.Exponential()
    assert [p.name for p in component.parameters] == ["A", "tau"]
    x = np.linspace(0.0, 3.0, 11)
    np.testing.assert_allclose(component.function(x), np.exp(-x))


FXC_COMPONENT = "Analysis/models/m1/_dict/_list_1_components/0"
FXC_PARAMETERS = FXC_COMPONENT + "/_list_2_parameters"


def build_display_name_model_file(tmp_path, free_b=True):
    data = np.arange(5 * 7 * 11, dtype=float).reshape(5, 7, 11) / 100 + 1
    s = hs.signals.Signal1D(data)
    model = s.create_model()
    component = hs.model.components1D.Expression("a*x + b", name="lin", a=1, b=0)
    if not free_b:
        component.b.free = False
    model.append(component)
    model.store("m1")
    path = tmp_path / (
        "display_names_free.hspy" if free_b else "display_names_fixed.hspy"
    )
    s.save(path)
    return path


def assert_display_name_file_truth(path, free_b=True):
    for index, name in (("0", "a"), ("1", "b")):
        attrs = _read_attrs(path, FXC_PARAMETERS + "/" + index)
        assert attrs["_id_name"] == name
        assert attrs["name"] == name
        assert attrs["_linear"] == np.True_
    if free_b:
        assert _read_attrs(path, FXC_PARAMETERS + "/0")["free"] == np.True_
        assert _read_attrs(path, FXC_PARAMETERS + "/1")["free"] == np.True_
    else:
        assert _read_attrs(path, FXC_PARAMETERS + "/0")["free"] == np.True_
        assert _read_attrs(path, FXC_PARAMETERS + "/1")["free"] == np.False_


def _restore_and_fit(path, optimizer):
    loaded = hs.load(path)
    model = loaded.models.restore("m1")
    model.fit(optimizer=optimizer, print_info=False)
    return model


def _fitted_values(model):
    return np.array([model[0].a.value, model[0].b.value], dtype=float)


def test_restore_then_fit_two_free_control(tmp_path):
    control = build_display_name_model_file(tmp_path)
    assert_display_name_file_truth(control)
    model = _restore_and_fit(control, "lstsq")
    np.testing.assert_allclose(
        _fitted_values(model), np.array([0.01, 1.0]), rtol=1e-8, atol=0
    )


def test_restore_then_fit_two_free_attack(tmp_path):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = build_display_name_model_file(tmp_path)
    assert_display_name_file_truth(control)
    control_model = _restore_and_fit(control, "lstsq")
    control_values = _fitted_values(control_model)
    new = "c + " + payload(sentinel)
    dst = tmp_path / "display_names_two_free.hspy"
    _tamper_attr(control, dst, FXC_PARAMETERS + "/1", "name", "b", new)
    loaded = hs.load(dst)
    model = loaded.models.restore("m1")
    names = [p.name for p in model[0].parameters]
    assert names == ["a", new]
    assert sorted(names) == names
    assert not sentinel.exists()
    model.fit(optimizer="lstsq", print_info=False)
    assert hasattr(model, "fit_output")
    assert not sentinel.exists()
    np.testing.assert_allclose(_fitted_values(model), control_values, rtol=1e-8, atol=0)
    print(f"case=C4-name-two-free sentinel_exists={sentinel.exists()}")


def test_restore_then_fit_one_free_control(tmp_path):
    control = build_display_name_model_file(tmp_path, free_b=False)
    assert_display_name_file_truth(control, free_b=False)
    model = _restore_and_fit(control, "lstsq")
    assert model[0].b.free is False
    assert np.isfinite(_fitted_values(model)).all()


def test_restore_then_fit_one_free_attack(tmp_path):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = build_display_name_model_file(tmp_path, free_b=False)
    assert_display_name_file_truth(control, free_b=False)
    control_model = _restore_and_fit(control, "lstsq")
    control_values = _fitted_values(control_model)
    new = "c + " + payload(sentinel)
    dst = tmp_path / "display_names_one_free.hspy"
    _tamper_attr(control, dst, FXC_PARAMETERS + "/0", "name", "a", new)
    loaded = hs.load(dst)
    model = loaded.models.restore("m1")
    free_parameters = [p for p in model[0].parameters if p.free]
    assert [p.name for p in free_parameters] == [new]
    assert not sentinel.exists()
    model.fit(optimizer="lstsq", print_info=False)
    assert hasattr(model, "fit_output")
    assert not sentinel.exists()
    np.testing.assert_allclose(_fitted_values(model), control_values, rtol=1e-8, atol=0)
    print(f"case=C5-name-one-free sentinel_exists={sentinel.exists()}")


def test_restore_then_fit_sorts_first_attack(tmp_path):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = build_display_name_model_file(tmp_path)
    assert_display_name_file_truth(control)
    new = payload(sentinel)
    dst = tmp_path / "display_names_sorts_first.hspy"
    _tamper_attr(control, dst, FXC_PARAMETERS + "/1", "name", "b", new)
    loaded = hs.load(dst)
    model = loaded.models.restore("m1")
    assert [p.name for p in model[0].parameters] == ["a", new]
    assert sorted([p.name for p in model[0].parameters]) != [
        p.name for p in model[0].parameters
    ]
    assert not sentinel.exists()
    model.fit(optimizer="lstsq", print_info=False)
    assert hasattr(model, "fit_output")
    assert not sentinel.exists()
    assert np.isfinite(_fitted_values(model)).all()
    print(f"case=C4b-name-sorts-first sentinel_exists={sentinel.exists()}")


def test_restore_then_fit_nonlinear_control(tmp_path):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = build_display_name_model_file(tmp_path)
    assert_display_name_file_truth(control)
    control_model = _restore_and_fit(control, "lm")
    control_values = _fitted_values(control_model)
    new = "c + " + payload(sentinel)
    dst = tmp_path / "display_names_nonlinear.hspy"
    _tamper_attr(control, dst, FXC_PARAMETERS + "/1", "name", "b", new)
    loaded = hs.load(dst)
    model = loaded.models.restore("m1")
    assert [p.name for p in model[0].parameters] == ["a", new]
    assert not sentinel.exists(), "restore must not parse display names"
    model.fit(optimizer="lm", print_info=False)
    assert hasattr(model, "fit_output")
    assert not sentinel.exists()
    np.testing.assert_allclose(_fitted_values(model), control_values, rtol=1e-6, atol=0)
    print(f"case=C6-name-nonlinear sentinel_exists={sentinel.exists()}")


def test_parameter_linearity_payload_attack(tmp_path):
    from hyperspy._components.expression import _check_parameter_linearity

    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    with pytest.raises(ValueError):
        _check_parameter_linearity("a + " + payload(sentinel), "a")
    assert not sentinel.exists()
    print(f"case=linearity-payload sentinel_exists={sentinel.exists()}")


def test_parameter_linearity_power_attack():
    from hyperspy._components.expression import _check_parameter_linearity
    from hyperspy.tests.misc.test_sympy_parse import tripwires

    with tripwires():
        with pytest.raises(ValueError) as info:
            _check_parameter_linearity("a*2**1025", "a")
    assert info.value.rule == "R-CONST-POW"


def test_parameter_linearity_name_attack(tmp_path):
    from hyperspy._components.expression import _check_parameter_linearity

    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    with pytest.raises(ValueError) as info:
        _check_parameter_linearity("a*x", payload(sentinel))
    assert info.value.rule == "R-IDENT-UNICODE"
    assert not sentinel.exists()


def test_parameter_linearity_existing_string_cases_control():
    from hyperspy._components.expression import _check_parameter_linearity

    expr = "a*x**2 + b*x + c"
    assert _check_parameter_linearity(expr, "a") is True
    assert _check_parameter_linearity(expr, "b") is True
    assert _check_parameter_linearity(expr, "c") is True

    expr = "a*sin(b*x)"
    assert _check_parameter_linearity(expr, "a") is True
    assert _check_parameter_linearity(expr, "b") is False

    expr = "a*exp(-b*x)"
    assert _check_parameter_linearity(expr, "a") is True
    assert _check_parameter_linearity(expr, "b") is False

    expr = "where(x > 10, a*sin(b*x), 0)"
    with pytest.warns(UserWarning):
        _check_parameter_linearity(expr, "a")
    with pytest.warns(UserWarning):
        _check_parameter_linearity(expr, "b")


def test_twin_sin_inversion_attack_is_bounded():
    # Deselected on the unfixed tree: the legacy solveset call does not return.
    from hyperspy.component import Parameter

    twin = Parameter()
    parameter = Parameter()
    parameter.twin = twin
    parameter.twin_function_expr = "sin(x)"
    assert parameter.twin_function_expr == "sin(x)"
    assert parameter._twin_inverse_sympy is None


def test_twin_rejected_value_attack(tmp_path):
    from hyperspy.component import Parameter

    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    twin = Parameter()
    parameter = Parameter()
    parameter.twin = twin
    parameter.twin_function_expr = "2*x + 1"
    before = (
        parameter.twin_function_expr,
        parameter._twin_inverse_sympy,
        parameter._twin_function,
        parameter.twin_inverse_function_expr,
    )
    with pytest.raises(ValueError):
        parameter.twin_function_expr = "2*x + " + payload(sentinel)
    assert not sentinel.exists()
    after = (
        parameter.twin_function_expr,
        parameter._twin_inverse_sympy,
        parameter._twin_function,
        parameter.twin_inverse_function_expr,
    )
    assert after[0] == before[0]
    assert after[1] is before[1]
    assert after[2] is before[2]
    assert after[3] == before[3]
    assert parameter.twin_function_expr == "2*x + 1"
    print(f"case=twin-rejected sentinel_exists={sentinel.exists()}")


# --- Task 3: native .zspy lane ---------------------------------------------

ZSPY_ROUTE_AXIS_ATTRS = {
    "_type": "FunctionalDataAxis",
    "a": 3,
    "b": 10,
    "expression": "a * x + b",
    "is_binned": False,
    "name": "_None_",
    "navigate": False,
    "size": 7,
    "units": "_None_",
}
ZSPY_ROUTE_X_ATTRS = {
    "_type": "UniformDataAxis",
    "is_binned": False,
    "name": "_None_",
    "navigate": False,
    "offset": 0.0,
    "scale": 1.0,
    "size": 7,
    "units": "_None_",
}
ZSPY_MAIN_AXIS_ATTRS = {
    **ZSPY_ROUTE_AXIS_ATTRS,
    "a": 100,
    "expression": "a / (x + 1) + b",
    "size": 11,
}
ZSPY_MAIN_X_ATTRS = {**ZSPY_ROUTE_X_ATTRS, "size": 11}


def _zarr_attrs(path, group):
    import zarr

    root = zarr.open_group(str(path), mode="r")
    assert group in root, group
    return dict(root[group].attrs)


def _zarr_children(path, group):
    import zarr

    root = zarr.open_group(str(path), mode="r")
    assert group in root, group
    return set(root[group].group_keys())


def _zarr_tree_hash(path):
    digest = hashlib.sha256()
    for child in sorted(Path(path).rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode())
            digest.update(child.read_bytes())
    return digest.hexdigest()


def _tamper_zarr_attr(src, dst, group, name, expected_old, new) -> None:
    import zarr

    shutil.copytree(src, dst)
    root = zarr.open_group(str(dst), mode="r+")
    assert group in root, group
    attrs = root[group].attrs
    assert name in attrs, name
    assert type(attrs[name]) is type(expected_old)
    assert attrs[name] == expected_old
    assert type(new) is type(expected_old)
    attrs[name] = new


def _tamper_zarr_element(src, dst, array_path, index, expected_old, new) -> None:
    import zarr

    shutil.copytree(src, dst)
    root = zarr.open_group(str(dst), mode="r+")
    assert array_path in root
    values = root[array_path]
    assert values.shape == (2,)
    assert values.dtype == object
    assert 0 <= index < 2
    assert type(expected_old) is str
    assert type(new) is str
    assert type(values[index]) is str
    assert values[index] == expected_old
    values[index] = new


def _zspy_expected_attrs(route):
    return ZSPY_MAIN_AXIS_ATTRS if route == "main-axis" else ZSPY_ROUTE_AXIS_ATTRS


def assert_zspy_route_file_truth(path, route):
    group = ROUTE_TARGETS[route][0]
    assert _zarr_attrs(path, group) == _zspy_expected_attrs(route)
    assert _zarr_children(path, group) == {"x"}
    assert _zarr_attrs(path, group + "/x") == (
        ZSPY_MAIN_X_ATTRS if route == "main-axis" else ZSPY_ROUTE_X_ATTRS
    )


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("route", ROUTE_CASES)
def test_zspy_route_control(tmp_path, lazy, route):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = _route_builders[route](tmp_path, suffix=".zspy")
    assert_zspy_route_file_truth(control, route)
    digest = _zarr_tree_hash(control)
    loaded = hs.load(control, lazy=lazy)
    try:
        assert loaded._lazy is lazy
        assert_route_loaded_axis(route, loaded)
        assert not sentinel.exists()
    finally:
        print(
            f"case=zspy-{route}-control lazy={lazy} sentinel_exists={sentinel.exists()}"
        )
        if lazy:
            loaded.close_file()
        assert _zarr_tree_hash(control) == digest, "control hash changed"
        assert_zspy_route_file_truth(control, route)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("route", ROUTE_CASES)
def test_zspy_route_attack(tmp_path, lazy, route):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = _route_builders[route](tmp_path, suffix=".zspy")
    assert_zspy_route_file_truth(control, route)
    digest = _zarr_tree_hash(control)
    group, name, old = ROUTE_TARGETS[route]
    new = old + " + " + payload(sentinel)
    dst = tmp_path / f"zspy_{route}_tampered.zspy"
    _tamper_zarr_attr(control, dst, group, name, old, new)
    assert _zarr_attrs(dst, group)[name] == new
    assert _zarr_attrs(control, group)[name] == old
    loaded = None
    try:
        if route == "component-signal":
            loaded = hs.load(dst, lazy=lazy)
            assert not sentinel.exists(), "restore must be the trigger"
            with pytest.raises(ValueError):
                loaded.models.restore("sfp")
        else:
            with pytest.raises(ValueError):
                loaded = hs.load(dst, lazy=lazy)
        assert not sentinel.exists()
    finally:
        print(
            f"case=zspy-{route}-attack lazy={lazy} "
            f"sentinel_exists={sentinel.exists()} "
            f"sentinel_text={sentinel.read_text() if sentinel.exists() else None!r}"
        )
        if loaded is not None and lazy:
            loaded.close_file()
        assert _zarr_tree_hash(control) == digest, "control hash changed"
        assert_zspy_route_file_truth(control, route)


ZSPY_MODEL_COMPONENTS = "Analysis/models/audit/_dict/_list_2_components"
ZSPY_MODEL_EXPRESSION = ZSPY_MODEL_COMPONENTS + "/1"
ZSPY_MODEL_PARAMETERS = ZSPY_MODEL_EXPRESSION + "/_list_2_parameters"
ZSPY_MODEL_CASES = (
    (
        "C1-expression-base",
        ZSPY_MODEL_EXPRESSION,
        "expression",
        "a*x + b; b = 2*c",
        "a*x + b + ",
        "; b = 2*c",
    ),
    (
        "C1-expression-substitution-rhs",
        ZSPY_MODEL_EXPRESSION,
        "expression",
        "a*x + b; b = 2*c",
        "a*x + b; b = 2*c + ",
        "",
    ),
    (
        "C2-twin-function",
        ZSPY_MODEL_PARAMETERS + "/0",
        "twin_function_expr",
        "2*x + 1",
        "2*x + 1 + ",
        "",
    ),
    (
        "C2-twin-inverse",
        ZSPY_MODEL_PARAMETERS + "/0",
        "twin_inverse_function_expr",
        "",
        "(x - 1)/2 + ",
        "",
    ),
)


def build_zspy_stored_model_file(tmp_path):
    data = np.arange(5 * 7 * 11, dtype=float).reshape(5, 7, 11) / 100 + 1
    s = hs.signals.Signal1D(data)
    model = s.create_model()
    gaussian = hs.model.components1D.Gaussian()
    expression = hs.model.components1D.Expression(
        "a*x + b; b = 2*c", name="lin", a=2.0, c=0.5
    )
    model.append(gaussian)
    model.append(expression)
    expression.a.twin = gaussian.A
    expression.a.twin_function_expr = "2*x + 1"
    model.store("audit")
    path = tmp_path / "zspy_model.zspy"
    s.save(path)
    return path


@pytest.mark.parametrize("lazy", (False, True))
def test_zspy_stored_model_control(tmp_path, lazy):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = build_zspy_stored_model_file(tmp_path)
    digest = _zarr_tree_hash(control)
    loaded = hs.load(control, lazy=lazy)
    try:
        assert loaded._lazy is lazy
        model = loaded.models.restore("audit")
        assert model[1].a.twin is model[0].A
        assert model[1].a.twin_function_expr == "2*x + 1"
        np.testing.assert_allclose(
            model[1].function(np.arange(11.0)), 3 * np.arange(11.0) + 1
        )
        assert not sentinel.exists()
    finally:
        if lazy:
            loaded.close_file()
        assert _zarr_tree_hash(control) == digest, "control hash changed"


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize(
    "case", ZSPY_MODEL_CASES, ids=[case[0] for case in ZSPY_MODEL_CASES]
)
def test_zspy_stored_model_attack(tmp_path, lazy, case):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = build_zspy_stored_model_file(tmp_path)
    digest = _zarr_tree_hash(control)
    case_id, group, name, old, prefix, suffix = case
    new = prefix + payload(sentinel) + suffix
    dst = tmp_path / "zspy_model_tampered.zspy"
    _tamper_zarr_attr(control, dst, group, name, old, new)
    assert _zarr_attrs(dst, group)[name] == new
    assert _zarr_attrs(control, group)[name] == old
    loaded = None
    try:
        loaded = hs.load(dst, lazy=lazy)
        assert not sentinel.exists(), "restore must be the trigger"
        with pytest.raises(ValueError):
            loaded.models.restore("audit")
        assert not sentinel.exists()
    finally:
        print(
            f"case=zspy-{case_id} lazy={lazy} sentinel_exists={sentinel.exists()} "
            f"sentinel_text={sentinel.read_text() if sentinel.exists() else None!r}"
        )
        if loaded is not None and lazy:
            loaded.close_file()
        assert _zarr_tree_hash(control) == digest, "control hash changed"


def assert_zspy_position_file_truth(path):
    import zarr

    attrs = _zarr_attrs(path, POSITION_COMPONENT)
    assert attrs["_id_name"] == "Expression"
    assert attrs["name"] == "g2"
    assert attrs["add_rotation"] is True
    assert attrs["expression"] == POSITION_EXPRESSION
    assert _zarr_attrs(path, POSITION_COMPONENT + "/rename_pars") == {"_s": "sigma"}
    root = zarr.open_group(str(path), mode="r")
    values = root[POSITION_ARRAY]
    assert values.shape == (2,)
    assert values.dtype == object
    assert [type(value) for value in values[:]] == [str, str]
    assert list(values[:]) == ["x0", "y0"]
    assert [type(f).__name__ for f in values.filters] == ["VLenUTF8"]


@pytest.mark.parametrize("lazy", (False, True))
def test_zspy_position_control(tmp_path, lazy):
    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = build_position_model_file(tmp_path, suffix=".zspy")
    assert_zspy_position_file_truth(control)
    digest = _zarr_tree_hash(control)
    loaded = hs.load(control, lazy=lazy)
    try:
        assert loaded._lazy is lazy
        model = loaded.models.restore("m2")
        component = model[0]
        assert component._position_x is component.x0
        assert component._position_y is component.y0
        x, y = np.meshgrid(
            np.linspace(-2.0, 3.0, 11),
            np.linspace(-1.5, 2.5, 7),
            indexing="ij",
        )
        np.testing.assert_allclose(
            component.function(x, y), _position_component().function(x, y)
        )
        assert not sentinel.exists()
    finally:
        if lazy:
            loaded.close_file()
        assert _zarr_tree_hash(control) == digest, "control hash changed"
        assert_zspy_position_file_truth(control)


@pytest.mark.parametrize("lazy", (False, True))
def test_zspy_position_attack(tmp_path, lazy):
    import zarr

    sentinel = tmp_path / SENTINEL
    assert not sentinel.exists()
    control = build_position_model_file(tmp_path, suffix=".zspy")
    assert_zspy_position_file_truth(control)
    digest = _zarr_tree_hash(control)
    new = payload(sentinel)
    dst = tmp_path / "zspy_position_tampered.zspy"
    _tamper_zarr_element(control, dst, POSITION_ARRAY, 1, "y0", new)
    root = zarr.open_group(str(dst), mode="r")
    assert list(root[POSITION_ARRAY][:]) == ["x0", new]
    root = zarr.open_group(str(control), mode="r")
    assert list(root[POSITION_ARRAY][:]) == ["x0", "y0"]
    loaded = None
    try:
        loaded = hs.load(dst, lazy=lazy)
        assert not sentinel.exists(), "restore must be the trigger"
        with pytest.raises(ValueError):
            loaded.models.restore("m2")
        assert not sentinel.exists()
    finally:
        print(
            f"case=zspy-C3-position lazy={lazy} "
            f"sentinel_exists={sentinel.exists()} "
            f"sentinel_text={sentinel.read_text() if sentinel.exists() else None!r}"
        )
        if loaded is not None and lazy:
            loaded.close_file()
        assert _zarr_tree_hash(control) == digest, "control hash changed"
        assert_zspy_position_file_truth(control)


def test_rotation_center_large_integer_control():
    from fractions import Fraction

    center = 18446744073709551616
    component = hs.model.components2D.Expression(
        "x+y",
        name="e",
        add_rotation=True,
        rotation_center=(center, 0),
        module="numpy",
        compute_gradients=False,
    )
    component.rotation_angle.value = np.pi / 4
    sine, cosine = np.sin(np.pi / 4), np.cos(np.pi / 4)
    rotated_y = (3.0 - center) * sine + 5.0 * cosine
    expected = center + (3.0 - center) * cosine - rotated_y * sine + rotated_y
    np.testing.assert_allclose(component.function(3.0, 5.0), expected, rtol=1e-13)

    hs.model.components2D.Expression(
        "x+y",
        name="e",
        add_rotation=True,
        rotation_center=(2**1100, 0),
        module="numpy",
        compute_gradients=False,
    )
    for center in (Fraction(1, 2), Fraction(2**1100, 3)):
        hs.model.components2D.Expression(
            "x+y",
            name="e",
            add_rotation=True,
            rotation_center=(center, 0),
            module="numpy",
            compute_gradients=False,
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rotation_center_nonfinite_rejected(value):
    with pytest.raises(ValueError):
        hs.model.components2D.Expression(
            "x+y", name="e", add_rotation=True, rotation_center=(value, 0)
        )


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_rotation_center_boolean_rejected(value):
    with pytest.raises(ValueError):
        hs.model.components2D.Expression(
            "x+y", name="e", add_rotation=True, rotation_center=(value, 0)
        )
