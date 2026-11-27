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


import pickle
from copy import deepcopy

import cloudpickle
import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.component import Component, Parameter
from hyperspy.components1d import Expression, Gaussian, Lorentzian, ScalableFixedPattern
from hyperspy.tests.model.test_model_storing import _pickle_marker


class StoredInitComponent(Component):
    """Benign unregistered component; stored files pickle its class."""

    def __init__(self, scale=1.0):
        Component.__init__(self, ("scale",))
        self.scale.value = scale

    def function(self, x):
        return self.scale.value * x


class TrustedInitComponent(Component):
    """Registered-legitimate component with an init argument named trusted."""

    def __init__(self, scale=1.0, trusted=False):
        Component.__init__(self, ("scale",))
        self.scale.value = scale
        self.trusted = trusted

    def function(self, x):
        return self.scale.value * x


MARKER_NAME = "hs3699_marker.txt"


def _unsafe_legacy_content():
    """Imported lazily: the exception class does not exist before the fix."""

    from hyperspy.exceptions import UnsafeLegacyContent

    return UnsafeLegacyContent


def _assert_no_marker(marker):
    assert not marker.exists()
    # control: the payload is armed - unpickling it creates the marker
    pickle.loads(_pickle_marker(marker))
    assert marker.exists()
    marker.unlink()


def remove_empty_numpy_strings(dic):
    for k, v in dic.items():
        if isinstance(v, dict):
            remove_empty_numpy_strings(v)
        elif isinstance(v, list):
            for vv in v:
                if isinstance(vv, dict):
                    remove_empty_numpy_strings(vv)
                elif isinstance(vv, np.bytes_) and len(vv) == 0:
                    vv = ""
        elif isinstance(v, np.bytes_) and len(v) == 0:
            del dic[k]
            dic[k] = ""


class DummyAxesManager:
    navigation_shape = [
        1,
    ]
    navigation_size = 2
    indices = ()

    @property
    def _navigation_shape_in_array(self):
        return self.navigation_shape[::-1]


class TestParameterDictionary:
    def setup_method(self, method):
        self.par = Parameter()
        self.par.name = "asd"
        self.par._id_name = "newone"
        self.par.twin_function_expr = "x * x"
        self.par.twin_inverse_function_expr = "sqrt(x)"
        self.par._axes_manager = DummyAxesManager()
        self.par._create_array()
        self.par.value = 1
        self.par.std = 0.1
        self.par.store_current_value_in_array()
        self.par.ext_bounded = False
        self.par.ext_force_positive = False

    def test_to_dictionary(self):
        d = self.par.as_dictionary()

        assert d["name"] == self.par.name
        assert d["_id_name"] == self.par._id_name
        np.testing.assert_equal(d["map"]["values"][0], 1)
        np.testing.assert_equal(d["map"]["std"][0], 0.1)
        assert d["map"]["is_set"][0]
        np.testing.assert_equal(d["value"], self.par.value)
        np.testing.assert_equal(d["std"], self.par.std)
        assert d["free"] is self.par.free
        assert d["self"] == id(self.par)
        np.testing.assert_equal(d["_bounds"], self.par._bounds)
        assert d["ext_bounded"] is self.par.ext_bounded
        assert d["ext_force_positive"] is self.par.ext_force_positive

    def test_load_dictionary(self):
        d = self.par.as_dictionary()
        p = Parameter()
        p._id_name = "newone"
        _id = p._load_dictionary(d)

        assert _id == id(self.par)
        assert p.name == self.par.name
        assert p._id_name == self.par._id_name
        np.testing.assert_equal(p.map["values"][0], 1)
        np.testing.assert_equal(p.map["std"][0], 0.1)
        assert p.map["is_set"][0]
        np.testing.assert_equal(p.value, self.par.value)
        np.testing.assert_equal(p.std, self.par.std)
        np.testing.assert_equal(p.free, self.par.free)
        np.testing.assert_equal(p._bounds, self.par._bounds)

        rn = np.random.random()
        np.testing.assert_equal(p._twin_function(rn), self.par._twin_function(rn))
        np.testing.assert_equal(
            p._twin_inverse_function(rn), self.par._twin_inverse_function(rn)
        )

    def test_invalid_name(self):
        d = self.par.as_dictionary()
        d["_id_name"] = "otherone"
        p = Parameter()
        p._id_name = "newone"
        with pytest.raises(ValueError):
            _ = p._load_dictionary(d)


class TestComponentDictionary:
    def setup_method(self, method):
        self.parameter_names = ["par1", "par2"]
        self.comp = Component(self.parameter_names)
        self.comp.name = "newname!"
        self.comp._id_name = "dummy names yay!"
        self.comp._axes_manager = DummyAxesManager()
        self.comp._create_arrays()
        self.comp.par1.value = 2.0
        self.comp.par2.value = 5.0
        self.comp.par1.std = 0.2
        self.comp.par2.std = 0.5
        self.comp.store_current_parameters_in_map()

    def test_to_dictionary(self):
        d = self.comp.as_dictionary()
        c = self.comp

        assert c.name == d["name"]
        assert c._id_name == d["_id_name"]
        assert not d["active_is_multidimensional"]
        assert d["active"]
        assert d["_active_array"] is None
        for ip, p in enumerate(c.parameters):
            assert p.as_dictionary() == d["parameters"][ip]

        c.active_is_multidimensional = True
        d1 = c.as_dictionary()
        assert d1["active_is_multidimensional"]
        np.testing.assert_array_equal(d1["_active_array"], c._active_array)

    def test_load_dictionary(self):
        c = self.comp
        c.par1.twin_function_expr = "x + 2"
        c.par2.twin_function_expr = "x - 2"
        d = c.as_dictionary(True)
        n = Component(self.parameter_names)

        n._id_name = "dummy names yay!"
        _ = n._load_dictionary(d)
        assert c.name == n.name
        assert c.active == n.active
        assert c.active_is_multidimensional == n.active_is_multidimensional

        for pn, pc in zip(n.parameters, c.parameters):
            rn = np.random.random()
            assert pn._twin_function(rn) == pc._twin_function(rn)
            assert pn._twin_inverse_function(rn) == pc._twin_inverse_function(rn)
            dn = pn.as_dictionary()
            del dn["self"]
            dc = pc.as_dictionary()
            del dc["self"]
            print(list(dn.keys()))
            print(list(dc.keys()))
            assert dn == dc

    def test_invalid_component_name(self):
        c = self.comp
        d = c.as_dictionary()
        n = Component(self.parameter_names)
        with pytest.raises(ValueError):
            _ = n._load_dictionary(d)

    def test_invalid_parameter_name(self):
        c = self.comp
        d = c.as_dictionary()
        n = Component([a + "s" for a in self.parameter_names])
        n._id_name = "dummy names yay!"
        with pytest.raises(ValueError):
            _ = n._load_dictionary(d)


class TestModelDictionary:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.array([1.0, 2, 4, 7, 12, 7, 4, 2, 1]))
        m = s.create_model()
        self.model = m
        self.s = s

        m.append(Gaussian())
        m.append(Gaussian())
        m.append(ScalableFixedPattern(s * 0.3))
        m[0].A.twin = m[1].A
        m.fit()

    def test_to_dictionary(self):
        m = self.model
        d = m.as_dictionary()

        np.testing.assert_allclose(m.chisq.data, d["chisq.data"])
        np.testing.assert_allclose(m.dof.data, d["dof.data"])
        np.testing.assert_equal(
            d["free_parameters_boundaries"], m.free_parameters_boundaries
        )

        for num, c in enumerate(m):
            tmp = c.as_dictionary()
            remove_empty_numpy_strings(tmp)
            assert d["components"][num]["name"] == tmp["name"]
            assert d["components"][num]["_id_name"] == tmp["_id_name"]
        np.testing.assert_equal(
            d["components"][-1]["signal1D"], (m.signal * 0.3)._to_dictionary()
        )

    def test_load_dictionary(self):
        d = self.model.as_dictionary()
        mn = self.s.create_model()
        mn.append(Lorentzian())
        mn._load_dictionary(d)
        mo = self.model

        # assert_true(np.allclose(mo.signal1D.data, mn.signal1D.data))
        np.testing.assert_allclose(mo.chisq.data, mn.chisq.data)
        np.testing.assert_allclose(mo.dof.data, mn.dof.data)

        np.testing.assert_equal(
            mn.free_parameters_boundaries, mo.free_parameters_boundaries
        )
        for i in range(len(mn)):
            assert mn[i]._id_name == mo[i]._id_name
            for po, pn in zip(mo[i].parameters, mn[i].parameters):
                np.testing.assert_allclose(po.map["values"], pn.map["values"])
                np.testing.assert_allclose(po.map["is_set"], pn.map["is_set"])

        assert mn[0].A.twin is mn[1].A


class TestPickledContentTrust:
    """Stored pickled content (a custom component class or a pickled
    function) is rejected on restore unless the call runs inside an
    explicitly trusted scope."""

    def setup_method(self, method):
        s = hs.signals.Signal1D(np.arange(4 * 7 * 48, dtype=float).reshape(4, 7, 48))
        m = s.create_model()
        m.append(Gaussian(A=2.0, sigma=1.5, centre=10.0))
        self.s = s
        self.m = m
        self.d = m.as_dictionary()

    def _marker_component(self, marker, class_dump=None):
        comp = deepcopy(self.d["components"][0])
        comp["_id_name"] = "UnregisteredCustom"
        comp["_class_dump"] = class_dump if class_dump else _pickle_marker(marker)
        comp["parameters"] = []
        return comp

    def test_reconstruct_component_class_dump_denied(self, tmp_path):
        from hyperspy.misc.export_dictionary import _trust_scope
        from hyperspy.model import reconstruct_component

        marker = tmp_path / MARKER_NAME
        comp = self._marker_component(marker)
        with pytest.raises(_unsafe_legacy_content()):
            reconstruct_component(comp)
        _assert_no_marker(marker)
        # control: inside a trusted scope the benign custom class is built
        comp = self._marker_component(marker, cloudpickle.dumps(StoredInitComponent))
        with _trust_scope(True):
            instance = reconstruct_component(comp)
        assert isinstance(instance, StoredInitComponent)

    def test_reconstruct_object_pickled_fn_denied(self, tmp_path):
        from hyperspy.misc.export_dictionary import reconstruct_object

        marker = tmp_path / MARKER_NAME
        with pytest.raises(_unsafe_legacy_content()):
            reconstruct_object("fn", (True, _pickle_marker(marker)))
        _assert_no_marker(marker)
        # control: the (None, obj) form passes through without unpickling
        obj = {"a": 1}
        assert reconstruct_object("fn", (None, obj)) is obj

    def test_load_from_dictionary_pickled_fn_denied(self, tmp_path):
        from hyperspy.misc.export_dictionary import load_from_dictionary

        marker = tmp_path / MARKER_NAME
        d = self.d["components"][0]
        d = deepcopy(d)
        d["_whitelist"]["special_fn"] = "fn"
        d["special_fn"] = (True, _pickle_marker(marker))
        target = Gaussian()
        with pytest.raises(_unsafe_legacy_content()):
            load_from_dictionary(target, d)
        _assert_no_marker(marker)
        assert not hasattr(target, "special_fn")

    def test_parameter_load_dictionary_fn_denied(self, tmp_path):
        marker = tmp_path / MARKER_NAME
        par = Parameter()
        par._id_name = "newone"
        par.twin_function_expr = "x * x"
        par.twin_inverse_function_expr = "sqrt(x)"
        par._axes_manager = DummyAxesManager()
        par._create_array()
        d = par.as_dictionary()
        d["_whitelist"]["extra_fn"] = "fn"
        d["extra_fn"] = (True, _pickle_marker(marker))
        target = Parameter()
        target._id_name = "newone"
        with pytest.raises(_unsafe_legacy_content()):
            target._load_dictionary(d)
        _assert_no_marker(marker)

    def test_component_load_dictionary_fn_denied(self, tmp_path):
        marker = tmp_path / MARKER_NAME
        d = deepcopy(self.d["components"][0])
        d["_whitelist"]["special_fn"] = "fn"
        d["special_fn"] = (True, _pickle_marker(marker))
        target = Gaussian()
        with pytest.raises(_unsafe_legacy_content()):
            target._load_dictionary(d)
        _assert_no_marker(marker)

    def test_base_model_preflight_preserves_components(self, tmp_path):
        marker = tmp_path / MARKER_NAME
        d = deepcopy(self.d)
        d["components"].append(self._marker_component(marker))
        m2 = self.s.create_model()
        m2.append(Gaussian())
        with pytest.raises(_unsafe_legacy_content()):
            m2._load_dictionary(d)
        # the tolerant preflight runs before anything is removed
        assert len(m2) == 1
        _assert_no_marker(marker)

    def test_forged_trusted_keys_inert(self, tmp_path):
        from hyperspy.misc.export_dictionary import _is_trusted

        marker = tmp_path / MARKER_NAME
        d = deepcopy(self.d)
        comp = self._marker_component(marker)
        comp["trusted"] = True
        comp["parameters"] = [{"trusted": True}]
        d["components"] = [comp]
        d["trusted"] = True
        m2 = self.s.create_model()
        # trust is never read from the file: all three forged keys are inert
        try:
            with pytest.raises(_unsafe_legacy_content()):
                m2._load_dictionary(d)
        finally:
            _assert_no_marker(marker)
            assert _is_trusted() is False

    def test_preflight_malformed_shapes_tolerated(self):
        from hyperspy.misc.export_dictionary import _preflight_pickled_content

        # the tolerant preflight skips shapes it does not recognize; the
        # guarded sinks still protect whatever it skips
        _preflight_pickled_content({"components": "not a list"})
        _preflight_pickled_content([None, 42, "a string"])
        _preflight_pickled_content(
            [
                {
                    "_id_name": None,
                    "_class_dump": b"junk",
                    "_whitelist": None,
                }
            ]
        )
        # a registered id with a broken whitelist is skipped, not raised
        _preflight_pickled_content(
            [
                {
                    "_id_name": "Gaussian",
                    "_class_dump": b"junk",
                    "_whitelist": {"k": 123},
                }
            ]
        )
        _preflight_pickled_content(
            [
                {
                    "_id_name": None,
                    "_class_dump": b"junk",
                    "_whitelist": {},
                    "parameters": [None, 42],
                }
            ]
        )


class TestStoredInitKeys:
    """File-controlled component init keys can never enable trust, collide
    with a loader argument or pollute a class: they are rejected with
    ``ValueError`` before reaching the component constructor."""

    def setup_method(self, method):
        s = hs.signals.Signal1D(np.arange(4 * 7 * 48, dtype=float).reshape(4, 7, 48))
        m = s.create_model()
        m.append(Expression("a * x + b", name="Line", a=1.0, b=2.0))
        self.s = s
        self.d = m.as_dictionary()

    def _comp_with_init_key(self, key, value):
        d = deepcopy(self.d)
        comp = d["components"][0]
        comp["_whitelist"][key] = "init"
        comp[key] = value
        return d

    def test_init_key_trusted_on_expression(self):
        d = self._comp_with_init_key("trusted", True)
        m2 = self.s.create_model()
        with pytest.raises(ValueError) as excinfo:
            m2._load_dictionary(d)
        assert type(excinfo.value) is not TypeError
        from hyperspy.misc.export_dictionary import _is_trusted

        assert _is_trusted() is False

    def test_init_key_trusted_on_class_dump_denied(self, tmp_path):
        marker = tmp_path / MARKER_NAME
        d = deepcopy(self.d)
        comp = d["components"][0]
        comp["_id_name"] = "UnregisteredCustom"
        comp["_class_dump"] = _pickle_marker(marker)
        comp["parameters"] = []
        comp["_whitelist"]["trusted"] = "init"
        comp["trusted"] = True
        m2 = self.s.create_model()
        with pytest.raises(ValueError):
            m2._load_dictionary(d)
        _assert_no_marker(marker)

    def test_init_key_comp_dictionary_collision(self):
        d = self._comp_with_init_key("comp_dictionary", {})
        m2 = self.s.create_model()
        with pytest.raises(ValueError):
            m2._load_dictionary(d)

    def test_init_key_dunder_class_rejected(self):
        d = self._comp_with_init_key("__class__", 12345.0)
        m2 = self.s.create_model()
        try:
            with pytest.raises(ValueError):
                m2._load_dictionary(d)
            assert "value" not in vars(Expression)
        finally:
            # must not leak into other tests if the guard is ever missing
            if "value" in vars(Expression):
                del Expression.value

    def test_init_param_named_trusted_legitimate(self, monkeypatch):
        from hyperspy.model import _COMPONENTS

        monkeypatch.setitem(
            _COMPONENTS,
            "TrustedInitComponent",
            {
                "module": TrustedInitComponent.__module__,
                "class": "TrustedInitComponent",
            },
        )
        s = hs.signals.Signal1D(np.arange(4 * 7 * 48, dtype=float).reshape(4, 7, 48))
        m = s.create_model()
        m.append(TrustedInitComponent(scale=3.0))
        d = m.as_dictionary()
        comp = d["components"][0]
        # a legacy file may legitimately store an init argument literally
        # named `trusted`; it reaches the component but enables no trust
        comp["_whitelist"]["trusted"] = "init"
        comp["trusted"] = True
        m2 = s.create_model()
        m2._load_dictionary(d)
        assert isinstance(m2[0], TrustedInitComponent)
        assert m2[0].trusted is True
        from hyperspy.misc.export_dictionary import _is_trusted

        assert _is_trusted() is False


# --- Task 7: dunder whitelist keys are rejected before traversal ---


def test_whitelist_dunder_key_rejected():
    from hyperspy.misc.export_dictionary import load_from_dictionary

    key = "__class__.__base__.__getattribute__"
    d = {"_whitelist": {key: ""}, key: "payload"}
    with pytest.raises(ValueError, match="__class__"):
        load_from_dictionary(Gaussian(), d)
    # Parameter/Component access still works in the same process
    p = Parameter()
    p.value = 2.5
    assert p.value == pytest.approx(2.5)
    g = Gaussian()
    g.A.value = 1.0
    assert g.A.value == pytest.approx(1.0)
    assert isinstance(g, Component)


def test_whitelist_ordinary_key_control():
    from hyperspy.misc.export_dictionary import load_from_dictionary

    g = Gaussian()
    g.A.value = 1.0
    d = {"_whitelist": {"A.value": ""}, "A.value": 3.0}
    load_from_dictionary(g, d)
    assert g.A.value == pytest.approx(3.0)


@pytest.mark.parametrize(
    "key, flags", [(5, "id"), (5, ""), (b"evil", "id"), (b"evil", "")]
)
def test_whitelist_non_string_key_rejected(key, flags):
    from hyperspy.misc.export_dictionary import load_from_dictionary

    # the id-flag shape reaches the boundary without attrsetter; the plain
    # shape would traverse: both must stop at the key check
    d = {"_whitelist": {key: flags}, key: "payload"}
    with pytest.raises(ValueError, match="non-string whitelist key"):
        load_from_dictionary(Gaussian(), d)
    # Parameter/Component access still works in the same process
    p = Parameter()
    p.value = 2.5
    assert p.value == pytest.approx(2.5)
    g = Gaussian()
    g.A.value = 1.0
    assert g.A.value == pytest.approx(1.0)
    assert isinstance(g, Component)


def test_whitelist_bang_dunder_key_rejected():
    from hyperspy.misc._utils import DictionaryTreeBrowser
    from hyperspy.misc.export_dictionary import load_from_dictionary

    # `!__class__` traverses to `__class__` after slugification, so the
    # dunder check must reject the normalized parts or attrsetter writes
    # onto the class itself.
    key = "signal.metadata.!__class__.process_lazy_attributes"
    d = {"_whitelist": {key: ""}, key: "payload"}
    s = hs.signals.Signal1D(np.arange(12))
    m = s.create_model()
    with pytest.raises(ValueError, match="dunder whitelist key"):
        load_from_dictionary(m, d)
    # The traversal must not have overwritten the class method with the
    # file-controlled payload
    assert callable(DictionaryTreeBrowser.process_lazy_attributes)
