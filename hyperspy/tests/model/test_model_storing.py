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


import gc
import pickle
from copy import deepcopy
from os import remove
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import hyperspy.api as hs
import hyperspy.model
from hyperspy.component import Component
from hyperspy.components1d import Expression, Gaussian, GaussianHF
from hyperspy.components2d import Expression as Expression2D
from hyperspy.components2d import Gaussian2D
from hyperspy.extensions import ALL_EXTENSIONS


class StoredCustomComponent(Component):
    """Benign unregistered component; a stored model pickles its class."""

    def __init__(self, scale=1.0):
        Component.__init__(self, ("scale",))
        self.scale.value = scale

    def function(self, x):
        return self.scale.value * x


class OverridingComponent(StoredCustomComponent):
    """Registered-legitimate component overriding _load_dictionary with the
    plain historical signature (no trust keyword)."""

    def _load_dictionary(self, dic):
        return super()._load_dictionary(dic)


class EdsLikeSignal1D(hs.signals.Signal1D):
    """Downstream signal with the eXSpy EDS-shaped fixed *kwargs
    create_model signature."""

    def create_model(self, auto_background=True, print_info=True, *args, **kwargs):
        from hyperspy.models.model1d import Model1D

        return Model1D(self, *args, **kwargs)


class OldSignal1D(hs.signals.Signal1D):
    """Downstream signal with the eXSpy EELS-shaped fixed create_model
    signature that has no trust keyword at all."""

    def create_model(self, dictionary=None):
        from hyperspy.models.model1d import Model1D

        m = Model1D(self)
        if dictionary is not None:
            m._load_dictionary(dictionary)
        return m


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


def clean_model_dictionary(d):
    for c in d["components"]:
        for p in c["parameters"]:
            del p["self"]
    return d


class TestModelStoring:
    def setup_method(self, method):
        s = hs.signals.Signal1D(range(100))
        m = s.create_model()
        m.append(Gaussian())
        m.fit()
        self.m = m

    def test_models_getattr(self):
        m = self.m
        s = m.signal
        m.store()
        assert s.models.a is s.models["a"]

    def test_models_stub_methods(self):
        m = self.m
        s = m.signal
        m.store()
        s.models.pop = mock.MagicMock()
        s.models.remove = mock.MagicMock()
        s.models.restore = mock.MagicMock()
        s.models.a.restore()
        s.models.a.remove()
        s.models.a.pop()

        assert s.models.pop.call_count == 1
        assert s.models.remove.call_count == 1
        assert s.models.restore.call_count == 1

        assert s.models.pop.call_args[0] == ("a",)
        assert s.models.remove.call_args[0] == ("a",)
        assert s.models.restore.call_args[0] == ("a",)

    def test_models_pop(self):
        m = self.m
        s = m.signal
        m.store()
        s.models.remove = mock.MagicMock()
        s.models.restore = mock.MagicMock()
        s.models.pop("a")
        assert s.models.remove.call_count == 1
        assert s.models.restore.call_count == 1
        assert s.models.remove.call_args[0] == ("a",)
        assert s.models.restore.call_args[0] == ("a",)

    def test_model_store(self):
        m = self.m
        m.store()
        d = m.as_dictionary(True)
        np.testing.assert_equal(d, m.signal.models._models.a._dict.as_dictionary())

    def test_actually_stored(self):
        m = self.m
        m.store()
        m[0].A.map["values"][0] += 13.33
        m1 = m.signal.models.a.restore()
        assert m[0].A.map["values"] != m1[0].A.map["values"]

    def test_models_restore_remove(self):
        m = self.m
        s = m.signal
        m.store("a")
        m1 = s.models.restore("a")
        m2 = s.models.a.restore()
        d_o = clean_model_dictionary(m.as_dictionary())
        d_1 = clean_model_dictionary(m1.as_dictionary())
        d_2 = clean_model_dictionary(m2.as_dictionary())
        np.testing.assert_equal(d_o, d_1)
        np.testing.assert_equal(d_o, d_2)
        assert 1 == len(s.models)
        s.models.a.remove()
        assert 0 == len(s.models)

    def test_store_name_error1(self):
        s = self.m.signal
        with pytest.raises(KeyError):
            s.models.restore("a")

    def test_store_name_error2(self):
        s = self.m.signal
        with pytest.raises(KeyError):
            s.models.restore(3)

    def test_store_name_error3(self):
        s = self.m.signal
        with pytest.raises(KeyError):
            s.models.restore("_a")

    def test_store_name_error4(self):
        s = self.m.signal
        with pytest.raises(KeyError):
            s.models.restore("a._dict")

    def test_store_name_error5(self):
        s = self.m.signal
        self.m.store("b")
        with pytest.raises(KeyError):
            s.models.restore("a")


@pytest.mark.parametrize("save_std", [True, False])
@pytest.mark.parametrize("only_free", [True, False])
@pytest.mark.parametrize("only_active", [True, False])
def test_model_export(tmp_path, save_std, only_free, only_active):
    s = hs.signals.Signal1D(range(100))
    m = s.create_model()
    m.append(Gaussian())
    m.fit()
    m.export_results(
        tmp_path, save_std=save_std, only_free=only_free, only_active=only_active
    )


class TestModelSaving:
    def setup_method(self, method):
        s = hs.signals.Signal1D(range(100))
        m = s.create_model()
        m.append(Gaussian(A=13))
        m[-1].name = "something"
        m.append(GaussianHF(module="numpy"))
        m[-1].height.value = 3
        m.append(
            Expression(
                name="Line", expression="a * x + b", a=1, c=0, rename_pars={"b": "c"}
            )
        )
        self.m = m

    def test_save_and_load_model(self):
        m = self.m
        m.save("tmp.hspy", overwrite=True)
        s = hs.load("tmp.hspy")
        assert hasattr(s.models, "a")
        mr = s.models.restore("a")
        assert mr.components.something.A.value == 13
        assert mr.components.GaussianHF.height.value == 3
        assert mr.components.Line.a.value == 1
        assert mr.components.Line.c.value == 0
        assert mr.components.Line.function(10) == 10

    def teardown_method(self, method):
        gc.collect()  # Make sure any memmaps are closed first!
        remove("tmp.hspy")


def test_EELSModel_saving(tmp_path):
    pytest.importorskip("exspy")

    s = hs.signals.Signal1D(range(100))
    s.axes_manager[0].offset = 280
    s.set_signal_type("EELS")
    s.add_elements(["C"])
    s.set_microscope_parameters(100, 10, 10)
    m = s.create_model(auto_background=False)
    m.components.C_K.fine_structure_smoothing = 0.5
    m.components.C_K.fine_structure_width = 50
    m.components.C_K.fine_structure_active = True

    m.save(tmp_path / "tmp.hspy")
    s2 = hs.load(tmp_path / "tmp.hspy")
    assert hasattr(s2.models, "a")
    n = s2.models.restore("a")
    assert n[0].fine_structure_width == 50


MARKER_NAME = "hs3699_marker.txt"


def _pickle_marker(marker_path) -> bytes:
    """Pickle of a benign object whose unpickling writes the marker file.

    Kept in this model-storing test module because this branch does not carry
    the expression-security test module; the cloudpickle tests that need the
    helper import it from here.
    """

    class _Marker:
        def __reduce__(self):
            return (Path(str(marker_path)).write_text, ("hs3699",))

    return pickle.dumps(_Marker())


def _assert_component_params_equal(model, expected):
    for component, reference in zip(model, expected):
        assert component._id_name == reference._id_name
        np.testing.assert_allclose(
            [p.value for p in component.parameters],
            [p.value for p in reference.parameters],
            rtol=1e-6,
        )


class Test2DComponentStoring:
    """Built-in 2D components are registered, so storing does not pickle
    their class and stored models restore without any flag."""

    def setup_method(self, method):
        self.s = hs.signals.Signal2D(
            np.arange(3 * 12 * 25, dtype=float).reshape(3, 12, 25)
        )
        m = self.s.create_model()
        m.append(
            Gaussian2D(A=2.5, sigma_x=1.5, sigma_y=2.5, centre_x=5.0, centre_y=7.0)
        )
        m.append(Expression2D(name="Plane", expression="a*x + b*y", a=1.0, b=2.0))
        self.m = m

    def test_2d_builtins_not_pickled_and_restore(self):
        m = self.m
        m.store("b")
        components = m.as_dictionary()["components"]
        assert len(components) == 2
        for comp in components:
            assert "_class_dump" not in comp
        m1 = self.s.create_model(dictionary=m.as_dictionary())
        _assert_component_params_equal(m1, m)
        m2 = self.s.models.restore("b")
        _assert_component_params_equal(m2, m)

    def test_legacy_2d_dump_file_restores(self, tmp_path, monkeypatch):
        m = self.m
        legacy_registry = {
            key: value
            for key, value in hyperspy.model._COMPONENTS.items()
            if key in ALL_EXTENSIONS["components1D"]
        }
        with monkeypatch.context() as mp:
            mp.setattr("hyperspy.model._COMPONENTS", legacy_registry)
            m.store("a")
            d = m.as_dictionary()
            path = tmp_path / "legacy2d.hspy"
            self.s.save(path)
        # the genuine writer produced a real `_class_dump` for Gaussian2D
        assert "_class_dump" in d["components"][0]
        restored = hs.load(path).models.restore("a")
        _assert_component_params_equal(restored, m)

    def test_registered_id_marker_class_dump_ignored(self, tmp_path):
        m = self.m
        marker = tmp_path / MARKER_NAME
        d = m.as_dictionary()
        d["components"][0]["_class_dump"] = _pickle_marker(marker)
        self.s.models._save("a", d)
        path = tmp_path / "marker2d.hspy"
        self.s.save(path)
        restored = hs.load(path).models.restore("a")
        _assert_component_params_equal(restored, m)
        assert not marker.exists()
        # control: the payload is armed - unpickling it creates the marker
        pickle.loads(_pickle_marker(marker))
        assert marker.exists()
        marker.unlink()


class TestTrustRestore:
    """Stored models restore without unpickling anything unless the call
    passes keyword-only ``trusted=True``, through every public entry point
    and through fixed-signature downstream ``create_model`` shapes."""

    def setup_method(self, method):
        self.s = hs.signals.Signal1D(
            np.arange(4 * 7 * 48, dtype=float).reshape(4, 7, 48)
        )

    def _store_custom(self, signal, name="custom"):
        m = signal.create_model()
        m.append(StoredCustomComponent(scale=2.0))
        d = m.as_dictionary()
        assert "_class_dump" in d["components"][0]
        m.store(name)
        return d

    def _save_tainted(self, signal, d, marker, name="tainted"):
        """Store a marker-dump copy under its own name so the genuine custom
        model stays available for the trusted control."""

        d = deepcopy(d)
        d["components"][0]["_class_dump"] = _pickle_marker(marker)
        signal.models._save(name, d)

    def test_edsl_like_registered_and_custom(self, tmp_path):
        s = EdsLikeSignal1D(np.arange(4 * 7 * 48, dtype=float).reshape(4, 7, 48))
        m = s.create_model()
        m.append(Gaussian(A=2.0))
        m.store("registered")
        restored = s.models.restore("registered")
        assert isinstance(restored[0], Gaussian)
        d = self._store_custom(s)
        marker = tmp_path / MARKER_NAME
        self._save_tainted(s, d, marker)
        try:
            with pytest.raises(_unsafe_legacy_content()):
                s.models.restore("tainted")
        finally:
            _assert_no_marker(marker)
        # control: the scope carries trust through the fixed signature
        m2 = s.models.restore("custom", trusted=True)
        assert isinstance(m2[0], StoredCustomComponent)
        np.testing.assert_allclose(m2[0].scale.value, 2.0)

    def test_old_signal_registered_and_custom(self, tmp_path):
        s = OldSignal1D(np.arange(4 * 7 * 48, dtype=float).reshape(4, 7, 48))
        m = s.create_model()
        m.append(Gaussian(A=2.0))
        m.store("registered")
        restored = s.models.restore("registered")
        assert isinstance(restored[0], Gaussian)
        d = self._store_custom(s)
        marker = tmp_path / MARKER_NAME
        self._save_tainted(s, d, marker)
        try:
            with pytest.raises(_unsafe_legacy_content()):
                s.models.restore("tainted")
        finally:
            _assert_no_marker(marker)
        # control: trust is honoured inside the fixed signature through the
        # private scope, not through any keyword the subclass must accept
        m2 = s.models.restore("custom", trusted=True)
        assert isinstance(m2[0], StoredCustomComponent)
        np.testing.assert_allclose(m2[0].scale.value, 2.0)

    def test_create_model_1d_deny_and_trusted(self):
        s = hs.signals.Signal1D(np.arange(4 * 7 * 48, dtype=float).reshape(4, 7, 48))
        d = self._store_custom(self.s)
        with pytest.raises(_unsafe_legacy_content()):
            s.create_model(dictionary=d)
        m = s.create_model(dictionary=d, trusted=True)
        assert isinstance(m[0], StoredCustomComponent)
        np.testing.assert_allclose(m[0].scale.value, 2.0)

    def test_create_model_2d_deny_and_trusted(self, tmp_path):
        marker = tmp_path / MARKER_NAME
        s = hs.signals.Signal2D(np.arange(3 * 12 * 25, dtype=float).reshape(3, 12, 25))
        m = s.create_model()
        m.append(
            Gaussian2D(A=2.5, sigma_x=1.5, sigma_y=2.5, centre_x=5.0, centre_y=7.0)
        )
        d_ok = m.as_dictionary()
        # control: the built-in 2D component is registered and restores
        m1 = s.create_model(dictionary=d_ok)
        assert isinstance(m1[0], Gaussian2D)
        # a genuine custom component dictionary, as a legacy file stores it
        m1d = hs.signals.Signal1D(
            np.arange(4 * 7 * 48, dtype=float).reshape(4, 7, 48)
        ).create_model()
        m1d.append(StoredCustomComponent(scale=2.0))
        d_custom = m1d.as_dictionary()
        assert "_class_dump" in d_custom["components"][0]
        d_marker = deepcopy(d_custom)
        d_marker["components"][0]["_class_dump"] = _pickle_marker(marker)
        with pytest.raises(_unsafe_legacy_content()):
            s.create_model(dictionary=d_marker)
        _assert_no_marker(marker)
        # control: the benign custom component restores under trusted=True
        m2 = s.create_model(dictionary=d_custom, trusted=True)
        assert isinstance(m2[0], StoredCustomComponent)
        np.testing.assert_allclose(m2[0].scale.value, 2.0)

    def test_restore_pop_entries_deny_then_trusted(self):
        s = self.s
        self._store_custom(s)
        with pytest.raises(_unsafe_legacy_content()):
            s.models.restore("custom")
        with pytest.raises(_unsafe_legacy_content()):
            s.models.custom.restore()
        with pytest.raises(_unsafe_legacy_content()):
            s.models.pop("custom")
        # a rejected pop leaves the model stored
        assert hasattr(s.models, "custom")
        with pytest.raises(_unsafe_legacy_content()):
            s.models.custom.pop()
        assert hasattr(s.models, "custom")
        m2 = s.models.restore("custom", trusted=True)
        np.testing.assert_allclose(m2[0].scale.value, 2.0)
        m3 = s.models.custom.pop(trusted=True)
        np.testing.assert_allclose(m3[0].scale.value, 2.0)
        assert not hasattr(s.models, "custom")

    def test_positional_trusted_typeerror(self):
        s = self.s
        m = s.create_model()
        m.append(Gaussian())
        m.store("g")
        with pytest.raises(TypeError):
            s.models.restore("g", True)
        with pytest.raises(TypeError):
            s.models.pop("g", True)

    def test_trusted_string_denied(self, tmp_path):
        marker = tmp_path / MARKER_NAME
        s = self.s
        d = self._store_custom(s)
        self._save_tainted(s, d, marker)
        # any value other than exactly True means "not trusted"
        with pytest.raises(_unsafe_legacy_content()):
            s.models.restore("tainted", trusted="yes")
        _assert_no_marker(marker)

    def test_scope_reset_after_success_and_importerror(self):
        from hyperspy.misc.export_dictionary import _is_trusted

        s = self.s
        d = self._store_custom(s)
        m2 = s.models.restore("custom", trusted=True)
        assert isinstance(m2[0], StoredCustomComponent)
        assert _is_trusted() is False
        with pytest.raises(_unsafe_legacy_content()):
            s.models.restore("custom")
        # an unknown id with no dump keeps ImportError even when trusted
        d_unknown = deepcopy(d)
        comp = d_unknown["components"][0]
        del comp["_class_dump"]
        comp["_id_name"] = "UnknownNoDump"
        s.models._save("unknown", d_unknown)
        with pytest.raises(ImportError):
            s.models.restore("unknown", trusted=True)
        assert _is_trusted() is False
        with pytest.raises(_unsafe_legacy_content()):
            s.models.restore("custom")

    def test_registered_subclass_override_no_trusted_param(self, monkeypatch):
        monkeypatch.setitem(
            hyperspy.model._COMPONENTS,
            "OverridingComponent",
            {
                "module": OverridingComponent.__module__,
                "class": "OverridingComponent",
            },
        )
        s = self.s
        m = s.create_model()
        m.append(OverridingComponent(scale=4.0))
        d = m.as_dictionary()
        assert "_class_dump" not in d["components"][0]
        m.store("ov")
        m1 = s.models.restore("ov")
        _assert_component_params_equal(m1, m)
        m2 = s.models.restore("ov", trusted=True)
        _assert_component_params_equal(m2, m)

    def test_slicing_restored_model(self):
        s = self.s
        m = s.create_model()
        m.append(Expression("a * x + b", name="Line", a=1.0, b=2.0))
        m.append(StoredCustomComponent(scale=3.0))
        m.store("sl")
        m1 = s.models.restore("sl", trusted=True)
        assert isinstance(m1[0], Expression)
        assert isinstance(m1[1], StoredCustomComponent)
        sliced = m1.inav[0:2]
        assert len(sliced) == 2
        assert isinstance(sliced[1], StoredCustomComponent)
        sliced_signal = m1.isig[10:20]
        assert len(sliced_signal) == 2
