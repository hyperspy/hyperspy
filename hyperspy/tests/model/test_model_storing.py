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
from os import remove
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import hyperspy.api as hs
import hyperspy.model
from hyperspy.components1d import Expression, Gaussian, GaussianHF
from hyperspy.components2d import Expression as Expression2D
from hyperspy.components2d import Gaussian2D
from hyperspy.extensions import ALL_EXTENSIONS


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
