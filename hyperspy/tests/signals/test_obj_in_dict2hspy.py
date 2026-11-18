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

import re
import types
from collections import UserDict

import h5py
import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.axes import AxesManager
from hyperspy.learn._mva import LearningResults
from hyperspy.signal import BaseSignal, ModelManager, _obj_in_dict2hspy


def test_obj_in_dict2hspy_signal():
    s_dict = BaseSignal([0, 1, 2])._to_dictionary()
    d = {"_sig_signal": s_dict}
    _obj_in_dict2hspy(d, False)
    assert isinstance(d["signal"], BaseSignal)
    np.testing.assert_allclose(d["signal"].data, [0, 1, 2])


def test_obj_in_dict2hspy_axes_manager():
    axes_list = [
        {
            "name": "a",
            "navigate": True,
            "offset": 0.0,
            "scale": 1.3,
            "size": 2,
            "units": "aa",
        },
        {
            "name": "b",
            "navigate": False,
            "offset": 1.0,
            "scale": 6.0,
            "size": 3,
            "units": "bb",
        },
    ]

    am = AxesManager(axes_list)
    d = {"_hspy_AxesManager_am": am._get_axes_dicts()}
    _obj_in_dict2hspy(d, False)
    assert isinstance(d["am"], AxesManager)
    assert am[0].scale == d["am"][0].scale


def test_signal_containers():
    s = BaseSignal([0, 1, 2])
    s2 = BaseSignal([4, 5, 6])
    s3 = BaseSignal([7, 8, 9])
    s.metadata.signal_container = [s2, s3]
    s_dict = s.metadata.as_dictionary()
    _obj_in_dict2hspy(s_dict, False)
    signal_container = s_dict["signal_container"]
    isinstance(signal_container[0], BaseSignal)
    np.testing.assert_allclose(signal_container[0].data, [4, 5, 6])
    isinstance(signal_container[1], BaseSignal)
    np.testing.assert_allclose(signal_container[1].data, [7, 8, 9])


# --- Task 7: dunder keys in the loaded-attributes dict are rejected ---
# (also used by the learning-results restoration checks below)


def _assert_no_class_mutation():
    # the rejection must leave the signal machinery untouched
    assert BaseSignal.__name__ == "BaseSignal"
    s = BaseSignal(np.arange(12))
    np.testing.assert_allclose(s.data, np.arange(12))
    assert isinstance(s.axes_manager, AxesManager)
    assert type(s.metadata).__name__ == "DictionaryTreeBrowser"


def test_load_dictionary_top_level_dunder_rejected():
    d = {
        "data": np.arange(12, dtype=float),
        "attributes": {"__class__": {"marker_attr": 1}},
    }
    with pytest.raises(ValueError, match="__class__"):
        hs.signals.Signal1D(**d)
    _assert_no_class_mutation()


def test_load_dictionary_nested_dunder_rejected():
    # "_lazy" is an allowlisted attributes key, so its nested mapping still
    # reaches the setter guard; unknown keys are ignored before it.
    d = {
        "data": np.arange(12, dtype=float),
        "attributes": {"_lazy": {"__class__": {"a": 1}}},
    }
    with pytest.raises(ValueError, match="__class__"):
        hs.signals.Signal1D(**d)
    _assert_no_class_mutation()


@pytest.mark.parametrize("carrier", ["_lazy", "ragged"])
def test_load_dictionary_nested_invalid_key_after_valid_rejected(carrier):
    # A valid-looking nested key ordered before the invalid one must not
    # reach the setters: the guard rejects the whole mapping first.
    for bad in ({"good": 1, "__class__": {"a": 1}}, {"good": 1, 5: 2}):
        d = {
            "data": np.arange(12, dtype=float),
            "attributes": {carrier: dict(bad)},
        }
        with pytest.raises(ValueError):
            hs.signals.Signal1D(**d)
        _assert_no_class_mutation()


def test_load_dictionary_learning_results_no_partial_mutation():
    # A benign nested key ordered before a rejected one must not land on
    # the learning results before the guard raises.
    s = hs.signals.Signal1D(np.arange(12, dtype=float))
    before = s.learning_results.factors
    for bad_key in ("__class__", 5):
        d = {
            "data": np.arange(12, dtype=float),
            "attributes": {"learning_results": {"factors": "mutated", bad_key: "bad"}},
        }
        with pytest.raises(ValueError):
            s._load_dictionary(d)
        assert s.learning_results.factors == before
    _assert_no_class_mutation()


@pytest.mark.parametrize("bad", [{"__class__": {"a": 1}}, {5: 2}])
def test_load_dictionary_no_attribute_applied_before_rejected(bad):
    # An earlier allowlisted entry must not be applied when a later nested
    # mapping is rejected: the whole attributes mapping is validated before
    # any attribute is set.
    d = {
        "data": np.arange(12, dtype=float),
        "attributes": {"_lazy": True, "ragged": {"good": 1, **bad}},
    }
    with pytest.raises(ValueError):
        hs.signals.Signal1D(**d)
    s = hs.signals.Signal1D(np.arange(12, dtype=float))
    with pytest.raises(ValueError):
        s._load_dictionary(d)
    assert s._lazy is False
    _assert_no_class_mutation()


def test_load_dictionary_learning_results_not_applied_before_rejected():
    # The nested learning-results route is deferred until every attributes
    # entry is validated, so a later invalid entry cannot leave the
    # learning results half-restored.
    s = hs.signals.Signal1D(np.arange(12, dtype=float))
    factors_before = s.learning_results.factors
    d = {
        "data": np.arange(12, dtype=float),
        "attributes": {
            "learning_results": {"factors": "mutated"},
            "ragged": {"good": 1, "__class__": "bad"},
        },
    }
    with pytest.raises(ValueError):
        s._load_dictionary(d)
    assert s.learning_results.factors == factors_before
    _assert_no_class_mutation()


def test_load_dictionary_legitimate_keys_unchanged():
    d = {
        "data": np.arange(12, dtype=float),
        "attributes": {"_lazy": False, "ragged": False, "metadata": {"x": 1}},
    }
    s = hs.signals.Signal1D(**d)
    assert not s._lazy
    assert not s.axes_manager._ragged
    # metadata is restored from the top-level "metadata" mapping only;
    # a "metadata" key inside the file-controlled "attributes" mapping is
    # ignored like any other non-allowlisted name
    assert not s.metadata.has_item("x")


@pytest.mark.parametrize("key", [5, b"evil"])
def test_load_dictionary_non_string_top_level_key_rejected(key):
    d = {"data": np.arange(12, dtype=float), "attributes": {key: 1}}
    with pytest.raises(ValueError, match="non-string attribute key"):
        hs.signals.Signal1D(**d)
    _assert_no_class_mutation()


def test_load_dictionary_non_string_nested_key_rejected():
    # same carrier rule as the nested dunder test above
    d = {
        "data": np.arange(12, dtype=float),
        "attributes": {"_lazy": {5: 1}},
    }
    with pytest.raises(ValueError, match="non-string attribute key"):
        hs.signals.Signal1D(**d)
    _assert_no_class_mutation()


def test_load_yaml_non_string_attribute_key_via_hs_load(tmp_path):
    # F4's real surface: a custom YAML reader handed to hs.load through the
    # supported file_format object, producing an integer attribute key
    import yaml

    path = tmp_path / "custom.yaml"
    path.write_text("data: [0, 1, 2, 3]\nmetadata: {}\nattributes:\n  1: crafted\n")

    class YamlReader:
        @staticmethod
        def file_reader(filename, **kwds):
            with open(filename) as f:
                signal_dict = yaml.safe_load(f)
            signal_dict["data"] = np.asarray(signal_dict["data"], dtype=float)
            return [signal_dict]

    with pytest.raises(ValueError, match="non-string attribute key"):
        hs.load(path, file_format=YamlReader)
    _assert_no_class_mutation()


# --- File-controlled "attributes" cannot clobber signal attributes ---

_FILE_ATTRIBUTE_ATTACK_KEYS = (
    "models",
    "axes_manager",
    "metadata",
    "original_metadata",
    "inav",
    "isig",
    "events",
    "plot",
    "tmp_parameters",
)


@pytest.mark.parametrize("key", _FILE_ATTRIBUTE_ATTACK_KEYS)
def test_load_dictionary_unknown_attribute_key_ignored(key):
    # Only the keys written by this library are restored; anything else in
    # the file-controlled mapping is ignored instead of being set on the
    # signal (reported privately 2026-11-24: "models" as a plain string
    # clobbered the ModelManager on a plain load).
    d = {
        "data": np.arange(12, dtype=float),
        "attributes": {key: "pwned-string"},
    }
    s = hs.signals.Signal1D(**d)
    assert not isinstance(getattr(s, key), str)
    assert isinstance(s.models, ModelManager)
    assert isinstance(s.axes_manager, AxesManager)
    _assert_no_class_mutation()


def test_load_dictionary_nested_unknown_attribute_injection_ignored():
    d = {
        "data": np.arange(12, dtype=float),
        "attributes": {"axes_manager": {"pwn": 1}, "metadata": {"x": 1}},
    }
    s = hs.signals.Signal1D(**d)
    assert not hasattr(s.axes_manager, "pwn")
    assert not s.metadata.has_item("x")
    _assert_no_class_mutation()


@pytest.mark.parametrize("lazy", [False, True])
def test_load_hspy_attributes_group_models_clobber_ignored(tmp_path, lazy):
    # Real-file variant of the reported sink: a tampered
    # Experiments/<name>/attributes group stores a string under "models".
    s = hs.signals.Signal1D(np.arange(12, dtype=float))
    path = tmp_path / "c.hspy"
    s.save(path)
    with h5py.File(path, "r+") as f:
        attributes = f["Experiments/__unnamed__"]["attributes"]
        attributes.attrs["models"] = "pwned-string"
    s2 = hs.load(path, lazy=lazy)
    assert isinstance(s2.models, ModelManager)
    _assert_no_class_mutation()


# --- Learning-results restoration boundary (direct and custom-reader routes) ---

_LR_DATA = np.arange(12 * 25 * 48, dtype=float).reshape(12, 25, 48)


def _lr_signal_dict(**extra):
    return {"data": _LR_DATA.copy(), **extra}


def _nested_lr_dict(mapping):
    return _lr_signal_dict(attributes={"learning_results": mapping})


def _lr_match(*parts):
    # Match the route and the offending key independent of message ordering;
    # base and fixed messages do not agree on word order.
    return "".join(
        f"(?=.*{re.escape(str(part))})" for part in ("learning_results", *parts)
    )


class _ExplodingLearningValue:
    """Value that fails if restoration evaluates or computes it."""

    def __array__(self, *args, **kwargs):
        raise AssertionError("learning-results values must not be evaluated")

    def compute(self, *args, **kwargs):
        raise AssertionError("learning-results values must not be computed")


class _LearningResultsYamlReader:
    """Custom reader with the loader metadata hs.load records."""

    __loader__ = types.SimpleNamespace(name=__name__)

    @staticmethod
    def file_reader(filename, **kwds):
        import yaml

        with open(filename) as f:
            signal_dict = yaml.safe_load(f)
        signal_dict["data"] = _LR_DATA.copy()
        return [signal_dict]


@pytest.mark.parametrize("key", ("summary", "save", "_transpose_results"))
def test_learning_results_direct_method_rejected(key):
    with pytest.raises(ValueError, match=_lr_match(key)):
        hs.signals.Signal1D(**_nested_lr_dict({key: np.arange(3.0)}))
    _assert_no_class_mutation()


def test_learning_results_reserved_class_state_rejected():
    with pytest.raises(ValueError, match=_lr_match("_LEGACY_ALIASES")):
        hs.signals.Signal1D(
            **_lr_signal_dict(learning_results={"_LEGACY_ALIASES": ("algorithm",)})
        )
    _assert_no_class_mutation()


@pytest.mark.parametrize(
    "key", ("__class__", "__getstate__", "__reduce_ex__", "__unknown_dunder__")
)
def test_learning_results_dunder_rejected(key):
    with pytest.raises(ValueError, match=_lr_match(key)):
        hs.signals.Signal1D(**_lr_signal_dict(learning_results={key: "value"}))
    _assert_no_class_mutation()


@pytest.mark.parametrize("key", (5, b"evil"), ids=("integer", "bytes"))
def test_learning_results_nonstring_rejected(key):
    with pytest.raises(ValueError, match=_lr_match(str(key))):
        hs.signals.Signal1D(**_lr_signal_dict(learning_results={key: "value"}))
    _assert_no_class_mutation()


@pytest.mark.parametrize(
    "container",
    (None, "not-a-mapping", [("factors", np.zeros(2))]),
    ids=("none", "scalar", "pair-list"),
)
def test_learning_results_container_rejected(container):
    expected_type = type(container).__name__
    with pytest.raises(ValueError, match=_lr_match("non-mapping", expected_type)):
        hs.signals.Signal1D(**_lr_signal_dict(learning_results=container))
    _assert_no_class_mutation()


def test_learning_results_compatibility_control():
    sentinel = object()
    alias_values = {
        "algorithm": "legacy-algorithm",
        "V": np.array([1.5]),
        "w": np.array([2.5]),
        "variance2one": True,
        "centered": False,
        "pca_algorithm": "PCA",
        "ica_algorithm": "FastICA",
        "ica_scores": np.array([3.5]),
        "ica_factors": np.array([4.5]),
        "v": np.array([5.5]),
        "scores": np.array([6.5]),
        "pc": np.array([7.5]),
    }
    mapping = {
        "factors": np.arange(6.0).reshape(3, 2),
        "decomposition_algorithm": "SVD",
        "custom_public": {"kept": True},
        "_object": sentinel,
        "cluster_sum_signals": np.arange(3.0),
        **alias_values,
    }
    s = hs.signals.Signal1D(**_lr_signal_dict(learning_results=mapping))
    lr = s.learning_results
    assert isinstance(lr, LearningResults)
    assert lr.factors is mapping["factors"]
    assert lr.custom_public is mapping["custom_public"]
    assert lr._object is sentinel
    for name, value in alias_values.items():
        assert getattr(lr, name) is value, name


def test_learning_results_userdict_control():
    sentinel = object()
    mapping = UserDict(
        {
            "factors": np.arange(6.0).reshape(3, 2),
            "custom_public": "kept",
            "_object": sentinel,
        }
    )
    s = hs.signals.Signal1D(**_lr_signal_dict(learning_results=mapping))
    assert s.learning_results.factors is mapping["factors"]
    assert s.learning_results.custom_public == "kept"
    assert s.learning_results._object is sentinel


def test_learning_results_nested_nonstring_control():
    with pytest.raises(ValueError, match=_lr_match(5)):
        hs.signals.Signal1D(**_nested_lr_dict({5: np.arange(3.0)}))
    _assert_no_class_mutation()


def test_learning_results_value_identity_control():
    sentinel = _ExplodingLearningValue()
    s = hs.signals.Signal1D(**_lr_signal_dict(learning_results={"_object": sentinel}))
    assert s.learning_results.__dict__["_object"] is sentinel


def test_learning_results_custom_method_rejected(tmp_path):
    path = tmp_path / "learning_results_method.yaml"
    path.write_text("metadata: {}\nlearning_results:\n  summary: shadow\n")
    with pytest.raises(ValueError, match=_lr_match("summary")):
        hs.load(path, file_format=_LearningResultsYamlReader)
    _assert_no_class_mutation()


def test_learning_results_custom_legitimate_control(tmp_path):
    path = tmp_path / "learning_results_legit.yaml"
    path.write_text(
        "metadata: {}\n"
        "learning_results:\n"
        "  decomposition_algorithm: SVD\n"
        "  custom_public: 7\n"
        "  _object: kept\n"
    )
    s = hs.load(path, file_format=_LearningResultsYamlReader)
    assert isinstance(s.learning_results, LearningResults)
    assert s.learning_results.decomposition_algorithm == "SVD"
    assert s.learning_results.custom_public == 7
    assert s.learning_results._object == "kept"


def _preseed_learning_results(s):
    s.learning_results.factors = np.arange(6.0).reshape(3, 2)
    s.learning_results._object = object()
    s.learning_results.custom_public = {"kept": True}
    return dict(s.learning_results.__dict__)


def _assert_learning_results_unchanged(s, expected):
    actual = s.learning_results.__dict__
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert actual[key] is value, key


@pytest.mark.parametrize(
    ("nested", "top"),
    (
        ({"nested_only": "kept"}, {"summary": "shadow"}),
        ({"summary": "shadow"}, {"top_only": "valid"}),
    ),
    ids=("top", "nested"),
)
def test_learning_results_cross_route_atomicity(nested, top):
    s = BaseSignal(np.zeros((12, 25, 48)))
    before = _preseed_learning_results(s)
    file_data_dict = {
        "data": _LR_DATA.copy(),
        "attributes": {"learning_results": nested},
        "learning_results": top,
    }
    with pytest.raises(ValueError, match=_lr_match("summary")):
        s._load_dictionary(file_data_dict)
    _assert_learning_results_unchanged(s, before)


def test_learning_results_cross_route_precedence_control():
    nested_factors = np.arange(6.0).reshape(3, 2)
    top_factors = nested_factors + 100
    s = hs.signals.Signal1D(
        **_lr_signal_dict(
            attributes={
                "learning_results": {
                    "factors": nested_factors,
                    "nested_only": "kept",
                }
            },
            learning_results={"factors": top_factors, "top_only": "kept"},
        )
    )
    assert s.learning_results.factors is top_factors
    assert s.learning_results.nested_only == "kept"
    assert s.learning_results.top_only == "kept"


def test_learning_results_single_route_atomicity():
    s = BaseSignal(np.zeros((12, 25, 48)))
    before = _preseed_learning_results(s)
    file_data_dict = {
        "data": _LR_DATA.copy(),
        "attributes": {
            "learning_results": {
                "factors": np.full((3, 2), 9.0),
                "summary": "shadow",
            }
        },
    }
    with pytest.raises(ValueError, match=_lr_match("summary")):
        s._load_dictionary(file_data_dict)
    _assert_learning_results_unchanged(s, before)
