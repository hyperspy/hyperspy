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


import hashlib
import importlib
import io
import shutil
import zipfile
from pathlib import Path
import io

import dask.array as da
import h5py
import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.learn._mva import LearningResults
from hyperspy.signals import Signal1D

sklearn = importlib.util.find_spec("sklearn")
skip_sklearn = pytest.mark.skipif(sklearn is None, reason="sklearn not installed")


def test_learning_results_decom():
    rng = np.random.default_rng(123)

    s1 = Signal1D(rng.random(size=(20, 100)))
    s1.decomposition(output_dimension=2)

    out = str(s1.learning_results)
    assert "Decomposition parameters" in out
    assert "algorithm=SVD" in out
    assert "output_dimension=2" in out
    assert "Demixing parameters" not in out


@skip_sklearn
def test_learning_results_bss():
    rng = np.random.default_rng(123)

    s1 = Signal1D(rng.random(size=(20, 100)))
    s1.decomposition(output_dimension=2)
    s1.blind_source_separation(number_of_components=2)

    out = str(s1.learning_results)
    assert "Decomposition parameters" in out
    assert "Demixing parameters" in out
    assert "algorithm=sklearn_fastica" in out
    assert "n_components=2" in out


def test_learning_results_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(123)

    s1 = Signal1D(rng.random(size=(6, 12, 25)))
    s1.decomposition(output_dimension=2)

    path = tmp_path / "learning_results.npz"
    s1.learning_results.save(path, overwrite=True)

    lr = LearningResults()
    lr.load(path)
    np.testing.assert_allclose(lr.factors, s1.learning_results.factors)
    np.testing.assert_allclose(lr.loadings, s1.learning_results.loadings)
    np.testing.assert_allclose(
        lr.explained_variance, s1.learning_results.explained_variance
    )
    assert lr.decomposition_algorithm == "SVD"
    # Object-valued members are not loaded (pickle is never enabled)
    for name in ("mean", "centre", "unmixing_matrix", "cluster_labels"):
        assert getattr(lr, name) is None, name


class _MarkerWrite:
    """Reducer-bearing object equivalent to ``_pickle_marker``.

    Unpickling it writes the marker file; pickling it (storing it in an
    NPZ) does NOT execute the reduction.
    """

    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return (Path(self.path).write_text, ("hs3699",))


def test_load_npz_benign_object_member_skipped(tmp_path):
    # The object member unpickles to a payload that writes a marker file;
    # loading must never unpickle it.
    marker_path = tmp_path / "marker.txt"
    obj = np.array([_MarkerWrite(marker_path)], dtype=object)
    factors = np.arange(12.0).reshape(6, 2)
    path = tmp_path / "learning_results.npz"
    assert not marker_path.exists()
    np.savez(path, factors=factors, mean=np.array([7.0]), unmixing_matrix=obj)
    # Pickling the object during savez must not execute its reduction
    assert not marker_path.exists()

    # Armed control: NPZ members load lazily, so the marker appears only
    # when the object member is accessed with pickle enabled.
    with np.load(path, allow_pickle=True) as f:
        assert not marker_path.exists()
        f["unmixing_matrix"]
    assert marker_path.exists()
    marker_path.unlink()

    lr = LearningResults()
    lr.load(path)
    np.testing.assert_array_equal(lr.factors, factors)
    assert lr.mean == pytest.approx(7.0)
    assert lr.unmixing_matrix is None
    assert not marker_path.exists()


def test_load_restores_clustering_fields(tmp_path):
    # cluster_analysis() sets these attributes on the instance instead of
    # the class, so load() needs them declared to restore what save()
    # writes.
    lr = LearningResults()
    lr.cluster_sum_signals = np.arange(12.0).reshape(6, 2)
    lr.cluster_centroid_signals = np.arange(12.0).reshape(6, 2) + 1.0
    lr.cluster_distances = np.arange(5.0)
    lr.cluster_centroids = np.arange(6.0).reshape(3, 2)
    path = tmp_path / "learning_results.npz"
    lr.save(path, overwrite=True)
    restored = LearningResults()
    restored.load(path)
    np.testing.assert_allclose(restored.cluster_sum_signals, lr.cluster_sum_signals)
    np.testing.assert_allclose(
        restored.cluster_centroid_signals, lr.cluster_centroid_signals
    )
    np.testing.assert_allclose(restored.cluster_distances, lr.cluster_distances)
    np.testing.assert_allclose(restored.cluster_centroids, lr.cluster_centroids)


def test_load_ignores_unknown_and_dunder_keys(tmp_path):
    factors = np.arange(6.0).reshape(2, 3)
    path = tmp_path / "learning_results.npz"
    np.savez(
        path,
        __class__=np.array([1, 2]),
        unknown_field=np.array([3.0]),
        factors=factors,
    )

    lr = LearningResults()
    with pytest.warns(UserWarning, match="unknown field"):
        lr.load(path)
    assert lr.__class__ is LearningResults
    assert not hasattr(lr, "unknown_field")
    np.testing.assert_array_equal(lr.factors, factors)


def test_load_corrupt_member_propagates(tmp_path):
    # A member with a valid npy magic but a corrupt header must raise out
    # of load() instead of being silently skipped as if it were an
    # object-valued member that only needs pickle.
    factors = np.arange(6.0).reshape(2, 3)
    path = tmp_path / "learning_results.npz"
    factors_path = tmp_path / "factors.npy"
    np.save(factors_path, factors)
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(factors_path, "factors.npy")
        archive.writestr("mean.npy", b"\x93NUMPY\x01\x00garbage-header-bytes!!")

    lr = LearningResults()
    with pytest.raises(ValueError, match="array header"):
        lr.load(path)


def test_load_keeps_dynamic_instance_attribute(tmp_path):
    path = tmp_path / "learning_results.npz"
    np.savez(path, extra=np.array([9.0]), explained_variance=np.array([1.0, 2.0]))

    lr = LearningResults()
    lr.extra = "keep"
    with pytest.warns(UserWarning, match="unknown field 'extra'"):
        lr.load(path)
    assert lr.extra == "keep"
    np.testing.assert_allclose(lr.explained_variance, [1.0, 2.0])


def test_load_legacy_alias_V(tmp_path):
    path = tmp_path / "learning_results.npz"
    np.savez(path, V=np.array([1.5, 2.5]))

    lr = LearningResults()
    lr.load(path)
    np.testing.assert_allclose(lr.explained_variance, [1.5, 2.5])
    assert not hasattr(lr, "V")


# --- Learning-results restoration from real HSPY/ZSPY files -----------------
# Relocated from hyperspy/tests/test_io_expression_security.py, which is not
# part of the RELEASE_next_minor base for this split.

EXPERIMENT = "Experiments/__unnamed__"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _zarr_tree_hash(path):
    digest = hashlib.sha256()
    for child in sorted(Path(path).rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode())
            digest.update(child.read_bytes())
    return digest.hexdigest()


LR_GROUP = EXPERIMENT + "/learning_results"
LR_ATTRIBUTES = EXPERIMENT + "/attributes"
LR_FACTORS = np.arange(24.0).reshape(12, 2)
LR_LOADINGS = np.arange(50.0).reshape(25, 2)
LR_EXPLAINED_VARIANCE = np.array([3.5, 1.25])
LR_CLUSTER_SUM = np.arange(3.0)
LR_CLUSTER_CENTROID = np.arange(4.0)
LR_CLUSTER_DISTANCES = np.arange(5.0)
LR_CLUSTER_CENTROIDS = np.arange(6.0)
LR_ALIAS_V = np.array([7.5, 8.5])


def _lr_match(*parts):
    # Match the route and the offending name independent of message ordering.
    import re

    return "".join(
        f"(?=.*{re.escape(str(part))})" for part in ("learning_results", *parts)
    )


def _build_learning_results_file(tmp_path, suffix):
    data = np.arange(12 * 25 * 48, dtype=float).reshape(12, 25, 48)
    s = hs.signals.Signal1D(data)
    lr = s.learning_results
    lr.factors = LR_FACTORS.copy()
    lr.loadings = LR_LOADINGS.copy()
    lr.explained_variance = LR_EXPLAINED_VARIANCE.copy()
    lr.decomposition_algorithm = "SVD"
    lr.output_dimension = 2
    lr.original_shape = (12, 25, 48)
    lr.cluster_sum_signals = LR_CLUSTER_SUM.copy()
    lr.cluster_centroid_signals = LR_CLUSTER_CENTROID.copy()
    lr.cluster_distances = LR_CLUSTER_DISTANCES.copy()
    lr.cluster_centroids = LR_CLUSTER_CENTROIDS.copy()
    lr.custom_public = "kept"
    lr._object = "serializable-private"
    lr.algorithm = "legacy-algorithm"
    lr.V = LR_ALIAS_V.copy()
    # Shape (1,) with chunks (1,) is required for lazy HDF5 access to reach
    # HyperSpy instead of failing inside dask chunk handling.
    lr.custom_lazy = da.ones(1, chunks=(1,))
    path = tmp_path / f"learning_results{suffix}"
    s.save(path)
    assert path.exists()
    return path


def _copy_learning_results_file(src, dst, suffix):
    if suffix == ".zspy":
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _lr_file_hash(path, suffix):
    if suffix == ".zspy":
        return _zarr_tree_hash(path)
    return _sha256(path)


def _tamper_learning_results_key(src, dst, suffix, key):
    _copy_learning_results_file(src, dst, suffix)
    values = np.arange(1.0, dtype=float)
    if suffix == ".hspy":
        with h5py.File(dst, "r+") as f:
            group = f[LR_GROUP]
            assert key not in group
            group.create_dataset(key, data=values, chunks=(1,))
        with h5py.File(dst, "r") as f:
            assert key in f[LR_GROUP]
    else:
        import zarr

        root = zarr.open_group(str(dst), mode="r+")
        assert key not in root[LR_GROUP]
        root[LR_GROUP].create_dataset(key, data=values, chunks=(1,))
        check = zarr.open_group(str(dst), mode="r")
        assert key in check[LR_GROUP]


def _tamper_learning_results_container(src, dst, suffix):
    marker = "not-a-mapping"
    _copy_learning_results_file(src, dst, suffix)
    if suffix == ".hspy":
        with h5py.File(dst, "r+") as f:
            exp = f[EXPERIMENT]
            assert "learning_results" in exp
            del exp["learning_results"]
            exp["attributes"].attrs["learning_results"] = marker
            assert "learning_results" not in exp
            assert exp["attributes"].attrs["learning_results"] == marker
    else:
        import zarr

        root = zarr.open_group(str(dst), mode="r+")
        assert LR_GROUP in root
        del root[LR_GROUP]
        root[LR_ATTRIBUTES].attrs["learning_results"] = marker
        check = zarr.open_group(str(dst), mode="r")
        assert LR_GROUP not in check
        assert check[LR_ATTRIBUTES].attrs["learning_results"] == marker


@pytest.mark.parametrize("suffix", (".hspy", ".zspy"))
@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("key", ("summary", "save"))
def test_learning_results_file_method_rejected(tmp_path, suffix, lazy, key):
    if suffix == ".zspy":
        pytest.importorskip("zarr")
    src = _build_learning_results_file(tmp_path, suffix)
    dst = tmp_path / f"method_{key}{suffix}"
    _tamper_learning_results_key(src, dst, suffix, key)
    loaded = None
    try:
        with pytest.raises(ValueError, match=_lr_match(key)):
            loaded = hs.load(dst, lazy=lazy)
    finally:
        if loaded is not None and lazy:
            loaded.close_file()


@pytest.mark.parametrize("suffix", (".hspy", ".zspy"))
@pytest.mark.parametrize("lazy", (False, True))
def test_learning_results_file_container_rejected(tmp_path, suffix, lazy):
    if suffix == ".zspy":
        pytest.importorskip("zarr")
    src = _build_learning_results_file(tmp_path, suffix)
    dst = tmp_path / f"container{suffix}"
    _tamper_learning_results_container(src, dst, suffix)
    loaded = None
    try:
        with pytest.raises(ValueError, match=_lr_match("non-mapping", "str")):
            loaded = hs.load(dst, lazy=lazy)
    finally:
        if loaded is not None and lazy:
            loaded.close_file()


@pytest.mark.parametrize("suffix", (".hspy", ".zspy"))
@pytest.mark.parametrize("lazy", (False, True))
def test_learning_results_file_legitimate_roundtrip(tmp_path, suffix, lazy):
    if suffix == ".zspy":
        pytest.importorskip("zarr")
    path = _build_learning_results_file(tmp_path, suffix)
    digest = _lr_file_hash(path, suffix)
    loaded = None
    try:
        loaded = hs.load(path, lazy=lazy)
        assert loaded._lazy is lazy
        lr = loaded.learning_results
        expected_arrays = {
            "factors": LR_FACTORS,
            "loadings": LR_LOADINGS,
            "explained_variance": LR_EXPLAINED_VARIANCE,
            "cluster_sum_signals": LR_CLUSTER_SUM,
            "cluster_centroid_signals": LR_CLUSTER_CENTROID,
            "cluster_distances": LR_CLUSTER_DISTANCES,
            "cluster_centroids": LR_CLUSTER_CENTROIDS,
            "V": LR_ALIAS_V,
        }
        for name, expected in expected_arrays.items():
            actual = getattr(lr, name)
            if lazy:
                assert isinstance(actual, da.Array), name
                actual = actual.compute()
            else:
                assert isinstance(actual, np.ndarray), name
            np.testing.assert_allclose(actual, expected)
        assert lr.explained_variance is not lr.V
        assert lr.decomposition_algorithm == "SVD"
        assert lr.algorithm == "legacy-algorithm"
        assert lr.custom_public == "kept"
        assert lr._object == "serializable-private"
        assert tuple(lr.original_shape) == (12, 25, 48)
        lazy_field = lr.custom_lazy
        if lazy:
            assert isinstance(lazy_field, da.Array)
            assert lazy_field.chunks == ((1,),)
            lazy_field = lazy_field.compute()
        else:
            assert isinstance(lazy_field, np.ndarray)
        np.testing.assert_allclose(lazy_field, np.ones(1))
    finally:
        if loaded is not None and lazy:
            loaded.close_file()
        assert _lr_file_hash(path, suffix) == digest


@pytest.mark.parametrize("suffix", (".hspy", ".zspy"))
@pytest.mark.parametrize("lazy", (False, True))
def test_learning_results_file_dunder_control(tmp_path, suffix, lazy):
    if suffix == ".zspy":
        pytest.importorskip("zarr")
    key = "__getstate__"
    src = _build_learning_results_file(tmp_path, suffix)
    dst = tmp_path / f"dunder{suffix}"
    _tamper_learning_results_key(src, dst, suffix, key)
    loaded = None
    try:
        with pytest.raises(ValueError, match=_lr_match(key)):
            loaded = hs.load(dst, lazy=lazy)
    finally:
        if loaded is not None and lazy:
            loaded.close_file()


class TestLearningResultsLoad:
    # Loading a malformed archive must fail with a clear ValueError
    # instead of an unhandled BadZipFile or EOFError.

    def test_valid_npz_accepted(self, tmp_path):
        path = tmp_path / "results.npz"
        np.savez(path, factors=np.arange(4.0))
        results = LearningResults()
        results.load(path)
        np.testing.assert_allclose(results.factors, np.arange(4.0))

    def test_corrupt_npz_rejected(self, tmp_path):
        path = tmp_path / "corrupt.npz"
        _write_corrupt_npz(path)
        with pytest.raises(ValueError, match="corrupt"):
            LearningResults().load(path)

    def test_empty_npz_rejected(self, tmp_path):
        path = tmp_path / "empty.npz"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="corrupt"):
            LearningResults().load(path)


def _write_corrupt_npz(path):
    # Flip one bit of the stored array data of a valid ``.npz`` member so
    # that its CRC no longer matches.
    buffer = io.BytesIO()
    np.save(buffer, np.arange(4.0))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("factors.npy", buffer.getvalue())
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("factors.npy")
        filename_length = len(info.filename.encode())
        extra_length = len(info.extra)
        data_offset = info.header_offset + 30 + filename_length + extra_length
        data_size = info.file_size
    contents = bytearray(path.read_bytes())
    contents[data_offset + data_size - 1] ^= 1
    path.write_bytes(bytes(contents))
