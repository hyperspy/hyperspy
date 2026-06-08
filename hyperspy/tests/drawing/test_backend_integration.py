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

"""Integration tests for the backend abstraction (Phases 2, 5–6)."""

import importlib.metadata
from unittest.mock import patch

import numpy as np
import pytest

import hyperspy.api as hs
import hyperspy.drawing  # noqa: F401 — ensure backend is registered


@pytest.mark.parametrize("signal_dim", [1, 2])
def test_signal_plot_uses_backend_explorer(signal_dim):
    """signal.plot() must instantiate the class returned by get_explorer()."""
    from hyperspy.drawing.backends import get_backend
    from hyperspy.drawing.he import HyperExplorer

    if signal_dim == 2:
        data = np.random.random((3, 4, 5, 6))
        s = hs.signals.Signal2D(data)
    else:
        data = np.random.random((3, 4, 5))
        s = hs.signals.Signal1D(data)

    instantiated = []
    original_get_explorer = get_backend().get_explorer

    def tracking_get_explorer(dim):
        cls = original_get_explorer(dim)

        class Tracking(cls):
            def __init__(self_inner):
                super().__init__()
                instantiated.append(cls)

        return Tracking

    with patch.object(get_backend(), "get_explorer", tracking_get_explorer):
        s.plot(navigator="slider")

    assert len(instantiated) == 1
    assert issubclass(instantiated[0], HyperExplorer)
    s._plot.close()


def test_external_backend_via_entry_point(monkeypatch):
    """A fake backend injected via entry-point mocking can be loaded."""
    from hyperspy.drawing.backends._registry import load_backend

    class _FakeBackend:
        pass

    class _FakeEP:
        name = "fake_integration_backend"

        def load(self):
            return _FakeBackend

    real_entry_points = importlib.metadata.entry_points

    def _patched(group=None, **kwargs):
        result = real_entry_points(group=group, **kwargs)
        if group == "hyperspy.backends":
            return list(result) + [_FakeEP()]
        return result

    monkeypatch.setattr(importlib.metadata, "entry_points", _patched)
    backend = load_backend("fake_integration_backend")
    assert isinstance(backend, _FakeBackend)


def test_load_backend_unknown_raises_valueerror():
    from hyperspy.drawing.backends._registry import load_backend

    with pytest.raises(ValueError, match="Unknown plotting backend"):
        load_backend("__totally_nonexistent__")


def test_right_pointer_warns_on_unsupported_backend():
    """BackendCapabilityError from _add_right_line → UserWarning, not crash."""
    from hyperspy.drawing.backends._protocol import BackendCapabilityError

    s = hs.signals.Signal1D(np.random.random((4, 8)))
    s.plot(navigator="slider")
    explorer = s._plot

    with patch.object(
        explorer, "_add_right_line", side_effect=BackendCapabilityError("no twin")
    ):
        with pytest.warns(UserWarning, match="Right pointer not available"):
            explorer.right_pointer_on = True

    s._plot.close()


def test_markers_warn_on_unsupported_backend():
    """BackendCapabilityError from add_collection → UserWarning, no crash."""
    from hyperspy.drawing.backends import get_backend
    from hyperspy.drawing.backends._protocol import BackendCapabilityError

    s = hs.signals.Signal1D(np.random.random((4, 8)))
    marker = hs.plot.markers.Lines(
        segments=np.array([[[0, 0], [1, 1]]]),
    )
    s.add_marker(marker, permanent=False)
    s.plot(navigator="slider")

    with patch.object(
        get_backend(),
        "add_collection",
        side_effect=BackendCapabilityError("no collections"),
    ):
        with pytest.warns(UserWarning, match="does not support markers"):
            # Force re-plot of the marker by calling plot directly
            marker.plot()

    s._plot.close()
