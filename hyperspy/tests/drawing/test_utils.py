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

from unittest.mock import Mock, patch

import matplotlib
import numpy as np
import pytest

from hyperspy.drawing.utils import (
    ColorCycle,
    contrast_stretching,
    create_figure,
    key_press_handler_custom,
)


def test_create_figure():
    # needs to run inside the function to make sure the correct backend is used
    # not possible to pytest.mark.skipif decorator
    if matplotlib.get_backend().lower() == "agg":
        pytest.skip("'agg' backend does not support on_close event.")

    dummy_function = Mock()
    fig = create_figure(
        window_title="test title", _on_figure_window_close=dummy_function
    )
    assert isinstance(fig, matplotlib.figure.Figure) is True
    matplotlib.pyplot.close(fig)
    dummy_function.assert_called_once_with()


def test_contrast_stretching():
    data = np.arange(100)
    assert contrast_stretching(data, 1, 99) == (1, 99)
    assert contrast_stretching(data, 1.0, 99.0) == (1, 99)
    assert contrast_stretching(data, "1th", "99th") == (0.99, 98.01)
    assert contrast_stretching(data, "0.05th", "99.95th") == (0.0495, 98.9505)
    # vmin, vmax are to set in conftest.py
    assert contrast_stretching(data, None, None) == (0.0, 99.0)


def test_color_cycle_refills_when_exhausted():
    cycle = ColorCycle()
    n_colors = len(cycle.color_cycle)
    colors = [cycle() for _ in range(n_colors)]
    assert cycle.color_cycle == []
    # Popping again once exhausted must refill and return the first color.
    next_color = cycle()
    assert next_color == colors[0]
    assert cycle.color_cycle != []


def test_key_press_handler_custom_forwards_unhandled_keys():
    """Keys other than the hyperspy-reserved k/l/L are forwarded to
    matplotlib's own key_press_handler."""
    with patch("matplotlib.backend_bases.key_press_handler") as mock_handler:
        canvas = Mock()
        event = Mock(key="a")
        key_press_handler_custom(event, canvas)
    mock_handler.assert_called_once_with(event, canvas, canvas.manager.toolbar)


@pytest.mark.parametrize("key", ["k", "l", "L"])
def test_key_press_handler_custom_ignores_reserved_keys(key):
    """Hyperspy-reserved keys must not reach matplotlib's key_press_handler."""
    with patch("matplotlib.backend_bases.key_press_handler") as mock_handler:
        canvas = Mock()
        event = Mock(key=key)
        key_press_handler_custom(event, canvas)
    mock_handler.assert_not_called()
