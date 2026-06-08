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

import warnings
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.drawing.mpl_he import (
    _is_marimo_backend,
    _marimo_display_figure,
    _marimo_interactive_display,
)

pytest.importorskip("marimo")

from marimo._runtime.context import ContextNotInitializedError  # noqa: E402

# ── _is_marimo_backend ────────────────────────────────────────────────


class TestIsMarimoBackend:
    def test_returns_true_when_in_marimo_context(self):
        with patch("marimo._runtime.context.get_context") as mock_get:
            mock_get.return_value = object()  # any truthy object
            assert _is_marimo_backend() is True

    def test_returns_false_when_context_not_initialized(self):
        with patch("marimo._runtime.context.get_context") as mock_get:
            mock_get.side_effect = ContextNotInitializedError()
            assert _is_marimo_backend() is False

    def test_returns_false_when_marimo_not_imported(self):
        """Covers the ImportError path — force `import marimo` to fail."""
        with patch.dict("sys.modules", {"marimo": None}):
            assert _is_marimo_backend() is False


# ── _marimo_interactive_display ───────────────────────────────────────


def _make_mock_figure():
    """Return a mock matplotlib Figure."""
    return MagicMock(name="Figure")


def _make_mock_explorer(signal_fig=None, navigator_fig=None):
    """Return a mock MPL_HyperExplorer with optional signal/navigator plots."""
    explorer = MagicMock()
    if signal_fig is not None:
        explorer.signal_plot = MagicMock()
        explorer.signal_plot.get_mpl_figure.return_value = signal_fig
    else:
        explorer.signal_plot = None
    if navigator_fig is not None:
        explorer.navigator_plot = MagicMock()
        explorer.navigator_plot.get_mpl_figure.return_value = navigator_fig
    else:
        explorer.navigator_plot = None
    return explorer


class TestMarimoInteractiveDisplay:
    def test_signal_only(self):
        sig_fig = _make_mock_figure()
        explorer = _make_mock_explorer(signal_fig=sig_fig)

        with (
            patch("marimo.mpl.interactive") as mock_interactive,
            patch("marimo.output.append") as mock_append,
        ):
            mock_interactive.return_value = "mocked-html"
            _marimo_interactive_display(explorer, plot_style="vertical")

        mock_interactive.assert_called_once_with(sig_fig)
        mock_append.assert_called_once_with("mocked-html")

    def test_navigator_only(self):
        nav_fig = _make_mock_figure()
        explorer = _make_mock_explorer(navigator_fig=nav_fig)

        with (
            patch("marimo.mpl.interactive") as mock_interactive,
            patch("marimo.output.append") as mock_append,
        ):
            mock_interactive.return_value = "mocked-html"
            _marimo_interactive_display(explorer, plot_style="vertical")

        mock_interactive.assert_called_once_with(nav_fig)
        mock_append.assert_called_once_with("mocked-html")

    def test_signal_and_navigator_horizontal(self):
        sig_fig = _make_mock_figure()
        nav_fig = _make_mock_figure()
        explorer = _make_mock_explorer(signal_fig=sig_fig, navigator_fig=nav_fig)

        with (
            patch("marimo.mpl.interactive") as mock_interactive,
            patch("marimo.hstack") as mock_hstack,
            patch("marimo.output.append") as mock_append,
        ):
            mock_interactive.side_effect = ["mocked-nav", "mocked-sig"]
            mock_hstack.return_value = "mocked-hstack"
            _marimo_interactive_display(explorer, plot_style="horizontal")

        assert mock_interactive.call_count == 2
        mock_interactive.assert_any_call(nav_fig)
        mock_interactive.assert_any_call(sig_fig)
        mock_hstack.assert_called_once_with(["mocked-nav", "mocked-sig"])
        mock_append.assert_called_once_with("mocked-hstack")

    def test_signal_and_navigator_vertical(self):
        sig_fig = _make_mock_figure()
        nav_fig = _make_mock_figure()
        explorer = _make_mock_explorer(signal_fig=sig_fig, navigator_fig=nav_fig)

        with (
            patch("marimo.mpl.interactive") as mock_interactive,
            patch("marimo.vstack") as mock_vstack,
            patch("marimo.output.append") as mock_append,
        ):
            mock_interactive.side_effect = ["mocked-nav", "mocked-sig"]
            mock_vstack.return_value = "mocked-vstack"
            _marimo_interactive_display(explorer, plot_style="vertical")

        assert mock_interactive.call_count == 2
        mock_vstack.assert_called_once_with(["mocked-nav", "mocked-sig"])
        mock_append.assert_called_once_with("mocked-vstack")

    def test_no_figures(self):
        explorer = _make_mock_explorer()

        with (
            patch("marimo.mpl.interactive") as mock_interactive,
            patch("marimo.output.append") as mock_append,
        ):
            _marimo_interactive_display(explorer, plot_style="vertical")

        mock_interactive.assert_not_called()
        mock_append.assert_not_called()

    def test_navigator_has_no_figure(self):
        """Navigator plot exists but get_mpl_figure returns None."""
        explorer = MagicMock()
        explorer.signal_plot = None
        explorer.navigator_plot = MagicMock()
        explorer.navigator_plot.get_mpl_figure.return_value = None

        with (
            patch("marimo.mpl.interactive") as mock_interactive,
            patch("marimo.output.append") as mock_append,
        ):
            _marimo_interactive_display(explorer, plot_style="vertical")

        mock_interactive.assert_not_called()
        mock_append.assert_not_called()

    def test_invalid_plot_style(self):
        explorer = _make_mock_explorer(signal_fig=_make_mock_figure())

        with pytest.raises(ValueError, match="plot_style"):
            _marimo_interactive_display(explorer, plot_style="diagonal")

    def test_plot_style_none_falls_back_to_preferences(self):
        sig_fig = _make_mock_figure()
        explorer = _make_mock_explorer(signal_fig=sig_fig)

        with (
            patch("marimo.mpl.interactive") as mock_interactive,
            patch("marimo.output.append") as mock_append,
            patch(
                "hyperspy.defaults_parser.preferences.Plot.widget_plot_style",
                "horizontal",
            ),
        ):
            mock_interactive.return_value = "mocked-html"
            _marimo_interactive_display(explorer, plot_style=None)

        mock_interactive.assert_called_once_with(sig_fig)
        mock_append.assert_called_once_with("mocked-html")


# ── plot() integration ────────────────────────────────────────────────


class TestMarimoPlotIntegration:
    def test_marimo_backend_triggers_ioff(self):
        """When _is_marimo_backend is True, plt.ioff() should be used."""
        matplotlib.use("Agg")

        with (
            patch("hyperspy.drawing.mpl_he._is_marimo_backend", return_value=True),
            patch("hyperspy.drawing.mpl_he._is_widget_backend", return_value=False),
            patch(
                "hyperspy.drawing.mpl_he._marimo_interactive_display"
            ) as mock_display,
        ):
            s = hs.signals.Signal1D(np.random.random((12, 25, 48)))
            # Use all-different dimensions: shape (12, 25, 48) → displays as (25, 12 | 48)
            s.plot()

            mock_display.assert_called_once()

    def test_marimo_skips_ipympl_branch(self):
        """When marimo is active, the ipympl display branch must not be used."""
        matplotlib.use("Agg")

        with (
            patch("hyperspy.drawing.mpl_he._is_marimo_backend", return_value=True),
            patch("hyperspy.drawing.mpl_he._is_widget_backend", return_value=False),
            patch("hyperspy.drawing.mpl_he._marimo_interactive_display") as mock_marimo,
        ):
            s = hs.signals.Signal2D(np.random.random((7, 9, 11, 13)))
            # shape (7, 9, 11, 13) → Signal2D displays as (9, 7 | 13, 11)
            s.plot(plot_style="horizontal")

        mock_marimo.assert_called_once()

    def test_warning_suppressed_for_marimo(self):
        """When marimo backend is used, the plot_style warning is NOT shown."""
        matplotlib.use("Agg")

        with (
            patch("hyperspy.drawing.mpl_he._is_marimo_backend", return_value=True),
            patch("hyperspy.drawing.mpl_he._is_widget_backend", return_value=False),
            patch("hyperspy.drawing.mpl_he._marimo_interactive_display"),
        ):
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                s = hs.signals.Signal1D(np.random.random((5, 100)))
                s.plot(plot_style="vertical")

            warning_msgs = [str(w.message) for w in record]
            assert not any("plot_style" in msg for msg in warning_msgs)

    def test_warning_shown_for_neither_backend(self):
        """When neither marimo nor ipympl is active, warning is shown."""
        matplotlib.use("Agg")

        with (
            patch("hyperspy.drawing.mpl_he._is_marimo_backend", return_value=False),
            patch("hyperspy.drawing.mpl_he._is_widget_backend", return_value=False),
        ):
            with pytest.warns(UserWarning, match="plot_style"):
                s = hs.signals.Signal1D(np.random.random((5, 100)))
                s.plot(plot_style="vertical")

    def test_marimo_detection_returns_false_by_default(self):
        """Without mocking, _is_marimo_backend should return False."""
        # marimo is installed but we're not inside a marimo notebook
        assert _is_marimo_backend() is False


# ── blit-cleanup fix ──────────────────────────────────────────────────


class TestMarimoBrokenBlitFix:
    """Regression tests for the blit-callback / ROI-visibility bug.

    Matplotlib shares a single callback registry across all canvases
    attached to the same figure.  When _marimo_interactive_display swaps
    the canvas to WebAgg (supports_blit=False), the _on_blit_draw handler
    registered during the original Agg setup remains active.  On the next
    draw() call _on_blit_draw fires, re-draws animated artists (AxesImage
    etc.) on top of any non-animated patches added after plot() returns —
    erasing ROI widgets.  The fix disconnects the handler and de-animates
    all artists so WebAgg renders them correctly.
    """

    def _setup_signal_and_display(self):
        """Return signal after calling _marimo_interactive_display."""
        matplotlib.use("Agg")
        from hyperspy.drawing.mpl_he import _marimo_interactive_display

        with (
            patch("hyperspy.drawing.mpl_he._is_marimo_backend", return_value=True),
            patch("hyperspy.drawing.mpl_he._is_widget_backend", return_value=False),
            patch("hyperspy.drawing.mpl_he._marimo_interactive_display"),
        ):
            s = hs.signals.Signal2D(
                np.arange(512 * 512).reshape(512, 512).astype(float)
            )
            s.plot()

        # Call the real _marimo_interactive_display with mo.mpl.interactive
        # and mo.output.append mocked so no actual marimo context is needed.
        with (
            patch("marimo.mpl.interactive", return_value=MagicMock()),
            patch("marimo.output.append"),
        ):
            _marimo_interactive_display(s._plot)

        return s

    def test_blit_draw_event_disconnected_after_display(self):
        """_on_blit_draw must be disconnected after _marimo_interactive_display."""
        s = self._setup_signal_and_display()
        assert s._plot.signal_plot._draw_event_cid is None

    def test_animated_artists_cleared_after_display(self):
        """All animated artists must be de-animated after _marimo_interactive_display."""
        s = self._setup_signal_and_display()
        fig = s._plot.signal_plot.figure
        animated = [a for ax in fig.axes for a in ax.get_children() if a.get_animated()]
        assert animated == [], f"Still animated: {animated}"


# ── _marimo_display_figure ───────────────────────────────────────────


class TestMarimoDisplayFigure:
    def test_calls_interactive_and_append_when_in_marimo(self):
        fig = MagicMock(name="Figure")
        with (
            patch("hyperspy.drawing.mpl_he._is_marimo_backend", return_value=True),
            patch("marimo.mpl.interactive") as mock_interactive,
            patch("marimo.output.append") as mock_append,
        ):
            mock_interactive.return_value = "mocked-html"
            _marimo_display_figure(fig)

        mock_interactive.assert_called_once_with(fig)
        mock_append.assert_called_once_with("mocked-html")

    def test_does_nothing_when_not_in_marimo(self):
        fig = MagicMock(name="Figure")
        with (
            patch("hyperspy.drawing.mpl_he._is_marimo_backend", return_value=False),
            patch("marimo.mpl.interactive") as mock_interactive,
            patch("marimo.output.append") as mock_append,
        ):
            _marimo_display_figure(fig)

        mock_interactive.assert_not_called()
        mock_append.assert_not_called()


# ── Standalone plot functions ─────────────────────────────────────────


class TestMarimoStandalonePlots:
    @pytest.mark.parametrize("plot_func", ["plot_images", "plot_spectra"])
    def test_marimo_display_called_when_fig_created(self, plot_func):
        """When fig is not provided, _marimo_display_figure should be called."""
        matplotlib.use("Agg")
        import hyperspy.api as hs

        signal = hs.signals.Signal1D(np.arange(100))
        if plot_func == "plot_images":
            signal2d = hs.signals.Signal2D(np.random.random((7, 9, 11, 13)))
            args = ([signal2d],)
        else:
            args = ([signal],)

        with (
            patch("hyperspy.drawing.mpl_he._marimo_display_figure") as mock_display,
        ):
            getattr(hs.plot, plot_func)(*args)

        mock_display.assert_called_once()

    def test_plot_images_no_display_when_fig_provided(self):
        """When fig is passed in, _marimo_display_figure should NOT be called."""
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: I001

        signal2d = hs.signals.Signal2D(np.random.random((7, 9, 11, 13)))
        fig = plt.figure()

        with (
            patch("hyperspy.drawing.mpl_he._marimo_display_figure") as mock_display,
        ):
            hs.plot.plot_images([signal2d], fig=fig)
            plt.close(fig)

        mock_display.assert_not_called()

    def test_plot_spectra_no_display_when_fig_provided(self):
        """When fig is passed in, _marimo_display_figure should NOT be called."""
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: I001

        signal = hs.signals.Signal1D(np.arange(100))
        fig = plt.figure()

        with (
            patch("hyperspy.drawing.mpl_he._marimo_display_figure") as mock_display,
        ):
            hs.plot.plot_spectra([signal], fig=fig)
            plt.close(fig)

        mock_display.assert_not_called()


# ── Signal method plots ───────────────────────────────────────────────


class TestMarimoSignalFactorPlots:
    def test_factors_same_window_marimo_display(self):
        """_plot_factors_or_pchars same_window=True should display single figure."""
        matplotlib.use("Agg")
        factors = np.random.random((100, 4))
        s = hs.signals.Signal1D(np.random.random((5, 100)))

        with (
            patch("hyperspy.drawing.mpl_he._marimo_display_figure") as mock_display,
            patch("hyperspy.drawing.mpl_he._is_marimo_backend", return_value=True),
        ):
            s._plot_factors_or_pchars(factors, same_window=True, per_row=2)

        # single figure displayed (same_window=True returns f, not fig_list)
        mock_display.assert_called_once()

    def test_factors_separate_windows_marimo_display(self):
        """_plot_factors_or_pchars same_window=False should display each figure."""
        matplotlib.use("Agg")
        factors = np.random.random((100, 3))
        s = hs.signals.Signal1D(np.random.random((5, 100)))

        with (
            patch("hyperspy.drawing.mpl_he._marimo_display_figure") as mock_display,
            patch("hyperspy.drawing.mpl_he._is_marimo_backend", return_value=True),
        ):
            s._plot_factors_or_pchars(factors, same_window=False)

        # 3 figures in separate windows
        assert mock_display.call_count == 3

    def test_loadings_same_window_marimo_display(self):
        """_plot_loadings same_window=True should display single figure."""
        matplotlib.use("Agg")
        # Loadings shape: (n_components, navigation_size) — must match
        # navigation dimension (5) of the signal.
        loadings = np.random.random((5, 5))
        s = hs.signals.Signal1D(np.random.random((5, 100)))

        with (
            patch("hyperspy.drawing.mpl_he._marimo_display_figure") as mock_display,
            patch("hyperspy.drawing.mpl_he._is_marimo_backend", return_value=True),
        ):
            s._plot_loadings(loadings, comp_ids=range(5), same_window=True, per_row=2)

        mock_display.assert_called_once()

    def test_no_marimo_display_when_not_in_marimo(self):
        """_plot_factors_or_pchars should not call display when not in marimo."""
        matplotlib.use("Agg")
        factors = np.random.random((100, 2))
        s = hs.signals.Signal1D(np.random.random((5, 100)))

        with (
            patch("hyperspy.drawing.mpl_he._marimo_display_figure") as mock_display,
            patch("hyperspy.drawing.mpl_he._is_marimo_backend", return_value=False),
        ):
            s._plot_factors_or_pchars(factors, same_window=True)

        mock_display.assert_not_called()


# ── MVA plots ─────────────────────────────────────────────────────────


class TestMarimoMVAPlots:
    def test_cumulative_explained_variance_marimo_display(self):
        """plot_cumulative_explained_variance_ratio should display in marimo."""
        matplotlib.use("Agg")
        s = hs.signals.Signal1D(np.random.random((5, 100)))
        s.decomposition(algorithm="SVD")
        assert s.learning_results.explained_variance is not None

        with (
            patch("hyperspy.drawing.mpl_he._marimo_display_figure") as mock_display,
        ):
            s.plot_cumulative_explained_variance_ratio()

        mock_display.assert_called_once()

    def test_cluster_analysis_marimo_display(self):
        """plot_cluster_metric should display in marimo."""
        matplotlib.use("Agg")
        s = hs.signals.Signal1D(np.random.random((5, 100)))
        s.learning_results.cluster_metric_data = np.random.random(10)
        s.learning_results.cluster_metric_index = list(range(2, 12))
        s.learning_results.cluster_metric = "silhouette"
        s.learning_results.number_of_clusters = None
        s.learning_results.estimated_number_of_clusters = None

        with (
            patch("hyperspy.drawing.mpl_he._marimo_display_figure") as mock_display,
        ):
            s.plot_cluster_metric()

        mock_display.assert_called_once()

    def test_explained_variance_no_display_when_fig_provided(self):
        """plot_explained_variance_ratio should NOT display when fig passed in."""
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: I001

        s = hs.signals.Signal1D(np.random.random((5, 100)))
        s.decomposition(algorithm="SVD")
        assert s.learning_results.explained_variance is not None

        fig = plt.figure()
        with (
            patch("hyperspy.drawing.mpl_he._marimo_display_figure") as mock_display,
        ):
            s.plot_explained_variance_ratio(fig=fig)
            plt.close(fig)

        mock_display.assert_not_called()


# ── Image shift estimation ────────────────────────────────────────────


class TestMarimoImageShift:
    def test_estimate_image_shift_marimo_display(self):
        """estimate_shift2D should use marimo display when plot=True."""
        matplotlib.use("Agg")
        s = hs.signals.Signal2D(np.random.random((2, 2, 7, 11)))
        # 2*2 = 4 navigation positions, each creates a figure

        with (
            patch("hyperspy.drawing.mpl_he._marimo_display_figure") as mock_display,
        ):
            s.estimate_shift2D(plot=True)

        # One figure per navigation position (2*2 = 4)
        assert mock_display.call_count == 4

    def test_estimate_image_shift_no_display_when_plot_false(self):
        """estimate_shift2D should NOT display when plot=False."""
        matplotlib.use("Agg")
        s = hs.signals.Signal2D(np.random.random((2, 2, 7, 11)))

        with (
            patch("hyperspy.drawing.mpl_he._marimo_display_figure") as mock_display,
        ):
            s.estimate_shift2D(plot=False)

        mock_display.assert_not_called()
