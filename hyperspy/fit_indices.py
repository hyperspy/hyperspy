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

"""FitIndices — decoupled fitting navigation state.

This module contains :class:`FitIndices`, which owns all per-run fitting
state for :class:`~hyperspy.model.BaseModel`.  It is intentionally
independent of :class:`~hyperspy.axes.AxesManager` and the plot/widget
system so that fitting an index never has side-effects on the plot cursor,
and the plot cursor never accidentally drives a fit.
"""

import inspect
import math
from contextlib import contextmanager
from typing import Generator

import numpy as np

from hyperspy.events import Event, Events


def _serpentine_iter(shape):
    """Re-export so callers don't need to import from axes."""
    from hyperspy.axes import _serpentine_iter as _s

    return _s(shape)


def _flyback_iter(shape):
    """Re-export so callers don't need to import from axes."""
    from hyperspy.axes import _flyback_iter as _f

    return _f(shape)


class FitIndices:
    """Owns all navigation state for a model fitting run.

    ``FitIndices`` is attached to :class:`~hyperspy.model.BaseModel` and is
    the single source of truth for:

    * **which pixel** is currently being fitted (``current_index``),
    * **which pixels** have been fitted already (``fitted`` bool array), and
    * **in what order** pixels are visited (``strategy``).

    It is entirely independent of :class:`~hyperspy.axes.AxesManager` and
    the plot/widget system.

    Parameters
    ----------
    navigation_shape : tuple of int
        The shape of the navigation space in HyperSpy order (x-first).
        Pass ``()`` for a 0-D signal (single spectrum).  Obtained from
        ``model.signal.axes_manager.navigation_shape``.
    strategy : {"serpentine", "flyback"} or iterable of index tuples
        Iteration order. Same semantics as
        :attr:`~hyperspy.axes.AxesManager.iterpath`.
        Default: ``"serpentine"``.

    Attributes
    ----------
    current_index : tuple of int or None
        The index currently being fitted.  ``None`` when no fit is running.
    fitted : numpy.ndarray of bool
        Shape ``navigation_shape[::-1]`` (numpy C order).  ``True`` at
        every position that has been successfully marked with
        :meth:`mark_fitted`.
    events : :class:`~hyperspy.events.Events`
        Container for the events described below.
    events.index_changed : :class:`~hyperspy.events.Event`
        Fires whenever :attr:`current_index` changes.
        Arguments: ``obj`` (this :class:`FitIndices`), ``index`` (new tuple).
    events.fitting_complete : :class:`~hyperspy.events.Event`
        Fires when :attr:`fitted_count` equals :attr:`total_count`.
        Argument: ``obj`` (this :class:`FitIndices`).

    Examples
    --------
    Basic usage — iterate and mark progress:

    >>> fi = FitIndices(navigation_shape=(3, 4), strategy="serpentine")
    >>> for index in fi:
    ...     do_fit(index)
    ...     fi.mark_fitted(index)
    >>> fi.fitted_count
    12

    Resume a partially-completed run:

    >>> fi.reset(clear_fitted=False)   # keep the fitted map
    >>> for index in fi.as_generator(skip_fitted=True):
    ...     do_fit(index)
    ...     fi.mark_fitted(index)
    """

    def __init__(self, navigation_shape, strategy="serpentine"):
        self.navigation_shape = tuple(navigation_shape)
        self._strategy = None  # set via property below (validates)
        self._generator = None  # active iterator; created lazily in __iter__
        self.current_index = None

        # Bool array in numpy (C) order, i.e. shape reversed from HyperSpy order.
        numpy_shape = self.navigation_shape[::-1] if self.navigation_shape else (1,)
        self.fitted = np.zeros(numpy_shape, dtype=bool)

        self.events = Events()
        self.events.index_changed = Event(
            """
            Event that fires whenever ``current_index`` changes.

            Parameters
            ----------
            obj : FitIndices
                The :class:`FitIndices` instance whose index changed.
            index : tuple of int
                The new current index (HyperSpy order, x-first).
            """,
            arguments=["obj", "index"],
        )
        self.events.fitting_complete = Event(
            """
            Event that fires when all navigation positions have been fitted.

            Parameters
            ----------
            obj : FitIndices
                The :class:`FitIndices` instance that completed.
            """,
            arguments=["obj"],
        )

        # Triggers validation and stores the strategy.
        self.strategy = strategy

    # ------------------------------------------------------------------
    # strategy property
    # ------------------------------------------------------------------

    @property
    def strategy(self):
        """Iteration order: ``"serpentine"``, ``"flyback"``, or a custom
        iterable of index tuples.

        Setting this property does **not** immediately create a generator;
        the generator is created lazily when iteration begins via
        :meth:`__iter__` or :meth:`as_generator`.  This means the property
        can be changed between runs without consuming any iterator.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        if isinstance(value, str):
            if value not in ("serpentine", "flyback"):
                raise ValueError(
                    f"strategy must be 'serpentine', 'flyback', or an iterable "
                    f"of index tuples, not {value!r}."
                )
        else:
            try:
                iter(value)
            except TypeError:
                raise TypeError(
                    f"strategy must be 'serpentine', 'flyback', or an iterable, "
                    f"not {type(value)!r}."
                )
            # Validate first element for non-generators (peek cheaply)
            if not (inspect.isgenerator(value) or _is_generator_len(value)):
                try:
                    first = value[0]
                    if not hasattr(first, "__iter__"):
                        raise TypeError(
                            f"Each element of strategy must be an iterable of "
                            f"indices (e.g. a tuple), not {type(first)!r}."
                        )
                    if self.navigation_shape and len(first) != len(
                        self.navigation_shape
                    ):
                        raise ValueError(
                            f"strategy yields index tuples of length "
                            f"{len(first)}, but navigation_shape has "
                            f"{len(self.navigation_shape)} dimensions."
                        )
                except (IndexError, KeyError):
                    pass  # empty custom iterpath — allow it
        self._strategy = value
        # Invalidate any running generator so the next __iter__ picks up the
        # new strategy.
        self._generator = None

    # ------------------------------------------------------------------
    # Progress properties
    # ------------------------------------------------------------------

    @property
    def total_count(self):
        """Total number of positions in the navigation space."""
        return int(np.prod(self.navigation_shape)) if self.navigation_shape else 1

    @property
    def fitted_count(self):
        """Number of positions marked as fitted."""
        return int(self.fitted.sum())

    @property
    def remaining_count(self):
        """Number of positions not yet marked as fitted."""
        return self.total_count - self.fitted_count

    def __len__(self):
        """Best-effort length (used by the progress bar).

        Raises ``TypeError`` for open-ended generators where the length
        cannot be determined.
        """
        if isinstance(self.strategy, str):
            return self.total_count
        try:
            return len(self.strategy)
        except TypeError:
            raise TypeError(
                "Cannot determine length of a generator-based strategy. "
                "Pass an explicit total to the progress bar or use a "
                "list/array-based strategy instead of a generator."
            )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def mark_fitted(self, index):
        """Record that *index* has been fitted successfully.

        Parameters
        ----------
        index : tuple of int
            Navigation index in HyperSpy (x-first) order.  Use ``()`` for
            a 0-D signal.
        """
        if self.navigation_shape:
            self.fitted[tuple(index[::-1])] = True
        else:
            self.fitted[0] = True

        if self.fitted_count == self.total_count:
            self.events.fitting_complete.trigger(obj=self)

    def is_fitted(self, index):
        """Return ``True`` if *index* has been marked fitted.

        Parameters
        ----------
        index : tuple of int
            Navigation index in HyperSpy (x-first) order.
        """
        if self.navigation_shape:
            return bool(self.fitted[tuple(index[::-1])])
        return bool(self.fitted[0])

    def reset(self, clear_fitted=True):
        """Reset iteration state, optionally clearing the fitted mask.

        Parameters
        ----------
        clear_fitted : bool, default ``True``
            When ``True`` the :attr:`fitted` array is zeroed so the next
            run starts fresh.  When ``False`` the array is preserved,
            enabling ``multifit(resume=True)`` to skip already-done pixels.
        """
        self.current_index = None
        self._generator = None
        if clear_fitted:
            self.fitted[:] = False

    # ------------------------------------------------------------------
    # Context manager for temporary index override
    # ------------------------------------------------------------------

    @contextmanager
    def at_index(self, index):
        """Context manager: temporarily set ``current_index`` to *index*.

        Restores the previous ``current_index`` on exit.  Used by
        ``fit(index=some_tuple)`` so a single-pixel fit can set its target
        without corrupting the state of a running ``multifit``.

        Parameters
        ----------
        index : tuple of int
            The index to use inside the context.
        """
        previous = self.current_index
        self.current_index = tuple(index) if index is not None else None
        if self.current_index is not None:
            self.events.index_changed.trigger(obj=self, index=self.current_index)
        try:
            yield self
        finally:
            self.current_index = previous

    # ------------------------------------------------------------------
    # Iteration interface
    # ------------------------------------------------------------------

    def _make_generator(self):
        """Build a fresh generator from the current strategy."""
        if not self.navigation_shape:
            # 0-D signal — single pixel, represented as empty tuple
            return iter([()])
        if self.strategy == "serpentine":
            return _serpentine_iter(self.navigation_shape)
        if self.strategy == "flyback":
            return _flyback_iter(self.navigation_shape)
        # Custom iterable/generator — wrap as iterator
        return iter(self.strategy)

    def __iter__(self):
        """Begin a fresh iteration.  Resets the internal generator."""
        self._generator = self._make_generator()
        return self

    def __next__(self):
        """Advance to the next index, update ``current_index``, fire event."""
        index = next(self._generator)  # propagates StopIteration when done
        self.current_index = index
        self.events.index_changed.trigger(obj=self, index=index)
        return index

    def as_generator(self, mask=None, skip_fitted=False) -> Generator:
        """Yield indices respecting an optional boolean mask and skip logic.

        Parameters
        ----------
        mask : numpy.ndarray of bool or None
            Shape ``navigation_shape[::-1]`` (numpy C order).  Positions
            where ``mask`` is ``True`` are **skipped** (same convention as
            :meth:`~hyperspy.model.BaseModel.multifit`).
        skip_fitted : bool, default ``False``
            When ``True``, positions already marked in :attr:`fitted` are
            skipped.  Enables ``multifit(resume=True)`` semantics.

        Yields
        ------
        index : tuple of int
            Navigation index in HyperSpy (x-first) order.
        """
        for index in self:
            if mask is not None and mask[tuple(index[::-1]) if index else (0,)]:
                continue
            if skip_fitted and self.is_fitted(index):
                continue
            yield index

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _compute_grid(self, max_cells=64):
        """Bin ``fitted`` into a ≤ *max_cells* × *max_cells* display grid.

        Each cell of the returned grid holds the **fraction** [0, 1] of
        navigation positions in the corresponding block that have been
        marked fitted.  Padding positions (when the navigation shape is not
        a multiple of the block size) are ``NaN`` and are excluded from the
        mean.

        Parameters
        ----------
        max_cells : int, default 64
            Maximum number of display cells along each axis.

        Returns
        -------
        grid : numpy.ndarray of float32, shape (display_rows, display_cols)
        current_row : int
            Row in *grid* that contains :attr:`current_index`; -1 if none.
        current_col : int
            Column in *grid* that contains :attr:`current_index`; -1 if none.
        """
        if not self.navigation_shape:
            # 0-D signal — single cell.
            grid = np.array([[float(self.fitted[0])]], dtype=np.float32)
            cr = 0 if self.current_index is not None else -1
            cc = 0 if self.current_index is not None else -1
            return grid, cr, cc

        # -------------------------------------------------------------------
        # Flatten to 2-D: shape (n_flat_rows, n_cols)
        #   The last numpy axis  == the x-axis (HyperSpy dim 0, fastest).
        #   All leading axes are flattened into a single row-axis.
        # -------------------------------------------------------------------
        flat_2d = self.fitted.reshape(-1, self.fitted.shape[-1]).astype(np.float32)
        n_flat_rows, n_cols = flat_2d.shape

        # Block sizes
        bs_row = max(1, math.ceil(n_flat_rows / max_cells))
        bs_col = max(1, math.ceil(n_cols / max_cells))

        disp_rows = math.ceil(n_flat_rows / bs_row)
        disp_cols = math.ceil(n_cols / bs_col)

        # Pad so the array is exactly divisible
        pad_r = disp_rows * bs_row - n_flat_rows
        pad_c = disp_cols * bs_col - n_cols
        if pad_r > 0 or pad_c > 0:
            flat_2d = np.pad(flat_2d, ((0, pad_r), (0, pad_c)), constant_values=np.nan)

        # nanmean over each block
        grid = np.nanmean(
            flat_2d.reshape(disp_rows, bs_row, disp_cols, bs_col),
            axis=(1, 3),
        ).astype(np.float32)

        # -------------------------------------------------------------------
        # Locate current_index in the grid
        # -------------------------------------------------------------------
        cur_row, cur_col = -1, -1
        if self.current_index is not None:
            ix = self.current_index[0]  # HyperSpy x → last numpy axis → col

            if len(self.current_index) > 1:
                # Leading HyperSpy indices (iy, iz,  …) → reverse to numpy
                # order (… iz, iy) to match the C-order leading shape.
                leading_hs = self.current_index[1:]  # (iy, iz, …) HS order
                leading_np = tuple(reversed(leading_hs))
                leading_shape = self.fitted.shape[:-1]  # (nz, ny, …) numpy
                flat_row = int(np.ravel_multi_index(leading_np, leading_shape))
            else:
                flat_row = 0

            cur_row = flat_row // bs_row
            cur_col = ix // bs_col

        return grid, cur_row, cur_col

    # ------------------------------------------------------------------

    def _repr_html_(self):
        """Static HTML snapshot of the fitting progress grid.

        Renders the ≤ 64 × 64 binned grid as an HTML table.  Each cell is
        coloured by the fraction fitted (grey → green).  The cell containing
        :attr:`current_index` gets an amber border.

        This is the *non-widget* fallback used by IDEs and by Jupyter when
        ``anywidget`` is not installed.  For a live-updating display call
        :meth:`display` instead.
        """
        grid, cur_row, cur_col = self._compute_grid()
        rows, cols = grid.shape

        strat = self.strategy if isinstance(self.strategy, str) else "custom"
        pct = (
            f"{self.fitted_count / self.total_count * 100:.1f}"
            if self.total_count
            else "0.0"
        )
        cur_str = (
            str(self.current_index)
            if self.current_index is not None and self.current_index != ()
            else "—"
        )

        # Info bar
        html = (
            '<div style="display:inline-block;font-family:monospace;">'
            '<div style="font-size:12px;padding:3px 6px;background:#f0f0f0;'
            "border:1px solid #ccc;border-bottom:none;"
            'border-radius:3px 3px 0 0;white-space:nowrap;">'
            f"{self.fitted_count}\u00a0/\u00a0{self.total_count} fitted "
            f"({pct}\u00a0%) \u2022 {strat} \u2022 current\u00a0{cur_str}"
            "</div>"
        )

        # Grid table
        CELL = 9  # px
        html += (
            '<table style="border-collapse:collapse;border:1px solid #ccc;'
            "border-radius:0 0 3px 3px;background:#e8e8e8;"
            f'padding:{CELL // 3}px;">'
        )

        for r in range(rows):
            html += "<tr>"
            for c in range(cols):
                frac = float(grid[r, c])
                if math.isnan(frac):
                    frac = 0.0
                # Grey (220,220,220) → Green (76,175,80)
                ri = int(220 + (76 - 220) * frac)
                gi = int(220 + (175 - 220) * frac)
                bi = int(220 + (80 - 220) * frac)
                bg = f"rgb({ri},{gi},{bi})"

                if r == cur_row and c == cur_col:
                    border = "2px solid #ffc107"
                else:
                    border = f"1px solid {bg}"

                html += (
                    f'<td style="width:{CELL}px;height:{CELL}px;'
                    f"background:{bg};border:{border};"
                    'padding:0;margin:1px;"></td>'
                )
            html += "</tr>"

        html += "</table></div>"
        return html

    # ------------------------------------------------------------------

    def display(self):
        """Display a live-updating fit-map widget in Jupyter.

        Connects :attr:`events.index_changed` and
        :attr:`events.fitting_complete` to the widget so the grid redraws
        automatically on every pixel fit.

        Returns
        -------
        widget : FitIndicesWidget or IPython DisplayHandle or None
            * :class:`~hyperspy.viewer.fit_indices_widget.FitIndicesWidget`
              when ``anywidget`` is available.
            * An ``IPython.display.DisplayHandle`` when only IPython is
              available (falls back to static HTML snapshots pushed via
              ``update_display``).
            * ``None`` when neither is available (falls back to
              ``print(self)``).

        Notes
        -----
        The widget stays live as long as its ``index_changed`` callback is
        connected.  The callback is disconnected automatically when
        ``fitting_complete`` fires.  If ``multifit`` raises before all
        pixels are fitted, call ``widget.disconnect()`` (anywidget) or
        ignore the orphaned callback (it is a no-op once the fitting object
        is garbage-collected).

        Examples
        --------
        >>> widget = model.fit_indices.display()   # show grid before fitting
        >>> model.multifit()                       # grid updates live
        """
        # ── Try anywidget first ───────────────────────────────────────────
        try:
            from IPython.display import display as _ipy_display

            from hyperspy.viewer.fit_indices_widget import FitIndicesWidget

            widget = FitIndicesWidget.from_fit_indices(self)
            _ipy_display(widget)
            return widget
        except ImportError:
            pass

        # ── Fall back: IPython display_id with static HTML snapshots ───────
        try:
            from IPython.display import HTML
            from IPython.display import display as _ipy_display

            handle = _ipy_display(HTML(self._repr_html_()), display_id=True)

            def _on_change(obj, **_):
                handle.update(HTML(obj._repr_html_()))

            self.events.index_changed.connect(_on_change, ["obj"])
            self.events.fitting_complete.connect(_on_change, ["obj"])
            return handle
        except ImportError:
            pass

        # ── Last resort: plain text ────────────────────────────────────────
        print(self)
        return None

    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"FitIndices("
            f"navigation_shape={self.navigation_shape}, "
            f"strategy={self.strategy!r}, "
            f"fitted={self.fitted_count}/{self.total_count}, "
            f"current_index={self.current_index})"
        )


# ---------------------------------------------------------------------------
# Small helper
# ---------------------------------------------------------------------------


def _is_generator_len(obj):
    """Return True if obj looks like a GeneratorLen (has __len__ and __iter__
    but is not a plain list/tuple/array).  Used to decide whether we can peek
    at obj[0] safely."""
    return (
        hasattr(obj, "__len__")
        and hasattr(obj, "__iter__")
        and not isinstance(obj, (list, tuple, np.ndarray))
    )
