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
