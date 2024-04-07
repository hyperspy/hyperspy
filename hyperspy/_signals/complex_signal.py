# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

from functools import wraps

import numpy as np
from packaging.version import Version

from hyperspy._signals.lazy import LazySignal
from hyperspy._signals.signal2d import Signal2D
from hyperspy.docstrings.plot import (
    BASE_PLOT_DOCSTRING,
    BASE_PLOT_DOCSTRING_PARAMETERS,
    COMPLEX_DOCSTRING,
    PLOT2D_KWARGS_DOCSTRING,
)
from hyperspy.docstrings.signal import (
    LAZYSIGNAL_DOC,
    NUM_WORKERS_ARG,
    SHOW_PROGRESSBAR_ARG,
)
from hyperspy.misc.utils import parse_quantity
from hyperspy.signal import BaseSignal

ERROR_MESSAGE_SETTER = (
    "Setting the {} with a complex signal is ambiguous, "
    "use a numpy array or a real signal."
)


PROPERTY_DOCSTRING_TEMPLATE = """Get/set the {} of the data."""


def format_title(thing):
    def title_decorator(func):
        @wraps(func)
        def signal_wrapper(*args, **kwargs):
            signal = func(*args, **kwargs)
            if signal.metadata.General.title:
                title = signal.metadata.General.title
            else:
                title = "Untitled Signal"
            signal.metadata.General.title = f"{thing}({title})"
            return signal

        return signal_wrapper

    return title_decorator


class ComplexSignal(BaseSignal):
    """General signal class for complex data."""

    _dtype = "complex"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # _plot_kwargs store the plot kwargs argument for convenience when
        # plotting ROI in order to use the same plotting options than the
        # original plot
        self._plot_kwargs = {}
        if not np.issubdtype(self.data.dtype, np.complexfloating):
            self.data = self.data.astype(np.complex128)

    @format_title("real")
    def _get_real(self):
        real = self._deepcopy_with_new_data(self.data.real)
        real._assign_subclass()
        return real

    real = property(
        lambda s: s._get_real(),
        lambda s, v: s._set_real(v),
        doc=PROPERTY_DOCSTRING_TEMPLATE.format("real part"),
    )

    def _set_real(self, real):
        if isinstance(real, self.__class__):
            raise TypeError(ERROR_MESSAGE_SETTER.format("real part"))
        elif isinstance(real, BaseSignal):
            real = real.data
        self.data = real + 1j * self.data.imag
        self.events.data_changed.trigger(self)

    @format_title("imag")
    def _get_imag(self):
        imag = self._deepcopy_with_new_data(self.data.imag)
        imag._assign_subclass()
        return imag

    imag = property(
        lambda s: s._get_imag(),
        lambda s, v: s._set_imag(v),
        doc=PROPERTY_DOCSTRING_TEMPLATE.format("imaginary part"),
    )

    def _set_imag(self, imag):
        if isinstance(imag, self.__class__):
            raise TypeError(ERROR_MESSAGE_SETTER.format("imaginary part"))
        elif isinstance(imag, BaseSignal):
            imag = imag.data
        self.data = self.data.real + 1j * imag
        self.events.data_changed.trigger(self)

    @format_title("amplitude")
    def _get_amplitude(self):
        amplitude = self._deepcopy_with_new_data(abs(self.data))
        amplitude._assign_subclass()
        return amplitude

    amplitude = property(
        lambda s: s._get_amplitude(),
        lambda s, v: s._set_amplitude(v),
        doc=PROPERTY_DOCSTRING_TEMPLATE.format("amplitude"),
    )

    def _set_amplitude(self, amplitude):
        if isinstance(amplitude, self.__class__):
            raise TypeError(ERROR_MESSAGE_SETTER.format("amplitude"))
        elif isinstance(amplitude, BaseSignal):
            amplitude = amplitude.data.real
        self.data = amplitude * np.exp(1j * np.angle(self.data))
        self.events.data_changed.trigger(self)

    @format_title("phase")
    def _get_phase(self):
        phase = self._deepcopy_with_new_data(np.angle(self.data))
        phase._assign_subclass()
        return phase

    phase = property(
        lambda s: s._get_phase(),
        lambda s, v: s._set_phase(v),
        doc=PROPERTY_DOCSTRING_TEMPLATE.format("phase"),
    )

    def _set_phase(self, phase):
        if isinstance(phase, self.__class__):
            raise TypeError(ERROR_MESSAGE_SETTER.format("phase"))
        elif isinstance(phase, BaseSignal):
            phase = phase.data
        self.data = abs(self.data) * np.exp(1j * phase)
        self.events.data_changed.trigger(self)

    def change_dtype(self, dtype):
        """Change the data type.

        Parameters
        ----------
        dtype : str or numpy.dtype
            Typecode or data-type to which the array is cast. For complex signals only other
            complex dtypes are allowed. If real valued properties are required use ``real``,
            ``imag``, ``amplitude`` and ``phase`` instead.
        """
        if np.issubdtype(dtype, np.complexfloating):
            self.data = self.data.astype(dtype)
        else:
            raise ValueError(
                "Complex data can only be converted into other complex dtypes!"
            )

    def unwrapped_phase(
        self, wrap_around=False, seed=None, show_progressbar=None, num_workers=None
    ):
        """Return the unwrapped phase as an appropriate HyperSpy signal.

        Parameters
        ----------
        wrap_around : bool or iterable of bool, default False
            When an element of the sequence is  `True`, the unwrapping process
            will regard the edges along the corresponding axis of the image to be
            connected and use this connectivity to guide the phase unwrapping
            process. If only a single boolean is given, it will apply to all axes.
            Wrap around is not supported for 1D arrays.
        seed : numpy.random.Generator, int or None, default None
            Pass to the `rng` argument of the :func:`~skimage.restoration.unwrap_phase`
            function. Unwrapping 2D or 3D images uses random initialization.
            This sets the seed of the PRNG to achieve deterministic behavior.
        %s
        %s

        Returns
        -------
        :class:`~hyperspy.api.signals.BaseSignal` (or subclass)
            The unwrapped phase.

        Notes
        -----
        Uses the :func:`~skimage.restoration.unwrap_phase` function from `skimage`.
        The algorithm is based on Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor,
        and Munther A. Gdeisat, “Fast two-dimensional phase-unwrapping algorithm based on sorting
        by reliability following a noncontinuous path”, Journal Applied Optics,
        Vol. 41, No. 35, pp. 7437, 2002

        """
        import skimage
        from skimage.restoration import unwrap_phase

        kwargs = {}
        if Version(skimage.__version__) >= Version("0.21"):
            kwargs["rng"] = seed
        else:
            kwargs["seed"] = seed

        phase = self.phase
        phase.map(
            unwrap_phase,
            wrap_around=wrap_around,
            show_progressbar=show_progressbar,
            ragged=False,
            num_workers=num_workers,
            **kwargs,
        )
        phase.metadata.General.title = f"unwrapped {phase.metadata.General.title}"
        return phase  # Now unwrapped!

    unwrapped_phase.__doc__ %= (SHOW_PROGRESSBAR_ARG, NUM_WORKERS_ARG)

    def _get_current_data(
        self, axes_manager=None, power_spectrum=False, fft_shift=False, as_numpy=None
    ):
        value = super()._get_current_data(
            axes_manager=axes_manager, fft_shift=fft_shift, as_numpy=as_numpy
        )
        if power_spectrum:
            value = abs(value) ** 2
        return value

    def plot(
        self,
        power_spectrum=False,
        representation="cartesian",
        same_axes=True,
        fft_shift=False,
        navigator="auto",
        axes_manager=None,
        norm="auto",
        **kwargs,
    ):
        """%s
        %s
        %s
        %s

        """
        if norm == "auto":
            norm = "log" if power_spectrum else "linear"

        kwargs.update(
            {
                "norm": norm,
                "fft_shift": fft_shift,
                "navigator": navigator,
                "axes_manager": self.axes_manager,
            }
        )
        if representation == "cartesian":
            if (
                same_axes and self.axes_manager.signal_dimension == 1
            ) or power_spectrum:
                kwargs["power_spectrum"] = power_spectrum
                super().plot(**kwargs)
            else:
                self.real.plot(**kwargs)
                self.imag.plot(**kwargs)
        elif representation == "polar":
            if same_axes and self.axes_manager.signal_dimension == 1:
                amp = self.amplitude
                amp.change_dtype("complex")
                amp.imag = self.phase
                amp.plot(**kwargs)
            else:
                self.amplitude.plot(**kwargs)
                self.phase.plot(**kwargs)
        else:
            raise ValueError(
                f"{representation} is not a valid input for "
                'representation (use "cartesian" or "polar")!'
            )

        self._plot_kwargs = {
            "power_spectrum": power_spectrum,
            "representation": representation,
            "norm": norm,
            "fft_shift": fft_shift,
            "same_axes": same_axes,
        }

    plot.__doc__ %= (
        BASE_PLOT_DOCSTRING,
        COMPLEX_DOCSTRING,
        BASE_PLOT_DOCSTRING_PARAMETERS,
        PLOT2D_KWARGS_DOCSTRING,
    )

    @format_title("angle")
    def angle(self, deg=False):
        r"""Return the angle (also known as phase or argument).
        If the data is real, the angle is 0
        for positive values and :math:`2\pi` for negative values.

        Parameters
        ----------
        deg : bool, default False
            Return angle in degrees if True, radians if False.

        Returns
        -------
        :class:`~hyperspy.api.signals.BaseSignal`
            The counterclockwise angle from the positive real axis on the complex plane,
            with dtype as numpy.float64.

        """
        angle = self._deepcopy_with_new_data(np.angle(self.data, deg))
        angle.set_signal_type("")

        return angle

    def argand_diagram(self, size=[256, 256], range=None):
        """
        Calculate and plot Argand diagram of complex signal.

        Parameters
        ----------
        size : list of int, optional
            Size of the Argand plot in pixels. Default is [256, 256].
        range : None, numpy.ndarray, default None
            The position of the edges of the diagram with shape (2, 2) or (2,).
            All values outside of this range will be considered outliers and not
            tallied in the histogram. If None use the mininum and maximum values.

        Returns
        -------
        :class:`~hyperspy.api.signals.Signal2D`
            The Argand diagram

        Examples
        --------
        >>> import holospy as holo  # doctest: +SKIP
        >>> hologram = holo.data.Fe_needle_hologram() # doctest: +SKIP
        >>> ref = holo.data.Fe_needle_reference_hologram() # doctest: +SKIP
        >>> w = hologram.reconstruct_phase(ref) # doctest: +SKIP
        >>> w.argand_diagram(range=[-3, 3]).plot() # doctest: +SKIP

        """
        if self._lazy:
            raise NotImplementedError(
                "Argand diagram is not implemented for lazy signals. Use "
                "`compute()` to convert the signal to a regular one."
            )

        for axis in self.axes_manager.signal_axes:
            if not axis.is_uniform:
                raise NotImplementedError(
                    "The function is not implemented for non-uniform axes."
                )
        im = self.imag.data.ravel()
        re = self.real.data.ravel()

        if range:
            if np.asarray(range).shape == (2,):
                range = [[range[0], range[1]], [range[0], range[1]]]
            elif np.asarray(range).shape != (2, 2):
                raise ValueError(
                    "display_range should be array_like, shape(2,2) or shape(2,)."
                )

        argand_diagram, real_edges, imag_edges = np.histogram2d(
            re, im, bins=size, range=range
        )
        argand_diagram = Signal2D(
            argand_diagram.T,
            metadata=self.metadata.as_dictionary(),
        )
        argand_diagram.metadata.General.title = (
            f"Argand diagram of {self.metadata.General.title}"
        )

        if self.real.metadata.Signal.has_item("quantity"):
            quantity_real, units_real = parse_quantity(
                self.real.metadata.Signal.quantity
            )
            argand_diagram.axes_manager.signal_axes[0].name = quantity_real
        else:
            argand_diagram.axes_manager.signal_axes[0].name = "Real"
            units_real = None
        argand_diagram.axes_manager.signal_axes[0].offset = real_edges[0]
        argand_diagram.axes_manager.signal_axes[0].scale = abs(
            real_edges[0] - real_edges[1]
        )

        if self.imag.metadata.Signal.has_item("quantity"):
            quantity_imag, units_imag = parse_quantity(
                self.imag.metadata.Signal.quantity
            )
            argand_diagram.axes_manager.signal_axes[1].name = quantity_imag
        else:
            argand_diagram.axes_manager.signal_axes[1].name = "Imaginary"
            units_imag = None
        argand_diagram.axes_manager.signal_axes[1].offset = imag_edges[0]
        argand_diagram.axes_manager.signal_axes[1].scale = abs(
            imag_edges[0] - imag_edges[1]
        )
        if units_real:
            argand_diagram.axes_manager.signal_axes[0].units = units_real
        if units_imag:
            argand_diagram.axes_manager.signal_axes[1].units = units_imag

        return argand_diagram


class LazyComplexSignal(ComplexSignal, LazySignal):
    """Lazy general signal class for complex data."""

    __doc__ += LAZYSIGNAL_DOC.replace("__BASECLASS__", "ComplexSignal")
