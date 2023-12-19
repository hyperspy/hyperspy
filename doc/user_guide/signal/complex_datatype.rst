.. _complex_data-label:

Complex datatype
----------------

The HyperSpy :class:`~.api.signals.ComplexSignal` signal class
and its subclasses for 1-dimensional and 2-dimensional data allow the user to
access complex properties like the ``real`` and ``imag`` parts of the data or the
``amplitude`` (also known as the modulus) and ``phase`` (also known as angle or
argument) directly. Getting and setting those properties can be done as
follows:

.. code-block:: python

  >>> s = hs.signals.ComplexSignal1D(np.arange(100) + 1j * np.arange(100))
  >>> real = s.real                   # real is a new HS signal accessing the same data
  >>> s.real = np.random.random(100)  # new_real can be an array or signal
  >>> imag = s.imag                   # imag  is a new HS signal accessing the same data
  >>> s.imag = np.random.random(100)  # new_imag can be an array or signal

It is important to note that `data` passed to the constructor of a
:class:`~.api.signals.ComplexSignal` (or to a subclass), which
is not already complex, will be converted to the numpy standard of
`np.complex`/`np.complex128`. `data` which is already complex will be passed
as is.

To transform a real signal into a complex one use:

.. code-block:: python

    >>> s.change_dtype(complex)

Changing the ``dtype`` of a complex signal to something real is not clearly
defined and thus not directly possible. Use the ``real``, ``imag``,
``amplitude`` or ``phase`` properties instead to extract the real data that is
desired.


Calculate the angle / phase / argument
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :meth:`~.api.signals.ComplexSignal.angle` function
can be used to calculate the angle, which is equivalent to using the ``phase``
property if no argument is used. If the data is real, the angle will be 0 for
positive values and 2$\pi$ for negative values. If the `deg` parameter is set
to ``True``, the result will be given in degrees, otherwise in rad (default).
The underlying function is the :func:`numpy.angle` function.
:meth:`~.api.signals.ComplexSignal.angle` will return
an appropriate HyperSpy signal.


Phase unwrapping
^^^^^^^^^^^^^^^^

With the :meth:`~.api.signals.ComplexSignal.unwrapped_phase`
method the complex phase of a signal can be unwrapped and returned as a new signal.
The underlying method is :func:`skimage.restoration.unwrap_phase`, which
uses the algorithm described in :ref:`[Herraez] <Herraez>`.


.. _complex.argand:

Calculate and display Argand diagram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it is convenient to visualize a complex signal as a plot of its
imaginary part versus real one. In this case so called Argand diagrams can
be calculated using :meth:`~.api.signals.ComplexSignal.argand_diagram`
method, which returns the plot as a :class:`~.api.signals.Signal2D`.
Optional arguments ``size`` and ``display_range`` can be used to change the
size (and therefore resolution) of the plot and to change the range for the
display of the plot respectively. The last one is especially useful in order to
zoom into specific regions of the plot or to limit the plot in case of noisy
data points.

An example of calculation of Aragand diagram is holospy:ref:`shown for electron
holography data <holo.argand-example>`.

Add a linear phase ramp
^^^^^^^^^^^^^^^^^^^^^^^

For 2-dimensional complex images, a linear phase ramp can be added to the
signal via the
:meth:`~.api.signals.ComplexSignal2D.add_phase_ramp` method.
The parameters ``ramp_x`` and ``ramp_y`` dictate the slope of the ramp in `x`-
and `y` direction, while the offset is determined by the ``offset`` parameter.
The fulcrum of the linear ramp is at the origin and the slopes are given in
units of the axis with the according scale taken into account. Both are
available via the :class:`~.axes.AxesManager` of the signal.
