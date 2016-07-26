Dielectric function tools
-------------------------

.. versionadded:: 0.7

The :py:class:`~.signals.dielectric_function.DielectricFunction` class inherits from
:py:class:`~.signals.complex_signal.ComplexSignal` and can thus access complex properties.
To convert a :py:class:`~.signals.complex_signal.ComplexSignal` to a
:py:class:`~.signals.dielectric_function.DielectricFunction`, make sure that the signal dimension
and signal type are properly set:

    .. code-block:: python

        >>> s.set_signal_type('DielectricFunction')

Note that :py:class:`~._signals.dielectric_function.DielectricFunction` is complex and therefore
is a subclass of :py:class:`~._signals.complex_signal1d.ComplexSignal1D`.


Number of effective electrons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 0.7

The Bethe f-sum rule gives rise to two definitions of the effective number (see
[Egerton2011]_):

.. math::

   n_{\mathrm{eff1}}\left(-\Im\left(\epsilon^{-1}\right)\right)=\frac{2\epsilon_{0}m_{0}}{\pi\hbar^{2}e^{2}n_{a}}\int_{0}^{E}E'\Im\left(\frac{-1}{\epsilon}\right)dE'

   n_{\mathrm{eff2}}\left(\epsilon_{2}\right)=\frac{2\epsilon_{0}m_{0}}{\pi\hbar^{2}e^{2}n_{a}}\int_{0}^{E}E'\epsilon_{2}\left(E'\right)dE'

where :math:`n_a` is the number of atoms (or molecules) per unit volume of the
sample, :math:`\epsilon_0` is the vacuum permittivity, :math:`m_0` is the
elecron mass and :math:`e` is the electron charge.

The
:py:meth:`~._signals.dielectric_function.DielectricFunction.get_number_of_effective_electrons`
method computes both.

Compute the electron energy-loss signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 0.7

The
:py:meth:`~._signals.dielectric_function.DielectricFunction.get_electron_energy_loss_spectrum`
"naively" computes the single-scattering electron-energy loss spectrum from the
dielectric function given the zero-loss peak (or its integral) and the sample
thickness using:

.. math::

    S\left(E\right)=\frac{2I_{0}t}{\pi
    a_{0}m_{0}v^{2}}\ln\left[1+\left(\frac{\beta}{\theta(E)}\right)^{2}\right]\Im\left[\frac{-1}{\epsilon\left(E\right)}\right]

where :math:`I_0` is the zero-loss peak integral, :math:`t` the sample
thickness, :math:`\beta` the collection semi-angle and :math:`\theta(E)` the
characteristic scattering angle.
