.. _signal1D-label:

Signal1D Tools
**************

The methods described in this section are only available for one-dimensional
signals in the Signal1D class.


.. _signal1D.crop:

Cropping
--------

The :meth:`~.api.signals.Signal1D.crop_signal` crops the
the signal object along the signal axis (e.g. the spectral energy range)
*in-place*. If no parameter is passed, a user interface
appears in which to crop the one dimensional signal. For example:

.. code-block:: python

    >>> s = hs.data.two_gaussians()
    >>> s.crop_signal(5, 15) # s is cropped in place

Additionally, cropping in HyperSpy can be performed using the :ref:`Signal
indexing <signal.indexing>` syntax. For example, the following crops a signal
to the 5.0-15.0 region:

.. code-block:: python

    >>> s = hs.data.two_gaussians()
    >>> sc = s.isig[5.:15.] # s is not cropped, sc is a "cropped view" of s

It is possible to crop interactively using :ref:`roi-label`. For example:

.. code-block:: python

    >>> s = hs.data.two_gaussians()
    >>> roi = hs.roi.SpanROI(left=5, right=15)
    >>> s.plot()
    >>> sc = roi.interactive(s)

.. _interactive_signal1d_cropping_image:

.. figure::  images/interactive_signal1d_cropping.png
   :align:   center

   Interactive spectrum cropping using a ROI.


.. _signal1D.remove_background:

Background removal
------------------

.. versionadded:: 1.4
    ``zero_fill`` and ``plot_remainder`` keyword arguments and big speed
    improvements.

The :meth:`~.api.signals.Signal1D.remove_background` method provides
background removal capabilities through both a CLI and a GUI. The GUI displays
an interactive preview of the remainder after background subtraction. Currently,
the following background types are supported: Doniach, Exponential, Gaussian,
Lorentzian, Polynomial, Power law (default), Offset, Skew normal, Split Voigt
and Voigt. By default, the background parameters are estimated using analytical
approximations (keyword argument ``fast=True``). The fast option is not accurate
for most background types - except Gaussian, Offset and Power law -
but it is useful to estimate the initial fitting parameters before performing a
full fit. For better accuracy, but higher processing time, the parameters can
be estimated using curve fitting by setting ``fast=False``.

Example of usage:

.. code-block:: python

    >>> s = exspy.data.EELS_MnFe(add_powerlaw=True) # doctest: +SKIP
    >>> s.remove_background() # doctest: +SKIP

.. figure::  images/signal_1d_remove_background.png
   :align:   center

   Interactive background removal. In order to select the region
   used to estimate the background parameters (red area in the
   figure) click inside the axes of the figure and drag to the right
   without releasing the button.


Calibration
-----------

The :meth:`~.api.signals.Signal1D.calibrate` method provides a user
interface to calibrate the spectral axis.


Alignment
---------

The following methods use sub-pixel cross-correlation or user-provided shifts
to align spectra. They support applying the same transformation to multiple
files.

* :meth:`~.api.signals.Signal1D.align1D`
* :meth:`~.api.signals.Signal1D.shift1D`


.. _integrate_1D-label:

Integration
-----------

To integrate signals use the :meth:`~.api.signals.BaseSignal.integrate1D` method.
Possibly in combination with a :ref:`ROI-label` if interactivity is required.
Otherwise, a signal subrange for integration can also be chosen with the
:attr:`~.api.signals.BaseSignal.isig` method.

.. code-block:: python

    >>> s.isig[0.2:0.5].integrate1D(axis=0) # doctest: +SKIP


Data smoothing
--------------

The following methods (that include user interfaces when no arguments are
passed) can perform data smoothing with different algorithms:

* :meth:`~.api.signals.Signal1D.smooth_lowess`
  (requires ``statsmodels`` to be installed)
* :meth:`~.api.signals.Signal1D.smooth_tv`
* :meth:`~.api.signals.Signal1D.smooth_savitzky_golay`


Spike removal
--------------

:meth:`~.api.signals.Signal1D.spikes_removal_tool` provides an user
interface to remove spikes from spectra. The ``derivative histogram`` allows to
identify the appropriate threshold. It is possible to use this tool
on a specific interval of the data by :ref:`slicing the data
<signal.indexing>`. For example, to use this tool in the signal between
indices 8 and 17:

.. code-block:: python

   >>> s = hs.signals.Signal1D(np.arange(5*10*20).reshape((5, 10, 20)))
   >>> s.isig[8:17].spikes_removal_tool() # doctest: +SKIP


The options ``navigation_mask`` or ``signal_mask`` provide more flexibility in the
selection of the data, but these require a mask (booleen array) as parameter, which needs
to be created manually:

.. code-block:: python

   >>> s = hs.signals.Signal1D(np.arange(5*10*20).reshape((5, 10, 20)))
   
   To get a signal mask, get the mean over the navigation space

   >>> s_mean = s.mean()
   >>> mask = s_mean > 495
   >>> s.spikes_removal_tool(signal_mask=mask) # doctest: +SKIP

.. figure::  images/spikes_removal_tool.png
   :align:   center
   :width:   500

   Spikes removal tool.


Peak finding
------------

A peak finding routine based on the work of T. O'Haver is available in HyperSpy
through the :meth:`~.api.signals.Signal1D.find_peaks1D_ohaver`
method.


Estimate peak width
-------------------

For asymmetric peaks, `fitted functions <model.fitting>` may not provide
an accurate description of the peak, in particular the peak width. The function
:meth:`~.api.signals.Signal1D.estimate_peak_width`
determines the width of a peak at a certain fraction of its maximum value.


Other methods
-------------

* Interpolate the spectra in between two positions
  :meth:`~.api.signals.Signal1D.interpolate_in_between`
* Convolve the spectra with a gaussian
  :meth:`~.api.signals.Signal1D.gaussian_filter`
* Apply a hanning taper to the spectra
  :meth:`~.api.signals.Signal1D.hanning_taper`
