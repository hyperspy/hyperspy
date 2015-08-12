
Spectrum tools
--------------

These methods are only available for Signal object with signal_dimension equal
to one.

.. _spectrum.crop:

Cropping
^^^^^^^^

In addition to cropping using the powerful and compact :ref:`Signal indexing
<signal.indexing>` syntax the following method is available to crop spectra
using a GUI:

The :py:meth:`~.signal.Signal1DTools.crop_spectrum`, method is used to crop the
spectral energy range. If no parameter is passed, a user interface appears in
which to crop the spectrum.

Background removal
^^^^^^^^^^^^^^^^^^

The :py:meth:`~.signal.Signal1DTools.remove_background` method provides
background removal capabilities through both a CLI and a GUI. Current
background type supported are power law, offset, polynomial and gaussian.

Calibration
^^^^^^^^^^^

The :py:meth:`~.signal.Signal1DTools.calibrate` method provides a user
interface to calibrate the spectral axis.

Spectral alignment
^^^^^^^^^^^^^^^^^^

The following methods use sub-pixel cross-correlation or user-provided shifts
to align spectra. They support applying the same transformation to multiple
files.

* :py:meth:`~.signal.Signal1DTools.align1D`
* :py:meth:`~.signal.Signal1DTools.shift1D`

.. _integrate_1D-label:

Integration
^^^^^^^^^^^

The :py:meth:`~.signal.Signal1DTools.integrate_in_range` method provides a GUI
and a CLI to integrate the 1D signal dimension in a given range using the
Simpson's rule.

Data smoothing
^^^^^^^^^^^^^^

The following methods (that include user interfaces when no arguments are
passed) can perform data smoothing with different algorithms:

* :py:meth:`~.signal.Signal1DTools.smooth_lowess`
* :py:meth:`~.signal.Signal1DTools.smooth_tv`
* :py:meth:`~.signal.Signal1DTools.smooth_savitzky_golay`

Spikes removal
^^^^^^^^^^^^^^
.. versionadded:: 0.5

:py:meth:`~._signals.spectrum.Spectrum.spikes_removal_tool` provides an user
interface to remove spikes from spectra.


.. figure::  images/spikes_removal_tool.png
   :align:   center
   :width:   500

   Spikes removal tool.


Other methods
^^^^^^^^^^^^^^

* Apply a hanning taper to the spectra
  :py:meth:`~.signal.Signal1DTools.hanning_taper`
* Find peaks in spectra
  :py:meth:`~.signal.Signal1DTools.find_peaks1D_ohaver`
* Interpolate the spectra in between two positions
  :py:meth:`~.signal.Signal1DTools.interpolate_in_between`
* Convolve the spectra with a gaussian
  :py:meth:`~.signal.Signal1DTools.gaussian_filter`
