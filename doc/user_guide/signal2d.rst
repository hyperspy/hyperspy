
Signal2D Tools
**************

The methods described in this section are only available for two-dimensional signals in the Signal2D class.

Two dimensional signal registration (alignment)
-----------------------------------------------

.. versionadded:: 0.5

The :py:meth:`~._signals.signal2d.Signal2DTools.align2D` method provides advanced image
alignment functionality, including subpixel alignment.

.. _image.crop:

Cropping an image
-----------------

In addition to cropping using the powerful and compact :ref:`signal.indexing`
the following method is available to crop spectra the familiar top, bottom,
left, right syntax.

* :py:meth:`~._signals.signal2d.Signal2DTools.crop_image`


Add a linear ramp
-----------------

A linear ramp can be added to the signal via the :py:func:`~._signals.signal2d.Signal2D.add_ramp`
method. The parameters `ramp_x` and `ramp_y` dictate the slope of the ramp in `x`- and `y` direction,
while the offset is determined by the `offset` parameter. The fulcrum of the linear ramp is at the origin
and the slopes are given in units of the axis with the according scale taken into account.
Both are available via the :py:class:`~.axes.AxesManager` of the signal.
