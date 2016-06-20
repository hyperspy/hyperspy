
Signal2D Tools
**************

The methods described in this section are only available for one-dimensional signals in the Signal1D class.

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

Peak finding
------------

.. versionadded:: 1.0.0

The :py:meth:`~.signal.Signal2DTools.find_peaks2D` method provides access to a
number of algorithms for that achieve peak finding in two dimensional signals.
The methods available are as follows:

Zaeferrer peak finder
^^^^^^^^^^^^^^^^^^^^^


Ball statistical peak finder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Matrix based peak finding
^^^^^^^^^^^^^^^^^^^^^^^^^


Masiel peak finder
^^^^^^^^^^^^^^^^^^
