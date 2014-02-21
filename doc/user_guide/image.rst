
Image tools
-----------

These methods are only available for Signal object with signal_dimension equal
to two.

Image registration (alignment)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 0.5

The :py:meth:`~.signal.Signal2DTools.align2D` method provides advanced image
alignment functionality, including subpixel alignment.

.. _image.crop:

Cropping an image
^^^^^^^^^^^^^^^^^

In addition to cropping using the powerful and compact :ref:`signal.indexing`
the following method is available to crop spectra the familiar top, bottom,
left, right syntax.

* :py:meth:`~.signal.Signal2DTools.crop_image`


