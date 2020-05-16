
Signal2D Tools
**************

The methods described in this section are only available for two-dimensional
signals in the :py:class:`~._signals.signal2d.Signal2D`. class.

.. _signal2D.align:

Signal registration and alignment
---------------------------------

The :py:meth:`~._signals.signal2d.Signal2D.align2D` and
:py:meth:`~._signals.signal2d.Signal2D.estimate_shift2D` methods provide
advanced image alignment functionality.

.. code-block:: python

    # Estimate shifts, then align the images
    >>> shifts = s.estimate_shift2D()
    >>> s.align2D(shifts=shifts)

    # Estimate and align in a single step
    >>> s.align2D()

.. warning::

    ``s.align2D()`` will modify the data **in-place**. If you don't want
    to modify your original data, first take a copy before aligning.

Sub-pixel accuracy can be achieved in two ways:

* `scikit-image's <https://scikit-image.org/>`_ upsampled matrix-multiplication DFT method
  :ref:`[Guizar2008] <Guizar2008>`, by setting the ``sub_pixel_factor``
  keyword argument
* for multi-dimensional datasets only, using the statistical
  method :ref:`[Schaffer2004] <Schaffer2004>`, by setting the ``reference``
  keyword argument to ``"stat"``

.. code-block:: python

    # skimage upsampling method
    >>> shifts = s.estimate_shift2D(sub_pixel_factor=20)

    # stat method
    >>> shifts = s.estimate_shift2D(reference="stat")

    # combined upsampling and statistical method
    >>> shifts = s.estimate_shift2D(reference="stat", sub_pixel_factor=20)

If you have a large stack of images, you can perform the image alignment step in
parallel by passing ``parallel=True``. You can control the number of threads used
with the ``max_workers`` argument. See the :ref:`map documentation <parallel-map-label>`
for more information.

.. code-block:: python

    # Estimate shifts
    >>> shifts = s.estimate_shift2D()

    # Align images in parallel using 4 threads
    >>> s.align2D(shifts=shifts, parallel=True, max_workers=4)

.. _signal2D.crop:

Cropping an image
-----------------

The :py:meth:`~._signals.signal2d.Signal2D.crop_image` method crops the
image *in-place* e.g.:

.. code-block:: python

    >>> im = hs.datasets.example_signals.object_hologram()
    >>> imc = im.crop(left=120, top=300, bottom=560) # im is cropped in-place


Cropping in HyperSpy is performed using the :ref:`Signal indexing
<signal.indexing>` syntax. For example, to crop an image:

.. code-block:: python

    >>> im = hs.datasets.example_signals.object_hologram()
    >>> # im is not cropped, imc is a "cropped view" of im
    >>> imc = im.isig[120.:, 300.:560.]


It is possible to crop interactively using :ref:`roi-label`. For example:

.. code-block:: python

    >>> im = hs.datasets.example_signals.object_hologram()
    >>> roi = hs.roi.RectangularROI(left=120, right=460., top=300, bottom=560)
    >>> im.plot()
    >>> imc = roi.interactive(im)
    >>> imc.plot()


.. _interactive_signal2d_cropping_image:

.. figure::  images/interactive_signal2d_cropping.png
   :align:   center

   Interactive image cropping using a ROI.



Add a linear ramp
-----------------

A linear ramp can be added to the signal via the
:py:meth:`~._signals.signal2d.Signal2D.add_ramp` method. The parameters
`ramp_x` and `ramp_y` dictate the slope of the ramp in `x`- and `y` direction,
while the offset is determined by the `offset` parameter. The fulcrum of the
linear ramp is at the origin and the slopes are given in units of the axis
with the according scale taken into account. Both are available via the
:py:class:`~.axes.AxesManager` of the signal.
