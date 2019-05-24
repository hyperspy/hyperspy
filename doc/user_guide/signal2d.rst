
Signal2D Tools
**************

The methods described in this section are only available for two-dimensional
signals in the Signal2D class.


.. _signal2D.align:

Two dimensional signal registration (alignment)
-----------------------------------------------

.. versionadded:: 1.4
   ``sub_pixel_factor`` keyword.

The :py:meth:`~._signals.signal2d.Signal2D.align2D` and 
:py:meth:`~._signals.signal2d.Signal2D.estimate_shift2D` methods provide
advanced image alignment functionality. Sub-pixel accuracy can be achieved
by using skimage's upsampled matrix-multiplication DFT method
:ref:`[Guizar2008] <Guizar2008>`—by setting the `sub_pixel_factor` keyword argument—
and/or, for multi-dimensional datasets only, using the statistical method
:ref:`[Schaffer2004] <Schaffer2004>`—by setting the ``reference`` keyword argument to ``"stat"``.

.. _image.crop:

Cropping an image
-----------------

The :py:meth:`~._signals.signal2d.Signal2DTools.crop_image` method crops the
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
:py:func:`~._signals.signal2d.Signal2D.add_ramp` method. The parameters
`ramp_x` and `ramp_y` dictate the slope of the ramp in `x`- and `y` direction,
while the offset is determined by the `offset` parameter. The fulcrum of the
linear ramp is at the origin and the slopes are given in units of the axis
with the according scale taken into account. Both are available via the
:py:class:`~.axes.AxesManager` of the signal.


Peak finding
------------

.. versionadded:: 1.1.2

The :py:meth:`~.signal.Signal2DTools.find_peaks2D` method provides access to a
number of algorithms for that achieve peak finding in two dimensional signals.
The methods available are as follows:

Zaeferrer peak finder
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> s.find_peaks2D(method='zaefferer')

This algorithm was developed by Zaefferer [1]_ and the
implementation here is after the description of the algorithm in the Ph.D.
thesis of Thomas A. White. It is based on a gradient threshold followed by a
local maximum search within a square window, which is moved until it is
centered on the brightest point, which is taken as a peak if it is within a
certain distance of the starting point.

Ball statistical peak finder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> s.find_peaks2D(method='stat')

Developed by Gordon Ball, and described in the Ph.D. thesis of Thomas A.
White, this method is based on finding points which have a statistically
higher value than the surrounding areas, then iterating between smoothing and
binarising until the number of peaks has converged. This method is slow, but
very robust to a variety of image types.

Matrix based peak finding
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> s.find_peaks2D(method='laplacian_of_gaussians')
    >>> s.find_peaks2D(method='difference_of_gaussians')

These methods are essentially wrappers around the
`scikit-image <http://scikit-image
.org/docs/dev/auto_examples/plot_blob.html>`_ Laplacian
of Gaussian and Difference of Gaussian methods, based on stacking the
Laplacian/difference of images convolved with Gaussian kernels of various
standard deviations. Both are very rapid and relatively robust, given
appropriate parameters.

Interactive Parametrization
---------------------------

.. code-block:: python

    >>> s.find_peaks2D_interactive()

Many of the peak finding algorithms implemented here have a number of
tuneable parameters that significantly affect their accuracy and speed. Finding
the correct parameters can be difficult. An interactive tool for the Jupyter
(originally IPython) notebook has been developed to help.

Several widgets are available:

.. figure::  images/widgets.png
   :align: center
   :width: 600

* The method selector is used to compare different methods. The last-set
  parameters are maintained.
* The signal navigator is used where a signal has navigation axes. The
  randomizer will select random indices.
* The parameter adjusters will update the parameters of the method and re-plot
  the new peaks.

.. note:: Some methods take significantly longer than others, particularly
    where there are a large number of peaks to be found. The plotting window
    may be inactive during this time.

References
----------

.. [1] S. Zaefferer, “New developments of computer-aided
   crystallographic analysis in transmission electron microscopy research
   papers,” J. Appl. Crystallogr., vol. 33, no. v, pp. 10–25, 2000.
