Masking Data
*************

Hyperspy supports masking data through the use of the hyperspy.api.ma module
This module mirrors the `numpy.ma <https://numpy.org/doc/stable/reference/maskedarray.generic.html>`_
or for the case of a lazy signal the dask.array.ma package.

Similar to the numpy.ma class there are many ways to access and change the mask.
Additionally, we added in the ability to use hyperspy's ROI functionality to decribe
and set a mask.

Something to note is that the behavior of masks for signals loaded into memory
and lazy signals is slightly different. This is mostly because changing values
while slicing in hyperspy is not supported with a lazy signal. However, changing the masks
using the methods in hyperspy.api.ma is supported.

To begin we can walk you through something like masking a beamstop in a
electron diffraction dataset.  In this case there are a couple of options
which will have about the same result. The first is using the :py:meth:`~.misc.ma.masked_below` method
which masks any value below some cutoff.  The second is using the slicing features in hyperspy
with :py:`s.isig[:,:]= hs.ma.masked`.  The final is using an :py:class:`~.roi.RectangularROI` with the
:py:meth:`~.misc.ma.masked_roi` method.  These are all three seen in the code block below.

.. code-block:: python
    import hyperspy.api as hs
    import numpy as np
    hs.signals.Signal


