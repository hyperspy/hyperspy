Vector Signal Tools
*******************

The methods described in this section are only available for vector signals
signals in the :py:class:`~._signals.vector_signal.BaseVectorSignal`. class.

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