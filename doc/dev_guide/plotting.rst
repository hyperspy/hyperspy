.. _plotting-label:

Interactive Plotting
====================
Interactive plotting in hyperspy is handled through `matplotlib` and is primarily driven though
event handling.

Specifically for some signal s when the `index` value for some :class:`~.axes.BaseDataAxis` is changed then
the signal plot is updated to reflect the data at that index.  Each signal has a `__call__` function which
will return the data at the current navigation index.

For lazy signals the __call__ function works slightly differently as the current chunk is cached.  As a result
the `__call__` function first checks if the current chunk is cached and then either computes the chunk where the
navigation index resides or just pulls the value from the cached chunk.

Interactive Markers
===================

:class`~.drawing.markers.Markers` operates in a similar way to signals when the data is
retrieved. The current ``index`` for the signal is used to retrieve the current array of markers at that ``index``.
Additionally, lazy markers are treated similarly where the current chunk for a marker is cached.

Adding new types of markers to hyperspy is relatively simple. Currently hyperspy supports any
:class:`matplotlib.collections.Collection` object. For most common cases this should be sufficient
as matplotlib has a large number of built in collections beyond what is available in hyperspy.

In the event that you want a specific shape that isn't supported you can define a custom
:class:`matplotlib.path.Path` object and then use the :class:`matplotlib.collections.PathCollection`
to add the markers to the plot. Currently there is no support for saving Path based markers but that can
be added if there are specific use cases.

Many times when annotating 1-D Plots you want to add markers which are relative to the data.  For example
you may want to add a line which goes from [0,y] where y is the value at x.  To do this you can set the
``offset_transform`` to "relative" or the ``transfrom`` to relative.

.. code::

    >>> s = hs.signals.Signal1D(np.random.rand(3, 15))
    >>> from matplotlib.collections import LineCollection
    >>> m = hs.plot.markers.Lines(segments=[[[2,0],[2,1.0]]], transform = "relative")
    >>> s.plot()
    >>> s.add_marker(m)

This marker will create a line at a value=2 which extends from 0 --> 1 and updates as the index changes.
