.. _plotting-label:

Interactive Plotting
====================
Interactive plotting in hyperspy is handled through `matplotlib` and is primarily driven though
event handling.

Specifically for some signal s when the `index` value for some :py:class:`~.axes.BaseDataAxis` is changed then
the signal plot is updated to reflect the data at that index.  Each signal has a `__call__` function which
will return the data at the current navigation index.

For lazy signals the __call__ function works slightly differently as the current chunk is cached.  As a result
the `__call__` function first checks if the current chunk is cached and then either computes the chunk where the
navigation index resides or just pulls the value from the cached chunk.

Interactive MarkerCollections
=============================

:py:class`~.drawing.marker_collections.MarkerCollections` operates in a similar way to signals when the data is
retrieved. The current `index` for the signal is used to retrieve the current array of markers at that `index`.
Additionally, lazy markers are treated similarly where the current chunk for a marker is cached.

One special type of marker is the :py:class:`~.drawing._markers.relative_collection.RelativeCollection`.

This is used to add markers relative to the data in a 1-D Plot.  For example if you want to add lines which go from
[0,y] where y is the value at x then you can

>>> s = hs.signals.Signal1D(np.random.rand(3, 15))
>>> from matplotlib.collections import LineCollection
>>> m = hs.plot.markers.RelativeMarkers(segments = [[[2,0],[2,1.0]]], collection_class=LineCollection)
>>> s.plot()
>>> s.add_marker(m)

This marker will create a line at index 2 which extends from 0 --> 1 and updates as the index changes.

Unlike the case above with uses ragged or precomputed lazy markers RelativeMarkers are computed when the
index is changed saving some extra computation as well as speeding up plotting.

This is used to add markers relative to the data in a 1-D Plot.  For example if you want to add lines which go from
[0,y] where y is the value at x then you can

>>> s = hs.signals.Signal1D(np.random.rand(3, 15))
>>> from matplotlib.collections import LineCollection
>>> m = hs.plot.markers.RelativeMarkers(segments = [[[2,0],[2,1.0]]], collection_class=LineCollection)
>>> s.plot()
>>> s.add_marker(m)

This marker will create a line at index 2 which extends from 0 --> 1 and updates as the index changes.

Unlike the case above with uses ragged or precomputed lazy markers RelativeMarkers are computed when the
index is changed saving some extra computation as well as speeding up plotting.

This is used to add markers relative to the data in a 1-D Plot.  For example if you want to add lines which go from
[0,y] where y is the value at x then you can

>>> s = hs.signals.Signal1D(np.random.rand(3, 15))
>>> from matplotlib.collections import LineCollection
>>> m = hs.plot.markers.RelativeCollection(segments = [[[2,0],[2,1.0]]], collection_class=LineCollection)
>>> s.plot()
>>> s.add_marker(m)

This marker will create a line at index 2 which extends from 0 --> 1 and updates as the index changes.

Unlike the case above with uses ragged or precomputed lazy markers RelativeMarkers are computed when the
index is changed saving some extra computation as well as speeding up plotting.