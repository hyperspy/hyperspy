"""
Line Segment Markers
====================

"""
import hyperspy.api as hs
import matplotlib as mpl
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng()
s = hs.signals.Signal2D(rng.random((25, 25, 100, 100)))

# Define the position of the lines
# line0: (x0, y0), (x1, y1)
# line1: (x0, y0), (x1, y1)
# ...
segments = rng.random((10, 2, 2)) * 100

m = hs.plot.markers.Markers(
    collection_class=mpl.collections.LineCollection,
    segments=segments,
    colors='r',
    )

s.plot()
s.add_marker(m)

"""
Dynamic Line Segment Markers
============================
"""

segments = np.empty(s.axes_manager.navigation_shape, dtype=object)
for ind in np.ndindex(segments.shape):
    segments[ind] = rng.random((10, 2, 2)) * 100

m = hs.plot.markers.Markers(
    collection_class=mpl.collections.LineCollection,
    segments=segments,
    colors='r',
    )

s.plot()
s.add_marker(m)
