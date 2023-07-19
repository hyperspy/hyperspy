"""
Star markers
============

"""
import hyperspy.api as hs
import matplotlib as mpl
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng()
s = hs.signals.Signal2D(rng.random((25, 25, 100, 100)))

# Define the position of the boxes
offsets = rng.random((10, 2)) * 100

# every other star is red/blue and has a size of 20/30
m = hs.plot.markers.MarkerCollection(collection_class=mpl.collections.StarPolygonCollection,
                                     offsets=offsets,
                                     numsides=5,
                                     color=("red", "blue",),
                                     sizes=(20,30))
s.plot()
s.add_marker(m)

"""
Dynamic Star Markers
====================
"""

# Create a ragged array of offsets
offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
for ind in np.ndindex(offsets.shape):
    offsets[ind] = rng.random((10, 2)) * 100

m = hs.plot.markers.MarkerCollection(collection_class=mpl.collections.StarPolygonCollection,
                                     offsets=offsets,
                                     numsides=5,
                                     color=("red", "blue",),
                                     sizes=(20, 30))

s.plot()
s.add_marker(m)

