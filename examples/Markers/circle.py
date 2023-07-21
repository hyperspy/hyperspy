"""
Circle Markers
==============

"""
import hyperspy.api as hs
import matplotlib as mpl
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng()
s = hs.signals.Signal2D(rng.random((25, 25, 100, 100)))

# Define the position of the circles
offsets = rng.random((10, 2)) * 100

m = hs.plot.markers.MarkerCollection(
    collection_class=mpl.collections.CircleCollection,
    sizes=10,
    offsets=offsets,
    facecolor='none',
    edgecolor='r',
    linewidth=5,
    )

s.plot()
s.add_marker(m)


"""
Dynamic Circle Markers
======================
"""

offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
sizes = np.empty(s.axes_manager.navigation_shape, dtype=object)

for ind in np.ndindex(offsets.shape):
    offsets[ind] = rng.random((5, 2)) * 100
    sizes[ind] = rng.random((5, )) * 20

m = hs.plot.markers.MarkerCollection(
    collection_class=mpl.collections.CircleCollection,
    sizes=sizes,
    offsets=offsets,
    facecolor='none',
    linewidth=10,
    )

s.plot()
s.add_marker(m)
