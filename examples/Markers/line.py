"""
Line markers
============

"""
import hyperspy.api as hs
import matplotlib as mpl
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng()
s = hs.signals.Signal2D(rng.random((25, 25, 100, 100)))

# Define the position of the lines
segments = rng.random((10, 2, 2)) * 100

m = hs.plot.markers.MarkerCollection(
    collection_class=mpl.collections.LineCollection,
    segments=segments,
    colors='r',
    )

s.plot()
s.add_marker(m)
