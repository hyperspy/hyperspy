"""
Rectangle markers
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

m = hs.plot.markers.MarkerCollection(collection_class=mpl.collections.PolyCollection,
                                     offsets=offsets,
                                     verts=np.array([[[-5, -5], [-5, 5], [5, 5], [5, -5]]]),
                                     color="red")
s.plot()
s.add_marker(m)