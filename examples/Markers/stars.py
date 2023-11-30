"""
Star Markers
============

"""

#%%
# Create a signal

import hyperspy.api as hs
import matplotlib as mpl
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng(0)
data = np.ones((25, 25, 100, 100))
s = hs.signals.Signal2D(data)

#%%
# This first example shows how to draw static stars markers using the matplotlib
# StarPolygonCollection

# Define the position of the boxes
offsets = rng.random((10, 2)) * 100

# every other star has a size of 50/100
m = hs.plot.markers.Markers(collection=mpl.collections.StarPolygonCollection,
                            offsets=offsets,
                            numsides=5,
                            color="orange",
                            sizes=(50, 100))
s.plot()
s.add_marker(m)

#%%
#
# Dynamic Star Markers
# ######################
#
# This second example shows how to draw dynamic stars markers, whose position
# depends on the navigation coordinates

# Create a Signal2D with 2 navigation dimensions
s2 = hs.signals.Signal2D(data)

# Create a ragged array of offsets
offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
for ind in np.ndindex(offsets.shape):
    offsets[ind] = rng.random((10, 2)) * 100

m2 = hs.plot.markers.Markers(collection=mpl.collections.StarPolygonCollection,
                            offsets=offsets,
                            numsides=5,
                            color="blue",
                            sizes=(50, 100))

s2.plot()
s2.add_marker(m2)

#%%
# sphinx_gallery_thumbnail_number = 2
