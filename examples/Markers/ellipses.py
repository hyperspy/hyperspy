"""
Ellipse markers
===============

"""
#%%
# Create a signal

import hyperspy.api as hs
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng(0)
data = np.ones((25, 25, 100, 100))
s = hs.signals.Signal2D(data)

#%%
# This first example shows how to draw static ellipses

# Define the position of the ellipses
offsets = rng.random((10, 2)) * 100

m = hs.plot.markers.Ellipses(
    widths=(8,),
    heights=(10,),
    angles=(45,),
    offsets=offsets,
    facecolor="red",
    )

s.plot()
s.add_marker(m)

#%%
#
# Dynamic Ellipse Markers
# #######################
#
# This first example shows how to draw dynamic ellipses, whose position, widths
# heights and angles depends on the navigation coordinates

s2 = hs.signals.Signal2D(data)

widths = np.empty(s.axes_manager.navigation_shape, dtype=object)
heights = np.empty(s.axes_manager.navigation_shape, dtype=object)
angles = np.empty(s.axes_manager.navigation_shape, dtype=object)
offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)

for index in np.ndindex(offsets.shape):
    widths[index] = rng.random((10, )) * 10
    heights[index] = rng.random((10, )) * 7
    angles[index] = rng.random((10, )) * 180
    offsets[index] = rng.random((10, 2)) * 100


m = hs.plot.markers.Ellipses(
    widths=widths,
    heights=heights,
    angles=angles,
    offsets=offsets,
    facecolor="red",
    )

s2.plot()
s2.add_marker(m)

#%%
# sphinx_gallery_thumbnail_number = 2
