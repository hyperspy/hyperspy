"""
Square Markers
==============

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
# This first example shows how to draw static square markers

# Define the position of the squares (start at (0, 0) and increment by 10)
offsets = np.array([np.arange(0, 100, 10)]*2).T

m = hs.plot.markers.Squares(
    offsets=offsets,
    widths=(5,),
    angles=(0,),
    color="orange",

    )
s.plot()
s.add_marker(m)

#%%
#
# Dynamic Square Markers
# #########################
#
# This first example shows how to draw dynamic squres markers, whose
# position, widths and angles depends on the navigation coordinates

s2 = hs.signals.Signal2D(data)

widths = np.empty(s.axes_manager.navigation_shape, dtype=object)
heights = np.empty(s.axes_manager.navigation_shape, dtype=object)
angles = np.empty(s.axes_manager.navigation_shape, dtype=object)
offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)

for index in np.ndindex(offsets.shape):
    widths[index] = rng.random((10, )) * 50
    heights[index] = rng.random((10, )) * 25
    angles[index] = rng.random((10, )) * 180
    offsets[index] = rng.random((10, 2)) * 100

m = hs.plot.markers.Squares(
    offsets=offsets,
    widths=widths,
    angles=angles,
    color="orange",
    facecolor="none",
    linewidth=3
    )


s2.plot()
s2.add_marker(m)

#%%
# sphinx_gallery_thumbnail_number = 4
