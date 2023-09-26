"""
Circle Markers
==============

"""
#%%
# Create a signal

import hyperspy.api as hs
import numpy as np

# Create a Signal2D with 1 navigation dimension
rng = np.random.default_rng(0)
data = np.ones((50, 100, 100))
s = hs.signals.Signal2D(data)

#%%
# This first example shows how to draw static circles

# Define the position of the circles (start at (0, 0) and increment by 10)
offsets = np.array([np.arange(0, 100, 10)]*2).T

m = hs.plot.markers.Circles(
    sizes=10,
    offsets=offsets,
    edgecolor='r',
    linewidth=5,
    )

s.plot()
s.add_marker(m)

#%%
#
# Dynamic Circle Markers
# ######################
#
# This second example shows how to draw dynamic circles whose position and
# radius change depending on the navigation position

s2 = hs.signals.Signal2D(data)

offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
sizes = np.empty(s.axes_manager.navigation_shape, dtype=object)

for ind in np.ndindex(offsets.shape):
    offsets[ind] = rng.random((5, 2)) * 100
    sizes[ind] = rng.random((5, )) * 10

m = hs.plot.markers.Circles(
    sizes=sizes,
    offsets=offsets,
    edgecolor='r',
    linewidth=5,
    )

s2.plot()
s2.add_marker(m)

#%%
# sphinx_gallery_thumbnail_number = 4
