"""
Filled Circle Markers
=====================

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
# This first example shows how to draw static filled circles

# Define the position of the circles
offsets = rng.random((10, 2)) * 100

m = hs.plot.markers.Points(
    sizes=20,
    offsets=offsets,
    facecolors="red",
    )

s.plot()
s.add_marker(m)

#%%
#
# Dynamic Filled Circle Markers
# #############################
#
# This second example shows how to draw dynamic filled circles, whose size,
# color and position change depending on the navigation position

s2 = hs.signals.Signal2D(data)

offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
sizes = np.empty(s.axes_manager.navigation_shape, dtype=object)
colors = list(mpl.colors.TABLEAU_COLORS.values())[:10]

for ind in np.ndindex(offsets.shape):
    offsets[ind] = rng.random((10, 2)) * 100
    sizes[ind] = rng.random((10, )) * 50

m = hs.plot.markers.Points(
    sizes=sizes,
    offsets=offsets,
    facecolors=colors,
    )

s2.plot()
s2.add_marker(m)

#%%
# sphinx_gallery_thumbnail_number = 2
