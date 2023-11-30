"""
Varying number of arrows per navigation position
================================================

"""
#%%
# Create a signal

import hyperspy.api as hs
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng(0)
data = np.ones((10, 100, 100))
s = hs.signals.Signal2D(data)

for axis in s.axes_manager.signal_axes:
    axis.scale = 2*np.pi / 100

# Select navigation position 5
s.axes_manager.indices = (5, )

#%%
#
# Dynamic Arrow Markers: Changing Length
# ######################################
#
# This example shows how to use the Arrows marker with a varying number of
# arrows per navigation position

# Define the position of the arrows, use ragged array to enable the navigation
# position dependence
offsets= np.empty(s.axes_manager.navigation_shape, dtype=object)
U = np.empty(s.axes_manager.navigation_shape, dtype=object)
V = np.empty(s.axes_manager.navigation_shape, dtype=object)
for ind in np.ndindex(U.shape):
    offsets[ind] = rng.random((ind[0]+1, 2)) * 6
    U[ind] = rng.random(ind[0]+1) * 2
    V[ind] = rng.random(ind[0]+1) * 2

m = hs.plot.markers.Arrows(
    offsets,
    U,
    V,
    )

s.plot()
s.add_marker(m)

#%%
# sphinx_gallery_thumbnail_number = 2
