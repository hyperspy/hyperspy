"""
Arrow markers
=============

"""
#%%
# Create a signal

import hyperspy.api as hs
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng(0)
data = np.ones((50, 100, 100))
s = hs.signals.Signal2D(data)

for axis in s.axes_manager.signal_axes:
    axis.scale = 2*np.pi / 100

#%%
#
# Dynamic Arrow Markers: Changing Length
# ######################################
#
# The first example shows how to change the length of the arrows when changing
# the navigation coordinates

X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
offsets = np.column_stack((X.ravel(), Y.ravel()))

weight = np.cos(np.linspace(0, 4*np.pi, num=50))

U = np.empty(s.axes_manager.navigation_shape, dtype=object)
V = np.empty(s.axes_manager.navigation_shape, dtype=object)
C = np.empty(s.axes_manager.navigation_shape, dtype=object)
for ind in np.ndindex(U.shape):
    U[ind] = np.cos(X).ravel() / 7.5 * weight[ind]
    V[ind] = np.sin(Y).ravel() / 7.5 * weight[ind]
    C[ind] = np.hypot(U[ind], V[ind])

m = hs.plot.markers.Arrows(
    offsets,
    U,
    V,
    C=C
    )

s.plot()
s.add_marker(m)

#%%
#
# Dynamic Arrow Markers: Changing Position
# ########################################
#
# The second example shows how to change the position of the arrows when changing
# the navigation coordinates

X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
U = np.cos(X).ravel() / 7.5
V = np.sin(Y).ravel() / 7.5
C = np.hypot(U, V)

weight_x = np.sin(np.linspace(0, 2*np.pi, num=50))
weight_y = np.cos(np.linspace(0, 2*np.pi, num=50))

offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
for ind in np.ndindex(offsets.shape):
    offsets[ind] = np.column_stack((X.ravel() + weight_x[ind], Y.ravel() + weight_y[ind]))

m = hs.plot.markers.Arrows(
    offsets,
    U,
    V,
    C=C
    )

s.plot()
s.add_marker(m)


#%%
# sphinx_gallery_thumbnail_number = 2
