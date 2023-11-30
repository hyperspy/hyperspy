"""
Polygon  Markers
================

"""

#%%
# Create a signal

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng(0)
data = np.ones((25, 25, 100, 100))
s = hs.signals.Signal2D(data)

#%%
# This first example shows how to draw static polygon markers using the matplotlib
# PolygonCollection

# Define the vertexes of the polygons
# poylgon1: [[x0, y0], [x1, y1], [x2, y2], [x3, x3]]
# poylgon2: [[x0, y0], [x1, y1], [x2, y2]]
# ...
poylgon1 = [[1, 1], [20, 20], [1, 20], [25, 5]]
poylgon2 = [[50, 60], [90, 40], [60, 40], [23, 60]]

verts = [poylgon1, poylgon2]

m = hs.plot.markers.Polygons(
    verts=verts,
    linewidth=3,
    facecolors=('g',),
    )

s.plot()
s.add_marker(m)

#%%
#
# Dynamic Polygon Markers
# #######################
#
# This example shows how to draw dynamic polygon markers, whose position
# depends on the navigation coordinates

verts = np.empty(s.axes_manager.navigation_shape, dtype=object)
for ind in np.ndindex(verts.shape):
    verts[ind] = rng.random((10, 4, 2)) * 100

# Get list of colors
colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])

m = hs.plot.markers.Polygons(
    verts=verts,
    facecolors=colors,
    linewidth=3,
    alpha=0.6
)
s.plot()
s.add_marker(m)

#%%
# sphinx_gallery_thumbnail_number = 2
