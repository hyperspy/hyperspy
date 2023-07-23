"""
Filled Circle Markers
=====================

"""
import hyperspy.api as hs
import matplotlib as mpl
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng()
s = hs.signals.Signal2D(rng.random((25, 25, 100, 100)))

# Define the position of the circles
offsets = rng.random((10, 2)) * 100

m = hs.plot.markers.Points(
    sizes=10,
    offsets=offsets,
    )

s.plot()
s.add_marker(m)


"""
Dynamic Filled Circle Markers
=============================
"""


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

s.plot()
s.add_marker(m)