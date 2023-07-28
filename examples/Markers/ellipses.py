"""
Ellipse markers
===============

"""
import hyperspy.api as hs
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng(0)
s = hs.signals.Signal2D(rng.random((25, 25, 100, 100)))

# Define the position of the ellipses
offsets = rng.random((10, 2)) * 100

m = hs.plot.markers.Ellipses(
    widths=(4,),
    heights=(2,),
    angles=(45,),
    offsets=offsets,
    )

s.plot()
s.add_marker(m)


"""
Dynamic Ellipse Markers
=======================
"""

widths = np.empty(s.axes_manager.navigation_shape, dtype=object)
heights = np.empty(s.axes_manager.navigation_shape, dtype=object)
angles = np.empty(s.axes_manager.navigation_shape, dtype=object)
offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)

for index in np.ndindex(offsets.shape):
    widths[index] = rng.random((10, )) * 4
    heights[index] = rng.random((10, )) * 2
    angles[index] = rng.random((10, )) * 180
    offsets[index] = rng.random((10, 2)) * 100


m = hs.plot.markers.Ellipses(
    widths=widths,
    heights=heights,
    angles=angles,
    offsets=offsets,
    )

s.plot()
s.add_marker(m)
