"""
Ellipse markers
===============

"""
import hyperspy.api as hs
import matplotlib as mpl
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng()
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

offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
for ind in np.ndindex(offsets.shape):
    offsets[ind] = rng.random((10, 2)) * 100

m = hs.plot.markers.Ellipses(
    widths=(4,),
    heights=(2,),
    angles=(45,),
    offsets=offsets,
    )

s.plot()
s.add_marker(m)
