"""
Rectangle Markers
=================

"""
import hyperspy.api as hs
import matplotlib as mpl
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng()
s = hs.signals.Signal2D(rng.random((25, 25, 100, 100)))

# Define the position of the ellipses
offsets = rng.random((10, 2)) * 100

m = hs.plot.markers.Rectangles(
    widths=(4,),
    heights=(2,),
    offsets=offsets,
    )

"""
Dynamic Rectangle Markers
=========================
"""

# Create a ragged array of offsets
offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
for ind in np.ndindex(offsets.shape):
    offsets[ind] = rng.random((10, 2)) * 100

m = hs.plot.markers.Rectangles(offsets=offsets, widths=(15,), heights=(15,),
                               color="blue", facecolor="none", linewidth=3)


s.plot()
s.add_marker(m)
