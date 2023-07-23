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

# Define the corners of the rectangles
rectangles = rng.random((10, 4) * 100)

m = hs.plot.markers.Rectangles(rectangles=rectangles,
                               color="red"
                               )

s.plot()
s.add_marker(m)

"""
Dynamic Rectangle Markers
=========================
"""

# Create a ragged array of offsets
rectangles = np.empty(s.axes_manager.navigation_shape, dtype=object)
for ind in np.ndindex(rectangles.shape):
    rectangles[ind] = rng.random((10, 2)) * 100

m = hs.plot.markers.Rectangles(rectangles=rectangles, color="red")

s.plot()
s.add_marker(m)
