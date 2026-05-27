"""
Vertical Line Markers
=====================

"""

# %%
# Create a signal

import hyperspy.api as hs
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng(0)
data = rng.random((25, 25, 100))
s = hs.signals.Signal1D(data)

# %%
# This first example shows how to draw 3 static (same position for all
# navigation coordinate) vetical lines

offsets = np.array([10, 20, 40])

m = hs.plot.markers.VerticalLines(
    offsets=offsets,
    linewidth=3,
    colors=['r', 'g', 'b'],
    )

s.plot()
s.add_marker(m)

# %%
#
# Dynamic Line Markers
# ####################
#
# This example shows how to draw dynamic lines markers, whose positions and
# numbers depends on the navigation coordinates

offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
for ind in np.ndindex(offsets.shape):
    offsets[ind] = rng.random(rng.integers(10)) * 100

# Get list of colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

m = hs.plot.markers.VerticalLines(
    offsets=offsets,
    linewidth=5,
    colors=colors,
    )

s.plot()
s.add_marker(m)

# %%
# sphinx_gallery_thumbnail_number = 2
