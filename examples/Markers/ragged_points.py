"""
Ragged Points
=============

As for ragged signals, the number of markers at each position can vary and this
is done by passing a ragged array to the constructor of the markers.

"""
#%%
# Create a signal

import hyperspy.api as hs
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng(0)
data = np.arange(25*100*100).reshape((25, 100, 100))
s = hs.signals.Signal2D(data)

#%%
# Create the ragged array with varying number of markers for each navigation
# position

offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
for ind in np.ndindex(offsets.shape):
    num = rng.integers(3, 10)
    offsets[ind] = rng.random((num, 2)) * 100

m = hs.plot.markers.Points(
    offsets=offsets,
    facecolor='orange',
    )

s.plot()
s.add_marker(m)

#%%
# sphinx_gallery_thumbnail_number = 2
