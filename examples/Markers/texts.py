"""
Text Markers
============

"""
#%%
# Create a signal

import hyperspy.api as hs
import numpy as np

# Create a Signal2D with 1 navigation dimension
rng = np.random.default_rng(0)
data = np.ones((10, 100, 100))
s = hs.signals.Signal2D(data)

#%%
# This first example shows how to draw static Text markers

# Define the position of the texts
offsets = np.stack([np.arange(0, 100, 10)]*2).T + np.array([5,]*2)
texts = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'f', 'h', 'i'])

m = hs.plot.markers.Texts(
    offsets=offsets,
    texts=texts,
    sizes=3,
    facecolor="black",
    )
s.plot()
s.add_marker(m)

#%%
#
# Dynamic Text Markers
# ####################
#

s2 = hs.signals.Signal2D(data)

offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)

for index in np.ndindex(offsets.shape):
    offsets[index] = rng.random((10, 2)) * 100

m2 = hs.plot.markers.Texts(
    offsets=offsets,
    texts=texts,
    sizes=3,
    facecolor="black",
    )

s2.plot()
s2.add_marker(m2)

#%%
# sphinx_gallery_thumbnail_number = 2
