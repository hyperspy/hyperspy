"""
Making Inset Images
===================

This example shows how to make images that overlay the main images.  These can be images overlaid using
some king of transparency or make an inset image.  It can also be useful for something like making something
like a circular color bar to show orientation.
"""

import numpy as np
import hyperspy.api as hs
import matplotlib

# %%
# Making some artificial data

s = hs.signals.Signal2D(np.random.rand(100, 100, 10, 10))
signal2 = hs.signals.Signal2D(np.random.rand(10, 10))

# %%
# Plotting signal2 as an inset image on s

signal2_shape = signal2.axes_manager._signal_shape_in_array

y = np.linspace(
    -0.1, 0.1, signal2_shape[0] + 1
)  # making a mesh grid 20% of the size of the main image
x = np.linspace(
    -0.1, 0.1, signal2_shape[1] + 1
)  # making a mesh grid 20% of the size of the main image
xx, yy = np.meshgrid(x, y)
coords = np.stack([xx, yy], axis=-1)

quad = hs.plot.markers.Markers(
    collection=matplotlib.collections.QuadMesh,
    coordinates=coords,
    array=signal2.data,
    cmap="hsv",
    transform="axes",
    offset_transform="axes",
    offsets=[[0.7, 0.7]],
    alpha=1,
)  # Put in top right corner

s.plot()
s.add_marker(quad)

# %%
#
# Dynamic Inset Markers
# #####################
#
# This example shows how to draw dynamic inset markers, whose position
# depends on the navigation coordinates

s = hs.signals.Signal2D(np.random.rand(100, 100, 10, 10))
signal2 = hs.signals.Signal2D(np.random.rand(100, 100, 10, 10))

signal2_shape = signal2.axes_manager._signal_shape_in_array


def to_ragged(s):
    return s


ragged_signal2 = signal2.map(to_ragged, ragged=True, inplace=False)

y = np.linspace(
    -0.1, 0.1, signal2_shape[0] + 1
)  # making a mesh grid 20% of the size of the main image
x = np.linspace(
    -0.1, 0.1, signal2_shape[1] + 1
)  # making a mesh grid 20% of the size of the main image
xx, yy = np.meshgrid(x, y)
coords = np.stack([xx, yy], axis=-1)

quad = hs.plot.markers.Markers(
    collection=matplotlib.collections.QuadMesh,
    coordinates=coords,
    array=ragged_signal2.data,
    cmap="hsv",
    transform="axes",
    offset_transform="axes",
    offsets=[[0.7, 0.7]],
    alpha=1,
)  # Put in top right corner

s.plot()
s.add_marker(quad)

# %%
#
# Transparent Overlay
# ###################
#
# This example shows how to draw overlay a dynamic image over the signal, whose position
# depends on the navigation coordinates

s = hs.signals.Signal2D(np.random.rand(100, 100, 10, 10))
signal2 = hs.signals.Signal2D(np.random.rand(100, 100, 10, 10))

y = np.linspace(
    0, 1, signal2_shape[0] + 1
)  # making a mesh grid 20% of the size of the main image
x = np.linspace(
    0, 1, signal2_shape[1] + 1
)  # making a mesh grid 20% of the size of the main image
xx, yy = np.meshgrid(x, y)
coords = np.stack([xx, yy], axis=-1)

quad = hs.plot.markers.Markers(
    collection=matplotlib.collections.QuadMesh,
    coordinates=coords,
    array=ragged_signal2.data,
    cmap="hsv",
    transform="axes",
    offset_transform="display",
    offsets=[[0, 0]],
    alpha=0.3,
)  # Put in top right corner

s.plot()
s.add_marker(quad)

# %%
#
# Circular ColorBar
# #################
#
# This example shows how to a circular Color Bar for shows orientation

n = 360  # the number of secants for the mesh
t = np.linspace(0, 2 * np.pi, n)  # theta values
r = np.linspace(0.6, 1, 2)  # radius values change 0.6 to 0 for full circle

y = np.sin(t[:, np.newaxis]) * r[np.newaxis, :] / 10
x = np.cos(t[:, np.newaxis]) * r[np.newaxis, :] / 10
coords = np.stack([x, y], axis=-1)

quad = hs.plot.markers.Markers(
    collection=matplotlib.collections.QuadMesh,
    coordinates=coords,
    array=t[1:],
    cmap="hsv",
    transform="axes",
    offset_transform="axes",
    offsets=[[0.75, 0.8]],
    alpha=1,
)


im = np.random.randint(1, 360, (100, 100))
nav = hs.signals.Signal2D(im).T

s.plot(navigator=nav, navigator_kwds=dict(cmap="hsv", colorbar=False))

s.add_marker(quad, plot_on_signal=False)

# %%
# sphinx_gallery_thumbnail_number = 2
