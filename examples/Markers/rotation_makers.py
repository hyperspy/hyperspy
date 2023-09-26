"""
Rotation of markers
===================

This example shows how markers are rotated.

"""
#%%
# Create a signal

import hyperspy.api as hs
import numpy as np

# Create a Signal2D
data = np.ones([100, 100])
s = hs.signals.Signal2D(data)

num = 2
angle = 25
color = ["tab:orange", "tab:blue"]

#%%
# Create the markers, the first and second elements are at 0 and 20 degrees

# Define the position of the markers
offsets = np.array([20*np.ones(num)]*2).T
angles = np.arange(0, angle*num, angle)

m1 = hs.plot.markers.Rectangles(
    offsets=offsets,
    widths=np.ones(num)*20,
    heights=np.ones(num)*10,
    angles=angles,
    facecolor='none',
    edgecolor=color,
    )

m2 = hs.plot.markers.Ellipses(
    offsets=offsets + np.array([0, 20]),
    widths=np.ones(num)*20,
    heights=np.ones(num)*10,
    angles=angles,
    facecolor='none',
    edgecolor=color,
    )

m3 = hs.plot.markers.Squares(
    offsets=offsets + np.array([0, 50]),
    widths=np.ones(num)*20,
    angles=angles,
    facecolor='none',
    edgecolor=color,
    )

#%%
# Plot the signals and add all the markers

s.plot()
s.add_marker([m1, m2, m3])

#%%
# sphinx_gallery_thumbnail_number = 1
