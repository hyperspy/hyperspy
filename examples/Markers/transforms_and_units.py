"""
Transforms and Units
====================

This example shows how to use both the ``offsets_transform`` and ```transforms``
parameters for markers
"""

#%%
# Create a signal

import hyperspy.api as hs
import numpy as np

signal = hs.signals.Signal1D((np.arange(100) + 1).reshape(10, 10))

#%%
# The first example shows how to draw markers which are relative to some
# 1D signal.  This is how the EDS and EELS Lines are implemented in the
# exspy package.


segments = np.zeros((10, 2, 2)) # line segemnts for realative markers
segments[:, 1, 1] = 1  # set y values end (1 means to the signal curve)
segments[:, 0, 0] = np.arange(10).reshape(10)  # set x for line start
segments[:, 1, 0] = np.arange(10).reshape(10)  # set x for line stop

offsets = np.zeros((10,2)) # offsets for texts positions
offsets[:, 1] = 1  # set y value for text position ((1 means to the signal curve))
offsets[:, 0] = np.arange(10).reshape(10)  # set x for line start

markers = hs.plot.markers.Lines(segments=segments,transform="relative")
texts = hs.plot.markers.Texts(offsets=offsets,
                              texts=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
                              sizes=10,
                              offsets_transform="relative",
                              shift=0.005)  # shift in axes units for some constant displacement
signal.plot()
signal.add_marker(markers)
signal.add_marker(texts)

#%%
# The second example shows how to draw markers which extend to the edges of the
# axes.  This is how the VerticalLines and HorizontalLines markers are implemented.

markers = hs.plot.markers.Lines(segments=segments,
                                transform="xaxis")

signal.plot()
signal.add_marker(markers)

#%%
# The third example shows how an offsets_transform of 'axes' can be used to annotate
# a signal.

offsets = [[.1, .5], ]  # offsets for positions
marker1text = hs.plot.markers.Texts(offsets=np.add(offsets,[[.1,0]]),
                      texts=["sizes=1: transform=`xaxis_scale`"],
                      sizes=2, transform="display", offsets_transform="axes")
marker = hs.plot.markers.Points(offsets=offsets,
                      sizes=1, transform="xaxis_scale", offsets_transform="axes")

offsets = [[.1, .1], ]  # offsets for positions
marker2 = hs.plot.markers.Points(offsets=offsets,
                      sizes=1, transform="yaxis_scale", offsets_transform="axes")

marker2text = hs.plot.markers.Texts(offsets=np.add(offsets,[[.1,0]]),
                      texts=["sizes=1: transform=`yaxis_scale`"],
                      sizes=2, transform="display", offsets_transform="axes")

offsets = [[.1, .8], ]  # offsets for positions
marker3 = hs.plot.markers.Points(offsets=offsets,
                      sizes=300, transform="display", offsets_transform="axes")

marker3text = hs.plot.markers.Texts(offsets=np.add(offsets,[[.1,0]]),
                      texts=["sizes=300: transform=`display`"],
                      sizes=2, transform="display", offsets_transform="axes")
signal.plot()
signal.add_marker(marker)
signal.add_marker(marker1text)
signal.add_marker(marker2)
signal.add_marker(marker2text)
signal.add_marker(marker3)
signal.add_marker(marker3text)

#sphinx_gallery_thumbnail_number = 2
