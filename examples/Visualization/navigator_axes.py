"""
Specify Navigator Axes
======================

This example shows how to specify the axes of the navigation space for
the navigator
"""

import hyperspy.api as hs
import numpy as np

#%%
# Create a 5D signal with signal dimension of 2:
shape = (3, 4, 5, 10, 10)
data = np.arange(np.prod(shape)).reshape(shape) + np.random.default_rng().random(shape) * 100
s = hs.signals.Signal2D(data)

#%%
# The signal has 3 navigation axes and 2 signal dimension:
s.axes_manager

#%%
# By default, the first two navigation axes will be used:
s.plot()

#%%
# The axes of the navigation can be specified using the `navigator_axes` argument,
# for example, to select the first (index 0) and third (index 2) axes:
s.plot(navigator_axes=[0, 2])
