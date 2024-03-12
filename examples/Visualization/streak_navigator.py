"""
Streak Navigator
================

Use a `streak image <https://en.wikipedia.org/wiki/Streak_camera>`_ as a navigator:
the navigator will have the signal axis for the horizontal axis and the last navigation
dimension for the vertical axis.

"""

import hyperspy.api as hs

#%%
# Create a signal:
s = hs.data.two_gaussians()

#%%
# Plot the navigator with a streak image:
s.plot(navigator="streak")
