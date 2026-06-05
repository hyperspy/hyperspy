"""
Specifying Matplotlib Axis
==========================

"""

#%%
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np

#%%
#
# Signal2D
# --------
#
# Create two Signal2D

#%%
s = hs.signals.Signal2D(np.arange(100).reshape(10, 10))
s2 = -s

#%%
# Create a list of :external+matplotlib:class:`matplotlib.axis.Axis` using
# :external+matplotlib:func:`matplotlib.pyplot.subplots` and
# specify the second matplotlib axis of the list to :func:`~.api.plot.plot_images`

fig, axs = plt.subplots(ncols=3, nrows=1)
hs.plot.plot_images(s, ax=axs[1], axes_decor="off")

#%%
# The same can be done for a list of signals and axis

fig, axs = plt.subplots(ncols=3, nrows=1)
hs.plot.plot_images([s, s2], ax=axs[1:3], axes_decor="off")

#%%
#
# Signal1D
# --------
#
# The same can be for :class:`~.api.signals.Signal1D`
#
# Create two Signal2D

#%%
s = hs.signals.Signal1D(np.arange(100))
s2 = -s

#%%
# Create an array of :external+matplotlib:class:`matplotlib.axis.Axis` using
# :external+matplotlib:func:`matplotlib.pyplot.subplots` and
# specify the two las matplotlib axis of the second line to :func:`~.api.plot.plot_spectra`

#%%
fig, axs = plt.subplots(ncols=3, nrows=2)
hs.plot.plot_spectra([s, s2], ax=axs[1, 1:3], style="mosaic")
