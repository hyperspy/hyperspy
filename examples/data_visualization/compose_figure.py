"""
Composing Figure
================

This example shows how to compose a figure using
:func:`~.api.plot.plot_images` and :func:`~.api.plot.plot_spectra`

.. note::
    Composing a figure means handing HyperSpy pre-made
    :class:`matplotlib.axes.Axes`, so this example needs the default
    matplotlib backend and the figures below are static images. Most of the
    gallery renders through the interactive ``anyplotlib`` backend instead.
"""

#%%
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np

#%%
# Create the 1D and 2D signals

#%%
s2D_0 = hs.signals.Signal2D(np.arange(100).reshape(10, 10))
s2D_1 = -s2D_0

s1D_0 = hs.signals.Signal1D(np.arange(100))
s1D_1 = -s1D_0

#%%
# Create an array of :external+matplotlib:class:`matplotlib.axis.Axis` using :external+matplotlib:func:`matplotlib.pyplot.subplots` 

#%%
fig, axs = plt.subplots(ncols=2, nrows=2)
hs.plot.plot_images([s2D_0, s2D_1], ax=axs[:, 0], axes_decor="off")
hs.plot.plot_spectra([s1D_0, s1D_1], ax=axs[:, 1], style="mosaic")
