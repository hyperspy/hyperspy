"""
PolygonROI
==========

Use a :class:`~.api.roi.PolygonROI` interactively on a :class:`~.api.signals.Signal2D`.

"""
#%%
import hyperspy.api as hs

# Render with anyplotlib so the figures below stay live in the browser.
hs.preferences.Plot.backend = "anyplotlib"

#%%
# Create a signal:
s = hs.data.atomic_resolution_image()

#%%
# Create the ROI, here a :class:`~.api.roi.PolygonROI`:
roi = hs.roi.PolygonROI()

#%%
# Initializing the ROI with no arguments puts you directly into constructing the 
# polygon. Do this by clicking where you want the vertices. Click the first vertex
# to complete the polygon. You can reset the polygon by pressing "Esc" and you
# can move the entire polygon by shift-clicking and dragging.
#
# An alternative to define the :class:`~.api.roi.PolygonROI` interactively is
# to set the vertices of the ROI using the :attr:`~.api.roi.PolygonROI.vertices`.
#
# We can use :meth:`~hyperspy.roi.BaseInteractiveROI.interactive` to add the ROI to the
# figure and get the signal from the ROI.
#
# The extracted signal is plotted next to the original, so that activating the
# figures and dragging the polygon updates the extraction on the right.

s.plot()
roi.vertices = [(2, 4.5), (4.5, 4.5), (4.5, 2), (3, 3)]
s_roi = roi.interactive(s, axes=s.axes_manager.signal_axes)

s_roi.plot()  # Interactive

#%%
# The signal will contain a lot of NaNs, so take this into consideration when
# doing further processing. E.g. use :meth:`~.api.signals.BaseSignal.nanmean`
# instead of :meth:`~.api.signals.BaseSignal.mean`.

mean_value = s_roi.nanmean(axis=(0,1)).data[0]
print("Mean value in ROI:", mean_value)

# %%
# In some cases, it is easier to choose the area to remove rather than keep.
# By using the ``inverted`` parameter, everything except the ROI will be retained:

s_roi_inv = roi(s, inverted=True)
s_roi_inv.plot()  # Interactive
