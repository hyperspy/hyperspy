"""
SpanROI on signal axis
======================

Use a SpanROI interactively on a Signal1D.

"""
#%%
import hyperspy.api as hs

#%%
# Create a signal:
s = hs.data.atomic_resolution_image(size=92)

#%%
# Create the roi, here a :py:class:`~.api.roi.PolygonROI`:
roi = hs.roi.PolygonROI()

#%%
# We can then plot the signal add use `add_widget` to start using the widget.
# Initializing the ROI with no arguments puts you directly into constructing the 
# polygon. Do this by clicking where you want the vertices. Click the first vertex
# to complete the polygon. You can reset the polygon by pressing "Esc" and you
# can move the entire polygon by shift-clicking and dragging.

s.plot()
roi.add_widget(s, axes=s.axes_manager.signal_axes)

#%%
# Then we can extract the roi from the signal and plot it.

s_roi = roi(s)
s_roi.plot()
#%%
# The signal will contain a lot of NaNs, so take this into consideration when
# doing further processing. E.g. use `nanmean` instead of `mean`.

mean_value = s_roi.nanmean(axis=(0,1)).data[0]
print("Mean value in ROI:", mean_value)
# %%
# In some cases, it is easier to choose the area to remove rather than keep.
# By using the `inverted` parameter, everything except the ROI will be retained:

s_roi_inv = roi(s, inverted=True)
s_roi_inv.plot()
mean_value = s_roi_inv.nanmean(axis=(0,1)).data[0]
print("Mean value outside of ROI:", mean_value)
# %%

