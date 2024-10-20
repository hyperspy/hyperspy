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
# Create the ROI, here a :py:class:`~.api.roi.PolygonROI`:
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
# Then we can extract the ROI from the signal and plot it.

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
# The desired ROI can sometimes be disjointed. In this case, it is possible to
# combine several ROI to extract data. To demonstrate this, first create a second
# ROI. By sending a list of vertices to the constructor, the polygon vertices will 
# be initalized to these:

s.plot()
roi2 = hs.roi.PolygonROI([(0.10,0.10), (0.60,0.50), (0.55,0.70), (0.10,0.75)])
roi2.add_widget(s, axes=s.axes_manager.signal_axes)

# %%
# Now that we have two ROIs, `roi` and `roi2`, we can combine them to slice a signal 
# by using the following function:

s_roi_combined = hs.roi.combine_rois(s, [roi, roi2])
s_roi_combined.plot()

# %%
# It is also possible to get a boolean mask from the ROIs, which can be useful for
# interacting with other libraries. You need to supply the signal's `axes_manager`
# to get the correct parameters for creating the mask:

boolean_mask = hs.roi.mask_from_rois([roi, roi2], s.axes_manager)
boolean_mask = hs.signals.Signal2D(boolean_mask)
boolean_mask.plot()