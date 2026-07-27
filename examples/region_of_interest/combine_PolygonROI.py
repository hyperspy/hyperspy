"""
Combine PolygonROI
==================

Combine several :class:`~.api.roi.PolygonROI`.

"""
#%%
import hyperspy.api as hs

# Render with anyplotlib so the figures below stay live in the browser.
hs.preferences.Plot.backend = "anyplotlib"

#%%
# Create a signal:
s = hs.data.atomic_resolution_image()

#%%
# Create the ROIs, here :class:`~.api.roi.PolygonROI`:
roi = hs.roi.PolygonROI([(2, 4.5), (4.5, 4.5), (4.5, 2), (3.5, 3.5)])
roi2 = hs.roi.PolygonROI([(0.5, 0.5), (1.2, 0.2), (1.5, 1), (0.2, 1.4)])

#%%
# We plot the signal and add the ROIs to the figure using
# :meth:`~hyperspy.roi.BaseInteractiveROI.add_widget`. Now that we have two
# ROIs, ``roi`` and ``roi2``, we can combine them to slice the signal with
# :func:`~.api.roi.combine_rois`, and plot the combination beside the original.
#
# To keep the combination in step with the polygons we wrap the call in
# :func:`~.api.interactive`, listening to both ROIs. Note that this passes the
# ROI events as ``recompute_out_event`` rather than ``event``: the bounding box
# of the combined region — and therefore the shape of the result — changes as
# the polygons move, so the output signal has to be rebuilt rather than
# updated in place.
#
# Activate the figures and drag either polygon to see the extraction follow.

s.plot()
roi.add_widget(s, axes=s.axes_manager.signal_axes)
roi2.add_widget(s, axes=s.axes_manager.signal_axes)

s_roi_combined = hs.interactive(
    hs.roi.combine_rois,
    event=None,
    recompute_out_event=[roi.events.changed, roi2.events.changed],
    signal=s,
    rois=[roi, roi2],
)
s_roi_combined.plot()  # Interactive

# %%
# It is also possible to get a boolean mask from the ROIs, which can be useful for
# interacting with other libraries. You need to supply the signal's ``axes_manager``
# to get the correct parameters for creating the mask:

boolean_mask = hs.roi.mask_from_rois([roi, roi2], s.axes_manager)
boolean_mask = hs.signals.Signal2D(boolean_mask)
boolean_mask.plot()  # Interactive
