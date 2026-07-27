"""
Interactive integration of one dimensional signal
=================================================

This example shows how to integrate a signal using an interactive ROI.

"""

import hyperspy.api as hs

# Render with anyplotlib so the figures below stay live in the browser.
hs.preferences.Plot.backend = "anyplotlib"

#%%
# Create a signal:
s = hs.data.two_gaussians()

#%%
# Create SpanROI:
roi = hs.roi.SpanROI(left=10, right=20)

#%%
# Now build the whole interactive chain in one go, so that the signal and the
# integrated intensity are drawn side by side and you can watch one drive the
# other. Activate the figures, then drag or resize the span on the left.
#
# The steps are:
#
# 1. plot the signal and slice it with the ROI — because we use
#    :meth:`~hyperspy.roi.BaseInteractiveROI.interactive`, ``sliced_signal``
#    re-slices itself whenever the ROI moves, and the ROI widget is added to
#    the signal figure automatically;
# 2. make a placeholder signal to hold the integrated intensity;
# 3. connect the integration to the ROI with :func:`~.api.interactive`, using
#    ``out`` so the result lands in that placeholder;
# 4. plot the placeholder.

s.plot()
sliced_signal = roi.interactive(s, axes=s.axes_manager.signal_axes)

integrated_sliced_signal = sliced_signal.sum(axis=-1).T
integrated_sliced_signal.metadata.General.title = "Integrated intensity"

hs.interactive(
    sliced_signal.sum,
    axis=sliced_signal.axes_manager.signal_axes,
    event=roi.events.changed,
    recompute_out_event=None,
    out=integrated_sliced_signal,
)

integrated_sliced_signal.plot()  # Interactive

# sphinx_gallery_thumbnail_number = 1
