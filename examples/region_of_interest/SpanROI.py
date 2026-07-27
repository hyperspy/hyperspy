"""
SpanROI on signal axis
======================

Use a SpanROI interactively on a Signal1D.

"""

import hyperspy.api as hs

# Render with anyplotlib so the figures below stay live in the browser.
hs.preferences.Plot.backend = "anyplotlib"

#%%
# Create a signal:
s = hs.data.two_gaussians()

#%%
# Create the roi, here a :py:class:`~.api.roi.SpanROI` for one dimensional ROI:
roi = hs.roi.SpanROI(left=10, right=20)

#%%
# Slice the signal with the ROI. By using the :meth:`~hyperspy.roi.BaseInteractiveROI.interactive`
# function, the output signal ``sliced_signal`` will update automatically.
# The ROI will be added automatically on the signal figure.
#
# Specify the ``axes`` to add the ROI on either the navigation or signal dimension.
#
# The sliced signal is plotted next to the original, with ``autoscale='xv'`` so
# its limits track the ROI. Activate the figures and drag the span on the left
# to see the right-hand plot follow it.

s.plot()
sliced_signal = roi.interactive(s, axes=s.axes_manager.signal_axes)

sliced_signal.plot(autoscale='xv')  # Interactive

# sphinx_gallery_thumbnail_number = 1
