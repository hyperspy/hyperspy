"""
Navigator ROI
=============

Use a RectangularROI to take the sum of an area of the navigation space.

"""

import hyperspy.api as hs

# Render with anyplotlib so the figures below stay live in the browser.
hs.preferences.Plot.backend = "anyplotlib"

#%%
# Create a signal:
s = hs.data.two_gaussians()

#%%
# Create the roi, here a :py:class:`~.api.roi.RectangularROI` for the two dimension navigation space:
roi = hs.roi.RectangularROI()

#%%
# Slice signal with roi with the ROI. By using the `interactive` function, the
# output signal ``s_roi`` will update automatically.
# The ROI will be added automatically on the signal figure.
#
# By default, the ROI will be added to the navigation or signal. We specify
# ``recompute_out_event=None`` to avoid redundant computation when changing the ROI

#
# The sum over the ROI is plotted beside the signal, so activating the figures
# and dragging the orange rectangle on the navigator updates the spectrum on
# the right.

s.plot()
s_roi = roi.interactive(s, recompute_out_event=None, color='C1')

# We use the :func:`~.api.interactive` function to compute the sum over the ROI
# interactively:

roi_sum = hs.interactive(s_roi.sum, recompute_out_event=None)
roi_sum.plot()  # Interactive

# sphinx_gallery_thumbnail_number = 1
