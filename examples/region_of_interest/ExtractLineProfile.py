"""
Extract line profile from image
===============================

Use a ``Line2DROI`` to interactively extract a line profile with a certain width
from an image.

"""

#%%
# Initialize image data as HyperSpy signal:
import hyperspy.api as hs
import scipy
img = hs.signals.Signal2D(scipy.datasets.ascent())

#%%
# Intialize Line-ROI from pixel (90,90) to pixel (200,250) of width 5.
# You can also use calibrated axes units by providing floats instead of integers.
line_roi = hs.roi.Line2DROI(90, 90, 200, 250, 5)

#%%
# Plot the image and display the ROI (creates new signal object):
img.plot(cmap='viridis')
roi1D = line_roi.interactive(img, color='yellow')

#%%
# You can drag and drop the ends of the ROI to adjust.
# Print the (updated) parameters of the ROI:
print('%.3f, %.3f, %.3f, %.3f, %.2f' % (line_roi.x1, line_roi.y1, line_roi.x2, line_roi.y2, line_roi.linewidth))

#%%
# You can also display the same ROI on a second image
# (e.g. to make sure that a profile is well placed on both images).
# In this example, we create a second image by differentiating the original image:
img2 = img.diff(axis=-1)
img2.plot()
roi1D = line_roi.interactive(img2, color='green')

#%%
# Extract data along ROI as new signal:
profile = line_roi(img)
profile

#%%
# Plot the profile:
profile.plot()

#%%
# Extract data along the same ROI from the second image and plot both profiles:
profile2 = line_roi(img2)
hs.plot.plot_spectra([profile, profile2])
# Choose the third figure as gallery thumbnail:
# sphinx_gallery_thumbnail_number = 4

#%%
# Save profile as `.msa` text file:
profile.save('extracted-line-profile.msa', format='XY')
