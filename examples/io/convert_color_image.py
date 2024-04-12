"""
Adjust contrast and save RGB images
===================================

This example shows how to adjust the contrast and intensities using
scikit-image and save it as an RGB image.

When saving an RGB image to ``jpg``, only 8 bits are supported and the image
intensity needs to be rescaled to 0-255 before converting to 8 bits,
otherwise, the intensities will be cropped at the value of 255.

"""

import hyperspy.api as hs
import numpy as np
import skimage as ski

#%%
#
# Adjust contrast
# ###############
#
# In hyperspy, color images are defined as Signal1D with the signal dimension
# corresponding to the color channel (red, green and blue)

# The dtype can be changed to a custom dtype, which is convenient to visualise
# the color image
s = hs.signals.Signal1D(ski.data.astronaut())
s.change_dtype("rgb8")
print(s)

#%%
# Display the color image
s.plot()

#%%
# Processing is usually performed on standard dtype (e.g. ``uint8``, ``uint16``), because
# most functions from scikit-image, numpy, scipy, etc. only support standard ``dtype``.
# Convert from RGB to unsigned integer 16 bits
s.change_dtype("uint8")
print(s)

#%%
# Adjust contrast (gamma correction)
s.data = ski.exposure.adjust_gamma(s.data, gamma=0.2)

#%%
#
# Save to ``jpg``
# ###############
#
# Change dtype back to custom dtype ``rgb8``
s.change_dtype("rgb8")

#%%
# Save as jpg
s.save("rgb8_image.jpg", overwrite=True)


#%%
#
# Save ``rgb16`` image to ``jpg``
# ###############################
#
# The last part of this example shows how to save ``rgb16`` to a ``jpg`` file
#
# Create a signal with ``rgb16`` dtype
s2 = hs.signals.Signal1D(ski.data.astronaut().astype("uint16") * 100)

#%%
# To save a color image to ``jpg``, the signal needs to be converted to ``rgb8`` because
# ``jpg`` only support 8-bit RGB
# Rescale intensity to fit the unsigned integer 8 bits (2**8 = 256 intensity level)
s2.data = ski.exposure.rescale_intensity(s2.data, out_range=(0, 255))

#%%
# Now that the values have been rescaled to the 0-255 range, we can convert the data type
# to unsigned integer 8 bit and then ``rgb8`` to be able to save the RGB image in ``jpg`` format
s2.change_dtype("uint8")
s2.change_dtype("rgb8")
s2.save("rgb16_image_saved_as_jpg.jpg", overwrite=True)
