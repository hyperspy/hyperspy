"""
Creating Signal1D from numpy array
===================================

This example demonstrates how to create a HyperSpy Signal1D from a numpy array
and understand the navigation vs signal dimensions concept.
"""

# %%
# Create a numpy array and convert it to a Signal1D
import numpy as np
import hyperspy.api as hs

# Create a random numpy array with shape (10, 20, 100)
# This will become a spectrum image with:
# - Navigation dimensions: (20, 10) 
# - Signal dimension: (100)
my_np_array = np.random.random((10, 20, 100))
s = hs.signals.Signal1D(my_np_array)

# %%
# **Signal1D created from numpy array**
#
# The Signal1D object has been created with the proper dimensional interpretation.
# HyperSpy automatically determines navigation vs signal dimensions.

# %%
# **Understanding array shapes and axes interpretation**
#
# HyperSpy rearranges and interprets the array dimensions for multidimensional data analysis workflows.

# **Array shape comparison:**
# - Original numpy array: (10, 20, 100)
# - HyperSpy signal data: maintains same shape but interprets dimensions
# - Signal representation: (20, 10|100) where | separates navigation from signal

# %%
# **Navigation and signal axes breakdown**
#
# HyperSpy distinguishes between navigation (spatial/scanning) and signal (spectral) dimensions.

# **Navigation axes:** Define the scanning/spatial dimensions
# - These represent the positions where spectra were acquired
# - In this case: 2 navigation axes with sizes 20 and 10

# **Signal axes:** Define the spectral/energy dimension  
# - These represent the measured spectrum at each position
# - In this case: 1 signal axis with size 100 (spectrum channels)

# %%
# Plot the signal - this will show a spectrum at current navigation position
# and a navigator showing the sum over the signal dimension
s.plot()
