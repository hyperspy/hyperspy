"""
Export single spectrum
======================

Creates a single spectrum image, saves it and plots it:

1. Create a single sprectrum using `Signal1D` signal.
2. Save signal as a msa file
3. Plot the signal using the `plot` method
4. Save the figure as a png file

"""

# Set the matplotlib backend of your choice, for example
# %matploltib qt
import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt

s = hs.signals.Signal1D(np.random.rand(1024))

# Export as msa file, which is a format very similar to a csv format but 
# contains standardised metadata
s.save('testSpectrum.msa')

# Plot it
s.plot()

# Save the plot
plt.savefig("testSpectrum.png")
