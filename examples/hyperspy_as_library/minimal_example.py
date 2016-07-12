""" Loads hyperspy as a regular python library, creates a spectrum with random numbers and plots it to a file"""

import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt

s = hs.signals.Signal1D(np.random.rand(1024))
s.plot()

plt.savefig("testSpectrum.png")
