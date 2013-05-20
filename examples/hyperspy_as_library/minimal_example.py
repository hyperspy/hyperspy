""" Loads hyperspy as a regular python library, creates a spectrum with random numbers and plots it to a file"""

import hyperspy.hspy as hspy
import numpy as np
import matplotlib.pyplot as plt

s = hspy.signals.Spectrum(np.random.rand(1024))
s.plot()

plt.savefig("testSpectrum.png")
