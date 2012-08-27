"""Creates a spectrum image and plots it
"""

s = signals.Spectrum({'data' : np.random.random((64, 64, 1024))})
s.plot()

