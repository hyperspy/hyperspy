"""Creates a line spectrum and plots it
"""

# If running from hyperspy's interactive the next two imports can be omitted 
# omitted (i.e. the next 4 lines)
import numpy as np
import matplotlib.pyplot as plt

from hyperspy.signals.spectrum import Spectrum

s = Spectrum({'data' : np.random.random((100,1024))})
s.plot()

# If running from hyperspy's interactive console the next line can be 
# omitted
plt.show()

