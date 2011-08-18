"""Creates a 3D image and plots it
"""
# If running from hyperspy's interactive the next two imports can be omitted 
# omitted (i.e. the next 4 lines)
import numpy as np
import matplotlib.pyplot as plt

from hyperspy.signals.image import Image

s = Image({'data' : np.random.random((16,32,32))})
s.plot()

# If running from hyperspy's interactive console the next line can be 
# omitted
plt.show()

