import numpy as np
import matplotlib.pyplot as plt

from hyperspy.signals.image import Image

if __name__ == '__main__':
    s = Image({'data' : np.random.random((16,32,32))})
    s.plot()
    plt.show()

