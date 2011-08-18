from hyperspy.signal import Signal
import numpy as np

s = Signal({'data' : np.random.random((16,16,32,32))})
s.axes_manager.set_view('image')
s.plot()

