s = Signal({'data' : np.random.random((16,16,32,32))})
s.axes_manager.axes[2].slice_bool = True
s.plot()

