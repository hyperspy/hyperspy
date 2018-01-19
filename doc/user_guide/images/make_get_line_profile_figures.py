import os
import numpy as np
import hyperspy.api as hs

my_path = os.path.dirname(__file__)

print(my_path)

# Figure for 2D signal
s = hs.signals.Signal2D(np.arange(10000).reshape((100, 100)))
s_line = s.get_line_profile()
s._plot.signal_plot.figure.savefig(os.path.join(
    my_path, "get_line_profile_2d_signal_0.png"))
s_line._plot.signal_plot.figure.savefig(os.path.join(
    my_path, "get_line_profile_2d_signal_1.png"))

# Figure for 4D signal
s = hs.signals.Signal1D(np.arange(10000).reshape((10, 10, 10, 10)))
s_line = s.get_line_profile(axes=(0, 2))
s._plot.signal_plot.figure.savefig(os.path.join(
    my_path, "get_line_profile_4d_signal_0_sig.png"))
s._plot.navigator_plot.figure.savefig(os.path.join(
    my_path, "get_line_profile_4d_signal_0_nav.png"))
s_line._plot.signal_plot.figure.savefig(os.path.join(
    my_path, "get_line_profile_4d_signal_1_sig.png"))
s_line._plot.navigator_plot.figure.savefig(os.path.join(
    my_path, "get_line_profile_4d_signal_1_nav.png"))
