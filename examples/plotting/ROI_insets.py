"""
==========
ROI Insets
==========

ROI's can be powerful tools to help visualize data.  In this case we will define ROI's in hyperspy, sum
the data within the ROI, and then plot the sum as a signal. Using the `matplotlib.figure.SubFigure` class
we can create a custom layout to visualize and interact with the data.
"""
import matplotlib.pyplot as plt
import hyperspy.api as hs
import numpy as np

rng = np.random.default_rng()

fig = plt.figure(figsize=(5, 3))
gs = fig.add_gridspec(6, 10)
sub1 = fig.add_subfigure(gs[0:6, 0:6])
sub2 = fig.add_subfigure(gs[0:2, 6:8])
sub3 = fig.add_subfigure(gs[2:4, 7:9])
sub4 = fig.add_subfigure(gs[4:6, 6:8])


s = hs.signals.Signal2D(rng.random((10, 10, 30, 30)))
r1 = hs.roi.RectangularROI(1, 1, 3, 3)
r2 = hs.roi.RectangularROI(4, 4, 6, 6)
r3 = hs.roi.RectangularROI(3, 7, 5, 9)


s2 = r1(s).sum()
s3 = r2(s).sum()
s4 = r3(s).sum()

navigator = s.sum(axis=(2, 3)).T  # create a navigator signal

navigator.plot(fig=sub1, colorbar=False, axes_off=True, title="", plot_indices=False)
s2.plot(fig=sub2, colorbar=False, axes_off=True, title="", plot_indices=False)
s3.plot(fig=sub3, colorbar=False, axes_off=True, title="", plot_indices=False)
s4.plot(fig=sub4, colorbar=False, axes_off=True, title="", plot_indices=False)

# %%
# Add ROI's to the navigator

r1.add_widget(navigator, color="r")
r2.add_widget(navigator, color="g")
r3.add_widget(navigator, color="y")

# %%
# Add Borders to the match the color of the ROI
red_edge = hs.plot.markers.Squares(
    offset_transform="axes",
    offsets=(0.5, 0.5),
    units="width",
    widths=1,
    color="r",
    linewidth=5,
    facecolor="none",
)
s2.add_marker(red_edge)

green_edge = hs.plot.markers.Squares(
    offset_transform="axes",
    offsets=(0.5, 0.5),
    units="width",
    widths=1,
    color="g",
    linewidth=5,
    facecolor="none",
)
s3.add_marker(green_edge)

yellow_edge = hs.plot.markers.Squares(
    offset_transform="axes",
    offsets=(0.5, 0.5),
    units="width",
    widths=1,
    color="y",
    linewidth=5,
    facecolor="none",
)

s4.add_marker(yellow_edge)

# %%
# Label the insets

label = hs.plot.markers.Texts(
    texts=("a.",), offsets=[[0.9, 0.9]], offset_transform="axes", sizes=10, color="w"
)
navigator.add_marker(label)
label = hs.plot.markers.Texts(
    texts=("b.",), offsets=[[0.8, 0.8]], offset_transform="axes", sizes=3, color="w"
)
s2.add_marker(label)
label = hs.plot.markers.Texts(
    texts=("c.",), offsets=[[0.8, 0.8]], offset_transform="axes", sizes=3, color="w"
)
s3.add_marker(label)
label = hs.plot.markers.Texts(
    texts=("d.",), offsets=[[0.8, 0.8]], offset_transform="axes", sizes=3, color="w"
)
s4.add_marker(label)

# %%
