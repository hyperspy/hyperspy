"""
================
Named Navigators
================

This example demonstrates ``signal.navigators``, a dict-like proxy that stores
named navigator signals. Named navigators survive ``inav`` slicing and ``map``
operations, and can be selected by name when calling ``plot()``.
"""

import matplotlib.pyplot as plt
import numpy as np

import hyperspy.api as hs

# %%
# Create a synthetic 4D-STEM-like dataset: navigation (10, 15), signal (64, 64)
rng = np.random.default_rng(0)
s = hs.signals.Signal2D(rng.random((10, 15, 64, 64)))
s.axes_manager.navigation_axes[0].name = "x"
s.axes_manager.navigation_axes[0].scale = 0.5
s.axes_manager.navigation_axes[0].units = "nm"
s.axes_manager.navigation_axes[1].name = "y"
s.axes_manager.navigation_axes[1].scale = 0.5
s.axes_manager.navigation_axes[1].units = "nm"

# %%
# Build a "Virtual Bright Field" navigator by summing signal pixels near the
# centre (a stand-in for a real VBF aperture).
centre = s.isig[24:40, 24:40].sum(axis=(2, 3)).T
centre.metadata.General.title = "Virtual Bright Field"

# %%
# Assign it to the navigators dict.  The proxy validates that the total shape
# matches the signal's navigation space.
s.navigators["Virtual Bright Field"] = centre

# %%
# A second named navigator: annular dark field using an outer ring.
adf = s.isig[:12, :].sum(axis=(2, 3)).T
adf.metadata.General.title = "ADF"
s.navigators["ADF"] = adf

print("Stored navigators:", list(s.navigators.keys()))

# %%
# Use ``navigators.set_default`` to promote one to the singular navigator so
# ``plot()`` uses it by default.
s.navigators.set_default("Virtual Bright Field")

# %%
# inav slicing propagates to all named navigators automatically.
sliced = s.inav[2:8, 3:12]
print("Original nav shape :", s.axes_manager.navigation_shape)
print("Sliced   nav shape :", sliced.axes_manager.navigation_shape)
print("Navigator shape    :", sliced.navigators["Virtual Bright Field"].data.shape)

# %%
# Plot the VBF and ADF navigators side-by-side using plain matplotlib to show
# what the navigator images look like.
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(s.navigators["Virtual Bright Field"].data.T, origin="lower")
axes[0].set_title("Virtual Bright Field")
axes[0].set_xlabel("x (nm)")
axes[0].set_ylabel("y (nm)")

axes[1].imshow(s.navigators["ADF"].data.T, origin="lower")
axes[1].set_title("ADF")
axes[1].set_xlabel("x (nm)")
axes[1].set_ylabel("y (nm)")

fig.tight_layout()

# %%
# ``compute_navigator()`` sums over all signal axes, saves the result as
# ``navigators["Signal Sum Image"]``, and sets it as the default navigator.
s2 = hs.signals.Signal2D(rng.random((5, 7, 32, 32)))
s2.compute_navigator()
print("After compute_navigator:", list(s2.navigators.keys()))
print("Navigator is set      :", s2.navigator is not None)
