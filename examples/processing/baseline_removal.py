"""
Baseline Removal
================

This example shows how to remove a baseline from a 1D signal using the
`pybaselines <https://pybaselines.readthedocs.io>`_ library.
"""

#%%
# Create a signal
import hyperspy.api as hs

# Render with anyplotlib so the figures below stay live in the browser.
hs.preferences.Plot.backend = "anyplotlib"

# pybaselines is not part of the Pyodide distribution, so the documentation's
# in-browser interpreter fetches it from PyPI.
_PYODIDE_MICROPIP = ["pybaselines"]

s = hs.data.two_gaussians()


#%%
# Remove baseline using :meth:`~.api.signals.Signal1D.remove_baseline`:
s2 = s.remove_baseline(method="aspls", lam=1E7, inplace=False)

#%%
# Plot the signal and its baseline: 
(s + (s-s2) * 1j).plot()  # Interactive
# sphinx_gallery_thumbnail_number = 1
