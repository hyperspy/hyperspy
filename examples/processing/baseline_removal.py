"""
Baseline Removal
================

This example shows how to remove a baseline from a 1D signal using the
`pybaselines <https://pybaselines.readthedocs.io>`_ library.
"""

#%%
# Create a signal
import hyperspy.api as hs
s = hs.data.two_gaussians()


#%%
# Remove baseline using :meth:`~.api.signals.Signal1D.remove_baseline`:
s2 = s.remove_baseline(algorithm="aspls", lam=1E7, inplace=False)

#%%
# Plot the signal and its baseline: 
(s + (s-s2) * 1j).plot()

#%%
# sphinx_gallery_thumbnail_number = 2