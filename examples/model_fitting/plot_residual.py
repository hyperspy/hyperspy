"""
Plot Residual
=============

Fit an affine function and plot the residual.

"""

import numpy as np
import hyperspy.api as hs

#%%
# Create a signal:
data = np.arange(1000, dtype=np.int64).reshape((10, 100))
s = hs.signals.Signal1D(data)

#%%
# Add noise:
s.add_poissonian_noise(random_state=0)

#%%
# Create model:
m = s.create_model()
line = hs.model.components1D.Expression("a * x + b", name="Affine")
m.append(line)

#%%
# Fit for all navigation positions:
m.multifit()

#%%
# Plot the fitted model with residual:
m.plot(plot_residual=True)
# Choose the second figure as gallery thumbnail:
# sphinx_gallery_thumbnail_number = 2

