"""
Simple simulation (2 Gaussians)
===============================

Creates a 2D hyperspectrum consisting of two Gaussians and plots it.

This example can serve as starting point to test other functionalities on the
simulated hyperspectrum.

"""
import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt

# %%
# **Creating empty spectrum and model**
#
# We start by creating an empty spectrum image and then build up the simulation
# using HyperSpy's model components. This approach gives us full control over
# the synthetic data parameters.

# Create an empty spectrum
s = hs.signals.Signal1D(np.zeros((32, 32, 1024)))

# Create a model from the signal
m = s.create_model()

# **Model setup:**
# - Spectrum image: 32×32 spatial pixels
# - Spectrum length: 1024 channels
# - Ready for component addition

# %%
# **Configuring first Gaussian component**
#
# We'll create a Gaussian peak that varies spatially across the spectrum image.
# This simulates realistic variations you might see in experimental data.

# Define the first gaussian
gs1 = hs.model.components1D.Gaussian()
# Add it to the model
m.append(gs1)

# Set the parameters with spatial variation
m.set_parameters_value('sigma', 10, component_list=[gs1])

# **Spatial parameter variations:**
# - Center: varies ±5 channels around position 256
# - Area: random values between 0 and 10,000
# - Width: constant at 10 channels

# Make the center vary in the -5,5 range around 256
gs1.centre.map['values'][:] = 256 + (np.random.random((32, 32)) - 0.5) * 10
gs1.centre.map['is_set'][:] = True

# Make the area vary between 0 and 10000
gs1.A.map['values'][:] = 10000 * np.random.random((32, 32))
gs1.A.map['is_set'][:] = True

# %%
# Configure second Gaussian component
# -----------------------------------
# Second gaussian
gs2 = hs.model.components1D.Gaussian()
# Add it to the model
m.append(gs2)

# Set the parameters
m.set_parameters_value('sigma', 20, component_list=[gs2])

# Make the center vary in the -10,10 range around 768
gs2.centre.map['values'][:] = 768 + (np.random.random((32, 32)) - 0.5) * 20
gs2.centre.map['is_set'][:] = True

# Make the area vary between 0 and 20000
gs2.A.map['values'][:] = 20000 * np.random.random((32, 32))
gs2.A.map['is_set'][:] = True

# %%
# Generate the dataset and add noise
# ----------------------------------
# Create the dataset
s_model = m.as_signal()

# Add noise
s_model.set_signal_origin("simulation")
s_model.add_poissonian_noise()

# %%
# Plot the result
# ---------------
# Plot the result
s_model.plot()

plt.show()
