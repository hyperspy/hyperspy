"""
Simple simulation (2 Gaussians)
===============================

Creates a 2D hyperspectrum consisting of two Gaussians and plots it.

This example can serve as starting point to test other functionalities on the
simulated hyperspectrum.

"""
import numpy as np
import hyperspy.api as hs

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
# **Setting up signal with physical units**
#
# Configure the energy axis with realistic calibration before adding components.
# This enables string-based indexing and physical interpretation.

s.axes_manager.signal_axes[0].name = 'Energy'
s.axes_manager.signal_axes[0].units = 'eV'
s.axes_manager.signal_axes[0].scale = 0.1
s.axes_manager.signal_axes[0].offset = 200

print(f"Energy range: {s.axes_manager[2].axis[0]:.1f} to {s.axes_manager[2].axis[-1]:.1f} eV")

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
# - Center: varies ±5 channels around position 256 (~225.6 eV)
# - Area: random values between 0 and 10,000
# - Width: constant at 10 channels (~1 eV)

# Make the center vary in the -5,5 range around 256
gs1.centre.map['values'][:] = 256 + (np.random.random((32, 32)) - 0.5) * 10
gs1.centre.map['is_set'][:] = True

# Make the area vary between 0 and 10000
gs1.A.map['values'][:] = 10000 * np.random.random((32, 32))
gs1.A.map['is_set'][:] = True

# %%
# **Configure second Gaussian component**
#
# Second gaussian at higher energy with different characteristics
gs2 = hs.model.components1D.Gaussian()
# Add it to the model
m.append(gs2)

# Set the parameters
m.set_parameters_value('sigma', 20, component_list=[gs2])

# Make the center vary around 768 (~276.8 eV)
gs2.centre.map['values'][:] = 768 + (np.random.random((32, 32)) - 0.5) * 20
gs2.centre.map['is_set'][:] = True

# Make the area vary between 0 and 20000
gs2.A.map['values'][:] = 20000 * np.random.random((32, 32))
gs2.A.map['is_set'][:] = True

# %%
# **Generate the dataset and add realistic noise**
#
# Create the simulated spectrum and add physics-based noise
s_model = m.as_signal()

# Set metadata for proper identification
s_model.set_signal_origin("simulation")

# Store ground truth model for reproducibility
m.signal = s_model
s_model.models.store(m, name="ground_truth")

# Add realistic noise in proper sequence
print("Adding realistic noise to simulation...")
print(f"Signal intensity range: {s_model.data.min():.0f} to {s_model.data.max():.0f}")

# 1. Convert to float for full noise compatibility
s_model.change_dtype('float64')

# 2. Add shot noise (Poisson statistics from counting)
s_model.add_poissonian_noise(random_state=42)

# 3. Add electronic noise (instrumentation)
s_model.add_gaussian_noise(std=50, random_state=43)

print(f"After noise: {s_model.data.min():.0f} to {s_model.data.max():.0f}")

# %%
# **Demonstrate string-based indexing with units**
#
# Now that we have a calibrated signal, we can use string-based indexing

# Extract first Gaussian region using physical units
first_peak = s_model.isig['220 eV':'235 eV']
print(f"First peak region shape: {first_peak.data.shape}")
print(f"Energy range: {first_peak.axes_manager[2].axis[0]:.1f} to {first_peak.axes_manager[2].axis[-1]:.1f} eV")

# Extract second Gaussian region  
second_peak = s_model.isig['270 eV':'285 eV']
print(f"Second peak region shape: {second_peak.data.shape}")

# Extract a spatial subset using relative indexing
spatial_roi = s_model.inav['rel0.25':'rel0.75', 'rel0.25':'rel0.75']
print(f"Central spatial ROI shape: {spatial_roi.data.shape}")

# %%
# **Plot the result**
#
# Visualize the simulated spectrum image with both peaks
s_model.plot()
