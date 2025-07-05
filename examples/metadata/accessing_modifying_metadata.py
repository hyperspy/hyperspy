"""
Accessing and Modifying Metadata
=================================

This example demonstrates how to access and modify metadata in HyperSpy signals.
Metadata provides important information about your data and how it was acquired.
"""

import numpy as np
import hyperspy.api as hs

# %%
# ## Create test signal with initial metadata
# 
# We'll create a test signal and populate it with realistic metadata

# Create a test signal with some initial metadata
data = np.random.random((50, 100)) * 1000
signal = hs.signals.Signal1D(data)

# Set up some basic metadata during signal creation
signal.metadata.General.title = "Example Spectrum"
signal.metadata.General.date = "2024-01-15"
signal.metadata.General.time = "14:30:00"

# Set acquisition parameters
signal.metadata.Acquisition_instrument.SEM.beam_energy = 15.0
signal.metadata.Acquisition_instrument.SEM.beam_current = 1.2

# %%
# ## Access metadata values
# 
# Metadata can be accessed using dot notation for easy reading
print(f"Title: {signal.metadata.General.title}")
print(f"Date: {signal.metadata.General.date}")
print(f"Beam energy: {signal.metadata.Acquisition_instrument.SEM.beam_energy} kV")

# %%
# ## Modify existing metadata
# 
# Metadata can be easily modified and new entries can be added
signal.metadata.Acquisition_instrument.SEM.beam_energy = 20.0
signal.metadata.Sample.name = "Test Sample"

# Display the updated values
print("Updated beam energy:", signal.metadata.Acquisition_instrument.SEM.beam_energy)
print("Sample name:", signal.metadata.Sample.name)

# %%
# ## Summary
# 
# Metadata in HyperSpy provides a flexible way to store and access information about your data
