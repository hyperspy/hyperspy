"""
Advanced Simulations with HyperSpy Models
==========================================

This example demonstrates how to create realistic simulations using HyperSpy's
model components, including the powerful Expression component for custom
mathematical functions, and best practices for simulation workflows.

"""

import numpy as np
import hyperspy.api as hs

# %%
# **Basic Simulation with Expression Components**
#
# The Expression component allows you to turn any mathematical function into 
# an optimizable component, making it extremely powerful for custom simulations.

# Create empty signal for 1D simulation
s1d = hs.signals.Signal1D(np.zeros(500))
s1d.axes_manager.signal_axes[0].name = 'Energy'
s1d.axes_manager.signal_axes[0].units = 'eV'
s1d.axes_manager.signal_axes[0].scale = 0.1
s1d.axes_manager.signal_axes[0].offset = 100

# Create model
m1d = s1d.create_model()

# Add custom background using Expression component with custom name
m1d.append(hs.model.components1D.Expression(
    "a * atan((x - b) / c) + d",
    name="Shirley_Background",
    a=1000,     # Step height
    b=120,      # Step position (eV)  
    c=5,        # Step width
    d=500       # Baseline offset
))

# Add multiple peaks with Expression components
# Asymmetric peak (simplified Doniach-Sunjic lineshape)
m1d.append(hs.model.components1D.Expression(
    "A * cos(pi*alpha/2) / ((1 + ((x-centre)/width)**2)**(1-alpha))",
    name="Doniach_Sunjic",
    A=5000,         # Amplitude
    centre=110,     # Peak center (eV)
    width=1.5,      # Width parameter
    alpha=0.1       # Asymmetry parameter
))

# Voigt profile using Expression (simplified version)
m1d.append(hs.model.components1D.Expression(
    "A * exp(-((x-centre)/sigma)**2/2) / (1 + ((x-centre)/width)**2)",
    name="Pseudo_Voigt",
    A=3000,
    centre=125,
    sigma=1.0,      # Gaussian width
    width=0.5       # Lorentzian width  
))

# Set parameter values using set_parameters_value (handles maps automatically)
m1d.set_parameters_value('a', 1000, component_list=[m1d.components.Shirley_Background], only_current=False)
m1d.set_parameters_value('b', 120, component_list=[m1d.components.Shirley_Background], only_current=False)
m1d.set_parameters_value('c', 5, component_list=[m1d.components.Shirley_Background], only_current=False)
m1d.set_parameters_value('d', 500, component_list=[m1d.components.Shirley_Background], only_current=False)

m1d.set_parameters_value('A', 5000, component_list=[m1d.components.Doniach_Sunjic], only_current=False)
m1d.set_parameters_value('centre', 110, component_list=[m1d.components.Doniach_Sunjic], only_current=False)
m1d.set_parameters_value('width', 1.5, component_list=[m1d.components.Doniach_Sunjic], only_current=False)
m1d.set_parameters_value('alpha', 0.1, component_list=[m1d.components.Doniach_Sunjic], only_current=False)

m1d.set_parameters_value('A', 3000, component_list=[m1d.components.Pseudo_Voigt], only_current=False)
m1d.set_parameters_value('centre', 125, component_list=[m1d.components.Pseudo_Voigt], only_current=False)
m1d.set_parameters_value('sigma', 1.0, component_list=[m1d.components.Pseudo_Voigt], only_current=False)
m1d.set_parameters_value('width', 0.5, component_list=[m1d.components.Pseudo_Voigt], only_current=False)

# Generate the simulation
sim1d = m1d.as_signal()
sim1d.set_signal_origin("simulation")
sim1d.metadata.General.title = "Advanced 1D Simulation"

# Store the simulation data and ground truth model
# Replace the model's signal with the simulation data
m1d.signal = sim1d

# Store the model as ground truth (preserves all parameters)
sim1d.models.store(m1d, name="ground_truth")

print("Created 1D simulation with Expression components:")
print(f"- Shirley background with arctangent step")
print(f"- Doniach-Sunjic asymmetric peak")
print(f"- Pseudo-Voigt profile peak")
print("- Ground truth model stored for future reference")

# %%
# **2D Simulation with Spatial Parameter Variations**
#
# Create a spectrum image where peak parameters vary spatially across the sample

# Create 2D signal (spectrum image)
nx, ny, n_energy = 32, 32, 256
s2d = hs.signals.Signal1D(np.zeros((nx, ny, n_energy)))

# Set up axes
s2d.axes_manager.navigation_axes[0].name = 'X'
s2d.axes_manager.navigation_axes[0].units = 'μm'
s2d.axes_manager.navigation_axes[0].scale = 0.1
s2d.axes_manager.navigation_axes[1].name = 'Y'
s2d.axes_manager.navigation_axes[1].units = 'μm'
s2d.axes_manager.navigation_axes[1].scale = 0.1
s2d.axes_manager.signal_axes[0].name = 'Energy'
s2d.axes_manager.signal_axes[0].units = 'eV'
s2d.axes_manager.signal_axes[0].scale = 0.5
s2d.axes_manager.signal_axes[0].offset = 100

# Create model
m2d = s2d.create_model()

# Add uniform background (using best practice: name at creation)
bg2d = hs.model.components1D.Polynomial(order=1, name="Background_2D")
m2d.append(bg2d)

# Set polynomial background parameters using set_parameters_value (handles maps automatically)
m2d.set_parameters_value('a0', 100, component_list=[bg2d], only_current=False)
m2d.set_parameters_value('a1', 0.5, component_list=[bg2d], only_current=False)

# Add main peak with spatial variations (using best practice: name at creation)
main_peak = hs.model.components1D.Gaussian(name="Main_Peak")
m2d.append(main_peak)

# Create spatial coordinate grids
x_coords, y_coords = np.ogrid[:nx, :ny]

# Peak center varies across the sample (simulating sample composition gradients)
center_variation = 120 + 10 * np.sin(2 * np.pi * x_coords / nx) * np.cos(2 * np.pi * y_coords / ny)
main_peak.centre.map['values'][:] = center_variation
main_peak.centre.map['is_set'][:] = True

# Peak intensity varies with distance from center (simulating beam effects)
center_x, center_y = nx // 2, ny // 2
distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
intensity_variation = 5000 * np.exp(-distance_from_center / 10)
main_peak.A.map['values'][:] = intensity_variation
main_peak.A.map['is_set'][:] = True

# Peak width varies slightly (simulating instrumental effects)
width_variation = 2.0 + 0.5 * np.random.random((nx, ny))
main_peak.sigma.map['values'][:] = width_variation
main_peak.sigma.map['is_set'][:] = True

# Set polynomial background parameters using direct access
bg2d.a0.value = 100
bg2d.a1.value = 0.5

# Generate the simulation
sim2d = m2d.as_signal()
sim2d.set_signal_origin("simulation")
sim2d.metadata.General.title = "2D Spatial Simulation"

# Store the simulation data and ground truth model
m2d.signal = sim2d
sim2d.models.store(m2d, name="ground_truth_2d")

print(f"\nCreated 2D simulation ({nx}×{ny} pixels):")
print(f"- Peak center varies: {center_variation.min():.1f} to {center_variation.max():.1f} eV")
print(f"- Intensity varies with position (beam effects)")
print(f"- Width has random variations")
print("- Ground truth model stored with spatial parameter maps")

# %%
# **Advanced Simulation: Multiple Phases with Custom Components**
#
# Simulate a sample with multiple crystalline phases, each with characteristic peaks

# Create signal for multi-phase simulation
s_phases = hs.signals.Signal1D(np.zeros((20, 20, 400)))
# Set up axes
s_phases.axes_manager.navigation_axes[0].name = 'X'
s_phases.axes_manager.navigation_axes[0].units = 'μm'
s_phases.axes_manager.navigation_axes[0].scale = 0.05
s_phases.axes_manager.navigation_axes[1].name = 'Y'
s_phases.axes_manager.navigation_axes[1].units = 'μm'
s_phases.axes_manager.navigation_axes[1].scale = 0.05
s_phases.axes_manager.signal_axes[0].name = '2theta'
s_phases.axes_manager.signal_axes[0].units = 'degrees'
s_phases.axes_manager.signal_axes[0].scale = 0.05
s_phases.axes_manager.signal_axes[0].offset = 20

# Create model
m_phases = s_phases.create_model()

# Add background (typical for diffraction)
bg_phases = hs.model.components1D.Expression(
    "a * exp(-b * x) + c",
    name="Exponential_Background",
    a=1000, b=0.1, c=50
)
m_phases.append(bg_phases)

# Phase 1: Major phase with multiple peaks (using best practice: name at creation)
phase1_peak1 = hs.model.components1D.Gaussian(name="Phase1_Peak1")
phase1_peak1.centre.value = 25.5  # degrees
phase1_peak1.sigma.value = 0.12   # width
phase1_peak1.A.value = 1000       # area
m_phases.append(phase1_peak1)

phase1_peak2 = hs.model.components1D.Gaussian(name="Phase1_Peak2")
phase1_peak2.centre.value = 31.2
phase1_peak2.sigma.value = 0.14   # width
phase1_peak2.A.value = 800        # area
m_phases.append(phase1_peak2)

# Phase 2: Minor phase (varies spatially) (using best practice: name at creation)
phase2_peak = hs.model.components1D.Gaussian(name="Phase2_Peak")
phase2_peak.centre.value = 28.8
phase2_peak.sigma.value = 0.16    # width
phase2_peak.A.value = 500         # area
m_phases.append(phase2_peak)

# Set spatial variations for phases
x_grid, y_grid = np.ogrid[:20, :20]

# Phase 1 is dominant everywhere but varies in intensity
phase1_intensity = 3000 + 1000 * np.random.random((20, 20))
phase1_peak1.A.map['values'][:] = phase1_intensity
phase1_peak1.A.map['is_set'][:] = True
phase1_peak2.A.map['values'][:] = 0.6 * phase1_intensity  # Related intensity
phase1_peak2.A.map['is_set'][:] = True

# Phase 2 only appears in certain regions
phase2_mask = (x_grid > 10) & (y_grid < 10)  # Bottom-right quadrant
phase2_intensity = np.where(phase2_mask, 1500 + 500 * np.random.random((20, 20)), 0)
phase2_peak.A.map['values'][:] = phase2_intensity
phase2_peak.A.map['is_set'][:] = True

# Set background parameters using set_parameters_value (handles maps automatically)
m_phases.set_parameters_value('a', 1000, component_list=[bg_phases], only_current=False)
m_phases.set_parameters_value('b', 0.1, component_list=[bg_phases], only_current=False)
m_phases.set_parameters_value('c', 50, component_list=[bg_phases], only_current=False)

# Set Gaussian component parameters using set_parameters_value
m_phases.set_parameters_value('centre', 25.5, component_list=[phase1_peak1], only_current=False)
m_phases.set_parameters_value('sigma', 0.12, component_list=[phase1_peak1], only_current=False)

m_phases.set_parameters_value('centre', 31.2, component_list=[phase1_peak2], only_current=False)
m_phases.set_parameters_value('sigma', 0.14, component_list=[phase1_peak2], only_current=False)

m_phases.set_parameters_value('centre', 28.8, component_list=[phase2_peak], only_current=False)
m_phases.set_parameters_value('sigma', 0.16, component_list=[phase2_peak], only_current=False)

# Generate the simulation
sim_phases = m_phases.as_signal()
sim_phases.set_signal_origin("simulation")
sim_phases.metadata.General.title = "Multi-phase Diffraction Simulation"

# Store the simulation data and ground truth model
m_phases.signal = sim_phases
sim_phases.models.store(m_phases, name="ground_truth_phases")

print(f"\nCreated multi-phase simulation:")
print(f"- Phase 1: Major phase (peaks at {phase1_peak1.centre.value}°, {phase1_peak2.centre.value}°)")
print(f"- Phase 2: Minor phase (peak at {phase2_peak.centre.value}°, spatial distribution)")
print(f"- Realistic background and peak profiles")
print("- Ground truth model stored with spatial phase distributions")

# %%
# **Adding Realistic Noise to Simulations**
#
# Demonstrate proper noise addition for different signal types

# Add realistic noise to 1D simulation
print("\nAdding realistic noise to simulations...")

# For spectroscopy data: Poisson + Gaussian noise
sim1d_noisy = sim1d.copy()
# Ensure positive values for Poisson noise (direct .data access needed for safety checks)
if sim1d_noisy.data.min() <= 0:
    sim1d_noisy.data += abs(sim1d_noisy.data.min()) + 1
sim1d_noisy.add_poissonian_noise(random_state=42)  # Shot noise
sim1d_noisy.add_gaussian_noise(std=20, random_state=43)  # Electronic noise

# For imaging data: primarily Poisson noise
sim2d_noisy = sim2d.copy()
# Ensure positive values for Poisson noise (direct .data access needed for safety checks)
if sim2d_noisy.data.min() <= 0:
    sim2d_noisy.data += abs(sim2d_noisy.data.min()) + 1
sim2d_noisy.add_poissonian_noise(random_state=44)

# For diffraction data: Poisson noise with low background
# Handle Poisson noise carefully to avoid numerical issues
sim_phases_noisy = sim_phases.copy()

# Clean up any NaN or infinite values (direct .data access needed for nan_to_num)
sim_phases_noisy.data = np.nan_to_num(sim_phases_noisy.data, nan=0.0, posinf=1000.0, neginf=0.0)

# Scale to realistic count range for Poisson noise (direct .data access needed for safety)
max_val = sim_phases_noisy.data.max()
if max_val > 1000:  # More conservative limit for diffraction data
    scale_factor = 1000 / max_val
    sim_phases_noisy.data *= scale_factor
    print(f"Scaled diffraction data by {scale_factor:.3f} for realistic count rates")

# Ensure all values are positive for Poisson noise (direct .data access needed for safety)
min_val = sim_phases_noisy.data.min()
if min_val <= 0:
    sim_phases_noisy.data = sim_phases_noisy.data - min_val + 1

# Verify values are in safe range and cap maximum to prevent Poisson errors
max_safe_value = 1000  # Conservative limit for NumPy Poisson generation
if sim_phases_noisy.data.max() > max_safe_value:
    sim_phases_noisy.data = np.clip(sim_phases_noisy.data, 0, max_safe_value)
    print(f"Clipped data to maximum value of {max_safe_value} for safe Poisson noise generation")

print(f"Data range before Poisson noise: {sim_phases_noisy.data.min():.1f} to {sim_phases_noisy.data.max():.1f}")

sim_phases_noisy.add_poissonian_noise(random_state=45)

print("Added appropriate noise models:")
print("- 1D spectroscopy: Poisson + Gaussian noise")
print("- 2D imaging: Poisson noise")
print("- Diffraction: Poisson noise")

# %%
# **Simulation Analysis and Validation**
#
# Demonstrate how to analyze and validate simulations

# Extract peak positions from 2D simulation
peak_map = sim2d_noisy.indexmax(axis='Energy')
peak_map.metadata.General.title = "Peak Position Map"

# Convert indices to energy values
energy_axis = sim2d_noisy.axes_manager.signal_axes[0].axis
peak_energy_map = peak_map.copy()
# Note: Direct .data access needed here for array indexing with peak positions
peak_energy_map.data = energy_axis[peak_map.data]
# Note: peak_energy_map is a 0D signal (no signal axes), so we add metadata directly
peak_energy_map.metadata.General.title = "Peak Energy Map (eV)"

# Calculate peak statistics (direct .data access needed for numerical comparison)
print(f"\nSimulation validation:")
print(f"Peak center range: {peak_energy_map.data.min():.1f} to {peak_energy_map.data.max():.1f} eV")
print(f"Expected range: {center_variation.min():.1f} to {center_variation.max():.1f} eV")
print(f"Simulation accuracy: ±{np.abs(peak_energy_map.data - center_variation).max():.2f} eV")

# %%
# **Plotting Results**
#
# Visualize the different simulations

# Plot 1D simulation
sim1d_noisy.plot()

# Plot 2D simulation overview
intensity_map = sim2d_noisy.max(axis='Energy')
intensity_map.metadata.General.title = "Maximum Intensity Map"
intensity_map.plot()

# Plot peak position map
peak_energy_map.plot()

print("\nSimulation complete! Generated multiple realistic datasets:")
print("- 1D spectroscopy with custom lineshapes")
print("- 2D spectrum image with spatial variations")  
print("- Multi-phase diffraction simulation")
print("- All with appropriate noise models")

# %%
# **Advanced Parameter Setting Demonstration**
#
# Show when to use set_parameters_value vs direct access

# Create a simple model with multiple peaks to demonstrate batch operations
s_demo = hs.signals.Signal1D(np.zeros(512))
s_demo.axes_manager.signal_axes[0].name = 'Energy'
s_demo.axes_manager.signal_axes[0].scale = 0.5
s_demo.axes_manager.signal_axes[0].offset = 0

m_demo = s_demo.create_model()

# Add multiple similar peaks (using best practice: name at creation)
peak_a = hs.model.components1D.Gaussian(name='Peak_A')
peak_b = hs.model.components1D.Gaussian(name='Peak_B')
peak_c = hs.model.components1D.Gaussian(name='Peak_C')
background = hs.model.components1D.Polynomial(order=1, name='Background')

m_demo.extend([peak_a, peak_b, peak_c, background])

# ✅ Excellent use case: Set same parameter on multiple components
print("\nBatch parameter setting with set_parameters_value:")
all_peaks = [peak_a, peak_b, peak_c]

# Set all peaks to have the same width (batch operation)
m_demo.set_parameters_value('sigma', 5.0, component_list=all_peaks)
print("Set sigma=5.0 for all peaks simultaneously")

# Set individual peak positions (different values, use direct access for this demo)
# Note: In practice for simulations, parameter maps should be set
peak_a.centre.value = 50
peak_b.centre.value = 100
peak_c.centre.value = 150
print("Set individual peak centers using direct access")

# Set different amplitudes for demonstration
peak_a.A.value = 1000
peak_b.A.value = 1500
peak_c.A.value = 800

# Set background parameters using set_parameters_value (handles maps)
m_demo.set_parameters_value('a0', 100, component_list=[background], only_current=False)
m_demo.set_parameters_value('a1', 0.5, component_list=[background], only_current=False)

# Set individual parameter maps for peaks (required for simulation)
for peak, centre_val, amp_val in zip(all_peaks, [50, 100, 150], [1000, 1500, 800]):
    m_demo.set_parameters_value('centre', centre_val, component_list=[peak], only_current=False)
    m_demo.set_parameters_value('A', amp_val, component_list=[peak], only_current=False)

# Generate and show the result
demo_sim = m_demo.as_signal()
demo_sim.set_signal_origin("simulation")
demo_sim.metadata.General.title = "Batch Parameter Setting Demo"

print(f"Created simulation with {len(all_peaks)} peaks")
print("- All peaks have identical width (set via batch operation)")
print("- Individual positions and amplitudes (set via direct access)")
