"""
Specialized Analysis Workflows
==============================

This example demonstrates complete analysis workflows for specific scientific
domains, showcasing how to combine multiple HyperSpy capabilities for real-world
analytical problems in electron microscopy and spectroscopy.
"""

import hyperspy.api as hs
import numpy as np

# %%
# ## Workflow 1: Core-Loss EELS Quantification
# 
# Complete workflow for quantitative EELS analysis including background subtraction,
# edge fitting, and elemental mapping

print("Creating simulated EELS spectrum image...")

# Create realistic EELS simulation
nx, ny, n_energy = 32, 32, 1024
energy_axis = np.linspace(200, 800, n_energy)

# Initialize data array
eels_data = np.zeros((nx, ny, n_energy))

# Add realistic EELS features
for i in range(nx):
    for j in range(ny):
        # Background (power law decay)
        background = 10000 * (energy_axis / 300)**(-3.5)
        
        # Carbon K-edge at ~284 eV (varying concentration)
        carbon_conc = 0.3 + 0.4 * np.sin(2 * np.pi * i / nx) * np.cos(2 * np.pi * j / ny)
        carbon_onset = 284
        carbon_edge_mask = energy_axis >= carbon_onset
        carbon_edge = np.zeros_like(energy_axis)
        carbon_edge[carbon_edge_mask] = carbon_conc * 2000 * (energy_axis[carbon_edge_mask] - carbon_onset)**(-0.5)
        
        # Oxygen K-edge at ~532 eV (varying concentration)
        oxygen_conc = 0.2 + 0.3 * np.cos(2 * np.pi * i / (nx/2))
        oxygen_onset = 532
        oxygen_edge_mask = energy_axis >= oxygen_onset
        oxygen_edge = np.zeros_like(energy_axis)
        oxygen_edge[oxygen_edge_mask] = oxygen_conc * 1500 * (energy_axis[oxygen_edge_mask] - oxygen_onset)**(-0.7)
        
        # Combine components with noise
        spectrum = background + carbon_edge + oxygen_edge
        noise = np.random.poisson(np.maximum(spectrum, 1))
        eels_data[i, j, :] = noise

# Create EELS signal - use Signal1D since EELSSpectrum is in exspy package
eels_signal = hs.signals.Signal1D(eels_data)
eels_signal.metadata.Signal.signal_type = 'EELS'
eels_signal.axes_manager[0].name = 'x'
eels_signal.axes_manager[0].units = 'nm'
eels_signal.axes_manager[0].scale = 0.5

eels_signal.axes_manager[1].name = 'y'
eels_signal.axes_manager[1].units = 'nm'
eels_signal.axes_manager[1].scale = 0.5

eels_signal.axes_manager[2].name = 'energy_loss'
eels_signal.axes_manager[2].units = 'eV'
eels_signal.axes_manager[2].scale = 0.6
eels_signal.axes_manager[2].offset = 200
eels_signal.metadata.General.title = 'Simulated EELS Spectrum Image'

print(f"EELS signal created: {eels_signal}")

# %%
# ### Background Subtraction Workflow

# Define pre-edge region for background fitting
pre_edge_region = eels_signal.isig[200.:280.]

# Fit power law background (using best practice: name at creation)
background_model = pre_edge_region.create_model()
power_law = hs.model.components1D.PowerLaw(name="PowerLaw_Background")
# Set initial parameter values for stable fitting
power_law.A.value = 1000
power_law.r.value = 3.0
background_model.append(power_law)

# Fit background to pre-edge region
print("Fitting power law background...")
try:
    background_model.fit()
    print(f"Fit successful. A={power_law.A.value:.2f}, r={power_law.r.value:.2f}")
except Exception as e:
    print(f"Fit failed: {e}. Using manual parameter values.")
    # Set reasonable values manually if fit fails
    power_law.A.value = 5000
    power_law.r.value = 2.5

# Create simple power law background using numpy operations
print("Creating power law background using direct calculation...")
# Get fitted parameters
A_val = power_law.A.value
r_val = power_law.r.value

# Create background using power law formula: A * x^(-r)
energy_axis = eels_signal.axes_manager.signal_axes[0].axis
background_data = A_val * energy_axis**(-r_val)

# Create background signal with proper shape
background_extrapolated = eels_signal.deepcopy()
# Broadcast background to all navigation positions  
background_extrapolated.data[:] = background_data[np.newaxis, np.newaxis, :]
background_extrapolated.metadata.General.title = 'Power Law Background'

# Subtract background
eels_background_subtracted = eels_signal - background_extrapolated

print("Background subtraction completed")

# %%
# ### Edge Detection and Quantification

# Extract Carbon K-edge region
carbon_edge_region = eels_background_subtracted.isig[280.:350.]
carbon_integral = carbon_edge_region.integrate1D(axis='energy_loss')
carbon_integral.metadata.General.title = 'Carbon K-edge Intensity Map'

# Extract Oxygen K-edge region  
oxygen_edge_region = eels_background_subtracted.isig[530.:600.]
oxygen_integral = oxygen_edge_region.integrate1D(axis='energy_loss')
oxygen_integral.metadata.General.title = 'Oxygen K-edge Intensity Map'

# Calculate elemental ratio map
ratio_map = carbon_integral / (oxygen_integral + 1e-6)  # Avoid division by zero
ratio_map.metadata.General.title = 'C/O Ratio Map'

print("Edge integration and ratio calculation completed")

# %%
# ## Workflow 2: 4D-STEM Virtual Detector Analysis
# 
# Complete workflow for 4D-STEM data including virtual detector creation,
# strain mapping, and orientation analysis

print("\nCreating simulated 4D-STEM dataset...")

# Create 4D-STEM simulation
scan_size, det_size = 16, 64
stem_data = np.zeros((scan_size, scan_size, det_size, det_size))

for sy in range(scan_size):
    for sx in range(scan_size):
        # Create diffraction pattern with position-dependent features
        y_det, x_det = np.ogrid[:det_size, :det_size]
        center_y, center_x = det_size // 2, det_size // 2
        
        # Main beam
        main_beam = 5000 * np.exp(-((y_det - center_y)**2 + (x_det - center_x)**2) / 50)
        
        # Diffracted beams with position-dependent shifts (strain simulation)
        strain_y = 2 * np.sin(2 * np.pi * sy / scan_size)
        strain_x = 2 * np.cos(2 * np.pi * sx / scan_size)
        
        beam1_y, beam1_x = center_y + 15 + strain_y, center_x + 15 + strain_x
        beam2_y, beam2_x = center_y - 15 - strain_y, center_x - 15 - strain_x
        
        beam1 = 1500 * np.exp(-((y_det - beam1_y)**2 + (x_det - beam1_x)**2) / 20)
        beam2 = 1500 * np.exp(-((y_det - beam2_y)**2 + (x_det - beam2_x)**2) / 20)
        
        # Add noise
        pattern = main_beam + beam1 + beam2 + 100 * np.random.random((det_size, det_size))
        stem_data[sy, sx, :, :] = np.random.poisson(pattern)

# Create 4D-STEM signal
stem_signal = hs.signals.Signal2D(stem_data)
stem_signal.axes_manager[0].name = 'scan_y'
stem_signal.axes_manager[0].units = 'nm'
stem_signal.axes_manager[0].scale = 0.2

stem_signal.axes_manager[1].name = 'scan_x'
stem_signal.axes_manager[1].units = 'nm'
stem_signal.axes_manager[1].scale = 0.2

stem_signal.axes_manager[2].name = 'detector_y'
stem_signal.axes_manager[2].units = 'mrad'
stem_signal.axes_manager[2].scale = 0.5

stem_signal.axes_manager[3].name = 'detector_x'
stem_signal.axes_manager[3].units = 'mrad'
stem_signal.axes_manager[3].scale = 0.5
stem_signal.metadata.General.title = '4D-STEM Dataset'

print(f"4D-STEM signal created: {stem_signal}")

# %%
# ### Virtual Detector Analysis

# Create different virtual detectors
bright_field = stem_signal.isig[28:36, 28:36].sum(axis=('detector_y', 'detector_x'))
bright_field.metadata.General.title = 'Bright Field Virtual Detector'

# Annular detector (ring-shaped)
detector_y, detector_x = np.ogrid[:det_size, :det_size]
center_y, center_x = det_size // 2, det_size // 2
radius = np.sqrt((detector_y - center_y)**2 + (detector_x - center_x)**2)
annular_mask = (radius > 20) & (radius < 30)

# Apply virtual detector mask
annular_detector_data = stem_signal.data * annular_mask[np.newaxis, np.newaxis, :, :]
# Create signal and copy axis configuration
annular_detector_signal = hs.signals.Signal2D(annular_detector_data)
annular_detector_signal.axes_manager = stem_signal.axes_manager.deepcopy()
annular_detector = annular_detector_signal.sum(axis=(2, 3))  # Use axis indices instead of names
annular_detector.metadata.General.title = 'Annular Dark Field Virtual Detector'

print("Virtual detector analysis completed")

# %%
# ### Center of Mass Analysis for Strain Mapping

# Calculate center of mass for each diffraction pattern
def calculate_com(pattern):
    """Calculate center of mass of diffraction pattern"""
    y_indices, x_indices = np.ogrid[:pattern.shape[0], :pattern.shape[1]]
    total_intensity = np.sum(pattern)
    if total_intensity > 0:
        com_y = np.sum(y_indices * pattern) / total_intensity
        com_x = np.sum(x_indices * pattern) / total_intensity
        return com_y, com_x
    return pattern.shape[0]/2, pattern.shape[1]/2

# Calculate center of mass maps
com_y_map = np.zeros((scan_size, scan_size))
com_x_map = np.zeros((scan_size, scan_size))

for sy in range(scan_size):
    for sx in range(scan_size):
        pattern = stem_signal.inav[sy, sx].data
        com_y, com_x = calculate_com(pattern)
        com_y_map[sy, sx] = com_y
        com_x_map[sy, sx] = com_x

# Create strain maps (deviation from average)
reference_com_y = np.mean(com_y_map)
reference_com_x = np.mean(com_x_map)

strain_y = hs.signals.Signal2D(com_y_map - reference_com_y)
strain_x = hs.signals.Signal2D(com_x_map - reference_com_x)

strain_y.axes_manager[0].name = 'scan_y'
strain_y.axes_manager[0].units = 'nm'
strain_y.axes_manager[0].scale = 0.2

strain_y.axes_manager[1].name = 'scan_x'
strain_y.axes_manager[1].units = 'nm'
strain_y.axes_manager[1].scale = 0.2

strain_x.axes_manager[0].name = 'scan_y'
strain_x.axes_manager[0].units = 'nm'
strain_x.axes_manager[0].scale = 0.2

strain_x.axes_manager[1].name = 'scan_x'
strain_x.axes_manager[1].units = 'nm'
strain_x.axes_manager[1].scale = 0.2

strain_y.metadata.General.title = 'Strain Map Y-direction'
strain_x.metadata.General.title = 'Strain Map X-direction'

print("Center of mass strain analysis completed")

# %%
# ## Workflow 3: Phase Analysis in Crystallography
# 
# Complete workflow for phase identification and quantification using peak fitting

print("\nCreating simulated XRD pattern...")

# Create simulated powder diffraction pattern
n_angles = 2000
two_theta = np.linspace(10, 80, n_angles)

# Background
background = 100 + 20 * np.exp(-(two_theta - 15)**2 / 100)

# Phase 1 peaks (e.g., quartz)
phase1_peaks = [26.6, 50.1, 59.9]  # 2θ positions
phase1_intensities = [1000, 600, 400]
phase1_widths = [0.3, 0.4, 0.35]

phase1_pattern = np.zeros_like(two_theta)
for pos, intensity, width in zip(phase1_peaks, phase1_intensities, phase1_widths):
    phase1_pattern += intensity * np.exp(-((two_theta - pos) / width)**2)

# Phase 2 peaks (e.g., feldspar) 
phase2_peaks = [27.9, 41.2, 56.1]
phase2_intensities = [800, 500, 300]
phase2_widths = [0.35, 0.3, 0.4]

phase2_pattern = np.zeros_like(two_theta)
for pos, intensity, width in zip(phase2_peaks, phase2_intensities, phase2_widths):
    phase2_pattern += intensity * np.exp(-((two_theta - pos) / width)**2)

# Combine with noise
total_pattern = background + phase1_pattern + phase2_pattern
noisy_pattern = np.random.poisson(np.maximum(total_pattern, 1))

# Create diffraction signal
xrd_signal = hs.signals.Signal1D(noisy_pattern)
xrd_signal.axes_manager[0].name = '2theta'
xrd_signal.axes_manager[0].units = 'degrees'
xrd_signal.axes_manager[0].scale = (80-10)/n_angles
xrd_signal.axes_manager[0].offset = 10
xrd_signal.metadata.General.title = 'Simulated Powder Diffraction Pattern'

print(f"XRD signal created: {xrd_signal}")

# %%
# ### Peak Fitting for Phase Quantification

# Create model for peak fitting
xrd_model = xrd_signal.create_model()

# Add background component (using best practice: name at creation)
background_component = hs.model.components1D.Exponential(name="XRD_Background")
xrd_model.append(background_component)

# Add Gaussian peaks for each phase (using best practice: name at creation)
phase1_components = []
for i, (pos, intensity, width) in enumerate(zip(phase1_peaks, phase1_intensities, phase1_widths)):
    peak = hs.model.components1D.Gaussian(name=f"Phase1_Peak_{i+1}")
    peak.centre.value = pos
    peak.A.value = intensity
    peak.sigma.value = width
    xrd_model.append(peak)
    phase1_components.append(peak)

phase2_components = []
for i, (pos, intensity, width) in enumerate(zip(phase2_peaks, phase2_intensities, phase2_widths)):
    peak = hs.model.components1D.Gaussian(name=f"Phase2_Peak_{i+1}")
    peak.centre.value = pos
    peak.A.value = intensity  
    peak.sigma.value = width
    xrd_model.append(peak)
    phase2_components.append(peak)

# Fit the model
print("Fitting peak model...")
xrd_model.fit()

# Calculate phase fractions based on integrated intensities
phase1_total = sum(comp.A.value for comp in phase1_components)
phase2_total = sum(comp.A.value for comp in phase2_components)
total_intensity = phase1_total + phase2_total

phase1_fraction = phase1_total / total_intensity
phase2_fraction = phase2_total / total_intensity

print(f"Phase 1 fraction: {phase1_fraction:.3f}")
print(f"Phase 2 fraction: {phase2_fraction:.3f}")

# %%
# ## Results Visualization and Export
# 
# Create comprehensive visualization of all analysis results

print("\nGenerating comprehensive analysis visualization...")

# EELS results
hs.plot.plot_images([carbon_integral, oxygen_integral, ratio_map], 
                    tight_layout=True, axes_decor='ticks')

# 4D-STEM results  
hs.plot.plot_images([bright_field, annular_detector, strain_y, strain_x],
                    tight_layout=True, axes_decor='ticks')

# XRD fitting results
xrd_signal.plot()
xrd_model.plot()

# %%
# ## Workflow Summary and Best Practices

print("\nSpecialized Analysis Workflows Completed:")
print("==========================================")

print("\n1. EELS Quantification Workflow:")
print("   - Power law background subtraction")
print("   - Edge integration for elemental mapping")  
print("   - Ratio map calculation")
print("   - Result: Carbon/Oxygen distribution maps")

print("\n2. 4D-STEM Analysis Workflow:")
print("   - Virtual detector creation (bright field, annular)")
print("   - Center of mass strain analysis")
print("   - Position-dependent diffraction analysis")
print("   - Result: Strain maps and virtual images")

print("\n3. Phase Analysis Workflow:")
print("   - Peak identification and fitting")
print("   - Multi-component model creation")
print("   - Quantitative phase fraction calculation")
print("   - Result: Phase composition analysis")

print("\nWorkflow Best Practices Demonstrated:")
print("- Domain-specific signal types (Signal1D with signal_type, Signal2D)")
print("- Model-based analysis for reproducibility")
print("- Comprehensive metadata management")
print("- Statistical analysis across multiple dimensions")
print("- Professional visualization and export")
print("- Error handling and validation steps")
