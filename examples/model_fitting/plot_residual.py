"""
Model Residual Analysis and Visualization
=========================================

This example demonstrates how to analyze and visualize model residuals in HyperSpy.
Residual analysis is crucial for assessing fit quality and identifying systematic
errors in your model. This example follows AI Guide best practices for model
evaluation and visualization.

Key concepts:
- Creating and fitting models with multidimensional data
- Calculating and interpreting residuals
- Visualizing residuals to assess fit quality
- Statistical analysis of residuals
- Identifying systematic patterns in residuals
"""

import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt
from scipy import stats

# %%
# **Creating synthetic multidimensional data**
#
# We'll create a 2D dataset with systematic variations to demonstrate
# residual analysis across multiple spectra.

# Create base signal with proper calibration
# Use different dimensions to clearly see HyperSpy's axis arrangement
data = np.arange(1500, dtype=np.float64).reshape((10, 15, 10))  # (10, 15, 10) → (15, 10|10)
s = hs.signals.Signal1D(data)

# ✅ BEST PRACTICE: Always calibrate axes immediately
s.axes_manager.navigation_axes[0].name = 'x'
s.axes_manager.navigation_axes[0].units = 'μm'
s.axes_manager.navigation_axes[0].scale = 0.1
s.axes_manager.navigation_axes[1].name = 'y'
s.axes_manager.navigation_axes[1].units = 'μm'
s.axes_manager.navigation_axes[1].scale = 0.1
s.axes_manager.signal_axes[0].name = 'Energy'
s.axes_manager.signal_axes[0].units = 'eV'
s.axes_manager.signal_axes[0].scale = 0.5
s.axes_manager.signal_axes[0].offset = 100

s.metadata.General.title = 'Synthetic Spectrum Image'

print(f"Created signal: {s}")
print(f"Navigation shape: {s.axes_manager.navigation_shape}")
print(f"Signal shape: {s.axes_manager.signal_shape}")

# %%
# **Add realistic noise and systematic variations**
#
# Add different types of noise and systematic variations to test
# the robustness of our residual analysis.

# Add Poissonian noise (realistic for count data)
s.add_poissonian_noise(random_state=0)

# Add systematic variations to make the problem more realistic
# Create a systematic trend across navigation dimensions
for i in range(s.axes_manager.navigation_shape[0]):
    for j in range(s.axes_manager.navigation_shape[1]):
        # Add position-dependent offset
        s.inav[i, j].data += i * 50 + j * 30
        
        # Add position-dependent slope variation
        energy_axis = s.axes_manager.signal_axes[0].axis
        s.inav[i, j].data += (i - 5) * 0.2 * energy_axis

print("Added noise and systematic variations")

# Plot the data to visualize
s.plot()

# %%
# **Create and configure the model**
#
# We'll fit a linear model to demonstrate residual analysis.

# Create model for all navigation positions
m = s.create_model()

# ✅ BEST PRACTICE: Use Expression component with descriptive name
line = hs.model.components1D.Expression("a * x + b", name="Linear")
m.append(line)

print(f"Created model with {len(m)} components:")
for i, comp in enumerate(m):
    if hasattr(comp, 'expression'):
        print(f"  {i+1}. {comp.name}: {comp.expression}")
    else:
        print(f"  {i+1}. {comp.name}: {comp.__class__.__name__}")

# %%
# **Parameter initialization across navigation dimensions**
#
# Initialize parameters with reasonable values for all navigation positions.

# ✅ BEST PRACTICE: Initialize parameters with reasonable estimates
line.a.value = 1.0  # Initial slope
line.b.value = 500  # Initial intercept

print("Initial parameters:")
print(f"Slope (a): {line.a.value:.3f}")
print(f"Intercept (b): {line.b.value:.3f}")

# %%
# **Robust model fitting with error handling**
#
# Perform multifit with proper error handling and monitoring.

print("Starting multifit for all navigation positions...")

try:
    # ✅ BEST PRACTICE: Use multifit for multidimensional data
    m.multifit(bounded=True, show_progressbar=True)
    
    print("✅ Multifit completed successfully!")
    
    # Display parameter statistics
    slopes = m.p_std[0].data if hasattr(m, 'p_std') else line.a.map['values']
    intercepts = m.p_std[1].data if hasattr(m, 'p_std') else line.b.map['values']
    
    print(f"\nParameter statistics:")
    print(f"Slopes: mean={np.mean(slopes):.3f}, std={np.std(slopes):.3f}")
    print(f"Intercepts: mean={np.mean(intercepts):.1f}, std={np.std(intercepts):.1f}")
    
except Exception as e:
    print(f"❌ Multifit failed: {e}")
    print("Attempting single position fit as fallback...")
    
    # Fallback to single position fit
    m.fit(bounded=True)
    print("✅ Single position fit completed")

# %%
# **Residual calculation and analysis**
#
# Calculate residuals and perform statistical analysis.

# Calculate residuals using HyperSpy signal arithmetic
model_signal = m.as_signal()
residuals = s - model_signal  # Preserves metadata automatically

# ✅ BEST PRACTICE: Calculate meaningful statistics
residual_stats = {
    'mean': np.mean(residuals.data),
    'std': np.std(residuals.data),
    'min': np.min(residuals.data),
    'max': np.max(residuals.data),
    'rms': np.sqrt(np.mean(residuals.data**2))
}

print(f"\n=== Residual Statistics ===")
for key, value in residual_stats.items():
    print(f"{key.upper()}: {value:.3f}")

# Calculate residual maps (statistics across signal dimension)
residual_mean_map = residuals.mean(axis='Energy')
residual_std_map = residuals.std(axis='Energy')
residual_max_map = residuals.max(axis='Energy')

print(f"\nResidual maps calculated:")
print(f"Mean residual map: {residual_mean_map}")
print(f"Std residual map: {residual_std_map}")
print(f"Max residual map: {residual_max_map}")

# %%
# **Comprehensive residual visualization**
#
# Create multiple visualizations to assess fit quality.

# 1. Plot the fitted model with residual using HyperSpy's built-in method
m.plot(plot_residual=True)

# 2. Plot residual maps to identify spatial patterns
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Mean residual map
im1 = axes[0].imshow(residual_mean_map.data, origin='lower', cmap='RdBu_r')
axes[0].set_title('Mean Residual Map')
axes[0].set_xlabel('X position')
axes[0].set_ylabel('Y position')
plt.colorbar(im1, ax=axes[0], label='Mean Residual')

# Residual standard deviation map
im2 = axes[1].imshow(residual_std_map.data, origin='lower', cmap='viridis')
axes[1].set_title('Residual Std Dev Map')
axes[1].set_xlabel('X position')
axes[1].set_ylabel('Y position')
plt.colorbar(im2, ax=axes[1], label='Residual Std Dev')

# Maximum residual map
im3 = axes[2].imshow(residual_max_map.data, origin='lower', cmap='plasma')
axes[2].set_title('Max Residual Map')
axes[2].set_xlabel('X position')
axes[2].set_ylabel('Y position')
plt.colorbar(im3, ax=axes[2], label='Max Residual')

plt.tight_layout()
plt.show()

# 3. Plot residual distribution
plt.figure(figsize=(10, 6))

# Histogram of all residuals
plt.subplot(2, 2, 1)
plt.hist(residuals.data.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True, alpha=0.3)

# Q-Q plot for normality assessment
from scipy import stats
plt.subplot(2, 2, 2)
stats.probplot(residuals.data.flatten(), dist="norm", plot=plt)
plt.title('Q-Q Plot (Normality Test)')
plt.grid(True, alpha=0.3)

# Residuals vs fitted values
plt.subplot(2, 2, 3)
plt.scatter(model_signal.data.flatten(), residuals.data.flatten(), alpha=0.5, s=1)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)

# Residuals vs observation order (for systematic patterns)
plt.subplot(2, 2, 4)
residual_flat = residuals.data.flatten()
plt.plot(range(len(residual_flat)), residual_flat, 'o-', alpha=0.5, markersize=1)
plt.xlabel('Observation Order')
plt.ylabel('Residuals')
plt.title('Residuals vs Observation Order')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# **Residual analysis and interpretation**
#
# Analyze residuals to assess model adequacy.

# Statistical tests for residual quality
print(f"\n=== Residual Analysis ===")

# Test for normality (should be approximately normal for good fit)
_, normality_p = stats.shapiro(residuals.data.flatten()[:5000])  # Sample for large datasets
print(f"Normality test p-value: {normality_p:.6f}")
if normality_p > 0.05:
    print("✅ Residuals appear normally distributed (good)")
else:
    print("⚠️  Residuals may not be normally distributed")

# Test for systematic patterns
residual_autocorr = np.corrcoef(residuals.data.flatten()[:-1], residuals.data.flatten()[1:])[0, 1]
print(f"Residual autocorrelation: {residual_autocorr:.4f}")
if abs(residual_autocorr) < 0.1:
    print("✅ Low autocorrelation (good)")
else:
    print("⚠️  High autocorrelation suggests systematic patterns")

# Calculate R-squared for overall fit quality
ss_res = np.sum(residuals.data**2)
ss_tot = np.sum((s.data - np.mean(s.data))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"\nOverall fit quality:")
print(f"R-squared: {r_squared:.4f}")
print(f"RMSE: {residual_stats['rms']:.3f}")

# Interpretation guidelines
print(f"\n=== Interpretation Guidelines ===")
if r_squared > 0.95:
    print("✅ Excellent fit (R² > 0.95)")
elif r_squared > 0.90:
    print("✅ Good fit (R² > 0.90)")
elif r_squared > 0.80:
    print("⚠️  Acceptable fit (R² > 0.80)")
else:
    print("❌ Poor fit (R² < 0.80) - consider model revision")

# %%
# **Advanced residual analysis techniques**
#
# Additional techniques for comprehensive residual analysis.

# Identify outliers using statistical methods
residual_data = np.array(residuals.data.flatten())
residual_z_scores = np.abs(stats.zscore(residual_data))
outliers = residual_z_scores > 3
n_outliers = np.sum(outliers)
outlier_percentage = (n_outliers / len(residual_z_scores)) * 100

print(f"\nOutlier analysis:")
print(f"Number of outliers (|z-score| > 3): {n_outliers}")
print(f"Percentage of outliers: {outlier_percentage:.2f}%")

if outlier_percentage < 1:
    print("✅ Low outlier percentage (good)")
elif outlier_percentage < 5:
    print("⚠️  Moderate outlier percentage")
else:
    print("❌ High outlier percentage - investigate data quality")

# Analyze residual patterns across navigation dimensions
print(f"\nSpatial residual patterns:")
mean_residual_by_x = np.mean(residuals.data, axis=(1, 2))
mean_residual_by_y = np.mean(residuals.data, axis=(0, 2))

print(f"X-direction residual trend: {np.std(mean_residual_by_x):.3f}")
print(f"Y-direction residual trend: {np.std(mean_residual_by_y):.3f}")

# %%
# **Best practices summary**
#
# Key takeaways for residual analysis in HyperSpy.

print(f"\n=== Best Practices Summary ===")
print("✅ Always calculate and visualize residuals")
print("✅ Use multiple visualization methods (histograms, Q-Q plots, spatial maps)")
print("✅ Test residuals for normality and systematic patterns")
print("✅ Calculate meaningful statistics (R², RMSE, outlier percentage)")
print("✅ Analyze residual patterns across navigation dimensions")
print("✅ Use statistical tests to assess model adequacy")
print("✅ Document residual analysis results")
print("✅ Consider model revision if residuals show systematic patterns")
print("✅ Compare residual statistics across different models")
print("✅ Use residual analysis to identify data quality issues")

print(f"\nModel residual analysis example completed successfully!")
print(f"Final R²: {r_squared:.4f}, RMSE: {residual_stats['rms']:.3f}")

# Choose the residual analysis plots as gallery thumbnails
# sphinx_gallery_thumbnail_number = 3

