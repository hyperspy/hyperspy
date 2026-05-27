"""
File Saving in Different Formats
=================================

This example demonstrates how to save HyperSpy signals in various file formats
and the considerations for each format.
"""

import numpy as np
import hyperspy.api as hs
import tempfile
import os

# %%
# **Creating test signals for file format demonstration**
#
# We'll create example signals with proper metadata to demonstrate how different 
# file formats preserve or lose information during save/load operations.

# Create a test signal for saving examples
data_1d = np.random.random(100) * 1000
signal_1d = hs.signals.Signal1D(data_1d)
signal_1d.axes_manager.signal_axes.set(
    name=['Energy'],
    units=['eV'],
    scale=[0.1]
)
signal_1d.metadata.General.title = "Test Spectrum"

data_2d = np.random.random((64, 64)) * 255
signal_2d = hs.signals.Signal2D(data_2d)
signal_2d.axes_manager.signal_axes.set(
    name=['x', 'y'],
    units=['nm', 'nm']
)
signal_2d.metadata.General.title = "Test Image"

# %%
# **Setting up temporary directory for file format examples**
#
# We'll save files to a temporary directory to demonstrate different formats.

# Create temporary directory for examples
temp_dir = tempfile.mkdtemp()

# %%
# **HyperSpy native formats (recommended)**
#
# The .hspy and .zspy formats preserve all metadata, axes information, and signal type.
# These are the recommended formats for scientific data analysis workflows.

# Save as .hspy - preserves all metadata and axes information
hspy_file = os.path.join(temp_dir, "spectrum.hspy")
signal_1d.save(hspy_file)
# File saved as .hspy format with complete metadata preservation

# Save as .zspy - compressed HyperSpy format
zspy_file = os.path.join(temp_dir, "spectrum.zspy")
signal_1d.save(zspy_file)
# File saved as .zspy format - compressed but retains all information

# %%
# **Common image formats**
#
# Standard image formats are convenient for visualization but may lose scientific metadata.
# Use these for publication figures or when interoperability is needed.

# Save 2D signal as TIFF
tiff_file = os.path.join(temp_dir, "image.tiff")
signal_2d.save(tiff_file)
# TIFF format saved - good for 2D images, preserves some metadata

# Save as PNG (will lose axes information)
png_file = os.path.join(temp_dir, "image.png")
# Convert to uint8 for PNG compatibility using HyperSpy's change_dtype
signal_2d_uint8 = signal_2d.deepcopy()
# First normalize to 0-255 range, then convert to uint8
signal_2d_uint8.data = ((signal_2d_uint8.data - signal_2d_uint8.data.min()) / 
                       (signal_2d_uint8.data.max() - signal_2d_uint8.data.min()) * 255)
signal_2d_uint8.change_dtype(np.uint8)
signal_2d_uint8.save(png_file)
# PNG format saved - good for web/presentations but loses scientific metadata

# %%
# **Other scientific formats**
#
# HDF5 and raw data formats provide different trade-offs between compatibility,
# file size, and metadata preservation.

# Save as HDF5
hdf5_file = os.path.join(temp_dir, "data.hdf5")
signal_1d.save(hdf5_file, file_format="HSPY")
# HDF5 format saved - excellent for large datasets and cross-platform compatibility

# Save as NumPy array (data only, no metadata)
npy_file = os.path.join(temp_dir, "data.npy")
np.save(npy_file, signal_1d.data)
# Raw data saved as NumPy array - fastest but loses all metadata

# %%
# **Demonstrate loading and comparing file formats**
#
# Different formats preserve different amounts of information. Let's compare what happens
# when we reload signals saved in various formats.

# Load HyperSpy format (preserves everything)
loaded_hspy = hs.load(hspy_file)
# Loaded .hspy format: complete signal with all metadata preserved
# Signal type and axes information remain intact

# Load TIFF (loses some metadata)
loaded_tiff = hs.load(tiff_file)
# Loaded .tiff format: image data preserved but some metadata may be lost
# Basic axes information often retained for TIFF files

# %%
# **File size comparison across formats**
#
# Different formats have different storage efficiencies and compression characteristics.

# Show file sizes for all saved formats
file_sizes = {}
for filename in [hspy_file, zspy_file, tiff_file, png_file, hdf5_file, npy_file]:
    if os.path.exists(filename):
        size_kb = os.path.getsize(filename) / 1024
        file_sizes[os.path.basename(filename)] = size_kb

# File size comparison reveals format efficiency:
# - .zspy files are typically smallest due to compression
# - .npy files are compact but lack metadata
# - .hspy files balance metadata preservation with reasonable size

# %%
# **Best practices summary for file format selection**
#
# Choose formats based on your analysis needs and collaboration requirements.

recommended_formats = {
    "hspy": "Best for preserving all HyperSpy information",
    "zspy": "Compressed version of .hspy, good for large datasets", 
    "hdf5": "Good for interoperability with other tools"
}

avoid_formats = {
    "png_jpg": "Lossy compression, no metadata",
    "npy": "Only raw data, no axes or metadata information"
}

# %%
# **Format selection guidelines:**
#
# **For scientific data preservation:**
# - .hspy: Best for preserving all HyperSpy information
# - .zspy: Compressed version of .hspy, good for large datasets  
# - .hdf5: Good for interoperability with other tools
#
# **Avoid for scientific data:**
# - .png/.jpg: Lossy compression, no metadata
# - .npy: Only raw data, no axes or metadata information

# %%
# **Clean up temporary files**

# Clean up temporary files
import shutil
shutil.rmtree(temp_dir)
# Temporary directory and all example files have been cleaned up
