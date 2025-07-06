"""
Basic file loading operations
=============================

This example demonstrates basic file loading operations in HyperSpy,
including loading different formats and working with metadata.
"""

# %%
# Create some sample data and save it in different formats
import numpy as np
import hyperspy.api as hs
import tempfile
import os

# Create a sample spectrum image
data = np.random.random((20, 30, 100))
s = hs.signals.Signal1D(data)

# Set up axes using batch assignment
s.axes_manager.signal_axes[0].name = 'Energy'
s.axes_manager.signal_axes[0].units = 'eV'
s.axes_manager.signal_axes[0].scale = 0.1
s.axes_manager.signal_axes[0].offset = 100
s.axes_manager.navigation_axes[0].name = 'X'
s.axes_manager.navigation_axes[0].units = 'nm'
s.axes_manager.navigation_axes[0].scale = 0.5
s.axes_manager.navigation_axes[1].name = 'Y'
s.axes_manager.navigation_axes[1].units = 'nm'
s.axes_manager.navigation_axes[1].scale = 0.5

s.metadata.General.title = 'Sample spectrum image'
s.metadata.General.author = 'HyperSpy example'

# %%
# ## Save the data in different formats
# 
# HyperSpy supports multiple file formats with varying levels of metadata preservation

with tempfile.TemporaryDirectory() as temp_dir:
    # Save as HyperSpy format (preserves all metadata)
    hspy_file = os.path.join(temp_dir, "sample.hspy")
    s.save(hspy_file)
    
    # Save as compressed HyperSpy format
    zspy_file = os.path.join(temp_dir, "sample.zspy")
    s.save(zspy_file)
    
    # %%
    # ## Loading files
    # 
    # Files can be loaded using different approaches with various format specifications
    
    # Load HyperSpy format - metadata and axis information preserved
    s_loaded = hs.load(hspy_file)
    
    # Verify the loaded signal matches original
    assert s_loaded.metadata.General.title == s.metadata.General.title
    assert s_loaded.metadata.General.author == s.metadata.General.author
    
    # %%
    # ## Loading with specific signal type
    # 
    # Override automatic signal type detection when needed
    
    s_loaded_spectrum = hs.load(hspy_file, signal_type="Signal1D") 
    
    # %%
    # ## Metadata structure examination
    # 
    # HyperSpy preserves both original file metadata and HyperSpy-specific metadata
    
    # Access original metadata from file format
    original_keys = list(s_loaded.original_metadata.keys()) if hasattr(s_loaded.original_metadata, 'keys') else []
    
    # HyperSpy metadata structure is organized and accessible
    metadata_structure = s_loaded.metadata

# %%
# ## Working with built-in datasets
# 
# HyperSpy provides several built-in datasets for testing and demonstration

# Load built-in test data - no file I/O required
built_in_data = hs.data.two_gaussians()

# Plot the built-in dataset to examine its structure
built_in_data.plot()
