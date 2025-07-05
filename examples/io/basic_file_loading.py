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

# Set up axes
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
# Save the data in different formats
with tempfile.TemporaryDirectory() as temp_dir:
    # Save as HyperSpy format (preserves all metadata)
    hspy_file = os.path.join(temp_dir, "sample.hspy")
    s.save(hspy_file)
    print(f"Saved as HyperSpy format: {hspy_file}")
    
    # Save as compressed HyperSpy format
    zspy_file = os.path.join(temp_dir, "sample.zspy")
    s.save(zspy_file)
    print(f"Saved as compressed HyperSpy format: {zspy_file}")
    
    # %%
    # ## Loading files
    # 
    # Files can be loaded using different approaches with various format specifications
    
    # Load HyperSpy format
    s_loaded = hs.load(hspy_file)
    print(f"Loaded from .hspy: {s_loaded}")
    print(f"Title: {s_loaded.metadata.General.title}")
    print(f"Author: {s_loaded.metadata.General.author}")
    
    # %%
    # Load with specific signal type (overriding automatic detection)
    s_loaded_spectrum = hs.load(hspy_file, signal_type="Signal1D") 
    print(f"\nLoaded with specific signal type: {s_loaded_spectrum}")
    
    # %%
    # ## Metadata structure
    # 
    # HyperSpy preserves both original file metadata and HyperSpy-specific metadata
    print("Original metadata keys:", list(s_loaded.original_metadata.keys()) if hasattr(s_loaded.original_metadata, 'keys') else "None")
    print("Metadata structure:")
    print(s_loaded.metadata)

# %%
# ## Working with built-in datasets
# 
# HyperSpy provides several built-in datasets for testing and demonstration

# Load built-in test data
built_in_data = hs.data.two_gaussians()
print(f"Built-in data: {built_in_data}")
built_in_data.plot()
