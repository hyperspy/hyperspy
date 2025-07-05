"""
Specifying File Format and Signal Type
=======================================

This example demonstrates how to specify file format and signal type
when loading data with HyperSpy, which is useful when working with files
that have non-standard extensions or when you want to explicitly specify
the data interpretation.

"""

import hyperspy.api as hs
import numpy as np
import tempfile
import os

# %%
# Create example datasets
# -----------------------
# Create some example data
print("Creating example datasets...")

# Create a spectrum (1D signal)
spectrum_data = np.random.random((10, 100))
spectrum = hs.signals.Signal1D(spectrum_data)
spectrum.metadata.General.title = "Test Spectrum"
spectrum.axes_manager.signal_axes[0].name = "Energy"
spectrum.axes_manager.signal_axes[0].units = "eV"
spectrum.axes_manager.signal_axes[0].scale = 0.1

# Create an image (2D signal)
image_data = np.random.random((5, 64, 64))
image = hs.signals.Signal2D(image_data)
image.metadata.General.title = "Test Image"
image.axes_manager.signal_axes[0].name = "x"
image.axes_manager.signal_axes[1].name = "y"

# %%
# Save files for testing
# ----------------------
print("Saving example files...")

# Create temporary directory for examples
with tempfile.TemporaryDirectory() as temp_dir:
    # Save in standard HyperSpy format
    hspy_file = os.path.join(temp_dir, "spectrum.hspy")
    spectrum.save(hspy_file)
    
    # Save with non-standard extension (save as hspy first, then rename)
    weird_extension_file = os.path.join(temp_dir, "spectrum.unknown_ext")
    temp_hspy = os.path.join(temp_dir, "temp.hspy")
    spectrum.save(temp_hspy)
    os.rename(temp_hspy, weird_extension_file)
    
    # Save image in HyperSpy format
    image_file = os.path.join(temp_dir, "image.hspy")
    image.save(image_file)
    
    # %%
    # ## 1. Loading with automatic format detection
    # 
    # HyperSpy can automatically detect file formats based on extension and content
    
    # Load normally - HyperSpy will detect format automatically
    s1 = hs.load(hspy_file)
    print(f"Loaded signal: {s1}")
    print(f"Signal type: {type(s1).__name__}")
    print(f"Shape: {s1.data.shape}")
    
    # %%
    # ## 2. Specifying file format explicitly
    # 
    # When files have unusual extensions, you can specify the format manually
    
    # Load file with unknown extension by specifying format
    print(f"Loading file with unknown extension: {os.path.basename(weird_extension_file)}")
    s2 = hs.load(weird_extension_file, file_format="hspy")
    print(f"Loaded signal: {s2}")
    print(f"Signal type: {type(s2).__name__}")
    print(f"Data matches original: {np.array_equal(s1.data, s2.data)}")
    
    # %%
    # ## 3. Available signal types
    # 
    # HyperSpy supports various signal types for different kinds of data
    
    # Print known signal types
    print("Available signal types on this system:")
    hs.print_known_signal_types()
    
    # %%
    # ## 4. Specifying signal type explicitly
    # 
    # You can override automatic signal type detection to ensure the right class is used
    
    # Load and explicitly specify signal type
    print("Loading spectrum as generic Signal1D:")
    s3 = hs.load(hspy_file, signal_type="Signal1D")
    print(f"Signal type: {type(s3).__name__}")
    
    # Load and specify different signal type (if available)
    # Note: Some signal types require extension packages like eXSpy
    print("\nLoading image as generic Signal2D:")
    s4 = hs.load(image_file, signal_type="Signal2D")
    print(f"Signal type: {type(s4).__name__}")
    print(f"Shape: {s4.data.shape}")
    
    print("\n" + "="*50)
    print("5. Working with metadata from different formats")
    print("="*50)
    
    # Demonstrate metadata handling
    print("Original metadata keys:")
    print(list(spectrum.metadata.General.keys()))
    
    # Load and check metadata preservation
    loaded_spectrum = hs.load(hspy_file)
    print("\nLoaded metadata:")
    print(f"Title: {loaded_spectrum.metadata.General.title}")
    print(f"Energy axis name: {loaded_spectrum.axes_manager.signal_axes[0].name}")
    print(f"Energy axis units: {loaded_spectrum.axes_manager.signal_axes[0].units}")
    print(f"Energy axis scale: {loaded_spectrum.axes_manager.signal_axes[0].scale}")
    
    print("\n" + "="*50)
    print("6. Error handling for unsupported formats")
    print("="*50)
    
    try:
        # Try to load with unsupported format
        hs.load(hspy_file, file_format="nonexistent_format")
    except (ValueError, OSError, Exception) as e:
        print(f"Expected error for unsupported format: {type(e).__name__}")
        print(f"Error message: {str(e)[:100]}...")  # Truncate long messages
    
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print("✓ File format can be explicitly specified using 'file_format' parameter")
    print("✓ Signal type can be specified using 'signal_type' parameter")
    print("✓ HyperSpy automatically detects format from file extension when possible")
    print("✓ Explicit specification is useful for files with unusual extensions")
    print("✓ Signal type specification controls how data is interpreted and processed")
    print("✓ Metadata is preserved during load/save operations in supported formats")
