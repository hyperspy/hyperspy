"""
Navigation Hotkeys and Controls
===============================

This example demonstrates navigation hotkeys and controls for interacting with
signals in HyperSpy plots. These shortcuts make it easier to navigate through
multidimensional datasets.

Key navigation shortcuts:
- Arrow keys: Navigate through signal dimensions
- Ctrl+Arrow keys: Jump to start/end of dimensions
- Home/End: Jump to first/last position
- Page Up/Down: Move in larger steps
"""

import numpy as np
import hyperspy.api as hs

# %%
# Create test signals for navigation demonstration
# ------------------------------------------------
# Create a 3D spectrum image for navigation
data = np.random.random((20, 30, 1000))

# Add some structure to make navigation more interesting
for i in range(20):
    for j in range(30):
        # Add Gaussian peaks at different positions
        center = 200 + i * 10 + j * 5
        sigma = 20 + i * 2
        x = np.arange(1000)
        data[i, j, :] += 1000 * np.exp(-(x - center)**2 / (2 * sigma**2))

s = hs.signals.Signal1D(data)

# Set up axes using batch assignment
s.axes_manager.signal_axes[0].set(name='Energy', units='eV', scale=0.1, offset=100)
s.axes_manager.navigation_axes[0].set(name='X', units='nm', scale=1.0)
s.axes_manager.navigation_axes[1].set(name='Y', units='nm', scale=1.0)

s.metadata.General.title = "Navigation Example"

# %%
# ## Navigation Hotkeys and Controls
# 
# Once you plot the signal, you can use these keyboard shortcuts to navigate:
# 
# ### Basic Navigation
# - **Arrow keys**: Move through navigation dimensions
# - **Home**: Jump to first position  
# - **End**: Jump to last position
# - **Page Up/Down**: Move in larger steps
# 
# ### Advanced Navigation  
# - **Ctrl+Arrow keys**: Jump to start/end of dimensions
# 
# > **Tip**: Make sure the plot window is active (clicked on) for the hotkeys to work!

s.plot()

# %%
# Additional navigation features
# ------------------------------
# You can also navigate programmatically
print(f"Current navigation position: {s.axes_manager.indices}")

# Set specific navigation position
s.axes_manager.indices = (10, 15)
print(f"New navigation position: {s.axes_manager.indices}")

# Navigate using axes managers
nav_x = s.axes_manager.navigation_axes[0]
nav_y = s.axes_manager.navigation_axes[1]

print(f"X axis range: {nav_x.low_value:.1f} to {nav_x.high_value:.1f} {nav_x.units}")
print(f"Y axis range: {nav_y.low_value:.1f} to {nav_y.high_value:.1f} {nav_y.units}")
