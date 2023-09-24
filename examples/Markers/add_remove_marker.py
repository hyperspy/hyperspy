"""
Add/Remove items from existing Markers
======================================

This example shows how to add or remove marker from an existing collection.
This is done by setting the parameters (offsets, sizes, etc.) of the collection.

"""
#%%
# Create a signal
import hyperspy.api as hs
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng(0)
data = np.arange(15*100*100).reshape((15, 100, 100))
s = hs.signals.Signal2D(data)

#%%
# Create text marker

# Define the position of the texts
offsets = np.stack([np.arange(0, 100, 10)]*2).T + np.array([5,]*2)
texts = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'f', 'h', 'i'])

m = hs.plot.markers.Texts(
    offsets=offsets,
    texts=texts,
    sizes=3,
    )

print(f'Number of markers is {len(m)}.')

s.plot()
s.add_marker(m)

#%%
# Remove the last text of the collection
# ######################################



# Set new texts and offsets parameters with one less item
m.remove_items(indices=-1)

print(f'Number of markers is {len(m)} after removing one marker.')

s.plot()
s.add_marker(m)

#%%
# Add another text of the collection
# ##################################

# Define the position in the middle of the axes

m.add_items(offsets=np.array([[50, 50]]), texts=np.array(["new text"]))

print(f'Number of markers is {len(m)} after adding the text {texts[-1]}.')

s.plot()
s.add_marker(m)

#%%
# sphinx_gallery_thumbnail_number = 2
