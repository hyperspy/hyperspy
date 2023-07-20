"""
Using a Dask Backend with a single thread
=========================================

There are many ways to set up a Dask backend. Here we show how to use the
'single-threaded' which is useful for debugging and testing.

"""
import dask
import hyperspy.api as hs
import numpy as np
import dask.array as da

# setting the scheduler to single-threaded globally
dask.config.set(scheduler='single-threaded')

# creating a lazy signal
s = hs.datasets.example_signals.EDS_SEM_Spectrum()

repeated_data = da.repeat(da.array(s.data[np.newaxis, :]), 10, axis=0)
s = hs.signals.Signal1D(repeated_data).as_lazy()

summed = s.map(np.sum, inplace=False)  # uses distributed scheduler

s.plot()  # uses single-threaded scheduler to compute each chunk and then passes one chunk the memory

s.compute()  # uses single-threaded scheduler

"""
Using a Dask Backend with a Single Thread
=========================================
Alternatively, you can set the scheduler to single-threaded for a single function call by
setting the ``scheduler`` keyword argument to ``'single-threaded'``.

Or for something like plotting you can set the scheduler to single-threaded for the
duration of the plotting call by using the ``with dask.config.set`` context manager.
"""


repeated_data = da.repeat(da.array(s.data[np.newaxis,:]), 10, axis=0)
s = hs.signals.Signal1D(repeated_data).as_lazy()

s.compute(scheduler="single-threaded")  # uses single-threaded scheduler

with dask.config.set(scheduler='single-threaded'):
    s.plot()  # uses single-threaded scheduler to compute each chunk and then passes one chunk the memory
