"""
Using a Distributed Backend
===========================

There are many ways to set up a Dask backend. Here we show how to use the
``dask.distributed`` scheduler to run a computation on a local cluster as well as the
``dask_jobqueue`` to run a computation on a remote cluster.

"""
from dask.distributed import Client
from dask.distributed import LocalCluster
import dask.array as da
import hyperspy.api as hs
import numpy as np

cluster = LocalCluster()

client = Client(cluster)
client

# Any calculation will now use the distributed scheduler

# creating a lazy signal
s = hs.datasets.example_signals.EDS_SEM_Spectrum()
repeated_data = da.repeat(da.array(s.data[np.newaxis,:]),10, axis=0)
s = hs.signals.Signal1D(repeated_data).as_lazy()


summed = s.map(np.sum, inplace=False)  # uses distributed scheduler

s.plot()  # uses distributed scheduler to compute each chunk and then passes one chunk the memory

s.compute()  # uses distributed scheduler

# Even when the signal is not lazy any function which calls the ``map`` method will use
# the distributed scheduler

summed = s.map(np.sum, inplace=False)  # uses distributed scheduler


"""
Using The Distributed Scheduler with Dask-Jobqueue
==================================================
"""

from dask_jobqueue import SLURMCluster # or what ever scheduler you use
from dask.distributed import Client
cluster = SLURMCluster(cores=48,
                       memory='120Gb',
                       walltime="01:00:00",
                       queue='research')
cluster.scale(jobs=3) # get 3 nodes
client = Client(cluster)
client

# Any calculation will now use the distributed scheduler

# creating a lazy signal
s = hs.datasets.example_signals.EDS_SEM_Spectrum()
repeated_data = da.repeat(da.array(s.data[np.newaxis, :]),10, axis=0)
s = hs.signals.Signal1D(repeated_data).as_lazy()


summed = s.map(np.sum, inplace=False)  # uses distributed scheduler

s.compute()  # uses distributed scheduler

# Even when the signal is not lazy any function which calls the ``map`` method will use
# the distributed scheduler

summed = s.map(np.sum, inplace=False)  # uses distributed scheduler


# uses distributed scheduler to compute the currently viewed chunk and then passes one chunk the memory
# This takes slightly longer than the local scheduler because of the overhead of passing from the
# cluster where the computation is taking place to the local machine
s.plot()
