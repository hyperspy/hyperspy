"""
Using a Distributed Backend
===========================

There are many ways to set up a Dask backend. Here we show how to use the
`dask.distributed <https://distributed.dask.org/>`_ scheduler to run a
computation on a local cluster or on a :ref:`remote cluster<remote_cluster-label>`.

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
