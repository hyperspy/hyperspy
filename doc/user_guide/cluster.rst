
Cluster analysis 
**********

Introduction
============

A Hyperspy signal can represent a number of large arrays of different signals such as a map of chemical composition (EDX,XRF),
a stack of elemental maps, chemical state (EELS/XANES) ,or structure (diffraction pattern). 
Identifying and extracting trends from large datasets is often difficult and both PCA and cluster analysis play an important role
in this process. 
Cluster analysis aims to group the data, or rather features in the data, into N sets of similar properties and it does this 
by comparing the "distances" (or similar metric) between different features and grouping those that are closest.   

Cluster analysis can be performed on raw data but is more commonly used on the PCA decomposition results as  
PCA reduces the data set into a smaller set of features (i.e the loadings). 
The loadings capture a core representation of the features in the spectra or images and the factors provide the mixing of these loadings
to describe that describe the original data.  
These mixing factors can be interrogated to find similar regions and for PCA these represent relatively small data sets (typically <20 values)
compared to clustering the raw data with the raw data which can be several thousand data points.

A detailed description of the application of cluster analysis in spectro-microscopy and further details on the theory and implementation can be found here.  
:ref:`[Lerotic2004] <Lerotic2004>`.


Nomenclature
============

Taking the example of a 1D Signal of dimensions `(20, 10|4)` containing the
dataset, we say there are 200 *samples*. The four measured parameters are the
*features*. If we choose to search for 3 clusters within this dataset, we
derive two main values: the `labels`, of dimensions `(20, 10|3)` (each
sample is assigned a label to each cluster), and the `centers`, of
dimensions `(3, 4)` (each centre has a coordinate in each feature).


Example
=======

We can use the `make_blobs` function supplied by `scikit-learn` to see how
clustering might work in practice.

.. code-block:: python

    import hyperspy.api as hs
    from sklearn.datasets import make_blobs
    data = hs.signals.Signal1D(
        make_blobs(
            n_samples=200,
            n_features=10,
            shuffle=False
            )[0].reshape(20, 10, 10))

The resultant signal contains 3 distinct "types" of signal. 
If we examine the signal using PCA we can see that there are 3 regions but
there interpretation of the signal is a little ambigous.  

To see how cluster analysis works it's best to first examine the signal.
Moving around the image you should be able to see 3 distinct regions in which
the 1D signal modulates slightly.  

.. code-block:: python

    data.plot()

.. code-block:: python

If we then perform PCA we start to see the 3 regions a little more clearly but
the factors and loadings don't match up with the original 1D signals or image.

.. code-block:: python

    data.decomposition()
    data.plot_decomposition_results()

.. code-block:: python

We can then cluster, using the decomposition results, to find similar regions
and the representative features in those regions. 
This indentifies 3 regions and the average or representative 1D signals in 
those regions

.. code-block:: python

    data.cluster(3)
    data.plot_cluster_results()

.. code-block:: python

In this case we know there are 3 signals but for real examples the difficulty
can be how to define the number of clusters to use.  
Silhouette analysis measures how well seperated clusters are and can be used to
determine the most likely number of clusters by scoring the clustering.   

.. code-block:: python
    max_number_of_clusters = 6
    best_cluster=data.evalute_no_of_clusters(max_number_of_clusters, 
                                             use_decomposition_results=True)
    print("best clusters and scores",best_cluster)
    

.. code-block:: python




