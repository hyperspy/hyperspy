
Cluster analysis
================

.. versionadded:: 1.6

A Hyperspy signal can represent a number of large arrays of different signals such as a map of chemical composition (EDS, XRF),
a stack of elemental maps, chemical state (EELS/XANES), or structure (diffraction pattern). 
Identifying and extracting trends from large datasets is often difficult and PCA, BSS, NMF and cluster analysis play an important role
in this process. 
Cluster analysis aims to group the data, or rather features in the data, into N sets with similar properties and it does this 
by comparing the "distances" (or similar metric) between different features and grouping those that are closest.   

Cluster analysis can be performed on raw data but is more commonly used on an extracted set of features.
PCA decomposition is one way to produces a set features as it reduces the description of the data into a set of loadings and factors. 
The loadings capture a core representation of the features in the spectra or images and the factors provide the mixing of these loadings
to describe that describe the original data.  
These mixing factors can be interrogated to find similar regions and for PCA these represent relatively small data sets (typically <20 values)
compared to clustering the raw data with the raw data which can be several thousand data points.

A detailed description of the application of cluster analysis in spectro-microscopy and further details on the theory and implementation can be found here.  
:ref:`[Lerotic2004] <Lerotic2004>`.


Nomenclature
------------

Taking the example of a 1D Signal of dimensions `(20, 10|4)` containing the
dataset, we say there are 200 *samples*. The four measured parameters are the
*features*. If we choose to search for 3 clusters within this dataset, we
derive two main values: the `labels`, of dimensions `(20, 10|3)` (each
sample is assigned a label to each cluster), and the `centers`, of
dimensions `(3, 4)` (each centre has a coordinate in each feature).


Example
-------

We can use the `make_blobs` function supplied by `scikit-learn` to see how
clustering might work in practice.

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> from sklearn.datasets import make_blobs
    >>> data = hs.signals.Signal1D(
    >>>     make_blobs(
    >>>         n_samples=200,
    >>>         n_features=10,
    >>>         shuffle=False)[0].reshape(20, 10, 10))
	

The resultant signal contains 3 distinct "types" of signal. 
If we examine the signal using PCA we can see that there are 3 regions but
there interpretation of the signal is a little ambigous.  

To see how cluster analysis works it's best to first examine the signal.
Moving around the image you should be able to see 3 distinct regions in which
the 1D signal modulates slightly.  

.. code-block:: python

    >>> data.plot()


If we then perform PCA we start to see the 3 regions a little more clearly but
the factors and loadings don't match up with the original 1D signals or image.

.. code-block:: python

    >>> data.decomposition()
    >>> data.plot_decomposition_results()


We can then cluster, using the decomposition results, to find similar regions
and the representative features in those regions. 
This indentifies 3 regions and the average or representative 1D signals in 
those regions

.. code-block:: python

    >>> data.cluster_analysis(3)
    >>> data.plot_cluster_results()


To see what the labels the cluster algorithm has assigned you can inspect

.. code-block:: python

    >>> data.learning_results.cluster_membership


This are split into a cluster_labels array to help plotting and masking

.. code-block:: python

    >>> data.learning_results.cluster_labels


kmeans and agglomerative clustering methods are currently supported and 
additional keywords can be passed directly to the scikit learn methods.

.. code-block:: python

    >>> data.cluster_analysis(3, algorithm='agglomerative',
                              kwargs={affinity='cosine', linkage='average'})
    >>> data.plot_cluster_results()


In this case we know there are 3 signals but for real examples the difficulty
can be how to define the number of clusters to use and a number of metrics
such as the elbow, Silhouette and Gap metrics can be used to determine the optimal 
number of clusters. 
The elbow method measures the sum-of-squares of the distances within a cluster
and like the PCA decomposition method an elbow or point where the gains 
diminish with increasing number of clusters indicates the ideal number of 
clusters.
Silhouette analysis measures how well seperated clusters are and can be used to
determine the most likely number of clusters. As the scoring is a measure of
seperation of clusters a number of solutions may occur and maxima in the scores
are used to indicate possible solutions.   
Gap analysis is similar but compares the "gap" between the clustered data 
results and those from a randomly data set of the same size. The largest gap
indicates the best clustering. The metric results can be plotted to
check how well defined the clustering is.   

.. code-block:: python

    >>> data.evaluate_number_of_clusters(
    >>>     use_decomposition_results=True,metric="gap")
    >>> data.plot_cluster_metric()
    
The optimal number of clusters can be set or accessed from the learning 
results

.. code-block:: python

    >>> data.learning_results.number_of_clusters
    
If running cluster analysis and the number of clusters have not been
specified the algorithm will attempt to use the estimated number of clusters

.. code-block:: python

    >>> data.cluster_analysis()







