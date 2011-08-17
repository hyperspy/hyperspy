Statistical analysis
********************

Introduction/Concepts 
===================== 

The amount of data collected often greatly exceeds the true
dimensionality of the system being measured.  For instance, a typical
spectrum image has a number of dimensions equal to the number of
channels in the spectrum at each beam position.  With present
technology, that is on the order of 1000 dimensions.  For stacks of
images of a repetitive structure, the number of dimensions is the
number of pixels in each image, and can easily reach the order of
10,000 or higher.  However, it is not each individual channel in a
spectrum, nor each pixel in an image that is of interest.  Rather, it
is a partcular range and shape of a spectrum that identifies the
chemical constituents of the sample, or a set of pixels that
represents a change in column population or position.  There are
likely far fewer of these "dimensions," which are really the physical
phenomena of interest.  Multivariate analysis provides a means of
simplifying the representation of data as channels, transforrming the
data into a representation of components.  Ideally, these components
would match pure physical components, allowing direct identification
of the species present in the sample.  In practice, the components
obtained by the most common multivariate analysis method, principal
component analysis (PCA), are not directly interpretable.  Other
methods that derive components using different mathematical methods
and with different criteria for separation can provide much more
meaningful components, making interpretation easier.  The method
demonstrated here, independent component anlaysis (ICA), derives
components with minimally Gaussian statistics.  Both EEL spectra and
structural variations captured in images are expected to be
non-Gaussian, thus ICA is well suited to studying these systems.

What can MSA be applied to?
===========================

MVA can be applied to any data in which dimensions are directly
comparable within a data set.  These dimensions can be channels of an
energy resolved spectrum, pixels across several images of a repeat
structure, or pixels from several images in a time-resolved series.
The interpretation of any MVA results of course depends on the data
being fed into it, but the principles are quite similar - what is the
simplest way to examine how the sample is varying over time or space?

PCA workflow
============

PCA is most commonly applied as a means of noise reduction and data
simplification.  To perform PCA on your data set, run the following
command:

.. code-block:: python

    s.principal_components_analysis()

There are several options for PCA, including data pre-treatment and
the algorithm used to compute the analysis.  If you're curious, please
review these options using the ? syntax of ipython:

.. code-block:: python

    s.principal_components_analysis?

The primary purpose of PCA is to estimate the dimensionality of your
system.  To do this visually, you must examine the scree plot for you
data, which is a plot of the eigenvalue vs eigenvalue index (in other
words, the amount of variance that each component accounts for.)
This plot drops quickly, eventually becoming a slowly descending
line.  The point at which it becomes linear (often referred to as an
elbow) is generally judged to be the dimensionality of your system.

To obtain a scree plot, run the following command:

.. code-block:: python

    s.plot_lev()


ICA workflow
============

ICA is advantageous over PCA for its ability to return components that
are often more physically meaningful.  The reasons behind this are
that ICA relaxes the orthogonality requirement among components, and
that ICA seeks non-Gaussian components.  These work out to better
results because in
reality, data are often correlated in observation space
(non-orthogonal), and the distributions of real data are not Gaussian.

To perform ICA on your data, run the following command:

.. code-block:: python

    s.independent_components_analysis(number_of_components)

.. NOTE::
    You must have run PCA before you attempt to run ICA.  ICA uses the
    components from PCA, but analyzes them to come up with a different
    set of components.

.. NOTE::
    You must pass an integer number of components to ICA.  The best
    way to estimate this number is by looking at the Scree plot, as
    described above in the PCA workflow.


Visualising results
===================


Saving and loading results
==========================

