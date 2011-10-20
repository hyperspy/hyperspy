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

Reconstructing data from a partial set of components
=====================================================

As we mentioned before, the primary purpose of PCA is often to filter
noise.  To reconstruct your data using a limited set of components,
omitting the later components that ideally contain only noise, do the
following:

.. code-block:: python

    s2=s.pca_build_SI(components)

.. NOTE:: 
    The components argument can be one of several things (None, int,
    or list of ints):

    * if None, rebuilds SI from all components
    * if int, rebuilds SI from components in range 0-given int
    * if list of ints, rebuilds SI from only components in given list

.. NOTE::
    Unlike most of the analysis functions, this function returns a new
    object.  The new object is something that you have to give a
    handle to, so that you can perform analysis on that object later.
    That is why we use the **s2=s.pca_build_SI()** syntax.

It is very important to examine the residuals between your original
data and your reconstructed data before you go any further.  If you
have any recognizable structure in either the image space or the
spectral space, that means you did not include enough components in
your reconstruction, and you have lost important experimental
information.  To examine residuals, use the plot_residual method on
the reconstructed object.

.. code-block:: python

    s2.plot_residual()


ICA workflow
============

ICA is advantageous over PCA for its ability to return components that
are often more physically meaningful.  The reasons behind this are
that ICA relaxes the orthogonality requirement among components, and
that ICA seeks non-Gaussian components.  These work out to better
results because in reality, data are often correlated in observation
space (non-orthogonal), and the distributions of real data are not
Gaussian.

To perform ICA on your data, run the following command:

.. code-block:: python

    s.independent_components_analysis(number_of_components)

.. NOTE::
    You must have run PCA before you attempt to run ICA.  ICA uses the
    components from PCA, but analyzes them to come up with a different
    set of components.

.. NOTE::
    If you reconstructed an SI from principal components, you need to
    run PCA again before you can run ICA.

.. NOTE::
    You must pass an integer number of components to ICA.  The best
    way to estimate this number is by looking at the Scree plot, as
    described above in the PCA workflow.


Visualising results
===================

Plot methods exist for both components and score maps.  Most of these
methods begin with plot.  Most of these methods take at least one
argument - the number of components or score maps to plot.

To explore the plot methods available for an object (we'll use s for
this example), type the following, and hit the tab key.

.. code-block:: python

    s.plot

You can then start typing whichever plot method you want, then hit the
tab key again.  It will auto-complete up to the point where there is
more than one match.

.. code-block:: python

    s.plotP (hit tab)
    s.plotPca_ (hit tab)
    s.plotPca_factors(number_of_components)
    s.plotPca_scores(number_of_components)


Saving and loading results
==========================

You can save the entire object on which you've done MVA (this saves
the data along with the MVA results).  For this, just use the base
**save** method.  Alternatively, to save just the MVA results, you can use
the specialized **exportPca_results** and
**exportIca_results** methods.  This has the advantage of being
easier to import into other data analysis programs, such as Digital
Micrograph.

.. code-block:: python

    # save the entire object
    s.save(filename)
    # save the principal components and scores to files themselves
    s.exportPca_results()
    # save the independent components to the rpl format, which is
    #   easily importable into Digital Micrograph
    s.exportIca_results(factor_format='rpl', score_format='rpl')



