SAMFire
*******

.. versionadded:: 1.0
    SAMFire

SAMFire (Smart Adaptive Multi-dimensional Fitting) is an algorithm created to
avoid the starting value (or local / false minima) problem, which often arises
when fitting multi-dimensional datasets.

The algorithm will be described in full when accompanying paper is published,
however we make the implementation available before that.

The idea
--------

The main idea of SAMFire is to change two things, when compared to the
traditional way of fitting datasets with many dimensions in the navigation
space:

 #. Pick a more sensible pixel fitting order.
 #. Calculate the pixel starting parameters from already fitted parts of the
     dataset.

Both of these aspects are linked one to another and are represented by two
different strategy families that SAMFfire uses while operating.

Strategies
**********

While operating, SAMFire uses a list of strategies to determine how to select
next pixel and estimate its starting parameters. Only one strategy is used at a
time. Next strategy is activated when no new pixels are able to fit using the
current one. Once either the strategy list is exhausted or the full dataset
fitted, the algorithm terminates.

There are two families of strategies. In each family there may be many
strategies, using different statistical or significance measures.

As a rule of thumb, the first strategy in the list should always be from the
local family, followed by a strategy from the global family. 

Local strategy family
---------------------

These strategies assume that locally neighbouring pixels are similar. As a
result, the pixel fitting order seems to follow data-suggested order, and the
starting values are computed from the surrounding already fitted pixels.

More information about the exact procedure will be available once the
accompanying paper is published.


Global strategy family
----------------------

Global strategies assume that the navigation coordinates of each pixel bear no
relation to it's signal (i.e. the location of pixels is meaningless). As a
result, the pixels are selected at random to ensure uniform sampling of the
navigation space. 

A number of candidate starting values are computed form global statistical
measures. These values are all attempted in order until a satisfactory result
is found (not necessarily testing all available starting guesses). As a result,
on average each pixel requires significantly more computations when compared to
a local strategy.

More information about the exact procedure will be available once the
accompanying paper is published.

Seed points
***********

Due to the strategies using already fitted pixels to estimate the starting
values, at least one pixel has to be fitted beforehand by the user.

The seed pixel(s) should be selected to require the most complex model present
in the dataset, however in-built goodness of fit checks ensure that only
sufficiently well fitted values are allowed to propagate.

If the dataset consists of regions (in the navigation space) of highly
dissimilar pixels, often called "domain structures", at least one seed pixel
should be given for each unique region.

If the starting pixels were not optimal, only part of the dataset will be
fitted. In such cases it is best to allow the algorithm terminate, then provide
new (better) seed pixels by hand, and restart SAMFire. It will use the
new seed together with the already computed parts of the data.

Examples
********

Once a model and suitable seed pixels are fitted, the SAMFire object is created
as follows:

.. code-block:: python

    >>> samf = m.create_samfire(workers=None, ipyparallel=False)

By default SAMFire will look for an ipyparallel cluster for the workers, and
will look for it for around 30 seconds. If none is available, it's recommended
to tell it explicitly via the ``ipyparallel=False`` argument, to use the
fall-back option of `multiprocessing`.

By default a new SAMFire object already has two (and currently only) strategies
added to its strategly list:

.. code-block:: python

    >>> samf.strategies
      A |    # | Strategy
     -- | ---- | -------------------------
      x |    0 | Reduced chi squared strategy
        |    1 | Histogram global strategy

The currently active strategy is marked by an 'x' in the first column.

If a new datapoint (i.e. pixel) is added manually, the "database" of the
currently active strategy has to be refreshed using the
:py:meth:`~.samfire.Samfire.refresh_database` call.

The current strategy "database" can be plotted using the
:py:meth:`~.samfire.Samfire.plot` method.

.. warning::
    It is a known bug that starting SAMFire after closing the previously
    plotted plot window crashes. In such cases please run ``samf._figure=None``
    before startng the samfire analysis.

Whilst the SAMFire is running, each pixel is checked by a ``goodness_test``,
which is by default :py:class:`~.fit_tests.red_chisq_test`, checking the
reduced chi-squared to be in the bounds of [0, 2]. 

This tolerance can (and most likely should!) be changed appropriatelly for the
data as follows:

.. code-block:: python

    >>> samf.metadata.goodness_test.tolerance = 0.3 # use a sensible value

The SAMFire calculations can be started using the
:py:meth:`~.samfire.Samfire.start` method. All keyword arguments are passed to
the underlying (i.e. usual) :py:meth:`~.model.BaseModel.fit` call:

.. code-block:: python

    >>> samf.start(fitter='mpfit', bounded=True)


