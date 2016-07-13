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


