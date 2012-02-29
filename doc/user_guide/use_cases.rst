Use cases
*********

Machine learning application to the analysis of image stacks
============================================================

.. WARNING::
    Some features mentioned in this section are no longer maintained since
    their author (and the author of this section), Michael Sarahan,
    left the project. Therefore, it is possible that they no longer work as 
    expected. If no one maintains the associated code it can be deprecated 
    in future versions of Hyperspy.

Types of data
-------------

The "data" we'll be addressing here will consist of several images of
some highly repetitive structure with some slight variations in the
structure from image to image.  For example, faces.  Or, more along
the lines of the author's research, images of a repetitive (though
slightly varying) atomic structure.

All images must be the same shape, same scale, and must be registered
with one another.  The idea is that any pixel in one image should be
directly comparable to the corresponding pixel in any other image.

Goals of analysis
-----------------

What this analysis can provide you is one of several things:

* Factor images, which give you visual insight into correlated regions
  of change in your data
* Peak locations and other characteristics (height, shape,
  orientation)
* Factors of peak characteristics, showing correlated changes in
  heights, positions, shapes, and orientations.
* Classifications of several distinct structural variations into
  categories

Getting Started
---------------

The best way to start is to pick out several images you've taken under
identical experimental conditions.  For microscopes, make sure the
focus on all your images is roughly similar, and that you haven't
changed any apertures or anything else that would change what the
image contrast means.

Load your image(s) into a hyperspy object:

.. code-block:: python

    d=load('filename.dm3')
    # or, for many files:
    d=load('*.dm3')

Cropping cells
--------------

To obtain the similar images that go into the analysis, we crop small
sub-images from your original images.  Thanks to Mike getting sick of
the tedium of doing this by hand, there's an interactive tool for
cropping cells from one or more images.

.. code-block:: python

    agg=d.cell_cropper()

This dialog serves two primary functions:

* Picking your template

  * Drag the cursor indicated by the intersection of the white lines
    overlaid on your image.

  * You can also adjust the top or left coordinates from the dialog in
    the upper right.

  * Adjust the size of the template by entering a value in the
    corresponding box.

* Locating cells that match the template on the image

  * Cross-correlation is automatically calculated on your image
    whenever the template changes.  You can view the cross-correlation
    map by checking the box below the image.

  * Matches to the template are located by finding peaks on the cross
    correlation map.  This is slow, so click the "Find Peaks" button
    and be patient.

  * Once peaks have been found, you can adjust the threshold to
    specify exactly which peaks should be cropped.  Right-click on the
    colorbar, and drag to select a threshold area.  Note that the
    transparency of the overlaid spots on the image changes as you
    change the threshold.  Opaque spots will be cropped, while
    transparent spots will not.

When you have found spots to your satisfaction, click OK (you won't be
able to click OK until you have at least some spots).  After clicking
OK, the agg object now contains all of your cropped cells.

Analysis in Image space
-----------------------

The aggregate object we've created functions just like any other
normal data object.  To run multivariate analysis on the stack of
cropped images,

.. code-block:: python

    agg.principal_components_analysis(True)

To view your factor images, use the **plotPca_factors** method:

.. code-block:: python

    agg.plotPca_factors(2)

Why ICA works for images
------------------------
For several real-world situations, the distributions of components
that make up the system are not Gaussian.  This means that PCA will
never work well for finding the original components - it can only
derive Gaussian components.  Independent component analysis, on the
other hand, derives components by intentionally maximizing the
non-Gaussianity of components.  The idea is encapsulated by the
`Central Limit Theorem
<http://en.wikipedia.org/wiki/Central_limit_theorem>`_, 
which states that the sum of any number of independent components 
will converge towards the Gaussian distribution.

Characterizing image peaks
--------------------------

To make the interpretation of factors more straightforward, you can
derive characteristics for image peaks, and run multivariate analysis
on the characteristics, rather than the image data.  Assuming that
your characteristics properly parameterize your peaks (i.e. the
parameterized peak does not lose important information), this
description of your data is equivalent to the image.

In hyperspy, you can quickly and easily characterize all the peaks in
a stack of images with the **peak_char_stack** function.  Its one
required parameter is the peak_width, which you should estimate
visually from your images.

.. code-block:: python

    agg.peak_char_stack(10)

Meaning of peak characteristics
-------------------------------

Peak characteristics currently come in sets of 7:

 #. 0 and 1: the peak position

 #. 2 and 3: the difference between peak position and target position

 #. 4: the peak height (or change in peak height for a component)

 #. 5: the peak orientation (or change in peak orientation for a
   component).  Orientation ranges from -pi/2 to pi/2 radians.
   
 #. 6: the peak eccentricity (or change in peak eccentricity for a
   component).  Eccentricity ranges from 0 to 1, with 0 being a
   perfectly round object, and 1 being a line.

Analysis in Peak space
----------------------

Running multivariate analysis on peak characteristics is similar to
analysis on image data, with the exception that you pass the on_peaks
parameter as True.

.. code-block:: python

    agg.principal_component_analysis(on_peaks=True)

Similarly, plotting functions also take the on_peaks parameter.  They
also offer a few more options to control exactly which of the peak
parameters are plotted.

.. code-block:: python

    # default view for peak plotting is to plot shifts,
    # magnified by a factor of 100, along with a scatter map
    # of peak heights.
    agg.plotPca_factors(3,on_peaks=True)
    # disable the vector plot of peak shifts
    agg.plotPca_factors(3,on_peaks=True,plot_shifts=False)
    # plot peak orientation instead of peak height
    agg.plotPca_factors(3,on_peaks=True,plot_char=5)
    
