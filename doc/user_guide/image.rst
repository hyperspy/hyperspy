Image Processing
============

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

Why ICA works for images
------------------------

Characterizing image peaks
--------------------------

Analysis in Peak space
----------------------

