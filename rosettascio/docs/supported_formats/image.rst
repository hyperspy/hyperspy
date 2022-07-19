.. _image-format:

Image formats
-------------

HyperSpy can read and write data to `all the image formats
<https://imageio.readthedocs.io/en/stable/formats.html>`_ supported by
`imageio <https://imageio.readthedocs.io/>`_, which uses the 
`Python Image Library (PIL/pillow) <https://pillow.readthedocs.io>`_.
This includes ``.jpg``, ``.gif``, ``.png``, ``.pdf``, ``.tif``, etc.
It is important to note that these image formats only support 8-bit files, and
therefore have an insufficient dynamic range for most scientific applications.
It is therefore highly discouraged to use any general image format (with the
exception of :ref:`tiff-format` which uses another library) to store data for
analysis purposes.

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``scalebar`` (bool, optional): Export the image with a scalebar. Default
  is False.
- ``scalebar_kwds`` (dict, optional): dictionary of keyword arguments for the
  scalebar. Useful to set formattiong, location, etc. of the scalebar. See the
  `matplotlib-scalebar <https://pypi.org/project/matplotlib-scalebar/>`_
  documentation for more information.
- ``output_size`` : (int, tuple of length 2 or None, optional): the output size
  of the image in pixels:

  * if ``int``, defines the width of the image, the height is
    determined from the aspect ratio of the image.
  * if ``tuple`` of length 2, defines the width and height of the
    image. Padding with white pixels is used to maintain the aspect
    ratio of the image.
  * if ``None``, the size of the data is used.

  For output sizes larger than the data size, "nearest" interpolation is
  used by default and this behaviour can be changed through the
  ``imshow_kwds`` dictionary.

- ``imshow_kwds`` (dict, optional):  Keyword arguments dictionary for
  :py:func:`~.matplotlib.pyplot.imshow`.
- ``**kwds`` : keyword arguments supported by the individual file
  writers as documented at
  https://imageio.readthedocs.io/en/stable/formats.html when exporting
  an image without scalebar. When exporting with a scalebar, the keyword
  arguments are passed to the `pil_kwargs` dictionary of
  :py:func:`matplotlib.pyplot.savefig`


When saving an image, a scalebar can be added to the image and the formatting,
location, etc. of the scalebar can be set using the ``scalebar_kwds``
arguments:

.. code-block:: python

    >>> s.save('file.jpg', scalebar=True)
    >>> s.save('file.jpg', scalebar=True, scalebar_kwds={'location':'lower right'})

In the example above, the image is created using
:py:func:`~.matplotlib.pyplot.imshow`, and additional keyword arguments can be
passed to this function using ``imshow_kwds``. For example, this can be used
to save an image displayed using a matplotlib colormap:

.. code-block:: python

    >>> s.save('file.jpg', imshow_kwds=dict(cmap='viridis'))


The resolution of the exported image can be adjusted:

.. code-block:: python

    >>> s.save('file.jpg', output_size=512)
