Visualisation of multi-dimensional data
****************************************

With the aim to ease the data analysis of multidimensionnal data, Hyperspy provides
a powerful and flexible :py:meth:`~.signal.Signal.plot` method to visualize 
n-dimensional data. In this chapter, the visualisation of data of 3 or more dimensions is exemplified with a image stack and a 4D hyperspectrum obtained by recording two signals in
parallele in a FIB/SEM: the intensity of the secondary electron emission (SE image) and the X-ray spectrum (EDS map).

The plot() function
-------------------

The visualisation with :py:meth:`~.signal.Signal.plot` of 1D and 2D signal
is given in :ref:`getting started <getting-help-label>`. Further options
are shown here.

Stack of images
^^^^^^^^^^^^^^^

Stack of 2D images can be imported as an 3D image and plotted with a slider.

.. code-block:: python

    >>> img = load('image*.tif', stack=True)
    >>> img.plot(navigator=None)
    
    
.. figure::  images/3D_image.png
   :align:   center
   :width:   500    

   Visualisation of a 3D image with a slider.   
   
Spectrum images
^^^^^^^^^^^^^^

A stack of 2D spectrum images can be imported as a 3D spectrum image and plotted 
with sliders.

.. code-block:: python

    >>> spec = load('spectrum_image*.tif', stack=True)
    >>> spec.plot()
    
    
.. figure::  images/3D_spectrum.png
   :align:   center
   :width:   650    

   Visualisation of a 3D spectrum image with sliders.
   
If the 3D images has the same spatial dimension as the 3D spectrum image,
it can be used as an external signal for the navigator.
   
   
.. code-block:: python

    >>> spec.plot(navigator=img)    
    
.. figure::  images/3D_spectrum_external.png
   :align:   center
   :width:   650    

   Visualisation of a 3D spectrum image. The navigator is an external signal.
   
Stack of spectral images
^^^^^^^^^^^^^^^^^^^^^^^^

The 3D spectrum image can be transformed in a stack of spectral images for 
an alternative display.

.. code-block:: python

    >>> imgSpec = spec.to_image((0, 1))
    >>> imgSpec.plot(navigator='spectrum')
    
    
.. figure::  images/3D_image_spectrum.png
   :align:   center
   :width:   650    

   Visualisation of a stack of 2D spectral images.
   
An external signal (e.g. a spectrum) can be used as a navigator, for example 
the "maximum spectrum" for which each channel is the maximum of all pixels. 

.. code-block:: python

    >>> specMax = spec.max(0).max(0).max(0)
    >>> imgSpec.plot(navigator=specMax)
    
    
.. figure::  images/3D_image_spectrum_external.png
   :align:   center
   :width:   650    

   Visualisation of a stack of 2D spectral images. 
   The navigator is the "maximum spectrum".

Using Mayavi to visualize 3D data
---------------------------------

Although Hyperspy does not currently support plotting when signal_dimension 
is greater than 2, `Mayavi <http://docs.enthought.com/mayavi/mayavi/>`_ 
can be used for this purpose.

In the following example we also use `scikit-image <http://scikit-image.org/>`_ for noise reduction: 

.. code-block:: python

    >>> #Import packages
    >>> from skimage import filter
    >>> from mayavi import mlab
    >>> #Generate the X-ray intensity map of Nickel L alpha
    >>> NiMap = specImg3Dc.get_intensity_map(['Ni_La'])[0]
    >>> #Reduce the noise
    >>> NiMapDenoise = filter.tv_denoise(NiMap.data)
    >>> #Plot isosurfaces
    >>> mlab.contour3d(NiMapDenoise)
    >>> mlab.outline()
        
    
.. figure::  images/mayavi.png
   :align:   center
   :width:   450    

   Visualisation of isosurfaces with mayavi.
   
.. NOTE::

    The sample and the data used in this chapter are described in 
    P. Burdet, `et al.`, Acta Materialia, 61, p. 3090-3098 (2013) (see
    `abstract <http://infoscience.epfl.ch/record/185861/>`_).
