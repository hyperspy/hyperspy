Visualisation of multi-dimensional data
****************************************

Hyperspy is built to process multi-dimensionnal data. In this chapter, 
the visualisation of signal of 3 or more dimensions is described. 
The example used here is an acquisition in 3D with two detectors. One 
detector records X-ray spectrum (EDS map), and the other detector 
records the intensity of secondary electron (SE image).

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
   
If the 3D images has the same spatial dimension than the 3D spectrum image,
it can be used as an external signals for the navigator.
   
   
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

    >>> imgSpec = spec.to_image(1)
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

External function
-----------------

3D plotting
^^^^^^^^^^^

Other python packages can be imported for further process or visualisation.
For example: 

* `mayavi <http://docs.enthought.com/mayavi/mayavi/>`_ for 3D plotting.
* `scikit-image <http://scikit-image.org/>`_ for image processing.

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
