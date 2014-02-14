
.. _visualization-label:

Visualisation of multi-dimensional data
****************************************

.. navigator_options:

Navigator options of the plot() function
----------------------------------------

With the aim to ease the data analysis of multidimensionnal data, Hyperspy provides
a powerful and flexible :py:meth:`~.signal.Signal.plot` method to visualize 
n-dimensional data. In this chapter, the visualisation of data of 3 or 
more dimensions is exemplified with a image stack and a 4D hyperspectrum
obtained by recording two signals in parallele in a FIB/SEM: the intensity
of the secondary electron emission (SE image) and the X-ray spectrum (EDS map).

The visualisation with :py:meth:`~.signal.Signal.plot` of 1D and 2D signal
is given in :ref:`getting started <getting-help-label>`. The options of
the navigator are shown here.

Stack of images
^^^^^^^^^^^^^^^

Stack of 2D images can be imported as an 3D image and plotted with a slider.

.. code-block:: python

    >>> img = load('image*.tif', stack=True)
    >>> img.plot(navigator="slider")
    
    
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

    >>> imgSpec = spec.as_image((0, 1))
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
   
Lastly, if no navigator is needed, "navigator=None" can be used.

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
    >>> NiMapDenoise = filter.denoise_tv_chambolle(NiMap.data)
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

Comparing objects
-----------------

Hyperspy provides two functions to compare different objects (spectra, images or
other signals) whatever their dimension. The two functions, 
:py:func:`~.drawing.utils.plot_spectra` and :py:func:`~.drawing.utils.plot_signals`
, are explained and exemplified in this chapter. 

Plotting several spectra
^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 0.7

:py:func:`~.drawing.utils.plot_spectra` is used to plot several spectra in the
same figure. It supports different styles, the default
being "overlap". The default style is configurable in :ref:`preferences
<configuring-hyperspy-label>`.

In the following example we create a list of 9 single spectra (gaussian
functions with different sigma values) and plot them in the same figure using
:py:func:`~.drawing.utils.plot_spectra`. Note that, in this case, the legend
labels are taken from the indivual spectrum titles. By clicking on the 
legended line, a spectrum can be toggled on and off.

 .. code-block:: python

     >>> s = signals.Spectrum(np.zeros((200)))
     >>> s.axes_manager[0].offset = -10
     >>> s.axes_manager[0].scale = 0.1
     >>> m = create_model(s)
     >>> g = components.Gaussian()
     >>> m.append(g)
     >>> gaussians = []
     >>> labels = []
     >>> 
     >>> for sigma in range(1, 10):
     ...         g.sigma.value = sigma
     ...         gs = m.as_signal()
     ...         gs.mapped_parameters.title = "sigma=%i" % sigma
     ...         gaussians.append(gs)
     ...         
     >>> utils.plot.plot_spectra(gaussians,legend='auto')
     <matplotlib.axes.AxesSubplot object at 0x4c28c90>


.. figure::  images/plot_spectra_overlap.png
  :align:   center
  :width:   500 
  

Another style, "cascade", can be useful when "overlap" results in a plot that
is too cluttered e.g. to visualize 
changes in EELS fine structure over a line scan. The following example 
shows how to plot a cascade style figure from a spectrum, and save it in 
a file:

.. code-block:: python

    >>> s = signals.Spectrum(np.random.random((6,1000)))
    >>> cascade_plot = utils.plot.plot_spectra(s, style='cascade')
    >>> cascade_plot.figure.savefig("cascade_plot.png")

.. figure::  images/plot_spectra_cascade.png
  :align:   center
  :width:   500    

The "cascade" `style` has a `padding` option. The default value, 1, keeps the 
individual plots from overlapping. However in most cases a lower 
padding value can be used, to get tighter plots.

Using the color argument one can assign a color to all the spectra, or specific
colors for each spectrum. In the same way, one can also assign the line style
and provide the legend labels:

.. code-block:: python

    >>> color_list = ['red', 'red', 'blue', 'blue', 'red', 'red']
    >>> line_style_list = ['-','--','steps','-.',':','-']
    >>> utils.plot.plot_spectra(s, style='cascade', color=color_list,
    >>> line_style=line_style_list,legend='auto')

.. figure::  images/plot_spectra_color.png
  :align:   center
  :width:   500    

There are also two other styles, "heatmap" and "mosaic":

.. code-block:: python

    >>> utils.plot.plot_spectra(s, style='heatmap')

.. figure::  images/plot_spectra_heatmap.png
  :align:   center
  :width:   500    

.. code-block:: python

    >>> s = signals.Spectrum(np.random.random((2,1000)))
    >>> utils.plot.plot_spectra(s, style='mosaic')
    
.. figure::  images/plot_spectra_mosaic.png
  :align:   center
  :width:   500    

The function returns a matplotlib ax object, which can be used to customize the figure:

.. code-block:: python

    >>> s = signals.Spectrum(np.random.random((6,1000)))
    >>> cascade_plot = utils.plot.plot_spectra(s)
    >>> cascade_plot.set_xlabel("An axis")
    >>> cascade_plot.set_ylabel("Another axis")
    >>> cascade_plot.set_title("A title!")
    >>> plt.draw()

.. figure::  images/plot_spectra_customize.png
  :align:   center
  :width:   500    

Plotting several signals
^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 0.7
:py:func:`~.drawing.utils.plot_signals` is used to plot several signals at the
same time. By default the navigation position of the signals will be synced, and the 
signals must have the same dimensions. To plot two spectra at the same time: 

.. code-block:: python

    >>> s1 = signals.Spectrum(np.random.random(10,10,100)) 
    >>> s2 = signals.Spectrum(np.random.random(10,10,100)) 
    >>> utils.plot.plot_signals([s1, s2])

.. figure::  images/plot_signals.png
  :align:   center
  :width:   500    

The navigator can be specified by using the navigator argument, where the 
different options are "auto", None, "spectrum", "slider" or Signal.  
For more details about the different navigators, 
see :ref:`navigator_options`.
To specify the navigator:

.. code-block:: python

    >>> s1 = signals.Spectrum(np.random.random(10,10,100)) 
    >>> s2 = signals.Spectrum(np.random.random(10,10,100)) 
    >>> utils.plot.plot_signals([s1, s2], navigator="slider")

.. figure::  images/plot_signals_slider.png
  :align:   center
  :width:   500    

Navigators can also be set differently for different plots using the 
navigator_list argument. Where the navigator_list be the same length
as the number of signals plotted, and only contain valid navigator options.
For example:

.. code-block:: python

    >>> s1 = signals.Spectrum(np.random.random(10,10,100)) 
    >>> s2 = signals.Spectrum(np.random.random(10,10,100)) 
    >>> s3 = signals.Spectrum(np.random.random(10,10)) 
    >>> utils.plot.plot_signals([s1, s2], navigator_list=["slider", s3])

.. figure::  images/plot_signals_navigator_list.png
  :align:   center
  :width:   500    

Several signals can also be plotted without syncing the navigation by using
sync=False. The navigator_list can still be used to specify a navigator for 
each plot:

.. code-block:: python

    >>> s1 = signals.Spectrum(np.random.random(10,10,100)) 
    >>> s2 = signals.Spectrum(np.random.random(10,10,100)) 
    >>> utils.plot.plot_signals([s1, s2], sync=False, navigator_list=["slider", "slider"])

.. figure::  images/plot_signals_sync.png
  :align:   center
  :width:   500    


