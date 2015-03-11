
.. _visualization-label:


Data visualization
******************

The object returned by :py:func:`~.io.load`, a :py:class:`~.signal.Signal`
instance, has a :py:meth:`~.signal.Signal.plot` method that is powerful and
flexible tools to visualize n-dimensional data. In this chapter, the
visualisation of multidimensional data  is exemplified with two experimental
datasets: an EELS spectrum image and an EDX dataset consisting of a secondary
electron emission image stack and a 3D hyperspectrum , both simoultaneously
acquired by recording two signals in parallel in a FIB/SEM.


.. code-block:: python
    
    >>> s = load('YourDataFilenameHere')
    >>> s.plot()

if the object is single spectrum or an image one window will appear when
calling the plot method.

Multidimensional spectral data
==============================

If the object is a 1D or 2D spectrum-image (i.e. with 2 or 3 dimensions when
including energy) two figures will appear, one containing a plot of the
spectrum at the current coordinates and the other an image of the data summed
over its spectral dimension if 2D or an image with the spectral dimension in
the x-axis if 1D:

.. _2d_SI:

.. figure::  images/2D_SI.png
   :align:   center
   :width:   500

   Visualisation of a 2D spectrum image
   
.. _1d_SI:

.. figure::  images/1D_SI.png
   :align:   center
   :width:   500

   Visualisation of a 1D spectrum image

To change the current coordinates, click on the pointer (which will be a line
or a square depending on the dimensions of the data) and drag it around. It is
also possible to move the pointer by using the numpad arrows **when numlock is
on and the spectrum or navigator figure is selected**.When using the numpad
arrows the PageUp and PageDown keys change the size of the step.

An extra cursor can be added by pressing the ``e`` key. Pressing ``e`` once
more will disable the extra cursor:

.. _second_pointer.png:

.. figure::  images/second_pointer.png
   :align:   center
   :width:   500

   Visualisation of a 2D spectrum image using two pointers.

Sometimes the default size of the rectangular cursors used to navigate images
can be too small to be dragged or even seen. It
is possible to change the size of the cursors by pressing the ``+`` and ``-``
keys  **when the navigator window is selected**.

=========   =============================
key         function    
=========   =============================
e           Switch second pointer on/off
Arrows      Change coordinates  
PageUp      Increase step size
PageDown    Decrease step size
``+``           Increase pointer size
``-``           Decrease pointer size
``h``       Launch the contrast adjustment tool (only for Image)
=========   =============================

To close all the figures run the following command:

.. code-block:: python

    close('all')

.. NOTE::

    This is a `matplotlib <http://matplotlib.sourceforge.net/>`_ command.
    Matplotlib is the library that hyperspy uses to produce the plots. You can
    learn how to pan/zoom and more  `in the matplotlib documentation
    <http://matplotlib.sourceforge.net/users/navigation_toolbar.html>`_

Multidimensional image data
===========================

Equivalently, if the object is a 1D or 2D image stack two figures will appear, 
one containing a plot of the image at the current coordinates and the other
a spectrum or an image obtained by summing over the image dimensions:
   
.. _1D_image_stack.png:

.. figure::  images/1D_image_stack.png
   :align:   center
   :width:   500    

   Visualisation of a 1D image stack
   
.. _2D_image_stack.png:

.. figure::  images/2D_image_stack.png
   :align:   center
   :width:   500
   
   Visualisation of a 2D image stack


The same keys can be used to explore an image stack.

Customizing the "navigator"
===========================

Stack of 2D images can be imported as an 3D image and plotted with a slider
instead of the 2D navigator as in the previous example.

.. code-block:: python

    >>> img = load('image*.tif', stack=True)
    >>> img.plot(navigator="slider")
    
    
.. figure::  images/3D_image.png
   :align:   center
   :width:   500    

   Visualisation of a 3D image with a slider.   
   

A stack of 2D spectrum images can be imported as a 3D spectrum image and
plotted with sliders.

.. code-block:: python

    >>> spec = load('spectrum_image*.rpl', stack=True)
    >>> spec.plot()
    
    
.. figure::  images/3D_spectrum.png
   :align:   center
   :width:   650    

   Visualisation of a 3D spectrum image with sliders.
   
If the 3D images has the same spatial dimension as the 3D spectrum image, it
can be used as an external signal for the navigator.
   
   
.. code-block:: python

    >>> spec.plot(navigator=img)    
    
.. figure::  images/3D_spectrum_external.png
   :align:   center
   :width:   650    

   Visualisation of a 3D spectrum image. The navigator is an external signal.
   
The 3D spectrum image can be transformed in a stack of spectral images for an
alternative display.

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
=================================

Although HyperSpy does not currently support plotting when signal_dimension is
greater than 2, `Mayavi <http://docs.enthought.com/mayavi/mayavi/>`_ can be
used for this purpose.

In the following example we also use `scikit-image <http://scikit-image.org/>`_
for noise reduction. More details about 
:py:meth:`~._signals.eds.EDSSpectrum.get_lines_intensity` method can be 
found in :ref:`EDS lines intensity<get_lines_intensity>`.

.. code-block:: python

    >>> #Import packages
    >>> from skimage import filter
    >>> from mayavi import mlab
    >>> #Generate the X-ray intensity map of Nickel L alpha
    >>> NiMap = specImg3Dc.get_lines_intensity(['Ni_La'])[0]
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

.. _plot_spectra:

Plotting multiple signals
=========================

HyperSpy provides three functions to plot multiple signals (spectra, images or
other signals): :py:func:`~.drawing.utils.plot_images`, :py:func:`~.drawing.utils.plot_spectra`, and
:py:func:`~.drawing.utils.plot_signals` in the ``utils.plot`` package.

.. _plot.images:

Plotting several images
-----------------------

.. versionadded:: 0.8

:py:func:`~.drawing.utils.plot_images` is used to plot several images in the
same figure. It supports many configurations and has many options available
to customize the resulting output. The function returns a list of
`matplotlib axes <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.axes>`_, which
can be used to further customize the figure. Some examples are given below.

A common usage for :py:func:`~.drawing.utils.plot_images` is to view the
different slices of a multidimensional image (a *hyperimage*):

 .. code-block:: python

    >>> im1_3 = signals.Image(np.random.random((2, 3, 64, 64)))
    >>> im1_3.metadata.General.title = 'multi-dimensional image'
    >>> utils.plot.plot_images(im1_3)

.. figure::  images/plot_images_defaults.png
  :align:   center
  :width:   500

By default, :py:func:`~.drawing.utils.plot_images` will attempt to auto-label the images
based on the Signal titles. The labels (and title) can be customized with the `suptitle` and `label` arguments.
In this example, the axes labels are also disabled with `axes_decor` so only ticks are shown:

 .. code-block:: python

    >>> im1_3 = signals.Image(np.random.random((2, 3, 64, 64)))
    >>> im1_3.metadata.General.title = 'multi-dimensional image'
    >>> utils.plot.plot_images(im1_3, suptitle='Custom figure title',
    ...                        label=['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5', 'Image 6'],
    ...                        axes_decor='ticks')

.. figure::  images/plot_images_custom-labels.png
  :align:   center
  :width:   500

:py:func:`~.drawing.utils.plot_images` can also be used to easily plot a list of `Images`, comparing
different `Signals`, including RGB images (example below available :download:`here <images/plot_images_rgb1.png>`).
This example also demonstrates how to wrap labels using `labelwrap` (for preventing overlap) and using a single
`colorbar` for all the Images, as opposed to multiple individual ones:

 .. code-block:: python

    >>> im0 = signals.Image(np.random.random((64,64)))
    >>> im1_3 = signals.Image(np.random.random((2, 3, 64, 64)))
    >>> im4 = signals.Image(np.random.random((64,32)))
    >>> rgb = load("plot_images_rgb1.png")
    >>> im0.metadata.General.title = 'im0'
    >>> im0.axes_manager[0].units = 'nm'
    >>> im0.axes_manager[1].units = 'nm'
    >>> im1_3.metadata.General.title = 'multi-dimensional image'
    >>> im4.metadata.General.title = 'im4'
    >>> rgb.metadata.General.title = 'RGB'

    >>> utils.plot.plot_images([im0, im1_3, im4, rgb], colorbar='single', labelwrap=20)

.. figure::  images/plot_images_image-list.png
  :align:   center
  :width:   500

Another example for this function is plotting EDS line intensities. Using a spectrum image with
EDS data (:download:`download <images/plot_images_eds.hdf5>`), one can use the following commands
to get a representative figure of the line intensities. This example also demonstrates changing the colormap (with `cmap`),
adding scalebars to the plots (with `scalebar`), and changing the `padding` between the images. The padding is specified as
a dictionary, which is used to call :py:func:`matplotlib.figure.Figure.subplots_adjust`
(see `documentation <http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.subplots_adjust>`_).

.. |subplots_adjust| image:: images/plot_images_subplots.png

*Note, this padding can also be changed interactively by clicking on the* |subplots_adjust|
*button in the GUI (button may be different when using different graphical backends).*

 .. code-block:: python

    >>> si_EDS = load("plot_images_eds.hdf5")
    >>> im = si_EDS.get_lines_intensity()
    >>> utils.plot.plot_images(im, per_row=2, tight_layout=True, axes_decor='off',
    ...                        suptitle_fontsize=20, colorbar='single', suptitle='EDS Line intensity',
    ...                        label=['La M$\\alpha$', 'Mn K$\\alpha$', 'Zr L$\\alpha$'],cmap='cubehelix',
    ...                        scalebar='all', scalebar_color='white',
    ...                        padding={'top':0.85,'bottom':0.10,'left':0.05,'right':0.85,'wspace':0.10,'hspace':0.10})

.. figure::  images/plot_images_eds.png
  :align:   center
  :width:   500

.. _plot.spectra:

Plotting several spectra
------------------------

.. versionadded:: 0.7

:py:func:`~.drawing.utils.plot_spectra` is used to plot several spectra in the
same figure. It supports different styles, the default
being "overlap". The default style is configurable in :ref:`preferences
<configuring-hyperspy-label>`.

In the following example we create a list of 9 single spectra (gaussian
functions with different sigma values) and plot them in the same figure using
:py:func:`~.drawing.utils.plot_spectra`. Note that, in this case, the legend
labels are taken from the individual spectrum titles. By clicking on the
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
     ...         gs.metadata.General.title = "sigma=%i" % sigma
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

For the "heatmap" style, different `matplotlib color schemes <http://matplotlib.org/examples/color/colormaps_reference.html>`_ can be used:

.. code-block:: python

    >>> import matplotlib.cm
    >>> ax = utils.plot.plot_spectra(s, style="heatmap")
    >>> ax.images[0].set_cmap(matplotlib.cm.jet)

.. figure::  images/plot_spectra_heatmap_jet.png
  :align:   center
  :width:   500

Any parameter that can be passed to matplotlib.pyplot.figure can also be used with plot_spectra()
to allow further customization  (when using the "overlap", "cascade", or "mosaic" styles).
In the following example, `dpi`, `facecolor`, `frameon`, and `num` are all parameters
that are passed directly to matplotlib.pyplot.figure as keyword arguments:

.. code-block:: python

    >>> s = signals.Spectrum(np.random.random((6,1000)))
    >>> legendtext = ['Plot 0', 'Plot 1', 'Plot 2', 'Plot 3', 'Plot 4', 'Plot 5']
    >>> cascade_plot = utils.plot.plot_spectra(s, style='cascade', legend=legendtext, dpi=60, facecolor='lightblue', frameon=True, num=5)
    >>> cascade_plot.set_xlabel("X-axis")
    >>> cascade_plot.set_ylabel("Y-axis")
    >>> cascade_plot.set_title("Cascade plot")
    >>> plt.draw()

.. figure:: images/plot_spectra_kwargs.png
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
  
A matplotlib ax and fig object can also be specified, which can be used to put several
subplots in the same figure. This will only work for "cascade" and "overlap" styles:

.. code-block:: python

    >>> fig, axarr = plt.subplots(1,2)
    >>> s1 = signals.Spectrum(np.random.random((6,1000)))
    >>> s2 = signals.Spectrum(np.random.random((6,1000)))
    >>> utils.plot.plot_spectra(s1, style='cascade',color='blue',ax=axarr[0],fig=fig)
    >>> utils.plot.plot_spectra(s2, style='cascade',color='red',ax=axarr[1],fig=fig)
    >>> fig.canvas.draw()

.. figure::  images/plot_spectra_ax_argument.png
  :align:   center
  :width:   500

.. _plot.signals:

Plotting several signals
^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 0.7
:py:func:`~.drawing.utils.plot_signals` is used to plot several signals at the
same time. By default the navigation position of the signals will be synced, and the 
signals must have the same dimensions. To plot two spectra at the same time: 

.. code-block:: python

    >>> s1 = signals.Spectrum(np.random.random((10,10,100))) 
    >>> s2 = signals.Spectrum(np.random.random((10,10,100)))
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

    >>> s1 = signals.Spectrum(np.random.random((10,10,100))) 
    >>> s2 = signals.Spectrum(np.random.random((10,10,100))) 
    >>> utils.plot.plot_signals([s1, s2], navigator="slider")

.. figure::  images/plot_signals_slider.png
  :align:   center
  :width:   500    

Navigators can also be set differently for different plots using the 
navigator_list argument. Where the navigator_list be the same length
as the number of signals plotted, and only contain valid navigator options.
For example:

.. code-block:: python

    >>> s1 = signals.Spectrum(np.random.random((10,10,100))) 
    >>> s2 = signals.Spectrum(np.random.random((10,10,100))) 
    >>> s3 = signals.Spectrum(np.random.random((10,10))) 
    >>> utils.plot.plot_signals([s1, s2], navigator_list=["slider", s3])

.. figure::  images/plot_signals_navigator_list.png
  :align:   center
  :width:   500    

Several signals can also be plotted without syncing the navigation by using
sync=False. The navigator_list can still be used to specify a navigator for 
each plot:

.. code-block:: python

    >>> s1 = signals.Spectrum(np.random.random((10,10,100))) 
    >>> s2 = signals.Spectrum(np.random.random((10,10,100))) 
    >>> utils.plot.plot_signals([s1, s2], sync=False, navigator_list=["slider", "slider"])

.. figure::  images/plot_signals_sync.png
  :align:   center
  :width:   500    


