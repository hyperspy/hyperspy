# -*- coding: utf-8 -*-
"""Common docstring snippets for plot.

"""


BASE_PLOT_DOCSTRING_PARAMETERS = \
    """navigator : str, None, or :py:class:`~hyperspy.signal.BaseSignal` (or subclass)
        Allowed string values are ``'auto'``, ``'slider'``, and ``'spectrum'``.

        If ``'auto'``:

            - If `navigation_dimension` > 0, a navigator is
              provided to explore the data.
            - If `navigation_dimension` is 1 and the signal is an image
              the navigator is a sum spectrum obtained by integrating
              over the signal axes (the image).
            - If `navigation_dimension` is 1 and the signal is a spectrum
              the navigator is an image obtained by stacking all the 
              spectra in the dataset horizontally.
            - If `navigation_dimension` is > 1, the navigator is a sum 
              image obtained by integrating the data over the signal axes.
            - Additionally, if `navigation_dimension` > 2, a window
              with one slider per axis is raised to navigate the data.
            - For example, if the dataset consists of 3 navigation axes `X`, 
              `Y`, `Z` and one signal axis, `E`, the default navigator will 
              be an image obtained by integrating the data over `E` at the 
              current `Z` index and a window with sliders for the `X`, `Y`, 
              and `Z` axes will be raised. Notice that changing the `Z`-axis 
              index changes the navigator in this case.

        If ``'slider'``:
        
            - If `navigation dimension` > 0 a window with one slider per 
              axis is raised to navigate the data.

        If ``'spectrum'``:
        
            - If `navigation_dimension` > 0 the navigator is always a 
              spectrum obtained by integrating the data over all other axes.

        If ``None``, no navigator will be provided.

        Alternatively a :py:class:`~hyperspy.signal.BaseSignal` (or subclass) 
        instance can be provided. The `signal_dimension` must be 1 (for a 
        spectrum navigator) or 2 (for a image navigator) and 
        `navigation_shape` must be 0 (for a static navigator) or 
        `navigation_shape` + `signal_shape` must be equal to the 
        `navigator_shape` of the current object (for a dynamic navigator).
        If the signal `dtype` is RGB or RGBA this parameter has no effect and 
        the value is always set to ``'slider'``.
    axes_manager : None or :py:class:`~hyperspy.axes.AxesManager`
        If None, the signal's `axes_manager` attribute is used.
    plot_markers : bool, default True
        Plot markers added using s.add_marker(marker, permanent=True).
        Note, a large number of markers might lead to very slow plotting.
    norm : str or :py:class:`matplotlib.colors.Normalize` 
        The function used to normalize the data prior to plotting.
        Allowable strings are: ``'auto'``, ``'linear'``, or ``'log'`` 
        (default value is ``'auto'``).
        If ``'auto'``, intensity is plotted on a linear scale except when
        the Signal's `power_spectrum` is ``True``, which can be used only for 
        compatible signals.  
        Alternatively, for a :py:class:`~hyperspy._signals.signal2d.Signal2D` 
        object, an instance of the :py:class:`~matplotlib.colors.Normalize` 
        class can be used to customize the normalization function.
    """


BASE_PLOT_DOCSTRING = \
    """Plot the signal at the current coordinates.

    For multidimensional datasets an optional figure,
    the "navigator", with a cursor to navigate that data is
    raised. In any case it is possible to navigate the data using
    the sliders. Currently only signals with signal_dimension equal to
    0, 1 and 2 can be plotted.

    Parameters
    ----------
    %s""" % BASE_PLOT_DOCSTRING_PARAMETERS


PLOT2D_DOCSTRING = \
    """colorbar : bool, optional
            If true, a colorbar is plotted for non-RGB images.
        scalebar : bool, optional
            If True and the units and scale of the x and y axes are the same a
            scale bar is plotted.
        scalebar_color : str, optional
            A valid MPL color string; will be used as the scalebar color.
        axes_ticks : {None, bool}, optional
            If True, plot the axes ticks. If None axes_ticks are only
            plotted when the scale bar is not plotted. If False the axes ticks
            are never plotted.
        saturated_pixels: scalar
            The percentage of pixels that are left out of the bounds.
            For example, the low and high bounds of a value of 1 are the 0.5%
            and 99.5% percentiles. It must be in the [0, 100] range.
        vmin, vmax : scalar, optional
            `vmin` and `vmax` are used to normalize the intensity scale.
        no_nans : bool, optional
            If True, set nans to zero for plotting.
        centre_colormap : {"auto", True, False}
            If True the centre of the color scheme is set to zero. This is
            specially useful when using diverging color schemes. If "auto"
            (default), diverging color schemes are automatically centred.
        min_aspect : float
            Set the minimum aspect ratio of the image and the figure. To
            keep the image in the aspect limit the pixels are made
            rectangular.
        gamma : float
            Value used for the gamma adjustement and not compatible norm='log'.
            See ``matplotlib.colors.PowerNorm`` for more information.
            Default is 1.0 (linear scale).
        linthresh : float, optional
            Only used with norm='log' and negative values: Range of value
            closed to zero, which are linearly extrapolated.
            See the ``matplotlib.colors.SymLogNorm`` for more information.
            Default is 0.01.
        linscale : float, optional
            Only used with norm='log' and negative values: Number of decades to
            use for each half of the linear range.
            See the ``matplotlib.colors.SymLogNorm`` for more information.
            Default is 0.1."""


COMPLEX_DOCSTRING = \
    """power_spectrum : bool, default is False.
            If True, plot the power spectrum instead of the actual signal, if
            False, plot the real and imaginary parts of the complex signal.
        representation : {'cartesian' or 'polar'}
            Determines if the real and imaginary part of the complex data is plotted ('cartesian',
            default), or if the amplitude and phase should be used ('polar').
        same_axes : bool, default True
            If True (default) plot the real and
            imaginary parts (or amplitude and phase) in the same figure if
            the signal is one-dimensional.
        fft_shift : bool, default False
            If True, shift the zero-frequency component.
            See `numpy.fft.fftshift` for more details.
        """


KWARGS_DOCSTRING = \
    """**kwargs
            Only for :py:class:`~hyperspy._signals.signal2d.Signal2D`: 
            additional (optional) keyword arguments for 
            :py:func:`matplotlib.pyplot.imshow`."""
