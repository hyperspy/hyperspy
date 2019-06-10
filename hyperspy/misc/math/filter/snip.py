import numpy as np

def snip_method(spectrum,
                width=13, decrease_factor=np.sqrt(2),
                iterations=16):
    """
    use snip algorithm to obtain background

    Parameters
    ----------
    spectrum : array
        intensity spectrum or array 
    width : int, optional
        window size in indices. Typically twice the fwhm of peaks in spectrum 
    decrease_factor : float, optional
        gradually decrease of window size, default as sqrt(2)
    iterations : int, optional
        Number of iterations in snip filter


    Returns
    -------
    background : array
        output background curve

    References
    ----------

    .. [1] C.G. Ryan etc, "SNIP, a statistics-sensitive background
           treatment for the quantitative analysis of PIXE spectra in
           geoscience applications", Nuclear Instruments and Methods in
           Physics Research Section B, vol. 34, 1998.
    """

    background = np.array(spectrum)
    spectra_size = len(background)
    window_p = int(width)
    background = np.log(np.log(background + 1) + 1)
    index      = np.arange(spectra_size)
    
    # snip 
    # reduce the filter width at the edges as this will cause large drop-offs
    # otherwise
    lo_index = np.where(index>=window_p,index-window_p,0)
    hi_index = np.where(index>=window_p,index+window_p,index+index)
    hi_index = np.where(hi_index<spectra_size,hi_index,spectra_size-1)
    lo_index = np.where(lo_index<spectra_size-2*window_p,lo_index,index - 
                        (spectra_size - index) +1)
    for j in range(iterations):
        temp = (background[lo_index] +
                background[hi_index]) / 2.

        bg_index = background > temp
        background[bg_index] = temp[bg_index]
     
    window_p=7
    # a final smoothing/filter with a reducing step size.
    for i in range(10):
        lo_index = np.where(index>=window_p,index-window_p,0)
        hi_index = np.where(index>=window_p,index+window_p,index+index)
        hi_index = np.where(hi_index<spectra_size,hi_index,spectra_size-1)
        lo_index = np.where(lo_index<spectra_size-2*window_p,lo_index,index - 
                            (spectra_size - index) +1)
        temp = (background[lo_index] +
                background[hi_index]) / 2.
 
        bg_index = background > temp
        background[bg_index] = temp[bg_index]
        window_p= max(window_p-1,1)

    background = np.exp(np.exp(background) - 1) - 1
    inf_ind = np.where(~np.isfinite(background))    
    background[inf_ind] = 0.0
    return background

