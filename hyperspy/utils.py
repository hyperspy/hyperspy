import tempfile

import numpy as np

from hyperspy.misc.utils import DictionaryBrowser

def stack(signal_list, axis=None, new_axis_name='stack_element', 
          mmap=False, mmap_dir=None,):
    """Concatenate the signals in the list over a given axis or a new axis.
    
    The title is set to that of the first signal in the list.
    
    Parameters
    ----------
    signal_list : list of Signal instances
    axis : {None, int, str}
        If None, the signals are stacked over a new axis. The data must 
        have the same dimensions. Otherwise the 
        signals are stacked over the axis given by its integer index or
        its name. The data must have the same shape, except in the dimension
        corresponding to `axis`.
    new_axis_name : string
        The name of the new axis when `axis` is None.
        If an axis with this name already 
        exists it automatically append '-i', where `i` are integers,
        until it finds a name that is not yet in use.
    mmap: bool
        If True and stack is True, then the data is stored
        in a memory-mapped temporary file.The memory-mapped data is 
        stored on disk, and not directly loaded into memory.  
        Memory mapping is especially useful for accessing small 
        fragments of large files without reading the entire file into 
        memory.
    mmap_dir : string
        If mmap_dir is not None, and stack and mmap are True, the memory
        mapped file will be created in the given directory,
        otherwise the default directory is used.
    
    Returns
    -------
    signal : Signal instance (or subclass, determined by the objects in
        signal list)
        
    Examples
    --------
    >>> data = np.arange(20)
    >>> s = utils.stack([signals.Spectrum(data[:10]), signals.Spectrum(data[10:])])
    >>> s
    <Spectrum, title: Stack of , dimensions: (2, 10)>
    >>> s.data
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    
    """

    for i, obj in enumerate(signal_list):    
        if i == 0:
            if axis is None:
                original_shape = obj.data.shape
                stack_shape = tuple([len(signal_list),]) + original_shape
                tempf = None
                if mmap is False:
                    data = np.empty(stack_shape,
                                           dtype=obj.data.dtype)
                else:
                    tempf = tempfile.NamedTemporaryFile(
                                                    dir=mmap_dir)
                    data = np.memmap(tempf,
                                     dtype=obj.data.dtype,
                                     mode = 'w+',
                                     shape=stack_shape,)
        
                signal = type(obj)(data=data)
                signal.axes_manager._axes[1:] = obj.axes_manager._axes
                axis_name = new_axis_name
                axis_names = [axis_.name for axis_ in 
                              signal.axes_manager._axes[1:]]
                j = 1
                while axis_name in axis_names:
                    axis_name = new_axis_name + "-%i" % j
                    j += 1             
                eaxis = signal.axes_manager._axes[0] 
                eaxis.name = axis_name           
                eaxis.navigate = True # This triggers _update_parameters
                signal.mapped_parameters = obj.mapped_parameters
                # Get the title from 1st object
                signal.mapped_parameters.title = (
                    "Stack of " + obj.mapped_parameters.title)
                signal.original_parameters = DictionaryBrowser({})
            else:
                axis = obj.axes_manager[axis]
                signal = obj.deepcopy()
            
            signal.original_parameters.add_node('stack_elements')

        # Store parameters
        signal.original_parameters.stack_elements.add_node(
            'element%i' % i)
        node = signal.original_parameters.stack_elements[
            'element%i' % i]
        node.original_parameters = \
            obj.original_parameters.as_dictionary()
        node.mapped_parameters = \
            obj.mapped_parameters.as_dictionary()

        if axis is None:            
            if obj.data.shape != original_shape:
                raise IOError(
              "Only files with data of the same shape can be stacked")
            signal.data[i,...] = obj.data
            del obj
    if axis is not None:
        signal.data = np.concatenate([signal_.data for signal_ in signal_list],
                                     axis=axis.index_in_array)
        signal.get_dimensions_from_data()
    return signal
                    