# -*- coding: utf-8 -*-


def set_log_level(level):
    """
    Convenience function to set the log level of all hyperspy modules.

    Note: The log level of all other modules are left untouched.

    Parameters
    ----------
        level : {int | str}
            The log level to set. Any values that `logging.Logger.setLevel()`
            accepts are valid. The default options are:
                - 'CRITICAL'
                - 'ERROR'
                - 'WARNING'
                - 'INFO'
                - 'DEBUG'
                - 'NOTSET'

    Example use
    -----------

    For normal logging of hyperspy functions, you can set the log level like
    this:

        >>> import hyperspy.api as hs
        >>> hs.set_log_level('INFO')
        >>> hs.load(r'my_file.dm3')
        INFO:hyperspy.io_plugins.digital_micrograph:DM version: 3
        INFO:hyperspy.io_plugins.digital_micrograph:size 4796607 B
        INFO:hyperspy.io_plugins.digital_micrograph:Is file Little endian? True
        INFO:hyperspy.io_plugins.digital_micrograph:Total tags in root group: 15
        <Signal2D, title: My file, dimensions: (|1024, 1024)>

    If you need the log output during the initial import of hyperspy, you
    should set the log level like this:

        >>> from hyperspy.logger import set_log_level
        >>> set_log_level('DEBUG')
        >>> import hyperspy.api as hs
        DEBUG:hyperspy.gui:Loading hyperspy.gui
        DEBUG:hyperspy.gui:Current MPL backend: TkAgg
        DEBUG:hyperspy.gui:Current ETS toolkit: qt4
        DEBUG:hyperspy.gui:Current ETS toolkit set to: null

    """
    import logging
    logging.basicConfig()  # Does nothing if already configured
    logging.getLogger('hyperspy').setLevel(level)
