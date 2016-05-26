from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.magic_arguments import magic_arguments, argument, parse_argstring
import warnings

from hyperspy.defaults_parser import preferences
from hyperspy.misc.hspy_warnings import VisibleDeprecationWarning


@magics_class
class HyperspyMagics(Magics):

    @line_magic
    @magic_arguments()
    @argument('-r', '--replace', action='store_true', default=None,
              help="""After running the the magic as usual, overwrites the current input cell with just executed
              code that can be run directly without magic"""
              )
    @argument('toolkit', nargs='?', default=None,
              help="""Name of the matplotlib backend to use.  If given, the corresponding matplotlib backend
              is used, otherwise it will be the HyperSpy's default.  Available toolkits: {qt4, wx, None, gtk,
              tk}. Note that gtk and tk toolkits are not fully supported
              """
              )
    def hyperspy(self, line):
        """
        Load HyperSpy, numpy and matplotlib to work interactively.

        %hyperspy runs the following commands in various cases:

        >>> # if toolkit is "None" only
        >>> import matplotlib
        >>> matplotlib.use('Agg')

        >>> # if toolkit is "qt4" only
        >>> import os
        >>> os.environ['QT_API'] = 'pyqt'

        >>> # if toolkit is not "None"
        >>> %matplotlib [toolkit]

        >>> # run in all cases
        >>> import numpy as np
        >>> import hyperspy.api as hs
        >>> import matplotlib.pyplot as plt

        """
        warnings.warn(
            "This magic is deprecated and will be removed in Hyperspy 1.0."
            "The reccomended way to start HyperSpy is:\n\n"
            ">>> import hyperspy.api as hs\n"
            ">>> %matplotlib\n\n"
            "See the online documentation for more details.",
            VisibleDeprecationWarning)
        sh = self.shell

        gui = False
        args = parse_argstring(self.hyperspy, line)
        overwrite = not args.replace is None
        toolkit = args.toolkit
        if toolkit is None:
            toolkit = preferences.General.default_toolkit

        if toolkit not in ['qt4', 'gtk', 'wx', 'tk', 'None']:
            raise ValueError("The '%s' toolkit is not supported.\n" % toolkit +
                             "Supported toolkits: {qt4, gtk, wx, tk, None}")

        mpl_code = ""
        if toolkit == "None":
            mpl_code = ("import matplotlib\n"
                        "matplotlib.use('Agg')\n")
        else:
            gui = True

        exec(mpl_code, sh.user_ns)
        if gui:
            sh.enable_matplotlib(toolkit)
        first_import_part = ("import numpy as np\n"
                             "import hyperspy.api as hs\n"
                             "import matplotlib.pyplot as plt\n")
        exec(first_import_part, sh.user_ns)

        if preferences.General.import_hspy:
            second_import_part = "from hyperspy.hspy import *\n"
            warnings.warn(
                "Importing everything from ``hyperspy.hspy`` will be removed in "
                "Hyperspy 1.0. Please use the new API imported as ``hs`` "
                "instead. See the "
                "`Getting started` section of the User Guide for details.",
                UserWarning)
            exec(second_import_part, sh.user_ns)
            first_import_part += second_import_part

        header = "\nHyperSpy imported!\nThe following commands were just executed:\n"
        header += "---------------\n"
        ans = mpl_code
        if gui:
            ans += "%matplotlib " + toolkit + "\n"

        ans += first_import_part
        print(header + ans)
        if overwrite:
            sh.set_next_input(
                "# %hyperspy -r " +
                toolkit +
                "\n" +
                ans +
                "\n\n",
                replace=True)
