from IPython.core.magic import Magics, magics_class, line_magic
import warnings

from hyperspy.defaults_parser import preferences


@magics_class
class HyperspyMagics(Magics):

    @line_magic
    def hyperspy(self, line):
        """
        %hyperspy [-r] [toolkit]

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

        If you pass `-r`, the current input cell will be overwritten with the above specified commands. As a
        consequence, all other code in the input cell will be deleted!

        Positional arguments:
        ---------------------
            toolkit : {qt4, gtk, wx, tk, None}
                Name of the matplotlib backend to use. If given, the corresponding matplotlib backend is used,
                otherwise it will be the HyperSpy's default.

        Optional arguments:
        -------------------
            -r
                After running the the magic as usual, overwrites the current input cell with just executed
                code that can be run directly without magic

        """
        sh = self.shell

        overwrite = False
        gui = False
        line = line.strip()
        if "-r" in line:
            overwrite = True
            before, after = line.split("-r")
            before, after = before.strip(), after.strip()
            if after:
                toolkit = after
            elif before:
                toolkit = before
            else:
                toolkit = preferences.General.default_toolkit

        elif line:
            toolkit = line.strip()
        else:
            toolkit = preferences.General.default_toolkit

        if toolkit not in ['qt4', 'gtk', 'wx', 'tk', 'None', 'none']:
            raise ValueError("The '%s' toolkit is not supported.\n" % toolkit +
                             "Supported toolkits: {qt4, gtk, wx, tk, None}")

        mpl_code = ""
        if toolkit in ["None", "none"]:
            mpl_code = ("import matplotlib\n"
                        "matplotlib.use('Agg')\n")
        elif toolkit == 'qt4':
            gui = True
            mpl_code = ("import os\n"
                        "os.environ['QT_API'] = 'pyqt'\n")
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
                "HyperSpy 0.9. Please use the new API imported as ``hs`` "
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
        print header + ans
        if overwrite:
            sh.set_next_input(
                "# %hyperspy -r " +
                toolkit +
                "\n" +
                ans +
                "\n\n",
                replace=True)
