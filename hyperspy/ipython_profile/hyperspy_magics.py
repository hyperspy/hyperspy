from IPython.core.magic import Magics, magics_class, line_magic


@magics_class
class HyperspyMagics(Magics):

    @line_magic
    def hyperspy(self, line):
        sh = self.shell
        first_import_part = ("import numpy as np\n"
                             "import hyperspy.hspy as hs\n")
        exec(first_import_part, sh.user_ns)

        if len(line) == 0:
            toolkit = hs.preferences.General.default_toolkit
        else:
            toolkit = line

        sh.enable_matplotlib(toolkit)

        second_import_part = "import matplotlib.pyplot as plt"
        exec(second_import_part, sh.user_ns)

        ans = "\nHyperSpy imported!\nThe following commands were just executed:\n"
        ans += "---------------\n"
        ans += first_import_part + "%matplotlib "+ toolkit + "\n" + second_import_part
        print ans

ip = get_ipython()
ip.register_magics(HyperspyMagics)
