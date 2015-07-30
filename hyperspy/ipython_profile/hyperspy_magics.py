from IPython.core.magic import Magics, magics_class, line_magic


@magics_class
class HyperspyMagics(Magics):

    @line_magic
    def hyperspy(self, line):
        sh = self.shell
        first_import_part = ("import numpy as np\n"
                             "import hyperspy.hspy as hs\n")
        exec(first_import_part, sh.user_ns)
        
        overwrite = False
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
                toolkit = hs.preferences.General.default_toolkit

        elif line:
            toolkit = line.strip()
        else:
            toolkit = hs.preferences.General.default_toolkit

        sh.enable_matplotlib(toolkit)

        second_import_part = "import matplotlib.pyplot as plt"
        exec(second_import_part, sh.user_ns)

        ans = "\nHyperSpy imported!\nThe following commands were just executed:\n"
        ans += "---------------\n"
        ans += first_import_part + "%matplotlib " + \
            toolkit + "\n" + second_import_part
        print ans
        if overwrite:
            sh.set_next_input("# %hyperspy -r "+toolkit+\
                    "\n"+first_import_part + \
                    "%matplotlib " + toolkit + "\n" +\
                    second_import_part + "\n\n", 
                    replace=True)

ip = get_ipython()
ip.register_magics(HyperspyMagics)
