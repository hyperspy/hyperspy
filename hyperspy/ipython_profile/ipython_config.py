from hyperspy import Release

# Configuration file for ipython.
c = get_config()
c.TerminalIPythonApp.ignore_old_config = True
c.TerminalInteractiveShell.banner2 = Release.info + \
    "\nHyperspy is loaded in hs.* namespace\n"

c.InteractiveShellApp.exec_lines = ['import hyperspy.hspy as hs', ]
