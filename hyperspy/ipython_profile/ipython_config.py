from hyperspy import Release

# Configuration file for ipython.
c = get_config()
c.TerminalIPythonApp.ignore_old_config = True
c.TerminalInteractiveShell.banner2 = Release.info
c.TerminalIPythonApp.exec_lines = ['from hyperspy.hspy import *']
