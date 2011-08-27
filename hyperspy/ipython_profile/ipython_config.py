from hyperspy import Release

# Configuration file for ipython.
c = get_config()


c.TerminalIPythonApp.exec_lines = [
    'from hyperspy.hspy import *']

c.TerminalIPythonApp.ignore_old_config = True
c.InteractiveShellApp.exec_lines = ['from hyperspy.hspy import *',
                                    'print(Release.info)']
c.TerminalInteractiveShell.banner2 = Release.info

