from hyperspy import Release

# Configuration file for ipython.
c = get_config()
c.TerminalIPythonApp.ignore_old_config = True
c.TerminalInteractiveShell.banner2 = Release.info
c.TerminalIPythonApp.exec_lines = [
    'import sys',
    'reload(sys)',
    "sys.setdefaultencoding('utf8')",
    'from hyperspy.hspy import *'
]
