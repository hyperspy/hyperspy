c = get_config()
c.IPKernelApp.exec_lines = ['import sys',
                            'reload(sys)',
                            "sys.setdefaultencoding('utf8')",
                            'from hyperspy.hspy import *',
                            'import hyperspy.Release',
                            'print hyperspy.Release.info']
c.IPKernelApp.pylab = "qt"
