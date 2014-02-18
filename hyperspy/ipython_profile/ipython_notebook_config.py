c = get_config()
c.IPKernelApp.exec_lines = ['from hyperspy.hspy import *',
                            'import hyperspy.Release',
                            'print hyperspy.Release.info']
c.IPKernelApp.pylab = "qt"
